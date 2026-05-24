/**
 * Browser-side CTC fingerspelling decoder.
 *
 * Loads asl_ctc.onnx (~18 MB) lazily on first use, normalizes the holistic landmark clip, runs onnxruntime-web, and decodes with prefix beam search.
 *
 * MUST stay in sync with openhand-model/fingerspelling/model/landmarks.py (training-side normalization).
 * The "missing" mask is part of the contract: an explicit per-landmark bool, not zero-equality
 */

import * as ort from "onnxruntime-web";
import { FRAME_FEATURES, FRAME_LANDMARKS, GROUP_OFFSETS } from "./landmarks";

const MODEL_URL = "/models/asl_ctc.onnx";
const META_URL = "/models/asl_ctc_meta.json";

const MAX_FRAMES = 256;        // matches training-time max_frames
const DEFAULT_BEAM_WIDTH = 10;
const NEG_INF = -1e30;
const NORM_PERCENTILE = 95;
const NORM_EPS = 1e-6;

interface Meta {
  idx_to_char: Record<string, string>;
  blank_idx: number;
  num_classes: number;
}

let _session: ort.InferenceSession | null = null;
let _meta: Meta | null = null;
let _idxToChar: string[] = []; // dense lookup by index, "" for unknown
let _blankIdx = -1;

async function loadMeta(): Promise<Meta> {
  const r = await fetch(META_URL);
  if (!r.ok) throw new Error(`Failed to load CTC meta from ${META_URL}`);
  return (await r.json()) as Meta;
}

async function init(): Promise<void> {
  if (_session) return;
  const [session, meta] = await Promise.all([
    ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }),
    loadMeta(),
  ]);
  _session = session;
  _meta = meta;
  _blankIdx = meta.blank_idx;
  // Dense lookup table for fast decode. Skip "<blank>" sentinel; emit "" for any unknown index
  const maxIdx = Math.max(...Object.keys(meta.idx_to_char).map((k) => parseInt(k, 10)));
  _idxToChar = new Array(maxIdx + 1).fill("");
  for (const [k, v] of Object.entries(meta.idx_to_char)) {
    const i = parseInt(k, 10);
    if (i !== _blankIdx) _idxToChar[i] = v;
  }
}

/** numpy.percentile with linear interpolation, on an unsorted array */
function percentile(values: Float32Array, p: number): number {
  if (values.length === 0) return 0;
  const sorted = Float32Array.from(values).sort();
  if (sorted.length === 1) return sorted[0];
  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  const frac = rank - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

/**
 * Per-sequence normalization.
 * Anchor on the more-present wrist's mean position, scale by p95 |coord| over present landmarks, re-zero missing
 */
function normalizeSequence(features: Float32Array, missing: Uint8Array, T: number): Float32Array {
  const out = new Float32Array(T * FRAME_FEATURES);
  out.set(features);
  // NaN -> 0; MediaPipe occasionally emits NaN on occluded landmarks
  for (let i = 0; i < out.length; i++) if (!isFinite(out[i])) out[i] = 0;

  const rhLm = GROUP_OFFSETS.rightHand;
  const lhLm = GROUP_OFFSETS.leftHand;

  // Count per-frame presence of each wrist (landmark 0 of the hand)
  let rightCount = 0;
  let leftCount = 0;
  for (let t = 0; t < T; t++) {
    if (missing[t * FRAME_LANDMARKS + rhLm] === 0) rightCount++;
    if (missing[t * FRAME_LANDMARKS + lhLm] === 0) leftCount++;
  }

  const useRight = rightCount >= leftCount;
  const anchorLm = useRight ? rhLm : lhLm;
  let anchorX = 0;
  let anchorY = 0;
  let anchorZ = 0;
  let anchorN = 0;
  for (let t = 0; t < T; t++) {
    if (missing[t * FRAME_LANDMARKS + anchorLm] !== 0) continue;
    const base = (t * FRAME_LANDMARKS + anchorLm) * 3;
    anchorX += out[base];
    anchorY += out[base + 1];
    anchorZ += out[base + 2];
    anchorN++;
  }
  if (anchorN > 0) {
    anchorX /= anchorN;
    anchorY /= anchorN;
    anchorZ /= anchorN;
  } else {
    anchorX = anchorY = anchorZ = 0;
  }

  // Subtract anchor from every landmark, regardless of missing.
  // Missing positions get re-zeroed after the scale step
  for (let t = 0; t < T; t++) {
    for (let lm = 0; lm < FRAME_LANDMARKS; lm++) {
      const base = (t * FRAME_LANDMARKS + lm) * 3;
      out[base] -= anchorX;
      out[base + 1] -= anchorY;
      out[base + 2] -= anchorZ;
    }
  }

  // Collect |coord| over present landmarks for the percentile.
  // Pre-count so the Float32Array can be sized exactly
  let presentCount = 0;
  for (let t = 0; t < T; t++) {
    for (let lm = 0; lm < FRAME_LANDMARKS; lm++) {
      if (missing[t * FRAME_LANDMARKS + lm] === 0) presentCount++;
    }
  }
  if (presentCount > 0) {
    const abs = new Float32Array(presentCount * 3);
    let cur = 0;
    for (let t = 0; t < T; t++) {
      for (let lm = 0; lm < FRAME_LANDMARKS; lm++) {
        if (missing[t * FRAME_LANDMARKS + lm] !== 0) continue;
        const base = (t * FRAME_LANDMARKS + lm) * 3;
        abs[cur++] = Math.abs(out[base]);
        abs[cur++] = Math.abs(out[base + 1]);
        abs[cur++] = Math.abs(out[base + 2]);
      }
    }
    const scale = percentile(abs, NORM_PERCENTILE);
    if (scale > NORM_EPS) {
      for (let i = 0; i < out.length; i++) out[i] /= scale;
    }
  }

  // Re-zero missing
  for (let t = 0; t < T; t++) {
    for (let lm = 0; lm < FRAME_LANDMARKS; lm++) {
      if (missing[t * FRAME_LANDMARKS + lm] !== 0) {
        const base = (t * FRAME_LANDMARKS + lm) * 3;
        out[base] = 0;
        out[base + 1] = 0;
        out[base + 2] = 0;
      }
    }
  }
  return out;
}

/** Numerically stable log(exp(a) + exp(b)) */
function logsumexp(a: number, b: number): number {
  if (a <= NEG_INF) return b;
  if (b <= NEG_INF) return a;
  const m = a > b ? a : b;
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** Get the top-K indices of a Float32Array by descending value */
function topK(lp: Float32Array, k: number): number[] {
  const idx = new Array<number>(lp.length);
  for (let i = 0; i < lp.length; i++) idx[i] = i;
  idx.sort((a, b) => lp[b] - lp[a]);
  return idx.slice(0, k);
}

/**
 * CTC prefix beam search.
 * logProbs: (T, V) flat row-major; V = vocab incl. blank.
 * Prefix keys are joined comma-separated id strings; JS Map keys hash by-identity for objects, so a string key is the cheap way to dedupe
 */
function beamSearchDecode(
  logProbs: Float32Array,
  T: number,
  V: number,
  blank: number,
  beamWidth: number,
): string {
  // Each beam: prefix string -> [pb, pnb]
  let beams = new Map<string, [number, number]>();
  beams.set("", [0.0, NEG_INF]);

  const topKSize = Math.min(beamWidth * 2, V);
  const ltMap = new Map<string, number[]>(); // prefix string -> list of int ids
  ltMap.set("", []);

  for (let t = 0; t < T; t++) {
    const lpT = logProbs.subarray(t * V, (t + 1) * V) as Float32Array;
    const topIdx = topK(lpT, topKSize);

    const next = new Map<string, [number, number]>();
    const nextIds = new Map<string, number[]>();

    for (const [prefix, [pB, pNB]] of beams) {
      const prefixIds = ltMap.get(prefix)!;
      for (const s of topIdx) {
        const pS = lpT[s];

        if (s === blank) {
          const existing = next.get(prefix);
          const eB = existing ? existing[0] : NEG_INF;
          const eNB = existing ? existing[1] : NEG_INF;
          const newPb = logsumexp(eB, logsumexp(pB + pS, pNB + pS));
          next.set(prefix, [newPb, eNB]);
          if (!nextIds.has(prefix)) nextIds.set(prefix, prefixIds);
          continue;
        }

        const last = prefixIds.length > 0 ? prefixIds[prefixIds.length - 1] : null;
        if (s === last) {
          // Extend: pb -> new prefix
          const newPrefix = prefix + "," + s;
          const ex1 = next.get(newPrefix);
          const e1B = ex1 ? ex1[0] : NEG_INF;
          const e1NB = ex1 ? ex1[1] : NEG_INF;
          const newPnb = logsumexp(e1NB, pB + pS);
          next.set(newPrefix, [e1B, newPnb]);
          if (!nextIds.has(newPrefix)) {
            const ids = prefixIds.slice();
            ids.push(s);
            nextIds.set(newPrefix, ids);
          }

          // Collapse: pnb -> same prefix
          const ex2 = next.get(prefix);
          const e2B = ex2 ? ex2[0] : NEG_INF;
          const e2NB = ex2 ? ex2[1] : NEG_INF;
          const newPnbSame = logsumexp(e2NB, pNB + pS);
          next.set(prefix, [e2B, newPnbSame]);
          if (!nextIds.has(prefix)) nextIds.set(prefix, prefixIds);
        } else {
          const newPrefix = prefix + "," + s;
          const ex = next.get(newPrefix);
          const eB = ex ? ex[0] : NEG_INF;
          const eNB = ex ? ex[1] : NEG_INF;
          const newPnb = logsumexp(eNB, logsumexp(pB + pS, pNB + pS));
          next.set(newPrefix, [eB, newPnb]);
          if (!nextIds.has(newPrefix)) {
            const ids = prefixIds.slice();
            ids.push(s);
            nextIds.set(newPrefix, ids);
          }
        }
      }
    }

    // Prune to top beam_width by combined log-prob
    const scored: [string, number, number, number][] = [];
    for (const [pre, [pb, pnb]] of next) {
      scored.push([pre, pb, pnb, logsumexp(pb, pnb)]);
    }
    scored.sort((a, b) => b[3] - a[3]);
    beams = new Map();
    const newLtMap = new Map<string, number[]>();
    for (let i = 0; i < Math.min(beamWidth, scored.length); i++) {
      const [pre, pb, pnb] = scored[i];
      beams.set(pre, [pb, pnb]);
      newLtMap.set(pre, nextIds.get(pre)!);
    }
    ltMap.clear();
    for (const [k, v] of newLtMap) ltMap.set(k, v);
  }

  // Pick best beam
  let bestPrefix = "";
  let bestScore = -Infinity;
  for (const [pre, [pb, pnb]] of beams) {
    const score = logsumexp(pb, pnb);
    if (score > bestScore) {
      bestScore = score;
      bestPrefix = pre;
    }
  }
  const ids = ltMap.get(bestPrefix) ?? [];
  let out = "";
  for (const i of ids) {
    const ch = _idxToChar[i];
    if (ch) out += ch;
  }
  return out;
}

export interface CTCOptions {
  beamWidth?: number;
}

/**
 * Transcribe a hand-frame buffer to text.
 *
 * features: flat Float32Array of length T * FRAME_FEATURES, row-major.
 * missing:  flat Uint8Array of length T * FRAME_LANDMARKS (1 = missing).
 *
 * The dynamo-exported ONNX needs batch >= 2 to keep the batch axis dynamic, so it's padded with an all-zero, fully-masked second item that's discarded after inference
 */
export async function transcribe(
  features: Float32Array,
  missing: Uint8Array,
  options: CTCOptions = {},
): Promise<string> {
  if (!_session || !_meta) await init();
  const session = _session!;
  const beamWidth = options.beamWidth ?? DEFAULT_BEAM_WIDTH;

  let T = features.length / FRAME_FEATURES;
  if (T === 0) return "";

  // Downsample with linspace if too long
  let feats = features;
  let miss = missing;
  if (T > MAX_FRAMES) {
    const keep = new Array<number>(MAX_FRAMES);
    for (let i = 0; i < MAX_FRAMES; i++) {
      keep[i] = Math.round((i * (T - 1)) / (MAX_FRAMES - 1));
    }
    feats = new Float32Array(MAX_FRAMES * FRAME_FEATURES);
    miss = new Uint8Array(MAX_FRAMES * FRAME_LANDMARKS);
    for (let i = 0; i < MAX_FRAMES; i++) {
      const src = keep[i];
      feats.set(
        features.subarray(src * FRAME_FEATURES, (src + 1) * FRAME_FEATURES),
        i * FRAME_FEATURES,
      );
      miss.set(
        missing.subarray(src * FRAME_LANDMARKS, (src + 1) * FRAME_LANDMARKS),
        i * FRAME_LANDMARKS,
      );
    }
    T = MAX_FRAMES;
  }

  const normalized = normalizeSequence(feats, miss, T);

  // Batch-pad to 2
  const xBatch = new Float32Array(2 * T * FRAME_FEATURES);
  xBatch.set(normalized, 0);
  // Second batch item stays all-zero (already initialized)

  // pad_mask shape (B, T) bool. ONNX bool is uint8 in onnxruntime-web
  const padMask = new Uint8Array(2 * T);
  // First half (real input) all false (0), second half (fake) all true (1)
  padMask.fill(0, 0, T);
  padMask.fill(1, T, 2 * T);

  const xTensor = new ort.Tensor("float32", xBatch, [2, T, FRAME_FEATURES]);
  const maskTensor = new ort.Tensor("bool", padMask, [2, T]);
  const out = await session.run({ landmarks: xTensor, pad_mask: maskTensor });
  const logProbsTensor = out[session.outputNames[0]];
  const logProbs = logProbsTensor.data as Float32Array;
  // ONNX output is (T, B, V)
  const dims = logProbsTensor.dims;
  const outT = dims[0];
  const outB = dims[1];
  const V = dims[2];
  if (outB !== 2 || outT !== T) {
    // Shape drift would be a config mismatch -- fail loudly
    throw new Error(
      `CTC output shape ${dims.join("x")} does not match expected (${T}, 2, V)`,
    );
  }

  // Extract batch index 0 (real input).
  // Flat layout is row-major (T, B, V), so for batch b at time t the offset is (t * B + b) * V
  const lp = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    const src = (t * outB + 0) * V;
    lp.set(logProbs.subarray(src, src + V), t * V);
  }

  return beamSearchDecode(lp, T, V, _blankIdx, beamWidth);
}

/** Eagerly start loading the CTC model. ~18 MB so do this off the main flow */
export function warmup(): Promise<void> {
  return init();
}
