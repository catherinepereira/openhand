/**
 * Browser-side ASL alphabet MLP
 *
 * Loads asl_classifier.onnx and model_meta.json from /models/, normalizes 21 hand landmarks (wrist anchor + p95 |value| scale), runs onnxruntime-web, and returns the predicted letter + confidence
 *
 * MUST stay in sync with openhand-model/alphabet/scripts/preprocess_alphabet.py, which uses the same normalization at training time
 */

import * as ort from "onnxruntime-web";
import { executionProviders, logProvider } from "./ortProvider";
import { runExclusive } from "./ortQueue";
import type { DetectedHand } from "../hooks/useSignDetection";

const MODEL_URL = "/models/asl_classifier.onnx";
const META_URL = "/models/model_meta.json";

export const N_LANDMARKS = 21;
export const N_FEATURES = N_LANDMARKS * 3;

const MIN_CONFIDENCE = 0.5;
const NORM_PERCENTILE = 95;
const NORM_EPS = 1e-6;

interface Meta {
  label_map: Record<string, string>;
  num_classes: number;
  input_dim: number;
}

let _session: ort.InferenceSession | null = null;
let _meta: Meta | null = null;
let _inputName = "";

async function loadMeta(): Promise<Meta> {
  const r = await fetch(META_URL);
  if (!r.ok) throw new Error(`Failed to load model meta from ${META_URL}`);
  return (await r.json()) as Meta;
}

async function init(): Promise<void> {
  if (_session) return;
  logProvider("alphabet");
  const [session, meta] = await Promise.all([
    ort.InferenceSession.create(MODEL_URL, {
      executionProviders,
      graphOptimizationLevel: "all",
    }),
    loadMeta(),
  ]);
  _session = session;
  _meta = meta;
  _inputName = session.inputNames[0];
}

/** Mirror of numpy.percentile with the default linear interpolation */
function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  const frac = rank - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

function normalize(vec: Float32Array): Float32Array {
  const wristX = vec[0];
  const wristY = vec[1];
  const wristZ = vec[2];
  const out = new Float32Array(N_FEATURES);
  const abs = new Array<number>(N_FEATURES);
  for (let i = 0; i < N_LANDMARKS; i++) {
    const base = i * 3;
    out[base] = vec[base] - wristX;
    out[base + 1] = vec[base + 1] - wristY;
    out[base + 2] = vec[base + 2] - wristZ;
    abs[base] = Math.abs(out[base]);
    abs[base + 1] = Math.abs(out[base + 1]);
    abs[base + 2] = Math.abs(out[base + 2]);
  }
  const scale = percentile(abs, NORM_PERCENTILE);
  if (scale > NORM_EPS) {
    for (let i = 0; i < N_FEATURES; i++) out[i] /= scale;
  }
  return out;
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exp = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exp[i] = Math.exp(logits[i] - max);
    sum += exp[i];
  }
  for (let i = 0; i < logits.length; i++) exp[i] /= sum;
  return exp;
}

export interface AlphabetResult {
  sign: string;
  confidence: number;
}

/**
 * Run inference on the right hand of a frame
 * Returns "-" with 0 confidence if no right hand is detected, or if the best class is below MIN_CONFIDENCE
 */
export async function classifyHand(
  hands: readonly DetectedHand[],
): Promise<AlphabetResult> {
  if (!_session || !_meta) await init();
  if (!hands.length) return { sign: "-", confidence: 0.0 };

  const target = hands.find(
    (h) => h.handedness === "Right" && h.landmarks.length === N_LANDMARKS,
  );
  if (!target) return { sign: "-", confidence: 0.0 };

  const vec = new Float32Array(N_FEATURES);
  for (let i = 0; i < N_LANDMARKS; i++) {
    const lm = target.landmarks[i];
    vec[i * 3] = lm.x;
    vec[i * 3 + 1] = lm.y;
    vec[i * 3 + 2] = lm.z;
  }
  const normalized = normalize(vec);

  const session = _session!;
  const meta = _meta!;
  const tensor = new ort.Tensor("float32", normalized, [1, N_FEATURES]);
  const output = await runExclusive(() => session.run({ [_inputName]: tensor }));
  const logitsTensor = output[session.outputNames[0]];
  const probs = softmax(logitsTensor.data as Float32Array);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++)
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  const confidence = probs[bestIdx];
  const sign =
    confidence >= MIN_CONFIDENCE ? meta.label_map[String(bestIdx)] : "-";
  return { sign, confidence };
}

/** Load model early */
export function warmup(): Promise<void> {
  return init();
}
