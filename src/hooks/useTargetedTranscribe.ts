import { useEffect, useRef, useState } from "react";
import {
  FRAME_FEATURES,
  FRAME_LANDMARKS,
  GROUP_OFFSETS,
  GROUP_SIZES,
  buildFrameFeatures,
} from "../lib/landmarks";
import { transcribe, warmup as warmupCTC } from "../lib/ctc";
import type { FrameDetection } from "../lib/mediapipe";

function hasHand(missing: Uint8Array): boolean {
  const lh = GROUP_OFFSETS.leftHand;
  const rh = GROUP_OFFSETS.rightHand;
  for (let i = 0; i < GROUP_SIZES.leftHand; i++) {
    if (missing[lh + i] === 0) return true;
  }
  for (let i = 0; i < GROUP_SIZES.rightHand; i++) {
    if (missing[rh + i] === 0) return true;
  }
  return false;
}

/**
 * Rolling-window CTC decode for the Learn screen's J/Z grading.
 * Keeps a ~4s buffer of hand-present frames and re-decodes on a timer.
 *
 * Runs CTC client-side via lib/ctc.ts.
 * Beam=3 instead of the default 10 because this fires every cadence tick and the latency budget is tighter
 */
const CONFIG = {
  bufferFrames: 40,
  decodeCadenceMs: 900,
  beamWidth: 3,
} as const;

interface FrameSample {
  features: Float32Array;
  missing: Uint8Array;
}

export interface TargetedTranscribeResult {
  /** The most recent CTC decode of the rolling buffer */
  latest: string;
  /** True if the CTC model failed to load */
  unavailable: boolean;
}

export function useTargetedTranscribe(
  detection: FrameDetection | null,
  active: boolean,
): TargetedTranscribeResult {
  const bufferRef = useRef<FrameSample[]>([]);
  const decodeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inflightRef = useRef(false);
  const lastHandTimeRef = useRef<number>(0);

  const [latest, setLatest] = useState("");
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    warmupCTC().catch((err) => {
      console.error("CTC warmup failed:", err);
      setUnavailable(true);
    });
  }, []);

  useEffect(() => {
    if (!active) {
      bufferRef.current = [];
      setLatest("");
      return;
    }
  }, [active]);

  // Push hand-present frames into the rolling buffer.
  // After 2s of no hand, clear the buffer + displayed text so what's on screen reflects current intent
  useEffect(() => {
    if (!active || !detection) return;
    const { features, missing } = buildFrameFeatures(detection.landmarks);
    if (!hasHand(missing)) {
      if (
        lastHandTimeRef.current &&
        performance.now() - lastHandTimeRef.current > 2000
      ) {
        if (bufferRef.current.length > 0) bufferRef.current = [];
        setLatest((cur) => (cur ? "" : cur));
      }
      return;
    }
    lastHandTimeRef.current = performance.now();
    const buf = bufferRef.current;
    buf.push({ features, missing });
    while (buf.length > CONFIG.bufferFrames) buf.shift();
  }, [detection, active]);

  // Periodic decode tick
  useEffect(() => {
    if (!active || unavailable) {
      if (decodeTimerRef.current) clearInterval(decodeTimerRef.current);
      decodeTimerRef.current = null;
      return;
    }

    decodeTimerRef.current = setInterval(() => {
      if (inflightRef.current) return;
      const buf = bufferRef.current;
      if (buf.length === 0) return;

      const T = buf.length;
      const flatFeatures = new Float32Array(T * FRAME_FEATURES);
      const flatMissing = new Uint8Array(T * FRAME_LANDMARKS);
      for (let t = 0; t < T; t++) {
        flatFeatures.set(buf[t].features, t * FRAME_FEATURES);
        flatMissing.set(buf[t].missing, t * FRAME_LANDMARKS);
      }

      inflightRef.current = true;
      transcribe(flatFeatures, flatMissing, { beamWidth: CONFIG.beamWidth })
        .then((text) => setLatest(text))
        .catch((err) => {
          console.error("CTC decode failed:", err);
          setUnavailable(true);
        })
        .finally(() => {
          inflightRef.current = false;
        });
    }, CONFIG.decodeCadenceMs);

    return () => {
      if (decodeTimerRef.current) clearInterval(decodeTimerRef.current);
      decodeTimerRef.current = null;
    };
  }, [active, unavailable]);

  return { latest, unavailable };
}
