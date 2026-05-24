import { useCallback, useEffect, useRef, useState } from "react";
import {
  FRAME_FEATURES,
  FRAME_LANDMARKS,
  GROUP_OFFSETS,
  GROUP_SIZES,
  buildFrameFeatures,
} from "../lib/landmarks";
import { transcribe, warmup as warmupCTC } from "../lib/ctc";
import type { FrameDetection } from "../lib/mediapipe";

/**
 * User-driven fingerspelling recorder.
 *
 * While recording, every MediaPipe frame with a visible hand pushes a (features, missing) sample into a buffer, and the latest per-letter MLP sign is appended to a parallel sequence.
 * On stop(), the full landmark buffer is run through the CTC model client-side and the decoded phrase is returned via `result`
 */
const MAX_FRAMES = 256; // matches CTC training-time max_frames
const MIN_FRAMES = 4;

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

interface FrameSample {
  features: Float32Array;
  missing: Uint8Array;
}

export type RecorderState = "idle" | "recording" | "decoding" | "error";

export interface FingerspellRecorder {
  state: RecorderState;
  /** CTC-decoded phrase from the last completed recording */
  result: string;
  /** Per-letter MLP outputs collected during the last recording, in order, with consecutive duplicates collapsed.
   *  del/space are kept as literal tokens */
  letters: string[];
  /** Number of hand-present frames captured so far in the current recording */
  frameCount: number;
  /** Non-fatal error message from the last decode attempt, if any */
  error: string | null;
  start: () => void;
  stop: () => void;
  reset: () => void;
}

export function useFingerspellRecorder(
  detection: FrameDetection | null,
  liveSign: string,
  active: boolean,
): FingerspellRecorder {
  const [state, setState] = useState<RecorderState>("idle");
  const [result, setResult] = useState("");
  const [letters, setLetters] = useState<string[]>([]);
  const [frameCount, setFrameCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const bufferRef = useRef<FrameSample[]>([]);
  const lettersRef = useRef<string[]>([]);
  const lastLetterRef = useRef<string>("-");

  // Kick off the ~18 MB CTC model fetch early so the first recording doesn't pay the load cost
  useEffect(() => {
    warmupCTC().catch((err) => console.error("CTC warmup failed:", err));
  }, []);

  const start = useCallback(() => {
    bufferRef.current = [];
    lettersRef.current = [];
    lastLetterRef.current = "-";
    setResult("");
    setLetters([]);
    setError(null);
    setFrameCount(0);
    setState("recording");
  }, []);

  const reset = useCallback(() => {
    bufferRef.current = [];
    lettersRef.current = [];
    lastLetterRef.current = "-";
    setResult("");
    setLetters([]);
    setError(null);
    setFrameCount(0);
    setState("idle");
  }, []);

  // Push frames + collapse-collect MLP letters while recording
  useEffect(() => {
    if (state !== "recording" || !active || !detection) return;
    const { features, missing } = buildFrameFeatures(detection.landmarks);
    if (!hasHand(missing)) return;
    if (bufferRef.current.length >= MAX_FRAMES) return;
    bufferRef.current.push({ features, missing });
    setFrameCount(bufferRef.current.length);
    if (liveSign !== "-" && liveSign !== lastLetterRef.current) {
      lastLetterRef.current = liveSign;
      lettersRef.current.push(liveSign);
    }
  }, [detection, liveSign, state, active]);

  // Decode buffer via in-browser CTC on stop
  const stop = useCallback(() => {
    if (state !== "recording") return;
    const buf = bufferRef.current;
    const collectedLetters = [...lettersRef.current];
    setLetters(collectedLetters);
    if (buf.length < MIN_FRAMES) {
      setState("idle");
      setError("Recording too short. Hold a sign for at least a few frames.");
      return;
    }
    setState("decoding");

    const T = buf.length;
    const flatFeatures = new Float32Array(T * FRAME_FEATURES);
    const flatMissing = new Uint8Array(T * FRAME_LANDMARKS);
    for (let t = 0; t < T; t++) {
      flatFeatures.set(buf[t].features, t * FRAME_FEATURES);
      flatMissing.set(buf[t].missing, t * FRAME_LANDMARKS);
    }

    transcribe(flatFeatures, flatMissing)
      .then((text) => {
        setResult(text);
        setState("idle");
      })
      .catch((err) => {
        console.error("CTC decode failed:", err);
        setError(err instanceof Error ? err.message : String(err));
        setState("error");
      });
  }, [state]);

  // Stop recording if the camera goes inactive mid-record
  useEffect(() => {
    if (!active && state === "recording") {
      setState("idle");
    }
  }, [active, state]);

  return { state, result, letters, frameCount, error, start, stop, reset };
}
