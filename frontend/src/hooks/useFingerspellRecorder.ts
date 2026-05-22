import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import {
  FRAME_FEATURES,
  FRAME_LANDMARKS,
  GROUP_OFFSETS,
  GROUP_SIZES,
  buildFrameFeatures,
} from "../lib/landmarks";
import type { FrameDetection } from "../lib/mediapipe";

/**
 * User-driven fingerspelling recorder. The caller drives start()/stop();
 * while recording, every MediaPipe frame with a visible hand pushes a
 * (features, missing) sample into a buffer and the latest per-letter MLP
 * sign is appended to a parallel sequence.
 *
 * On stop(), the full landmark buffer is sent in one shot to the CTC
 * WebSocket, and the decoded phrase is returned via `result`.
 *
 * The MLP-letter sequence is exposed too so the caller can use it
 * separately (e.g. as a second-opinion lookup or search hint).
 */
const MAX_FRAMES = 512; // matches backend MAX_FRAMES guard
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
  /** CTC-decoded phrase from the last completed recording. */
  result: string;
  /** Per-letter MLP outputs collected during the last recording, in order
   *  with consecutive duplicates collapsed. del/space are kept as literal
   *  tokens. */
  letters: string[];
  /** Number of hand-present frames captured so far in the current recording. */
  frameCount: number;
  /** Non-fatal error message from the last decode attempt, if any. */
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
  const wsRef = useRef<WebSocket | null>(null);

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

  // Push frames + collapse-collect MLP letters while recording.
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

  // Send buffer to CTC when the user stops.
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

    const ws = new WebSocket(WS_ENDPOINTS.transcribeStream);
    wsRef.current = ws;

    const cleanup = () => {
      if (wsRef.current === ws) wsRef.current = null;
      try { ws.close(); } catch { /* ignore */ }
    };

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "decode",
          frame_count: T,
          features: Array.from(flatFeatures),
          missing: Array.from(flatMissing, (b) => b !== 0),
        }),
      );
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "result") {
          setResult(typeof data.text === "string" ? data.text : "");
          setState("idle");
        } else if (data.type === "error") {
          setError(String(data.message ?? "CTC decode failed"));
          setState("error");
        }
      } catch {
        setError("Malformed response from CTC server");
        setState("error");
      }
      cleanup();
    };
    ws.onerror = () => {
      setError("Failed to reach the transcription service");
      setState("error");
      cleanup();
    };
    ws.onclose = () => {
      // If we never received a result message and we're still in
      // 'decoding', surface the silent close.
      setState((s) => (s === "decoding" ? "error" : s));
      if (wsRef.current === ws) wsRef.current = null;
    };
  }, [state]);

  // Stop recording if the camera goes inactive mid-record.
  useEffect(() => {
    if (!active && state === "recording") {
      setState("idle");
    }
  }, [active, state]);

  // Close any open socket on unmount.
  useEffect(() => () => {
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
  }, []);

  return { state, result, letters, frameCount, error, start, stop, reset };
}
