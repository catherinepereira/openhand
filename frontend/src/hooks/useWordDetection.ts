import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import { FRAME_FEATURES, FRAME_LANDMARKS, buildFrameFeatures } from "../lib/landmarks";
import type { FrameDetection } from "../lib/mediapipe";

/**
 * Per-clip word classification for the Learn-the-words screen.
 *
 * Keeps a rolling buffer of MediaPipe frames, sends the whole buffer to
 * the sign-classifier WebSocket every `decodeCadenceMs`, and exposes the
 * latest top-K predictions. Different shape from useTargetedTranscribe:
 * isolated signs need a *complete clip* (~1-2 seconds), so the buffer
 * length matters more here and the cadence is slower.
 *
 * Wire payload matches the backend's /ws/classify-sign route: raw
 * (T, 127, 3) landmarks + (T, 127) missing mask. Normalization happens
 * server-side.
 */
const CONFIG = {
  bufferFrames: 30,        // ~3s at 10 fps; isolated signs are short
  decodeCadenceMs: 800,    // how often to re-classify
  topK: 5,
} as const;

interface FrameSample {
  features: Float32Array;  // length 381 (127 * 3), per-frame landmark x/y/z
  missing: Uint8Array;     // length 127
}

export type WordPrediction = readonly [sign: string, prob: number];

export interface WordDetectionResult {
  /** Top-K predictions on the latest decode. Sign name + probability. */
  predictions: WordPrediction[];
  /** True if the sign-classifier ONNX is missing on the server. */
  unavailable: boolean;
}

export function useWordDetection(
  detection: FrameDetection | null,
  active: boolean,
): WordDetectionResult {
  const wsRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef<FrameSample[]>([]);
  const decodeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [predictions, setPredictions] = useState<WordPrediction[]>([]);
  const [unavailable, setUnavailable] = useState(false);

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (
      existing &&
      (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    const ws = new WebSocket(WS_ENDPOINTS.classifySign);
    ws.onopen = () => setUnavailable(false);
    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "result") {
          setPredictions((data.predictions ?? []) as WordPrediction[]);
        } else if (data.type === "error") {
          setUnavailable(true);
        }
      } catch {
        // ignore malformed payloads
      }
    };
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    if (!active) {
      wsRef.current?.close();
      wsRef.current = null;
      bufferRef.current = [];
      setPredictions([]);
      return;
    }
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [active, connect]);

  // Push each MediaPipe frame into the rolling buffer.
  useEffect(() => {
    if (!active || !detection) return;
    const { features, missing } = buildFrameFeatures(detection.landmarks);
    const buf = bufferRef.current;
    buf.push({ features, missing });
    while (buf.length > CONFIG.bufferFrames) buf.shift();
  }, [detection, active]);

  // Periodic classify tick.
  useEffect(() => {
    if (!active || unavailable) {
      if (decodeTimerRef.current) clearInterval(decodeTimerRef.current);
      decodeTimerRef.current = null;
      return;
    }
    decodeTimerRef.current = setInterval(() => {
      const ws = wsRef.current;
      if (ws?.readyState !== WebSocket.OPEN) {
        connect();
        return;
      }
      const buf = bufferRef.current;
      if (buf.length < 4) return;  // need at least a few frames to bother

      const T = buf.length;
      const flatLandmarks = new Float32Array(T * FRAME_FEATURES);
      const flatMissing = new Uint8Array(T * FRAME_LANDMARKS);
      for (let t = 0; t < T; t++) {
        flatLandmarks.set(buf[t].features, t * FRAME_FEATURES);
        flatMissing.set(buf[t].missing, t * FRAME_LANDMARKS);
      }
      ws.send(
        JSON.stringify({
          type: "classify",
          frame_count: T,
          landmarks: Array.from(flatLandmarks),
          missing: Array.from(flatMissing, (b) => b !== 0),
          top_k: CONFIG.topK,
        }),
      );
    }, CONFIG.decodeCadenceMs);
    return () => {
      if (decodeTimerRef.current) clearInterval(decodeTimerRef.current);
      decodeTimerRef.current = null;
    };
  }, [active, connect, unavailable]);

  return { predictions, unavailable };
}
