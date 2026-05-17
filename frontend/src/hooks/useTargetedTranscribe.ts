import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import { FRAME_FEATURES, FRAME_LANDMARKS, buildFrameFeatures } from "../lib/landmarks";
import type { FrameDetection } from "../lib/mediapipe";

/**
 * Per-target CTC scoring for the Learn screen.
 *
 * Keeps a rolling ~2s window of frames and re-decodes it on a timer,
 * exposing the latest decoded string so the caller can match it against
 * a target letter.
 *
 * The CTC model needs motion to disambiguate letters like J and Z, so
 * the rolling window covers ~2 seconds at 10 fps. Decode cadence is
 * 600ms which keeps feedback snappy without saturating CPU.
 */
const CONFIG = {
  bufferFrames: 20,
  decodeCadenceMs: 600,
} as const;

interface FrameSample {
  features: Float32Array;
  missing: Uint8Array;
}

export interface TargetedTranscribeResult {
  /** The most recent CTC decode of the rolling buffer. */
  latest: string;
  /** True if the CTC model failed to load on the server side. */
  unavailable: boolean;
}

export function useTargetedTranscribe(
  detection: FrameDetection | null,
  active: boolean,
): TargetedTranscribeResult {
  const wsRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef<FrameSample[]>([]);
  const decodeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [latest, setLatest] = useState("");
  const [unavailable, setUnavailable] = useState(false);

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (
      existing &&
      (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    const ws = new WebSocket(WS_ENDPOINTS.transcribeStream);
    ws.onopen = () => setUnavailable(false);
    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
    };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "result") {
          setLatest(data.text ?? "");
        } else if (data.type === "error") {
          // The backend sends this then closes if the CTC ONNX is missing.
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
      setLatest("");
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

  // Periodic decode tick.
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
      if (buf.length === 0) return;

      const T = buf.length;
      const flatFeatures = new Float32Array(T * FRAME_FEATURES);
      const flatMissing = new Uint8Array(T * FRAME_LANDMARKS);
      for (let t = 0; t < T; t++) {
        flatFeatures.set(buf[t].features, t * FRAME_FEATURES);
        flatMissing.set(buf[t].missing, t * FRAME_LANDMARKS);
      }

      ws.send(
        JSON.stringify({
          type: "decode",
          frame_count: T,
          features: Array.from(flatFeatures),
          missing: Array.from(flatMissing, (b) => b !== 0),
        }),
      );
    }, CONFIG.decodeCadenceMs);

    return () => {
      if (decodeTimerRef.current) clearInterval(decodeTimerRef.current);
      decodeTimerRef.current = null;
    };
  }, [active, connect, unavailable]);

  return { latest, unavailable };
}
