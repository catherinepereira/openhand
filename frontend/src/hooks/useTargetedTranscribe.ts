import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import { FRAME_FEATURES, FRAME_LANDMARKS, GROUP_OFFSETS, GROUP_SIZES, buildFrameFeatures } from "../lib/landmarks";
import type { FrameDetection } from "../lib/mediapipe";

function hasHand(missing: Uint8Array): boolean {
  // Hand-present = at least one non-missing landmark in either hand group.
  // The CTC model learned to read fingerspelling from the hands; sending a
  // frame with both hands absent feeds it pure noise and produces spurious
  // decodes.
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
 * Per-target CTC scoring for the Learn screen.
 *
 * Keeps a rolling ~4s window of frames and re-decodes it on a timer,
 * exposing the latest decoded string. 4s covers most short fingerspelled
 * words end-to-end, which matters because the CTC model is trained on
 * whole sequences; a 2s slice catches mid-word and decodes badly.
 *
 * Decode cadence of 900ms is below the buffer length, so each tick
 * mostly re-decodes the same content (stable text) plus a sliver of new
 * frames at the tail.
 */
const CONFIG = {
  bufferFrames: 40,
  decodeCadenceMs: 900,
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

  // Push each MediaPipe frame into the rolling buffer, but only when a
  // hand is actually visible. Hand-absent frames feed the CTC model noise
  // and cause spurious decodes.
  const lastHandTimeRef = useRef<number>(0);
  useEffect(() => {
    if (!active || !detection) return;
    const { features, missing } = buildFrameFeatures(detection.landmarks);
    if (!hasHand(missing)) {
      // Clear stale buffer + decode once the user has put their hands down
      // long enough that the displayed text is unrelated to current intent.
      if (lastHandTimeRef.current && performance.now() - lastHandTimeRef.current > 2000) {
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
