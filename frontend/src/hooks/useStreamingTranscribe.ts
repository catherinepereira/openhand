import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import { FRAME_FEATURES, FRAME_LANDMARKS, buildFrameFeatures } from "../lib/landmarks";
import type { FrameDetection } from "../lib/mediapipe";

/** Timing and smoothing knobs for streaming transcription. */
const TRANSCRIBE_CONFIG = {
  /** Rolling-buffer length, in frames. At 10fps this is ~3 seconds. */
  bufferFrames: 30,
  /** How often we send the buffer to the backend for re-decode, in ms. */
  decodeCadenceMs: 750,
  /** Per-frame landmark-movement threshold (in MediaPipe normalized
   *  coords) below which the hand is considered "still." */
  stillnessThreshold: 0.01,
  /** Consecutive still-hand frames after which we commit the current
   *  tentative decode to the final text. At 10fps, 6 frames = 600 ms,
   *  which is a comfortable mid-phrase pause. */
  stillnessFramesToCommit: 6,
  /** Consecutive no-hand frames after which we wipe both the buffer
   *  and the tentative text, ending the current phrase silently. */
  silenceFramesToClear: 10,
} as const;

export interface StreamingTranscribeResult {
  /** Committed transcription so far: phrases the user has paused on,
   *  joined with single spaces. Accumulates until clear() is called. */
  text: string;
  /** Latest in-flight decode of the rolling buffer. Updates every tick;
   *  expected to fluctuate as the model revises its guess. */
  tentative: string;
  /** Backend-reported per-decode latency, last call only. */
  lastDecodeMs: number;
  /** True while the decode WebSocket is open. */
  connected: boolean;
  /** Imperative clear of the committed text. */
  clear: () => void;
}

interface FrameSample {
  features: Float32Array;
  missing: Uint8Array;
}

/**
 * Always-on streaming CTC transcription with commit-on-pause semantics.
 *
 * Behavior:
 *  1. Every webcam frame the user's dominant hand is detected, we push
 *     its packed features into a rolling buffer.
 *  2. Every decodeCadenceMs we send the buffer to the backend and receive
 *     a fresh decode. This is shown as tentative and fluctuates per tick.
 *  3. When the hand goes still for stillnessFramesToCommit frames (a
 *     natural mid-phrase pause), we commit the latest tentative decode
 *     to text, append a separator, and reset for the next phrase.
 *  4. When the hand leaves the frame for silenceFramesToClear frames, we
 *     discard the tentative text without committing.
 *
 * The model's per-tick output fluctuates a lot across sub-windows, so we
 * don't smooth across decodes; we trust the last guess at the pause point,
 * when the buffer fully covers a single phrase.
 */
export function useStreamingTranscribe(
  detection: FrameDetection | null,
  active: boolean,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef<FrameSample[]>([]);
  const decodeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tentativeRef = useRef("");
  const lastWristRef = useRef<{ x: number; y: number; z: number } | null>(null);
  const stillCountRef = useRef(0);
  const silenceCountRef = useRef(0);

  const [text, setText] = useState("");
  const [tentative, setTentative] = useState("");
  const [lastDecodeMs, setLastDecodeMs] = useState(0);
  const [connected, setConnected] = useState(false);

  const resetPhraseState = useCallback(() => {
    bufferRef.current = [];
    tentativeRef.current = "";
    setTentative("");
    lastWristRef.current = null;
    stillCountRef.current = 0;
  }, []);

  const clear = useCallback(() => {
    setText("");
    resetPhraseState();
  }, [resetPhraseState]);

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (
      existing &&
      (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    const ws = new WebSocket(WS_ENDPOINTS.transcribeStream);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      if (wsRef.current === ws) wsRef.current = null;
    };
    ws.onerror = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type !== "result") return;
        const newDecode: string = data.text ?? "";
        tentativeRef.current = newDecode;
        setTentative(newDecode);
        setLastDecodeMs(data.elapsed_ms ?? 0);
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
      resetPhraseState();
      silenceCountRef.current = 0;
      return;
    }
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [active, connect, resetPhraseState]);

  useEffect(() => {
    if (!active || !detection) return;

    const right = detection.landmarks.rightHand;
    const left = detection.landmarks.leftHand;
    const dominant = right ?? left;

    if (dominant === null) {
      silenceCountRef.current++;
      stillCountRef.current = 0;
      lastWristRef.current = null;
      if (silenceCountRef.current >= TRANSCRIBE_CONFIG.silenceFramesToClear) {
        if (bufferRef.current.length > 0 || tentativeRef.current.length > 0) {
          resetPhraseState();
        }
      }
      return;
    }

    silenceCountRef.current = 0;

    const { features, missing } = buildFrameFeatures(detection.landmarks);
    const buf = bufferRef.current;
    buf.push({ features, missing });
    while (buf.length > TRANSCRIBE_CONFIG.bufferFrames) buf.shift();

    const wrist = dominant[0];
    const prev = lastWristRef.current;
    lastWristRef.current = { x: wrist.x, y: wrist.y, z: wrist.z };
    if (prev === null) return;

    const dx = wrist.x - prev.x;
    const dy = wrist.y - prev.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < TRANSCRIBE_CONFIG.stillnessThreshold) {
      stillCountRef.current++;
      if (stillCountRef.current === TRANSCRIBE_CONFIG.stillnessFramesToCommit) {
        const decoded = tentativeRef.current.trim();
        if (decoded.length > 0) {
          setText((prev) => (prev ? prev + " " : "") + decoded);
        }
        resetPhraseState();
      }
    } else {
      stillCountRef.current = 0;
    }
  }, [detection, active, resetPhraseState]);

  useEffect(() => {
    if (!active) {
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
    }, TRANSCRIBE_CONFIG.decodeCadenceMs);

    return () => {
      if (decodeTimerRef.current) clearInterval(decodeTimerRef.current);
      decodeTimerRef.current = null;
    };
  }, [active, connect]);

  return { text, tentative, lastDecodeMs, connected, clear };
}
