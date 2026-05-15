import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import type { FrameDetection } from "../lib/mediapipe";

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface DetectedHand {
  /** MediaPipe-camera-POV label: "Left" / "Right" / "" if unknown. */
  handedness: string;
  landmarks: Landmark[];
}

export interface DetectionResult {
  sign: string;
  confidence: number;
  /** Every visible hand in raw-video coords. Populated client-side so
   *  the overlay stays in sync with the camera regardless of backend
   *  round-trip latency. */
  hands: DetectedHand[];
}

/**
 * Live per-frame letter detection. Consumes MediaPipe output from
 * useMediaPipe and forwards each frame's hands to the backend alphabet
 * MLP over a WebSocket.
 */
export function useSignDetection(
  detection: FrameDetection | null,
  active: boolean,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const overlayHandsRef = useRef<DetectedHand[]>([]);
  const [result, setResult] = useState<DetectionResult>({
    sign: "-",
    confidence: 0,
    hands: [],
  });
  const [connected, setConnected] = useState(false);

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (
      existing &&
      (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    const ws = new WebSocket(WS_ENDPOINTS.detect);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      if (wsRef.current === ws) wsRef.current = null;
    };
    ws.onerror = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setResult({
          sign: data.sign ?? "-",
          confidence: data.confidence ?? 0,
          hands: overlayHandsRef.current,
        });
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
      setResult({ sign: "-", confidence: 0, hands: [] });
      return;
    }
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [active, connect]);

  useEffect(() => {
    if (!active || !detection) return;

    const hands: DetectedHand[] = [];
    const left = detection.landmarks.leftHand;
    const right = detection.landmarks.rightHand;
    if (left !== null && left.length === 21) {
      hands.push({
        handedness: "Left",
        landmarks: left.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
      });
    }
    if (right !== null && right.length === 21) {
      hands.push({
        handedness: "Right",
        landmarks: right.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
      });
    }
    overlayHandsRef.current = hands;

    if (hands.length === 0) {
      setResult((prev) =>
        prev.sign === "-" && prev.hands.length === 0
          ? prev
          : { sign: "-", confidence: 0, hands: [] },
      );
      return;
    }

    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      connect();
      return;
    }
    wsRef.current.send(JSON.stringify({ hands }));
  }, [detection, active, connect]);

  return { result, connected };
}
