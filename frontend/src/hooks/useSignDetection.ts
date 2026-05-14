import { useCallback, useEffect, useRef, useState } from "react";
import { WS_ENDPOINTS } from "../config";
import { createDetectors, detectFrame, type MediaPipeDetectors } from "../lib/mediapipe";

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
  /** Every visible hand in raw-video coords, populated client-side from
   *  MediaPipe so the overlay stays in sync regardless of backend
   *  round-trip latency. */
  hands: DetectedHand[];
}

const FRAME_INTERVAL_MS = 100;

/**
 * Live per-frame letter detection.
 *
 * Runs MediaPipe in the browser, sends only the 21 hand landmarks over
 * the WebSocket (~1-2 KB/frame instead of ~30-50 KB of JPEG).
 */
export function useSignDetection(
  videoRef: React.RefObject<HTMLVideoElement>,
  active: boolean,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const detectorsRef = useRef<MediaPipeDetectors | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [result, setResult] = useState<DetectionResult>({
    sign: "—",
    confidence: 0,
    hands: [],
  });
  const [connected, setConnected] = useState(false);

  // Latest client-side overlay hands (raw-video coords). Spliced into
  // `result` when the backend replies so the overlay always reflects the
  // current frame, not the one we sent.
  const overlayHandsRef = useRef<DetectedHand[]>([]);

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
          sign: data.sign ?? "—",
          confidence: data.confidence ?? 0,
          hands: overlayHandsRef.current,
        });
      } catch {
        // ignore malformed payloads
      }
    };
    wsRef.current = ws;
  }, []);

  // Mount detectors once for the lifetime of this hook instance.
  useEffect(() => {
    let cancelled = false;
    createDetectors("image")
      .then((d) => {
        if (cancelled) {
          d.close();
          return;
        }
        detectorsRef.current = d;
      })
      .catch((err) => {
        console.error("MediaPipe detector init failed:", err);
      });
    return () => {
      cancelled = true;
      detectorsRef.current?.close();
      detectorsRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!active) {
      intervalRef.current && clearInterval(intervalRef.current);
      intervalRef.current = null;
      wsRef.current?.close();
      wsRef.current = null;
      setResult({ sign: "—", confidence: 0, hands: [] });
      return;
    }

    connect();

    intervalRef.current = setInterval(() => {
      const detectors = detectorsRef.current;
      const video = videoRef.current;
      if (!detectors || !video || video.readyState < 2) return;

      let detection;
      try {
        detection = detectFrame(detectors, video, performance.now());
      } catch {
        return;
      }

      // Collect every visible hand for the overlay, tagged with its
      // MediaPipe handedness label.
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
          prev.sign === "—" && prev.hands.length === 0
            ? prev
            : { sign: "—", confidence: 0, hands: [] },
        );
        return;
      }

      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        connect();
        return;
      }

      // Send every detected hand with its handedness label; the backend
      // picks the "Right" hand to classify.
      wsRef.current.send(JSON.stringify({ hands }));
    }, FRAME_INTERVAL_MS);

    return () => {
      intervalRef.current && clearInterval(intervalRef.current);
      intervalRef.current = null;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [active, connect, videoRef]);

  return { result, connected };
}
