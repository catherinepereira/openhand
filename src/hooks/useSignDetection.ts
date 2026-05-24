import { useEffect, useRef, useState } from "react";
import { classifyHand, warmup } from "../lib/alphabet";
import type { FrameDetection } from "../lib/mediapipe";

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface DetectedHand {
  /** MediaPipe-camera-POV label: "Left" / "Right" / "" if unknown */
  handedness: string;
  landmarks: Landmark[];
}

export interface DetectionResult {
  sign: string;
  confidence: number;
  /** Every visible hand in raw-video coords.
   *  Populated so the overlay stays in sync with the camera regardless of inference latency */
  hands: DetectedHand[];
}

/**
 * Live per-frame letter detection.
 * Runs the alphabet MLP in-browser via onnxruntime-web on each MediaPipe detection
 */
export function useSignDetection(
  detection: FrameDetection | null,
  active: boolean,
) {
  const [result, setResult] = useState<DetectionResult>({
    sign: "-",
    confidence: 0,
    hands: [],
  });
  // Only one classify in flight at a time.
  // Frames arriving during inference are dropped; the next frame catches up
  const inflightRef = useRef(false);

  useEffect(() => {
    warmup().catch((err) => console.error("Alphabet model warmup failed:", err));
  }, []);

  useEffect(() => {
    if (!active) {
      setResult({ sign: "-", confidence: 0, hands: [] });
      return;
    }
  }, [active]);

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

    if (hands.length === 0) {
      setResult((prev) =>
        prev.sign === "-" && prev.hands.length === 0
          ? prev
          : { sign: "-", confidence: 0, hands: [] },
      );
      return;
    }

    if (inflightRef.current) return;
    inflightRef.current = true;
    classifyHand(hands)
      .then(({ sign, confidence }) => {
        setResult({ sign, confidence, hands });
      })
      .catch((err) => {
        console.error("Alphabet inference failed:", err);
      })
      .finally(() => {
        inflightRef.current = false;
      });
  }, [detection, active]);

  return { result };
}
