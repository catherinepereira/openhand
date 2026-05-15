import { useEffect, useRef, useState } from "react";
import {
  createDetectors,
  detectFrame,
  type FrameDetection,
  type MediaPipeDetectors,
} from "../lib/mediapipe";

/** Frame cadence for MediaPipe detection: 10 fps. */
const FRAME_INTERVAL_MS = 100;

/**
 * Shared MediaPipe producer. Initializes Hand/Pose/Face detectors once for
 * the page lifetime, runs them on every webcam frame, and exposes the
 * latest result to any number of consumer hooks.
 *
 * Returns:
 *   - detection:  React state, the latest detection (null until the
 *                 detectors finish loading and the first frame is processed).
 *   - latestRef:  a ref to the same value, for consumers that need to read
 *                 the freshest detection inside a callback without
 *                 re-triggering effects.
 */
export function useMediaPipe(
  videoRef: React.RefObject<HTMLVideoElement>,
  active: boolean,
) {
  const detectorsRef = useRef<MediaPipeDetectors | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const latestRef = useRef<FrameDetection | null>(null);
  const [detection, setDetection] = useState<FrameDetection | null>(null);

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
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      latestRef.current = null;
      setDetection(null);
      return;
    }

    intervalRef.current = setInterval(() => {
      const detectors = detectorsRef.current;
      const video = videoRef.current;
      if (!detectors || !video || video.readyState < 2) return;

      let result: FrameDetection;
      try {
        result = detectFrame(detectors, video, performance.now());
      } catch {
        return;
      }
      latestRef.current = result;
      setDetection(result);
    }, FRAME_INTERVAL_MS);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [active, videoRef]);

  return { detection, latestRef };
}
