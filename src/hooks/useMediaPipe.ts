import { useEffect, useRef, useState } from "react";
import {
  createDetectors,
  detectFrame,
  type FrameDetection,
  type MediaPipeDetectors,
} from "../lib/mediapipe";

/**
 * Shared MediaPipe producer
 * Initializes Hand/Pose/Face detectors once for the page lifetime, runs them on every new webcam frame, and exposes the latest detection as React state.
 * The state is null until the detectors finish loading and the first frame is processed
 *
 * The capture loop is requestAnimationFrame, matching the official MediaPipe web demos, so detection is paced to the display refresh instead of a fixed 10fps timer.
 * An unchanged video frame is skipped so the same image is never detected twice
 */
export function useMediaPipe(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  active: boolean,
) {
  const detectorsRef = useRef<MediaPipeDetectors | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
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
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastVideoTimeRef.current = -1;
      setDetection(null);
      return;
    }

    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);
      const detectors = detectorsRef.current;
      const video = videoRef.current;
      if (!detectors || !video || video.readyState < 2) return;
      // Skip frames the camera has not advanced past yet, so the same image is
      // never run through the three detectors twice
      if (video.currentTime === lastVideoTimeRef.current) return;
      lastVideoTimeRef.current = video.currentTime;

      let result: FrameDetection;
      try {
        result = detectFrame(detectors, video, performance.now());
      } catch {
        return;
      }
      setDetection(result);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [active, videoRef]);

  return { detection };
}
