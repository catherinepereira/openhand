import { useCallback, useEffect, useRef, useState } from "react";
import { HTTP_ENDPOINTS } from "../config";
import {
  FRAME_FEATURES,
  FRAME_LANDMARKS,
  buildFrameFeatures,
} from "../lib/landmarks";
import { createDetectors, detectFrame, type MediaPipeDetectors } from "../lib/mediapipe";

const FRAME_INTERVAL_MS = 100;
// Hard cap on how long a single hold can be — 12s at 10fps = 120 frames.
const MAX_FRAMES = 120;

export interface TranscribeResult {
  text: string;
  frameCount: number;
  elapsedMs: number;
}

interface FrameBuffer {
  /** Flat (T * FRAME_FEATURES,) Float32Array, row-major. */
  features: number[];
  /** Flat (T * FRAME_LANDMARKS,) bool, row-major. */
  missing: boolean[];
  frameCount: number;
}

/**
 * Hold-to-record phrase transcription.
 *
 * While the user holds the button, MediaPipe extracts landmarks in the
 * browser every 100ms and pushes the packed features + missing mask into
 * a rolling buffer. On release we POST the buffer to the backend, which
 * runs the CTC model and returns the transcription.
 */
export function useTranscribe(videoRef: React.RefObject<HTMLVideoElement>) {
  const detectorsRef = useRef<MediaPipeDetectors | null>(null);
  const bufferRef = useRef<FrameBuffer>({ features: [], missing: [], frameCount: 0 });
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lastResult, setLastResult] = useState<TranscribeResult | null>(null);

  // Eager-init detectors so the first hold doesn't pay the model-load cost.
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

  const captureFrame = useCallback(() => {
    const detectors = detectorsRef.current;
    const video = videoRef.current;
    if (!detectors || !video || video.readyState < 2) return;

    let detection;
    try {
      detection = detectFrame(detectors, video, performance.now());
    } catch {
      return;
    }

    const { features, missing } = buildFrameFeatures(detection.landmarks);
    const buf = bufferRef.current;
    for (let i = 0; i < features.length; i++) buf.features.push(features[i]);
    for (let i = 0; i < missing.length; i++) buf.missing.push(missing[i] !== 0);
    buf.frameCount++;
  }, [videoRef]);

  const start = useCallback(() => {
    if (recording || busy) return;
    bufferRef.current = { features: [], missing: [], frameCount: 0 };
    setRecording(true);
    intervalRef.current = setInterval(() => {
      if (bufferRef.current.frameCount >= MAX_FRAMES) return;
      captureFrame();
    }, FRAME_INTERVAL_MS);
  }, [recording, busy, captureFrame]);

  const stop = useCallback(async (): Promise<TranscribeResult | null> => {
    if (!recording) return null;
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    setRecording(false);

    const buf = bufferRef.current;
    bufferRef.current = { features: [], missing: [], frameCount: 0 };
    if (buf.frameCount === 0) return null;

    setBusy(true);
    try {
      const res = await fetch(HTTP_ENDPOINTS.transcribe, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame_count: buf.frameCount,
          features: buf.features,
          missing: buf.missing,
        }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      const result: TranscribeResult = {
        text: data.text ?? "",
        frameCount: data.frame_count ?? 0,
        elapsedMs: data.elapsed_ms ?? 0,
      };
      setLastResult(result);
      return result;
    } catch {
      return null;
    } finally {
      setBusy(false);
    }
  }, [recording]);

  return { recording, busy, lastResult, start, stop };
}

// Module-load sanity: detect at build time if FRAME_FEATURES /
// FRAME_LANDMARKS ever diverges, since this file constructs the payload
// directly without further reshaping.
if (FRAME_FEATURES !== FRAME_LANDMARKS * 3) {
  throw new Error(
    `useTranscribe expects FRAME_FEATURES == FRAME_LANDMARKS * 3, got ` +
      `${FRAME_FEATURES} vs ${FRAME_LANDMARKS * 3}`,
  );
}
