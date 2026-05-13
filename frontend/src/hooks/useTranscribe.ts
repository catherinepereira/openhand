import { useCallback, useRef, useState } from "react";

const TRANSCRIBE_ENDPOINT = "http://localhost:8273/api/transcribe";
const FRAME_INTERVAL_MS = 100;
// Hard cap on how long a single hold can be — 12s at 10fps = 120 frames.
const MAX_FRAMES = 120;

export interface TranscribeResult {
  text: string;
  frameCount: number;
  elapsedMs: number;
}

export function useTranscribe(videoRef: React.RefObject<HTMLVideoElement>) {
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));
  const framesRef = useRef<string[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lastResult, setLastResult] = useState<TranscribeResult | null>(null);

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.6);
  }, [videoRef]);

  const start = useCallback(() => {
    if (recording || busy) return;
    framesRef.current = [];
    setRecording(true);
    intervalRef.current = setInterval(() => {
      if (framesRef.current.length >= MAX_FRAMES) return;
      const f = captureFrame();
      if (f) framesRef.current.push(f);
    }, FRAME_INTERVAL_MS);
  }, [recording, busy, captureFrame]);

  const stop = useCallback(async (): Promise<TranscribeResult | null> => {
    if (!recording) return null;
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    setRecording(false);
    const frames = framesRef.current;
    framesRef.current = [];
    if (frames.length === 0) return null;

    setBusy(true);
    try {
      const res = await fetch(TRANSCRIBE_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames }),
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
