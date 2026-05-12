import { useEffect, useRef, useState } from "react";

export type WebcamStatus = "idle" | "requesting" | "active" | "error";

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    setStatus("requesting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus("active");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Camera access denied");
      setStatus("error");
    }
  };

  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  };

  useEffect(() => () => { stop(); }, []);

  return { videoRef, status, error, start, stop };
}
