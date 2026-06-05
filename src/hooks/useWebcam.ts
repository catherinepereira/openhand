import { useCallback, useEffect, useRef, useState } from "react";

export type WebcamStatus = "idle" | "requesting" | "active" | "error";

export function useWebcam() {
  // videoRef is exposed as a normal ref so callers can pass it to <video>, but the *current* element is also tracked via a callback ref.
  // When the consumer remounts the <video> in a different view (e.g. switching from the home screen to the Learn screen), the callback re-attaches the live MediaStream to the new element.
  // Without this, the stream stays bound to the unmounted element and the new <video> is blank
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const attachStreamTo = useCallback((el: HTMLVideoElement | null) => {
    if (!el) return;
    if (el.srcObject !== streamRef.current) {
      el.srcObject = streamRef.current;
    }
    if (streamRef.current && el.paused) {
      el.play().catch(() => {
        // Autoplay can fail on iOS Safari without a user gesture; ignore
      });
    }
  }, []);

  const setVideoRef = useCallback(
    (el: HTMLVideoElement | null) => {
      videoRef.current = el;
      attachStreamTo(el);
    },
    [attachStreamTo],
  );

  const start = async () => {
    setStatus("requesting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      attachStreamTo(videoRef.current);
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

  useEffect(
    () => () => {
      stop();
    },
    [],
  );

  return {
    /** Pass to JSX: `<video ref={videoRef} />`.
     *  Re-attaches the stream if React remounts the element (e.g. switching views) */
    videoRef: setVideoRef,
    /** For consumers that need to read the underlying element (canvas sizing, MediaPipe input, etc).
     *  Always points at the most recent mounted <video> */
    videoElementRef: videoRef,
    status,
    error,
    start,
    stop,
  };
}
