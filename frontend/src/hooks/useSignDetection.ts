import { useCallback, useEffect, useRef, useState } from "react";

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface DetectionResult {
  sign: string;
  confidence: number;
  landmarks: Landmark[];
}

const WS_URL = "ws://localhost:8273/ws/detect";
const FRAME_INTERVAL_MS = 100;

export function useSignDetection(
  videoRef: React.RefObject<HTMLVideoElement>,
  active: boolean
) {
  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [result, setResult] = useState<DetectionResult>({ sign: "—", confidence: 0, landmarks: [] });
  const [connected, setConnected] = useState(false);

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      if (wsRef.current === ws) wsRef.current = null;
    };
    ws.onerror = () => { setConnected(false); };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setResult({
          sign: data.sign ?? "—",
          confidence: data.confidence ?? 0,
          landmarks: Array.isArray(data.landmarks) ? data.landmarks : [],
        });
      } catch {
        // ignore malformed frames
      }
    };
    wsRef.current = ws;
  }, []);

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

  useEffect(() => {
    if (!active) {
      intervalRef.current && clearInterval(intervalRef.current);
      intervalRef.current = null;
      wsRef.current?.close();
      wsRef.current = null;
      setResult({ sign: "—", confidence: 0, landmarks: [] });
      return;
    }

    connect();

    intervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        connect();
        return;
      }
      const frame = captureFrame();
      if (frame) wsRef.current.send(frame);
    }, FRAME_INTERVAL_MS);

    return () => {
      intervalRef.current && clearInterval(intervalRef.current);
      intervalRef.current = null;
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [active, connect, captureFrame]);

  return { result, connected };
}
