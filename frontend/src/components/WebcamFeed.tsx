import { useEffect, useRef } from "react";
import { HandIcon } from "./icons";
import type { Landmark } from "../hooks/useSignDetection";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  status: "idle" | "requesting" | "active" | "error";
  error: string | null;
  landmarks: Landmark[];
}

const STATUS_LABEL: Record<string, string> = {
  idle: "Waiting for camera",
  requesting: "Requesting access…",
  active: "Camera active",
  error: "Camera error",
};

// MediaPipe hand connections — pairs of landmark indices forming bones
const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

export function WebcamFeed({ videoRef, status, error, landmarks }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas pixel size to its CSS size so coordinates are in pixels
    const { width, height } = canvas.getBoundingClientRect();
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks.length) return;

    // Video is mirrored via CSS (scaleX(-1)); mirror landmark x so the overlay
    // visually aligns with the user's hand on screen.
    const points = landmarks.map((lm) => ({
      x: (1 - lm.x) * canvas.width,
      y: lm.y * canvas.height,
    }));

    // Bones
    ctx.strokeStyle = "rgba(60, 130, 240, 0.85)";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (const [a, b] of HAND_CONNECTIONS) {
      if (!points[a] || !points[b]) continue;
      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    }

    // Joints
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.strokeStyle = "rgba(60, 130, 240, 1)";
    ctx.lineWidth = 1.5;
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [landmarks]);

  return (
    <div className="webcam-wrap">
      <div className="webcam-frame">
        {status !== "active" && (
          <div className="webcam-placeholder">
            <HandIcon />
          </div>
        )}
        <video
          ref={videoRef}
          playsInline
          muted
          className={`webcam-video ${status === "active" ? "visible" : ""}`}
        />
        <canvas ref={overlayRef} className="webcam-overlay" />
      </div>
      <p className="webcam-status">
        <span className={`status-dot ${status}`} />
        {error ?? STATUS_LABEL[status]}
      </p>
    </div>
  );
}
