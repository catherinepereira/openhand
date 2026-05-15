import { useEffect, useRef } from "react";
import { HandIcon } from "./icons";
import type { DetectedHand } from "../hooks/useSignDetection";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  status: "idle" | "requesting" | "active" | "error";
  error: string | null;
  /** Every visible hand, tagged with handedness, in MediaPipe normalized
   *  coords (x, y in [0,1] of the raw unmirrored frame). */
  hands: DetectedHand[];
}

const STATUS_LABEL: Record<string, string> = {
  idle: "Waiting for camera",
  requesting: "Requesting access…",
  active: "Camera active",
  error: "Camera error",
};

// MediaPipe hand connections: pairs of landmark indices forming bones.
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],                  // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],                  // Index
  [5, 9], [9, 10], [10, 11], [11, 12],             // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],           // Ring
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
];

const BONE_COLOR = "rgba(60, 130, 240, 0.85)";
const JOINT_COLOR = "rgba(60, 130, 240, 1)";
const JOINT_FILL = "rgba(255, 255, 255, 0.95)";
const LABEL_BG = "rgba(60, 130, 240, 0.9)";

function drawHand(
  ctx: CanvasRenderingContext2D,
  hand: DetectedHand,
  canvasW: number,
  canvasH: number,
) {
  if (hand.landmarks.length !== 21) return;

  // MediaPipe coords are raw-video (unmirrored). The displayed video has
  // CSS scaleX(-1), so flip x to align the overlay with the mirrored visual.
  const points = hand.landmarks.map((lm) => ({
    x: (1 - lm.x) * canvasW,
    y: lm.y * canvasH,
  }));

  ctx.strokeStyle = BONE_COLOR;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(points[a].x, points[a].y);
    ctx.lineTo(points[b].x, points[b].y);
    ctx.stroke();
  }

  ctx.fillStyle = JOINT_FILL;
  ctx.strokeStyle = JOINT_COLOR;
  ctx.lineWidth = 1.5;
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Handedness label near the wrist (only when we have a confident label).
  if (hand.handedness) {
    const wrist = points[0];
    ctx.font = "12px system-ui, sans-serif";
    const pad = 4;
    const w = ctx.measureText(hand.handedness).width + pad * 2;
    const h = 16;
    const x = wrist.x + 8;
    const y = wrist.y - h - 4;
    ctx.fillStyle = LABEL_BG;
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = "white";
    ctx.fillText(hand.handedness, x + pad, y + 12);
  }
}

export function WebcamFeed({ videoRef, status, error, hands }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const video = videoRef.current;
    const videoW = video?.videoWidth ?? 0;
    const videoH = video?.videoHeight ?? 0;
    if (videoW === 0 || videoH === 0) return;

    // Match canvas backing buffer to the video stream's native resolution.
    // MediaPipe x/y are normalized to source frame size; CSS scales the
    // canvas display rect down to share the same box as the video.
    if (canvas.width !== videoW) canvas.width = videoW;
    if (canvas.height !== videoH) canvas.height = videoH;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const h of hands) drawHand(ctx, h, canvas.width, canvas.height);
  }, [hands, videoRef]);

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
