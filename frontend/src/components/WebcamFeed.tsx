import { useEffect, useRef } from "react";
import { HandIcon } from "./icons";
import type { DetectedHand } from "../hooks/useSignDetection";

interface Props {
  videoRef: (el: HTMLVideoElement | null) => void;
  status: "idle" | "requesting" | "active" | "error";
  error: string | null;
  hands: DetectedHand[];
  showSkeleton: boolean;
  onShowSkeletonChange: (next: boolean) => void;
}

const STATUS_LABEL: Record<string, string> = {
  idle: "Waiting for camera",
  requesting: "Requesting access...",
  active: "Camera active",
  error: "Camera error",
};

const STATUS_DOT: Record<string, string> = {
  idle: "bg-muted",
  requesting: "bg-amber-500",
  active: "bg-green-500",
  error: "bg-red-500",
};

const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
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

export function WebcamFeed({
  videoRef,
  status,
  error,
  hands,
  showSkeleton,
  onShowSkeletonChange,
}: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const videoEl = useRef<HTMLVideoElement | null>(null);

  const setVideo = (el: HTMLVideoElement | null) => {
    videoEl.current = el;
    videoRef(el);
  };

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const video = videoEl.current;
    const videoW = video?.videoWidth ?? 0;
    const videoH = video?.videoHeight ?? 0;
    if (videoW === 0 || videoH === 0) return;

    if (canvas.width !== videoW) canvas.width = videoW;
    if (canvas.height !== videoH) canvas.height = videoH;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const h of hands) drawHand(ctx, h, canvas.width, canvas.height);
  }, [hands]);

  return (
    <div className="flex w-full flex-col items-center gap-3">
      <div className="relative flex aspect-[4/3] w-[min(760px,96%)] items-center justify-center overflow-hidden rounded-3xl border border-border-app bg-surface shadow-[0_2px_16px_rgba(0,0,0,0.05)]">
        {status !== "active" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <HandIcon />
          </div>
        )}
        <video
          ref={setVideo}
          playsInline
          muted
          className={`absolute inset-0 h-full w-full object-fill transition-opacity duration-300 [transform:scaleX(-1)] ${status === "active" ? "opacity-100" : "opacity-0"}`}
        />
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />

        {status === "active" && (
          <label className="absolute top-3 right-3 z-10 inline-flex cursor-pointer select-none items-center gap-[0.55rem] rounded-full bg-black/55 px-[0.65rem] py-[0.4rem] text-[0.78rem] font-medium text-white backdrop-blur-md">
            <span className="leading-none">Skeleton</span>
            <span className="relative inline-block h-4 w-[30px]">
              <input
                type="checkbox"
                checked={showSkeleton}
                onChange={(e) => onShowSkeletonChange(e.target.checked)}
                className="peer absolute inset-0 z-10 m-0 cursor-pointer opacity-0"
              />
              <span
                aria-hidden="true"
                className="absolute inset-0 rounded-full bg-white/30 transition-colors peer-checked:bg-blue-500 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-white/70"
              />
              <span
                aria-hidden="true"
                className="pointer-events-none absolute left-[2px] top-[2px] h-3 w-3 rounded-full bg-white shadow-[0_1px_2px_rgba(0,0,0,0.35)] transition-transform peer-checked:translate-x-[14px]"
              />
            </span>
          </label>
        )}
      </div>
      <p className="flex items-center gap-[0.4rem] text-[0.82rem] text-muted">
        <span className={`h-[7px] w-[7px] shrink-0 rounded-full ${STATUS_DOT[status]}`} />
        {error ?? STATUS_LABEL[status]}
      </p>
    </div>
  );
}
