import { useEffect, useRef, useState } from "react";

type Mode = "media" | "animation";

interface Props {
  /** Path under /public, e.g. "/reference-letters/A.svg" or "/reference-signs/airplane.mp4". */
  imageSrc: string;
  /** Rendered when mode is "animation" or when imageSrc fails to load. */
  animation: React.ReactNode;
  /** Defaults to "media". */
  initialMode?: Mode;
}

const VIDEO_EXTS = [".mp4", ".webm", ".mov"];

function isVideoPath(src: string): boolean {
  const lower = src.toLowerCase();
  return VIDEO_EXTS.some((ext) => lower.endsWith(ext));
}

export function ReferenceView({ imageSrc, animation, initialMode = "media" }: Props) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [mediaMissing, setMediaMissing] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    setMediaMissing(false);
  }, [imageSrc]);

  const isVideo = isVideoPath(imageSrc);
  const showAnimation = mode === "animation" || mediaMissing;
  const mediaLabel = isVideo ? "Video" : "Image";

  return (
    <div className="relative h-full w-full">
      {showAnimation ? (
        animation
      ) : (
        <div className="flex h-full w-full items-center justify-center">
          {isVideo ? (
            <video
              ref={videoRef}
              src={imageSrc}
              autoPlay
              loop
              muted
              playsInline
              onError={() => setMediaMissing(true)}
              className="max-h-full max-w-full object-contain"
            />
          ) : (
            <img
              src={imageSrc}
              alt=""
              onError={() => setMediaMissing(true)}
              className="max-h-[40%] max-w-[40%] object-contain"
            />
          )}
        </div>
      )}
      <div className="absolute right-2 top-2 flex overflow-hidden rounded-md border-[1.5px] border-border-app bg-bg/90 text-[0.72rem] font-medium backdrop-blur">
        <button
          onClick={() => setMode("media")}
          disabled={mediaMissing}
          className={[
            "px-2 py-1 transition-colors",
            mode === "media" && !mediaMissing
              ? "bg-ink text-white"
              : "text-ink hover:enabled:bg-surface disabled:opacity-40",
          ].join(" ")}
        >
          {mediaLabel}
        </button>
        <button
          onClick={() => setMode("animation")}
          className={[
            "px-2 py-1 transition-colors",
            mode === "animation" || mediaMissing
              ? "bg-ink text-white"
              : "text-ink hover:bg-surface",
          ].join(" ")}
        >
          3D
        </button>
      </div>
    </div>
  );
}
