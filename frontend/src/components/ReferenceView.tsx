import { useEffect, useState } from "react";

type Mode = "image" | "animation";

interface Props {
  /** Path under /public, e.g. "/reference-letters/A.svg". */
  imageSrc: string;
  /** Rendered when mode is "animation" or when imageSrc fails to load. */
  animation: React.ReactNode;
  /** Defaults to "image". */
  initialMode?: Mode;
}

export function ReferenceView({ imageSrc, animation, initialMode = "image" }: Props) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [imageMissing, setImageMissing] = useState(false);

  useEffect(() => {
    setImageMissing(false);
  }, [imageSrc]);

  const showAnimation = mode === "animation" || imageMissing;

  return (
    <div className="relative h-full w-full">
      {showAnimation ? (
        animation
      ) : (
        <div className="flex h-full w-full items-center justify-center">
          <img
            src={imageSrc}
            alt=""
            onError={() => setImageMissing(true)}
            className="max-h-[40%] max-w-[40%] object-contain"
          />
        </div>
      )}
      <div className="absolute right-2 top-2 flex overflow-hidden rounded-md border-[1.5px] border-border-app bg-bg/90 text-[0.72rem] font-medium backdrop-blur">
        <button
          onClick={() => setMode("image")}
          disabled={imageMissing}
          className={[
            "px-2 py-1 transition-colors",
            mode === "image" && !imageMissing
              ? "bg-ink text-white"
              : "text-ink hover:enabled:bg-surface disabled:opacity-40",
          ].join(" ")}
        >
          Image
        </button>
        <button
          onClick={() => setMode("animation")}
          className={[
            "px-2 py-1 transition-colors",
            mode === "animation" || imageMissing
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
