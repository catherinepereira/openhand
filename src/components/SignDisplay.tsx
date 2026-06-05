import { useEffect, useRef, useState } from "react";

interface Props {
  sign: string;
  confidence: number;
}

const CONFIDENCE_UPDATE_MS = 500;

export function SignDisplay({ sign, confidence }: Props) {
  const [displayConfidence, setDisplayConfidence] = useState(confidence);
  const lastUpdateRef = useRef(0);
  const pendingRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const now = performance.now();
    const elapsed = now - lastUpdateRef.current;
    if (elapsed >= CONFIDENCE_UPDATE_MS) {
      lastUpdateRef.current = now;
      setDisplayConfidence(confidence);
      return;
    }
    if (pendingRef.current) clearTimeout(pendingRef.current);
    pendingRef.current = setTimeout(() => {
      lastUpdateRef.current = performance.now();
      setDisplayConfidence(confidence);
    }, CONFIDENCE_UPDATE_MS - elapsed);
  }, [confidence]);

  useEffect(
    () => () => {
      if (pendingRef.current) clearTimeout(pendingRef.current);
    },
    [],
  );

  const pct =
    displayConfidence > 0 ? `${Math.round(displayConfidence * 100)}%` : "-";

  return (
    <div className="flex w-full max-w-[360px] gap-3">
      <div className="border-border-app bg-bg flex flex-1 flex-col gap-[0.4rem] rounded-[14px] border-[1.5px] px-[1.2rem] py-4">
        <span className="label-caps">DETECTED</span>
        <span className="text-ink text-[1.6rem] font-light tracking-tight">
          {sign}
        </span>
      </div>
      <div className="border-border-app bg-bg flex flex-1 flex-col gap-[0.4rem] rounded-[14px] border-[1.5px] px-[1.2rem] py-4">
        <span className="label-caps">CONFIDENCE</span>
        <span className="text-ink text-[1.6rem] font-light tracking-tight">
          {pct}
        </span>
      </div>
    </div>
  );
}
