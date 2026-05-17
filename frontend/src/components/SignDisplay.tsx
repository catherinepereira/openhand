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

  useEffect(() => () => {
    if (pendingRef.current) clearTimeout(pendingRef.current);
  }, []);

  const pct = displayConfidence > 0 ? `${Math.round(displayConfidence * 100)}%` : "-";

  return (
    <div className="sign-display-row">
      <div className="stat-card">
        <span className="stat-label">DETECTED</span>
        <span className="stat-value">{sign}</span>
      </div>
      <div className="stat-card">
        <span className="stat-label">CONFIDENCE</span>
        <span className="stat-value">{pct}</span>
      </div>
    </div>
  );
}
