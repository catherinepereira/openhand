interface Props {
  sign: string;
  confidence: number;
}

export function SignDisplay({ sign, confidence }: Props) {
  const pct = confidence > 0 ? `${Math.round(confidence * 100)}%` : "—";

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
