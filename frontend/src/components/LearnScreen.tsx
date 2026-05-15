import { useEffect, useMemo, useState } from "react";
import { HTTP_ENDPOINTS } from "../config";
import type { DetectionResult } from "../hooks/useSignDetection";
import { HandModel3D } from "./HandModel3D";
import { WebcamFeed } from "./WebcamFeed";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

interface ReferencePayload {
  format: string;
  n_features: number;
  letters: Record<string, number[]>;
  sample_counts: Record<string, number>;
}

type Grade = "green" | "yellow" | "red" | "none";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  status: "idle" | "requesting" | "active" | "error";
  error: string | null;
  detection: DetectionResult;
  onExit: () => void;
}

/**
 * Map the live classifier output for the *target* letter onto a grade.
 * The classifier returns a single argmax letter plus its confidence; if
 * the argmax is the target letter, the confidence number directly tells
 * us how close the user is. If it's a different letter, we have no
 * direct readout on "how close to target" so we grade red.
 *
 * Thresholds picked so a correctly-formed sign sits comfortably in
 * green: the alphabet model averages ~0.95+ confidence on its own
 * test set.
 */
function gradeFor(targetLetter: string, detection: DetectionResult): Grade {
  if (detection.hands.length === 0) return "none";
  if (detection.sign === "-") return "red";
  if (detection.sign.toUpperCase() !== targetLetter) return "red";
  if (detection.confidence >= 0.85) return "green";
  if (detection.confidence >= 0.5) return "yellow";
  return "red";
}

const GRADE_LABEL: Record<Grade, string> = {
  green: "Almost exact",
  yellow: "Close, keep adjusting",
  red: "Way off",
  none: "Show your hand",
};

const GRADE_COLOR: Record<Grade, string> = {
  green: "#22c55e",
  yellow: "#f59e0b",
  red: "#ef4444",
  none: "#888",
};

export function LearnScreen({ videoRef, status, error, detection, onExit }: Props) {
  const [refs, setRefs] = useState<ReferencePayload | null>(null);
  const [refsError, setRefsError] = useState<string | null>(null);
  const [target, setTarget] = useState<string>("A");

  useEffect(() => {
    let cancelled = false;
    fetch(HTTP_ENDPOINTS.referenceLandmarks)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ReferencePayload) => {
        if (cancelled) return;
        setRefs(data);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setRefsError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const targetLandmarks = useMemo(() => {
    if (!refs) return null;
    return refs.letters[target] ?? null;
  }, [refs, target]);

  const grade: Grade = useMemo(
    () => gradeFor(target, detection),
    [target, detection],
  );

  return (
    <div className="learn-screen">
      <header className="learn-nav">
        <button className="btn-launch" onClick={onExit}>
          ← Back
        </button>
        <h2 className="learn-title">Learn the signs</h2>
        <div style={{ width: 80 }} />
      </header>

      <div className="letter-grid">
        {LETTERS.map((l) => {
          const available = refs?.letters[l] !== undefined;
          return (
            <button
              key={l}
              className={`letter-tile ${l === target ? "selected" : ""} ${available ? "" : "disabled"}`}
              onClick={() => available && setTarget(l)}
              disabled={!available}
            >
              {l}
            </button>
          );
        })}
      </div>

      <div className="learn-main">
        <div className="learn-panel">
          <div className="learn-panel-label">REFERENCE - {target}</div>
          <div className="learn-3d-wrap">
            {refsError && (
              <div className="learn-error">
                Could not load reference: {refsError}
              </div>
            )}
            {!refs && !refsError && (
              <div className="learn-loading">Loading reference poses...</div>
            )}
            {targetLandmarks && (
              <HandModel3D landmarks={targetLandmarks} color="#3c82f0" autoRotate />
            )}
          </div>
        </div>

        <div className="learn-panel">
          <div className="learn-panel-label">YOUR HAND</div>
          <WebcamFeed
            videoRef={videoRef}
            status={status}
            error={error}
            hands={detection.hands}
          />
          <div
            className="grade-bar"
            style={{ borderColor: GRADE_COLOR[grade] }}
          >
            <span
              className="grade-dot"
              style={{ background: GRADE_COLOR[grade] }}
            />
            <span className="grade-label">{GRADE_LABEL[grade]}</span>
            {grade !== "none" && detection.sign !== "-" && (
              <span className="grade-detail">
                detected <strong>{detection.sign}</strong>
                {" "}({Math.round(detection.confidence * 100)}%)
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
