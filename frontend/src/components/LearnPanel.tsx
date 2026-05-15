import { useEffect, useMemo, useState } from "react";
import { HTTP_ENDPOINTS } from "../config";
import type { DetectionResult } from "../hooks/useSignDetection";
import { useTargetedTranscribe } from "../hooks/useTargetedTranscribe";
import type { FrameDetection } from "../lib/mediapipe";
import { HandModel3D } from "./HandModel3D";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

/**
 * Letters where the static-frame classifier can't disambiguate because
 * the sign is defined by motion (not just handshape). For these we score
 * against the CTC model's decoded text instead of the per-frame argmax.
 *
 * J is "I + drop-and-hook"; Z is "D-finger + trace a Z". The static
 * frames look like I and D respectively.
 */
const MOTION_LETTERS = new Set(["J", "Z"]);

interface ReferencePayload {
  format: string;
  n_features: number;
  letters: Record<string, number[]>;
  sample_counts: Record<string, number>;
}

type Grade = "green" | "yellow" | "red" | "none";

interface Props {
  /** Live classifier output (per-frame argmax + confidence). */
  detection: DetectionResult;
  /** Raw MediaPipe output, needed for CTC scoring on motion letters. */
  frameDetection: FrameDetection | null;
  active: boolean;
  onExit: () => void;
}

/**
 * Static-frame grading via the alphabet MLP. If the classifier's argmax
 * matches the target, its confidence number directly tells us how close
 * the user is. If the argmax is a different letter, we grade red.
 *
 * Thresholds are loose-ish (0.85 / 0.5) so a correctly-formed sign sits
 * comfortably in green; the alphabet model averages ~0.95+ confidence on
 * its test set, so we have headroom.
 */
function gradeStatic(targetLetter: string, detection: DetectionResult): Grade {
  if (detection.hands.length === 0) return "none";
  if (detection.sign === "-") return "red";
  if (detection.sign.toUpperCase() !== targetLetter) return "red";
  if (detection.confidence >= 0.85) return "green";
  if (detection.confidence >= 0.5) return "yellow";
  return "red";
}

/**
 * CTC-based grading for motion letters. The decoded string from the
 * rolling 2-second window will contain garbage between meaningful gestures
 * (the CTC model was trained on continuous fingerspelling, not isolated
 * letters), so we look for the target letter anywhere in the decode.
 */
function gradeMotion(
  targetLetter: string,
  detection: DetectionResult,
  ctcText: string,
): Grade {
  if (detection.hands.length === 0) return "none";
  const decoded = ctcText.toLowerCase();
  const target = targetLetter.toLowerCase();
  if (decoded.includes(target)) return "green";
  if (decoded.length > 0) return "yellow";
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

/**
 * Left-column panel for the Learn view. Renders alongside the persistent
 * webcam feed in the hero layout, so the camera doesn't have to remount
 * when entering/leaving Learn mode.
 */
export function LearnPanel({ detection, frameDetection, active, onExit }: Props) {
  const [refs, setRefs] = useState<ReferencePayload | null>(null);
  const [refsError, setRefsError] = useState<string | null>(null);
  const [target, setTarget] = useState<string>("A");

  const isMotion = MOTION_LETTERS.has(target);
  // CTC decode for motion letters only.
  const ctc = useTargetedTranscribe(frameDetection, active && isMotion);

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

  const grade: Grade = useMemo(() => {
    if (isMotion) {
      if (ctc.unavailable) return gradeStatic(target, detection);
      return gradeMotion(target, detection, ctc.latest);
    }
    return gradeStatic(target, detection);
  }, [isMotion, target, detection, ctc.latest, ctc.unavailable]);

  return (
    <div className="learn-panel-col">
      <div className="learn-panel-header">
        <button className="btn-launch" onClick={onExit}>← Back</button>
        <h2 className="learn-title">Learn the signs</h2>
      </div>

      <div className="reference-block">
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
            <HandModel3D landmarks={targetLandmarks} />
          )}
        </div>
      </div>

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

      <div
        className="grade-bar"
        style={{ borderColor: GRADE_COLOR[grade] }}
      >
        <span
          className="grade-dot"
          style={{ background: GRADE_COLOR[grade] }}
        />
        <span className="grade-label">{GRADE_LABEL[grade]}</span>
        {isMotion ? (
          <span className="grade-detail">
            {ctc.unavailable
              ? "CTC model unavailable"
              : detection.hands.length === 0
                ? null
                : ctc.latest
                  ? <>decode <strong>{ctc.latest.slice(-1).toUpperCase()}</strong></>
                  : "move your hand"}
          </span>
        ) : (
          grade !== "none" && detection.sign !== "-" && (
            <span className="grade-detail">
              detected <strong>{detection.sign}</strong>
              {" "}({Math.round(detection.confidence * 100)}%)
            </span>
          )
        )}
      </div>

      {isMotion && (
        <p className="learn-hint">
          {target} is a motion sign. Trace the letter in front of the
          camera; we score on the CTC model's decode of the last ~2 seconds.
        </p>
      )}
    </div>
  );
}
