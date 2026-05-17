import { useEffect, useMemo, useState } from "react";
import { HTTP_ENDPOINTS } from "../config";
import { useWordDetection, type WordPrediction } from "../hooks/useWordDetection";
import type { FrameDetection } from "../lib/mediapipe";
import { SignAnimation3D } from "./SignAnimation3D";

interface ReferenceClip {
  label: number;
  n_frames: number;
  landmarks: number[];
  missing: number[];
}

interface ReferencesPayload {
  n_landmarks: number;
  n_coords: number;
  sign_to_idx: Record<string, number>;
  signs: Record<string, ReferenceClip>;
}

type Grade = "green" | "yellow" | "red" | "none";

interface Props {
  frameDetection: FrameDetection | null;
  active: boolean;
  onExit: () => void;
}

const GRADE_LABEL: Record<Grade, string> = {
  green: "Nailed it",
  yellow: "Top 5 - close",
  red: "Not yet",
  none: "Show your hands",
};
const GRADE_COLOR: Record<Grade, string> = {
  green: "#22c55e",
  yellow: "#f59e0b",
  red: "#ef4444",
  none: "#888",
};

function gradeFor(target: string, predictions: WordPrediction[]): Grade {
  if (predictions.length === 0) return "none";
  const targetLc = target.toLowerCase();
  const top = predictions[0];
  if (top[0].toLowerCase() === targetLc && top[1] >= 0.5) return "green";
  if (predictions.slice(0, 5).some(([s]) => s.toLowerCase() === targetLc)) return "yellow";
  return "red";
}

export function WordsPanel({ frameDetection, active, onExit }: Props) {
  const [refs, setRefs] = useState<ReferencesPayload | null>(null);
  const [refsError, setRefsError] = useState<string | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  // Live classifier output (rolling buffer + per-tick top-K).
  const { predictions, unavailable } = useWordDetection(frameDetection, active);

  useEffect(() => {
    let cancelled = false;
    fetch(HTTP_ENDPOINTS.signReferences)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ReferencesPayload) => {
        if (cancelled) return;
        setRefs(data);
        // Pick a default target the first time we land here.
        const names = Object.keys(data.signs).sort();
        if (names.length > 0) setTarget(names[0]);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setRefsError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const allSigns = useMemo(() => (refs ? Object.keys(refs.signs).sort() : []), [refs]);
  const filtered = useMemo(() => {
    const f = filter.trim().toLowerCase();
    if (!f) return allSigns;
    return allSigns.filter((s) => s.toLowerCase().includes(f));
  }, [allSigns, filter]);

  const targetClip = useMemo(() => {
    if (!refs || !target) return null;
    return refs.signs[target] ?? null;
  }, [refs, target]);

  const grade: Grade = useMemo(() => {
    if (!target) return "none";
    if (unavailable) return "none";
    return gradeFor(target, predictions);
  }, [target, predictions, unavailable]);

  return (
    <div className="learn-panel-col">
      <div className="learn-panel-header">
        <button className="btn-launch" onClick={onExit}>← Back</button>
        <h2 className="learn-title">Learn the words</h2>
      </div>

      <div className="reference-block">
        <div className="learn-panel-label">
          REFERENCE - {target ?? "..."}
        </div>
        <div className="learn-3d-wrap">
          {refsError && (
            <div className="learn-error">
              Could not load references: {refsError}
            </div>
          )}
          {!refs && !refsError && (
            <div className="learn-loading">Loading sign references...</div>
          )}
          {targetClip && (
            <SignAnimation3D
              landmarks={targetClip.landmarks}
              missing={targetClip.missing}
              frames={targetClip.n_frames}
            />
          )}
        </div>
      </div>

      <input
        type="search"
        className="word-search"
        placeholder="Search 250 signs..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />

      <div className="word-list">
        {filtered.map((sign) => (
          <button
            key={sign}
            className={`word-tile ${sign === target ? "selected" : ""}`}
            onClick={() => setTarget(sign)}
          >
            {sign}
          </button>
        ))}
        {filtered.length === 0 && (
          <div className="word-list-empty">No signs match "{filter}"</div>
        )}
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
        <span className="grade-detail">
          {unavailable
            ? "Sign model unavailable"
            : predictions[0]
              ? <>top: <strong>{predictions[0][0]}</strong> ({Math.round(predictions[0][1] * 100)}%)</>
              : "decoding..."}
        </span>
      </div>
    </div>
  );
}
