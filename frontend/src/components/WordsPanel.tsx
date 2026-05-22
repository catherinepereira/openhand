import { useEffect, useMemo, useState } from "react";
import { HTTP_ENDPOINTS } from "../config";
import { useWordDetection, type WordPrediction } from "../hooks/useWordDetection";
import type { FrameDetection } from "../lib/mediapipe";
import { ReferenceView } from "./ReferenceView";
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
    <div className="flex w-full min-w-0 flex-col gap-4">
      <div className="flex items-center gap-4">
        <button
          onClick={onExit}
          className="whitespace-nowrap rounded-lg border-[1.5px] border-border-app px-[1.1rem] py-[0.45rem] text-[0.85rem] font-medium text-ink transition-colors hover:bg-surface"
        >
          ← Back
        </button>
        <h2 className="text-[1.3rem] font-semibold tracking-tight">Learn the words</h2>
      </div>

      <div className="flex flex-col gap-2 rounded-[14px] border-[1.5px] border-border-app bg-bg px-4 py-[0.85rem]">
        <div className="label-caps text-[0.7rem]">REFERENCE - {target ?? "..."}</div>
        <div className="relative h-[420px] overflow-hidden rounded-xl bg-surface max-[700px]:h-[300px]">
          {refsError && (
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-[0.9rem] text-[#b14242]">
              Could not load references: {refsError}
            </div>
          )}
          {!refs && !refsError && (
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-[0.9rem] text-muted">
              Loading sign references...
            </div>
          )}
          {targetClip && target && (
            <ReferenceView
              imageSrc={`/reference-signs/${target}.png`}
              animation={
                <SignAnimation3D
                  landmarks={targetClip.landmarks}
                  missing={targetClip.missing}
                  frames={targetClip.n_frames}
                />
              }
            />
          )}
        </div>
      </div>

      <input
        type="search"
        placeholder="Search 250 signs..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        className="w-full rounded-lg border-[1.5px] border-border-app bg-bg px-[0.8rem] py-[0.55rem] font-sans text-[0.9rem] text-ink outline-none transition-colors focus:border-ink"
      />

      <div className="grid max-h-[220px] grid-cols-[repeat(auto-fill,minmax(130px,1fr))] gap-[0.35rem] overflow-y-auto rounded-[10px] border-[1.5px] border-border-app bg-bg p-1">
        {filtered.map((sign) => {
          const selected = sign === target;
          return (
            <button
              key={sign}
              onClick={() => setTarget(sign)}
              className={[
                "min-h-[2.2rem] rounded-md border px-[0.7rem] py-2 text-left text-[0.85rem] font-medium leading-tight transition-colors active:scale-[0.97] [word-break:break-word]",
                selected
                  ? "border-ink bg-ink text-white"
                  : "border-border-app bg-bg text-ink hover:bg-surface",
              ].join(" ")}
            >
              {sign}
            </button>
          );
        })}
        {filtered.length === 0 && (
          <div className="col-span-full p-6 text-center text-[0.85rem] text-muted">
            No signs match "{filter}"
          </div>
        )}
      </div>

      <div
        className="flex items-center gap-[0.7rem] rounded-[10px] border-[1.5px] bg-bg px-4 py-[0.8rem] transition-colors"
        style={{ borderColor: GRADE_COLOR[grade] }}
      >
        <span
          className="h-3 w-3 shrink-0 rounded-full transition-colors"
          style={{ background: GRADE_COLOR[grade] }}
        />
        <span className="text-[0.95rem] font-semibold">{GRADE_LABEL[grade]}</span>
        <span className="ml-auto text-[0.85rem] text-muted">
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
