import { useEffect, useMemo, useState } from "react";
import { HTTP_ENDPOINTS } from "../config";
import type { DetectionResult } from "../hooks/useSignDetection";
import { useTargetedTranscribe } from "../hooks/useTargetedTranscribe";
import type { FrameDetection } from "../lib/mediapipe";
import { HandModel3D } from "./HandModel3D";
import { ReferenceView } from "./ReferenceView";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const MOTION_LETTERS = new Set(["J", "Z"]);

interface ReferencePayload {
  format: string;
  n_features: number;
  letters: Record<string, number[]>;
  sample_counts: Record<string, number>;
}

type Grade = "green" | "yellow" | "red" | "none";

interface Props {
  detection: DetectionResult;
  frameDetection: FrameDetection | null;
  active: boolean;
  onExit: () => void;
}

function gradeStatic(targetLetter: string, detection: DetectionResult): Grade {
  if (detection.hands.length === 0) return "none";
  if (detection.sign === "-") return "red";
  if (detection.sign.toUpperCase() !== targetLetter) return "red";
  if (detection.confidence >= 0.85) return "green";
  if (detection.confidence >= 0.5) return "yellow";
  return "red";
}

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

export function LearnPanel({ detection, frameDetection, active, onExit }: Props) {
  const [refs, setRefs] = useState<ReferencePayload | null>(null);
  const [refsError, setRefsError] = useState<string | null>(null);
  const [target, setTarget] = useState<string>("A");

  const isMotion = MOTION_LETTERS.has(target);
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
    <div className="flex w-full min-w-0 flex-col gap-4">
      <div className="flex items-center gap-4">
        <button
          onClick={onExit}
          className="whitespace-nowrap rounded-lg border-[1.5px] border-border-app px-[1.1rem] py-[0.45rem] text-[0.85rem] font-medium text-ink transition-colors hover:bg-surface"
        >
          ← Back
        </button>
        <h2 className="text-[1.3rem] font-semibold tracking-tight">Learn the signs</h2>
      </div>

      <div className="flex flex-col gap-2 rounded-[14px] border-[1.5px] border-border-app bg-bg px-4 py-[0.85rem]">
        <div className="label-caps text-[0.7rem]">REFERENCE - {target}</div>
        <div className="relative h-[420px] overflow-hidden rounded-xl bg-surface max-[700px]:h-[300px]">
          {refsError && (
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-[0.9rem] text-[#b14242]">
              Could not load reference: {refsError}
            </div>
          )}
          {!refs && !refsError && (
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-[0.9rem] text-muted">
              Loading reference poses...
            </div>
          )}
          {targetLandmarks && (
            <ReferenceView
              imageSrc={`/reference-letters/${target}.svg`}
              animation={<HandModel3D landmarks={targetLandmarks} />}
            />
          )}
        </div>
      </div>

      <div className="grid w-full grid-cols-[repeat(13,1fr)] gap-[0.3rem] max-[1100px]:grid-cols-[repeat(9,1fr)] max-[700px]:grid-cols-[repeat(7,1fr)]">
        {LETTERS.map((l) => {
          const available = refs?.letters[l] !== undefined;
          const selected = l === target;
          return (
            <button
              key={l}
              onClick={() => available && setTarget(l)}
              disabled={!available}
              className={[
                "aspect-square rounded-[7px] border-[1.5px] text-[0.9rem] font-semibold transition-colors active:enabled:scale-95",
                selected
                  ? "border-ink bg-ink text-white"
                  : "border-border-app bg-bg text-ink hover:enabled:bg-surface",
                !available && "cursor-not-allowed opacity-35",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {l}
            </button>
          );
        })}
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
        {isMotion ? (
          <span className="ml-auto text-[0.85rem] text-muted">
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
            <span className="ml-auto text-[0.85rem] text-muted">
              detected <strong>{detection.sign}</strong>
              {" "}({Math.round(detection.confidence * 100)}%)
            </span>
          )
        )}
      </div>

      {isMotion && (
        <p className="px-1 text-[0.82rem] leading-relaxed text-muted">
          {target} is a motion sign. Trace the letter in front of the
          camera; we score on the CTC model's decode of the last ~2 seconds.
        </p>
      )}
    </div>
  );
}
