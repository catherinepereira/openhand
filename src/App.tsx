import { useCallback, useEffect, useRef, useState } from "react";
import { WebcamFeed } from "./components/WebcamFeed";
import { SignDisplay } from "./components/SignDisplay";
import { TextOutput } from "./components/TextOutput";
import { LearnPanel } from "./components/LearnPanel";
import { SiteHeader } from "./components/SiteHeader";
import { RecordIcon, StopIcon } from "./components/icons";
import { useWebcam } from "./hooks/useWebcam";
import { useMediaPipe } from "./hooks/useMediaPipe";
import { useSignDetection } from "./hooks/useSignDetection";
import { useFingerspellRecorder } from "./hooks/useFingerspellRecorder";
import "./App.css";

// How long a letter must remain the live prediction before it's committed to the accumulator
const SIGN_DEBOUNCE_MS = 800;
const OUTPUT_MAX_CHARS = 80;

type View = "home" | "learn";

export default function App() {
  const { videoRef, videoElementRef, status, error, start, stop } = useWebcam();
  const isActive = status === "active";
  const [view, setView] = useState<View>("home");

  const { detection } = useMediaPipe(videoElementRef, isActive);
  const { result: liveResult } = useSignDetection(detection, isActive);

  // Recording session
  // While recording, every hand-present MediaPipe frame is buffered and the live MLP letter is appended
  // On stop, the full landmark buffer is sent to the CTC model in one shot for a clean phrase decode
  const recorder = useFingerspellRecorder(detection, liveResult.sign, isActive);

  const [showSkeleton, setShowSkeleton] = useState(true);

  // Live letter accumulator
  // Visible in OUTPUT whenever there isn't a completed CTC result
  // A finished recording overwrites the displayed text and clear resets both back to empty
  const [accumulated, setAccumulated] = useState("");
  const lastCommittedRef = useRef<string>("-");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Don't accumulate while recording
  // The recorder captures its own letter sequence, and the user is signing for the CTC decode, not the live bar
  const accumulate =
    isActive && recorder.state !== "recording" && recorder.state !== "decoding";

  useEffect(() => {
    if (!accumulate) return;
    const sign = liveResult.sign;
    if (sign === "-" || sign === lastCommittedRef.current) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      lastCommittedRef.current = sign;
      setAccumulated((prev) => {
        let next: string;
        if (sign === "del") next = prev.slice(0, -1);
        else if (sign === "space") next = prev + " ";
        else next = prev + sign;
        return next.length > OUTPUT_MAX_CHARS
          ? next.slice(-OUTPUT_MAX_CHARS)
          : next;
      });
    }, SIGN_DEBOUNCE_MS);
  }, [liveResult.sign, accumulate]);

  useEffect(
    () => () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    },
    [],
  );

  // Camera start/stop is always user-driven via the manual toggle
  // The Learn view works without the webcam (read-only reference browsing)
  const enterLearn = useCallback(() => setView("learn"), []);

  const exitToHome = useCallback(() => setView("home"), []);

  const handleClear = useCallback(() => {
    recorder.reset();
    setAccumulated("");
    lastCommittedRef.current = "-";
  }, [recorder]);

  // Recording result comes first otherwise show the live accumulation
  const outputText = recorder.result || accumulated;

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader title="OpenHand" repo="openhand" modelRepo="openhand-model" />

      <main className="border-border-app grid flex-1 grid-cols-2 border-b max-[700px]:grid-cols-1">
        <div className="border-border-app max-[700px]:border-border-app flex flex-col justify-center gap-6 border-r px-12 py-16 max-[700px]:border-r-0 max-[700px]:border-b max-[700px]:px-6 max-[700px]:py-10">
          {view === "home" && (
            <>
              <div className="border-border-app text-muted inline-flex w-fit items-center rounded-full border-[1.5px] px-3 py-[0.35rem] text-[0.72rem] font-semibold tracking-widest">
                OPEN SOURCE · REAL-TIME
              </div>
              <h1 className="text-[clamp(2.4rem,4vw,3.2rem)] leading-[1.1] font-normal tracking-tight">
                ASL fingerspelling,
                <br />
                to text.
              </h1>
              <p className="max-w-[38ch] text-[0.95rem] leading-[1.65] text-[#555]">
                OpenHand reads ASL fingerspelling from your webcam and
                transcribes it in real time, all in the browser.
              </p>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={isActive ? stop : start}
                  className="bg-ink rounded-[9px] px-[1.4rem] py-[0.6rem] text-[0.88rem] font-medium text-white transition-opacity hover:opacity-80"
                >
                  {isActive ? "Stop camera" : "Start camera"}
                </button>
                <button
                  onClick={enterLearn}
                  className="border-border-app bg-bg hover:bg-surface rounded-[9px] border-[1.5px] px-[1.4rem] py-[0.6rem] text-[0.88rem] font-medium transition-colors"
                >
                  Learn the letters
                </button>
              </div>
            </>
          )}
          {view === "learn" && (
            <LearnPanel
              detection={liveResult}
              frameDetection={detection}
              active={isActive}
              onExit={exitToHome}
              onStartCamera={start}
              onStopCamera={stop}
            />
          )}
        </div>

        <div className="bg-bg flex flex-col items-center justify-center gap-6 px-8 py-12 max-[700px]:px-6 max-[700px]:py-8">
          <WebcamFeed
            videoRef={videoRef}
            status={status}
            error={error}
            hands={showSkeleton ? liveResult.hands : []}
            showSkeleton={showSkeleton}
            onShowSkeletonChange={setShowSkeleton}
          />

          <SignDisplay
            sign={liveResult.sign}
            confidence={liveResult.confidence}
          />

          {(isActive || outputText) && (
            <div className="flex w-full max-w-[min(760px,96%)] items-stretch gap-3">
              {isActive &&
                (recorder.state === "recording" ? (
                  <button
                    onClick={recorder.stop}
                    aria-label="Stop recording"
                    title="Stop recording"
                    className="flex w-12 shrink-0 items-center justify-center rounded-xl bg-[#dc2626] text-white transition-opacity hover:opacity-85"
                  >
                    <StopIcon />
                  </button>
                ) : (
                  <button
                    onClick={recorder.start}
                    disabled={recorder.state === "decoding"}
                    aria-label={
                      recorder.state === "decoding" ? "Decoding" : "Record"
                    }
                    title={
                      recorder.state === "decoding" ? "Decoding" : "Record"
                    }
                    className="bg-ink flex w-12 shrink-0 items-center justify-center rounded-xl text-white transition-opacity hover:enabled:opacity-80 disabled:opacity-60"
                  >
                    <RecordIcon />
                  </button>
                ))}
              <div className="min-w-0 flex-1">
                <TextOutput text={outputText} onClear={handleClear} />
              </div>
            </div>
          )}

          {recorder.error && (
            <span className="text-[0.8rem] text-[#b14242]">
              {recorder.error}
            </span>
          )}
        </div>
      </main>

      <footer className="border-border-app mt-auto grid grid-cols-2 border-t max-[700px]:grid-cols-1">
        <div className="border-border-app max-[700px]:border-border-app flex flex-col gap-[0.3rem] border-r px-10 py-[1.4rem] max-[700px]:border-r-0 max-[700px]:border-b">
          <span className="label-caps">DETECTION</span>
          <span className="text-ink text-[0.95rem] font-medium">
            MediaPipe Hand + Pose + Face Landmarks (JS)
          </span>
        </div>
        <div className="flex flex-col gap-[0.3rem] px-10 py-[1.4rem]">
          <span className="label-caps">CLASSIFICATION</span>
          <span className="text-ink text-[0.95rem] font-medium">
            MLP + CTC in ONNX Runtime Web
          </span>
        </div>
      </footer>
    </div>
  );
}
