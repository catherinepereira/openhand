import { useCallback, useState } from "react";
import { WebcamFeed } from "./components/WebcamFeed";
import { SignDisplay } from "./components/SignDisplay";
import { TextOutput } from "./components/TextOutput";
import { LearnPanel } from "./components/LearnPanel";
import { useWebcam } from "./hooks/useWebcam";
import { useMediaPipe } from "./hooks/useMediaPipe";
import { useSignDetection } from "./hooks/useSignDetection";
import { useFingerspellRecorder } from "./hooks/useFingerspellRecorder";
import { HTTP_ENDPOINTS } from "./config";
import "./App.css";

type View = "home" | "learn";

export default function App() {
  const { videoRef, videoElementRef, status, error, start, stop } = useWebcam();
  const isActive = status === "active";
  const [view, setView] = useState<View>("home");

  const { detection } = useMediaPipe(videoElementRef, isActive);
  const { result: liveResult } = useSignDetection(detection, isActive);

  // Recording session. While recording, every hand-present MediaPipe frame
  // is buffered and the live MLP letter is collapse-appended. On stop, the
  // full landmark buffer is sent to the CTC model in one shot for a clean
  // phrase decode.
  const recorder = useFingerspellRecorder(detection, liveResult.sign, isActive);

  const [speaking, setSpeaking] = useState(false);
  const [ttsEnabled] = useState(!!import.meta.env.VITE_TTS_ENABLED);
  const [showSkeleton, setShowSkeleton] = useState(true);

  // Camera start/stop is always user-driven via the manual toggle. The
  // Learn view works without the webcam (read-only reference browsing).
  const enterLearn = useCallback(() => setView("learn"), []);

  const exitToHome = useCallback(() => setView("home"), []);

  const handleClear = useCallback(() => {
    recorder.reset();
  }, [recorder]);

  const handleSpeak = useCallback(async () => {
    if (!recorder.result || speaking) return;
    setSpeaking(true);
    try {
      const res = await fetch(HTTP_ENDPOINTS.tts, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: recorder.result }),
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => { setSpeaking(false); URL.revokeObjectURL(url); };
        audio.play();
      } else {
        setSpeaking(false);
      }
    } catch {
      setSpeaking(false);
    }
  }, [recorder.result, speaking]);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex h-16 items-center gap-8 border-b border-border-app bg-bg px-10 max-[700px]:px-5">
        <div className="text-[1.05rem] font-semibold tracking-tight whitespace-nowrap">
          OpenHand <span>🤟</span>
        </div>
        <nav className="flex flex-1 justify-center gap-6 text-sm text-muted max-[700px]:hidden">
          <a
            href="https://github.com/catherinepereira/openhand"
            target="_blank"
            rel="noreferrer noopener"
            className="hover:text-ink"
          >
            GitHub
          </a>
        </nav>
        <button
          onClick={isActive ? stop : start}
          className="whitespace-nowrap rounded-lg border-[1.5px] border-border-app px-[1.1rem] py-[0.45rem] text-[0.85rem] font-medium text-ink transition-colors hover:bg-surface"
        >
          {isActive ? "Stop" : "Launch app"}
        </button>
      </header>

      <main className="grid flex-1 grid-cols-2 border-b border-border-app max-[700px]:grid-cols-1">
        <div className="flex flex-col gap-6 border-r border-border-app px-12 pt-16 pb-12 max-[700px]:border-r-0 max-[700px]:border-b max-[700px]:border-border-app max-[700px]:px-6 max-[700px]:py-10">
          {view === "home" && (
            <>
              <div className="inline-flex w-fit items-center rounded-full border-[1.5px] border-border-app px-3 py-[0.35rem] text-[0.72rem] font-semibold tracking-widest text-muted">
                OPEN SOURCE · REAL-TIME
              </div>
              <h1 className="text-[clamp(2.4rem,4vw,3.2rem)] font-normal leading-[1.1] tracking-tight">
                ASL fingerspelling,<br />
                <strong className="font-bold">to speech.</strong>
              </h1>
              <p className="max-w-[38ch] text-[0.95rem] leading-[1.65] text-[#555]">
                OpenHand reads ASL fingerspelling from your webcam and converts it to text and speech in real time.
              </p>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={isActive ? stop : start}
                  className="rounded-[9px] bg-ink px-[1.4rem] py-[0.6rem] text-[0.88rem] font-medium text-white transition-opacity hover:opacity-80"
                >
                  {isActive ? "Stop camera" : "Start camera"}
                </button>
                <button
                  onClick={enterLearn}
                  className="rounded-[9px] border-[1.5px] border-border-app bg-bg px-[1.4rem] py-[0.6rem] text-[0.88rem] font-medium transition-colors hover:bg-surface"
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
            />
          )}
        </div>

        <div className="flex flex-col items-center justify-center gap-6 bg-bg px-8 py-12 max-[700px]:px-6 max-[700px]:py-8">
          <WebcamFeed
            videoRef={videoRef}
            status={status}
            error={error}
            hands={showSkeleton ? liveResult.hands : []}
            showSkeleton={showSkeleton}
            onShowSkeletonChange={setShowSkeleton}
          />

          <SignDisplay sign={liveResult.sign} confidence={liveResult.confidence} />

          {isActive && (
            <div className="flex w-full max-w-[640px] items-center gap-3">
              {recorder.state === "recording" ? (
                <button
                  onClick={recorder.stop}
                  className="flex items-center gap-2 rounded-[9px] bg-[#dc2626] px-[1.4rem] py-[0.6rem] text-[0.88rem] font-medium text-white transition-opacity hover:opacity-85"
                >
                  <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-white" />
                  Stop ({recorder.frameCount} frames)
                </button>
              ) : (
                <button
                  onClick={recorder.start}
                  disabled={recorder.state === "decoding"}
                  className="flex items-center gap-2 rounded-[9px] bg-ink px-[1.4rem] py-[0.6rem] text-[0.88rem] font-medium text-white transition-opacity hover:enabled:opacity-80 disabled:opacity-60"
                >
                  <span className="h-2.5 w-2.5 rounded-full bg-[#dc2626]" />
                  {recorder.state === "decoding" ? "Decoding..." : "Record"}
                </button>
              )}
              {recorder.error && (
                <span className="text-[0.8rem] text-[#b14242]">{recorder.error}</span>
              )}
            </div>
          )}

          {(isActive || recorder.result) && (
            <TextOutput
              text={recorder.result}
              onClear={handleClear}
              onSpeak={handleSpeak}
              ttsEnabled={ttsEnabled}
              speaking={speaking}
            />
          )}
        </div>
      </main>

      <footer className="mt-auto grid grid-cols-3 border-t border-border-app max-[700px]:grid-cols-1">
        <div className="flex flex-col gap-[0.3rem] border-r border-border-app px-10 py-[1.4rem] max-[700px]:border-r-0 max-[700px]:border-b max-[700px]:border-border-app">
          <span className="label-caps">DETECTION</span>
          <span className="text-[0.95rem] font-medium text-ink">MediaPipe Hand Landmarks (JS)</span>
        </div>
        <div className="flex flex-col gap-[0.3rem] border-r border-border-app px-10 py-[1.4rem] max-[700px]:border-r-0 max-[700px]:border-b max-[700px]:border-border-app">
          <span className="label-caps">CLASSIFICATION</span>
          <span className="text-[0.95rem] font-medium text-ink">MLP (per letter) and CTC (phrase transcription)</span>
        </div>
        <div className="flex flex-col gap-[0.3rem] px-10 py-[1.4rem]">
          <span className="label-caps">OUTPUT</span>
          <span className="text-[0.95rem] font-medium text-ink">TTS (ElevenLabs)</span>
        </div>
      </footer>
    </div>
  );
}
