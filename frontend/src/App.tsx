import { useCallback, useEffect, useRef, useState } from "react";
import { WebcamFeed } from "./components/WebcamFeed";
import { SignDisplay } from "./components/SignDisplay";
import { TextOutput } from "./components/TextOutput";
import { LearnPanel } from "./components/LearnPanel";
import { useWebcam } from "./hooks/useWebcam";
import { useMediaPipe } from "./hooks/useMediaPipe";
import { useSignDetection } from "./hooks/useSignDetection";
import { HTTP_ENDPOINTS } from "./config";
import "./App.css";

const SIGN_DEBOUNCE_MS = 800;
// Sliding-window cap on the accumulated output. Older letters scroll off
// the front so the bar stays a reasonable length without ever needing a
// manual clear.
const OUTPUT_MAX_CHARS = 30;

type View = "home" | "learn";

export default function App() {
  const { videoRef, videoElementRef, status, error, start, stop } = useWebcam();
  const isActive = status === "active";
  const [view, setView] = useState<View>("home");

  const { detection } = useMediaPipe(videoElementRef, isActive);
  const { result: liveResult } = useSignDetection(detection, isActive);

  const [outputText, setOutputText] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [ttsEnabled] = useState(!!import.meta.env.VITE_TTS_ENABLED);
  const [showSkeleton, setShowSkeleton] = useState(true);

  const enterLearn = useCallback(async () => {
    if (!isActive) await start();
    setView("learn");
  }, [isActive, start]);

  const exitLearn = useCallback(() => setView("home"), []);

  // Debounce detected letters into the accumulated output text.
  const lastSignRef = useRef<string>("-");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const sign = liveResult.sign;
    if (sign === "-" || sign === lastSignRef.current) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      lastSignRef.current = sign;
      setOutputText((prev) => {
        const next = prev + sign;
        return next.length > OUTPUT_MAX_CHARS
          ? next.slice(-OUTPUT_MAX_CHARS)
          : next;
      });
    }, SIGN_DEBOUNCE_MS);
  }, [liveResult.sign]);

  useEffect(() => () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
  }, []);

  const handleClear = useCallback(() => {
    setOutputText("");
    lastSignRef.current = "-";
  }, []);

  const handleSpeak = useCallback(async () => {
    if (!outputText || speaking) return;
    setSpeaking(true);
    try {
      const res = await fetch(HTTP_ENDPOINTS.tts, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: outputText }),
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
  }, [outputText, speaking]);

  return (
    <div className="app">
      <header className="nav">
        <div className="nav-logo">OpenHand <span>🤟</span></div>
        <nav className="nav-links">
          <a
            href="https://github.com/catherinepereira/openhand"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub
          </a>
        </nav>
        <button
          className="btn-launch"
          onClick={isActive ? stop : start}
        >
          {isActive ? "Stop" : "Launch app"}
        </button>
      </header>

      <main className="hero">
        {/* Left column swaps content based on the current view. The right
            column (webcam + overlays) is identical in both views, so the
            <video> element and its MediaStream stay mounted. */}
        <div className="hero-left">
          {view === "home" ? (
            <>
              <div className="badge">OPEN SOURCE · REAL-TIME</div>
              <h1 className="hero-heading">
                Sign language,<br />
                <strong>to speech.</strong>
              </h1>
              <p className="hero-sub">
                OpenHand detects signs as you make them and converts to text and speech in real time.
              </p>
              <div className="hero-ctas">
                <button className="btn-primary" onClick={isActive ? stop : start}>
                  {isActive ? "Stop camera" : "Start camera"}
                </button>
                <button className="btn-secondary" onClick={enterLearn}>Learn the signs</button>
              </div>
            </>
          ) : (
            <LearnPanel
              detection={liveResult}
              frameDetection={detection}
              active={isActive}
              onExit={exitLearn}
            />
          )}
        </div>

        <div className="hero-right">
          <WebcamFeed
            videoRef={videoRef}
            status={status}
            error={error}
            hands={showSkeleton ? liveResult.hands : []}
            showSkeleton={showSkeleton}
            onShowSkeletonChange={setShowSkeleton}
          />

          <SignDisplay sign={liveResult.sign} confidence={liveResult.confidence} />

          {(isActive || outputText) && (
            <TextOutput
              text={outputText}
              onClear={handleClear}
              onSpeak={handleSpeak}
              ttsEnabled={ttsEnabled}
              speaking={speaking}
            />
          )}
        </div>
      </main>

      <footer className="info-strip">
        <div className="info-block">
          <span className="info-label">DETECTION</span>
          <span className="info-value">MediaPipe Hand Landmarks (JS)</span>
        </div>
        <div className="info-block">
          <span className="info-label">CLASSIFICATION</span>
          <span className="info-value">MLP (per sign) and CTC (phrase transcription)</span>
        </div>
        <div className="info-block">
          <span className="info-label">OUTPUT</span>
          <span className="info-value">TTS (ElevenLabs)</span>
        </div>
      </footer>
    </div>
  );
}
