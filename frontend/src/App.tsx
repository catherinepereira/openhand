import { useCallback, useEffect, useRef, useState } from "react";
import { WebcamFeed } from "./components/WebcamFeed";
import { SignDisplay } from "./components/SignDisplay";
import { TextOutput } from "./components/TextOutput";
import { LearnScreen } from "./components/LearnScreen";
import { useWebcam } from "./hooks/useWebcam";
import { useMediaPipe } from "./hooks/useMediaPipe";
import { useSignDetection } from "./hooks/useSignDetection";
import { useStreamingTranscribe } from "./hooks/useStreamingTranscribe";
import { HTTP_ENDPOINTS } from "./config";
import "./App.css";

const SIGN_DEBOUNCE_MS = 800;

type View = "home" | "learn";

export default function App() {
  const { videoRef, status, error, start, stop } = useWebcam();
  const isActive = status === "active";
  const [view, setView] = useState<View>("home");

  const { detection } = useMediaPipe(videoRef, isActive);
  const { result: liveResult } = useSignDetection(detection, isActive);
  const transcribe = useStreamingTranscribe(detection, isActive);

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
      setOutputText((prev) => prev + sign);
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

  if (view === "learn") {
    return (
      <div className="app">
        <LearnScreen
          videoRef={videoRef}
          status={status}
          error={error}
          detection={liveResult}
          onExit={exitLearn}
        />
      </div>
    );
  }

  return (
    <div className="app">
      <header className="nav">
        <div className="nav-logo">OpenHand <span>🤟</span></div>
        <nav className="nav-links">
          <a href="#">How it works</a>
          <a href="#">Docs</a>
          <a href="#">GitHub</a>
        </nav>
        <button
          className="btn-launch"
          onClick={isActive ? stop : start}
        >
          {isActive ? "Stop" : "Launch app"}
        </button>
      </header>

      <main className="hero">
        <div className="hero-left">
          <div className="badge">OPEN SOURCE · REAL-TIME</div>
          <h1 className="hero-heading">
            Sign language,<br />
            <strong>understood.</strong>
          </h1>
          <p className="hero-sub">
            Point your camera. OpenHand detects signs as you make them and converts to text and speech in real time.
          </p>
          <div className="hero-ctas">
            <button className="btn-primary" onClick={isActive ? stop : start}>
              {isActive ? "Stop camera" : "Start camera"}
            </button>
            <button className="btn-secondary" onClick={enterLearn}>Learn the signs</button>
          </div>
        </div>

        <div className="hero-right">
          <WebcamFeed
            videoRef={videoRef}
            status={status}
            error={error}
            hands={showSkeleton ? liveResult.hands : []}
          />

          <label className="skeleton-toggle">
            <input
              type="checkbox"
              checked={showSkeleton}
              onChange={(e) => setShowSkeleton(e.target.checked)}
            />
            Show skeleton
          </label>

          <SignDisplay sign={liveResult.sign} confidence={liveResult.confidence} />

          {isActive && (
            <div className="phrase-display">
              <span className="phrase-label">PHRASE</span>
              <span className="phrase-value">
                {transcribe.text ? (
                  <>
                    <span className="phrase-committed">{transcribe.text}</span>
                    {transcribe.tentative && (
                      <span className="phrase-tentative"> {transcribe.tentative}</span>
                    )}
                  </>
                ) : transcribe.tentative ? (
                  <span className="phrase-tentative">{transcribe.tentative}</span>
                ) : (
                  <em className="phrase-placeholder">...</em>
                )}
              </span>
            </div>
          )}
        </div>
      </main>

      {(isActive || outputText) && (
        <div className="output-bar">
          <TextOutput
            text={outputText}
            onClear={handleClear}
            onSpeak={handleSpeak}
            ttsEnabled={ttsEnabled}
            speaking={speaking}
          />
        </div>
      )}

      <footer className="info-strip">
        <div className="info-block">
          <span className="info-label">DETECTION</span>
          <span className="info-value">MediaPipe + classifier</span>
        </div>
        <div className="info-block">
          <span className="info-label">OUTPUT</span>
          <span className="info-value">Text · Speech (ElevenLabs)</span>
        </div>
        <div className="info-block">
          <span className="info-label">PRIVACY</span>
          <span className="info-value">Runs locally</span>
        </div>
      </footer>
    </div>
  );
}
