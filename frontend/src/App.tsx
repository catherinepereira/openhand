import { useCallback, useEffect, useRef, useState } from "react";
import { WebcamFeed } from "./components/WebcamFeed";
import { SignDisplay } from "./components/SignDisplay";
import { TextOutput } from "./components/TextOutput";
import { useWebcam } from "./hooks/useWebcam";
import { useSignDetection } from "./hooks/useSignDetection";
import "./App.css";

const SIGN_DEBOUNCE_MS = 800;
const TTS_ENDPOINT = "http://localhost:8273/api/tts";

export default function App() {
  const { videoRef, status, error, start, stop } = useWebcam();
  const isActive = status === "active";
  const { result } = useSignDetection(videoRef, isActive);

  const [outputText, setOutputText] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [ttsEnabled] = useState(!!import.meta.env.VITE_TTS_ENABLED);

  // Debounce detected sign into output text
  const lastSignRef = useRef<string>("—");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const sign = result.sign;
    if (sign === "—" || sign === lastSignRef.current) return;
    debounceRef.current && clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      lastSignRef.current = sign;
      setOutputText((prev) => prev + sign);
    }, SIGN_DEBOUNCE_MS);
  }, [result.sign]);

  const handleClear = useCallback(() => {
    setOutputText("");
    lastSignRef.current = "—";
  }, []);

  const handleSpeak = useCallback(async () => {
    if (!outputText || speaking) return;
    setSpeaking(true);
    try {
      const res = await fetch(TTS_ENDPOINT, {
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
      {/* Nav */}
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

      {/* Hero */}
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
            <button className="btn-secondary">Learn the signs</button>
          </div>
        </div>

        <div className="hero-right">
          <WebcamFeed
            videoRef={videoRef}
            status={status}
            error={error}
            landmarks={result.landmarks}
          />
          <SignDisplay sign={result.sign} confidence={result.confidence} />
        </div>
      </main>

      {/* Text output bar — visible when there's output or camera is active */}
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

      {/* Footer info strip */}
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
