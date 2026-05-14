# OpenHand 🤟

> Real-time sign language recognition. Live video → detected signs → text → optional speech.

## Stack
- **Backend**: Python 3.10+, FastAPI, MediaPipe Tasks API, onnxruntime, uvicorn
- **Frontend**: React 18, TypeScript, Vite 6, WebRTC
- **Model**: trained in [`../openhand-model`](../openhand-model) — 62K-param MLP on MediaPipe landmarks, 98.95% test accuracy
- **Optional**: ElevenLabs API (text → speech)

## Features (MVP)
- [x] Live webcam feed (circular preview, mirrored)
- [x] Hand landmark detection (MediaPipe Tasks API, 21 landmarks)
- [x] **ML-based ASL classifier — A–Z, ONNX, ~0.02 ms/frame on CPU**
- [x] Real-time text output (debounced sign accumulation)
- [x] WebSocket-based frame streaming (10fps)
- [x] Optional speech output (ElevenLabs)

## Phase 2
- [ ] Learning mode: match signs to prompts
- [ ] Words/phrases (will need a temporal model — the ASL Fingerspelling
      dataset is already downloaded in `../openhand-model/data/raw/`)
- [ ] Mirror-augment training so left-handed signers work as well

## Structure
```
openhand/
  backend/
    main.py               # FastAPI app, CORS config
    api/
      routes.py           # WebSocket /ws/detect, POST /api/tts, GET /api/health
    services/
      mediapipe_service.py  # Decodes base64 JPEG frames, runs MediaPipe Tasks HandLandmarker
      classifier.py         # ONNX MLP → DetectionResult (A–Z + confidence)
      tts.py                # ElevenLabs TTS (async, optional)
    models/
      schemas.py            # Pydantic: Landmark, DetectionResult, TTSRequest/Response
      artifacts/
        asl_classifier.onnx     # Trained MLP (250 KB)
        model_meta.json         # Label map + training metadata
        hand_landmarker.task    # MediaPipe hand landmarker model (~8 MB)
    requirements.txt
    .env.example
  frontend/
    src/
      App.tsx               # Root: webcam state, sign debounce, TTS playback
      App.css               # All styles — off-white, minimal, matches design.png
      components/
        WebcamFeed.tsx      # Circular <video> preview + status dot
        SignDisplay.tsx     # DETECTED / CONFIDENCE stat cards
        TextOutput.tsx      # Accumulated text, clear + speak buttons
        icons.tsx           # HandIcon SVG
      hooks/
        useWebcam.ts        # getUserMedia start/stop, status: idle|requesting|active|error
        useSignDetection.ts # WebSocket client, canvas frame capture loop
    index.html
    vite.config.ts          # @vitejs/plugin-react
    tsconfig.json
    package.json
    .env.example
  .claude/
    settings.json           # Project permissions
  Design.png                # UI reference
  start-backend.ps1
  start-frontend.ps1
```

## Running locally

The alphabet MLP (`asl_classifier.onnx`, ~250 KB) is checked into the
repo. The MediaPipe `.task` files (~17 MB total) and the CTC transcribe
model (`asl_ctc.onnx`, 116 MB) are not — they need to be fetched on
first setup. See "First-time setup" below.

### First-time setup
```powershell
# Backend
cd openhand/backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download MediaPipe Hand/Pose/Face .task files (idempotent)
python scripts/download_models.py

# CTC transcribe model isn't downloaded automatically — copy it from
# the training repo. The transcribe endpoint will fail if missing, but
# the live letter path still works without it.
cp ../../openhand-model/exports/ctc/asl_ctc.onnx       models/artifacts/
cp ../../openhand-model/exports/ctc/model_meta.json    models/artifacts/asl_ctc_meta.json
```

### Backend
```powershell
cd openhand
.\backend\venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8273 --reload
```

You should see two MediaPipe info lines on startup (XNNPACK delegate, feedback
manager warning) — both are harmless. Hit http://localhost:8273/api/health to
confirm — should return `{"status":"ok"}`.

### Frontend
```powershell
cd openhand/frontend
npm install
npm run dev
# → http://localhost:5273
```

Allow webcam access in the browser. Hold up A–Z hand shapes — predictions
should appear with confidence around 95–98% per frame.

### Environment variables
Copy `.env.example` files and fill in as needed:
- `backend/.env` — `ELEVENLABS_API_KEY` (optional, enables TTS)
- `frontend/.env` — `VITE_TTS_ENABLED=true` (shows speak button in UI)

### Retraining the model
See [`../openhand-model/README.md`](../openhand-model/README.md). The training
pipeline is fully separate from the backend — it produces
`asl_classifier.onnx` + `model_meta.json`, which you copy into
`backend/models/artifacts/` to update the deployed model.

## Architecture notes

- **Frame pipeline**: frontend captures video frames to an offscreen `<canvas>` at 100ms intervals, encodes as base64 JPEG (quality 0.6), and sends over a WebSocket to the backend.
- **Detection**: backend decodes the frame with OpenCV, passes an `mp.Image` to MediaPipe Tasks API `HandLandmarker`, extracts 21 3D landmarks per hand.
- **Classification**: `SignClassifier` loads `asl_classifier.onnx` once at startup via `onnxruntime`. Per frame it builds the 63-float landmark vector, wrist-centres + 95th-percentile-scales it (matching the training-time normalisation in `openhand-model`), runs the MLP, softmaxes the logits, and returns the argmax letter + confidence. Below 0.5 confidence we return `"—"` rather than guessing.
- **Concurrency**: the MediaPipe `HandLandmarker` in IMAGE mode is not thread-safe, so `routes.py` serialises calls under a single `threading.Lock`. WebSocket clients are still handled concurrently — only the per-frame detection step is single-flighted.
- **Text accumulation**: detected signs are debounced (800ms) and appended to an output string in the frontend. The same sign is not repeated consecutively.
- **TTS**: on demand via ElevenLabs REST API; audio is streamed back as `audio/mpeg` and played via the Web Audio API.
- **WebSocket reconnection**: `useSignDetection` retries the WebSocket connection on each capture interval if the socket is not open.

## Model

| Metric | Value |
|--------|-------|
| Architecture | MLP `[63 → 256 → 128 → 64 → 26]`, BatchNorm + Dropout(0.3), ReLU |
| Parameters | 62,267 |
| Trained on | ASL Alphabet (Kaggle, 87K images → 62,819 landmark vectors) |
| Test accuracy | 98.95% (held-out 5%) |
| CPU inference | 0.019 ms/frame (onnxruntime) |
| Input | 63 floats — 21 MediaPipe landmarks × (x, y, z), wrist at origin, unit-scaled |
| Output | 26 logits A–Z |

Full training pipeline + retraining instructions live in
[`../openhand-model/README.md`](../openhand-model/README.md).

### Future work
- Temporal model (LSTM / 1D-CNN / Transformer + CTC) trained on the
  fingerspelling dataset already at `../openhand-model/data/raw/` — would
  recover J/Z motion-dependent signs and enable phrase-level transcription.
- Mirror-augment training samples for left-handed signers.

## Known limitations
- **J and Z** require motion to disambiguate from I and D. The single-frame
  model occasionally confuses these — a temporal model is the proper fix.
- **Single dominant hand** — `num_hands=1` in MediaPipe config.
- **Single signer in training data** — the ASL Alphabet dataset is captured
  from one signer. Real-world accuracy on unseen hands/lighting will be
  lower than the 98.95% test number.
- Node 22.11 triggers engine warnings from Vite 6 (requires 22.12+); warnings are harmless.
