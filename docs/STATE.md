# OpenHand — Current State

> Snapshot of what's live, what's pending, and how the pieces fit together.
> Companion to `PROGRESS.md` (how we got here).
> Last verified: 2026-05-13.

## Two repos, two responsibilities

| Repo | Role |
|---|---|
| **`openhand-model/`** | Training pipelines. Produces ONNX files and metadata. Has its own venv, never imports from `openhand/`. |
| **`openhand/`** | The application. Backend (FastAPI) + frontend (React/Vite). Consumes ONNX files copied from the training repo. |

The training repo is otherwise self-contained: model definitions, dataset
classes, training scripts, ONNX export. To redeploy a new model, you
retrain there, run the export script, and copy two files into
`openhand/backend/models/artifacts/`. The application has no PyTorch
dependency — just `onnxruntime`.

## What runs today

### Two ML paths in production

**Path A: Live letter classifier** (per-frame, real-time)

```
webcam frame (10 fps)
    → MediaPipe HandLandmarker (21 landmarks)
    → wrist-centred & scaled (63 floats)
    → MLP ONNX → 26 logits (A–Z)
    → softmax → letter + confidence (or "—" if < 0.5)
    → WebSocket /ws/detect → frontend
```

Latency budget: ~25 ms/frame end-to-end (mostly MediaPipe). Inference
itself is 0.02 ms.

**Path B: Phrase transcription** (hold-to-record, accurate)

```
hold button → buffer 100ms frames (max 120 = 12s)
release → POST /api/transcribe { frames: [base64...] }
    → MediaPipe Hand + Pose + Face per frame (~15-25 ms each)
    → build 381-feature vector + missing mask
    → wrist-anchored normalisation
    → CTC Transformer ONNX → log-probs (T, V=60)
    → greedy or beam decode
    → string
```

Latency: ~16 ms inference + frame extraction time (~5 s of frames takes
~1.5 s to fully process).

### Frontend toggle

A pill toggle in the UI switches between the two paths:
- "Live letter" — the existing per-frame flow, with the live skeleton
  overlay drawn over the webcam
- "Phrase transcribe" — a hold-to-record button that captures a clip,
  posts it, and appends the result to the output text bar

Both share the same TTS button at the bottom (ElevenLabs API, optional).

## Performance numbers

| Path | Metric | Value |
|---|---|---|
| **Live letter** | Test accuracy (held-out alphabet split) | 98.95% |
| | CPU inference (per frame) | 0.019 ms |
| | Model size | 250 KB (62K params) |
| **CTC transcribe** | Val CER (signer-held-out) | 0.249 |
| | CPU inference (per 128-frame clip) | 16 ms |
| | Model size | 116 MB (27.5M params) |
| | Beam search delta vs greedy | ~0.002 CER (model too confident) |

For reference, the Kaggle 1st-place CTC solution on this dataset hit
CER ~0.21 on test. We're at 0.249 on a different (signer-held-out) split
with a smaller model.

## File-by-file map

### `openhand-model/` (training)

```
model/
  landmarks.py             # 127-landmark selection + per-sequence normalisation
  ctc_transformer.py       # Conv1d stem → Transformer encoder → CTC head
  fingerspelling_dataset.py  # Dataset over per-sequence .npz, w/ augmentation
  mlp.py                   # The 62K-param alphabet MLP
  dataset.py               # PyTorch Dataset for the alphabet .npy splits
scripts/
  download_data.py             # Kaggle CLI wrapper for fingerspelling dataset
  preprocess_alphabet.py       # MediaPipe Hands over ASL Alphabet → X.npy/y.npy
  preprocess_fingerspelling.py # Parquet → per-sequence .npz (one-time)
  preprocess.py                # ABANDONED — old fingerspelling-as-classification
  train.py                     # Train the alphabet MLP
  train_ctc.py                 # Train the CTC transformer
  export_onnx.py               # Export alphabet MLP → ONNX
  export_ctc_onnx.py           # Export CTC transformer → ONNX (dynamo path)
  evaluate.py                  # Per-class accuracy + confusion matrix
  infer.py                     # Single-frame inference smoke test
  _download_model.py           # One-off MediaPipe .task downloader
exports/
  asl_classifier.onnx        # Alphabet MLP ONNX (deployed)
  best.pt                    # Alphabet MLP PyTorch checkpoint
  model_meta.json            # Alphabet vocab + accuracy
  training_curves.png        # Alphabet training plot
  ctc/
    asl_ctc.onnx             # CTC ONNX (deployed)
    best.pt                  # CTC PyTorch checkpoint (run 3)
    model_meta.json          # CTC vocab + training history
    *.bak                    # Run-2 and run-1 backups (gitignored)
data/                        # All gitignored — downloaded via scripts
  asl-alphabet/                # ~87K JPEGs
  raw/                         # Fingerspelling Parquet (~160 GB)
  processed_alphabet/          # X.npy, y.npy, label_map.json (62K samples)
  processed_fingerspelling/    # Per-sequence .npz (67K, ~11 GB)
  hand_landmarker.task         # MediaPipe model (used during preprocessing)
CTC_TRAINING_STATE.md          # Living doc for the CTC training effort
README.md                      # Top-level overview + how to retrain
```

### `openhand/` (application)

```
backend/
  main.py                  # FastAPI app, CORS
  api/
    routes.py              # /ws/detect, /api/transcribe, /api/tts, /api/health
  services/
    mediapipe_service.py   # HandDetector for live path
    holistic_service.py    # HolisticDetector (hand+pose+face) for CTC path
    classifier.py          # SignClassifier — MLP ONNX wrapper (live letter)
    ctc_classifier.py      # CTCClassifier — transformer ONNX + greedy/beam decode
    ctc_landmarks.py       # Mirror of training-time landmark selection
    tts.py                 # ElevenLabs TTS (async, optional)
  models/
    schemas.py             # Pydantic models (Landmark, DetectionResult, …)
    artifacts/
      asl_classifier.onnx        # Alphabet MLP (250 KB)
      asl_ctc.onnx               # CTC transformer (116 MB)
      asl_ctc_meta.json          # CTC vocab + meta
      model_meta.json            # Alphabet vocab + meta
      hand_landmarker.task       # MediaPipe Hand (8 MB)
      pose_landmarker.task       # MediaPipe Pose (5.5 MB)
      face_landmarker.task       # MediaPipe Face (3.6 MB)
  _download_models.py      # Setup script — downloads the .task files
frontend/
  src/
    App.tsx                # Top-level UI, mode toggle, debounce, TTS
    App.css                # All styles
    main.tsx               # React root
    components/
      WebcamFeed.tsx       # Video + canvas overlay (landmark skeleton)
      SignDisplay.tsx      # Live mode's "Detected / Confidence" stat cards
      TextOutput.tsx       # Accumulated output bar, clear/speak buttons
      icons.tsx            # Inline SVG icons
    hooks/
      useWebcam.ts         # getUserMedia start/stop
      useSignDetection.ts  # WebSocket client (live letter mode)
      useTranscribe.ts     # Hold-to-record + POST /api/transcribe
docs/
  PROGRESS.md              # The journey
  STATE.md                 # This file
README.md                  # Run instructions
start-backend.ps1          # Setup-and-run helper
start-frontend.ps1         # (likewise for the frontend)
```

## How the two paths share data

- **Frontend** is a single React app. It owns the webcam stream
  (`useWebcam`), and forwards frames to either the WebSocket
  (`useSignDetection`) or the buffered POST (`useTranscribe`) based on
  the current mode.
- **Backend** keeps both paths in `routes.py`. They share the FastAPI app
  but have independent state: `HandDetector` + `SignClassifier` for live,
  `HolisticDetector` + `CTCClassifier` for transcribe.
- The CTC components are **lazy-loaded** on first transcribe call so the
  backend starts up fast even if the user never hits transcribe.
- Concurrency: `HandDetector` is protected by `_detect_lock`,
  `HolisticDetector` by `_holistic_lock`. WebSocket clients are still
  served concurrently, only the per-frame detect calls are serialised.

## Endpoints

| Route | Method | Use |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/tts` | POST | ElevenLabs synthesis (optional, requires API key) |
| `/api/transcribe` | POST | Phrase transcription via CTC |
| `/ws/detect` | WS | Live letter detection (frame stream in, result stream out) |

## Configuration

| Variable | Source | Used in |
|---|---|---|
| `ELEVENLABS_API_KEY` | `backend/.env` | `services/tts.py` |
| `VITE_TTS_ENABLED` | `frontend/.env` | Shows/hides speak button |
| Backend port | hardcoded `8273` | both backend startup + frontend hooks |
| Frontend port | `vite.config.ts` (`5273`) | Vite dev server |
| CORS allowed origins | `backend/main.py` | `http://localhost:5273`, `:3000` |

## How to run

The README in `openhand/` has the full instructions. Quick version:

```powershell
# One-time
cd openhand/backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python _download_models.py

cd ../frontend
npm install

# Each session
cd openhand
.\start-backend.ps1   # → http://localhost:8273
.\start-frontend.ps1  # → http://localhost:5273
```

## Limitations & known issues

| Issue | Severity | Notes |
|---|---|---|
| J and Z imperfect even in CTC path | Medium | Motion-defined letters; CTC handles them but accuracy depends on signing speed |
| Left-handed signers underperform | Medium | Training data is right-hand dominant; mirror augmentation would help |
| One-signer alphabet model overfits | High in production | Live letter mode struggles outside the training distribution |
| Short phrases transcribe poorly | Medium | Common pattern: `"713-809-2808"` → `"i.0/270"`. Likely a function of CTC needing length |
| Backend exits if `HolisticLandmarker` (legacy) is used | Resolved | Switched to three separate detectors |
| Vite dev server sometimes binds IPv6 only | Cosmetic | `localhost` works either way; CORS allows both |
| First `/api/transcribe` call is slow (~2-3 s) | Cosmetic | Detectors + ONNX session lazy-load on first call |

## Watch list (next session)

Carried over from `CTC_TRAINING_STATE.md`:

- [ ] **Language model fusion in beam search** — compound with beam much
      more than greedy. Couple hundred LOC, no retraining. Best expected
      gain-per-effort
- [ ] **Mirror-augment training data** — handles left-handed signers,
      cheap retrain
- [ ] **Streaming inference** — rolling-window decode during live signing
      instead of hold-to-record interaction
- [ ] **Resume-from-checkpoint flag** for `train_ctc.py` — currently a
      crashed run loses optimizer state and restarts from epoch 1
- [ ] **Use the supplemental data** — there's a `supplemental_metadata.csv`
      with 53K extra sequences we haven't trained on

## What's *not* in scope yet

- **Word-level signs** (the 250-class ASL Signs dataset). Different
  vocabulary, different model — would need a separate effort
- **Speaker adaptation** — fine-tuning on a few seconds of the user's own
  signing could massively close the cross-signer gap
- **Mobile deployment** — current architecture targets desktop browser +
  Python backend. ONNX is portable but the full app isn't packaged
- **Two-hand signs and BSL** — model only sees one dominant hand at a
  time in the live path; CTC sees both but the training set is ASL-only
