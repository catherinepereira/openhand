# OpenHand

Point a webcam at your hand, get text back. OpenHand recognizes American
Sign Language fingerspelling in the browser and pipes it through to text
(and, if you want, speech).

It does two things:

1. **Per-frame letter detection.** Hold up a static A-Z handshape; you get
   one letter back at a time, debounced into an output bar. Uses a small
   MLP (~62K params) trained on the Kaggle ASL Alphabet dataset.
2. **Streaming phrase transcription.** Sign a whole word or phrase
   naturally; the rolling buffer of the last ~3 seconds is decoded by a
   CTC transformer every 750ms and committed when you pause. Trained on
   the Kaggle ASL Fingerspelling competition data.

MediaPipe runs in the browser to extract landmarks; the trained ONNX
models run server-side via `onnxruntime`. There is no GPU in the runtime
path. Inference latency for the alphabet model is well under a
millisecond on CPU.

## Repo layout

```
openhand/
  backend/                 FastAPI, MediaPipe-less (browser does that part)
    main.py
    api/routes.py          WebSocket + HTTP routes
    services/
      classifier.py        Alphabet MLP (ONNX)
      ctc_classifier.py    CTC transformer + beam search (ONNX)
      ctc_landmarks.py     Feature packing + normalization (mirrors training)
      tts.py               ElevenLabs (optional)
    models/
      schemas.py           Pydantic request/response types
      artifacts/           Model files live here at runtime
    tests/                 pytest
    requirements.txt
    requirements-dev.txt
    pyproject.toml
    .env.example
  frontend/                React 19 + Vite + TypeScript
    src/
      App.tsx              Top-level layout + sign debounce
      components/          Webcam frame, sign cards, output bar
      hooks/
        useWebcam.ts         getUserMedia lifecycle
        useMediaPipe.ts      Shared per-frame detector loop
        useSignDetection.ts  Alphabet WebSocket client
        useStreamingTranscribe.ts  CTC WebSocket client
      lib/
        landmarks.ts       127-landmark packing + normalization (mirrors backend)
        mediapipe.ts       MediaPipe Tasks wrappers
    scripts/
      download_mediapipe_models.mjs  Pulls the .task files post-install
    public/
      models/              .task files land here (gitignored)
    vite.config.ts
    tsconfig.json
    .env.example
  start-backend.ps1        Convenience launcher (Windows)
  start-frontend.ps1
```

Trained model weights and training code live in a sibling repo,
[openhand-model](../openhand-model). The alphabet model is small enough
(~250KB ONNX) that it's checked into this repo at
`backend/models/artifacts/asl_classifier.onnx`. The CTC model is 116MB
and isn't; you fetch it from the model repo (see below).

## Setup

You'll need:

- Python 3.10 or newer
- Node 22.12+ (Vite 6 will warn on 22.11 but still run)
- A webcam

```powershell
git clone https://github.com/<you>/openhand
cd openhand

# Backend
python -m venv backend/venv
backend\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
node scripts/download_mediapipe_models.mjs   # pulls ~17MB of .task files
cd ..
```

On macOS / Linux the venv activation is `source backend/venv/bin/activate`
and the rest is the same.

That's enough to run the alphabet path. The phrase transcription path
needs the CTC ONNX too:

```powershell
# Train and export it from the model repo (see openhand-model/README.md)
# then copy the artifacts in:
copy ..\openhand-model\exports\ctc\asl_ctc.onnx    backend\models\artifacts\
copy ..\openhand-model\exports\ctc\model_meta.json backend\models\artifacts\asl_ctc_meta.json
```

If you skip this, the phrase display will sit at `...` and the WebSocket
will close with an error message; the alphabet path still works.

## Running

Two terminals:

```powershell
# Terminal 1
.\start-backend.ps1
# uvicorn on http://localhost:8273

# Terminal 2
.\start-frontend.ps1
# vite on http://localhost:5273
```

Or, without the convenience scripts:

```powershell
# Backend
backend\venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8273 --reload

# Frontend
cd frontend
npm run dev
```

Open http://localhost:5273, allow webcam access, and start signing.

### Configuration

| File | What it does |
|------|-------------|
| `backend/.env` | Optional. Only used for the ElevenLabs API key (text-to-speech). |
| `frontend/.env` | Optional. `VITE_TTS_ENABLED=true` shows the speak button; `VITE_API_BASE=...` lets you point the frontend at a non-default backend origin. |
| `backend/pyproject.toml` | pytest config; nothing user-tunable. |
| `frontend/vite.config.ts` | Standard Vite + React plugin. Dev server port 5273. |
| `frontend/tsconfig.json` | TypeScript strict mode. |

Copy `*.env.example` to `*.env` to start; both example files document the
keys.

## Testing

```powershell
# Backend
backend\venv\Scripts\Activate.ps1
pip install -r backend/requirements-dev.txt
pytest backend/tests

# Frontend
cd frontend
npm test
```

Tests that depend on the CTC ONNX (116MB, not checked in) skip
automatically if the file isn't present.

## How it fits together

1. The frontend captures a frame from the webcam every 100ms.
2. MediaPipe Tasks runs in-browser on each frame to pull out 21 hand
   landmarks (alphabet path) plus the 127-landmark Holistic set (phrase
   path).
3. For the alphabet path, the 63 floats for the dominant hand go over a
   WebSocket to the backend, which runs the MLP and returns a letter +
   confidence. Below 0.5 confidence we return `-` and don't accumulate.
4. For the phrase path, the 381 floats per frame are buffered into a
   rolling ~3-second window. Every 750ms the frontend sends the buffer
   to a different WebSocket, the backend runs the CTC transformer + beam
   search, and the decoded string comes back. When you pause (the wrist
   stops moving for ~600ms), the latest decode is committed.
5. The "speak" button calls ElevenLabs over HTTP and plays the returned
   MP3 through `<audio>`.

The frontend also draws a hand skeleton overlay on the video feed, fed
straight from the MediaPipe output so it stays in sync with the camera
regardless of backend latency.

## The model

Both models are trained in [openhand-model](../openhand-model). Short
version:

| | Alphabet | CTC phrase |
|-|----------|------------|
| Architecture | 3-layer MLP, BatchNorm + Dropout | Conv1D stem + 6-layer transformer encoder + CTC head |
| Input | 21 landmarks * (x, y, z) = 63 floats | 127 landmarks * 3 = 381 floats per frame, variable length |
| Output | 26 logits (A-Z) | log-probs over 59 chars + blank |
| Params | ~62K | ~5.5M |
| CPU inference | 0.019ms / frame | ~50-150ms / decode (sequence-length dependent) |
| Dataset | Kaggle ASL Alphabet (87K images, one signer) | Kaggle ASL Fingerspelling (~67K phrases, 100+ signers) |
| Validation | 98.95% top-1 on held-out 5% | CER 0.235 on held-out signers (beam search) |

The training story (data pipeline rewrites, dealing with NaN landmark
sentinels, why CTC, etc.) is in `openhand-model/README.md`.

## Known limitations

- The alphabet model is trained on a single signer. Real-world accuracy
  on unseen hands and lighting is meaningfully lower than the 98.95%
  test number. Cross-signer fine-tuning would help.
- J and Z need motion to disambiguate from I and D. The static-frame
  alphabet model gets these wrong some of the time; the CTC model
  handles them fine because it sees a window of frames.
- MediaPipe's handedness label is camera-POV, not user-POV. A
  right-handed signer with a non-mirrored feed has their right hand
  labeled "Left". The classifier picks "Right" (camera-POV) as the
  dominant hand by convention, so left-handed signers may work better
  out of the box than right-handed ones.
- No support for multiple simultaneous hands. `numHands` is 2 in
  MediaPipe but only one feeds the classifier.

## License

MIT, see [LICENSE](LICENSE).
