# OpenHand

Point a webcam at your hand, get text back. OpenHand recognizes American
Sign Language fingerspelling in the browser and pipes it through to text
and, if you want, speech.

There are two paths in the app:

1. **Per-frame letter detection.** Hold up an A-Z handshape, get one
   letter back at a time, debounced into a sliding-window output bar.
   Backed by a small MLP (~62K params) trained on the [Kaggle ASL
   Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
2. **Streaming phrase transcription.** Sign a full word or phrase; a
   rolling buffer of the last ~3 seconds is decoded by a CTC transformer
   every 750ms and committed when you pause. Trained on the [Kaggle ASL
   Fingerspelling competition data](https://www.kaggle.com/competitions/asl-fingerspelling/).

The CTC path is wired end to end but the phrase display is hidden in the
UI right now; the focus is on getting per-letter recognition reliable
first. The Learn screen uses the CTC path for J and Z (motion letters)
behind the scenes.

MediaPipe runs in the browser to extract landmarks. The trained ONNX
models run server-side via `onnxruntime`. No GPU in the runtime path.
Alphabet inference is ~0.02ms per frame on CPU.

None of the trained model files are checked in. Train them in the
sibling repo [openhand-model](https://www.github.com/catherinepereira/openhand-model) and copy the outputs
into `backend/models/artifacts/`. See "Setting up the models" below.

## Setup

Requirements:

- Python 3.10+
- Node 22.12+ (Vite 6 warns on 22.11 but still runs)
- A webcam

```powershell
git clone https://github.com/catherinepereira/openhand
cd openhand

# Backend
python -m venv backend/venv
backend\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
node scripts/download_mediapipe_models.mjs   # ~17 MB of .task files
cd ..
```

On macOS / Linux the venv activation is `source backend/venv/bin/activate`;
everything else is the same.

## Running

```powershell
# Terminal 1
.\start-backend.ps1
# uvicorn on http://localhost:8273

# Terminal 2
.\start-frontend.ps1
# vite on http://localhost:5273
```

Without the launchers:

```powershell
# Backend
backend\venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8273 --reload

# Frontend
cd frontend
npm run dev
```

### Configuration

| File | What it does |
|------|--------------|
| `backend/.env` | Optional. Holds `ELEVENLABS_API_KEY` for TTS. |
| `frontend/.env` | Optional. `VITE_TTS_ENABLED=true` shows the speak button. `VITE_API_BASE=...` overrides the backend origin. |

## How it fits together

1. The frontend captures a frame from the webcam every 100ms.
2. MediaPipe Tasks runs in-browser on each frame to produce 21 hand
   landmarks (alphabet path) and the 127-landmark Holistic set (CTC
   path).
3. For the alphabet path, the 63 floats for the dominant hand go over a
   WebSocket to the backend, which runs the MLP and returns a letter +
   confidence. Below 0.5 confidence the backend returns `-` and the
   frontend doesn't accumulate.
4. For the CTC path (currently used only on the Learn screen for J/Z),
   the 381 floats per frame are buffered into a rolling ~2-second
   window. Every 600ms the frontend sends the buffer to a separate
   WebSocket, the backend runs the CTC transformer + beam search, and
   the decoded string comes back.
5. The "speak" button posts the accumulated text to `/api/tts`, gets
   MP3 audio from ElevenLabs, plays it through `<audio>`.

The hand skeleton overlay on the video is rendered straight from
MediaPipe output, so it stays in sync with the camera regardless of
backend latency.

## Known limitations

- The alphabet model is trained on a single signer. Real-world accuracy
  on unseen hands and lighting is lower than the 98.95% test number;
  cross-signer fine-tuning would help.
- J and Z need motion to disambiguate from I and D. The static-frame
  MLP gets them wrong some of the time; the CTC model handles them
  fine because it sees a window of frames.
- The MLP is rotation-sensitive (it sees raw camera-frame landmarks),
  so signs whose handshape depends on orientation (P, G, H) may need
  exaggerated angles to register.
- MediaPipe's handedness label is camera-POV, not user-POV. A
  right-handed signer with a non-mirrored feed has their right hand
  labeled "Left". The classifier picks "Right" (camera-POV) as the
  dominant hand by convention.
- Only right hand is supported. `numHands` is 2 in MediaPipe but only one (right hand) feeds
  the classifier.

## License

MIT. See [LICENSE](LICENSE).
