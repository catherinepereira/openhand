# OpenHand

Point a webcam at your hand, get text back. OpenHand recognizes American
Sign Language fingerspelling in the browser and pipes it through to text
and, if you want, speech.

There are three paths in the app:

1. **Per-frame letter detection.** Hold up an A-Z handshape, get one
   letter back at a time, debounced into a sliding-window output bar.
   Backed by a small MLP (~62K params) trained on the [Kaggle ASL
   Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
2. **Streaming phrase transcription.** Sign a full fingerspelled word
   or phrase; a rolling buffer of the last ~3 seconds is decoded by a
   CTC transformer. Trained on the [Kaggle ASL Fingerspelling
   competition data](https://www.kaggle.com/competitions/asl-fingerspelling/).
3. **Isolated word recognition.** Sign a whole word (not fingerspelled
   letter-by-letter) and the app classifies it against 250 common ASL
   signs. Backed by a Conv1D-plus-Transformer trained on the [Google
   Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
   competition data. Architecture follows the patterns from the 1st-place
   competition solution.

The CTC path is wired end to end but the phrase display is hidden in the
UI right now; the focus is on getting per-letter recognition reliable
first. The Learn screen uses the CTC path for J and Z (motion letters)
behind the scenes. A "Learn the words" view drives the isolated-sign
classifier with an animated reference clip (medoid clip per sign, chosen
in the trained encoder's embedding space).

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

### Setting up the models

The backend won't start without the alphabet artifacts. Everything below
lives under `backend/models/artifacts/` (gitignored).

| File | Size | Required? | Source |
|------|------|-----------|--------|
| `asl_classifier.onnx` | ~250 KB | yes | `openhand-model/scripts/train.py` + `export_onnx.py` |
| `model_meta.json` | ~1 KB | yes | written by `train.py` |
| `reference_landmarks.json` | ~35 KB | yes (Learn signs) | `openhand-model/scripts/build_reference_landmarks.py` |
| `asl_ctc.onnx` | ~116 MB | optional (phrase + Learn J/Z) | `openhand-model/scripts/train_ctc.py` + `export_ctc_onnx.py` |
| `asl_ctc_meta.json` | ~15 KB | optional | written by `train_ctc.py` |
| `sign_classifier.onnx` | ~10-20 MB | optional (Learn the words) | `openhand-model/scripts/train_signs.py` + `export_signs_onnx.py` |
| `sign_classifier_meta.json` | ~10 KB | optional | written by `train_signs.py` (rename `model_meta.json`) |
| `sign_references.json` | ~30-60 MB | optional | `openhand-model/scripts/build_sign_references.py` |

Once the model repo has produced its artifacts, copy them over:

```powershell
# Alphabet path + Learn-the-signs
copy ..\openhand-model\exports\asl_classifier.onnx       backend\models\artifacts\
copy ..\openhand-model\exports\model_meta.json           backend\models\artifacts\
copy ..\openhand-model\exports\reference_landmarks.json  backend\models\artifacts\

# CTC (optional; needed for J/Z Learn grading + future phrase view)
copy ..\openhand-model\exports\ctc\asl_ctc.onnx          backend\models\artifacts\
copy ..\openhand-model\exports\ctc\model_meta.json       backend\models\artifacts\asl_ctc_meta.json

# Words / isolated signs (optional; needed for Learn-the-words)
copy ..\openhand-model\exports\signs\sign_classifier.onnx  backend\models\artifacts\
copy ..\openhand-model\exports\signs\model_meta.json       backend\models\artifacts\sign_classifier_meta.json
copy ..\openhand-model\exports\signs\sign_references.json  backend\models\artifacts\
```

Without the word artifacts, the "Learn the words" view shows a clear
"sign model unavailable" message and the rest of the app keeps working.

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
   landmarks (alphabet path) and the 127-landmark Holistic set (CTC +
   word paths).
3. For the alphabet path, the 63 floats for the dominant hand go over a
   WebSocket to the backend, which runs the MLP and returns a letter +
   confidence. Below 0.5 confidence the backend returns `-` and the
   frontend doesn't accumulate.
4. For the CTC path (currently used only on the Learn screen for J/Z),
   the 381 floats per frame are buffered into a rolling ~2-second
   window. Every 600ms the frontend sends the buffer to a separate
   WebSocket, the backend runs the CTC transformer + beam search, and
   the decoded string comes back.
5. For the word path (Learn-the-words view), a 3-second rolling buffer
   of (T, 127, 3) raw landmarks plus a missing mask goes over its own
   WebSocket every 800ms. The backend normalizes the clip, builds
   engineered features (motion deltas + hand-to-lip distances), runs
   the Conv1D + Transformer encoder, and returns top-5 (sign, prob).
6. The "speak" button posts the accumulated text to `/api/tts`, gets
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
