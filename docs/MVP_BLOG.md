# OpenHand — building a sign-language interpreter MVP

> A real-time sign-language recognition app. Webcam in, detected letters
> and transcribed phrases out, optional speech synthesis on top. Built
> in roughly a week.

## What it does

OpenHand opens your webcam, recognises American Sign Language in two
modes, and turns it into text — and optionally into spoken audio.

**Live letter mode** continuously detects a single ASL letter from the
camera feed, draws a skeleton overlay on the user's hand for visual
confirmation, and appends each new letter to an output bar with light
debouncing. Roughly 10 frames per second, real-time, no waiting.

**Phrase transcribe mode** lets the user hold a button to record a
short clip (up to about 12 seconds) of continuous fingerspelling. On
release, the clip is sent to the backend, transcribed in one go, and
the result appended to the output.

The output text bar has clear / speak buttons; if an ElevenLabs API key
is configured, the speak button plays the accumulated text aloud.

It runs entirely on a laptop. No GPU needed at inference time. The
heaviest model is 116 MB.

## Why two modes

Initially I built only the live letter mode. It worked, but it had real
limitations:

- **One signer in training data.** The static-image dataset I used has
  87,000 photos but all from a single person, so generalisation to
  anyone else's hands was poor.
- **No motion.** Letters like J and Z are *defined* by hand motion in
  ASL. A model trained on still photos sees arbitrary frozen frames of
  these gestures and learns nothing meaningful about the motion.
- **Letter-by-letter only.** Real fingerspelling flows continuously
  between letters with no discrete pauses. Asking the user to deliberately
  hold each letter felt clunky.

After hitting these ceilings I built a second model on a real sequence
dataset (Google's ASL Fingerspelling Kaggle dataset, 67K phrases, 100+
signers, with multi-frame motion) using a Transformer + CTC loss.
That's the phrase transcribe mode. The blog post
[CTC_BLOG.md](CTC_BLOG.md) covers how that model was trained.

The two modes coexist in the app via a toggle. The live mode is
near-instant per frame; the transcribe mode is more accurate and
handles phrases, motion letters, and cross-signer variation, but
requires the user to deliberately record a clip.

## The architecture in one picture

```
                                    ┌──────────────────────────────┐
                                    │  React + Vite frontend       │
                                    │  (mode toggle, hold-to-record, │
                                    │   skeleton overlay, TTS)     │
                                    └──────────────┬───────────────┘
                                                   │
                              ┌────────────────────┴────────────────────┐
                              │                                         │
                  WebSocket /ws/detect                          POST /api/transcribe
                  (1 frame / 100ms)                            (buffered clip of frames)
                              │                                         │
                              ▼                                         ▼
                   ┌──────────────────────┐               ┌───────────────────────┐
                   │ MediaPipe Hand only  │               │ MediaPipe Hand+Pose+Face │
                   │ (21 landmarks/frame) │               │ (127 landmarks/frame)  │
                   └──────────┬───────────┘               └────────────┬──────────┘
                              │                                        │
                              ▼                                        ▼
                   ┌──────────────────────┐               ┌───────────────────────┐
                   │  Alphabet MLP        │               │  CTC Transformer       │
                   │  • 250 KB ONNX       │               │  • 116 MB ONNX         │
                   │  • 62K params        │               │  • 27.5M params        │
                   │  • 0.02 ms/frame CPU │               │  • 16 ms / 128 frames  │
                   │  • 26 logits A–Z     │               │  • CTC greedy/beam     │
                   └──────────┬───────────┘               └────────────┬──────────┘
                              │                                        │
                              ▼                                        ▼
                       letter + conf                                 phrase
```

The backend is FastAPI; both paths live in the same process and share
the same MediaPipe wheel and the same `onnxruntime` runtime. The
frontend is a single Vite + React app.

## The tech stack

**Backend:**
- Python 3.10+, FastAPI, uvicorn
- MediaPipe Tasks API (Hand, Pose, Face landmark detectors)
- onnxruntime for both ML models
- OpenCV for image decoding
- ElevenLabs HTTP API for optional TTS

**Frontend:**
- React 19, TypeScript, Vite 6
- WebRTC for camera access
- Canvas API for the live skeleton overlay
- Native fetch + WebSocket — no client libraries

**Training (separate repo):**
- PyTorch 2.x with AMP
- nn.CTCLoss for sequence training
- torch.onnx dynamo exporter
- pandas + numpy for data prep

The training repo and the application repo are deliberately separate.
The training repo produces ONNX files; the app repo consumes them.
There's no Python-side dependency between them, just a copy-the-file
step at deploy time.

## Numbers

**Alphabet model (live letter path)**
- Trained on Kaggle ASL Alphabet, 87K images → 62,819 after MediaPipe
  filtering for "hand detected"
- 4-layer MLP, 62K params
- 98.95% test accuracy on the held-out split (same signer)
- 0.019 ms inference per frame on CPU

**CTC model (phrase transcribe path)**
- Trained on Kaggle ASL Fingerspelling, 67,208 phrases × ~150 frames each
- 12-layer Transformer encoder + CTC head, 27.5M params
- Val character error rate 0.249 on 5 held-out signers
- 16 ms inference per 128-frame clip on CPU

For context the Kaggle 1st-place CTC solution reached about 0.21 CER on
the held-out test set. We're 4 points behind a competition winner with
greedy decoding and significantly less hyperparameter tuning.

## Three things that turned out to matter most

Looking back, the bulk of the value came from things I would have
guessed beforehand were "infrastructure" or "setup" work.

**1. Getting the live mode shipped first, even with a worse model.**
The first working version used an MLP on a single-signer alphabet
dataset. The model itself wasn't great in production conditions, but
having the end-to-end pipeline working — webcam → MediaPipe → ONNX →
WebSocket → frontend → debounce → output bar — exposed every other
problem clearly. By the time the CTC model was ready, the integration
was a known quantity.

**2. The data pipeline ate most of the time.**
The CTC training data is 160 GB of Parquet files. My first Dataset
class loaded shards lazily under a shuffled DataLoader, which thrashed
through the same files dozens of times per step. Training was running
at 57 minutes per epoch. Pre-extracting each sequence to a small `.npz`
once took 10 minutes and produced 11 GB on disk. Training dropped to
17 seconds per epoch — about 200× faster. The boring fix dominated
everything else.

**3. Anti-blank-collapse for CTC.**
CTC training has a well-known failure mode where the model learns to
predict "blank" on every frame as a local minimum, and never escapes.
My first CTC run produced a CER of 0.94 — essentially zero output —
after a full 30 epochs. Three fixes worked: a linear LR warmup, a
secondary KL-divergence-to-uniform regularisation term, and cosine LR
decay. Together they turned a non-functional model into one with
CER 0.27 in fewer epochs than the failed attempt.

The lesson, restated: model architecture mattered less than I expected.
Bumping from 5.5M to 27.5M parameters moved CER from 0.27 to 0.25.
Fixing data loading, fixing the missing-data sentinel, and fixing
blank-collapse moved CER from "unusable" to "deployable."

## Three things I didn't expect

**MediaPipe's new Tasks API broke the old API in a backwards-incompatible
way.** The legacy `mp.solutions.hands` was removed in MediaPipe 0.10.20.
The new `mediapipe.tasks.python.vision.HandLandmarker` requires a
separate `.task` model file (8 MB) that has to be downloaded once. The
backend has a setup script that fetches it.

**The all-in-one `HolisticLandmarker` crashed the backend.** On Windows
+ MediaPipe 0.10.21, when an internal sub-task produces an empty packet,
the next stage triggers a fatal C++ assertion that takes down the whole
Python process — no traceback, just a dead process. I replaced it with
three separate Tasks detectors (Hand, Pose, Face) running in series.
Slower per frame but stable.

**The PyTorch → ONNX export was a maze.** The legacy tracer baked the
dummy time dimension into MultiheadAttention's internal reshape ops.
The new dynamo-based exporter handled that but couldn't convert
BatchNorm in eval mode. Fix: fuse BN into the preceding Conv1d weights
before export. Then I discovered the dynamo exporter prints a unicode
checkmark on success which crashed Windows' cp1252 console. Then I
discovered the dynamo exporter needs batch ≥ 2 at runtime to preserve
batch as a dynamic axis, so the backend pads inference with a fully-
masked dummy second batch item.

Each one was 10-30 minutes to diagnose and fix, and each one would have
been a half-day if encountered cold without the surrounding context.

## What this MVP is *not*

To be clear about scope:

- **Not a complete ASL recogniser.** Only fingerspelling. No
  word-level or grammatical signs.
- **Not real-time phrase transcription.** The phrase mode is
  hold-to-record, not streaming. Streaming would require solving the
  "when has the user finished a phrase" problem, which is harder than
  it sounds.
- **Not multi-handed or bilingual.** ASL fingerspelling only, single
  dominant hand in the live mode.
- **Not deployed.** Runs locally on a development machine. Packaging
  for distribution is future work.
- **Not perfect on novel signers.** The CTC model is signer-agnostic
  by design (100+ signers in training) but the alphabet model in the
  live mode is one-signer-trained and will struggle outside its
  distribution.

## What's where in the repo

```
openhand/
├── backend/                  FastAPI app, both ONNX models, MediaPipe
│   ├── api/routes.py             /ws/detect, /api/transcribe, /api/tts
│   ├── services/                 detectors, classifiers, frame decode
│   └── models/artifacts/         ONNX + MediaPipe .task files
├── frontend/                 React + Vite single-page app
│   └── src/
│       ├── hooks/                useWebcam, useSignDetection, useTranscribe
│       └── components/           WebcamFeed with overlay, etc.
├── docs/
│   ├── MVP_BLOG.md               this file
│   ├── CTC_BLOG.md               training the sequence model
│   ├── PROGRESS.md               chronological project journal
│   ├── STATE.md                  current technical state
│   └── CODE_REVIEW.md            review of dead code & efficiency
└── README.md                 how to run locally
```

The training repo (`openhand-model/`) lives separately and is only
needed if you want to retrain the models. The application repo above
ships pre-trained ONNX files.

## How to run it

The README has the full version. Short story:

```powershell
# One-time
cd openhand
.\start-backend.ps1   # creates venv, installs deps, downloads task files
.\start-frontend.ps1  # installs npm deps

# Subsequent
.\start-backend.ps1     # http://localhost:8273
.\start-frontend.ps1    # http://localhost:5273
```

Then open `http://localhost:5273` in a browser, allow webcam access,
and pick a mode. The live letter mode works immediately. The phrase
transcribe mode needs the 116 MB CTC ONNX manually copied from the
training repo (see README) — the first run gives you a clear error
message if it's missing.

## Where it goes from here

The most promising next steps, roughly in order of bang-for-the-buck:

1. **Language model fusion in beam search.** Beam search on its own
   barely helps the CTC model (the softmax outputs are too confident to
   benefit from beam exploration). But beam + character n-gram LM
   compounds — the LM catches realistic URL/address patterns that the
   model alone has no reason to prefer.
2. **Mirror augmentation for left-handed signers.** Cheap retrain,
   doubles effective training samples by flipping x-coordinates.
3. **Streaming inference.** Sliding-window CTC decode during live
   signing would feel much more natural than hold-to-record.
4. **Word-level signs.** Different model, different dataset (Google
   ASL Signs has 250 isolated word classes), but plugs into the same
   architecture cleanly.

I don't have a deployment plan yet. Browser-side inference via
ONNX Runtime Web would mean no backend at all, which would be ideal for
privacy and distribution. The model sizes (250 KB alphabet, 116 MB CTC)
are at the boundary of "ship it to every user." Worth exploring.
