# OpenHand: Current State

> Snapshot of what's live, what's pending, and how the pieces fit together.
> Companion to `PROGRESS.md` (how we got here) and `OVERLAY_OFFSET_BUG.md`
> (a debugging post-mortem).
> Last verified: after the browser-MediaPipe migration.

## Two repos, two responsibilities

| Repo | Role |
|---|---|
| **`openhand-model/`** | Training pipelines. Produces ONNX files and metadata. Has its own venv, never imports from `openhand/`. |
| **`openhand/`** | The application. Backend (FastAPI + onnxruntime) + frontend (React/Vite + MediaPipe Tasks JS). Consumes ONNX files copied from the training repo. |

The training repo is self-contained. To redeploy a new model: retrain
there, run the export script, copy the resulting `asl_*.onnx` +
`*_meta.json` into `openhand/backend/models/artifacts/`. The
application has no PyTorch or MediaPipe Python dependency; just
`onnxruntime` and `numpy`.

## Architecture

The app is now **MediaPipe-on-the-frontend, ONNX-on-the-backend**.
Browser MediaPipe Tasks JS extracts hand/pose/face landmarks from the
webcam; only landmark coordinates ever leave the user's machine. The
backend is inference-only.

```
                              +--------------------------------+
                              |  React + Vite frontend         |
                              |                                |
                              |  useMediaPipe -- shared        |
                              |   producer (one MediaPipe pass |
                              |   per frame, fan-out below)    |
                              |                                |
                              |  |- useSignDetection           |
                              |  |    live letter detection    |
                              |  |    over WebSocket           |
                              |  |                             |
                              |  +- useStreamingTranscribe     |
                              |      continuous CTC over a     |
                              |      rolling buffer            |
                              +-------------+------------------+
                                            |
                          +-----------------+-----------------+
                          |                                   |
                  WS /ws/detect-landmarks            WS /ws/transcribe-stream
                  (per-frame DetectionResult)       (rolling decode every ~750ms)
                          |                                   |
                          v                                   v
                +----------------------+         +--------------------------+
                |  Alphabet MLP ONNX   |         |  CTC Transformer ONNX    |
                |  - 250 KB, 62K params|         |  - 116 MB, 27.5M params  |
                |  - 0.02 ms/frame CPU |         |  - ~16 ms / 128 frames   |
                |  - 26 logits A-Z     |         |  - greedy CTC decode     |
                +----------------------+         +--------------------------+
```

## What's deployed

| Surface | Status |
|---|---|
| Browser MediaPipe (Hand, Pose, Face) | Running. Detectors instantiated once via `lib/mediapipe.ts`, fed an offscreen-canvas snapshot of the video at native resolution (see overlay-offset bug doc for why). |
| 127-landmark feature packing + normalization | `lib/landmarks.ts`. Strongly typed (group-named, not positional). Validation harness in `lib/__tests__/`. |
| Skeleton overlay | Drawn on a canvas sized to the video stream resolution. Aligns pixel-perfectly with the displayed video. |
| Live-letter via `/ws/detect-landmarks` | Working. Frontend sends `DetectLandmarksRequest` per frame, backend runs alphabet MLP on the "Right" hand (MediaPipe camera POV). |
| Phrase transcription | **Mid-migration.** Currently hold-to-record (`useTranscribe` -> POST `/api/transcribe-landmarks`). About to be replaced with always-on streaming. |

## What's about to ship (in this session)

The remaining migration work, in order:

1. **`useMediaPipe` shared producer hook**: currently `useSignDetection`
   and `useTranscribe` each call `createDetectors()` and run
   `detectFrame()` independently, paying 2x the MediaPipe cost per frame.
   Consolidate into one producer that fans out to both consumers.
2. **Always-on streaming transcription**: replace `useTranscribe` (hold
   to record) with `useStreamingTranscribe` (continuous WebSocket).
   Rolling buffer of ~30 frames (3s at 10fps), re-decoded every ~750ms.
3. **Backend WebSocket `/ws/transcribe-stream`**: accepts decode
   requests, runs the CTC model, returns text. Message shapes documented
   inline in the route.
4. **Drop the mode toggle**: both displays (live-letter card and phrase
   text) render simultaneously, all the time.
5. **"Show skeleton" UI toggle**: for low-end devices that can't draw
   the overlay smoothly.

Configuration values for #2 live in a single named `TRANSCRIBE_CONFIG`
object at the top of `useStreamingTranscribe`, documented inline so
they're not magic numbers.

### Streaming-transcription smoothing heuristics

The raw output of "re-decode the last 3 seconds every 750 ms and show
it" was unusable on its own; letters flickered, swapped and reordered
between decodes as the rolling buffer shifted by a few frames each tick.
Worse, text appeared even when no hand was visible (the model would
hallucinate phrases on empty frames).

We tried **stable-prefix locking** first (display the longest character
prefix common to the last N decodes) but it doesn't work for this
dataset: the model's decodes for ~3 seconds of fingerspelling fluctuate
*wildly* between ticks, so the common prefix is almost always empty.

What actually works is **commit-on-pause** with a tentative display.
Implemented in `useStreamingTranscribe`:

1. **Tentative display.** Every tick the latest decode shows in faded /
   italic text. It twitches and revises; that's fine, it visibly
   indicates the model is working on the current phrase.

2. **Stillness-triggered commit.** Per-frame the hook measures wrist
   movement between consecutive frames. When the dominant hand stays
   still (movement < `stillnessThreshold`) for `stillnessFramesToCommit`
   frames in a row (default 6 = 600 ms, a natural mid-phrase pause),
   the latest decode commits into the solid-text output. The rolling
   buffer resets, so the next phrase starts fresh.

3. **Hand-gated buffering + silence discard.** Frames with no detected
   hand are *not* pushed into the buffer (so the model never sees
   empty-frame stretches to hallucinate from). After
   `silenceFramesToClear` consecutive no-hand frames (default 10, 1s),
   the buffer + tentative text both clear *without* committing.
   Lowering the hand silently ends a phrase you don't want.

Tunable constants live in `TRANSCRIBE_CONFIG` at the top of the hook:

- `stillnessThreshold`: bigger value = need bigger hand movement to
  *avoid* commit. Tune lower if commit fires too often during signing,
  higher if it doesn't fire when pausing.
- `stillnessFramesToCommit`: number of consecutive still frames to
  trigger commit. Lower = commits sooner, but accidental brief pauses
  inside a phrase split it in two.
- `silenceFramesToClear`: how long the hand must be absent before
  discard. Lower = quicker recovery from glitches, but a quick reach
  off-camera mid-signing wipes the phrase.

## Performance numbers

| Metric | Value |
|---|---|
| **Live letter** | |
| Frontend MediaPipe per-frame | ~10-20 ms (browser WASM) |
| Backend MLP inference | 0.019 ms |
| Per-frame bandwidth (landmarks JSON) | ~1-2 KB |
| WebSocket round-trip end-to-end | ~25 ms |
| **CTC transcribe** | |
| Backend CTC inference per 128-frame clip | 16 ms |
| Per-frame bandwidth (features+missing) | ~2 KB |
| Val CER (signer-held-out, 5 signers) | 0.249 |

## File-by-file map

### `openhand-model/` (training)

```
model/
  landmarks.py                 # 127-landmark selection + normalization (source of truth)
  ctc_transformer.py           # Conv1d + Transformer + CTC head
  fingerspelling_dataset.py    # .npz Dataset with 6 augmentation tricks
  mlp.py                       # 62K-param alphabet MLP
  dataset.py                   # Alphabet .npy splits
scripts/
  preprocess_alphabet.py       # ASL Alphabet JPEGs -> MediaPipe -> landmarks
  preprocess_fingerspelling.py # ASL Fingerspelling Parquet -> per-sequence .npz
  train.py                     # Alphabet MLP trainer
  train_ctc.py                 # CTC transformer trainer
  export_onnx.py               # Alphabet -> ONNX
  export_ctc_onnx.py           # CTC -> ONNX (dynamo path, BN-fused)
exports/
  asl_classifier.onnx          # Deployed alphabet model
  ctc/
    asl_ctc.onnx               # Deployed CTC model (best of 3 training runs)
    model_meta.json
```

### `openhand/` (application)

```
backend/
  main.py                      # FastAPI app, CORS for :5273 and :3000
  api/
    routes.py                  # /ws/detect-landmarks, /ws/transcribe-stream (new),
                               #   /api/transcribe-landmarks (legacy, may remove),
                               #   /api/tts, /api/health
  services/
    classifier.py              # SignClassifier: alphabet MLP wrapper
    ctc_classifier.py          # CTCClassifier: transformer ONNX + decode
    ctc_landmarks.py           # MIRROR of openhand-model/model/landmarks.py
    tts.py                     # ElevenLabs (async, optional)
  models/
    schemas.py                 # Pydantic: DetectedHand, DetectionResult, etc.
    artifacts/
      asl_classifier.onnx      # Alphabet MLP (250 KB)
      asl_ctc.onnx             # CTC transformer (116 MB)
      asl_ctc_meta.json
      model_meta.json
  requirements.txt             # No mediapipe, no opencv: inference-only

frontend/
  scripts/
    download_mediapipe_models.mjs   # First-time setup: fetches .task files
  public/
    models/                          # MediaPipe .task files (served statically)
      hand_landmarker.task
      pose_landmarker.task
      face_landmarker.task
  src/
    App.tsx                          # Top-level UI
    config.ts                        # API_BASE, endpoint URLs
    lib/
      landmarks.ts                   # 127-landmark selection (mirrors backend)
      mediapipe.ts                   # createDetectors, detectFrame, splitHands
      __tests__/                     # Numerical-parity harness vs Python
    hooks/
      useWebcam.ts                   # getUserMedia
      useMediaPipe.ts                # NEW: shared MediaPipe producer
      useSignDetection.ts            # Live-letter WebSocket client
      useStreamingTranscribe.ts      # NEW: continuous CTC over WS
    components/
      WebcamFeed.tsx                 # Video + skeleton overlay
      SignDisplay.tsx                # Live letter card
      TextOutput.tsx                 # Accumulated text + TTS bar
docs/
  STATE.md                           # This file
  PROGRESS.md                        # Chronological journal
  CTC_BLOG.md                        # How the CTC model was trained
  MVP_BLOG.md                        # Project overview blog
  CODE_REVIEW.md                     # Code review findings
  OVERLAY_OFFSET_BUG.md              # Debugging post-mortem
```

## How to run

```powershell
cd openhand
.\start-backend.ps1      # -> http://localhost:8273
.\start-frontend.ps1     # -> http://localhost:5273
```

Both scripts are idempotent. First run creates venv / installs deps /
downloads the MediaPipe `.task` files. Subsequent runs just launch.

## Known issues / open questions

- The CTC ONNX (116 MB) is gitignored. Copy from
  `openhand-model/exports/ctc/asl_ctc.onnx` after retraining.
- The legacy `/api/transcribe-landmarks` POST will be removed once
  streaming is verified working.
- The frontend "Show skeleton" toggle for low-end devices is the next UI
  addition.

## Watch list (next session)

- [ ] **Language model fusion in beam search**: biggest expected CER
      win for cheapest effort. Beam search exists but greedy and beam
      currently produce essentially identical CER because the model's
      softmax is too confident; an LM would change that.
- [ ] **Mirror augmentation** for left-handed signers (training-side).
- [ ] **Browser ONNX inference**: eventually skip the backend entirely
      for the alphabet model (250 KB is small enough). Bigger CTC model
      needs int8 quantization first.
- [ ] **Stable/tentative streaming decode**: right now we re-decode
      the full rolling window every 750ms and replace the displayed
      text. A two-state decoder (stable prefix that doesn't change,
      tentative suffix that may revise) would feel smoother.
