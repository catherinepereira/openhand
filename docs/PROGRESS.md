# OpenHand — Progress Log

> A retrospective of the work so far, in roughly chronological order.
> Companion to `STATE.md` (current state) and the per-repo READMEs (how to
> run things today).

## The arc

We started with a rule-based ASL letter classifier covering 7 letters. We
ended with two real ML models in production:

1. **A lightweight per-frame MLP** trained on the ASL Alphabet image dataset
   (62K samples after MediaPipe filtering), serving the live "letter as
   you sign" path at **0.019 ms/frame on CPU**, **98.95% test accuracy** on
   the held-out alphabet split.
2. **A 27.5M-param Transformer + CTC** trained on the Kaggle ASL
   Fingerspelling dataset (67K sequences, 100+ signers), serving the
   "press-to-transcribe-a-phrase" path at **~16 ms inference per 128-frame
   sequence**, **CER 0.249** on signer-held-out validation.

The journey from rule-based to two ML models involved one big detour, a few
near-disasters, and roughly a dozen iterations on the training pipeline.

## Phase 1 — Live alphabet classifier

### Initial premise (wrong)

The original plan was the Kaggle ASL Fingerspelling dataset — 67K
sequences, 100+ signers, MediaPipe Holistic landmarks pre-extracted into
Parquet files. The intuition: this is the "real" benchmark dataset,
should give us the best model.

We downloaded the full 160 GB of Parquet data, wrote a preprocessor that
chopped each sequence into per-character frames by dividing frames evenly
across the phrase's characters, then trained a small MLP on the resulting
"static letter" samples.

**Result: 30% test accuracy.** Garbage in, garbage out — the dataset has no
per-frame alignments. Dividing frames evenly across characters labels
mostly-transition frames with arbitrary letters. The model learned what
amounted to noise.

### The pivot

Realised mid-debug that the fingerspelling dataset is designed for
**sequence-to-sequence** transcription (CTC), not per-frame classification.
The dataset's labels are entire phrases, not per-frame letters.

Switched to the **ASL Alphabet dataset** (Kaggle, `grassknoted/asl-alphabet`):
87K labelled per-letter images of one signer, A–Z + DEL/NOTHING/SPACE.

### MediaPipe Tasks API migration

The backend was importing `mediapipe.solutions.hands` — that legacy API was
removed in MediaPipe 0.10.20+. Switched to the new
`mediapipe.tasks.python.vision.HandLandmarker`, which requires loading a
`.task` model file (8 MB). Same 21 landmarks per hand, just a different
API surface.

### The pipeline that worked

For each of the 87K alphabet images:
1. Run MediaPipe Hands → 21 landmarks (63 floats: x, y, z per landmark)
2. Drop image if no hand detected (15K of 87K — usually awkward angles)
3. Wrist-centre + 95th-percentile-scale the landmark vector
4. Save as a single `X.npy` / `y.npy` pair: **62,819 samples × 63 floats**

Then a 4-layer MLP on top: `[63 → 256 → 128 → 64 → 26]` with BatchNorm +
Dropout. 62K params total.

### Results

- 60 epochs, batch 512, AdamW lr=1e-3, cosine LR
- **Test accuracy: 98.95%** (held-out 5% split)
- Trained in 5–10 minutes on an RTX 4070 Super
- ONNX exported, 250 KB on disk
- Deployed to backend at `openhand/backend/models/artifacts/asl_classifier.onnx`

### Honest assessment

The 98.95% is real but narrow. The training data is **one signer**,
**consistent lighting**, **consistent camera angle**. Real-world accuracy
on diverse users will be lower — and the model has no way to handle J
or Z (which require motion).

## Phase 2 — Backend integration

### Replacing the rule-based classifier

The backend's `SignClassifier` was a stack of `if`-statements over landmark
geometry (finger extension, thumb position, etc.) covering 7 letters.
Replaced with an ONNX-loading classifier that runs the trained MLP, returns
A–Z + confidence. Below 0.5 confidence the result is `"—"` to avoid bad
guesses.

### MediaPipe upgrade

While we were in `mediapipe_service.py` anyway, migrated from the legacy
`mp.solutions.hands` to the new Tasks API. This was required by our
training pipeline using mediapipe 0.10.21, and meant the backend now
needed a `.task` model file shipped alongside the code.

### Thread safety

MediaPipe `HandLandmarker` in IMAGE mode is **not thread-safe** —
concurrent calls from multiple WebSocket clients would race and corrupt
internal state. Added a `threading.Lock` around `detector.detect()` in
`routes.py`. WebSocket clients are still handled concurrently by FastAPI,
but the per-frame detect call is single-flighted.

### Live testing — and the frontend bug

First end-to-end test: camera worked, but predictions never updated.
Backend logs showed WebSocket connections opening and immediately closing
with 0 frames received.

Root cause: React 18 StrictMode in dev double-invokes effects, which
interacted badly with the WebSocket lifecycle in `useSignDetection`. The
hook checked for `OPEN` state before opening a new WebSocket, but
ignored the `CONNECTING` state — so StrictMode's double-mount opened two
WebSockets, both pointed at the same ref. The second one overwrote the
first, the original was orphaned, and the cleanup function didn't close
WebSockets properly.

Fix: check both `OPEN` and `CONNECTING` in the connect guard, and close
the WebSocket in the cleanup function. Worked first try after that.

### Cosmetic improvements

- Made the camera view a rounded rectangle (~33% of viewport) instead of
  a small circle
- Added landmark overlay — drew the 21-joint skeleton on a canvas on top
  of the video feed, mirrored to match the user's actual hand. This both
  looks great and confirms visually that the model is "seeing" what we
  think it's seeing.

## Phase 3 — Returning to fingerspelling, doing it right

### Why come back

The alphabet model worked, but with caveats:
- **One signer in training** → poor generalisation
- **Static frames** → can't handle J/Z motion
- **Per-letter only** → can't transcribe phrases

The fingerspelling dataset has all three: 100+ signers, multi-frame
sequences (so motion is captured), and phrase-level transcription
implicitly trains the model on letter context.

Just needed to use it correctly this time — **sequence-to-sequence
training with CTC loss**.

### The plan

| Component | Choice | Why |
|---|---|---|
| Landmarks | 127-of-543 (lips, eyes, nose, pose, both hands) | Mirrors Kaggle 1st-place approach. 381 features/frame |
| Architecture | Conv1d stem → Transformer encoder → CTC head | Standard; Kaggle winners used same family |
| Loss | `nn.CTCLoss` with `blank` at last vocab index | Standard |
| Decoder | Greedy with repeat collapse | Beam search added later |
| Augmentation | Time crop + frame masking + (later) much more | Per Kaggle winners' notes |

### The first training disaster: 30% accuracy

Initial config: 5.5M-param transformer, full 67K dataset, 30 epochs.
Trained for ~50 minutes. Final val CER: **0.94** (essentially nothing).

The model had collapsed into the well-known **CTC blank-collapse** local
minimum: predicting "blank" for every frame is a low-loss attractor early
in training, and the model never escapes it.

**Fixes that mattered:**
1. **Linear warmup over 500-1500 steps** — prevents the optimizer from
   slamming into the blank-collapse basin during the first noisy updates
2. **KL-to-uniform label smoothing** (weight 0.1) — penalises peaky
   "always blank" outputs, nudging the model toward real character
   emissions
3. **Cosine LR decay after warmup** — standard for transformers

The smoke test confirmed these fixes worked: first-epoch hyps changed from
`""` to actual characters (e.g. `"e"`, `"eee"`) within the first epoch.

### The second disaster: slow as hell

After fixing blank-collapse, training was running at ~57 minutes per
epoch — 20 epochs would have taken 19 hours.

**Root cause**: the original Dataset class loaded each Parquet shard
(1.4 GB) from scratch every time a sample from that shard was needed.
With shuffled DataLoader, that meant 16-24 Parquet decodes per training
step. The "one shard cached at a time" LRU was thrashing constantly.

**Fix**: pre-extract every sequence to its own small `.npz` file once
(`scripts/preprocess_fingerspelling.py`). Each file is ~165 KB, contains
the (T, 390) landmark tensor plus an explicit missing-data mask, plus
the encoded target. Saved 11 GB total.

After this fix:
- **Epoch time dropped from 57 min → 17 seconds** (~200× speedup)
- The bottleneck moved from "decode Parquet" to "actual training compute"

### A subtler correctness fix

While auditing the data pipeline I caught a bug in `normalize_sequence`:
it was treating zero-valued landmarks as "missing." But MediaPipe normalises
landmarks to [0,1], so x=0 is a real value ("left edge of frame"), not
absence. The function was conflating real near-origin landmarks with
missing data, and after wrist-centring (which writes zeros), legitimate
landmarks were getting re-masked as missing.

Fixed by saving an explicit `missing` boolean mask alongside the features
in each .npz, and rewriting `normalize_sequence` to consume the mask
directly. Probably worth ~2-3 CER points.

### Training run history

| Run | Architecture | Aug | Epochs | Val CER | Notes |
|---|---|---|---|---|---|
| Subset smoke | 5.5M (d=192, 4L) | mild | 8 | 1.00 | Pipeline check on 2K seq |
| Subset proper | 5.5M (d=192, 4L) | mild | 20 | **0.516** | Validated architecture |
| Full 1 | 5.5M (d=256, 6L) | mild | 40 | **0.274** | First "real" model |
| Full 2 | 27.5M (d=512, 12L) | mild | 100 | **0.248** | Bigger model, marginal gain |
| Full 3 | 27.5M (d=512, 12L) | strong | 80 | **0.249** | Strong aug, no further gain |

**Cohort discovered**: by run 3 the augmentation closed the train/val gap
(train loss 1.31 vs val 0.99) but val CER plateaued at the same place. We
hit a different ceiling — likely data-limited or decoder-limited rather
than model-limited.

### Strong augmentation recipe

What we settled on for the data pipeline:
1. **Random time crop** (70-100% of frames) — robustness to clip boundaries
2. **Time stretch** (0.85-1.15× speed via frame resampling) — different
   signing speeds
3. **Landmark group dropout** (8% chance to zero entire face/pose/hand) —
   handles detector failures in production
4. **Affine jitter** (scale ±10%, translate ±0.05) — camera-distance
   variation
5. **Per-frame masking** (5% zero-out) — local noise robustness
6. **Contiguous time-mask spans** (1-2 spans, up to 10% of T) — classic
   SpecAugment

## Phase 4 — Deploying the CTC model

### ONNX export

PyTorch `torch.onnx.export` (the legacy tracer) baked the dummy time
dimension as a constant — exported models worked at T=64 but failed at
any other T with `Reshape` errors from `MultiheadAttention`'s internal
permute logic.

Fixed by switching to the new `torch.onnx.dynamo_export` (requires
`onnxscript`). The dynamo path correctly preserves time and batch as
dynamic axes. Also had to **fuse BatchNorm into the preceding Conv1d**
because the dynamo exporter currently can't convert BatchNorm in
eval/no-training mode.

One more nit: dynamo-exported ONNX requires batch ≥ 2 at runtime
(otherwise the batch axis becomes a constant). The classifier pads with
an all-zero, fully-masked second item it ignores.

### Backend integration

Added three new components:
- `services/ctc_landmarks.py` — mirrors the training repo's landmark
  selection + normalisation
- `services/holistic_service.py` — runs three MediaPipe Tasks detectors
  (Hand + Pose + Face) in series, produces the 381-feature vector
- `services/ctc_classifier.py` — ONNX inference + CTC greedy/beam decode

And a new route: `POST /api/transcribe` accepts `{frames: [base64...]}`,
returns `{text, frame_count, elapsed_ms}`.

### Why not MediaPipe Holistic

We initially tried the all-in-one `HolisticLandmarker` (the new Tasks API
version). It crashed the entire backend process with a fatal C++ assertion
(`packet is empty`) when an internal sub-task failed on a particular
frame — no traceback, just a dead process.

Replaced it with three separate Tasks-API detectors (`HandLandmarker`,
`PoseLandmarker`, `FaceLandmarker`) running in series. Slower per frame
(~15-25 ms total) but stable.

### Frontend integration

Added a mode toggle: "Live letter" (existing per-frame MLP) vs "Phrase
transcribe" (new CTC). The transcribe button is hold-to-record: while
held, captures frames into a rolling buffer at 100ms intervals; on
release, posts the buffer to `/api/transcribe` and appends the response
to the output text. Visual states: idle / recording (red pulsing) /
transcribing (busy).

The live-letter mode was kept entirely untouched — both paths coexist.

## Phase 5 — Decoder upgrade

### Beam search

Added CTC prefix beam search (`_beam_search_decode` in `ctc_classifier.py`)
with the standard two-state bookkeeping (blank-ending vs non-blank-ending
probabilities per prefix). Beam width 10, beam pruning to top-K candidate
symbols per frame.

**Result**: CER 0.237 (greedy) → 0.235 (beam) on a 40-sample val eval.
Essentially noise.

**Diagnosis**: The trained model's softmax is too confident for beam to
help — at each frame the argmax probability is typically >0.95, so all
beams collapse to the same prefix. Beam search dominates greedy when the
model is *uncertain*; with label smoothing + a well-trained 27.5M-param
model, our outputs aren't uncertain enough.

The decoder is now there for when it actually matters — e.g., if we add a
language model on top of beam search, that would compound much more
strongly than it does over greedy alone.

## What's deployed today

**Live letter path** (per-frame MLP, fast):
- `backend/models/artifacts/asl_classifier.onnx` (250 KB, 26-class A–Z)
- `backend/models/artifacts/hand_landmarker.task` (8 MB, MediaPipe)
- Routed via WebSocket `/ws/detect`, ~0.02 ms inference per frame

**Phrase transcribe path** (CTC transformer, accurate):
- `backend/models/artifacts/asl_ctc.onnx` (116 MB, 60-class incl. blank)
- `backend/models/artifacts/asl_ctc_meta.json` (vocab, training history)
- `backend/models/artifacts/{hand,pose,face}_landmarker.task` (17 MB total)
- Routed via POST `/api/transcribe`, ~50-100 ms inference for a 5-second phrase

Both paths are exposed in the frontend via a mode toggle.

## What we tried that didn't work

| Attempt | Outcome |
|---|---|
| Per-frame labels from fingerspelling phrases via even frame division | 30% accuracy — fundamentally broken approach |
| Bigger model (5.5M → 27.5M params) | CER 0.274 → 0.248. Small gain, overfit at end |
| Even bigger model (d=384, 8L attempt) | Killed early on perceived slowness — real steady-state was faster than wall-clock suggested |
| MediaPipe `HolisticLandmarker` (all-in-one) | Crashed backend with C++ assertion. Replaced with 3 separate detectors |
| Beam search vs greedy | ~0.002 CER reduction. Model too confident for beam to matter |
| Strong augmentation (6 tricks) | Closed train/val gap but CER same. Hit a different ceiling |

## Honest reflection on what dominates real accuracy

Looking at the runs in aggregate, here's what actually moved the needle:

1. **Fixing the broken data pipeline** (per-Parquet re-reads → per-sequence .npz):
   200× speed, no CER change but enabled much longer experiments
2. **Fixing the missing-data sentinel** (zero → explicit mask): probably
   2-3 CER points, hard to isolate
3. **Anti-blank-collapse tricks** (warmup + label smoothing): turned a
   non-functional model into a working one (CER 0.94 → 0.27)
4. **More data** (16K → 67K sequences): CER 0.52 → 0.27 (cleanest signal)
5. **More epochs and slightly bigger model**: 0.27 → 0.25 (modest)
6. **Stronger augmentation**: no CER gain but closed train/val gap
7. **Beam search**: negligible

The lesson: **data pipeline correctness and anti-blank-collapse > model
size > augmentation > decoder**. Most of our actual progress was in the
boring stuff at the bottom of the stack.
