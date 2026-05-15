# Code Review: Dead Code and Efficiency

> Issues found while sweeping `openhand-model/` and `openhand/` after the
> CTC integration shipped. Severity tags: **B**locking (correctness, prod
> risk), **A**ction (clear improvement, low risk), **N**it (style / minor).

Many of the "small" items below are cumulative; together they make a
meaningful difference in repo cleanliness and onboarding clarity.

---

## openhand-model

### 1. Dead one-off scripts at the repo root

- **A**: `scripts/_download_model.py` was a one-time helper to fetch the
  MediaPipe `hand_landmarker.task` for the alphabet training run. The
  same file now lives in the backend (`openhand/backend/_download_models.py`)
  and is unused here. **Delete.**

### 2. Abandoned fingerspelling-as-classification path

- **A**: `scripts/preprocess.py` is the **wrong** fingerspelling
  preprocessor: the one that produced the 30% disaster by treating
  frames as static letters. We pivoted to ASL Alphabet for the
  per-frame path and to CTC for the sequence path. This file no longer
  has a caller and its docstring is misleading. **Delete.**

### 3. Alphabet-era `processed/` outputs still on disk

- **A**: `data/processed/` (the output of the abandoned `preprocess.py`)
  exists but is unused. **Delete**; `data/processed_alphabet/` is the
  real one for the alphabet path.

### 4. `train.py` and friends: keep but rename for clarity

- **N**: `scripts/train.py`, `scripts/evaluate.py`, `scripts/infer.py`,
  `scripts/export_onnx.py` all serve the alphabet MLP path, but the
  filenames don't say so. We now also have `train_ctc.py`,
  `export_ctc_onnx.py`. Consider renaming the alphabet ones to
  `train_alphabet.py`, `evaluate_alphabet.py`, etc. **Optional.**

### 5. `model/dataset.py` only used by alphabet path

- **N**: `model/dataset.py` is the alphabet `ASLDataset` + `load_splits`.
  Only `scripts/train.py` (alphabet) imports it. Could be renamed
  `alphabet_dataset.py` to match `fingerspelling_dataset.py`. **Optional.**

### 6. Inline `import traceback` removed but `try/except` style still
       inconsistent

- **N**: `scripts/train_ctc.py:212-254` wraps the per-step training in a
  `try/except` to surface tracebacks for silent crashes; fair, given
  the Windows DataLoader crash history. But it's now defensive code for
  a problem we've fixed (the .npz preprocessing eliminated the original
  cause). Could be removed for cleaner reading. **Leave for now**:
  costs nothing, and any future weird crash will be much easier to
  diagnose with the trace.

### 7. `greedy_decode` and `char_error_rate` duplicated in train_ctc and backend

- **A**: The exact same logic exists in `scripts/train_ctc.py:36-78` and
  `openhand/backend/services/ctc_classifier.py:36-47`. The training-time
  one operates on a batched (T, B, V) tensor, the backend on a single
  (T, V); slightly different shapes but the algorithm is identical.
  Acceptable today (clean cross-repo boundary), but if we ever need
  beam search at training time too, it'd be worth pulling into a shared
  `model/decoder.py` and re-importing.

### 8. CER function isn't normalized consistently

- **N**: `char_error_rate` in `train_ctc.py:74-78` does
  `total += lev(h, r); chars += max(len(r), 1); return total / chars`.
  That's "total edits / total ref chars"; fine. But when `ref` is
  empty and `hyp` is non-empty, this returns the edit distance / 1,
  which can be >1.0 and inflate the average. Real-world impact: low
  (empty refs are dropped upstream). **Optional.**

### 9. `landmarks.py` defines `GROUP_OFFSETS` but only `fingerspelling_dataset.py` imports it

- **N**: That's fine; `GROUP_OFFSETS` IS the public API of the module.
  The backend's mirror in `services/ctc_landmarks.py` defines its own
  copy. **Leave as-is** but note the duplication: any change to the
  127-landmark selection must be made in both places.

### 10. Per-shard preprocessing prints float-precision progress

- **N**: `scripts/preprocess_fingerspelling.py` uses `tqdm` which prints
  ~150 it/s status; clean. Final size print is meaningful. No issues.

### 11. `_train_loop` was extracted, then reverted, left a comment trail

- **A**: During debugging I extracted a `_train_loop` helper into the
  script, then reverted because it broke variable scope for the meta
  JSON write. The current code is back to a single `main()`, but the
  flow is messier than necessary. The `import traceback` hoisted to
  the top is good; the wrapper try/except is the only relic. Cosmetic.

### 12. `CTC_TRAINING_STATE.md`: keep, but reaches 280 LOC

- **N**: Useful institutional knowledge but quite long. Won't actually
  block anything. Once we're stable, could fold the bug-fix history
  into `docs/PROGRESS.md` and slim this down to current-status only.

---

## openhand (backend)

### 13. `_validate_new_ctc.py` at repo root

- **A**: One-off validation script from when I deployed the new model.
  No longer needed. **Delete.**

### 14. `backend/_download_models.py`: keep, rename, document

- **A**: This downloader is genuinely useful (first-time setup needs
  three `.task` files). But it's named with a leading underscore,
  which conventionally means "private/internal." Rename to
  `scripts/download_models.py` (or just move out of the package) and
  reference it from the top-level README. **Easy win.**

### 15. Two parallel landmark/normalization implementations

- **A**: `openhand-model/model/landmarks.py` and
  `openhand/backend/services/ctc_landmarks.py` are duplicate
  implementations of the same 127-landmark spec + normalization. They
  must stay in sync or models will fail at inference. Today they're
  identical (I verified by eye). A drift would be silent and very
  hard to debug.

  Options:
  - **Vendor as one file**, copied at deploy time by a script
  - **Publish openhand-model as a pip package**, depend on it from the
    backend
  - **Document the requirement**, add a checksum / hash comparison
    test in CI

  Realistic choice for now: just **add a comment in both files**
  explicitly stating "this MUST match the other" and link them.

### 16. `routes.py` uses `np` import for one `np.stack` per request

- **N**: We import numpy just for stacking lists of arrays. Could inline
  but the cost is negligible (`numpy` is already loaded by classifier).
  **Leave as-is.**

### 17. Two locks, two detectors, possible deadlock?

- **N**: `_detect_lock` (HandDetector) and `_holistic_lock`
  (HolisticDetector) are independent. Live-letter path uses the first,
  transcribe uses the second. No nested locking. Confirmed safe.

### 18. Lazy-loaded CTC components ignore startup-time logging

- **N**: First `/api/transcribe` call takes ~2-3 seconds because the
  three MediaPipe detectors and the 116 MB ONNX all initialize at that
  moment. Frontend says "Transcribing..." but it's actually "Loading...".
  Could add an `/api/warmup` endpoint or eager-load on first import.
  **Optional, only matters for first-time UX.**

### 19. `bare except Exception` in HolisticDetector wraps each detect call

- **N**: `holistic_service.py:105-116` swallows exceptions for hands /
  pose / face individually so one weird frame doesn't kill the whole
  transcribe. Fair, but `except Exception` is the broad blanket. At
  minimum we should log the error type when it fires, so we know
  what's going wrong. **Easy improvement.**

### 20. Pose `min_pose_presence_confidence` vs others' `min_*_landmarks_confidence`

- **B**: `holistic_service.py:68`: actually look at the parameter name.
  It's `min_pose_presence_confidence`. The hand uses
  `min_hand_presence_confidence`, face uses
  `min_face_presence_confidence`. **All three are correct**; the
  MediaPipe Tasks API names them this way. False alarm; closing this.

### 21. `process_frame` decode logic duplicated in HandDetector and HolisticDetector

- **A**: Both `mediapipe_service.py` and `holistic_service.py` have
  near-identical `_decode` methods (split data URL, base64-decode,
  cv2.imdecode). Pull into a tiny `services/frame_io.py` helper.
  Saves ~10 LOC and one bug surface. **Easy refactor.**

### 22. The handedness-vs-mirroring comment is honest but suggests a latent bug

- **A**: `holistic_service.py:122-125` notes that MediaPipe reports
  handedness from the camera POV while the frontend mirrors the video.
  We "trust MediaPipe's label," which means the user's *physical*
  right hand is reported as "Left" by MediaPipe (because in the
  camera image, with no mirror, it's on the camera's left). The
  training data was generated with non-mirrored frames, so this is
  actually correct, but it's confusing and easy to break. Worth a
  paragraph in `STATE.md` or an inline assertion that checks a known
  test case. **Action item.**

### 23. Lazy `_get_ctc` isn't thread-safe under concurrent first calls

- **N**: Two simultaneous first transcribe calls could both check
  `_holistic is None`, both construct a new `HolisticDetector`. The
  loser's instance gets garbage-collected. No functional bug since
  the holistic detector has no side effects in `__init__`, but slightly
  wasteful. Easy fix: wrap `_get_ctc()` body in a lock too.
  **Optional.**

### 24. `CTCClassifier.transcribe` does a numpy `np.stack` of [x, zeros] every call

- **N**: Re-allocates a (2, T, 390) array every call to satisfy the
  ONNX dynamo batch>=2 requirement. Could pre-allocate a max-size
  buffer once and slice; saves maybe 1-2ms. **Not worth it** at our
  scale; calls are 16ms minimum, the alloc is sub-millisecond.

### 25. `tts.py` uses `httpx.AsyncClient` once per call

- **N**: New client per request means a new TCP/TLS handshake per request.
  For interactive use this is fine (a few seconds between clicks),
  but for a high-rate use we'd want a module-level client with
  connection reuse. **Optional, only if TTS gets heavy traffic.**

---

## openhand (frontend)

### 26. Hardcoded backend URL in two places

- **A**: `useTranscribe.ts:3` has `http://localhost:8273/api/transcribe`,
  `useSignDetection.ts:8` has `ws://localhost:8273/ws/detect`,
  `App.tsx:10` has `http://localhost:8273/api/tts`. Three places to
  update when the port changes (and we've changed the port once
  already). Move to a shared `src/config.ts` or read from
  `import.meta.env.VITE_API_BASE`. **Easy win.**

### 27. `useSignDetection` retries connect-on-each-tick instead of with backoff

- **N**: If the WebSocket fails, the 100ms interval keeps trying to
  reconnect at full rate. For a healthy backend this is fine; for a
  backend that's down it spams the network. Exponential backoff would
  be nicer. **Optional; current behavior is acceptable.**

### 28. `useTranscribe` ignores `videoRef` changes after first capture

- **N**: `captureFrame` is `useCallback([videoRef])`, so if the ref
  changes the callback updates; but it only matters at the moment of
  re-render, and the recording interval has already captured the
  callback. In practice this is fine because `videoRef` is stable
  across the component lifetime. **No action.**

### 29. `App.tsx` debounces sign output but only on `result.sign` changes

- **N**: The debounce logic in `App.tsx:25-34` looks correct, but if
  the user holds the same letter, only the first sighting gets added.
  For deliberate "AA" or "BB" or "00" patterns this is wrong. Probably
  intentional ("don't spam the same letter") but worth a comment that
  says so. **Minor.**

### 30. `useTranscribe.captureFrame` creates a new canvas context per call

- **N**: `getContext("2d")` is cached internally by the browser, so this
  is cheap. **No action.**

---

## Suggested action plan, prioritized

The biggest cleanup wins, in order:

1. **Delete dead code** (~10 minutes total):
   - `openhand-model/scripts/_download_model.py`
   - `openhand-model/scripts/preprocess.py`
   - `openhand-model/data/processed/` directory
   - `openhand/_validate_new_ctc.py`

2. **Centralize the backend URL** in one place in the frontend (~5 min).

3. **Deduplicate the frame decode** between `HandDetector` and
   `HolisticDetector` (~10 min).

4. **Rename `_download_models.py`** to drop the leading underscore, mention
   it in the README (~5 min).

5. **Add cross-file linkage comments** in `model/landmarks.py` and
   `services/ctc_landmarks.py` pointing at each other and noting the
   contract that they must stay in sync (~5 min).

Total: ~35 minutes of cleanup for a noticeably tighter repo.

The rest are either optional polish or intentional behavior worth
documenting rather than changing.
