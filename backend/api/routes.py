import asyncio
import json
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ..models.schemas import (
    DetectionResult,
    DetectLandmarksRequest,
    TTSRequest,
)
from ..services.classifier import SignClassifier
from ..services.ctc_classifier import CTCClassifier
from ..services.ctc_landmarks import N_FEATURES, N_LANDMARKS
from ..services.sign_classifier import SignClassifier as WordClassifier
from ..services.signs_landmarks import N_LANDMARKS as SIGNS_N_LANDMARKS
from ..services.tts import text_to_speech

router = APIRouter()

# Upper bounds on per-message tensor sizes
MAX_FRAMES = 512

# Alphabet classifier is small and fast to load; instantiate eagerly.
classifier = SignClassifier()

# Lazy-load these
_ctc: CTCClassifier | None = None
_words: WordClassifier | None = None


def _get_ctc() -> CTCClassifier:
    global _ctc
    if _ctc is None:
        _ctc = CTCClassifier()
    return _ctc


def _get_words() -> WordClassifier:
    global _words
    if _words is None:
        _words = WordClassifier()
    return _words


@router.websocket("/ws/detect-landmarks")
async def detect_landmarks_ws(websocket: WebSocket):
    """Per-frame letter detection from pre-extracted hand landmarks.

    Browser runs MediaPipe locally and sends one DetectLandmarksRequest per
    frame; backend runs the alphabet MLP against the camera-POV "Right" hand
    and returns a DetectionResult.
    """
    await websocket.accept()
    empty = DetectionResult(sign="-", confidence=0.0, hands=[]).model_dump_json()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                req = DetectLandmarksRequest.model_validate(payload)
            except Exception:
                await websocket.send_text(empty)
                continue
            result = classifier.classify(req.hands)
            await websocket.send_text(result.model_dump_json())
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/transcribe-stream")
async def transcribe_stream_ws(websocket: WebSocket):
    """Streaming CTC transcription.

    The browser sends decode messages containing a rolling buffer of landmark
    frames every ~750ms; the route runs the CTC model on each batch and returns
    the decoded string.

    Wire format (both directions JSON):

      client -> server:  {
        "type": "decode",
        "frame_count": int,
        "features": float[],  // flat, T * N_FEATURES, row-major
        "missing":  bool[],   // flat, T * N_LANDMARKS, row-major
      }

      server -> client:  {"type": "result", "text": str, "elapsed_ms": float}
                         {"type": "error",  "message": str}

    The model is stateless across messages; each request re-decodes the full
    provided buffer.
    """
    await websocket.accept()
    try:
        ctc = _get_ctc()
    except Exception as exc:
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"CTC model load failed: {exc}"})
        )
        await websocket.close()
        return

    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "decode":
                continue
            try:
                T = int(payload["frame_count"])
                features = payload["features"]
                missing = payload["missing"]
            except (KeyError, TypeError, ValueError):
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "bad payload"})
                )
                continue
            if T < 0 or T > MAX_FRAMES:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"frame_count out of range [0, {MAX_FRAMES}]"})
                )
                continue
            if T == 0:
                await websocket.send_text(json.dumps({"type": "result", "text": "", "elapsed_ms": 0.0}))
                continue
            if len(features) != T * N_FEATURES or len(missing) != T * N_LANDMARKS:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": (
                                f"length mismatch: expected features={T * N_FEATURES} "
                                f"missing={T * N_LANDMARKS}, got "
                                f"features={len(features)} missing={len(missing)}"
                            ),
                        }
                    )
                )
                continue

            arr = np.array(features, dtype=np.float32).reshape(T, N_FEATURES)
            miss = np.array(missing, dtype=bool).reshape(T, N_LANDMARKS)

            t0 = time.perf_counter()
            text = await asyncio.to_thread(ctc.transcribe, arr, miss)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            await websocket.send_text(
                json.dumps({"type": "result", "text": text, "elapsed_ms": elapsed_ms})
            )
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/classify-sign")
async def classify_sign_ws(websocket: WebSocket):
    """Isolated-sign classification (Google ISLR / "Words" path).

    The browser sends classify messages with a rolling buffer of
    127-landmark frames (+ missing mask). The route runs the trained
    transformer on the clip and returns the top-K (sign, prob) pairs.

    Wire format:
      client -> server: {
        "type": "classify",
        "frame_count": int,
        "landmarks": float[],   // flat T * 127 * 3, row-major
        "missing":   bool[],    // flat T * 127, row-major
        "top_k":     int        // optional, default 5
      }
      server -> client:
        {"type": "result", "predictions": [["airplane", 0.81], ["alligator", 0.05], ...], "elapsed_ms": float}
        {"type": "error",  "message": str}
    """
    await websocket.accept()
    try:
        words = _get_words()
    except Exception as exc:
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Sign model load failed: {exc}"})
        )
        await websocket.close()
        return

    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") != "classify":
                continue
            try:
                T = int(payload["frame_count"])
                landmarks = payload["landmarks"]
                missing = payload["missing"]
                top_k = int(payload.get("top_k", 5))
            except (KeyError, TypeError, ValueError):
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "bad payload"})
                )
                continue
            if T < 0 or T > MAX_FRAMES:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"frame_count out of range [0, {MAX_FRAMES}]"})
                )
                continue
            if not (1 <= top_k <= 250):
                top_k = 5
            if T == 0:
                await websocket.send_text(
                    json.dumps({"type": "result", "predictions": [], "elapsed_ms": 0.0})
                )
                continue
            expected_lm = T * SIGNS_N_LANDMARKS * 3
            expected_miss = T * SIGNS_N_LANDMARKS
            if len(landmarks) != expected_lm or len(missing) != expected_miss:
                await websocket.send_text(
                    json.dumps({
                        "type": "error",
                        "message": (
                            f"length mismatch: expected landmarks={expected_lm} "
                            f"missing={expected_miss}, got "
                            f"landmarks={len(landmarks)} missing={len(missing)}"
                        ),
                    })
                )
                continue

            arr = np.array(landmarks, dtype=np.float32).reshape(T, SIGNS_N_LANDMARKS, 3)
            miss = np.array(missing, dtype=bool).reshape(T, SIGNS_N_LANDMARKS)

            t0 = time.perf_counter()
            predictions = await asyncio.to_thread(words.classify, arr, miss, top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            await websocket.send_text(
                json.dumps({
                    "type": "result",
                    "predictions": predictions,
                    "elapsed_ms": elapsed_ms,
                })
            )
    except WebSocketDisconnect:
        pass


@router.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    audio = await text_to_speech(req.text, req.voice_id or "21m00Tcm4TlvDq8ikWAM")
    if audio is None:
        return {"error": "ElevenLabs API key not configured or request failed"}
    return Response(content=audio, media_type="audio/mpeg")


_REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "artifacts" / "reference_landmarks.json"
)
_reference_cache: dict | None = None


@router.get("/api/reference-landmarks")
async def reference_landmarks():
    """Return the per-letter mean landmark vectors used by the Learn screen.

    Payload shape:
      {
        "format": str,
        "n_features": 63,
        "letters": { "A": [x0, y0, z0, ..., x20, y20, z20], ... },
        "sample_counts": { "A": int, ... }
      }
    """
    global _reference_cache
    if _reference_cache is None:
        if not _REFERENCE_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"reference_landmarks.json not found at {_REFERENCE_PATH}. "
                    "Generate it from openhand-model via "
                    "scripts/build_reference_landmarks.py."
                ),
            )
        with open(_REFERENCE_PATH) as f:
            _reference_cache = json.load(f)
    return _reference_cache


_SIGN_REFERENCES_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "artifacts" / "sign_references.json"
)
_sign_refs_cache: dict | None = None


@router.get("/api/sign-references")
async def sign_references():
    """Return the per-sign medoid-clip landmark sequences for the
    Learn-the-words 3D animated reference.

    Payload:
      {
        "n_landmarks": 127,
        "n_coords": 3,
        "sign_to_idx": {...},
        "signs": {
          "airplane": {
            "label": 0,
            "n_frames": 47,
            "landmarks": [...flat T*127*3...],
            "missing":   [...flat T*127...]
          },
          ...
        }
      }
    """
    global _sign_refs_cache
    if _sign_refs_cache is None:
        if not _SIGN_REFERENCES_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"sign_references.json not found at {_SIGN_REFERENCES_PATH}. "
                    "Generate via openhand-model/signs/scripts/build_sign_references.py."
                ),
            )
        with open(_SIGN_REFERENCES_PATH) as f:
            _sign_refs_cache = json.load(f)
    return _sign_refs_cache


@router.get("/api/health")
async def health():
    return {"status": "ok"}
