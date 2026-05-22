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
from ..services.tts import text_to_speech

router = APIRouter()

# Upper bounds on per-message tensor sizes
MAX_FRAMES = 512

# Alphabet classifier is small and fast to load; instantiate eagerly.
classifier = SignClassifier()

# CTC is lazy-loaded so the alphabet path can serve traffic before the
# bigger ONNX session finishes opening.
_ctc: CTCClassifier | None = None


def _get_ctc() -> CTCClassifier:
    global _ctc
    if _ctc is None:
        _ctc = CTCClassifier()
    return _ctc


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


@router.get("/api/health")
async def health():
    return {"status": "ok"}
