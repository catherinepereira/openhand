import asyncio
import time

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ..models.schemas import (
    DetectionResult,
    DetectLandmarksRequest,
    TTSRequest,
    TranscribeLandmarksRequest,
    TranscribeResponse,
)
from ..services.classifier import SignClassifier
from ..services.ctc_classifier import CTCClassifier
from ..services.ctc_landmarks import N_FEATURES, N_LANDMARKS
from ..services.tts import text_to_speech

router = APIRouter()

# Alphabet classifier is small + fast to load; instantiate eagerly so the
# first WebSocket frame doesn't pay startup cost.
classifier = SignClassifier()

# CTC classifier ONNX is 116 MB; lazy-load on first transcribe call so
# the live-letter flow starts up fast even if nobody uses transcribe.
_ctc: CTCClassifier | None = None


def _get_ctc() -> CTCClassifier:
    global _ctc
    if _ctc is None:
        _ctc = CTCClassifier()
    return _ctc


# ─── live-letter (browser MediaPipe → backend ONNX MLP) ─────────────────

@router.websocket("/ws/detect-landmarks")
async def detect_landmarks_ws(websocket: WebSocket):
    """Per-frame letter detection from pre-extracted hand landmarks.

    Browser runs MediaPipe locally and sends one ``DetectLandmarksRequest``
    per frame; the backend runs the alphabet MLP against the "Right" hand
    (camera POV) and returns a ``DetectionResult``.
    """
    await websocket.accept()
    empty = DetectionResult(sign="—", confidence=0.0, hands=[]).model_dump_json()
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


# ─── phrase transcription (browser MediaPipe → backend CTC ONNX) ────────

@router.post("/api/transcribe-landmarks", response_model=TranscribeResponse)
async def transcribe_landmarks_endpoint(req: TranscribeLandmarksRequest):
    """Transcribe a fingerspelling clip whose landmarks were extracted by
    MediaPipe in the browser. The body shape matches the CTC model's
    input directly — no MediaPipe runs server-side.
    """
    T = req.frame_count
    if T == 0:
        return TranscribeResponse(text="", frame_count=0, elapsed_ms=0.0)
    if len(req.features) != T * N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"features length {len(req.features)} != frame_count {T} * {N_FEATURES}",
        )
    if len(req.missing) != T * N_LANDMARKS:
        raise HTTPException(
            status_code=422,
            detail=f"missing length {len(req.missing)} != frame_count {T} * {N_LANDMARKS}",
        )

    features = np.array(req.features, dtype=np.float32).reshape(T, N_FEATURES)
    missing = np.array(req.missing, dtype=bool).reshape(T, N_LANDMARKS)

    ctc = _get_ctc()
    t0 = time.perf_counter()
    text = await asyncio.to_thread(ctc.transcribe, features, missing)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return TranscribeResponse(text=text, frame_count=T, elapsed_ms=elapsed_ms)


# ─── misc ────────────────────────────────────────────────────────────────

@router.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    audio = await text_to_speech(req.text, req.voice_id or "21m00Tcm4TlvDq8ikWAM")
    if audio is None:
        return {"error": "ElevenLabs API key not configured or request failed"}
    return Response(content=audio, media_type="audio/mpeg")


@router.get("/api/health")
async def health():
    return {"status": "ok"}
