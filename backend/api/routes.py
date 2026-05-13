import asyncio
import json
import time
from threading import Lock

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ..models.schemas import TTSRequest, TranscribeRequest, TranscribeResponse
from ..services.classifier import SignClassifier
from ..services.ctc_classifier import CTCClassifier
from ..services.holistic_service import HolisticDetector
from ..services.mediapipe_service import HandDetector
from ..services.tts import text_to_speech

router = APIRouter()

detector = HandDetector()
classifier = SignClassifier()

# Heavy CTC components are lazy-loaded on first transcribe call so the live
# alphabet flow starts up fast even if these aren't needed yet.
_holistic: HolisticDetector | None = None
_ctc: CTCClassifier | None = None

# MediaPipe HandLandmarker in IMAGE mode is not safe to call from multiple
# threads at once. Serialise calls so concurrent WebSocket clients don't race.
_detect_lock = Lock()
_holistic_lock = Lock()


def _detect(data: str):
    with _detect_lock:
        return detector.process_frame(data)


def _get_ctc() -> tuple[HolisticDetector, CTCClassifier]:
    global _holistic, _ctc
    if _holistic is None:
        _holistic = HolisticDetector()
    if _ctc is None:
        _ctc = CTCClassifier()
    return _holistic, _ctc


def _run_transcribe(frames: list[str]) -> tuple[str, int]:
    holistic, ctc = _get_ctc()
    feats, masks = [], []
    with _holistic_lock:
        for f in frames:
            r = holistic.process_frame(f)
            if r is None:
                continue
            vec, miss = r
            feats.append(vec)
            masks.append(miss)
    if not feats:
        return "", 0
    arr = np.stack(feats, axis=0)
    miss = np.stack(masks, axis=0)
    return ctc.transcribe(arr, miss), len(feats)


@router.websocket("/ws/detect")
async def detect_ws(websocket: WebSocket):
    """Receive base64 frames, return detection results as JSON."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            landmarks, _ = await asyncio.to_thread(_detect, data)
            result = classifier.classify(landmarks)
            if result:
                await websocket.send_text(result.model_dump_json())
            else:
                await websocket.send_text(
                    json.dumps({"sign": "—", "confidence": 0.0, "landmarks": []})
                )
    except WebSocketDisconnect:
        pass


@router.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    audio = await text_to_speech(req.text, req.voice_id or "21m00Tcm4TlvDq8ikWAM")
    if audio is None:
        return {"error": "ElevenLabs API key not configured or request failed"}
    return Response(content=audio, media_type="audio/mpeg")


@router.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(req: TranscribeRequest):
    """Transcribe a short clip of fingerspelling.

    Accepts up to ~10 seconds of webcam frames at ~10 fps (~100 frames).
    Runs each through MediaPipe Holistic, builds the (T, 390) landmark
    sequence, and runs the CTC transformer to produce a string."""
    if not req.frames:
        return TranscribeResponse(text="", frame_count=0, elapsed_ms=0.0)
    t0 = time.perf_counter()
    text, frame_count = await asyncio.to_thread(_run_transcribe, req.frames)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return TranscribeResponse(text=text, frame_count=frame_count, elapsed_ms=elapsed_ms)


@router.get("/api/health")
async def health():
    return {"status": "ok"}
