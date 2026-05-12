import asyncio
import json
from threading import Lock

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ..models.schemas import TTSRequest
from ..services.classifier import SignClassifier
from ..services.mediapipe_service import HandDetector
from ..services.tts import text_to_speech

router = APIRouter()

detector = HandDetector()
classifier = SignClassifier()

# MediaPipe HandLandmarker in IMAGE mode is not safe to call from multiple
# threads at once. Serialise calls so concurrent WebSocket clients don't race.
_detect_lock = Lock()


def _detect(data: str):
    with _detect_lock:
        return detector.process_frame(data)


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


@router.get("/api/health")
async def health():
    return {"status": "ok"}
