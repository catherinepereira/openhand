from pydantic import BaseModel
from typing import Optional


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class DetectionResult(BaseModel):
    sign: str
    confidence: float
    landmarks: list[Landmark] = []


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"


class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    error: Optional[str] = None
