from pydantic import BaseModel, Field


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class DetectedHand(BaseModel):
    """A single detected hand with its MediaPipe handedness label.

    handedness is the camera-POV label: "Left" / "Right" / "" when the
    detector couldn't classify with confidence.
    """
    handedness: str
    landmarks: list[Landmark]


class DetectionResult(BaseModel):
    sign: str
    confidence: float
    hands: list[DetectedHand] = []


class TTSRequest(BaseModel):
    # Length cap stops a caller from burning ElevenLabs quota on huge inputs
    text: str = Field(..., min_length=1, max_length=500)
    voice_id: str | None = Field("21m00Tcm4TlvDq8ikWAM", max_length=64)


class DetectLandmarksRequest(BaseModel):
    """Pre-extracted hand frames for the live-letter MLP.

    Sent over the browser-side WebSocket at one message per video frame.
    hands mirrors DetectionResult.hands: one entry per detected hand with
    the MediaPipe handedness label so the classifier can pick "Right".
    """
    hands: list[DetectedHand]
