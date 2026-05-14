from pydantic import BaseModel
from typing import Optional


class Landmark(BaseModel):
    x: float
    y: float
    z: float


class DetectedHand(BaseModel):
    """A single detected hand with its MediaPipe handedness label."""
    # MediaPipe label, from the camera's POV: "Left" / "Right" / "" if
    # the detector couldn't classify with confidence.
    handedness: str
    landmarks: list[Landmark]


class DetectionResult(BaseModel):
    sign: str
    confidence: float
    # All detected hands in the frame. Empty list when no hand present.
    hands: list[DetectedHand] = []


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"


class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    error: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    frame_count: int
    elapsed_ms: float


class TranscribeLandmarksRequest(BaseModel):
    """Pre-extracted landmark sequence ready for the CTC model.

    ``features`` is a (T, N_FEATURES) float matrix flattened in row-major
    order — i.e. ``len(features) == T * N_FEATURES``.

    ``missing`` is the (T, N_LANDMARKS) bool mask, flattened the same way.

    Both arrays must have lengths consistent with ``frame_count`` and the
    feature dimensions baked into the model (see backend
    ``ctc_landmarks.py``); mismatched lengths return 422.
    """
    frame_count: int
    features: list[float]
    missing: list[bool]


class DetectLandmarksRequest(BaseModel):
    """Pre-extracted hand frames for the live-letter MLP.

    Sent over the new browser-side WebSocket route at one message per
    video frame. ``hands`` is the same shape ``DetectionResult.hands``
    uses for the response — one entry per detected hand, with the
    MediaPipe handedness label so the classifier can pick "Right".
    """
    hands: list[DetectedHand]
