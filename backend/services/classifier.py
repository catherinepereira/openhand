"""
ASL letter classifier backed by an ONNX MLP trained in openhand-model/.

Loads the model once at import time. Consumes the 21 MediaPipe hand landmarks
produced by the browser detector, normalizes them the same way training did
(wrist at origin, scaled by 95th-percentile |value|), and returns the
predicted letter and confidence.

To retrain or update the model, regenerate exports/asl_classifier.onnx in
openhand-model/ and copy it (along with model_meta.json) into
backend/models/artifacts/.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..models.schemas import DetectedHand, DetectionResult, Landmark

_ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
_MODEL_PATH = _ARTIFACTS / "asl_classifier.onnx"
_META_PATH = _ARTIFACTS / "model_meta.json"

N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3

# Below this confidence we return the empty sentinel rather than guessing.
MIN_CONFIDENCE = 0.5
NORM_PERCENTILE = 95
NORM_EPS = 1e-6


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Match openhand-model/alphabet/scripts/preprocess_alphabet.py::normalize.

    Subtracts the wrist (landmark 0) so the hand is anchored at the origin,
    then scales so the 95th-percentile absolute coordinate is 1.
    """
    pts = vec.reshape(N_LANDMARKS, 3) - vec[:3]
    scale = np.percentile(np.abs(pts), NORM_PERCENTILE)
    if scale > NORM_EPS:
        pts = pts / scale
    return pts.reshape(N_FEATURES).astype(np.float32)


class SignClassifier:
    def __init__(self):
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ASL classifier ONNX not found at {_MODEL_PATH}. "
                "Train it via ../openhand-model and copy the exports."
            )
        with open(_META_PATH) as f:
            meta = json.load(f)
        self.label_map: dict[str, str] = meta["label_map"]
        self.session = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, hands: list[DetectedHand]) -> DetectionResult:
        """Classify a frame's hands into a single letter prediction.

        Runs the MLP against the camera-POV "Right" hand. The returned hands
        field echoes every detected hand so the frontend can render overlays.
        """
        if not hands:
            return DetectionResult(sign="-", confidence=0.0, hands=[])

        target: DetectedHand | None = next(
            (h for h in hands if h.handedness == "Right" and len(h.landmarks) == 21),
            None,
        )
        if target is None:
            return DetectionResult(sign="-", confidence=0.0, hands=hands)

        vec = np.empty(N_FEATURES, dtype=np.float32)
        for i, lm in enumerate(target.landmarks):
            vec[i * 3 : i * 3 + 3] = (lm.x, lm.y, lm.z)
        vec = _normalize(vec)

        logits = self.session.run(None, {self.input_name: vec.reshape(1, N_FEATURES)})[0][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        sign = self.label_map[str(idx)] if confidence >= MIN_CONFIDENCE else "-"

        return DetectionResult(sign=sign, confidence=confidence, hands=hands)
