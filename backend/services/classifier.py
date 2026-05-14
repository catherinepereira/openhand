"""
ASL letter classifier backed by an ONNX MLP trained in openhand-model/.

Loads the model once at import time. The classifier consumes the 21 MediaPipe
hand landmarks already produced by HandDetector, normalises them the same way
the training pipeline did (wrist at origin, scaled by 95th-percentile |value|),
and returns the predicted letter + confidence.

The training pipeline lives in ../../../openhand-model/. To retrain or update
the model, regenerate exports/asl_classifier.onnx there and copy it (along
with model_meta.json) into backend/models/artifacts/.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..models.schemas import DetectedHand, DetectionResult, Landmark

_ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
_MODEL_PATH = _ARTIFACTS / "asl_classifier.onnx"
_META_PATH = _ARTIFACTS / "model_meta.json"

# MediaPipe Hands returns this many landmarks per hand; each has (x, y, z).
N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3

# Below this confidence the classifier returns "—" rather than guessing.
MIN_CONFIDENCE = 0.5
# Percentile used by the training preprocessor for scale normalisation.
NORM_PERCENTILE = 95
# Guards division when the entire hand is at the origin.
NORM_EPS = 1e-6


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Match openhand-model/scripts/preprocess_alphabet.py::normalize.

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

        Runs the MLP against the hand MediaPipe labelled "Right" (camera
        POV). The returned ``hands`` field carries every detected hand
        through unchanged so the frontend can render overlays for both.
        """
        if not hands:
            return DetectionResult(sign="—", confidence=0.0, hands=[])

        target: DetectedHand | None = next(
            (h for h in hands if h.handedness == "Right" and len(h.landmarks) == 21),
            None,
        )
        if target is None:
            return DetectionResult(sign="—", confidence=0.0, hands=hands)

        vec = np.array([[lm.x, lm.y, lm.z] for lm in target.landmarks], dtype=np.float32).reshape(63)
        vec = _normalize(vec)

        logits = self.session.run(None, {self.input_name: vec.reshape(1, 63)})[0][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        sign = self.label_map[str(idx)] if confidence >= MIN_CONFIDENCE else "—"

        return DetectionResult(sign=sign, confidence=confidence, hands=hands)
