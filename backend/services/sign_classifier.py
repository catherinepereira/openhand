"""
Isolated-sign classifier service. Wraps the ONNX-exported transformer
trained on the Google ISLR (Kaggle asl-signs) dataset.

Input:  a (T, 127, 3) float32 raw landmark tensor + (T, 127) bool
        missing mask for one clip. The backend normalizes and builds
        engineered features here, runs ONNX, returns top-K predictions.
Output: list of (sign_name, probability) tuples, length K.

Lazy-loaded because the ONNX is ~10-20 MB and we don't want to pay
startup cost if the user never opens the Words view.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .signs_landmarks import N_FEATURES, N_LANDMARKS, build_features, normalize_clip

_ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
_ONNX = _ARTIFACTS / "sign_classifier.onnx"
_META = _ARTIFACTS / "sign_classifier_meta.json"

MAX_FRAMES = 80


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class SignClassifier:
    def __init__(self) -> None:
        if not _ONNX.exists():
            raise FileNotFoundError(
                f"Sign classifier ONNX not found at {_ONNX}. "
                "Train it via openhand-model/signs/scripts/run_pipeline.py, "
                "then copy the artifacts here."
            )
        with open(_META) as f:
            meta = json.load(f)
        self.idx_to_sign: dict[int, str] = {
            int(k): v for k, v in meta["idx_to_sign"].items()
        }
        self.sign_to_idx: dict[str, int] = meta["sign_to_idx"]
        self.num_classes: int = int(meta["num_classes"])
        self.max_frames: int = int(meta.get("max_frames", MAX_FRAMES))
        self.session = ort.InferenceSession(str(_ONNX), providers=["CPUExecutionProvider"])
        self.input_x = "features"
        self.input_mask = "pad_mask"

    def classify(
        self,
        landmarks: np.ndarray,   # (T, N_LANDMARKS, 3)
        missing: np.ndarray,     # (T, N_LANDMARKS)
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        if landmarks.shape[0] == 0:
            return []
        if landmarks.shape[1] != N_LANDMARKS:
            raise ValueError(
                f"Expected (T, {N_LANDMARKS}, 3) landmarks, got {landmarks.shape}"
            )

        # Stride-sample if too long.
        if landmarks.shape[0] > self.max_frames:
            idxs = np.linspace(0, landmarks.shape[0] - 1, self.max_frames).astype(int)
            landmarks = landmarks[idxs]
            missing = missing[idxs]

        x = normalize_clip(landmarks, missing)
        features = build_features(x, missing)  # (T, N_FEATURES)
        T = features.shape[0]

        # Dynamo-exported ONNX needs batch >= 2 for the batch axis to
        # stay dynamic. Pad with an all-masked second item we discard.
        x_batch = np.stack([features, np.zeros_like(features)], axis=0).astype(np.float32)
        mask = np.zeros((2, T), dtype=bool)
        mask[1, :] = True

        logits = self.session.run(
            None,
            {self.input_x: x_batch, self.input_mask: mask},
        )[0]
        # (B=2, num_classes)
        probs = _softmax(logits[0].astype(np.float32))

        # Top-k
        k = min(top_k, self.num_classes)
        top_idx = np.argpartition(-probs, k - 1)[:k]
        top_idx = top_idx[np.argsort(-probs[top_idx])]
        return [
            (self.idx_to_sign[int(i)], float(probs[int(i)]))
            for i in top_idx
        ]


# A bare assertion the feature dim matches what the model expects, so
# wiring bugs show up at import time rather than at first inference.
assert N_FEATURES == 804, f"Sign classifier feature dim drift: {N_FEATURES}"
