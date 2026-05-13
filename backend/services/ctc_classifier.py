"""
CTC fingerspelling classifier — wraps the ONNX-exported transformer.

Input:  a (T, N_FEATURES) float32 numpy array (one signed phrase, T frames)
        plus an explicit (T, N_LANDMARKS) bool missing mask.
Output: decoded string + per-step token list (kept for debugging / future
        beam search).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .ctc_landmarks import N_FEATURES, normalize_sequence

_ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
_ONNX = _ARTIFACTS / "asl_ctc.onnx"
_META = _ARTIFACTS / "asl_ctc_meta.json"

MAX_FRAMES = 256  # match training-time max_frames


def _greedy_decode(log_probs: np.ndarray, blank: int, idx_to_char: dict[int, str]) -> str:
    """log_probs: (T, V). Greedy collapse repeats, drop blanks."""
    preds = log_probs.argmax(axis=-1)
    out: list[str] = []
    prev = -1
    for v in preds:
        v = int(v)
        if v != prev:
            if v != blank and v in idx_to_char:
                out.append(idx_to_char[v])
            prev = v
    return "".join(out)


class CTCClassifier:
    def __init__(self):
        if not _ONNX.exists():
            raise FileNotFoundError(f"CTC ONNX not found at {_ONNX}.")
        with open(_META) as f:
            meta = json.load(f)
        self.idx_to_char: dict[int, str] = {int(k): v for k, v in meta["idx_to_char"].items()}
        self.blank_idx: int = int(meta["blank_idx"])
        self.session = ort.InferenceSession(str(_ONNX), providers=["CPUExecutionProvider"])
        self.input_x = "landmarks"
        self.input_mask = "pad_mask"

    def transcribe(self, features: np.ndarray, missing: np.ndarray) -> str:
        """features: (T, N_FEATURES) raw (un-normalised, zero-filled missing) frames.
        missing: (T, N_LANDMARKS) bool — True where landmark was absent."""
        if features.shape[0] == 0:
            return ""
        # Cap to training-time max_frames to stay within the model's known regime.
        if features.shape[0] > MAX_FRAMES:
            keep = np.linspace(0, features.shape[0] - 1, MAX_FRAMES).astype(int)
            features = features[keep]
            missing = missing[keep]

        x = normalize_sequence(features, missing)
        T = x.shape[0]

        # The dynamo-exported ONNX requires batch >= 2 to keep the batch axis
        # dynamic; pad with an all-zero, fully-masked second item we ignore.
        x_batch = np.stack([x, np.zeros_like(x)], axis=0).astype(np.float32)
        mask = np.zeros((2, T), dtype=bool)
        mask[1, :] = True

        log_probs = self.session.run(
            None,
            {self.input_x: x_batch, self.input_mask: mask},
        )[0]
        # (T, B=2, V) — take first batch element
        return _greedy_decode(log_probs[:, 0, :], self.blank_idx, self.idx_to_char)
