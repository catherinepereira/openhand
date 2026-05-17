"""
Landmark selection + engineered features for the isolated-sign classifier.

MUST stay in sync with openhand-model/model/signs_landmarks.py. Same
file, intentional duplicate so the backend doesn't import from the
training repo. Drift here means the deployed model silently sees inputs
in the wrong format.

The frontend extracts the 127-landmark subset client-side and sends the
raw (T, 127, 3) tensor + (T, 127) missing mask over the WebSocket; this
module normalizes per-clip and builds the engineered features that the
ONNX expects.
"""

from __future__ import annotations

import numpy as np

LIPS_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE_IDX = [1, 2, 98, 327]
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
HAND_IDX = list(range(21))

N_FACE_LM = len(LIPS_IDX) + len(LEFT_EYE_IDX) + len(RIGHT_EYE_IDX) + len(NOSE_IDX)
N_POSE_LM = len(POSE_IDX)
N_HAND_LM = len(HAND_IDX)
N_LANDMARKS = N_FACE_LM + N_POSE_LM + 2 * N_HAND_LM
N_COORDS = 3
N_BASE_FEATURES = N_LANDMARKS * N_COORDS                        # 381
N_HAND_LIP_DISTANCES = 2 * N_HAND_LM                            # 42
N_MOTION_FEATURES = N_BASE_FEATURES                             # 381
N_FEATURES = N_BASE_FEATURES + N_MOTION_FEATURES + N_HAND_LIP_DISTANCES  # 804

LIPS_SLICE = slice(0, len(LIPS_IDX))
LEFT_HAND_SLICE = slice(N_FACE_LM + N_POSE_LM, N_FACE_LM + N_POSE_LM + N_HAND_LM)
RIGHT_HAND_SLICE = slice(N_FACE_LM + N_POSE_LM + N_HAND_LM, N_LANDMARKS)


def normalize_clip(arr: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Per-clip mean/std normalization. Mirror of training-side function."""
    arr = arr.copy().astype(np.float32, copy=False)
    present_mask = ~missing
    flat = arr.reshape(-1, 3)
    flat_mask = present_mask.reshape(-1)
    present = flat[flat_mask]
    if present.size:
        mean = present.mean(axis=0)
        std = np.maximum(present.std(axis=0), 1e-6)
    else:
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)
    arr = (arr - mean) / std
    arr[missing] = 0.0
    return arr.astype(np.float32, copy=False)


def build_features(
    arr: np.ndarray,
    missing: np.ndarray,
) -> np.ndarray:
    """Mirror of training-side build_features.

    arr:     (T, 127, 3) normalized landmarks
    missing: (T, 127) bool
    Returns: (T, N_FEATURES) float32
    """
    T = arr.shape[0]
    if T == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    base = arr.reshape(T, N_BASE_FEATURES)
    motion = np.zeros_like(base)
    if T > 1:
        motion[1:] = base[1:] - base[:-1]
    lip_centroid = arr[:, LIPS_SLICE, :].mean(axis=1)
    left_dist = np.linalg.norm(
        arr[:, LEFT_HAND_SLICE, :] - lip_centroid[:, None, :], axis=-1
    ).astype(np.float32)
    right_dist = np.linalg.norm(
        arr[:, RIGHT_HAND_SLICE, :] - lip_centroid[:, None, :], axis=-1
    ).astype(np.float32)
    hand_lip = np.concatenate([left_dist, right_dist], axis=1)
    features = np.concatenate([base, motion, hand_lip], axis=1).astype(np.float32)
    return features
