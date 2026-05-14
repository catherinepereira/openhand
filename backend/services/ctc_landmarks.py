"""
Landmark selection for the CTC fingerspelling model.

**MUST stay in sync with** ``openhand-model/model/landmarks.py``. This is a
deliberate copy (the backend doesn't import from the training repo). The
index lists, group order, and ``normalize_sequence`` body must match
exactly — drift here means the deployed CTC ONNX silently sees inputs in
the wrong format and produces garbage.

The CTC ONNX consumes (B, T, N_FEATURES) tensors where each frame's floats
are laid out as:
  40 lips × 3 + 16 left-eye × 3 + 16 right-eye × 3 + 4 nose × 3
  + 9 pose × 3 + 21 left-hand × 3 + 21 right-hand × 3
  = 127 landmarks × 3 = 381 floats
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

# MediaPipe Pose indices: 0=nose, 11/12=shoulders, 13/14=elbows, 15/16=wrists, 23/24=hips
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
HAND_IDX = list(range(21))

# Group offsets in the feature vector (in units of *features*, i.e. ×3 of
# the landmark indices). The right-wrist anchor lives at offset
# GROUP_OFFSETS["right_hand"] (this is what training's normalize_sequence
# uses).
N_FACE_LM = len(LIPS_IDX) + len(LEFT_EYE_IDX) + len(RIGHT_EYE_IDX) + len(NOSE_IDX)
N_POSE_LM = len(POSE_IDX)
N_HAND_LM = len(HAND_IDX)

N_LANDMARKS = N_FACE_LM + N_POSE_LM + 2 * N_HAND_LM
N_FEATURES = N_LANDMARKS * 3

GROUP_OFFSETS = {
    "face":       0,
    "pose":       N_FACE_LM * 3,
    "left_hand":  (N_FACE_LM + N_POSE_LM) * 3,
    "right_hand": (N_FACE_LM + N_POSE_LM + N_HAND_LM) * 3,
}


def build_frame_features(
    face_lms: list | None,
    pose_lms: list | None,
    left_hand_lms: list | None,
    right_hand_lms: list | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one frame's MediaPipe Holistic landmarks into a (N_FEATURES,)
    feature vector + (N_LANDMARKS,) missing mask.

    Each input is either ``None`` (entire group missing) or a list of mediapipe
    NormalizedLandmark with .x .y .z attributes covering ALL landmarks for
    that group (the full 468 face landmarks, 33 pose, 21 hand — we slice the
    subset of interest by index).
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    missing = np.ones(N_LANDMARKS, dtype=bool)
    lm_cursor = 0

    def write(group_lms, indices):
        nonlocal lm_cursor
        if group_lms is None:
            lm_cursor += len(indices)
            return
        for i in indices:
            lm = group_lms[i]
            base = lm_cursor * 3
            vec[base]     = lm.x
            vec[base + 1] = lm.y
            vec[base + 2] = lm.z
            missing[lm_cursor] = False
            lm_cursor += 1

    write(face_lms, LIPS_IDX)
    write(face_lms, LEFT_EYE_IDX)
    write(face_lms, RIGHT_EYE_IDX)
    write(face_lms, NOSE_IDX)
    write(pose_lms, POSE_IDX)
    write(left_hand_lms,  HAND_IDX)
    write(right_hand_lms, HAND_IDX)

    return vec, missing


def normalize_sequence(arr: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Match openhand-model/model/landmarks.py::normalize_sequence.

    arr:     (T, N_FEATURES) float32  — already NaN-filled with 0
    missing: (T, N_LANDMARKS) bool    — True where landmark was absent
    """
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32, copy=True)
    pts = arr.reshape(arr.shape[0], -1, 3)  # (T, N_LANDMARKS, 3)
    missing = missing.astype(bool, copy=False)

    rh_lm = GROUP_OFFSETS["right_hand"] // 3
    lh_lm = GROUP_OFFSETS["left_hand"]  // 3
    right_present = ~missing[:, rh_lm]
    left_present  = ~missing[:, lh_lm]

    if right_present.sum() >= left_present.sum():
        wrist_xyz = pts[:, rh_lm, :]
        wrist_mask = right_present
    else:
        wrist_xyz = pts[:, lh_lm, :]
        wrist_mask = left_present

    if wrist_mask.any():
        anchor = wrist_xyz[wrist_mask].mean(axis=0)
    else:
        anchor = np.zeros(3, dtype=np.float32)

    pts = pts - anchor
    present_pts = pts[~missing]
    if present_pts.size:
        scale = np.percentile(np.abs(present_pts), 95)
        if scale > 1e-6:
            pts = pts / scale

    pts[missing] = 0.0
    return pts.reshape(arr.shape[0], -1).astype(np.float32)
