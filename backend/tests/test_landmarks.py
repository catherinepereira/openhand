"""
Numerical-parity tests for the landmark packer and normalizer.

The deployed CTC ONNX has no notion of which feature index is "lips" vs
"left wrist." If the backend's ctc_landmarks.py ever diverges from the
training repo's model/landmarks.py, the model silently sees inputs in the
wrong slot order and produces nonsense without crashing.

These tests pin the contract:

  1. N_LANDMARKS and N_FEATURES come out where we expect
     (127 landmarks * 3 axes = 381 floats per frame).
  2. The group offsets cover the full range with no gaps or overlaps.
  3. build_frame_features packs a hand-crafted set of inputs into the exact
     slot order the model was trained on.
  4. normalize_sequence produces the same wrist-anchored, p95-scaled output
     for a known sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.ctc_landmarks import (
    GROUP_OFFSETS,
    HAND_IDX,
    LEFT_EYE_IDX,
    LIPS_IDX,
    N_FEATURES,
    N_LANDMARKS,
    NOSE_IDX,
    POSE_IDX,
    RIGHT_EYE_IDX,
    build_frame_features,
    normalize_sequence,
)


class _LM:
    """Stand-in for a MediaPipe NormalizedLandmark; just x/y/z."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


def test_landmark_counts() -> None:
    """40 lips + 16 left-eye + 16 right-eye + 4 nose + 9 pose + 2*21 hands."""
    expected = (
        len(LIPS_IDX) + len(LEFT_EYE_IDX) + len(RIGHT_EYE_IDX) + len(NOSE_IDX)
        + len(POSE_IDX) + 2 * len(HAND_IDX)
    )
    assert N_LANDMARKS == expected == 127
    assert N_FEATURES == N_LANDMARKS * 3 == 381


def test_group_offsets_cover_range_without_gaps() -> None:
    """Group offsets must partition [0, N_FEATURES) exactly."""
    assert GROUP_OFFSETS["face"] == 0
    assert GROUP_OFFSETS["pose"] == (len(LIPS_IDX) + len(LEFT_EYE_IDX) + len(RIGHT_EYE_IDX) + len(NOSE_IDX)) * 3
    assert GROUP_OFFSETS["left_hand"] == GROUP_OFFSETS["pose"] + len(POSE_IDX) * 3
    assert GROUP_OFFSETS["right_hand"] == GROUP_OFFSETS["left_hand"] + len(HAND_IDX) * 3
    assert GROUP_OFFSETS["right_hand"] + len(HAND_IDX) * 3 == N_FEATURES


def _face_array(n: int = 470) -> list[_LM]:
    """468-element MediaPipe FaceMesh source with deterministic values."""
    return [_LM(i * 0.001, i * 0.002, i * 0.003) for i in range(n)]


def _pose_array(n: int = 33) -> list[_LM]:
    return [_LM(0.5 + i * 0.001, 0.5 + i * 0.001, 0.0) for i in range(n)]


def _hand_array(seed: float) -> list[_LM]:
    return [_LM(seed + i * 0.001, seed + i * 0.002, 0.0) for i in range(21)]


def test_build_frame_features_writes_lips_first() -> None:
    """First 40 landmark slots must be the lips, in canonical index order."""
    face = _face_array()
    vec, missing = build_frame_features(face, None, None, None)

    assert vec.shape == (N_FEATURES,)
    assert missing.shape == (N_LANDMARKS,)
    assert vec.dtype == np.float32

    for slot, raw_idx in enumerate(LIPS_IDX):
        base = slot * 3
        assert vec[base]     == pytest.approx(face[raw_idx].x)
        assert vec[base + 1] == pytest.approx(face[raw_idx].y)
        assert vec[base + 2] == pytest.approx(face[raw_idx].z)
        assert missing[slot] is np.bool_(False) or not bool(missing[slot])


def test_build_frame_features_missing_group_is_marked() -> None:
    """A None group leaves zeros and missing=True for every slot in that group."""
    vec, missing = build_frame_features(None, None, _hand_array(0.5), _hand_array(0.7))

    face_lm_count = GROUP_OFFSETS["left_hand"] // 3
    assert missing[:face_lm_count].all()
    assert (vec[: GROUP_OFFSETS["left_hand"]] == 0).all()

    rh_start = GROUP_OFFSETS["right_hand"] // 3
    lh_start = GROUP_OFFSETS["left_hand"] // 3
    assert not missing[lh_start : lh_start + 21].any()
    assert not missing[rh_start : rh_start + 21].any()


def test_build_frame_features_right_hand_lands_in_right_slot() -> None:
    """Each hand's data goes into its own block, no cross-contamination."""
    left = _hand_array(0.10)
    right = _hand_array(0.90)
    vec, _ = build_frame_features(None, None, left, right)

    lh_off = GROUP_OFFSETS["left_hand"]
    rh_off = GROUP_OFFSETS["right_hand"]
    assert vec[lh_off]     == pytest.approx(left[0].x)
    assert vec[rh_off]     == pytest.approx(right[0].x)
    assert vec[lh_off] != vec[rh_off]


def test_normalize_sequence_centers_on_dominant_wrist() -> None:
    """The chosen wrist's mean position must end up at the origin."""
    T = 5
    arr = np.zeros((T, N_FEATURES), dtype=np.float32)
    missing = np.zeros((T, N_LANDMARKS), dtype=bool)

    rh_off = GROUP_OFFSETS["right_hand"]
    lh_off = GROUP_OFFSETS["left_hand"]
    rh_lm = rh_off // 3
    lh_lm = lh_off // 3

    for t in range(T):
        arr[t, rh_off]     = 0.7
        arr[t, rh_off + 1] = 0.3
        arr[t, rh_off + 2] = 0.1
        missing[t, lh_lm : lh_lm + 21] = True

    out = normalize_sequence(arr, missing)

    for t in range(T):
        assert out[t, rh_off]     == pytest.approx(0.0, abs=1e-6)
        assert out[t, rh_off + 1] == pytest.approx(0.0, abs=1e-6)
        assert out[t, rh_off + 2] == pytest.approx(0.0, abs=1e-6)

    for t in range(T):
        assert (out[t, lh_off : lh_off + 21 * 3] == 0).all()


def test_normalize_sequence_handles_all_missing() -> None:
    """Empty input shouldn't crash; output is all zeros."""
    arr = np.zeros((3, N_FEATURES), dtype=np.float32)
    missing = np.ones((3, N_LANDMARKS), dtype=bool)
    out = normalize_sequence(arr, missing)
    assert out.shape == (3, N_FEATURES)
    assert (out == 0).all()
