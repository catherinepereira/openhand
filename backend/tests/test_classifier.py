"""
Behavior tests for SignClassifier: input shape contracts, hand selection,
confidence-gated output. Skipped automatically if the alphabet ONNX isn't
on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.models.schemas import DetectedHand, Landmark
from backend.services.classifier import SignClassifier

_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "models" / "artifacts" / "asl_classifier.onnx"
)
pytestmark = pytest.mark.skipif(
    not _ARTIFACT.exists(),
    reason=f"alphabet ONNX missing at {_ARTIFACT}",
)


def _hand(handedness: str, x: float = 0.5) -> DetectedHand:
    """A 21-landmark hand at a fixed-ish position. Not a real ASL letter;
    these tests check classifier mechanics, not predictions."""
    return DetectedHand(
        handedness=handedness,
        landmarks=[Landmark(x=x + i * 0.001, y=0.5, z=0.0) for i in range(21)],
    )


@pytest.fixture(scope="module")
def classifier() -> SignClassifier:
    return SignClassifier()


def test_classify_empty_hands_returns_dash(classifier: SignClassifier) -> None:
    result = classifier.classify([])
    assert result.sign == "-"
    assert result.confidence == 0.0
    assert result.hands == []


def test_classify_wrong_landmark_count_returns_dash(classifier: SignClassifier) -> None:
    """A Right hand with fewer than 21 landmarks falls through to the sentinel."""
    short = DetectedHand(
        handedness="Right",
        landmarks=[Landmark(x=0.5, y=0.5, z=0.0)] * 10,
    )
    result = classifier.classify([short])
    assert result.sign == "-"
    assert len(result.hands) == 1


def test_classify_picks_right_hand(classifier: SignClassifier) -> None:
    """When both hands are present, the Right one feeds the model."""
    left = _hand("Left", x=0.1)
    right = _hand("Right", x=0.9)
    result = classifier.classify([left, right])
    assert len(result.hands) == 2
    assert isinstance(result.sign, str)
    assert 0.0 <= result.confidence <= 1.0


def test_classify_no_right_hand_returns_dash(classifier: SignClassifier) -> None:
    result = classifier.classify([_hand("Left", x=0.1)])
    assert result.sign == "-"
    assert len(result.hands) == 1
