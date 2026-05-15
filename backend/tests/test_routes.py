"""
End-to-end tests over the FastAPI surface using TestClient. Locks in the
wire format and happy-path round-trips for the two WebSocket routes plus
the trivial HTTP routes.

The CTC ONNX is large (~116 MB) and slow to load; tests touching
/ws/transcribe-stream are skipped if the file isn't on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.ctc_landmarks import N_FEATURES, N_LANDMARKS

_ALPHABET_ONNX = (
    Path(__file__).resolve().parent.parent / "models" / "artifacts" / "asl_classifier.onnx"
)
_CTC_ONNX = (
    Path(__file__).resolve().parent.parent / "models" / "artifacts" / "asl_ctc.onnx"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(not _ALPHABET_ONNX.exists(), reason="alphabet ONNX missing")
def test_detect_landmarks_round_trip(client: TestClient) -> None:
    """Send a valid 1-hand frame, expect a DetectionResult back."""
    payload = {
        "hands": [
            {
                "handedness": "Right",
                "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)],
            }
        ]
    }
    with client.websocket_connect("/ws/detect-landmarks") as ws:
        ws.send_json(payload)
        data = ws.receive_json()
    assert set(data.keys()) >= {"sign", "confidence", "hands"}
    assert isinstance(data["sign"], str)
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["hands"]) == 1
    assert data["hands"][0]["handedness"] == "Right"


@pytest.mark.skipif(not _ALPHABET_ONNX.exists(), reason="alphabet ONNX missing")
def test_detect_landmarks_invalid_payload_returns_empty(client: TestClient) -> None:
    """Malformed payload returns the sentinel empty response."""
    with client.websocket_connect("/ws/detect-landmarks") as ws:
        ws.send_json({"not_hands": "garbage"})
        data = ws.receive_json()
    assert data["sign"] == "—"
    assert data["confidence"] == 0.0
    assert data["hands"] == []


@pytest.mark.skipif(not _ALPHABET_ONNX.exists(), reason="alphabet ONNX missing")
def test_detect_landmarks_no_hands(client: TestClient) -> None:
    """Empty hand list returns the sentinel empty response."""
    with client.websocket_connect("/ws/detect-landmarks") as ws:
        ws.send_json({"hands": []})
        data = ws.receive_json()
    assert data["sign"] == "—"
    assert data["hands"] == []


@pytest.mark.skipif(not _CTC_ONNX.exists(), reason="CTC ONNX missing")
def test_transcribe_stream_empty_buffer(client: TestClient) -> None:
    """A zero-frame decode request must return an empty result, not crash."""
    payload = {
        "type": "decode",
        "frame_count": 0,
        "features": [],
        "missing": [],
    }
    with client.websocket_connect("/ws/transcribe-stream") as ws:
        ws.send_json(payload)
        data = ws.receive_json()
    assert data["type"] == "result"
    assert data["text"] == ""


@pytest.mark.skipif(not _CTC_ONNX.exists(), reason="CTC ONNX missing")
def test_transcribe_stream_length_mismatch_returns_error(client: TestClient) -> None:
    """features/missing arrays whose length doesn't match frame_count fail
    with a structured error message rather than a crash."""
    payload = {
        "type": "decode",
        "frame_count": 2,
        "features": [0.0] * (N_FEATURES * 2 + 1),  # one extra
        "missing": [False] * (N_LANDMARKS * 2),
    }
    with client.websocket_connect("/ws/transcribe-stream") as ws:
        ws.send_json(payload)
        data = ws.receive_json()
    assert data["type"] == "error"
    assert "length" in data["message"].lower()


@pytest.mark.skipif(not _CTC_ONNX.exists(), reason="CTC ONNX missing")
def test_transcribe_stream_decode_returns_text(client: TestClient) -> None:
    """A real-shape (synthetic) decode round-trips and returns a string;
    we only check the response envelope here."""
    T = 5
    payload = {
        "type": "decode",
        "frame_count": T,
        "features": [0.0] * (T * N_FEATURES),
        "missing": [True] * (T * N_LANDMARKS),
    }
    with client.websocket_connect("/ws/transcribe-stream") as ws:
        ws.send_json(payload)
        data = ws.receive_json()
    assert data["type"] == "result"
    assert isinstance(data["text"], str)
    assert isinstance(data["elapsed_ms"], (int, float))
