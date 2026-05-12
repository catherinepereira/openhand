"""
HandDetector — decode incoming base64 webcam frames and extract 21 MediaPipe
hand landmarks, ready for the ONNX sign classifier.

Uses the modern MediaPipe Tasks API (the legacy ``mp.solutions.hands`` was
removed in MediaPipe 0.10.20+). The .task model file is shipped at
``backend/models/artifacts/hand_landmarker.task``.
"""

import base64
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ..models.schemas import Landmark

_TASK_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "artifacts" / "hand_landmarker.task"
)


class HandDetector:
    def __init__(self):
        if not _TASK_MODEL.exists():
            raise FileNotFoundError(
                f"MediaPipe hand_landmarker.task not found at {_TASK_MODEL}. "
                "Copy it from openhand-model/data/."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_TASK_MODEL)),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(
        self, image_data: str
    ) -> tuple[Optional[list[Landmark]], Optional[np.ndarray]]:
        """Decode base64 JPEG, run MediaPipe, return (landmarks, raw_frame).

        Returns ``(None, None)`` if the frame can't be decoded; ``(None, frame)``
        if decoding succeeded but no hand was found.
        """
        _, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        try:
            img_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            return None, None

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None, frame

        return (
            [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in result.hand_landmarks[0]],
            frame,
        )

    def close(self):
        self.detector.close()
