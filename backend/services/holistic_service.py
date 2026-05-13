"""
HolisticDetector — produces the 390-feature vector the CTC fingerspelling
model expects by running THREE independent MediaPipe Tasks detectors per
frame (HandLandmarker for both hands, PoseLandmarker for upper body,
FaceLandmarker for the lips/eyes/nose subset).

We initially tried the all-in-one ``HolisticLandmarker``, but on Windows /
MediaPipe 0.10.21 it crashes the entire process with a C++ check failure
(``packet is empty``) on frames where one sub-task produces an empty packet
the next stage doesn't tolerate. Running the three sub-tasks separately
avoids the cross-stage packet plumbing entirely.

This is only used by ``/api/transcribe``; the live alphabet flow keeps using
the lighter, single-purpose ``HandDetector`` in ``mediapipe_service.py``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .ctc_landmarks import build_frame_features

_ARTIFACTS = Path(__file__).resolve().parent.parent / "models" / "artifacts"
_HAND_TASK = _ARTIFACTS / "hand_landmarker.task"
_POSE_TASK = _ARTIFACTS / "pose_landmarker.task"
_FACE_TASK = _ARTIFACTS / "face_landmarker.task"


def _require(path: Path, url_hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found at {path}. Download from {url_hint}"
        )
    return path


class HolisticDetector:
    def __init__(self):
        _require(_HAND_TASK,
                 "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        _require(_POSE_TASK,
                 "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        _require(_FACE_TASK,
                 "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

        # Hands — two simultaneously (left + right)
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_HAND_TASK)),
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_POSE_TASK)),
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_FACE_TASK)),
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                running_mode=vision.RunningMode.IMAGE,
            )
        )

    def _decode(self, image_data: str) -> Optional[np.ndarray]:
        _, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        try:
            img_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            return None
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def process_frame(self, image_data: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Decode one base64 frame and return (feature_vec(390,), missing_mask(130,))."""
        frame = self._decode(image_data)
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Each detector wraps its own internal Try / produces no result on
        # failure. Wrap in try/except defensively so a single weird frame can't
        # bring the whole transcribe down.
        try:
            hands_r = self.hands.detect(mp_image)
        except Exception:
            hands_r = None
        try:
            pose_r = self.pose.detect(mp_image)
        except Exception:
            pose_r = None
        try:
            face_r = self.face.detect(mp_image)
        except Exception:
            face_r = None

        left_hand = right_hand = None
        if hands_r and hands_r.hand_landmarks:
            for lm_list, handed in zip(hands_r.hand_landmarks, hands_r.handedness):
                top = handed[0].category_name if handed else ""
                # MediaPipe reports handedness from the camera's POV, but we
                # also flip the video in the frontend (transform: scaleX(-1)).
                # The training data uses the dataset's own left/right
                # convention — we match it by trusting MediaPipe's label.
                if top.lower() == "left":
                    left_hand = lm_list
                elif top.lower() == "right":
                    right_hand = lm_list

        pose_lms = pose_r.pose_landmarks[0] if (pose_r and pose_r.pose_landmarks) else None
        face_lms = face_r.face_landmarks[0] if (face_r and face_r.face_landmarks) else None

        return build_frame_features(face_lms, pose_lms, left_hand, right_hand)

    def close(self):
        for d in (self.hands, self.pose, self.face):
            try:
                d.close()
            except Exception:
                pass
