import urllib.request
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

for name, url in MODELS.items():
    dest = ARTIFACTS / name
    if dest.exists():
        print(f"OK    {name} ({dest.stat().st_size/1e6:.1f} MB)")
        continue
    print(f"DL    {name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"      saved {dest.stat().st_size/1e6:.1f} MB")
