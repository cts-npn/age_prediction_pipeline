"""Shared face-landmark detector used by align_faces.py and the backend at
inference time. Wraps MediaPipe's FaceLandmarker task and exposes a single
function returning 5 points in the same order/shape MTCNN used to produce
(left eye, right eye, nose tip, left mouth corner, right mouth corner), so
align_image() in align_faces.py needs no changes.

Requires the FaceLandmarker model bundle at MODEL_PATH (download once):
    curl -L -o models/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Landmark index mapping (verified visually against MediaPipe's 478-point
face mesh, image coordinates, not mirrored):
    left eye (image-left)   -> 468
    right eye (image-right) -> 473
    nose tip                -> 4
    left mouth corner       -> 61
    right mouth corner      -> 291
"""
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

# Order matches align_faces.REFERENCE_LANDMARKS: left eye, right eye, nose
# tip, left mouth corner, right mouth corner.
LANDMARK_INDICES = [468, 473, 4, 61, 291]

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {MODEL_PATH}. Download it with:\n"
                f"curl -L -o {MODEL_PATH} "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def detect_landmarks(img: Image.Image) -> np.ndarray | None:
    """Detect a single face and return its 5 alignment landmarks in pixel
    coordinates as a (5, 2) float32 array, or None if no face was found."""
    import mediapipe as mp

    landmarker = _get_landmarker()
    w, h = img.size
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img.convert("RGB")))
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]
    points = np.array([[face[i].x * w, face[i].y * h] for i in LANDMARK_INDICES], dtype=np.float32)
    return points
