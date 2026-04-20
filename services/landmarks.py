import os
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import base64
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "/tmp/hand_landmarker.task"


def _download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


_landmarker: Optional[HandLandmarker] = None


def _get_landmarker() -> HandLandmarker:
    global _landmarker
    if _landmarker is None:
        _download_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        _landmarker = HandLandmarker.create_from_options(options)
    return _landmarker


def decode_image(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image — invalid base64 or unsupported format")
    return img


def extract_landmarks(image_b64: str) -> dict:
    img = decode_image(image_b64)
    landmarker = _get_landmarker()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        raise ValueError("NO_HAND_DETECTED")

    lm = result.hand_landmarks[0]
    landmarks = [{"x": p.x, "y": p.y, "z": p.z} for p in lm]
    chirality = result.handedness[0][0].display_name
    confidence = float(result.handedness[0][0].score)

    return {
        "landmarks": landmarks,
        "chirality": chirality,
        "confidence": confidence,
        "image": img,
    }
