import base64
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

_hands: Optional[mp.solutions.hands.Hands] = None


def _get_hands() -> mp.solutions.hands.Hands:
    global _hands
    if _hands is None:
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
        )
    return _hands


def decode_image(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image — invalid base64 or unsupported format")
    return img


def extract_landmarks(image_b64: str) -> dict:
    img = decode_image(image_b64)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands = _get_hands()
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        raise ValueError("NO_HAND_DETECTED")

    lm = results.multi_hand_landmarks[0]
    landmarks = [{"x": p.x, "y": p.y, "z": p.z} for p in lm.landmark]

    chirality = "Right"
    confidence = 0.0
    if results.multi_handedness:
        h = results.multi_handedness[0].classification[0]
        chirality = h.label
        confidence = float(h.score)

    return {
        "landmarks": landmarks,
        "chirality": chirality,
        "confidence": confidence,
        "image": img,
    }
