from typing import List

import cv2
import numpy as np

PALM_INDICES = [0, 1, 5, 9, 13, 17]
ROI_SIZE = 256


def crop_palm_roi(image: np.ndarray, landmarks: List[dict], size: int = ROI_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    pts = np.array(
        [[landmarks[i]["x"] * w, landmarks[i]["y"] * h] for i in PALM_INDICES],
        dtype=np.float32,
    )
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pw = (x2 - x1) * 0.20
    ph = (y2 - y1) * 0.20
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(w, int(x2 + pw))
    y2 = min(h, int(y2 + ph))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        roi = image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    return cv2.resize(gray, (size, size))


def skeletonize(roi: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2,
    )
    skeleton = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton


def extract_line_segments(skeleton: np.ndarray, n: int = 4) -> List[np.ndarray]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    order = sorted(
        range(1, n_labels),
        key=lambda l: stats[l, cv2.CC_STAT_AREA],
        reverse=True,
    )
    lines = []
    for label_idx in order[:n]:
        mask = np.zeros_like(skeleton)
        mask[labels == label_idx] = 255
        lines.append(mask)
    while len(lines) < n:
        lines.append(np.zeros_like(skeleton))
    return lines
