from typing import List

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as nd_label
from skimage.morphology import skeletonize as ski_skeletonize

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
    if len(roi.shape) == 3:
        gray = (0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]).astype(np.uint8)
    else:
        gray = roi.astype(np.uint8)
    return np.array(PILImage.fromarray(gray).resize((size, size), PILImage.LANCZOS))


def skeletonize(roi: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(roi.astype(np.float64), sigma=1.1)
    local_mean = gaussian_filter(blurred, sigma=11 / 6.0)
    thresh = ((local_mean - blurred) > 2.0).astype(np.uint8) * 255
    skel = ski_skeletonize(thresh > 0)
    return (skel.astype(np.uint8) * 255)


def extract_line_segments(skeleton: np.ndarray, n: int = 4) -> List[np.ndarray]:
    labeled, n_labels = nd_label(skeleton > 0)
    if n_labels == 0:
        return [np.zeros_like(skeleton)] * n
    areas = np.bincount(labeled.ravel())
    order = sorted(range(1, n_labels + 1), key=lambda i: areas[i], reverse=True)
    lines = []
    for idx in order[:n]:
        mask = np.zeros_like(skeleton)
        mask[labeled == idx] = 255
        lines.append(mask)
    while len(lines) < n:
        lines.append(np.zeros_like(skeleton))
    return lines
