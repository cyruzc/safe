from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


# ============ IO Functions ============

def load_split_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.float32)


def save_response(response: np.ndarray, npy_path: Path, png_path: Path | None) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, response.astype(np.float32))
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((response * 255.0).clip(0, 255).astype(np.uint8)).save(png_path)


def save_prior(prior: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prior * 255).astype(np.uint8)).save(path)


# ============ Prior Building Functions ============

@dataclass
class SplitSummary:
    split: str
    count: int
    mean_inner_pixels: float
    mean_outer_pixels: float


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Find connected components in a binary mask using 8-connectivity."""
    binary = mask.astype(bool)
    try:
        cv2 = require_cv2()
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return [np.argwhere(labels == label).astype(np.int32) for label in range(1, num_labels)]
    except ModuleNotFoundError:
        visited = np.zeros(binary.shape, dtype=bool)
        height, width = binary.shape
        components: list[np.ndarray] = []
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                queue = deque([(y, x)])
                visited[y, x] = True
                pixels: list[tuple[int, int]] = []
                while queue:
                    cy, cx = queue.popleft()
                    pixels.append((cy, cx))
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                components.append(np.array(pixels, dtype=np.int32))
        return components


def component_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    """Extract per-component centers using floor() to match MATLAB centroid labels."""
    centers: list[tuple[int, int]] = []
    for component in connected_components(mask):
        if len(component) == 0:
            continue
        ys = component[:, 0].astype(np.float32)
        xs = component[:, 1].astype(np.float32)
        cy = int(np.floor(float(ys.mean())))
        cx = int(np.floor(float(xs.mean())))
        cy = min(max(cy, 0), mask.shape[0] - 1)
        cx = min(max(cx, 0), mask.shape[1] - 1)
        centers.append((cy, cx))
    return centers


def positive_pixel_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(mask.astype(np.uint8) > 0)
    return [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]


def make_anchor_mask(shape: tuple[int, int], centers: list[tuple[int, int]], radius: int) -> np.ndarray:
    cv2 = require_cv2()
    mask = np.zeros(shape, dtype=np.uint8)
    for cy, cx in centers:
        cv2.circle(mask, (cx, cy), radius, color=1, thickness=-1)
    return mask


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    cv2 = require_cv2()
    binary = (mask > 0).astype(np.uint8)
    if radius <= 0:
        return binary
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary, kernel, iterations=1)


def keep_connected_to_anchors(candidate: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    num_labels, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(candidate, dtype=np.uint8)
    for label in range(1, num_labels):
        component = labels == label
        if np.any(anchor_mask[component] > 0):
            kept[component] = 1
    return kept


def build_prior_from_response(
    response: np.ndarray,
    centers: list[tuple[int, int]],
    percentile: float,
    anchor_radius: int,
    dilate_iters: int = 0,
) -> np.ndarray:
    cv2 = require_cv2()
    threshold = float(np.percentile(response, percentile))
    candidate = (response >= threshold).astype(np.uint8)
    if dilate_iters > 0:
        candidate = cv2.dilate(candidate, np.ones((3, 3), np.uint8), iterations=dilate_iters)
    anchor_mask = make_anchor_mask(response.shape, centers, anchor_radius)
    prior = keep_connected_to_anchors(candidate, anchor_mask)
    prior = np.maximum(prior, anchor_mask)
    return (prior > 0).astype(np.uint8)


# ============ Prior Generation Methods ============

@lru_cache(maxsize=1)
def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for prior generation. Install `opencv-python` or `opencv-python-headless`."
        ) from exc
    return cv2


def normalize_to_unit_interval(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def safe_std(image: np.ndarray, ksize: int) -> np.ndarray:
    cv2 = require_cv2()
    mean = cv2.blur(image, (ksize, ksize))
    sq_mean = cv2.blur(image * image, (ksize, ksize))
    variance = np.maximum(sq_mean - mean * mean, 0.0)
    return np.sqrt(variance, dtype=np.float32)


def lagdm_response(image_u8: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    image = image_u8.astype(np.float32) / 255.0
    g_small = cv2.GaussianBlur(image, (0, 0), sigmaX=0.8)
    g_mid = cv2.GaussianBlur(image, (0, 0), sigmaX=1.6)
    g_large = cv2.GaussianBlur(image, (0, 0), sigmaX=3.2)
    dog_near = np.maximum(g_small - g_mid, 0.0)
    dog_far = np.maximum(g_mid - g_large, 0.0)
    local_mean = cv2.blur(image, (9, 9))
    local_std = safe_std(image, 9)
    saliency = np.maximum(image - local_mean, 0.0) / (local_std + 1e-6)
    response = (0.65 * dog_near + 0.35 * dog_far) * (1.0 + saliency)
    return normalize_to_unit_interval(response)


def dog_response(image_u8: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    image = image_u8.astype(np.float32) / 255.0
    g_small = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    g_large = cv2.GaussianBlur(image, (0, 0), sigmaX=2.4)
    response = np.maximum(g_small - g_large, 0.0)
    return normalize_to_unit_interval(response)


def wslcm_response(image_u8: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    image = image_u8.astype(np.float32) / 255.0
    local_mean_small = cv2.blur(image, (5, 5))
    local_mean_large = cv2.blur(image, (15, 15))
    local_std_large = safe_std(image, 15)
    top_hat = cv2.morphologyEx((image * 255.0).astype(np.uint8), cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8))
    top_hat = top_hat.astype(np.float32) / 255.0
    contrast = np.maximum(local_mean_small - local_mean_large, 0.0)
    weight = 1.0 / (1.0 + local_std_large)
    response = np.maximum(top_hat, contrast * weight)
    return normalize_to_unit_interval(response)


def mpcm_response(image_u8: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    image = image_u8.astype(np.float32) / 255.0
    center3 = cv2.blur(image, (3, 3))
    center5 = cv2.blur(image, (5, 5))
    surround9 = cv2.blur(image, (9, 9))
    surround15 = cv2.blur(image, (15, 15))
    contrast_small = np.maximum(center3 - surround9, 0.0)
    contrast_mid = np.maximum(center5 - surround15, 0.0)
    response = np.maximum(contrast_small, contrast_mid)
    return normalize_to_unit_interval(response)


METHOD_REGISTRY = {
    "dog": dog_response,
    "lagdm": lagdm_response,
    "mpcm": mpcm_response,
    "wslcm": wslcm_response,
}


__all__ = [
    # IO functions
    "load_split_file",
    "load_grayscale",
    "save_response",
    "save_prior",
    # Prior building
    "SplitSummary",
    "connected_components",
    "component_centers",
    "positive_pixel_centers",
    "make_anchor_mask",
    "dilate_binary_mask",
    "keep_connected_to_anchors",
    "build_prior_from_response",
    # Prior generation methods
    "require_cv2",
    "normalize_to_unit_interval",
    "safe_std",
    "lagdm_response",
    "dog_response",
    "wslcm_response",
    "mpcm_response",
    "METHOD_REGISTRY",
]
