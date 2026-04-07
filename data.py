"""
Data loading and processing for IRSTD datasets.

Supports:
- Multiple datasets: SIRST3, IRSTD-1K, NUAA-SIRST, NUDT-SIRST
- Point supervision with centroid/coarse labels
- SAFE method with tri-zone priors
- Full supervision baseline

Key functions:
- build_dataset_config: Configure dataset paths
- IRSTDPointDataset: Main dataset class with data augmentation
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from utils import load_grayscale, load_split_file, require_cv2


# ============ Dataset Config ============

@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for IRSTD datasets."""
    name: str
    root: Path
    train_image_dir: str
    train_mask_dir: str
    test_image_dir: str
    test_mask_dir: str
    train_split: str | None
    test_split: str
    centroid_label_dir: str | None = None
    coarse_label_dir: str | None = None
    img_mean: float = 0.0
    img_std: float = 1.0

    def point_label_dir_for(self, label_mode: str | None) -> str | None:
        """Get point label directory for a given label mode."""
        if label_mode == "centroid":
            return self.centroid_label_dir
        if label_mode == "coarse":
            return self.coarse_label_dir
        return None


# ============ Dataset Registry ============

_DATA_ROOT = Path(__file__).parent / "datasets"

DATASET_REGISTRY = {
    "sirst3": DatasetConfig(
        name="sirst3",
        root=_DATA_ROOT / "SIRST3",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_SIRST3.txt",
        test_split="img_idx/test_SIRST3.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=95.010,
        img_std=41.511,
    ),
    "irstd1k": DatasetConfig(
        name="irstd1k",
        root=_DATA_ROOT / "IRSTD-1K",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_IRSTD-1K.txt",
        test_split="img_idx/test_IRSTD-1K.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=87.466,
        img_std=39.720,
    ),
    "nuaa_sirst": DatasetConfig(
        name="nuaa_sirst",
        root=_DATA_ROOT / "NUAA-SIRST",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_NUAA-SIRST.txt",
        test_split="img_idx/test_NUAA-SIRST.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=101.064,
        img_std=34.620,
    ),
    "nudt_sirst": DatasetConfig(
        name="nudt_sirst",
        root=_DATA_ROOT / "NUDT-SIRST",
        train_image_dir="images",
        train_mask_dir="masks",
        test_image_dir="images",
        test_mask_dir="masks",
        train_split="img_idx/train_NUDT-SIRST.txt",
        test_split="img_idx/test_NUDT-SIRST.txt",
        centroid_label_dir="masks_centroid",
        coarse_label_dir="masks_coarse",
        img_mean=95.0,
        img_std=40.0,
    ),
}


# ============ Factory Functions ============

def build_dataset_config(args) -> DatasetConfig:
    """Build dataset configuration from command-line args."""
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name:
        try:
            config = DATASET_REGISTRY[dataset_name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {dataset_name}") from exc
    else:
        dataset_root = getattr(args, "dataset_root", None)
        if dataset_root in (None, ""):
            raise ValueError("Either --dataset-name or --dataset-root must be provided.")
        config = DatasetConfig(
            name="custom",
            root=Path(dataset_root),
            train_image_dir=getattr(args, "image_dir_name", None) or "images",
            train_mask_dir=getattr(args, "mask_dir_name", None) or "masks",
            test_image_dir=getattr(args, "test_image_dir_name", None) or getattr(args, "image_dir_name", None) or "images",
            test_mask_dir=getattr(args, "test_mask_dir_name", None) or getattr(args, "mask_dir_name", None) or "masks",
            train_split=getattr(args, "train_split", None),
            test_split=getattr(args, "test_split", None) or "test.txt",
            centroid_label_dir=None,
            coarse_label_dir=None,
        )

    config = replace(
        config,
        root=Path(getattr(args, "dataset_root", None) or str(config.root)),
        train_image_dir=getattr(args, "image_dir_name", None) or config.train_image_dir,
        train_mask_dir=getattr(args, "mask_dir_name", None) or config.train_mask_dir,
        test_image_dir=getattr(args, "test_image_dir_name", None) or config.test_image_dir,
        test_mask_dir=getattr(args, "test_mask_dir_name", None) or config.test_mask_dir,
        train_split=getattr(args, "train_split", None) or config.train_split,
        test_split=getattr(args, "test_split", None) or config.test_split,
        centroid_label_dir=getattr(args, "point_label_dir", None) if getattr(args, "label_mode", None) == "centroid" else config.centroid_label_dir,
        coarse_label_dir=getattr(args, "point_label_dir", None) if getattr(args, "label_mode", None) == "coarse" else config.coarse_label_dir,
    )
    return config


def validate_dataset_config(config: DatasetConfig, require_train_split: bool = True) -> None:
    """Validate dataset configuration."""
    if not config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.root}")
    if require_train_split and config.train_split in (None, ""):
        raise ValueError("Training requires a dataset with a train split.")
    if config.test_split in (None, ""):
        raise ValueError("Dataset config is missing a test split.")


# ============ Dataset Loading ============

@dataclass(frozen=True)
class SampleRecord:
    """Record containing file paths for a single sample."""
    name: str
    image_path: Path
    mask_path: Path
    point_label_path: Path | None = None
    inner_prior_path: Path | None = None
    outer_prior_path: Path | None = None


def resolve_records(
    dataset_root: Path,
    split_name: str,
    image_dir_name: str = "images",
    mask_dir_name: str = "masks",
    point_label_dir: str | Path | None = None,
    inner_prior_dir: str | Path | None = None,
    outer_prior_dir: str | Path | None = None,
) -> list[SampleRecord]:
    """Resolve file paths for dataset samples."""
    def resolve_optional_path(path_like: str | Path | None, *, base_dir: Path) -> Path | None:
        if path_like is None:
            return None
        path_obj = Path(path_like)
        if path_obj.is_absolute():
            return path_obj
        cwd_candidate = path_obj.resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return base_dir / path_obj

    image_dir = dataset_root / image_dir_name
    mask_dir = dataset_root / mask_dir_name
    split_path = dataset_root / split_name
    resolved_point_dir = resolve_optional_path(point_label_dir, base_dir=dataset_root)
    resolved_inner_dir = resolve_optional_path(inner_prior_dir, base_dir=dataset_root)
    resolved_outer_dir = resolve_optional_path(outer_prior_dir, base_dir=dataset_root)

    names = load_split_file(split_path)
    records: list[SampleRecord] = []
    for name in names:
        image_path = image_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        point_label_path = None
        if resolved_point_dir is not None:
            point_label_path = resolved_point_dir / f"{name}.png"
            if not point_label_path.exists():
                raise FileNotFoundError(f"Missing point label: {point_label_path}")

        inner_prior_path = None
        if resolved_inner_dir is not None:
            inner_prior_path = resolved_inner_dir / f"{name}.png"
            if not inner_prior_path.exists():
                raise FileNotFoundError(f"Missing inner prior: {inner_prior_path}")

        outer_prior_path = None
        if resolved_outer_dir is not None:
            outer_prior_path = resolved_outer_dir / f"{name}.png"
            if not outer_prior_path.exists():
                raise FileNotFoundError(f"Missing outer prior: {outer_prior_path}")

        records.append(
            SampleRecord(
                name=name,
                image_path=image_path,
                mask_path=mask_path,
                point_label_path=point_label_path,
                inner_prior_path=inner_prior_path,
                outer_prior_path=outer_prior_path,
            )
        )
    return records


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Find connected components in binary mask using 8-connectivity."""
    binary = mask.astype(bool)
    try:
        cv2 = require_cv2()
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return [np.argwhere(labels == label).astype(np.int32) for label in range(1, num_labels)]
    except ModuleNotFoundError:
        # Fallback without cv2
        visited = np.zeros(binary.shape, dtype=bool)
        height, width = binary.shape
        components = []
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                queue = deque([(y, x)])
                visited[y, x] = True
                pixels = []
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
    """
    Extract centers of connected components.

    Uses floor() to match MATLAB's centroid_anno.m implementation.
    """
    centers = []
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


def create_point_map_from_centers(shape: tuple[int, int], centers: list[tuple[int, int]], point_radius: float = 0.0) -> np.ndarray:
    """
    Create point supervision map from center coordinates.

    Args:
        shape: Image shape (height, width)
        centers: List of (y, x) center coordinates
        point_radius: If 0, create single-point map. If >0, create Gaussian-blurred points.

    Returns:
        Binary point map with 1s at center locations.
    """
    height, width = shape
    point_map = np.zeros((height, width), dtype=np.float32)

    if point_radius <= 0:
        # Single-point map (no blur) - matches MATLAB centroid_anno.m
        for cy, cx in centers:
            if 0 <= cy < height and 0 <= cx < width:
                point_map[cy, cx] = 1.0
    else:
        # Gaussian-blurred point map (if needed in future)
        yy, xx = np.mgrid[0:height, 0:width]
        sigma2 = point_radius * point_radius
        for cy, cx in centers:
            if 0 <= cy < height and 0 <= cx < width:
                gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma2))
                point_map = np.maximum(point_map, gaussian.astype(np.float32))

    return point_map


def crop_arrays(
    image: np.ndarray,
    arrays: list[np.ndarray],
    crop_size: int,
    focus_prob: float,
    rng: random.Random,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Crop arrays around focus regions."""
    height, width = image.shape
    reference = arrays[0]
    if height < crop_size or width < crop_size:
        pad_h = max(0, crop_size - height)
        pad_w = max(0, crop_size - width)
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        arrays = [np.pad(array, ((0, pad_h), (0, pad_w)), mode="constant") for array in arrays]
        height, width = image.shape

    ys, xs = np.where(reference > 0)
    if len(ys) > 0 and rng.random() < focus_prob:
        idx = rng.randrange(len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        top = min(max(cy - crop_size // 2, 0), height - crop_size)
        left = min(max(cx - crop_size // 2, 0), width - crop_size)
    else:
        top = rng.randint(0, height - crop_size)
        left = rng.randint(0, width - crop_size)

    cropped_arrays = [array[top:top + crop_size, left:left + crop_size] for array in arrays]
    return image[top:top + crop_size, left:left + crop_size], cropped_arrays


class IRSTDPointDataset(Dataset):
    """
    IRSTD dataset with point supervision support.

    Supports:
    - Full supervision: uses full pixel masks
    - Point supervision: uses centroid or coarse point labels
    - SAFE method: uses point labels + tri-zone priors

    Data augmentation:
    - Random cropping (focused on target regions)
    - Random horizontal flip
    - Random rotation (0, 90, 180, 270 degrees)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split_name: str,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        point_label_dir: str | Path | None = None,
        inner_prior_dir: str | Path | None = None,
        outer_prior_dir: str | Path | None = None,
        crop_size: int = 256,
        point_sigma: float = 2.0,
        use_loaded_point_labels: bool = False,
        train: bool = True,
        focus_prob: float = 0.7,
        seed: int = 42,
        cache_data: bool = False,
        supervision_masks: dict[str, np.ndarray] | None = None,
        include_names: set[str] | None = None,
        img_mean: float = 0.0,
        img_std: float = 1.0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.img_mean = img_mean
        self.img_std = img_std
        records = resolve_records(
            self.dataset_root,
            split_name,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
            point_label_dir=point_label_dir,
            inner_prior_dir=inner_prior_dir,
            outer_prior_dir=outer_prior_dir,
        )
        if include_names is not None:
            records = [record for record in records if record.name in include_names]
        self.records = records
        self.crop_size = crop_size
        self.point_sigma = point_sigma
        self.use_loaded_point_labels = use_loaded_point_labels
        self.train = train
        self.focus_prob = focus_prob
        self._seed = seed
        self.rng = random.Random(seed)
        self.cache_data = cache_data
        self.supervision_masks = supervision_masks
        self.cached_samples: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]
        ] | None = None
        if self.cache_data:
            self.cached_samples = [
                self._load_sample_arrays(record)
                for record in self.records
            ]

    def __len__(self) -> int:
        return len(self.records)

    def _build_point_map(self, mask: np.ndarray, loaded_point_label: np.ndarray | None) -> np.ndarray:
        """
        Build point supervision map with fallback for robustness.

        Priority:
        1. Use pre-generated labels (masks_centroid/masks_coarse) if available
        2. Fallback: compute online from full mask (robustness)

        This ensures experiments can run even if pre-generated labels are missing.
        """
        if loaded_point_label is not None:
            return loaded_point_label.astype(np.float32, copy=False)

        # Fallback: compute from full mask (robustness for missing labels)
        print(f"[WARNING] Pre-generated point labels not found, computing online from full mask")
        print(f"[WARNING] This should only happen if masks_centroid/masks_coarse are missing or corrupted")
        centers = component_centers(mask)
        return create_point_map_from_centers(mask.shape, centers, point_radius=0.0)

    def _load_sample_arrays(
        self,
        record: SampleRecord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Load and preprocess a single sample."""
        image = load_grayscale(record.image_path)
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        loaded_point_label = (
            (load_grayscale(record.point_label_path) > 0).astype(np.float32)
            if record.point_label_path is not None
            else None
        )

        # Build point_map with fallback for robustness
        if loaded_point_label is not None:
            point_map = self._build_point_map(mask, loaded_point_label)
        else:
            # Fallback: compute online (should not happen with proper config)
            print(f"[WARNING] loaded_point_label is None, computing online from full mask")
            centers = component_centers(mask)
            point_map = create_point_map_from_centers(mask.shape, centers, point_radius=0.0)

        inner_prior = (
            (load_grayscale(record.inner_prior_path) > 0).astype(np.float32)
            if record.inner_prior_path is not None
            else None
        )
        outer_prior = (
            (load_grayscale(record.outer_prior_path) > 0).astype(np.float32)
            if record.outer_prior_path is not None
            else None
        )
        return image, mask, point_map, loaded_point_label, inner_prior, outer_prior

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        if self.cached_samples is not None:
            image, mask, point_map, loaded_point_label, inner_prior, outer_prior = self.cached_samples[index]
        else:
            image, mask, point_map, loaded_point_label, inner_prior, outer_prior = self._load_sample_arrays(record)
        supervision_mask = None
        if self.supervision_masks is not None:
            supervision_mask = self.supervision_masks[record.name].astype(np.float32, copy=True)

        if self.train:
            named_arrays: list[tuple[str, np.ndarray]] = [("mask", mask)]
            if point_map is not None:
                named_arrays.append(("point_map", point_map))
            if supervision_mask is not None:
                named_arrays.append(("supervision_mask", supervision_mask))
            if inner_prior is not None:
                named_arrays.append(("inner_prior", inner_prior))
            if outer_prior is not None:
                named_arrays.append(("outer_prior", outer_prior))

            if supervision_mask is not None:
                crop_reference_name = "supervision_mask"
            elif point_map is not None:
                crop_reference_name = "point_map"
            else:
                crop_reference_name = "mask"

            crop_reference = dict(named_arrays)[crop_reference_name]
            arrays = [crop_reference]
            array_names = ["crop_reference"]
            for name, array in named_arrays:
                if array is crop_reference:
                    continue
                arrays.append(array)
                array_names.append(name)
            image, cropped_arrays = crop_arrays(image, arrays, self.crop_size, self.focus_prob, self.rng)
            cropped_map = dict(zip(array_names, cropped_arrays))
            crop_reference = cropped_map["crop_reference"]
            mask = crop_reference if crop_reference_name == "mask" else cropped_map["mask"]
            if supervision_mask is not None:
                supervision_mask = (
                    crop_reference if crop_reference_name == "supervision_mask" else cropped_map["supervision_mask"]
                )
            if point_map is not None:
                point_map = cropped_map["point_map"] if crop_reference_name == "point_map" else crop_reference
            if inner_prior is not None:
                inner_prior = crop_reference if crop_reference_name == "inner_prior" else cropped_map["inner_prior"]
            if outer_prior is not None:
                outer_prior = crop_reference if crop_reference_name == "outer_prior" else cropped_map["outer_prior"]

            if self.rng.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                if point_map is not None:
                    point_map = np.fliplr(point_map).copy()
                if supervision_mask is not None:
                    supervision_mask = np.fliplr(supervision_mask).copy()
                if inner_prior is not None:
                    inner_prior = np.fliplr(inner_prior).copy()
                if outer_prior is not None:
                    outer_prior = np.fliplr(outer_prior).copy()

            rotate_k = self.rng.randrange(4)
            if rotate_k:
                image = np.rot90(image, rotate_k).copy()
                mask = np.rot90(mask, rotate_k).copy()
                if point_map is not None:
                    point_map = np.rot90(point_map, rotate_k).copy()
                if supervision_mask is not None:
                    supervision_mask = np.rot90(supervision_mask, rotate_k).copy()
                if inner_prior is not None:
                    inner_prior = np.rot90(inner_prior, rotate_k).copy()
                if outer_prior is not None:
                    outer_prior = np.rot90(outer_prior, rotate_k).copy()

        elif self.cached_samples is None:
            # Rebuild point_map only if needed
            if loaded_point_label is not None:
                point_map = self._build_point_map(mask, loaded_point_label)
            else:
                # Fallback: compute online (should not happen with proper config)
                print(f"[WARNING] loaded_point_label is None, computing online from full mask")
                centers = component_centers(mask)
                point_map = create_point_map_from_centers(mask.shape, centers, point_radius=0.0)

        image_tensor = torch.from_numpy((image - self.img_mean) / self.img_std).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # Build point_map with fallback for robustness
        if loaded_point_label is not None:
            point_tensor = torch.from_numpy(self._build_point_map(mask, loaded_point_label)).unsqueeze(0).float()
        else:
            # Fallback: compute online (should not happen with proper config)
            print(f"[WARNING] loaded_point_label is None, computing online from full mask")
            centers = component_centers(mask)
            point_map = create_point_map_from_centers(mask.shape, centers, point_radius=0.0)
            point_tensor = torch.from_numpy(point_map).unsqueeze(0).float()

        inner_prior_tensor = torch.from_numpy(
            inner_prior if inner_prior is not None else np.zeros_like(mask, dtype=np.float32)
        ).unsqueeze(0).float()
        outer_prior_tensor = torch.from_numpy(
            outer_prior if outer_prior is not None else np.ones_like(mask, dtype=np.float32)
        ).unsqueeze(0).float()
        supervision_mask_tensor = torch.from_numpy(
            supervision_mask if supervision_mask is not None else np.zeros_like(mask, dtype=np.float32)
        ).unsqueeze(0).float()

        return {
            "name": record.name,
            "image": image_tensor,
            "mask": mask_tensor,
            "point": point_tensor,
            "supervision_mask": supervision_mask_tensor,
            "inner_prior": inner_prior_tensor,
            "outer_prior": outer_prior_tensor,
        }


def worker_init_fn(worker_id: int) -> None:
    """Re-seed per-worker RNG so augmentations are independent across workers."""
    info = get_worker_info()
    info.dataset.rng = random.Random(info.dataset._seed + worker_id)


__all__ = [
    # Config
    "DatasetConfig",
    "DATASET_REGISTRY",
    "build_dataset_config",
    "validate_dataset_config",
    # Data
    "SampleRecord",
    "IRSTDPointDataset",
    "resolve_records",
    "connected_components",
    "component_centers",
    "create_point_map_from_centers",
    "crop_arrays",
    "worker_init_fn",
]
