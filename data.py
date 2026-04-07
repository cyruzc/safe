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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from dataset_config import DATASET_REGISTRY, DatasetConfig, build_dataset_config, validate_dataset_config
from utils import load_grayscale, load_split_file


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
        train: bool = True,
        focus_prob: float = 0.7,
        seed: int = 42,
        cache_data: bool = False,
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
        self.records = records
        self.crop_size = crop_size
        self.train = train
        self.focus_prob = focus_prob
        self._seed = seed
        self.rng = random.Random(seed)
        self.cache_data = cache_data
        self.cached_samples: list[
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]
        ] | None = None
        if self.cache_data:
            self.cached_samples = [
                self._load_sample_arrays(record)
                for record in self.records
            ]

    def __len__(self) -> int:
        return len(self.records)

    def _build_point_map(self, mask: np.ndarray, loaded_point_label: np.ndarray | None) -> np.ndarray:
        if loaded_point_label is not None:
            return loaded_point_label.astype(np.float32, copy=False)
        return np.zeros_like(mask, dtype=np.float32)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).unsqueeze(0).float()

    @staticmethod
    def _apply_transform_to_named_arrays(
        named_arrays: dict[str, np.ndarray | None],
        transform,
    ) -> dict[str, np.ndarray | None]:
        return {
            name: (transform(array).copy() if array is not None else None)
            for name, array in named_arrays.items()
        }

    def _apply_train_transforms(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        point_map: np.ndarray | None,
        inner_prior: np.ndarray | None,
        outer_prior: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        crop_reference = point_map if point_map is not None and point_map.any() else mask
        named_arrays: dict[str, np.ndarray | None] = {
            "mask": mask,
            "point_map": point_map,
            "inner_prior": inner_prior,
            "outer_prior": outer_prior,
        }
        ordered_items = [("crop_reference", crop_reference)]
        for name, array in named_arrays.items():
            if array is None or array is crop_reference:
                continue
            ordered_items.append((name, array))
        ordered_names = [name for name, _ in ordered_items]
        ordered_arrays = [array for _, array in ordered_items]
        image, cropped_arrays = crop_arrays(image, ordered_arrays, self.crop_size, self.focus_prob, self.rng)
        cropped_map = dict(zip(ordered_names, cropped_arrays))
        transformed = {
            "mask": cropped_map["crop_reference"] if crop_reference is mask else cropped_map["mask"],
            "point_map": cropped_map["crop_reference"] if crop_reference is point_map else cropped_map.get("point_map"),
            "inner_prior": cropped_map["crop_reference"] if crop_reference is inner_prior else cropped_map.get("inner_prior"),
            "outer_prior": cropped_map["crop_reference"] if crop_reference is outer_prior else cropped_map.get("outer_prior"),
        }

        if self.rng.random() < 0.5:
            image = np.fliplr(image).copy()
            transformed = self._apply_transform_to_named_arrays(transformed, np.fliplr)

        rotate_k = self.rng.randrange(4)
        if rotate_k:
            image = np.rot90(image, rotate_k).copy()
            transformed = self._apply_transform_to_named_arrays(
                transformed,
                lambda array: np.rot90(array, rotate_k),
            )

        return (
            image,
            transformed["mask"],
            transformed["point_map"],
            transformed["inner_prior"],
            transformed["outer_prior"],
        )

    def _load_sample_arrays(
        self,
        record: SampleRecord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Load and preprocess a single sample."""
        image = load_grayscale(record.image_path)
        mask = (load_grayscale(record.mask_path) > 0).astype(np.float32)
        loaded_point_label = (
            (load_grayscale(record.point_label_path) > 0).astype(np.float32)
            if record.point_label_path is not None
            else None
        )
        point_map = self._build_point_map(mask, loaded_point_label)

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
        return image, mask, point_map, inner_prior, outer_prior

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        if self.cached_samples is not None:
            image, mask, point_map, inner_prior, outer_prior = self.cached_samples[index]
        else:
            image, mask, point_map, inner_prior, outer_prior = self._load_sample_arrays(record)

        if self.train:
            image, mask, point_map, inner_prior, outer_prior = self._apply_train_transforms(
                image,
                mask,
                point_map,
                inner_prior,
                outer_prior,
            )

        image_tensor = self._to_tensor((image - self.img_mean) / self.img_std)
        mask_tensor = self._to_tensor(mask)
        point_tensor = self._to_tensor(
            point_map if point_map is not None else np.zeros_like(mask, dtype=np.float32)
        )
        inner_prior_tensor = self._to_tensor(
            inner_prior if inner_prior is not None else np.zeros_like(mask, dtype=np.float32)
        )
        outer_prior_tensor = self._to_tensor(
            outer_prior if outer_prior is not None else np.ones_like(mask, dtype=np.float32)
        )

        return {
            "name": record.name,
            "image": image_tensor,
            "mask": mask_tensor,
            "point": point_tensor,
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
    "crop_arrays",
    "worker_init_fn",
]
