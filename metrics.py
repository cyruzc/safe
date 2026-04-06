from __future__ import annotations

import numpy as np
import torch


def require_skimage_measure():
    try:
        from skimage import measure  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-image is required for SAFE Pd/Fa metrics. Install `scikit-image`.") from exc
    return measure


class BasicIRSTDMetrics:
    """
    Metric definition aligned with BasicIRSTD.

    Returned metrics:
    - IoU: global pixel IoU over the whole test set
    - nIoU: sample-wise IoU mean
    - Pd: object-level detection probability via centroid-distance matching
    - Fa: false-alarm pixels normalized by total evaluated pixels
    - Th: fixed evaluation threshold
    """

    def __init__(self, thresholds: list[float] | tuple[float, ...] | None = None) -> None:
        self.thresholds = [float(threshold) for threshold in (thresholds or [0.5])]
        self.reset()

    def reset(self) -> None:
        self.stats_by_threshold: dict[float, dict[str, int | float | list[float]]] = {
            threshold: {
                "total_inter": 0,
                "total_union": 0,
                "total_fa": 0,
                "total_pixels": 0,
                "total_tp": 0,
                "total_gt": 0,
                "sample_ious": [],
            }
            for threshold in self.thresholds
        }

    # -- pixel-level (GPU-friendly, accepts torch tensors) --

    def update_pixel(self, pred_prob: torch.Tensor, gt: torch.Tensor) -> None:
        """Accumulate pixel-level IoU / nIoU from tensors on any device."""
        gt_bin = gt > 0
        for threshold in self.thresholds:
            stats = self.stats_by_threshold[threshold]
            pred_bin = pred_prob > threshold
            inter = (pred_bin & gt_bin).sum().item()
            union = (pred_bin | gt_bin).sum().item()
            stats["total_inter"] += inter
            stats["total_union"] += union
            sample_ious = stats["sample_ious"]
            assert isinstance(sample_ious, list)
            sample_ious.append(inter / union if union > 0 else 0.0)

    # -- object-level (CPU, requires scikit-image) --

    def update_pd_fa(self, pred_prob: np.ndarray, gt_bin: np.ndarray) -> None:
        """Accumulate object-level Pd / Fa from numpy arrays (CPU)."""
        gt = (gt_bin > 0).astype(np.uint8)
        for threshold in self.thresholds:
            stats = self.stats_by_threshold[threshold]
            pred = (pred_prob > threshold).astype(np.uint8)
            tp_count, fa_pix, gt_count = self._calculate_pd_fa(pred, gt)
            stats["total_fa"] += fa_pix
            stats["total_pixels"] += int(pred.size)
            stats["total_tp"] += tp_count
            stats["total_gt"] += gt_count

    @staticmethod
    def _calculate_pd_fa(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[int, int, int]:
        measure = require_skimage_measure()
        pred_regions = list(measure.regionprops(measure.label(pred_bin, connectivity=2)))
        gt_regions = list(measure.regionprops(measure.label(gt_bin, connectivity=2)))

        total_pred_pixels = int(sum(int(region.area) for region in pred_regions))
        matched_pred_pixels = 0
        matched = 0
        matched_pred: set[int] = set()

        for gt_region in gt_regions:
            gt_centroid = np.array(list(gt_region.centroid), dtype=np.float32)
            for idx, pred_region in enumerate(pred_regions):
                if idx in matched_pred:
                    continue
                pred_centroid = np.array(list(pred_region.centroid), dtype=np.float32)
                if np.linalg.norm(pred_centroid - gt_centroid) < 3.0:
                    matched += 1
                    matched_pred_pixels += int(pred_region.area)
                    matched_pred.add(idx)
                    break

        fa_pixels = total_pred_pixels - matched_pred_pixels
        return matched, fa_pixels, len(gt_regions)

    @staticmethod
    def _format_threshold_key(threshold: float) -> str:
        return f"{threshold:g}"

    @classmethod
    def result_keys(
        cls,
        thresholds: list[float] | tuple[float, ...] | None = None,
        include_object_metrics: bool = True,
    ) -> list[str]:
        threshold_list = [float(threshold) for threshold in (thresholds or [0.5])]
        if len(threshold_list) == 1:
            keys = ["IoU", "nIoU"]
            if include_object_metrics:
                keys.extend(["Pd", "Fa"])
            keys.append("Th")
            return keys

        keys: list[str] = []
        for threshold in threshold_list:
            suffix = cls._format_threshold_key(threshold)
            keys.extend([f"IoU@{suffix}", f"nIoU@{suffix}"])
            if include_object_metrics:
                keys.extend([f"Pd@{suffix}", f"Fa@{suffix}"])
        return keys

    @classmethod
    def primary_iou_key(cls, thresholds: list[float] | tuple[float, ...] | None = None) -> str:
        threshold_list = [float(threshold) for threshold in (thresholds or [0.5])]
        if len(threshold_list) == 1:
            return "IoU"
        return f"IoU@{cls._format_threshold_key(threshold_list[0])}"

    def get_results(self, include_object_metrics: bool = True) -> dict[str, float]:
        if len(self.thresholds) == 1:
            threshold = self.thresholds[0]
            stats = self.stats_by_threshold[threshold]
            total_inter = float(stats["total_inter"])
            total_union = float(stats["total_union"])
            sample_ious = stats["sample_ious"]
            assert isinstance(sample_ious, list)
            result = {
                "IoU": (total_inter / total_union * 100.0) if total_union > 0 else 0.0,
                "nIoU": (float(np.mean(sample_ious)) * 100.0) if sample_ious else 0.0,
            }
            if include_object_metrics:
                total_tp = float(stats["total_tp"])
                total_gt = float(stats["total_gt"])
                total_fa = float(stats["total_fa"])
                total_pixels = float(stats["total_pixels"])
                result["Pd"] = (total_tp / total_gt * 100.0) if total_gt > 0 else 0.0
                result["Fa"] = (total_fa / total_pixels) if total_pixels > 0 else 0.0
            result["Th"] = threshold
            return result

        result: dict[str, float] = {}
        for threshold in self.thresholds:
            suffix = self._format_threshold_key(threshold)
            stats = self.stats_by_threshold[threshold]
            total_inter = float(stats["total_inter"])
            total_union = float(stats["total_union"])
            sample_ious = stats["sample_ious"]
            assert isinstance(sample_ious, list)
            result[f"IoU@{suffix}"] = (total_inter / total_union * 100.0) if total_union > 0 else 0.0
            result[f"nIoU@{suffix}"] = (float(np.mean(sample_ious)) * 100.0) if sample_ious else 0.0
            if include_object_metrics:
                total_tp = float(stats["total_tp"])
                total_gt = float(stats["total_gt"])
                total_fa = float(stats["total_fa"])
                total_pixels = float(stats["total_pixels"])
                result[f"Pd@{suffix}"] = (total_tp / total_gt * 100.0) if total_gt > 0 else 0.0
                result[f"Fa@{suffix}"] = (total_fa / total_pixels) if total_pixels > 0 else 0.0
        return result
