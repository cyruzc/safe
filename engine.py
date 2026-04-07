from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from metrics import BasicIRSTDMetrics
from models import prepare_model_input


def is_cuda_device(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


@torch.no_grad()
def predict_prob_and_mask(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | str],
    device: torch.device,
    use_amp: bool,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_image = batch["image"].to(device, non_blocking=True)
    orig_h, orig_w = raw_image.shape[-2], raw_image.shape[-1]
    image = prepare_model_input(raw_image, model_name)
    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(image)
    prob = torch.sigmoid(logits)
    if prob.shape[-2:] != (orig_h, orig_w):
        prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    mask_tensor = batch["mask"].to(device, non_blocking=True)
    if mask_tensor.shape[-2:] != (orig_h, orig_w):
        mask_tensor = F.interpolate(mask_tensor, size=(orig_h, orig_w), mode="nearest")
    return prob, mask_tensor


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    use_amp: bool,
    model_name: str,
    compute_object_metrics: bool = True,
) -> dict[str, float]:
    model.eval()
    metrics = BasicIRSTDMetrics(thresholds=thresholds)
    for batch_idx, batch in enumerate(loader):
        prob, mask_tensor = predict_prob_and_mask(model, batch, device, use_amp, model_name)
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            print(f"Warning: Invalid predictions in batch {batch_idx}, skipping...")
            continue

        metrics.update_pixel(prob, mask_tensor)
        if compute_object_metrics:
            for sample_idx in range(prob.shape[0]):
                metrics.update_pd_fa(
                    prob[sample_idx, 0].cpu().numpy(),
                    mask_tensor[sample_idx, 0].cpu().numpy(),
                )
    return metrics.get_results(include_object_metrics=compute_object_metrics)


@torch.no_grad()
def evaluate_iou_only(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    model_name: str,
) -> float:
    result = evaluate_model(
        model,
        loader,
        device,
        [threshold],
        use_amp,
        model_name,
        compute_object_metrics=False,
    )
    return min(result["IoU"], 100.0)
