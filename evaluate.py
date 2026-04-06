from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional
from torch.amp import autocast
from torch.utils.data import DataLoader

from data import IRSTDPointDataset, build_dataset_config, validate_dataset_config
from metrics import BasicIRSTDMetrics
from models import build_model as build_model_from_registry, get_model_names, prepare_model_input


def parse_args() -> argparse.Namespace:
    model_choices = get_model_names()
    parser = argparse.ArgumentParser(description="Evaluate a trained IRSTD checkpoint")
    parser.add_argument("--dataset-name", type=str, choices=["sirst3", "irstd1k", "nuaa_sirst"], default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-dir-name", type=str, default=None)
    parser.add_argument("--mask-dir-name", type=str, default=None)
    parser.add_argument("--test-image-dir-name", type=str, default=None)
    parser.add_argument("--test-mask-dir-name", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--model-name", type=str, choices=model_choices, default=None)
    parser.add_argument("--eval-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument(
        "--cache-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload images and masks into RAM",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    return parser.parse_args()


def is_cuda_device(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    validate_dataset_config(dataset_config, require_train_split=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_name = args.model_name or config.get("model_name", "lightweight_unet")
    image_dir_name = args.image_dir_name or config.get("test_image_dir_name") or dataset_config.test_image_dir
    mask_dir_name = args.mask_dir_name or config.get("test_mask_dir_name") or dataset_config.test_mask_dir

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this execution environment.")
    model = build_model_from_registry(model_name).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    use_amp = bool(args.amp and device.type == "cuda")

    dataset = IRSTDPointDataset(
        dataset_root=dataset_config.root,
        split_name=args.test_split or dataset_config.test_split,
        image_dir_name=args.test_image_dir_name or image_dir_name,
        mask_dir_name=args.test_mask_dir_name or mask_dir_name,
        train=False,
        cache_data=args.cache_data,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=is_cuda_device(device))

    metrics = BasicIRSTDMetrics(thresholds=args.eval_thresholds)
    with torch.no_grad():
        for batch in loader:
            raw_image = batch["image"].to(device, non_blocking=True)
            orig_h, orig_w = raw_image.shape[-2], raw_image.shape[-1]
            image = prepare_model_input(raw_image, model_name)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
            prob = torch.sigmoid(logits)

            # Ensure output matches original input size
            if prob.shape[-2:] != (orig_h, orig_w):
                prob = torch.nn.functional.interpolate(
                    prob, size=(orig_h, orig_w), mode='bilinear', align_corners=False
                )

            # Get ground truth and ensure it matches prediction size
            mask_tensor = batch["mask"].to(device, non_blocking=True)
            if mask_tensor.shape[-2:] != (orig_h, orig_w):
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor, size=(orig_h, orig_w), mode='nearest'
                )

            metrics.update_pixel(prob, mask_tensor)
            metrics.update_pd_fa(prob[0, 0].cpu().numpy(), mask_tensor[0, 0].cpu().numpy())

    result = metrics.get_results()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
