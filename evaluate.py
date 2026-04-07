from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import IRSTDPointDataset
from dataset_config import build_dataset_config, validate_dataset_config
from engine import evaluate_model, is_cuda_device
from models import build_model as build_model_from_registry, get_model_names


def parse_args() -> argparse.Namespace:
    model_choices = get_model_names()
    parser = argparse.ArgumentParser(description="Evaluate a trained IRSTD checkpoint")
    parser.add_argument("--dataset-name", type=str, choices=["sirst3", "irstd1k", "nuaa_sirst", "nudt_sirst"], default=None)
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
        "--allow-nonfull-eval-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow evaluation against a mask directory other than the dataset full masks.",
    )
    parser.add_argument(
        "--cache-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload images and masks into RAM",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision on CUDA",
    )
    return parser.parse_args()


def resolve_eval_mask_dir(args: argparse.Namespace, dataset_config, config: dict) -> str:
    mask_dir_name = args.mask_dir_name or config.get("test_mask_dir_name") or dataset_config.test_mask_dir
    eval_mask_dir = args.test_mask_dir_name or mask_dir_name
    if not args.allow_nonfull_eval_mask and eval_mask_dir != dataset_config.test_mask_dir:
        raise ValueError(
            f"Evaluation must use the dataset full masks ('{dataset_config.test_mask_dir}'). "
            f"Got '{eval_mask_dir}'. Use --allow-nonfull-eval-mask only for non-paper debugging."
        )
    return eval_mask_dir


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    validate_dataset_config(dataset_config, require_train_split=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_name = args.model_name or config.get("model_name", "lightweight_unet")
    image_dir_name = args.image_dir_name or config.get("test_image_dir_name") or dataset_config.test_image_dir
    mask_dir_name = resolve_eval_mask_dir(args, dataset_config, config)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this execution environment.")
    model = build_model_from_registry(model_name).to(device)
    model.load_state_dict(checkpoint["model"])
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

    result = evaluate_model(
        model,
        loader,
        device,
        args.eval_thresholds,
        use_amp,
        model_name,
        compute_object_metrics=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
