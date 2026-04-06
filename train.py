from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data import IRSTDPointDataset, worker_init_fn, build_dataset_config, validate_dataset_config
from experiment_paths import build_run_output_dir, resolve_supervision_tag
from metrics import BasicIRSTDMetrics
from losses import build_criterion, resolve_method_name
from models import build_model as build_model_from_registry, get_default_channels, get_model_names, prepare_model_input


def parse_args() -> argparse.Namespace:
    model_choices = get_model_names()
    parser = argparse.ArgumentParser(description="Train SAFE and point/full IRSTD models")
    parser.add_argument("--dataset-name", type=str, choices=["sirst3", "irstd1k", "nuaa_sirst", "nudt_sirst"], default=None)
    parser.add_argument("--label-mode", type=str, choices=["centroid", "coarse"], default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit run directory.")
    parser.add_argument("--experiments-root", type=str, default="experiments", help="Root directory for auto-generated experiment paths.")
    parser.add_argument("--exp-tag", type=str, default="base", help="Short tag appended to the auto-generated run directory.")
    parser.add_argument("--model-name", type=str, choices=model_choices, default="lightweight_unet")
    parser.add_argument(
        "--method",
        type=str,
        choices=["point", "safe", "full"],
        default="point",
        help="Training method.",
    )
    parser.add_argument("--image-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--mask-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-image-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-mask-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--train-split", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-split", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--point-label-dir", type=str, default=None, help="Advanced override. Prefer --dataset-name with --label-mode.")
    parser.add_argument(
        "--use-loaded-point-labels",
        action="store_true",
        help="Use weak point/coarse labels from --point-label-dir directly instead of generating point maps from full masks.",
    )
    parser.add_argument("--inner-prior-dir", type=str, default=None)
    parser.add_argument("--outer-prior-dir", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--point-sigma", type=float, default=2.0)
    parser.add_argument("--focus-prob", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload images and masks into RAM",
    )
    parser.add_argument("--eval-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--full-pos-weight", type=float, default=1.0)
    parser.add_argument("--inner-loss-weight", type=float, default=0.0)
    parser.add_argument("--outer-loss-weight", type=float, default=0.0)
    parser.add_argument("--prior-warmup-epochs", type=int, default=0)
    parser.add_argument("--inner-decay-start-epoch", type=int, default=None)
    parser.add_argument("--inner-decay-end-epoch", type=int, default=None)
    parser.add_argument("--outer-boost-start-epoch", type=int, default=None)
    parser.add_argument("--outer-boost-end-epoch", type=int, default=None)
    parser.add_argument("--outer-boost-scale", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="none", help="LR scheduler.")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run IoU validation every N epochs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    return parser.parse_args()


def is_cuda_device(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


def should_run(epoch: int, interval: int) -> bool:
    return interval > 0 and epoch % interval == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    args: argparse.Namespace,
    dataset_config,
    split_name: str,
    train: bool,
    supervision_masks: dict[str, np.ndarray] | None = None,
    include_names: set[str] | None = None,
    include_point_labels: bool = False,
) -> DataLoader:
    image_dir_name = dataset_config.train_image_dir if train else dataset_config.test_image_dir
    mask_dir_name = dataset_config.train_mask_dir if train else dataset_config.test_mask_dir
    point_label_dir = None
    if args.method != "full" and (train or include_point_labels):
        point_label_dir = dataset_config.point_label_dir_for(args.label_mode) or args.point_label_dir
    dataset = IRSTDPointDataset(
        dataset_root=dataset_config.root,
        split_name=split_name,
        image_dir_name=image_dir_name,
        mask_dir_name=mask_dir_name,
        point_label_dir=point_label_dir,
        inner_prior_dir=args.inner_prior_dir if train else None,
        outer_prior_dir=args.outer_prior_dir if train else None,
        crop_size=args.crop_size,
        point_sigma=args.point_sigma,
        use_loaded_point_labels=(args.use_loaded_point_labels and args.method != "full" and (train or include_point_labels)),
        train=train,
        focus_prob=args.focus_prob,
        seed=args.seed,
        cache_data=args.cache_data,
        supervision_masks=supervision_masks,
        include_names=include_names,
        img_mean=dataset_config.img_mean,
        img_std=dataset_config.img_std,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else 1,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device(args.device),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )


@torch.no_grad()
def evaluate(
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
    for batch in loader:
        raw_image = batch["image"].to(device, non_blocking=True)
        orig_h, orig_w = raw_image.shape[-2], raw_image.shape[-1]
        image = prepare_model_input(raw_image, model_name)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
        prob = torch.sigmoid(logits)
        # Crop back to original size if padding was applied
        if prob.shape[-2] != orig_h or prob.shape[-1] != orig_w:
            prob = prob[:, :, :orig_h, :orig_w]

        mask_tensor = batch["mask"]
        metrics.update_pixel(prob, mask_tensor.to(device, non_blocking=True))
        if compute_object_metrics:
            metrics.update_pd_fa(prob[0, 0].cpu().numpy(), mask_tensor[0, 0].numpy())
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
    model.eval()
    total_inter = 0.0
    total_union = 0.0
    for batch in loader:
        raw_image = batch["image"].to(device, non_blocking=True)
        orig_h, orig_w = raw_image.shape[-2], raw_image.shape[-1]
        image = prepare_model_input(raw_image, model_name)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
        prob = torch.sigmoid(logits)
        if prob.shape[-2] != orig_h or prob.shape[-1] != orig_w:
            prob = prob[:, :, :orig_h, :orig_w]

        pred_bin = prob > threshold
        gt_bin = batch["mask"].to(device, non_blocking=True) > 0
        total_inter += float((pred_bin & gt_bin).sum().item())
        total_union += float((pred_bin | gt_bin).sum().item())
    return (total_inter / total_union * 100.0) if total_union > 0 else 0.0


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    epoch: int,
    prior_warmup_epochs: int,
    model_name: str,
) -> dict[str, float]:
    model.train()
    totals = {
        "train_loss": 0.0,
        "point_loss": 0.0,
        "inner_loss": 0.0,
        "outer_loss": 0.0,
    }
    use_prior_terms = epoch > prior_warmup_epochs
    inner_weight_scale = get_inner_weight_scale(criterion, epoch, prior_warmup_epochs)
    outer_weight_scale = get_outer_weight_scale(criterion, epoch, prior_warmup_epochs)
    for batch in loader:
        image = prepare_model_input(batch["image"].to(device, non_blocking=True), model_name)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            if hasattr(criterion, "point_loss"):
                point = batch["point"].to(device, non_blocking=True)
                inner_prior = batch["inner_prior"].to(device, non_blocking=True)
                outer_prior = batch["outer_prior"].to(device, non_blocking=True)
                loss, stats = criterion(
                    logits,
                    point,
                    inner_prior,
                    outer_prior,
                    inner_weight_scale=inner_weight_scale,
                    outer_weight_scale=outer_weight_scale,
                    enable_inner=use_prior_terms,
                    enable_outer=use_prior_terms,
                )
            else:
                mask = batch["mask"].to(device, non_blocking=True)
                loss = criterion(logits, mask)
                stats = {
                    "point_loss": float(loss.detach().item()),
                    "inner_loss": 0.0,
                    "outer_loss": 0.0,
                }
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        totals["train_loss"] += float(loss.item())
        totals["point_loss"] += stats["point_loss"]
        totals["inner_loss"] += stats["inner_loss"]
        totals["outer_loss"] += stats["outer_loss"]

    denom = max(len(loader), 1)
    return {key: value / denom for key, value in totals.items()}


def get_inner_weight_scale(
    criterion,
    epoch: int,
    prior_warmup_epochs: int,
) -> float:
    schedule = getattr(criterion, "inner_decay_schedule", None)
    if schedule is None:
        return 1.0
    start_epoch, end_epoch = schedule
    if epoch <= max(prior_warmup_epochs, start_epoch):
        return 1.0
    if epoch >= end_epoch:
        return 0.0
    span = max(end_epoch - start_epoch, 1)
    progress = (epoch - start_epoch) / span
    return max(0.0, 1.0 - progress)


def get_outer_weight_scale(
    criterion,
    epoch: int,
    prior_warmup_epochs: int,
) -> float:
    schedule = getattr(criterion, "outer_boost_schedule", None)
    if schedule is None:
        return 1.0
    start_epoch, end_epoch, boost_scale = schedule
    if epoch <= max(prior_warmup_epochs, start_epoch):
        return 1.0
    if epoch >= end_epoch:
        return boost_scale
    span = max(end_epoch - start_epoch, 1)
    progress = (epoch - start_epoch) / span
    return 1.0 + (boost_scale - 1.0) * progress


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    set_seed(args.seed)
    method_name = resolve_method_name(args)
    validate_dataset_config(dataset_config, require_train_split=True)
    if args.inner_loss_weight > 0 and args.inner_prior_dir is None:
        raise ValueError("--inner-prior-dir is required when --inner-loss-weight > 0.")
    if args.outer_loss_weight > 0 and args.outer_prior_dir is None:
        raise ValueError("--outer-prior-dir is required when --outer-loss-weight > 0.")
    point_label_dir = dataset_config.point_label_dir_for(args.label_mode) or args.point_label_dir
    if args.use_loaded_point_labels and point_label_dir is None:
        raise ValueError("Point supervision with loaded weak labels requires a point-label directory or dataset label mode.")
    if args.method == "full" and args.label_mode is not None:
        raise ValueError("--label-mode is not used with --method full.")
    if args.method == "full" and (args.inner_loss_weight > 0 or args.outer_loss_weight > 0):
        raise ValueError("Tri-zone prior losses are only supported with --method point or --method safe.")
    if args.method == "full" and args.use_loaded_point_labels:
        raise ValueError("--use-loaded-point-labels is only supported with --method point or --method safe.")
    if args.label_mode is not None and args.method == "full":
        raise ValueError("--label-mode is only supported with --method point or --method safe.")
    if (args.inner_decay_start_epoch is None) != (args.inner_decay_end_epoch is None):
        raise ValueError("--inner-decay-start-epoch and --inner-decay-end-epoch must be set together.")
    if args.inner_decay_start_epoch is not None and args.inner_decay_end_epoch <= args.inner_decay_start_epoch:
        raise ValueError("--inner-decay-end-epoch must be greater than --inner-decay-start-epoch.")
    if (args.outer_boost_start_epoch is None) != (args.outer_boost_end_epoch is None):
        raise ValueError("--outer-boost-start-epoch and --outer-boost-end-epoch must be set together.")
    if args.outer_boost_start_epoch is not None and args.outer_boost_end_epoch <= args.outer_boost_start_epoch:
        raise ValueError("--outer-boost-end-epoch must be greater than --outer-boost-start-epoch.")
    if args.outer_boost_scale < 1.0:
        raise ValueError("--outer-boost-scale must be >= 1.0.")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be >= 1.")
    supervision_tag = resolve_supervision_tag(args.method, args.label_mode)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else build_run_output_dir(
            args.experiments_root,
            dataset_config.name,
            args.model_name,
            supervision_tag,
            args.exp_tag,
            args.seed,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_channels = get_default_channels(args.model_name)
    config_payload = vars(args).copy()
    config_payload["dataset_root"] = str(dataset_config.root)
    config_payload["image_dir_name"] = dataset_config.train_image_dir
    config_payload["mask_dir_name"] = dataset_config.train_mask_dir
    config_payload["test_image_dir_name"] = dataset_config.test_image_dir
    config_payload["test_mask_dir_name"] = dataset_config.test_mask_dir
    config_payload["train_split"] = dataset_config.train_split
    config_payload["test_split"] = dataset_config.test_split
    config_payload["resolved_point_label_dir"] = point_label_dir
    config_payload["resolved_model_channels"] = list(resolved_channels) if resolved_channels is not None else None
    config_payload["resolved_method_name"] = method_name
    config_payload["resolved_supervision_tag"] = supervision_tag
    config_payload["resolved_output_dir"] = str(output_dir.resolve())
    save_json(output_dir / "config.json", config_payload)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this execution environment.")
        torch.backends.cudnn.benchmark = True
    train_loader = make_loader(
        args,
        dataset_config,
        dataset_config.train_split,
        train=True,
    )
    test_loader = make_loader(args, dataset_config, dataset_config.test_split, train=False)

    model = build_model_from_registry(args.model_name).to(device)
    method_bundle = build_criterion(args, method_name)
    criterion = method_bundle.criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        config_payload["lr_scheduler"] = args.scheduler
        config_payload["eta_min"] = args.eta_min
        save_json(output_dir / "config.json", config_payload)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_iou = -1.0
    primary_iou_key = BasicIRSTDMetrics.primary_iou_key(args.eval_thresholds)
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "point_loss", "inner_loss", "outer_loss", primary_iou_key],
        )
        writer.writeheader()

        print(
            f"device={device.type} amp={use_amp} dataset={dataset_config.name} "
            f"model={args.model_name} method={method_name} seed={args.seed}"
        )
        for epoch in range(1, args.epochs + 1):
            print(f"epoch={epoch:03d} start", flush=True)
            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                use_amp,
                epoch=epoch,
                prior_warmup_epochs=args.prior_warmup_epochs,
                model_name=args.model_name,
            )
            row = {fieldname: "" for fieldname in writer.fieldnames if fieldname is not None}
            row.update({"epoch": epoch, **train_stats})
            ran_eval = should_run(epoch, args.eval_every) or epoch == args.epochs
            if ran_eval:
                epoch_iou = evaluate_iou_only(
                    model,
                    test_loader,
                    device,
                    threshold=args.eval_thresholds[0],
                    use_amp=use_amp,
                    model_name=args.model_name,
                )
                row[primary_iou_key] = epoch_iou
            writer.writerow(row)
            file.flush()

            checkpoint = {
                "model": model.state_dict(),
                "channels": list(resolved_channels) if resolved_channels is not None else None,
                "epoch": epoch,
                "metrics": ({primary_iou_key: epoch_iou} if ran_eval else {}),
                "config": config_payload,
            }
            torch.save(checkpoint, output_dir / "last.pt")

            if ran_eval and epoch_iou > best_iou:
                best_iou = epoch_iou
                torch.save(checkpoint, output_dir / "best_iou.pt")

            message = (
                f"epoch={epoch:03d} "
                f"loss={train_stats['train_loss']:.4f} "
                f"point={train_stats['point_loss']:.4f} "
                f"inner={train_stats['inner_loss']:.4f} "
                f"outer={train_stats['outer_loss']:.4f}"
            )
            if ran_eval:
                message += f" {primary_iou_key}={epoch_iou:.2f}"
            else:
                message += " val=skipped"
            print(message, flush=True)

            if scheduler is not None:
                scheduler.step()

    best_checkpoint_path = output_dir / "best_iou.pt"
    if not best_checkpoint_path.exists():
        raise RuntimeError("Training finished without producing best_iou.pt. Check --eval-every.")

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    final_metrics = evaluate(
        model,
        test_loader,
        device,
        args.eval_thresholds,
        use_amp,
        args.model_name,
        compute_object_metrics=True,
    )
    save_json(output_dir / "final_metrics.json", final_metrics)
    best_checkpoint["metrics"] = final_metrics
    torch.save(best_checkpoint, best_checkpoint_path)
    print(f"final_best {json.dumps(final_metrics, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
