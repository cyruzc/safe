from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data import IRSTDPointDataset, worker_init_fn
from dataset_config import build_dataset_config, validate_dataset_config
from engine import evaluate_iou_only, evaluate_model, is_cuda_device
from experiment_paths import build_run_output_dir, resolve_supervision_tag
from metrics import BasicIRSTDMetrics
from losses import PointSupervisionLoss, TriZonePartialLoss, build_criterion, resolve_method_name
from models import build_model as build_model_from_registry, get_default_channels, get_model_names, prepare_model_input


def parse_args() -> argparse.Namespace:
    model_choices = get_model_names()
    parser = argparse.ArgumentParser(
        description="Train SAFE (Self-Adaptive prior-enhanced weak supervision), point, and full IRSTD models"
    )
    parser.add_argument("--dataset-name", type=str, choices=["sirst3", "irstd1k", "nuaa_sirst", "nudt_sirst"], default=None)
    parser.add_argument("--label-mode", type=str, choices=["full", "centroid", "coarse"], default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit run directory.")
    parser.add_argument("--experiments-root", type=str, default="experiments", help="Root directory for auto-generated experiment paths.")
    parser.add_argument("--exp-tag", type=str, default="base", help="Short tag appended to the auto-generated run directory.")
    parser.add_argument("--model-name", type=str, choices=model_choices, default="lightweight_unet")
    parser.add_argument(
        "--method",
        type=str,
        choices=["none", "safe"],
        default="none",
        help="Training method: none (baseline), safe (tri-zone priors). Future: lesps, pal.",
    )
    parser.add_argument("--image-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--mask-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-image-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-mask-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--train-split", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-split", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--point-label-dir", type=str, default=None, help="Advanced override. Prefer --dataset-name with --label-mode.")
    parser.add_argument("--inner-prior-dir", type=str, default=None)
    parser.add_argument("--outer-prior-dir", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--focus-prob", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)  # LESPS-validated learning rate
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
    parser.add_argument("--loss-type", type=str, choices=["bce", "focal"], default="focal", help="Loss function type: bce (weighted BCE) or focal (LESPS-style)")
    parser.add_argument("--inner-loss-weight", type=float, default=0.0)
    parser.add_argument("--outer-loss-weight", type=float, default=0.0)
    parser.add_argument("--hn-loss-weight", type=float, default=0.0)
    parser.add_argument("--tv-loss-weight", type=float, default=0.0)
    parser.add_argument("--tau-in", type=float, default=0.6)
    parser.add_argument("--tau-out", type=float, default=0.1)
    parser.add_argument("--hard-negative-topk", type=int, default=64)
    parser.add_argument("--prior-warmup-epochs", type=int, default=0)
    parser.add_argument("--inner-decay-start-epoch", type=int, default=None)
    parser.add_argument("--inner-decay-end-epoch", type=int, default=None)
    parser.add_argument("--outer-boost-start-epoch", type=int, default=None)
    parser.add_argument("--outer-boost-end-epoch", type=int, default=None)
    parser.add_argument("--outer-boost-scale", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="cosine", help="LR scheduler.")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run IoU validation every N epochs.")
    parser.add_argument(
        "--allow-nonfull-eval-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow validation against a mask directory other than the dataset full masks.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer deterministic CUDA behavior for reproducible experiments.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision on CUDA",
    )
    return parser.parse_args()

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
) -> DataLoader:
    image_dir_name = dataset_config.train_image_dir if train else dataset_config.test_image_dir
    mask_dir_name = dataset_config.train_mask_dir if train else dataset_config.test_mask_dir

    point_label_dir = None
    if train and args.label_mode in {"centroid", "coarse"}:
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
        train=train,
        focus_prob=args.focus_prob,
        seed=args.seed,
        cache_data=args.cache_data,
        img_mean=dataset_config.img_mean,
        img_std=dataset_config.img_std,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed if train else args.seed + 1)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else 1,
        shuffle=train,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=is_cuda_device(args.device),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

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
        "loss": 0.0,
        "main_loss": 0.0,
        "inner_loss": 0.0,
        "outer_loss": 0.0,
        "inner_mean": 0.0,
        "outer_mean": 0.0,
        "hn_loss": 0.0,
        "tv_loss": 0.0,
        "inner_weighted": 0.0,
        "outer_weighted": 0.0,
        "hn_weighted": 0.0,
        "tv_weighted": 0.0,
    }
    use_prior_terms = epoch > prior_warmup_epochs
    inner_weight_scale = get_inner_weight_scale(criterion, epoch, prior_warmup_epochs)
    outer_weight_scale = get_outer_weight_scale(criterion, epoch, prior_warmup_epochs)
    for batch in loader:
        image = prepare_model_input(batch["image"].to(device, non_blocking=True), model_name)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            if isinstance(criterion, TriZonePartialLoss):
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
                    hn_weight_scale=outer_weight_scale,
                    enable_inner=use_prior_terms,
                    enable_outer=use_prior_terms,
                    enable_hn=use_prior_terms,
                    enable_tv=True,
                )
            elif isinstance(criterion, PointSupervisionLoss):
                point = batch["point"].to(device, non_blocking=True)
                loss = criterion(logits, point)
                stats = {
                    "loss": float(loss.detach().item()),
                    "main_loss": float(loss.detach().item()),
                    "inner_loss": 0.0,
                    "outer_loss": 0.0,
                    "inner_mean": 0.0,
                    "outer_mean": 0.0,
                    "hn_loss": 0.0,
                    "tv_loss": 0.0,
                    "inner_weighted": 0.0,
                    "outer_weighted": 0.0,
                    "hn_weighted": 0.0,
                    "tv_weighted": 0.0,
                }
            else:
                mask = batch["mask"].to(device, non_blocking=True)
                loss = criterion(logits, mask)
                stats = {
                    "loss": float(loss.detach().item()),
                    "main_loss": float(loss.detach().item()),
                    "inner_loss": 0.0,
                    "outer_loss": 0.0,
                    "inner_mean": 0.0,
                    "outer_mean": 0.0,
                    "hn_loss": 0.0,
                    "tv_loss": 0.0,
                    "inner_weighted": 0.0,
                    "outer_weighted": 0.0,
                    "hn_weighted": 0.0,
                    "tv_weighted": 0.0,
                }
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss components
        totals["loss"] += stats["loss"]
        totals["main_loss"] += stats["main_loss"]
        totals["inner_loss"] += stats["inner_loss"]
        totals["outer_loss"] += stats["outer_loss"]
        totals["inner_mean"] += stats["inner_mean"]
        totals["outer_mean"] += stats["outer_mean"]
        totals["hn_loss"] += stats["hn_loss"]
        totals["tv_loss"] += stats["tv_loss"]
        totals["inner_weighted"] += stats["inner_weighted"]
        totals["outer_weighted"] += stats["outer_weighted"]
        totals["hn_weighted"] += stats["hn_weighted"]
        totals["tv_weighted"] += stats["tv_weighted"]

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


def resolve_dataset_relative_path(path_like: str | Path, dataset_root: Path) -> Path:
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj.resolve()
    cwd_candidate = path_obj.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (dataset_root / path_obj).resolve()


def resolve_point_label_dir(args: argparse.Namespace, dataset_config) -> str | None:
    if args.label_mode in {"centroid", "coarse"}:
        return dataset_config.point_label_dir_for(args.label_mode) or args.point_label_dir
    return None


def resolve_eval_mask_dir(args: argparse.Namespace, dataset_config) -> str:
    eval_mask_dir = args.test_mask_dir_name or dataset_config.test_mask_dir
    if not args.allow_nonfull_eval_mask and eval_mask_dir != dataset_config.test_mask_dir:
        raise ValueError(
            f"Evaluation must use the dataset full masks ('{dataset_config.test_mask_dir}'). "
            f"Got '{eval_mask_dir}'. Use --allow-nonfull-eval-mask only for non-paper debugging."
        )
    return eval_mask_dir


def load_prior_manifest(prior_dir: str, expected_key: str) -> dict:
    prior_path = Path(prior_dir).resolve()
    if not prior_path.exists():
        raise FileNotFoundError(f"Prior directory does not exist: {prior_path}")
    try:
        manifest_path = prior_path.parents[1] / "manifest.json"
    except IndexError as exc:
        raise ValueError(f"Prior directory has unexpected layout: {prior_path}") from exc
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing prior manifest for {expected_key}: expected {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_prior_dir = Path(manifest.get(expected_key, "")).resolve()
    if resolved_prior_dir != prior_path:
        raise ValueError(
            f"Prior manifest mismatch for {expected_key}: expected {prior_path}, manifest recorded {resolved_prior_dir}"
        )
    return manifest


def infer_prior_context(prior_dir: str) -> dict[str, Path | str]:
    prior_path = Path(prior_dir).resolve()
    try:
        output_root = prior_path.parents[1].resolve()
        label_mode = prior_path.parents[2].name
        dataset_name = prior_path.parents[3].name
    except IndexError as exc:
        raise ValueError(f"Prior directory has unexpected layout: {prior_path}") from exc
    return {
        "output_root": output_root,
        "label_mode": label_mode,
        "dataset_name": dataset_name,
    }


def validate_safe_priors(
    args: argparse.Namespace,
    dataset_config,
    point_label_dir: str,
) -> None:
    manifests: list[tuple[dict, str]] = []
    if args.inner_loss_weight > 0:
        manifests.append((load_prior_manifest(args.inner_prior_dir, "inner_prior_dir"), args.inner_prior_dir))
    if args.outer_loss_weight > 0:
        manifests.append((load_prior_manifest(args.outer_prior_dir, "outer_prior_dir"), args.outer_prior_dir))
    if not manifests:
        return

    expected_anchor_dir = resolve_dataset_relative_path(point_label_dir, dataset_config.root)
    first_output_root: Path | None = None
    for manifest, prior_dir in manifests:
        inferred = infer_prior_context(prior_dir)
        manifest_dataset_name = manifest.get("dataset_name") or inferred["dataset_name"]
        if manifest_dataset_name != dataset_config.name:
            raise ValueError(
                f"SAFE prior dataset mismatch: expected '{dataset_config.name}', got '{manifest_dataset_name}'."
            )
        manifest_label_mode = manifest.get("label_mode") or inferred["label_mode"]
        if manifest_label_mode != args.label_mode:
            raise ValueError(
                f"SAFE prior label-mode mismatch: expected '{args.label_mode}', got '{manifest_label_mode}'."
            )
        manifest_anchor_dir = manifest.get("anchor_label_dir")
        resolved_anchor_dir = (
            expected_anchor_dir
            if manifest_anchor_dir is None
            else resolve_dataset_relative_path(manifest_anchor_dir, dataset_config.root)
        )
        if resolved_anchor_dir != expected_anchor_dir:
            raise ValueError(
                f"SAFE prior anchor mismatch: expected anchor dir '{expected_anchor_dir}', got '{resolved_anchor_dir}'."
            )
        output_root_raw = manifest.get("output_root")
        resolved_output_root = (
            inferred["output_root"]
            if output_root_raw is None
            else Path(output_root_raw).resolve()
        )
        if first_output_root is None:
            first_output_root = resolved_output_root
        if resolved_output_root != first_output_root:
            raise ValueError("Inner and outer SAFE priors must come from the same prior-generation run.")


def validate_training_args(args: argparse.Namespace, dataset_config, point_label_dir: str | None) -> None:
    if args.inner_loss_weight > 0 and args.inner_prior_dir is None:
        raise ValueError("--inner-prior-dir is required when --inner-loss-weight > 0.")
    if (args.outer_loss_weight > 0 or args.hn_loss_weight > 0) and args.outer_prior_dir is None:
        raise ValueError("--outer-prior-dir is required when --outer-loss-weight > 0 or --hn-loss-weight > 0.")
    if args.label_mode is None:
        raise ValueError("--label-mode is required. Choose from: full, centroid, coarse.")
    if args.label_mode in {"centroid", "coarse"} and point_label_dir is None:
        raise ValueError(
            f"--label-mode={args.label_mode} requires pre-generated weak labels "
            "(dataset masks_centroid/masks_coarse or --point-label-dir)."
        )
    if args.label_mode == "full":
        if args.method != "none":
            raise ValueError("--label-mode=full only supports --method=none (full supervision baseline).")
        if args.inner_loss_weight > 0 or args.outer_loss_weight > 0 or args.hn_loss_weight > 0 or args.tv_loss_weight > 0:
            raise ValueError("Tri-zone prior losses are not supported with full supervision.")
    if args.method == "none" and (args.inner_loss_weight > 0 or args.outer_loss_weight > 0 or args.hn_loss_weight > 0 or args.tv_loss_weight > 0):
        raise ValueError("Tri-zone prior losses require --method=safe.")
    if args.method == "safe" and args.inner_loss_weight <= 0 and args.outer_loss_weight <= 0 and args.hn_loss_weight <= 0 and args.tv_loss_weight <= 0:
        raise ValueError(
            "--method=safe requires at least one positive loss weight among "
            "--inner-loss-weight/--outer-loss-weight/--hn-loss-weight/--tv-loss-weight."
        )
    if not (0.0 <= args.tau_in <= 1.0):
        raise ValueError("--tau-in must be in [0, 1].")
    if not (0.0 <= args.tau_out <= 1.0):
        raise ValueError("--tau-out must be in [0, 1].")
    if args.hard_negative_topk <= 0:
        raise ValueError("--hard-negative-topk must be > 0.")
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
    resolve_eval_mask_dir(args, dataset_config)
    if args.method == "safe":
        assert point_label_dir is not None
        validate_safe_priors(args, dataset_config, point_label_dir)


def configure_runtime(device: torch.device, deterministic: bool) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this execution environment.")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    set_seed(args.seed)
    method_name = resolve_method_name(args)
    validate_dataset_config(dataset_config, require_train_split=True)
    point_label_dir = resolve_point_label_dir(args, dataset_config)
    validate_training_args(args, dataset_config, point_label_dir)
    dataset_config = replace(dataset_config, test_mask_dir=resolve_eval_mask_dir(args, dataset_config))
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
    configure_runtime(device, args.deterministic)
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
            fieldnames=[
                "epoch",
                "loss",
                "main_loss",
                "inner_loss",
                "outer_loss",
                "hn_loss",
                "tv_loss",
                "inner_mean",
                "outer_mean",
                "inner_weighted",
                "outer_weighted",
                "hn_weighted",
                "tv_weighted",
                primary_iou_key,
            ],
        )
        writer.writeheader()

        print(
            f"device={device.type} amp={use_amp} dataset={dataset_config.name} "
            f"model={args.model_name} method={method_name} seed={args.seed}"
        )
        for epoch in range(1, args.epochs + 1):
            # Update learning rate at the start of each epoch (after optimizer.step() has been called in previous epoch)
            if scheduler is not None and epoch > 1:
                scheduler.step()

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
            # Prepare CSV row
            row = {fieldname: "" for fieldname in writer.fieldnames if fieldname is not None}
            row["epoch"] = epoch
            row["loss"] = f"{train_stats['loss']:.4f}"
            row["main_loss"] = f"{train_stats['main_loss']:.4f}"
            row["inner_loss"] = f"{train_stats['inner_loss']:.4f}"
            row["outer_loss"] = f"{train_stats['outer_loss']:.4f}"
            row["hn_loss"] = f"{train_stats['hn_loss']:.4f}"
            row["tv_loss"] = f"{train_stats['tv_loss']:.4f}"
            row["inner_mean"] = f"{train_stats['inner_mean']:.4f}"
            row["outer_mean"] = f"{train_stats['outer_mean']:.4f}"
            row["inner_weighted"] = f"{train_stats['inner_weighted']:.4f}"
            row["outer_weighted"] = f"{train_stats['outer_weighted']:.4f}"
            row["hn_weighted"] = f"{train_stats['hn_weighted']:.4f}"
            row["tv_weighted"] = f"{train_stats['tv_weighted']:.4f}"

            ran_eval = should_run(epoch, args.eval_every) or epoch == args.epochs
            if ran_eval:
                try:
                    epoch_iou = evaluate_iou_only(
                        model,
                        test_loader,
                        device,
                        threshold=args.eval_thresholds[0],
                        use_amp=use_amp,
                        model_name=args.model_name,
                    )
                    row[primary_iou_key] = f"{epoch_iou:.2f}"
                except Exception as e:
                    print(f"Warning: IoU calculation failed at epoch {epoch}: {e}")
                    row[primary_iou_key] = "0.00"

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
                f"loss={train_stats['loss']:.4f} "
                f"main={train_stats['main_loss']:.4f} "
                f"inner={train_stats['inner_loss']:.4f} "
                f"inner_w={train_stats['inner_weighted']:.4f} "
                f"outer={train_stats['outer_loss']:.4f} "
                f"outer_w={train_stats['outer_weighted']:.4f} "
                f"hn={train_stats['hn_loss']:.4f} "
                f"hn_w={train_stats['hn_weighted']:.4f} "
                f"tv={train_stats['tv_loss']:.4f} "
                f"tv_w={train_stats['tv_weighted']:.4f}"
            )
            if ran_eval:
                message += f" {primary_iou_key}={epoch_iou:.2f}"
            else:
                message += " val=skipped"
            print(message, flush=True)

    best_checkpoint_path = output_dir / "best_iou.pt"
    if not best_checkpoint_path.exists():
        raise RuntimeError("Training finished without producing best_iou.pt. Check --eval-every.")

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    final_metrics = evaluate_model(
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
