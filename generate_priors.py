from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dataset_config import build_dataset_config, validate_dataset_config
from experiment_paths import build_prior_output_root
from utils import (
    METHOD_REGISTRY,
    SplitSummary,
    build_prior_from_response,
    dilate_binary_mask,
    load_grayscale,
    load_split_file,
    make_anchor_mask,
    positive_pixel_centers,
    save_prior,
    save_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline response maps and binary priors for SAFE method"
    )
    parser.add_argument("--dataset-name", type=str, choices=["sirst3", "irstd1k", "nuaa_sirst", "nudt_sirst"], default=None)
    parser.add_argument("--label-mode", type=str, choices=["centroid", "coarse"], default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None, help="Optional explicit prior directory.")
    parser.add_argument("--experiments-root", type=str, default="experiments", help="Root directory for auto-generated prior paths.")
    parser.add_argument("--prior-tag", type=str, default=None, help="Optional custom tag for the generated prior directory.")
    parser.add_argument("--splits", type=str, nargs="+", default=None)
    parser.add_argument("--image-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--mask-dir-name", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--train-split", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument("--test-split", type=str, default=None, help="Advanced override. Prefer --dataset-name.")
    parser.add_argument(
        "--anchor-label-dir",
        type=str,
        default=None,
        help="Optional weak-label directory used to derive anchor points for prior construction. "
        "If omitted, anchors are derived from the dataset's weak-label directory for --label-mode.",
    )
    parser.add_argument("--inner-method", type=str, default="lagdm")
    parser.add_argument("--outer-method", type=str, default="wslcm")
    parser.add_argument("--inner-percentile", type=float, default=99.9)
    parser.add_argument("--outer-percentile", type=float, default=99.5)
    parser.add_argument("--inner-anchor-radius", type=int, default=1)
    parser.add_argument("--outer-anchor-radius", type=int, default=2)
    parser.add_argument("--outer-dilate-iters", type=int, default=1)
    parser.add_argument(
        "--oracle-outer-radius",
        type=int,
        default=None,
        help="Required when --outer-method=oracle. Builds M_out = Image_Domain \\ Dilate(full_gt, r).",
    )
    parser.add_argument("--write-preview-png", action="store_true")
    return parser.parse_args()


def validate_prior_args(args: argparse.Namespace) -> None:
    if args.label_mode not in {"centroid", "coarse"}:
        raise ValueError("--label-mode must be either 'centroid' or 'coarse' for SAFE prior generation.")
    if args.inner_method not in METHOD_REGISTRY and args.inner_method != "anchor":
        raise ValueError(
            f"Unsupported --inner-method '{args.inner_method}'. "
            f"Expected one of: {', '.join(sorted(METHOD_REGISTRY))}, anchor."
        )
    if args.outer_method not in METHOD_REGISTRY and args.outer_method != "oracle":
        raise ValueError(
            f"Unsupported --outer-method '{args.outer_method}'. "
            f"Expected one of: {', '.join(sorted(METHOD_REGISTRY))}, oracle."
        )
    if args.outer_method == "oracle" and (args.oracle_outer_radius is None or args.oracle_outer_radius < 0):
        raise ValueError("--outer-method=oracle requires --oracle-outer-radius >= 0.")


def resolve_prior_tag(args: argparse.Namespace) -> str:
    if args.prior_tag is not None:
        return args.prior_tag
    inner_part = (
        f"anchor_r{args.inner_anchor_radius}"
        if args.inner_method == "anchor"
        else f"{args.inner_method}_p{str(args.inner_percentile).replace('.', '_')}"
    )
    outer_part = (
        f"oracle_r{args.oracle_outer_radius}"
        if args.outer_method == "oracle"
        else f"{args.outer_method}_p{str(args.outer_percentile).replace('.', '_')}"
    )
    return f"{inner_part}_{outer_part}"


def resolve_prior_subdir_tags(args: argparse.Namespace) -> tuple[str, str]:
    inner_tag = (
        f"inner_anchor_r{args.inner_anchor_radius}"
        if args.inner_method == "anchor"
        else f"inner_{args.inner_method}_p{str(args.inner_percentile).replace('.', '_')}"
    )
    outer_tag = (
        f"outer_oracle_r{args.oracle_outer_radius}"
        if args.outer_method == "oracle"
        else f"outer_{args.outer_method}_p{str(args.outer_percentile).replace('.', '_')}"
    )
    return inner_tag, outer_tag


def main() -> None:
    args = parse_args()
    validate_prior_args(args)
    dataset_config = build_dataset_config(args)
    validate_dataset_config(dataset_config, require_train_split=False)
    dataset_root = dataset_config.root

    inner_fn = METHOD_REGISTRY.get(args.inner_method)
    outer_fn = METHOD_REGISTRY.get(args.outer_method)
    resolved_prior_tag = resolve_prior_tag(args)
    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        output_root = build_prior_output_root(
            args.experiments_root,
            dataset_config.name,
            args.label_mode,
            resolved_prior_tag,
        )

    inner_tag, outer_tag = resolve_prior_subdir_tags(args)

    image_dir_name = args.image_dir_name or dataset_config.train_image_dir
    mask_dir_name = args.mask_dir_name or dataset_config.train_mask_dir
    image_dir = dataset_root / image_dir_name
    mask_dir = dataset_root / mask_dir_name

    resolved_anchor_label_dir = args.anchor_label_dir
    if resolved_anchor_label_dir is None:
        resolved_anchor_label_dir = dataset_config.point_label_dir_for(args.label_mode)
    if resolved_anchor_label_dir is None:
        raise ValueError("SAFE prior generation requires weak-label anchors. Provide --anchor-label-dir or a valid --label-mode.")
    anchor_dir = dataset_root / resolved_anchor_label_dir

    splits = args.splits
    if not splits:
        splits = [dataset_config.train_split, dataset_config.test_split]
    splits = [split for split in splits if split not in (None, "")]

    summaries: list[SplitSummary] = []
    for split in splits:
        names = load_split_file(dataset_root / split)
        inner_pixels: list[int] = []
        outer_pixels: list[int] = []

        for name in names:
            image = load_grayscale(image_dir / f"{name}.png")
            anchor_map = (load_grayscale(anchor_dir / f"{name}.png") > 0).astype(np.uint8)
            centers = positive_pixel_centers(anchor_map)
            full_mask = (load_grayscale(mask_dir / f"{name}.png") > 0).astype(np.uint8)

            if args.inner_method == "anchor":
                inner_prior = make_anchor_mask(image.shape, centers, args.inner_anchor_radius).astype(np.uint8)
                inner_response = inner_prior.astype(np.float32)
            else:
                assert inner_fn is not None
                inner_response = inner_fn(image)
                inner_prior = build_prior_from_response(
                    inner_response,
                    centers,
                    percentile=args.inner_percentile,
                    anchor_radius=args.inner_anchor_radius,
                )

            if args.outer_method == "oracle":
                dilated_gt = dilate_binary_mask(full_mask, args.oracle_outer_radius or 0)
                outer_prior = (1 - dilated_gt).astype(np.uint8)
                outer_response = outer_prior.astype(np.float32)
            else:
                assert outer_fn is not None
                outer_response = outer_fn(image)
                outer_prior = build_prior_from_response(
                    outer_response,
                    centers,
                    percentile=args.outer_percentile,
                    anchor_radius=args.outer_anchor_radius,
                    dilate_iters=args.outer_dilate_iters,
                )

            inner_pixels.append(int(inner_prior.sum()))
            outer_pixels.append(int(outer_prior.sum()))

            inner_npy = output_root / "responses" / args.inner_method / f"{name}.npy"
            outer_npy = output_root / "responses" / args.outer_method / f"{name}.npy"
            inner_png = output_root / "responses_png" / args.inner_method / f"{name}.png" if args.write_preview_png else None
            outer_png = output_root / "responses_png" / args.outer_method / f"{name}.png" if args.write_preview_png else None
            save_response(inner_response, inner_npy, inner_png)
            save_response(outer_response, outer_npy, outer_png)
            save_prior(inner_prior, output_root / "priors" / inner_tag / f"{name}.png")
            save_prior(outer_prior, output_root / "priors" / outer_tag / f"{name}.png")

        summaries.append(
            SplitSummary(
                split=split,
                count=len(names),
                mean_inner_pixels=float(np.mean(inner_pixels)) if inner_pixels else 0.0,
                mean_outer_pixels=float(np.mean(outer_pixels)) if outer_pixels else 0.0,
            )
        )

    manifest = {
        "dataset_name": dataset_config.name,
        "label_mode": args.label_mode,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "resolved_prior_tag": resolved_prior_tag,
        "image_dir_name": image_dir_name,
        "mask_dir_name": mask_dir_name,
        "anchor_label_dir": str(anchor_dir.resolve()),
        "inner_method": args.inner_method,
        "outer_method": args.outer_method,
        "inner_prior_dir": str((output_root / "priors" / inner_tag).resolve()),
        "outer_prior_dir": str((output_root / "priors" / outer_tag).resolve()),
        "splits": splits,
        "params": {
            "inner_percentile": args.inner_percentile,
            "outer_percentile": args.outer_percentile,
            "inner_anchor_radius": args.inner_anchor_radius,
            "outer_anchor_radius": args.outer_anchor_radius,
            "outer_dilate_iters": args.outer_dilate_iters,
            "oracle_outer_radius": args.oracle_outer_radius,
        },
        "summaries": [asdict(summary) for summary in summaries],
        "notes": {
            "lagdm": "Current implementation is a lightweight engineering proxy used for idea verification.",
            "wslcm": "Current implementation is a lightweight engineering proxy used for idea verification.",
            "anchor": "Uses weak-label anchors directly as the inner prior.",
            "oracle": "Constructs M_out = Image_Domain minus Dilate(full_gt, r). For oracle diagnosis only, not fair weak-supervision comparison.",
        },
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
