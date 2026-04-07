from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data import build_dataset_config, validate_dataset_config
from experiment_paths import build_prior_output_root, build_prior_tag
from utils import (
    METHOD_REGISTRY,
    SplitSummary,
    build_prior_from_response,
    component_centers,
    load_grayscale,
    load_split_file,
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
    parser.add_argument(
        "--anchor-label-dir",
        type=str,
        default=None,
        help="Optional weak-label directory used to derive anchor points for prior construction. "
        "If omitted, anchors are derived from --label-mode or from --mask-dir-name.",
    )
    parser.add_argument("--inner-method", type=str, default="lagdm")
    parser.add_argument("--outer-method", type=str, default="wslcm")
    parser.add_argument("--inner-percentile", type=float, default=99.9)
    parser.add_argument("--outer-percentile", type=float, default=99.5)
    parser.add_argument("--inner-anchor-radius", type=int, default=1)
    parser.add_argument("--outer-anchor-radius", type=int, default=2)
    parser.add_argument("--outer-dilate-iters", type=int, default=1)
    parser.add_argument("--write-preview-png", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    validate_dataset_config(dataset_config, require_train_split=False)
    dataset_root = dataset_config.root

    inner_fn = METHOD_REGISTRY[args.inner_method]
    outer_fn = METHOD_REGISTRY[args.outer_method]
    resolved_prior_tag = args.prior_tag or build_prior_tag(
        args.inner_method,
        args.outer_method,
        args.inner_percentile,
        args.outer_percentile,
    )
    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        if args.label_mode is None:
            raise ValueError("--label-mode is required when --output-root is not provided.")
        output_root = build_prior_output_root(
            args.experiments_root,
            dataset_config.name,
            args.label_mode,
            resolved_prior_tag,
        )

    inner_tag = f"inner_{args.inner_method}_p{str(args.inner_percentile).replace('.', '_')}"
    outer_tag = f"outer_{args.outer_method}_p{str(args.outer_percentile).replace('.', '_')}"

    image_dir_name = args.image_dir_name or dataset_config.train_image_dir
    mask_dir_name = args.mask_dir_name or dataset_config.train_mask_dir
    image_dir = dataset_root / image_dir_name
    mask_dir = dataset_root / mask_dir_name

    resolved_anchor_label_dir = args.anchor_label_dir
    if resolved_anchor_label_dir is None and args.label_mode is not None:
        resolved_anchor_label_dir = dataset_config.point_label_dir_for(args.label_mode)
    anchor_dir = dataset_root / resolved_anchor_label_dir if resolved_anchor_label_dir is not None else None

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
            mask = (load_grayscale(mask_dir / f"{name}.png") > 0).astype(np.uint8)
            if anchor_dir is not None:
                anchor_map = (load_grayscale(anchor_dir / f"{name}.png") > 0).astype(np.uint8)
                centers = positive_pixel_centers(anchor_map)
            else:
                centers = component_centers(mask)

            inner_response = inner_fn(image)
            outer_response = outer_fn(image)

            inner_prior = build_prior_from_response(
                inner_response,
                centers,
                percentile=args.inner_percentile,
                anchor_radius=args.inner_anchor_radius,
            )
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
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "resolved_prior_tag": resolved_prior_tag,
        "image_dir_name": image_dir_name,
        "mask_dir_name": mask_dir_name,
        "anchor_label_dir": (str((dataset_root / resolved_anchor_label_dir).resolve()) if resolved_anchor_label_dir is not None else None),
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
        },
        "summaries": [asdict(summary) for summary in summaries],
        "notes": {
            "lagdm": "Current implementation is a lightweight engineering proxy used for idea verification.",
            "wslcm": "Current implementation is a lightweight engineering proxy used for idea verification.",
        },
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
