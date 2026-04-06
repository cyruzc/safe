from __future__ import annotations

from pathlib import Path


def sanitize_tag(value: str) -> str:
    sanitized = value.strip().replace(" ", "_").replace("/", "_")
    if not sanitized:
        raise ValueError("Tag values must not be empty.")
    return sanitized


def format_percentile_tag(value: float) -> str:
    return str(value).replace(".", "")


def resolve_supervision_tag(method: str, label_mode: str | None) -> str:
    if method == "full":
        return "full"
    if label_mode not in {"centroid", "coarse"}:
        raise ValueError(f"Method '{method}' requires --label-mode to be one of: centroid, coarse.")
    return f"{method}_{label_mode}"


def build_run_output_dir(
    experiments_root: str | Path,
    dataset_name: str,
    model_name: str,
    supervision_tag: str,
    exp_tag: str,
    seed: int,
) -> Path:
    return (
        Path(experiments_root)
        / "runs"
        / sanitize_tag(dataset_name)
        / sanitize_tag(model_name)
        / sanitize_tag(supervision_tag)
        / f"{sanitize_tag(exp_tag)}_seed{seed}"
    )


def build_prior_tag(
    inner_method: str,
    outer_method: str,
    inner_percentile: float,
    outer_percentile: float,
) -> str:
    return (
        f"{sanitize_tag(inner_method)}_{sanitize_tag(outer_method)}"
        f"_p{format_percentile_tag(inner_percentile)}"
        f"_p{format_percentile_tag(outer_percentile)}"
    )


def build_prior_output_root(
    experiments_root: str | Path,
    dataset_name: str,
    label_mode: str,
    prior_tag: str,
) -> Path:
    return (
        Path(experiments_root)
        / "priors"
        / sanitize_tag(dataset_name)
        / sanitize_tag(label_mode)
        / sanitize_tag(prior_tag)
    )
