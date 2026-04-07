from __future__ import annotations

from pathlib import Path


def sanitize_tag(value: str) -> str:
    sanitized = value.strip().replace(" ", "_").replace("/", "_")
    if not sanitized:
        raise ValueError("Tag values must not be empty.")
    return sanitized


def format_percentile_tag(value: float) -> str:
    return str(value).replace(".", "")


def resolve_supervision_tag(method: str, label_mode: str) -> str:
    """Resolve supervision tag for experiment directory naming.

    New orthogonal design:
    - label-mode=full, method=none → "full"
    - label-mode=centroid/coarse, method=none → "point_{label_mode}"
    - label-mode=centroid/coarse, method=safe → "safe_{label_mode}"

    Future:
    - method=lesps → "lesps_{label_mode}"
    - method=pal → "pal_{label_mode}"
    """
    if label_mode == "full":
        if method != "none":
            raise ValueError(f"label-mode=full only supports method=none, got '{method}'")
        return "full"
    if label_mode not in {"centroid", "coarse"}:
        raise ValueError(f"label-mode must be one of: full, centroid, coarse. Got '{label_mode}'")
    if method == "none":
        return f"point_{label_mode}"
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
