from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model_ACM import ACM
from .model_ALCNet import ASKCResNetFPN
from .model_DNANet import DNANet
from .model_ISTDUNet import ISTDUNet
from .model_MSHNet import MSHNet
from .model_RDIAN import RDIAN
from .model_SCTransNet import SCTransNet, get_CTranS_config
from .model_UIUNet import UIUNet
from .model_lightweight_unet import LightweightUNet


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: callable
    default_channels: tuple[int, ...] | None = None
    input_align: int | None = None


# Builders for each model
def _build_lightweight_unet() -> nn.Module:
    return LightweightUNet(channels=(16, 32, 64))


def _build_acm() -> nn.Module:
    return ACM(in_channels=1, channels=(8, 16, 32, 64))


def _build_alcnet() -> nn.Module:
    return ASKCResNetFPN(in_channels=1, channels=(8, 16, 32, 64))


def _build_dnanet() -> nn.Module:
    return DNANet(num_classes=1, input_channels=1)


def _build_istdunet() -> nn.Module:
    return ISTDUNet(in_channels=1)


def _build_mshnet() -> nn.Module:
    return MSHNet(input_channels=1)


def _build_rdian() -> nn.Module:
    return RDIAN(in_channels=1)


def _build_sctransnet() -> nn.Module:
    return SCTransNet(get_CTranS_config(), n_channels=1, n_classes=1)


def _build_uiunet() -> nn.Module:
    return UIUNet(in_ch=1, out_ch=1)


# Model registry
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # SAFE models
    "lightweight_unet": ModelSpec(
        name="lightweight_unet",
        builder=_build_lightweight_unet,
        default_channels=(16, 32, 64),
        input_align=None,
    ),
    "acm": ModelSpec(
        name="acm",
        builder=_build_acm,
        default_channels=(8, 16, 32, 64),
        input_align=32,
    ),
    # SOTA models
    "alcnet": ModelSpec(
        name="alcnet",
        builder=_build_alcnet,
        default_channels=(8, 16, 32, 64),
        input_align=32,
    ),
    "dnanet": ModelSpec(
        name="dnanet",
        builder=_build_dnanet,
        default_channels=None,
    ),
    "istdunet": ModelSpec(
        name="istdunet",
        builder=_build_istdunet,
        default_channels=None,
    ),
    "mshnet": ModelSpec(
        name="mshnet",
        builder=_build_mshnet,
        default_channels=None,
    ),
    "rdian": ModelSpec(
        name="rdian",
        builder=_build_rdian,
        default_channels=None,
    ),
    "sctransnet": ModelSpec(
        name="sctransnet",
        builder=_build_sctransnet,
        default_channels=None,
    ),
    "uiunet": ModelSpec(
        name="uiunet",
        builder=_build_uiunet,
        default_channels=None,
    ),
}


def get_model_names() -> list[str]:
    """Return the registered model names in declaration order."""
    return list(MODEL_REGISTRY.keys())


def get_input_align(model_name: str) -> int | None:
    """Return required spatial alignment for a model, if any."""
    try:
        return MODEL_REGISTRY[model_name].input_align
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def build_model(model_name: str) -> nn.Module:
    """Build a model by name from the registry."""
    try:
        spec = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported model: {model_name}. Available: {available}") from exc
    return spec.builder()


def get_default_channels(model_name: str) -> tuple[int, ...] | None:
    """Get default channel configuration for a model."""
    try:
        return MODEL_REGISTRY[model_name].default_channels
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def prepare_model_input(image: torch.Tensor, model_name: str) -> torch.Tensor:
    """Pad input to the model's required spatial alignment, if any."""
    align = get_input_align(model_name)
    if align is not None and align > 1:
        _, _, h, w = image.shape
        pad_h = (align - h % align) % align
        pad_w = (align - w % align) % align
        if pad_h > 0 or pad_w > 0:
            image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return image
