from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============ Loss Classes ============

class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    LESPS uses: alpha=2.0, gamma=4.0 for infrared small target detection.

    Key difference from standard Focal Loss:
    - Normalization by POSITIVE SAMPLES only, not total pixels
    - This prevents gradient dilution in extremely imbalanced scenarios
    """
    def __init__(self, alpha: float = 2.0, gamma: float = 4.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.alpha = alpha  # Modulating factor for positive samples
        self.gamma = gamma  # Modulating factor for negative samples
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # LESPS-style focal loss computation
        pos_weights = targets
        neg_weights = (1 - targets).pow(self.gamma)

        pos_loss = -(probs + self.eps).log() * (1 - probs).pow(self.alpha) * pos_weights
        neg_loss = -(1 - probs + self.eps).log() * probs.pow(self.alpha) * neg_weights

        loss = pos_loss + neg_loss

        # 🔑 KEY FIX: Normalize by positive samples only (LEPS-style)
        # This prevents gradient dilution in extremely imbalanced scenarios
        avg_factor = targets.eq(1).sum().clamp_min(1)
        return loss.sum() / avg_factor


class PointSupervisionLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 10.0,
        positive_threshold: float = 0.5,
        loss_type: str = "bce",
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.positive_threshold = positive_threshold
        self.loss_type = loss_type

        if loss_type == "focal":
            self.focal_loss = FocalLoss(alpha=2.0, gamma=4.0)

    def _positive_only_focal(self, logits: torch.Tensor, point_targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        positive_mask = (point_targets >= self.positive_threshold).float()
        pos_loss = -(probs + self.focal_loss.eps).log() * (1 - probs).pow(self.focal_loss.alpha) * positive_mask
        avg_factor = positive_mask.sum().clamp_min(1)
        return pos_loss.sum() / avg_factor

    def _positive_only_bce(self, logits: torch.Tensor, point_targets: torch.Tensor) -> torch.Tensor:
        positive_mask = (point_targets >= self.positive_threshold).float()
        positive_targets = torch.ones_like(point_targets)
        positive_loss = F.binary_cross_entropy_with_logits(
            logits,
            positive_targets,
            reduction="none",
        )
        avg_factor = positive_mask.sum().clamp_min(1)
        return (positive_loss * positive_mask).sum() / avg_factor

    def forward(self, logits: torch.Tensor, point_targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "focal":
            # Partial-label point supervision: only labeled points are positive;
            # unlabeled pixels are ignored rather than treated as background.
            return self._positive_only_focal(logits, point_targets)
        else:  # BCE
            return self._positive_only_bce(logits, point_targets)


class FullSupervisionLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, loss_type: str = "bce") -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.loss_type = loss_type

        if loss_type == "focal":
            self.focal_loss = FocalLoss(alpha=2.0, gamma=4.0)

    def forward(self, logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "focal":
            return self.focal_loss(logits, mask_targets)
        else:  # BCE
            weight = torch.ones_like(mask_targets)
            if self.pos_weight != 1.0:
                weight = torch.where(mask_targets >= 0.5, self.pos_weight, weight)
            return F.binary_cross_entropy_with_logits(logits, mask_targets, weight=weight)


class TriZonePartialLoss(nn.Module):
    """Tri-Zone Partial Loss for SAFE method.

    SAFE (Self-Adaptive ...) uses a three-zone approach:
    1. Point zone: Direct point supervision
    2. Inner prior zone: Encourages high probability near points
    3. Outer prior zone: Encourages low probability in background regions

    This implements the core technical innovation of the SAFE method.
    """
    def __init__(
        self,
        point_loss: PointSupervisionLoss,
        inner_weight: float = 0.0,
        outer_weight: float = 0.0,
        eps: float = 1e-6,
        inner_decay_schedule: tuple[int, int] | None = None,
        outer_boost_schedule: tuple[int, int, float] | None = None,
    ) -> None:
        super().__init__()
        self.point_loss = point_loss
        self.inner_weight = inner_weight
        self.outer_weight = outer_weight
        self.eps = eps
        self.inner_decay_schedule = inner_decay_schedule
        self.outer_boost_schedule = outer_boost_schedule

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask).sum() / mask.sum().clamp_min(self.eps)

    def _point_normalized_masked_sum(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        point_targets: torch.Tensor,
    ) -> torch.Tensor:
        point_count = point_targets.ge(0.5).sum().clamp_min(1)
        return (value * mask).sum() / point_count

    def forward(
        self,
        logits: torch.Tensor,
        point_targets: torch.Tensor,
        inner_prior: torch.Tensor,
        outer_prior: torch.Tensor,
        inner_weight_scale: float = 1.0,
        outer_weight_scale: float = 1.0,
        enable_inner: bool = True,
        enable_outer: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        point_term = self.point_loss(logits, point_targets)
        prob = torch.sigmoid(logits)
        effective_inner_weight = self.inner_weight * inner_weight_scale
        effective_outer_weight = self.outer_weight * outer_weight_scale

        inner_term = logits.new_zeros(())
        inner_mean = logits.new_zeros(())
        if effective_inner_weight > 0 and enable_inner:
            inner_mask = (inner_prior > 0.5).float()
            inner_value = (1.0 - prob) ** 2
            inner_mean = self._masked_mean(inner_value, inner_mask)
            inner_term = self._point_normalized_masked_sum(inner_value, inner_mask, point_targets)

        outer_term = logits.new_zeros(())
        outer_mean = logits.new_zeros(())
        if effective_outer_weight > 0 and enable_outer:
            outer_forbidden = (outer_prior <= 0.5).float()
            outer_value = prob ** 2
            outer_mean = self._masked_mean(outer_value, outer_forbidden)
            outer_term = outer_mean

        inner_weighted = effective_inner_weight * inner_term
        outer_weighted = effective_outer_weight * outer_term
        total = point_term + inner_weighted + outer_weighted
        stats = {
            "loss": float(total.detach().item()),
            "main_loss": float(point_term.detach().item()),
            "inner_loss": float(inner_term.detach().item()),
            "outer_loss": float(outer_term.detach().item()),
            "inner_mean": float(inner_mean.detach().item()),
            "outer_mean": float(outer_mean.detach().item()),
            "inner_weighted": float(inner_weighted.detach().item()),
            "outer_weighted": float(outer_weighted.detach().item()),
            "effective_inner_weight": float(effective_inner_weight),
            "effective_outer_weight": float(effective_outer_weight),
        }
        return total, stats


# ============ Method Bundle ============

@dataclass(frozen=True)
class MethodBundle:
    name: str
    criterion: nn.Module


# ============ Factory Functions ============

def resolve_method_name(args: argparse.Namespace) -> str:
    """Resolve method name based on label-mode and method.

    New orthogonal design:
    - label-mode=full, method=none → "full" (full supervision)
    - label-mode=centroid/coarse, method=none → "point" (point supervision baseline)
    - label-mode=centroid/coarse, method=safe → "safe" (SAFE with tri-zone priors)
    """
    if args.label_mode == "full":
        return "full"
    if args.method == "safe":
        return "safe"
    # label-mode=centroid/coarse + method=none → point supervision baseline
    return "point"


def build_point_supervision_loss(args: argparse.Namespace, loss_type: str) -> PointSupervisionLoss:
    return PointSupervisionLoss(
        pos_weight=args.pos_weight,
        positive_threshold=args.positive_threshold,
        loss_type=loss_type,
    )


def build_criterion(args: argparse.Namespace, method_name: str) -> MethodBundle:
    """Build loss criterion based on resolved method name.

    Orthogonal CLI design:
    - label-mode=full, method=none → FullSupervisionLoss
    - label-mode=centroid/coarse, method=none → PointSupervisionLoss
    - label-mode=centroid/coarse, method=safe → TriZonePartialLoss with priors

    Future extensions:
    - method=lesps → LESPS-specific loss
    - method=pal → PAL-specific loss
    """
    loss_type = getattr(args, 'loss_type', 'bce')  # Default to BCE for backward compatibility

    if method_name == "full":
        return MethodBundle(
            name="full",
            criterion=FullSupervisionLoss(pos_weight=args.full_pos_weight, loss_type=loss_type),
        )

    if method_name == "point":
        return MethodBundle(
            name="point",
            criterion=build_point_supervision_loss(args, loss_type),
        )

    if method_name == "safe":
        criterion = TriZonePartialLoss(
            point_loss=build_point_supervision_loss(args, loss_type),
            inner_weight=args.inner_loss_weight,
            outer_weight=args.outer_loss_weight,
            inner_decay_schedule=(
                (args.inner_decay_start_epoch, args.inner_decay_end_epoch)
                if args.inner_decay_start_epoch is not None else None
            ),
            outer_boost_schedule=(
                (args.outer_boost_start_epoch, args.outer_boost_end_epoch, args.outer_boost_scale)
                if args.outer_boost_start_epoch is not None else None
            ),
        )
        return MethodBundle(name=method_name, criterion=criterion)

    raise ValueError(f"Unsupported method: {method_name}")


__all__ = [
    "FocalLoss",
    "FullSupervisionLoss",
    "PointSupervisionLoss",
    "TriZonePartialLoss",  # Core implementation of SAFE method
    "MethodBundle",
    "resolve_method_name",
    "build_point_supervision_loss",
    "build_criterion",
]

# Naming convention:
# - SAFE: Project name and user-facing method name (--method safe)
# - TriZone: Technical implementation details (TriZonePartialLoss class)
# - This keeps API clean while preserving descriptive technical names
