from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============ Loss Classes ============

class PointSupervisionLoss(nn.Module):
    def __init__(self, pos_weight: float = 10.0, positive_threshold: float = 0.5) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.positive_threshold = positive_threshold

    def forward(self, logits: torch.Tensor, point_targets: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(point_targets)
        weight = torch.where(point_targets >= self.positive_threshold, self.pos_weight, weight)
        return F.binary_cross_entropy_with_logits(logits, point_targets, weight=weight)


class FullSupervisionLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(mask_targets)
        if self.pos_weight != 1.0:
            weight = torch.where(mask_targets >= 0.5, self.pos_weight, weight)
        return F.binary_cross_entropy_with_logits(logits, mask_targets, weight=weight)


class TriZonePartialLoss(nn.Module):
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
        if effective_inner_weight > 0 and enable_inner:
            inner_mask = (inner_prior > 0.5).float()
            inner_term = self._masked_mean((1.0 - prob) ** 2, inner_mask)

        outer_term = logits.new_zeros(())
        if effective_outer_weight > 0 and enable_outer:
            outer_forbidden = (outer_prior <= 0.5).float()
            outer_term = self._masked_mean(prob ** 2, outer_forbidden)

        total = point_term + effective_inner_weight * inner_term + effective_outer_weight * outer_term
        stats = {
            "point_loss": float(point_term.detach().item()),
            "inner_loss": float(inner_term.detach().item()),
            "outer_loss": float(outer_term.detach().item()),
            "total_loss": float(total.detach().item()),
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
    if args.method == "safe":
        return "trizone"
    return args.method


def build_criterion(args: argparse.Namespace, method_name: str) -> MethodBundle:
    if method_name == "full":
        return MethodBundle(
            name="full",
            criterion=FullSupervisionLoss(pos_weight=args.full_pos_weight),
        )

    if method_name in {"point", "trizone"}:
        criterion = TriZonePartialLoss(
            point_loss=PointSupervisionLoss(
                pos_weight=args.pos_weight,
                positive_threshold=args.positive_threshold,
            ),
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
    "FullSupervisionLoss",
    "PointSupervisionLoss",
    "TriZonePartialLoss",
    "MethodBundle",
    "resolve_method_name",
    "build_criterion",
]
