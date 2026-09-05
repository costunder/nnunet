"""Curriculum ranking objective with online semi-hard negative mining.

All per-batch diagnostics remain as device tensors. The training loop transfers
one compact accumulator to the CPU at epoch end, avoiding repeated CUDA
synchronizations from ``Tensor.item()`` and Python ``bool(Tensor)`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from hiercp.schema import (
    DIFFICULTY_EASY,
    DIFFICULTY_INTER_REGION,
    DIFFICULTY_INTRA_CORRUPTED,
)


@dataclass(frozen=True)
class CurriculumConfig:
    easy_epochs: int = 8
    inter_epochs: int = 18
    intra_epochs: int = 28
    model_mine_start_epoch: int = 29
    semi_hard_low_percentile: float = 0.70
    semi_hard_high_percentile: float = 0.95
    cross_entropy_weight: float = 1.0
    pairwise_weight: float = 1.0
    ordinal_weight: float = 0.2
    mined_weight: float = 0.5
    easy_margin: float = 1.0
    inter_margin: float = 0.7
    intra_margin: float = 0.4
    ordinal_margin: float = 0.05

    def validate(self) -> None:
        if not 0 <= self.semi_hard_low_percentile < self.semi_hard_high_percentile <= 1:
            raise ValueError("Invalid semi-hard percentile band")
        if self.easy_epochs < 1 or self.inter_epochs < self.easy_epochs:
            raise ValueError("Curriculum epoch boundaries are invalid")
        if self.intra_epochs < self.inter_epochs:
            raise ValueError("intra_epochs must be >= inter_epochs")
        if self.model_mine_start_epoch <= self.intra_epochs:
            raise ValueError(
                "model_mine_start_epoch must be after the intra-region curriculum stage"
            )
        weights = (
            self.cross_entropy_weight,
            self.pairwise_weight,
            self.ordinal_weight,
            self.mined_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Loss weights must be non-negative")


def _active_max_difficulty(epoch: int, config: CurriculumConfig) -> int:
    if epoch <= config.easy_epochs:
        return DIFFICULTY_EASY
    if epoch <= config.inter_epochs:
        return DIFFICULTY_INTER_REGION
    return DIFFICULTY_INTRA_CORRUPTED


def _margins(difficulty: Tensor, config: CurriculumConfig) -> Tensor:
    output = torch.full_like(difficulty, config.easy_margin, dtype=torch.float32)
    output = torch.where(
        difficulty == DIFFICULTY_INTER_REGION,
        torch.full_like(output, config.inter_margin),
        output,
    )
    return torch.where(
        difficulty == DIFFICULTY_INTRA_CORRUPTED,
        torch.full_like(output, config.intra_margin),
        output,
    )


def _semi_hard_mask(scores: Tensor, config: CurriculumConfig) -> Tensor:
    if scores.numel() < 3:
        return torch.zeros_like(scores, dtype=torch.bool)
    # AMP may produce float16/bfloat16 candidate scores, while
    # torch.quantile requires float32 or float64. Semi-hard selection is
    # detached and non-differentiable, so compute only its thresholds in FP32.
    detached = scores.detach().to(dtype=torch.float32)
    low = torch.quantile(
        detached,
        float(config.semi_hard_low_percentile),
    )
    high = torch.quantile(
        detached,
        float(config.semi_hard_high_percentile),
    )
    return (detached >= low) & (detached <= high)


def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weights = mask.to(values.dtype)
    count = weights.sum()
    mean = (values * weights).sum() / count.clamp_min(1.0)
    return mean, count


def _padded_scores(score_list: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Pack variable-length cases once; all subsequent score math is batched."""

    if not score_list:
        raise ValueError("score_list must be non-empty")
    if any(score.ndim != 1 or score.numel() < 1 for score in score_list):
        raise ValueError("Each score tensor must be a non-empty vector")
    reference = score_list[0]
    if any(score.device != reference.device or score.dtype != reference.dtype
           for score in score_list):
        raise ValueError("All score tensors must have the same device and dtype")
    scores = pad_sequence(score_list, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(
        [score.numel() for score in score_list], device=scores.device,
        dtype=torch.long,
    )
    valid = torch.arange(scores.shape[1], device=scores.device)[None] < lengths[:, None]
    return scores, valid


def _masked_row_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1)
    return (values * weights).sum(dim=-1) / count.clamp_min(1.0), count


def _batched_semi_hard_mask(
    scores: Tensor, valid: Tensor, config: CurriculumConfig
) -> Tensor:
    """Exact per-case torch.quantile(linear) semantics without a case loop.

    Padding sorts after every real negative. Interpolated order statistics use
    each case's own count, not the padded width or the pooled batch quantiles.
    As in the original objective, mining considers *all* real negatives.
    """

    detached = scores.detach().float()
    ordered = detached.masked_fill(~valid, torch.inf).sort(dim=-1).values
    count = valid.sum(dim=-1)
    quantiles = detached.new_tensor(
        [config.semi_hard_low_percentile, config.semi_hard_high_percentile]
    )
    positions = (count - 1).clamp_min(0)[:, None] * quantiles[None]
    lower = positions.floor().long()
    upper = positions.ceil().long()
    lower_values = ordered.gather(1, lower)
    upper_values = ordered.gather(1, upper)
    thresholds = torch.lerp(lower_values, upper_values, positions - lower)
    return (
        valid & (count[:, None] >= 3)
        & (detached >= thresholds[:, :1])
        & (detached <= thresholds[:, 1:])
    )


def curriculum_ranking_loss(
    score_list: Sequence[Tensor],
    difficulty_list: Sequence[Tensor],
    *,
    epoch: int,
    config: CurriculumConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    config.validate()
    if len(score_list) != len(difficulty_list):
        raise ValueError("score_list and difficulty_list lengths differ")
    if any(scores.numel() != difficulties.numel() or scores.numel() < 2
           or difficulties.ndim != 1
           for scores, difficulties in zip(score_list, difficulty_list)):
        raise ValueError("Each sample needs one positive and at least one negative")
    scores, valid = _padded_scores(score_list)
    difficulties = pad_sequence(
        difficulty_list, batch_first=True, padding_value=-1
    ).to(device=scores.device, non_blocking=True)
    active_max = _active_max_difficulty(epoch, config)
    negatives, negative_valid = scores[:, 1:], valid[:, 1:]
    negative_difficulties = difficulties[:, 1:]
    active = (negative_valid & (negative_difficulties >= DIFFICULTY_EASY)
              & (negative_difficulties <= active_max))
    logits = torch.cat([scores[:, :1], negatives.masked_fill(~active, -torch.inf)], dim=1)
    ce = F.cross_entropy(
        logits, torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device),
        reduction="none",
    )
    pair, _ = _masked_row_mean(
        F.softplus(_margins(negative_difficulties, config) - scores[:, :1] + negatives),
        active,
    )
    levels = torch.tensor(
        [DIFFICULTY_EASY, DIFFICULTY_INTER_REGION, DIFFICULTY_INTRA_CORRUPTED],
        device=scores.device,
    )
    group_mask = (
        negative_valid[:, None, :]
        & (negative_difficulties[:, None, :] == levels[None, :, None])
        & (levels[None, :, None] <= active_max)
    )
    group_means, group_counts = _masked_row_mean(negatives[:, None, :], group_mask)
    ordinal, _ = _masked_row_mean(
        F.softplus(config.ordinal_margin - group_means[:, 1:] + group_means[:, :-1]),
        (group_counts[:, 1:] > 0) & (group_counts[:, :-1] > 0),
    )
    mined = scores.new_zeros((scores.shape[0],))
    if epoch >= config.model_mine_start_epoch:
        mined, _ = _masked_row_mean(
            F.softplus(config.intra_margin - scores[:, :1] + negatives),
            _batched_semi_hard_mask(negatives, negative_valid, config),
        )
    total = (
        config.cross_entropy_weight * ce + config.pairwise_weight * pair
        + config.ordinal_weight * ordinal + config.mined_weight * mined
    ).mean()
    metrics = {
        "loss": total.detach(),
        "ce": ce.mean().detach(),
        "pair": pair.mean().detach(),
        "ordinal": ordinal.mean().detach(),
        "mined": mined.mean().detach(),
        "active_max_difficulty": total.new_tensor(float(active_max)),
    }
    return total, metrics


def ranking_metric_sums(score_list: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Return device-side accuracy sum, reciprocal-rank sum and sample count."""

    scores, valid = _padded_scores(score_list)
    rank = 1 + ((scores[:, 1:] >= scores[:, :1]) & valid[:, 1:]).sum(dim=1)
    return (
        (rank == 1).float().sum(),
        rank.float().reciprocal().sum(),
        scores.new_tensor(float(len(score_list)), dtype=torch.float32),
    )


def ranking_metrics(score_list: Sequence[Tensor]) -> tuple[float, float]:
    """Compatibility helper for tests; training uses ``ranking_metric_sums``."""

    accuracy_sum, reciprocal_rank_sum, count = ranking_metric_sums(score_list)
    values = torch.stack(
        [accuracy_sum / count.clamp_min(1.0), reciprocal_rank_sum / count.clamp_min(1.0)]
    ).detach().cpu().tolist()
    return float(values[0]), float(values[1])
