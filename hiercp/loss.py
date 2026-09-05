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
    active_max = _active_max_difficulty(epoch, config)
    total_terms: list[Tensor] = []
    ce_terms: list[Tensor] = []
    pair_terms: list[Tensor] = []
    ordinal_terms_all: list[Tensor] = []
    mined_terms: list[Tensor] = []

    for scores, difficulties in zip(score_list, difficulty_list):
        difficulties = difficulties.to(device=scores.device, non_blocking=True)
        if scores.numel() != difficulties.numel() or scores.numel() < 2:
            raise ValueError("Each sample needs one positive and at least one negative")

        negative_difficulties_all = difficulties[1:]
        negative_scores_all = scores[1:]
        active_negative_mask = (
            (negative_difficulties_all >= DIFFICULTY_EASY)
            & (negative_difficulties_all <= active_max)
        )

        # Keep tensor dimensions static: inactive logits are masked rather than
        # dynamically indexed into a variable-length CUDA tensor.
        masked_negative_logits = torch.where(
            active_negative_mask,
            negative_scores_all,
            torch.full_like(negative_scores_all, -torch.inf),
        )
        active_logits = torch.cat([scores[:1], masked_negative_logits], dim=0)
        ce = F.cross_entropy(
            active_logits[None],
            torch.zeros(1, dtype=torch.long, device=scores.device),
        )
        ce_terms.append(ce)

        pair_values = F.softplus(
            _margins(negative_difficulties_all, config).to(scores.device)
            - scores[0]
            + negative_scores_all
        )
        pair, _ = _masked_mean(pair_values, active_negative_mask)
        pair_terms.append(pair)

        group_means: dict[int, Tensor] = {}
        group_counts: dict[int, Tensor] = {}
        for level in (
            DIFFICULTY_EASY,
            DIFFICULTY_INTER_REGION,
            DIFFICULTY_INTRA_CORRUPTED,
        ):
            mask = (negative_difficulties_all == level) & (level <= active_max)
            mean, count = _masked_mean(negative_scores_all, mask)
            group_means[level] = mean
            group_counts[level] = count

        ordinal_sum = scores.new_zeros(())
        ordinal_count = scores.new_zeros(())
        easy_inter_valid = (
            (group_counts[DIFFICULTY_EASY] > 0)
            & (group_counts[DIFFICULTY_INTER_REGION] > 0)
        )
        easy_inter = F.softplus(
            config.ordinal_margin
            - group_means[DIFFICULTY_INTER_REGION]
            + group_means[DIFFICULTY_EASY]
        )
        easy_inter_weight = easy_inter_valid.to(scores.dtype)
        ordinal_sum = ordinal_sum + easy_inter * easy_inter_weight
        ordinal_count = ordinal_count + easy_inter_weight

        inter_intra_valid = (
            (group_counts[DIFFICULTY_INTER_REGION] > 0)
            & (group_counts[DIFFICULTY_INTRA_CORRUPTED] > 0)
        )
        inter_intra = F.softplus(
            config.ordinal_margin
            - group_means[DIFFICULTY_INTRA_CORRUPTED]
            + group_means[DIFFICULTY_INTER_REGION]
        )
        inter_intra_weight = inter_intra_valid.to(scores.dtype)
        ordinal_sum = ordinal_sum + inter_intra * inter_intra_weight
        ordinal_count = ordinal_count + inter_intra_weight
        ordinal = ordinal_sum / ordinal_count.clamp_min(1.0)
        ordinal_terms_all.append(ordinal)

        mined = scores.new_zeros(())
        if epoch >= config.model_mine_start_epoch:
            semi_hard_mask = _semi_hard_mask(negative_scores_all, config)
            mined_values = F.softplus(
                config.intra_margin - scores[0] + negative_scores_all
            )
            mined, _ = _masked_mean(mined_values, semi_hard_mask)
        mined_terms.append(mined)

        total_terms.append(
            config.cross_entropy_weight * ce
            + config.pairwise_weight * pair
            + config.ordinal_weight * ordinal
            + config.mined_weight * mined
        )

    total = torch.stack(total_terms).mean()
    metrics = {
        "loss": total.detach(),
        "ce": torch.stack(ce_terms).mean().detach(),
        "pair": torch.stack(pair_terms).mean().detach(),
        "ordinal": torch.stack(ordinal_terms_all).mean().detach(),
        "mined": torch.stack(mined_terms).mean().detach(),
        "active_max_difficulty": total.new_tensor(float(active_max)),
    }
    return total, metrics


def ranking_metric_sums(score_list: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Return device-side accuracy sum, reciprocal-rank sum and sample count."""

    if not score_list:
        raise ValueError("score_list must be non-empty")
    device = score_list[0].device
    accuracy_sum = torch.zeros((), device=device, dtype=torch.float32)
    reciprocal_rank_sum = torch.zeros((), device=device, dtype=torch.float32)
    for scores in score_list:
        rank = 1 + torch.sum(scores[1:] >= scores[0])
        accuracy_sum = accuracy_sum + (rank == 1).to(torch.float32)
        reciprocal_rank_sum = reciprocal_rank_sum + rank.to(torch.float32).reciprocal()
    return (
        accuracy_sum,
        reciprocal_rank_sum,
        torch.tensor(float(len(score_list)), device=device),
    )


def ranking_metrics(score_list: Sequence[Tensor]) -> tuple[float, float]:
    """Compatibility helper for tests; training uses ``ranking_metric_sums``."""

    accuracy_sum, reciprocal_rank_sum, count = ranking_metric_sums(score_list)
    values = torch.stack(
        [accuracy_sum / count.clamp_min(1.0), reciprocal_rank_sum / count.clamp_min(1.0)]
    ).detach().cpu().tolist()
    return float(values[0]), float(values[1])
