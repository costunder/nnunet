"""DEBUG synthetic tensors only: vectorized curriculum versus scalar reference.

No medical data, checkpoint, training run, or production configuration is used.
The deliberately scalar reference is test-only and freezes the prior objective.
"""

from __future__ import annotations

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss, ranking_metric_sums


def scalar_reference(score_list, difficulty_list, *, epoch, config):
    active_max = 1 if epoch <= config.easy_epochs else 2 if epoch <= config.inter_epochs else 3
    rows = []
    for scores, difficulty in zip(score_list, difficulty_list):
        negatives, difficulty = scores[1:], difficulty[1:]
        active = (difficulty >= 1) & (difficulty <= active_max)

        def mean(values, mask):
            weights = mask.to(values.dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1)

        logits = torch.cat([scores[:1], negatives.masked_fill(~active, -torch.inf)])
        ce = F.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long))
        margins = torch.full_like(difficulty, config.easy_margin, dtype=torch.float32)
        margins = torch.where(difficulty == 2, config.inter_margin, margins)
        margins = torch.where(difficulty == 3, config.intra_margin, margins)
        pair = mean(F.softplus(margins - scores[0] + negatives), active)
        group_masks = [(difficulty == level) & (level <= active_max) for level in (1, 2, 3)]
        group_means = [mean(negatives, mask) for mask in group_masks]
        ordinal_terms = []
        for left, right in ((0, 1), (1, 2)):
            if group_masks[left].any() and group_masks[right].any():
                ordinal_terms.append(F.softplus(
                    config.ordinal_margin - group_means[right] + group_means[left]
                ))
        ordinal = torch.stack(ordinal_terms).mean() if ordinal_terms else scores.new_zeros(())
        mined = scores.new_zeros(())
        if epoch >= config.model_mine_start_epoch and negatives.numel() >= 3:
            detached = negatives.detach().float()
            low = torch.quantile(detached, config.semi_hard_low_percentile)
            high = torch.quantile(detached, config.semi_hard_high_percentile)
            mined = mean(F.softplus(config.intra_margin - scores[0] + negatives),
                         (detached >= low) & (detached <= high))
        total = (config.cross_entropy_weight * ce + config.pairwise_weight * pair
                 + config.ordinal_weight * ordinal + config.mined_weight * mined)
        rows.append(torch.stack([total, ce, pair, ordinal, mined]))
    return torch.stack(rows).mean(0)


@unittest.skipUnless(TORCH_AVAILABLE, "DEBUG tensor tests require PyTorch")
class VectorizedCurriculumDebugTests(unittest.TestCase):
    def fixtures(self, dtype=None):
        dtype = torch.float64 if dtype is None else dtype
        generator = torch.Generator().manual_seed(921)
        lengths = (2, 4, 7, 11, 5)
        scores = [torch.randn(n, generator=generator, dtype=dtype).requires_grad_()
                  for n in lengths]
        difficulties = [torch.tensor([0] + [1 + i % 3 for i in range(n - 1)])
                        for n in lengths]
        return scores, difficulties

    def test_debug_scalar_value_and_gradient_equivalence_all_stages(self):
        config = CurriculumConfig()
        for epoch in (1, 8, 9, 18, 19, 28, 29, 40):
            with self.subTest(epoch=epoch):
                actual_scores, difficulties = self.fixtures()
                reference_scores = [s.detach().clone().requires_grad_() for s in actual_scores]
                total, metrics = curriculum_ranking_loss(
                    actual_scores, difficulties, epoch=epoch, config=config,
                )
                expected = scalar_reference(reference_scores, difficulties, epoch=epoch, config=config)
                values = torch.stack([total, metrics["ce"], metrics["pair"],
                                      metrics["ordinal"], metrics["mined"]])
                torch.testing.assert_close(values, expected, rtol=1e-7, atol=1e-9)
                actual_grads = torch.autograd.grad(total, actual_scores)
                expected_grads = torch.autograd.grad(expected[0], reference_scores)
                for actual, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, reference, rtol=1e-7, atol=1e-9)

    def test_debug_quantiles_use_real_case_length_and_ties(self):
        scores = [torch.tensor([1., 2., 2., 2., 2.], requires_grad=True),
                  torch.tensor([-1., 8.], requires_grad=True),
                  torch.tensor([0., -20., -4., 1., 3., 10., 12.], requires_grad=True)]
        difficulties = [torch.tensor([0] + [3] * (s.numel() - 1)) for s in scores]
        config = CurriculumConfig()
        actual, stats = curriculum_ranking_loss(scores, difficulties, epoch=30, config=config)
        expected = scalar_reference(scores, difficulties, epoch=30, config=config)
        torch.testing.assert_close(actual, expected[0])
        torch.testing.assert_close(stats["mined"], expected[4])

    def test_debug_missing_difficulty_groups_and_inactive_negatives(self):
        scores = [torch.tensor([1., 3., -1.], requires_grad=True),
                  torch.tensor([2., -3., 0., 1.], requires_grad=True)]
        difficulties = [torch.tensor([0, 3, 3]), torch.tensor([0, 1, 1, 1])]
        for epoch in (1, 10, 30):
            actual, _ = curriculum_ranking_loss(scores, difficulties, epoch=epoch, config=CurriculumConfig())
            expected = scalar_reference(scores, difficulties, epoch=epoch, config=CurriculumConfig())
            torch.testing.assert_close(actual, expected[0])
            self.assertTrue(torch.isfinite(actual))

    def test_debug_batch_permutation_and_padding_invariance(self):
        scores, difficulties = self.fixtures(torch.float32)
        actual, _ = curriculum_ranking_loss(scores, difficulties, epoch=30, config=CurriculumConfig())
        reversed_loss, _ = curriculum_ranking_loss(
            scores[::-1], difficulties[::-1], epoch=30, config=CurriculumConfig(),
        )
        individual = [curriculum_ranking_loss([s], [d], epoch=30, config=CurriculumConfig())[0]
                      for s, d in zip(scores, difficulties)]
        torch.testing.assert_close(actual, reversed_loss)
        torch.testing.assert_close(actual, torch.stack(individual).mean())

    def test_debug_ranking_ties_and_variable_lengths(self):
        scores = [torch.tensor([3., 3., 4.]), torch.tensor([1.]), torch.tensor([4., 1.])]
        accuracy, reciprocal, count = ranking_metric_sums(scores)
        torch.testing.assert_close(accuracy, torch.tensor(2.))
        torch.testing.assert_close(reciprocal, torch.tensor(2. + 1. / 3.))
        torch.testing.assert_close(count, torch.tensor(3.))

    def test_debug_rejects_empty_and_misaligned_cases(self):
        with self.assertRaises(ValueError):
            curriculum_ranking_loss([], [], epoch=1, config=CurriculumConfig())
        with self.assertRaises(ValueError):
            curriculum_ranking_loss([torch.ones(2)], [torch.ones(3)], epoch=1, config=CurriculumConfig())
        with self.assertRaises(ValueError):
            curriculum_ranking_loss([torch.ones(1)], [torch.ones(1)], epoch=1, config=CurriculumConfig())


if __name__ == "__main__":
    unittest.main()
