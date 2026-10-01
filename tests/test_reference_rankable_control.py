"""UNIT checks for explicit native ranking controls; no CT performance claims.

Selection fixtures preserve a complete LiveContext, including pure-class and
natural partial tiles. Scalar gradient fixtures exercise the ranking-only gate
without a model forward, optimizer update, or training experiment.
"""
import copy
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from l0_regions.donor_learning import LiveContext
from l0_regions.training import hash_state
from tools.local_cnn_reference_runtime import (
    probe_updates,
    ranking_parameter_gradients,
    select_update_tiles,
)


def context_fixture(batch=32):
    rows = []
    # Insertion order deliberately starts with zero-pair tiles. Each subsequent
    # case uses every P x U pair under the existing complete-cohort tiling rule.
    for case, positives, unobserved in (
        ('pure_U', 0, 32), ('pure_P', 32, 0),
        ('mixed_A', 5, 65), ('mixed_B', 3, 57), ('mixed_C', 2, 30),
    ):
        for offset in range(positives + unobserved):
            rows.append(dict(
                id=f'UNIT_{case}_{offset}', case_id=case, patient_group=case,
                donor_case_id='UNIT_DONOR', donor_component=0,
                donor_group='UNIT_DONOR', target=int(offset < positives),
                bounds=dict(edges=offset),
            ))
    return LiveContext(SimpleNamespace(rows=rows), batch)


def context_snapshot(context):
    return copy.deepcopy(dict(
        rows=context.rows, order=context.order, uses=context.uses,
        counts=context.counts, pairs=context.pairs, steps=context.steps,
        audit=context.audit,
    ))


def full_mixed_indices(tiles, context, batch):
    return [position for position, tile in enumerate(tiles)
            if len(tile) == batch
            and any(context.rows[i]['target'] == 1 for i in tile)
            and any(context.rows[i]['target'] == 0 for i in tile)]


class NativeRankableSelectionChecks(unittest.TestCase):
    def test_explicit_rankable_filter_retains_native_tiles_and_complete_context(self):
        context = context_fixture()
        tiles = copy.deepcopy(context.order)
        before = context_snapshot(context)
        original_tiles = copy.deepcopy(tiles)
        expected = full_mixed_indices(tiles, context, 32)
        self.assertEqual(len(expected), 4)
        self.assertEqual(sum(context.rows[i]['target'] for i in tiles[0]), 0)
        self.assertEqual(sum(context.rows[i]['target'] for i in tiles[1]), 32)
        self.assertTrue(any(len(tile) < 32 for tile in tiles))

        report = select_update_tiles(
            tiles, steps=4, loss_context=context, physical_batch=32,
            selection_policy='rankable_full_batch_prefix',
        )
        self.assertEqual(report['selection_policy'], 'rankable_full_batch_prefix')
        self.assertEqual(report['selected_schedule_indices'], expected)
        self.assertEqual(report['eligible_tile_count'], len(expected))
        self.assertEqual(report['full_schedule_tiles'], context.steps)
        self.assertEqual(report['full_cohort_observations'], len(context.rows))
        self.assertEqual(report['full_schedule_audit'], before['audit'])
        normalization = report['normalization']
        self.assertEqual(normalization['steps'], context.steps)
        self.assertEqual(normalization['pairs'], context.pairs)
        self.assertEqual(normalization['class_counts'], dict(context.counts))
        self.assertEqual(normalization['observation_multiplicities_sha256'],
                         hash_state(dict(context.uses)))
        self.assertEqual(normalization['ranking_coefficient'], context.steps / context.pairs)
        self.assertNotEqual(normalization['steps'], 4)
        for position, selected in zip(expected, report['selected_tiles']):
            native = original_tiles[position]
            observed = sum(context.rows[i]['target'] for i in native)
            self.assertEqual(selected['schedule_index'], position)
            self.assertEqual(selected['indices'], native)
            self.assertEqual(selected['physical_batch'], 32)
            self.assertEqual(selected['observed'], observed)
            self.assertEqual(selected['unobserved'], 32 - observed)
            self.assertEqual(selected['ranking_pairs'], observed * (32 - observed))
            self.assertGreater(selected['ranking_pairs'], 0)
        self.assertEqual(tiles, original_tiles)
        self.assertEqual(context_snapshot(context), before)

        # The report is not permitted to alias the complete audit and alter the
        # original normalization contract through a caller's later mutation.
        report['full_schedule_audit']['actual_batch_sizes'].append(999)
        self.assertEqual(context_snapshot(context), before)

    def test_default_complete_prefix_preserves_historical_zero_pair_tiles(self):
        context = context_fixture()
        report = select_update_tiles(
            context.order, steps=2, loss_context=context, physical_batch=32,
        )
        self.assertEqual(report['selection_policy'], 'complete_prefix')
        self.assertEqual(report['selected_schedule_indices'], [0, 1])
        self.assertEqual([row['indices'] for row in report['selected_tiles']], context.order[:2])
        self.assertEqual([row['ranking_pairs'] for row in report['selected_tiles']], [0, 0])
        self.assertEqual(inspect.signature(select_update_tiles).parameters['selection_policy'].default,
                         'complete_prefix')
        complete = select_update_tiles(
            context.order, steps=context.steps, loss_context=context, physical_batch=32,
        )
        self.assertEqual([row['indices'] for row in complete['selected_tiles']], context.order)
        self.assertTrue(any(row['physical_batch'] < 32 for row in complete['selected_tiles']))

    def test_selection_is_deterministic_in_the_original_supplied_schedule(self):
        context = context_fixture()
        # A caller may supply its already shuffled complete epoch schedule.
        # Selection keeps that schedule's order and never sorts by scores.
        tiles = copy.deepcopy(list(reversed(context.order)))
        original = copy.deepcopy(tiles)
        before = context_snapshot(context)
        expected = full_mixed_indices(tiles, context, 32)[:3]
        left = select_update_tiles(
            tiles, steps=3, loss_context=context, physical_batch=32,
            selection_policy='rankable_full_batch_prefix',
        )
        right = select_update_tiles(
            tiles, steps=3, loss_context=context, physical_batch=32,
            selection_policy='rankable_full_batch_prefix',
        )
        self.assertEqual(left, right)
        self.assertEqual(left['selected_schedule_indices'], expected)
        self.assertEqual([row['indices'] for row in left['selected_tiles']], [tiles[i] for i in expected])
        self.assertEqual(tiles, original)
        self.assertEqual(context_snapshot(context), before)

    def test_requested_updates_are_never_silently_shortened_or_replayed(self):
        context = context_fixture()
        before = context_snapshot(context)
        with self.assertRaises(ValueError):
            select_update_tiles(
                context.order, steps=5, loss_context=context, physical_batch=32,
                selection_policy='rankable_full_batch_prefix',
            )
        self.assertEqual(context_snapshot(context), before)

    def test_no_full_mixed_tile_fails_without_replacing_native_candidates(self):
        context = context_fixture(batch=128)
        # This is still a valid full ranking cohort, but every mixed native
        # tile is a natural partial batch. No synthesized physical128 tile is
        # allowed, and repeating rows cannot manufacture eligibility.
        self.assertGreater(context.pairs, 0)
        self.assertEqual(full_mixed_indices(context.order, context, 128), [])
        before = context_snapshot(context)
        with self.assertRaises(ValueError):
            select_update_tiles(
                context.order, steps=1, loss_context=context, physical_batch=128,
                selection_policy='rankable_full_batch_prefix',
            )
        self.assertEqual(context_snapshot(context), before)

    def test_invalid_requests_and_changed_full_schedule_are_explicit_errors(self):
        context = context_fixture()
        for steps in (0, -1, True, 1.0, len(context.order) + 1):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                select_update_tiles(context.order, steps=steps,
                                    loss_context=context, physical_batch=32)
        for policy in (None, '', 'automatic', 'positive_only', 1):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                select_update_tiles(context.order, steps=1,
                                    loss_context=context, physical_batch=32,
                                    selection_policy=policy)
        for altered in (context.order[:-1], context.order + [context.order[0]],
                        [context.order[0]] + context.order[:-1]):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                select_update_tiles(altered, steps=1,
                                    loss_context=context, physical_batch=32)
        with self.assertRaises(ValueError):
            select_update_tiles(context.order, steps=1,
                                loss_context=context, physical_batch=16)

    def test_transfer_is_explicit_and_invalid_policy_fails_before_queries_or_clones(self):
        self.assertEqual(inspect.signature(probe_updates).parameters['transfer_policy'].default,
                         'raw_columns')
        self.assertEqual(inspect.signature(probe_updates).parameters['selection_policy'].default,
                         'complete_prefix')
        context = context_fixture()
        batch_provider, support_provider, evaluation_provider = Mock(), Mock(), Mock()
        with patch('tools.local_cnn_reference_l1.clone_reference') as historical, \
                patch('tools.local_cnn_reference_transfer.clone_reference_control') as controlled:
            for policy in (None, '', 'automatic', 'affine', 1):
                with self.subTest(policy=policy), self.assertRaises(ValueError):
                    probe_updates(
                        object(), steps=1, train_tiles=context.order,
                        batch_provider=batch_provider, support_provider=support_provider,
                        loss_context=context, physical_batch=32, budget=Mock(),
                        training={}, seed=7, evaluation_provider=evaluation_provider,
                        selection_policy='rankable_full_batch_prefix', transfer_policy=policy,
                        progress=False,
                    )
            historical.assert_not_called()
            controlled.assert_not_called()
        batch_provider.assert_not_called()
        support_provider.assert_not_called()
        evaluation_provider.assert_not_called()


def gradient_fixture():
    cnn = nn.Parameter(torch.tensor([2., -3.]))
    l1 = nn.Parameter(torch.tensor([4.]))
    l2 = nn.Parameter(torch.tensor([5., 6.]))
    unused = nn.Parameter(torch.tensor([7.]))
    named = [('local.cnn.weight', cnn), ('l1.0.weight', l1),
             ('l2.0.weight', l2), ('l2_updates.0.weight', unused)]
    return named, cnn, l1, l2, unused


class RankingGradientGateChecks(unittest.TestCase):
    def test_ranking_only_gradient_is_measured_without_changing_buffers_or_graph(self):
        named, cnn, l1, l2, unused = gradient_fixture()
        for _, parameter in named:
            parameter.grad = torch.full_like(parameter, 19.)
        before = [parameter.grad.clone() for _, parameter in named]
        ranking = cnn.square().sum() + l1.square().sum() + l2.square().sum()
        report = ranking_parameter_gradients(ranking, named, require_nonzero=True)
        self.assertEqual(report['status'], 'MEASURED')
        self.assertEqual(report['required_nonzero_modules'], ['CNN', 'L1', 'L2'])
        self.assertAlmostEqual(report['module_gradient_norms']['CNN'], float((2 * cnn).detach().norm()), places=6)
        self.assertAlmostEqual(report['module_gradient_norms']['L1'], 8., places=6)
        self.assertAlmostEqual(report['module_gradient_norms']['L2'], float((2 * l2).detach().norm()), places=6)
        self.assertAlmostEqual(report['module_gradient_norms']['global'],
                               float(torch.cat((2 * cnn, 2 * l1, 2 * l2)).detach().norm()), places=6)
        self.assertEqual(report['module_parameter_connections']['L2'],
                         dict(trainable_parameters=2, connected_parameters=1))
        for old, (_, parameter) in zip(before, named):
            torch.testing.assert_close(parameter.grad, old, rtol=0, atol=0)
        # The production objective must still be able to backward after the
        # gate. In this unit fixture the auxiliary-only term touches unused.
        (ranking + unused.square().sum()).backward()
        torch.testing.assert_close(cnn.grad, before[0] + 2 * cnn)
        torch.testing.assert_close(unused.grad, before[3] + 2 * unused)

    def test_auxiliary_route_cannot_mask_missing_ranking_module_connections(self):
        named, cnn, l1, l2, _ = gradient_fixture()
        ranking = l1.square().sum()
        auxiliary = cnn.square().sum() + l2.square().sum()
        # The total objective connects all modules; the ranking-only gate must
        # independently reject the missing CNN and L2 ranking routes.
        total_grads = torch.autograd.grad(ranking + auxiliary,
                                         (cnn, l1, l2), retain_graph=True)
        self.assertTrue(all(bool(gradient.abs().sum() > 0) for gradient in total_grads))
        report = ranking_parameter_gradients(ranking, named)
        self.assertIsNone(report['module_gradient_norms']['CNN'])
        self.assertIsNone(report['module_gradient_norms']['L2'])
        with self.assertRaises(ValueError):
            ranking_parameter_gradients(ranking, named, require_nonzero=True)
        self.assertTrue(all(parameter.grad is None for _, parameter in named))

    def test_connected_but_zero_ranking_gradient_is_rejected(self):
        named, cnn, l1, l2, _ = gradient_fixture()
        ranking = (cnn * 0).sum() + l1.square().sum() + l2.square().sum()
        report = ranking_parameter_gradients(ranking, named)
        self.assertEqual(report['module_gradient_norms']['CNN'], 0.)
        with self.assertRaises(ValueError):
            ranking_parameter_gradients(ranking, named, require_nonzero=True)

    def test_nonfinite_ranking_gradient_or_loss_is_rejected(self):
        for mode in ('nonfinite_gradient', 'nonfinite_loss'):
            named, cnn, l1, l2, _ = gradient_fixture()
            if mode == 'nonfinite_gradient':
                # sqrt(0) has a finite value and infinite derivative.
                ranking = (cnn - cnn.detach()).sum().sqrt() + l1.sum() + l2.sum()
                self.assertTrue(bool(torch.isfinite(ranking)))
            else:
                ranking = cnn.sum() * float('inf') + l1.sum() + l2.sum()
            expected_error = FloatingPointError if mode == 'nonfinite_gradient' else ValueError
            with self.subTest(mode=mode), self.assertRaises(expected_error):
                ranking_parameter_gradients(ranking, named, require_nonzero=True)


if __name__ == '__main__':
    unittest.main()
