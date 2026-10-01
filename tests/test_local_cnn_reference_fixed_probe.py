"""UNIT tensor fixtures for fixed-weight controls; not CT accuracy evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_execution import rng_state
from l0_regions.donor_learning import LiveContext
from l0_regions.training import hash_state
from tools.local_cnn_reference_fixed_probe import MODES, probe_fixed_weights


ROOT = Path(__file__).resolve().parents[1]


class UnitBudget:
    def __init__(self):
        self.calls = 0

    def check(self):
        self.calls += 1


def fixture(device='cpu'):
    torch.manual_seed(419)
    cfg = json.loads((ROOT/'config/prompt_graph_v222_v1_l0.json').read_text())
    base = json.loads((ROOT/'config/train.json').read_text())
    net = PromptGraphModel(cfg, base, {}, local_encoder=nn.Identity()).to(device).eval()
    net.checkpoint_support = False
    query = torch.randn(32, 128, device=device)
    support = (torch.randn(256, 128, device=device),
               torch.arange(16, device=device).repeat_interleave(16),
               torch.arange(256, device=device).remainder(2))
    rows = [dict(id=f'UNIT{i}', case_id='UNIT_QUERY', patient_group='Q',
        donor_case_id='D', donor_component=0, donor_group='D',
        target=int(i < 5), bounds=dict(edges=1)) for i in range(32)]
    context = LiveContext(SimpleNamespace(rows=rows), 32)
    truth = torch.tensor([row['target'] for row in rows], device=device)
    return net, query, support, context, truth


def execute(net, query, support, context, truth, policies):
    ids = context.order[0]
    return probe_fixed_weights(net, query[ids], support, policies=policies, truth=truth[ids],
        budget=UnitBudget(), loss_context=context, indices=ids)


class FixedReferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_original_state_gradients_mixed_modes_rng_and_inputs_preserved(self):
        net, query, support, context, truth = fixture()
        net.train()
        net.l2[0].eval()
        next(net.parameters()).grad = torch.ones_like(next(net.parameters()))
        before = hash_state(dict(model=net.state_dict(), gradients={n:p.grad for n,p in net.named_parameters()},
            rng=rng_state(), query=query, support=support, truth=truth))
        modes = [m.training for m in net.modules()]
        report = execute(net, query, support, context, truth, ['affine_relations_zero_out_bias'])
        after = hash_state(dict(model=net.state_dict(), gradients={n:p.grad for n,p in net.named_parameters()},
            rng=rng_state(), query=query, support=support, truth=truth))
        self.assertEqual(before, after)
        self.assertEqual(modes, [m.training for m in net.modules()])
        self.assertTrue(report['original_state_gradients_modes_rng_and_inputs_preserved'])
        self.assertFalse(report['optimizer_created'])
        self.assertFalse(report['production_checkpoint_written'])
        self.assertFalse(report['production_ready'])
        json.dumps(report, allow_nan=False)

    def test_bn_one_pass_and_separate_alignment_query_route(self):
        args = fixture()
        report = execute(*args, ['affine_relations_zero_out_bias'])
        legacy, reference = report['branches']
        self.assertFalse(legacy['per_loss_query_gradient']['losses']['alignment']['query_embedding_dependency'])
        self.assertIsNone(legacy['per_loss_query_gradient']['losses']['alignment']['query_embedding_gradient_norm'])
        self.assertTrue(reference['per_loss_query_gradient']['losses']['alignment']['query_embedding_dependency'])
        self.assertGreater(reference['per_loss_query_gradient']['losses']['alignment']['query_embedding_gradient_norm'], 0)
        self.assertEqual(tuple(row['mode'] for row in reference['modes']), MODES)
        fresh, joint, once = reference['modes']
        self.assertEqual(fresh['batch_norm_before'], fresh['batch_norm_after'])
        self.assertEqual(once['batch_norm_before'], once['batch_norm_after'])
        for name in fresh['batch_norm_before']:
            self.assertEqual(fresh['batch_norm_after'][name]['num_batches_tracked'], 0)
            self.assertEqual(joint['batch_norm_after'][name]['num_batches_tracked'], 1)
            self.assertEqual(once['batch_norm_after'][name]['num_batches_tracked'], 1)
            self.assertEqual(reference['per_loss_query_gradient']['batch_norm'][name]['num_batches_tracked'], 1)
        for layer in fresh['layers']:
            self.assertEqual(set(layer['stages']), {'input','aggregate','pre_bn','operator_output','output'})
            self.assertIsInstance(layer['normalized_centered_energy_output_over_input'], float)
        degree = reference['degree_bias']['layers'][0]['roles']
        self.assertEqual(degree['support_data']['degree_min'], 3)
        self.assertEqual(degree['patient_label']['degree_min'], 17)
        self.assertEqual(degree['query']['degree_min'], 33)
        self.assertEqual(degree['query']['official_bias_component_norm_max'], 0)

    def test_teacher_refit_once_per_own_branch_and_shared_l0(self):
        args = fixture()
        original = PromptGraphModel.fit_support_clusters
        calls = []

        def track(model, *support):
            calls.append(model)
            return original(model, *support)

        with patch.object(PromptGraphModel, 'fit_support_clusters', track):
            report = execute(*args, ['raw_columns','affine_relations','affine_relations_zero_out_bias'])
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({id(model) for model in calls}), 4)
        self.assertEqual(len({row['same_initial_L0_embeddings_sha256'] for row in report['branches']}), 1)
        for row in report['branches']:
            self.assertEqual(row['frozen_teacher']['fit_calls'], 1)
            self.assertTrue(row['frozen_teacher']['own_branch'])
            self.assertEqual(len({mode['teacher_plan_sha256'] for mode in row['modes']}), 1)

    def test_evaluation_provider_runs_fresh_eval_and_cannot_mutate_state(self):
        net, query, support, context, truth = fixture()
        seen = []

        def provider(model):
            seen.append((model.training, [int(layer.bn.num_batches_tracked) for layer in model.l1
                                        if hasattr(layer, 'bn')]))
            return {'status':'UNIT', 'scope':'no accuracy claim'}

        ids = context.order[0]
        report = probe_fixed_weights(net, query[ids], support, policies=['raw_columns'], truth=truth[ids],
            budget=UnitBudget(), loss_context=context, indices=ids, evaluation_provider=provider)
        self.assertEqual(seen, [(False, []), (False, [0,0])])
        self.assertEqual(report['branches'][1]['initial_full_case_evaluation']['status'], 'UNIT')

        def corrupt(model):
            with torch.no_grad():
                model.label_seed.add_(1)
            return {'status':'bad'}

        with self.assertRaisesRegex(AssertionError, 'mutated'):
            probe_fixed_weights(net, query[ids], support, policies=['raw_columns'], truth=truth[ids],
                budget=UnitBudget(), loss_context=context, indices=ids, evaluation_provider=corrupt)

    def test_unchanged_pure_unobserved_tile_explicit_not_evaluable(self):
        net, query, support, context, truth = fixture()
        rows = context.rows + [dict(id=f'UNIT{i}', case_id='UNIT_NO_P', patient_group='Q2',
            donor_case_id='D', donor_component=0, donor_group='D',
            target=0, bounds=dict(edges=1)) for i in range(32,36)]
        context = LiveContext(SimpleNamespace(rows=rows), 32)
        ids = next(tile for tile in context.order if rows[tile[0]]['case_id']=='UNIT_NO_P')
        report = probe_fixed_weights(net, query[:4], support, policies=['affine_relations'],
            truth=torch.zeros(4,dtype=torch.long), budget=UnitBudget(), loss_context=context, indices=ids)
        self.assertEqual(report['indices'], ids)
        for branch in report['branches']:
            for mode in branch['modes']:
                self.assertEqual(mode['score']['status'], 'NOT_EVALUABLE')
            gradient = branch['per_loss_query_gradient']
            self.assertEqual(gradient['ranking_pairs'], 0)
            self.assertEqual(gradient['losses']['ranking']['weighted_loss'], 0)
            self.assertEqual(gradient['losses']['ranking']['query_embedding_gradient_norm'], 0)

    def test_reject_nonfinite_and_wrong_truth_without_fallback(self):
        net, query, support, context, truth = fixture()
        bad = query.clone()
        bad[0,0] = float('nan')
        with self.assertRaises(FloatingPointError):
            execute(net, bad, support, context, truth, ['raw_columns'])
        with self.assertRaises(ValueError):
            execute(net, query, support, context, 1-truth, ['raw_columns'])
        with self.assertRaises(ValueError):
            execute(net, query, support, context, truth, [])
        with self.assertRaisesRegex(ValueError, 'Exact ordered tile'):
            probe_fixed_weights(net, query[:4], support, policies=['raw_columns'], truth=truth[:4],
                budget=UnitBudget(), loss_context=context, indices=list(range(4)))


if __name__ == '__main__':
    unittest.main()
