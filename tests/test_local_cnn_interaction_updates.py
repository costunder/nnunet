"""Explicit short UNIT native tensors; no trained CT/CP accuracy assertion."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_execution import rng_state
from l0_local_cnn.model import LocalBatch, LocalCNN, MODE
from l0_regions.donor_learning import LiveContext, configuration
from l0_regions.training import hash_state
from tools.local_cnn_interaction_candidate import clone_candidate
from tools.local_cnn_interaction_updates import probe_candidate_updates


class Budget:
    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail
    def check(self):
        self.calls += 1
        if self.calls == self.fail:
            raise MemoryError('UNIT explicit resource rejection')


def fixture(device):
    torch.set_num_threads(4)
    torch.manual_seed(929)
    rows = []
    for case_id, count in [('UNIT query', 48), ('UNIT other query', 48),
                           ('UNIT S0', 4), ('UNIT S1', 4), ('UNIT S2', 4)]:
        for i in range(count):
            rows.append(dict(id=f'{case_id}:{i}', case_id=case_id, patient_group=case_id,
                donor_case_id='UNIT independent donor', donor_component=1,
                donor_group='UNIT independent donor', target=i % 3 == 0 if count == 48 else i % 2,
                bounds=dict(edges=i)))
    context = LiveContext(SimpleNamespace(rows=rows), 32)
    config = dict(architecture=MODE, channels=[12,24,32], convolutions=[2,3,3], hidden_dim=128,
        margin_mm=10., input='native_spacing_organ_only', readout='organ_masked_mean_each_scale',
        fusion='donor_target_difference_product', initialization='fresh_seed42', learning_policy='same_donor_live_v1')
    base = json.loads(Path('config/train.json').read_text())
    cfg = json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
    local = LocalCNN(config).to(device)
    net = PromptGraphModel(cfg, base, {}, local_encoder=local).to(device).eval()
    net.checkpoint_support = False
    images = torch.randn(len(rows) + 1, 1, 6, 6, 6, device=device)
    support_indices = list(range(96, len(rows)))
    support = (torch.randn(12, 128, device=device), torch.arange(3, device=device).repeat_interleave(4),
               torch.tensor([rows[i]['target'] for i in support_indices], dtype=torch.long, device=device))
    support_ids = [rows[i]['id'] for i in support_indices]
    def batch_provider(ids):
        selection = torch.tensor([0] + [i + 1 for i in ids], device=device)
        values = images.index_select(0, selection)
        audit = [dict(case='UNIT independent donor', origin=[0,0,0], shape=[6,6,6],
                      spacing=[1.,1.,1.], anchor_in_organ=True)]
        audit.extend(dict(case=rows[i]['case_id'], origin=[0,0,0], shape=[6,6,6],
                          spacing=[1.,1.,1.], anchor_in_organ=True) for i in ids)
        return LocalBatch(values, torch.ones_like(values, dtype=torch.bool),
            torch.zeros(len(ids), device=device, dtype=torch.long),
            torch.arange(1, len(ids) + 1, device=device),
            torch.tensor(ids, device=device), audit).validate()
    def support_provider(ids):
        return support, support_ids, rows[ids[0]]['patient_group']
    kwargs = dict(scale=1., steps=2, lr=.0001, train_tiles=context.order,
                  batch_provider=batch_provider, support_provider=support_provider,
                  loss_context=context, settings=configuration(), physical_batch=32,
                  budget=Budget(), weight_decay=.0001, grad_clip=5., seed=42,
                  fused_optimizer=device == 'cuda')
    return net, kwargs, support


class Checks(unittest.TestCase):
    def test_zero_candidate_matches_baseline_training_and_callback_rng_isolated(self):
        net, kwargs, support = fixture('cpu')
        kwargs.update(scale=0., steps=1)
        original = hash_state(net.state_dict())
        modes = tuple(m.training for m in net.modules())
        before_rng = hash_state(rng_state())
        input_hash = hash_state(support)
        def evaluation(model):
            self.assertFalse(model.training)
            return dict(scope='UNIT operator only', random_number=float(torch.rand(())),
                        measure=float(next(model.parameters()).detach().sum()))
        report = probe_candidate_updates(net, **kwargs, evaluation_provider=evaluation)
        self.assertEqual(hash_state(net.state_dict()), original)
        self.assertEqual(tuple(m.training for m in net.modules()), modes)
        self.assertEqual(hash_state(rng_state()), before_rng)
        self.assertEqual(hash_state(support), input_hash)
        baseline, candidate = report['branches']
        self.assertEqual(baseline['cloned_weight_hash_after'], candidate['cloned_weight_hash_after'])
        self.assertEqual(baseline['updates'][0]['loss'], candidate['updates'][0]['loss'])
        self.assertEqual(baseline['updates'][0]['physical_batch'], 32)
        without_eval = probe_candidate_updates(net, **kwargs)
        self.assertEqual(baseline['cloned_weight_hash_after'], without_eval['branches'][0]['cloned_weight_hash_after'])
        progress_options, progress_postfixes = [], []
        class QuietProgress:
            def __init__(self, iterable, **options):
                self.iterable = iterable
                progress_options.append(options)
            def __iter__(self):
                return iter(self.iterable)
            def set_postfix(self, **values):
                progress_postfixes.append(values)
            def close(self):
                pass
        with patch('tools.local_cnn_interaction_updates.tqdm', QuietProgress):
            shown = probe_candidate_updates(net, **kwargs, progress=True)
        self.assertEqual([value['desc'] for value in progress_options], ['CLONED L1 baseline', 'CLONED L1 candidate'])
        self.assertEqual([value['disable'] for value in progress_options], [False, False])
        self.assertEqual([value['actual_batch'] for value in progress_postfixes], [32, 32])
        self.assertEqual(baseline['cloned_weight_hash_after'], shown['branches'][0]['cloned_weight_hash_after'])
        # A caller's autocast must not silently change the diagnostic teacher,
        # native CNN/query forward or loss from the pinned FP32 equation.
        with torch.autocast('cpu', dtype=torch.bfloat16):
            ambient_amp = probe_candidate_updates(net, **kwargs, evaluation_provider=evaluation)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(baseline['cloned_weight_hash_after'], ambient_amp['branches'][0]['cloned_weight_hash_after'])
        self.assertEqual(baseline['updates'][0]['loss'], ambient_amp['branches'][0]['updates'][0]['loss'])
        self.assertEqual(ambient_amp['precision'], 'FP32')
        self.assertFalse(ambient_amp['autocast_enabled'])
        self.assertFalse(ambient_amp['execution_runtime']['autocast_enabled'])
        self.assertIn('does not guarantee bitwise', ambient_amp['stochastic_comparison'])
        self.assertFalse(ambient_amp['timing_scope']['speedup_claim'])
        self.assertTrue(report['original_weights_types_methods_modes_preserved'])
        self.assertTrue(report['caller_rng_restored'])
        self.assertFalse(report['exact_resume'])
        self.assertFalse(report['next_saved_update'])
        self.assertEqual(report['checkpoints_written'], 0)

    def test_each_branch_fits_own_teacher_once_per_patient_episode(self):
        net, kwargs, _ = fixture('cpu')
        fitted = {0.: 0, 1.: 0}
        def tracking_clone(original, scale):
            clone = clone_candidate(original, scale)
            old = clone.fit_support_clusters
            def fit(*args):
                fitted[scale] += 1
                return old(*args)
            clone.fit_support_clusters = fit
            return clone
        with patch('tools.local_cnn_interaction_candidate.clone_candidate', tracking_clone):
            report = probe_candidate_updates(net, **kwargs)
        self.assertEqual(fitted, {0.: 1, 1.: 1})
        self.assertEqual(report['full_schedule_tiles'], len(kwargs['train_tiles']))
        self.assertEqual(report['full_cohort_observations'], len(kwargs['loss_context'].rows))
        for branch in report['branches']:
            self.assertEqual([row['physical_batch'] for row in branch['updates']], [32,32])
            self.assertEqual([row['support_records'] for row in branch['updates']], [12,12])
            for update in branch['updates']:
                self.assertGreater(update['module_gradient_norms']['CNN'], 0)
                self.assertGreater(update['module_gradient_norms']['L1'], 0)
                self.assertGreater(update['module_gradient_norms']['L2'], 0)
                self.assertAlmostEqual(update['loss'], update['terms']['ranking_loss']+
                    update['terms']['observation_auxiliary_loss']+update['weighted_alignment_loss'], places=5)

    def test_incomplete_schedule_and_support_patient_observation_drop_rejected(self):
        net, kwargs, _ = fixture('cpu')
        original = hash_state(net.state_dict())
        with self.assertRaisesRegex(ValueError, 'complete current diagnostic epoch schedule'):
            probe_candidate_updates(net, **(kwargs | dict(train_tiles=kwargs['train_tiles'][:1])))
        provider = kwargs['support_provider']
        def incomplete(ids):
            (vectors, owners, targets), record_ids, group = provider(ids)
            return (vectors[:-1], owners[:-1], targets[:-1]), record_ids[:-1], group
        with self.assertRaisesRegex(ValueError, 'dropped eligible observations'):
            probe_candidate_updates(net, **(kwargs | dict(support_provider=incomplete)))
        self.assertEqual(hash_state(net.state_dict()), original)

    def test_role_mismatch_and_changed_branch_data_fail_without_fallback(self):
        net, kwargs, _ = fixture('cpu')
        provider = kwargs['batch_provider']
        def wrong_role(ids):
            batch = provider(ids)
            batch.audit[0]['case'] = 'UNIT wrong donor'
            return batch.validate()
        with self.assertRaisesRegex(ValueError, 'crop role binding differs'):
            probe_candidate_updates(net, **(kwargs | dict(batch_provider=wrong_role)))
        calls = 0
        def changed(ids):
            nonlocal calls
            batch = provider(ids)
            calls += 1
            if calls > 1:
                batch.images.add_(1)
                batch.validate()
            return batch
        with self.assertRaisesRegex(ValueError, 'native query or eligible support data differ'):
            probe_candidate_updates(net, **(kwargs | dict(steps=1, batch_provider=changed)))

    def test_failure_restores_rng_modes_weights_and_gradient_buffers(self):
        net, kwargs, _ = fixture('cpu')
        for parameter in net.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients = hash_state({name: p.grad for name, p in net.named_parameters()})
        weights = hash_state(net.state_dict())
        before_rng = hash_state(rng_state())
        def bad_evaluation(model):
            next(model.parameters()).add_(1)
            return dict(scope='UNIT deliberate mutation')
        with self.assertRaisesRegex(AssertionError, 'evaluation mutated cloned model'):
            probe_candidate_updates(net, **kwargs, evaluation_provider=bad_evaluation)
        self.assertEqual(hash_state(rng_state()), before_rng)
        self.assertEqual(hash_state(net.state_dict()), weights)
        self.assertEqual(hash_state({name: p.grad for name, p in net.named_parameters()}), gradients)
        with self.assertRaisesRegex(MemoryError, 'UNIT explicit resource rejection'):
            probe_candidate_updates(net, **(kwargs | dict(budget=Budget(fail=2))))
        self.assertEqual(hash_state(rng_state()), before_rng)

    def test_cuda_fused_native32_prefix_connects_original_full_model(self):
        if not torch.cuda.is_available():
            self.skipTest('Explicit CUDA native batch mechanics requires GPU')
        # CUBLAS reads workspace configuration when its context is initialized.
        # Earlier tests in an integrated suite may already have initialized it;
        # use an isolated CUDA process with the setting present before importing
        # PyTorch. This is the same GPU test, not a CPU or reduced-size fallback.
        if os.environ.get('CP_INTERACTION_PARITY_UNIT_CHILD') != '1':
            environment = os.environ.copy()
            environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CP_INTERACTION_PARITY_UNIT_CHILD='1')
            name = 'tests.test_local_cnn_interaction_updates.Checks.test_cuda_fused_native32_prefix_connects_original_full_model'
            completed = subprocess.run([sys.executable, '-B', '-m', 'unittest', name, '-v'],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                text=True, capture_output=True, check=False)
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            self.assertIn('Ran 1 test', output)
            self.assertIn('... ok', output)
            self.assertNotIn('skipped', output.lower())
            return
        # Exact AMP-vs-FP32 hashes require deterministic CUDA reduction order.
        # Without this, even two normal FP32 runs can differ through atomic
        # index-add backward. Keep the caller's execution policy unchanged.
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                torch.use_deterministic_algorithms(True)
                self._cuda_fused_native32_precision_parity()
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    def _cuda_fused_native32_precision_parity(self):
        net, kwargs, _ = fixture('cuda')
        report = probe_candidate_updates(net, **kwargs)
        with torch.autocast('cuda', dtype=torch.float16):
            under_amp = probe_candidate_updates(net, **kwargs)
            self.assertTrue(torch.is_autocast_enabled('cuda'))
        self.assertEqual(report['branches'][0]['cloned_weight_hash_after'], under_amp['branches'][0]['cloned_weight_hash_after'])
        self.assertEqual(report['branches'][1]['cloned_weight_hash_after'], under_amp['branches'][1]['cloned_weight_hash_after'])
        self.assertTrue(report['optimizer']['fused'])
        self.assertTrue(report['execution_runtime']['deterministic_algorithms'])
        self.assertFalse(report['execution_runtime']['autocast_enabled'])
        self.assertEqual(report['execution_runtime']['cublas_workspace_config'], ':4096:8')
        self.assertEqual(report['optimizer']['history'], 'fresh reset; no saved optimizer moments')
        self.assertEqual(report['cloned_optimizer_updates_per_branch'], 2)
        self.assertEqual(report['production_optimizer_updates'], 0)
        for branch in report['branches']:
            self.assertGreater(branch['observed_cuda_peak_bytes'], 0)
            self.assertGreater(branch['module_parameter_delta_norms']['CNN'], 0)
            self.assertGreater(branch['module_parameter_delta_norms']['readout_fusion'], 0)
            self.assertEqual([r['physical_batch'] for r in branch['updates']], [32,32])
            for update in branch['updates']:
                self.assertGreater(update['module_gradient_norms']['CNN'], 0)
                self.assertGreater(update['synchronized_update_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
