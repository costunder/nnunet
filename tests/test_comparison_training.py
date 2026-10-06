"""CPU UNIT scalar/policy/receipt checks only; no model or CUDA execution."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x import comparison_training as comparison
from hiercp_v1x import u_bridge_training as engine


class ObjectiveUNIT(unittest.TestCase):
    def test_listwise_problem_mean_and_original_consistency_gradients(self):
        logits = [torch.zeros(8, requires_grad=True) for _ in range(2)]
        consistency = torch.tensor(.25, requires_grad=True)
        loss, terms = comparison.objective(logits, consistency, arm='native_listwise')
        self.assertAlmostEqual(float(terms['ranking'].detach()), math.log(8), places=6)
        self.assertAlmostEqual(float(loss.detach()), math.log(8) + .025, places=6)
        loss.backward()
        for scores in logits:
            self.assertAlmostEqual(float(scores.grad[0]), -7/16, places=7)
            self.assertTrue(torch.equal(scores.grad[1:], torch.full((7,), 1/16)))
            self.assertEqual(float(scores.grad.sum()), 0.)
        self.assertAlmostEqual(float(consistency.grad), .1, places=7)

    def test_three_pairwise_arms_preserve_exact_original_objective(self):
        for arm in ('selected', 'native', 'native_fixed'):
            values = [torch.linspace(-2, 3, 8, requires_grad=True),
                      torch.linspace(4, -1, 8, requires_grad=True)]
            consistency = torch.tensor(.75, requires_grad=True)
            expected, expected_terms = engine.pair_objective(values, consistency)
            actual, actual_terms = comparison.objective(values, consistency, arm=arm)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_terms['ranking'], expected_terms['ranking']))
            expected_grad = torch.autograd.grad(expected, [*values, consistency], retain_graph=True)
            actual_grad = torch.autograd.grad(actual, [*values, consistency])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad)))

    def test_raw_listwise_scale_is_disclosed_and_no_rescaling_occurs(self):
        scores = torch.zeros(8, requires_grad=True)
        loss, _ = comparison.objective([scores], torch.tensor(0.), arm='native_listwise')
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), math.log(8), places=6)
        self.assertAlmostEqual(float(scores.grad[0]), -7/8, places=7)
        policy = comparison.arm_policy('native_listwise')
        self.assertIs(policy['natural_loss_scale_matched'], False)
        self.assertIn('different natural loss/gradient scales', policy['loss_scale_confound'])
        self.assertEqual(policy['objective'], 'listwise_cross_entropy')

    def test_listwise_rejects_other_candidate_counts_and_missing_consistency(self):
        for size in (1, 7, 9, 129):
            with self.assertRaises(ValueError):
                comparison.objective([torch.zeros(size)], torch.tensor(0.), arm='native_listwise')
        with self.assertRaises(ValueError):
            comparison.objective([], torch.tensor(0.), arm='native_listwise')
        with self.assertRaises(ValueError):
            comparison.objective([torch.zeros(8)], torch.zeros(2), arm='native_listwise')


class CandidatePolicyUNIT(unittest.TestCase):
    def test_forty_epoch_exposure_fixed7_vs_rotating128(self):
        for arm, count in (('selected', 7), ('native', 128),
                           ('native_fixed', 7), ('native_listwise', 128)):
            union = set()
            for epoch in range(1, 41):
                keys = comparison.expected_keys(arm, epoch)
                self.assertEqual(keys[0], 'P')
                self.assertEqual(len(keys), 8)
                self.assertEqual(len(set(keys)), 8)
                union.update(keys[1:])
            self.assertEqual(len(union), count)
            self.assertEqual(comparison.arm_policy(arm)['expected_unique_U_per_source'], count)
        self.assertEqual(comparison.expected_keys('native', 2)[1:], tuple(f'U:{i}' for i in range(7, 14)))
        self.assertEqual(comparison.expected_keys('native', 19)[1:], ('U:126', 'U:127', 'U:0', 'U:1', 'U:2', 'U:3', 'U:4'))
        self.assertEqual(comparison.expected_keys('native_fixed', 40), comparison.expected_keys('native_fixed', 1))

    def test_full129_is_fixed_for_all_arms_and_each_has_one_source_P(self):
        expected = ('P', *(f'U:{i}' for i in range(128)))
        for arm in comparison.ARMS:
            self.assertEqual(comparison.expected_keys(arm, 0, full=True), expected)
            self.assertEqual(comparison.expected_keys(arm, 40, full=True), expected)
            with self.assertRaises(ValueError): comparison.expected_keys(arm, 0)

    def test_fixed_candidate_proxy_keeps_actual_view_epoch_and_source_ids(self):
        class Provider:
            global_rng_free = True
            def __init__(self): self.calls = []
            def candidate_keys(self, index, arm, epoch, full=False):
                return comparison.expected_keys(arm, epoch, full=full)
            def batch(self, indices, arm, epoch, training, full=False):
                self.calls.append((list(indices), arm, epoch, training, full))
                return SimpleNamespace(bridge_indices=tuple(indices),
                    bridge_candidate_keys=tuple(self.candidate_keys(i, arm, epoch, full=full) for i in indices))
        original = Provider(); provider = comparison._PolicyProvider(original, 'native_fixed')
        batch = provider.batch([3, 19], 'native_fixed', 29, True)
        self.assertEqual(original.calls, [([3, 19], 'native_fixed', 29, True, False)])
        self.assertEqual(batch.bridge_candidate_keys, (comparison.expected_keys('native_fixed', 29),) * 2)
        self.assertIs(provider.global_rng_free, True)
        with self.assertRaises(ValueError): provider.batch([3], 'native', 29, True)
        original.candidate_keys = lambda *args, **kwargs: comparison.expected_keys('native', 29)
        with self.assertRaises(ValueError): provider.batch([3], 'native_fixed', 29, True)


class FrozenBindingUNIT(unittest.TestCase):
    def test_private_namespaces_reuse_exact_code_and_never_mutate_engine(self):
        names = ('ARMS', 'FORMAT', 'pair_objective', '_write_new', '_append',
                 'atomic_save', 'run_arm', 'calibrate_batches')
        before = {name: getattr(engine, name) for name in names}
        source_hash = hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()
        bindings = []
        for arm in comparison.ARMS:
            functions = comparison._engine_functions(comparison.arm_policy(arm))
            bindings.append(functions['run_arm'].__globals__)
            for name in ('run_arm', 'calibrate_batches'):
                self.assertIs(functions[name].__code__, getattr(engine, name).__code__)
                self.assertEqual(functions[name].__kwdefaults__, getattr(engine, name).__kwdefaults__)
            self.assertEqual(functions['run_arm'].__globals__['FORMAT'], comparison.FORMAT)
            self.assertEqual(functions['run_arm'].__globals__['ARMS'], comparison.ARMS)
            with patch('torch.cuda.is_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'actual CUDA'):
                    functions['calibrate_batches'](None, None, {}, arm=arm, candidates=[2],
                                                  workers=2, budget={}, debug=True)
        self.assertEqual(len({id(value) for value in bindings}), 4)
        for name in names: self.assertIs(getattr(engine, name), before[name])
        self.assertEqual(hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(), source_hash)
        self.assertEqual(source_hash, comparison.ENGINE_SHA256)

    def test_policy_seal_binds_loss_schedule_and_source_bytes(self):
        policies = [comparison.arm_policy(arm) for arm in comparison.ARMS]
        self.assertEqual(len({policy['sha256'] for policy in policies}), 4)
        for policy in policies:
            payload = copy.deepcopy(policy); expected = payload.pop('sha256')
            actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False).encode()).hexdigest()
            self.assertEqual(expected, actual)
            self.assertEqual(policy['facade_sha256'], hashlib.sha256(Path(comparison.__file__).read_bytes()).hexdigest())
            self.assertEqual(policy['engine_sha256'], comparison.ENGINE_SHA256)
            self.assertIs(policy['architecture_changed'], False)
            self.assertIs(policy['optimizer_or_schedule_changed'], False)

    def test_changed_policy_configuration_and_calibration_fail_before_execution(self):
        native = comparison.arm_policy('native'); listwise = comparison.arm_policy('native_listwise')
        config = {'comparison_policy': native}
        original = copy.deepcopy(config)
        with self.assertRaisesRegex(ValueError, 'no silent resume migration'):
            comparison._bind_config(config, listwise)
        self.assertEqual(config, original)
        receipt = dict(format=comparison.FORMAT, comparison_policy=native)
        config = {'u_bridge_runtime': {'batch_calibration': receipt}}
        for arm in ('native_fixed', 'native_listwise'):
            with self.assertRaisesRegex(ValueError, 'calibration belongs to another'):
                comparison.run_arm(None, None, config, arm=arm, output='unused', physical_batch=2,
                    workers=2, epochs=40, identity={}, budget={})
        config['u_bridge_runtime']['batch_calibration'] = dict(format=engine.FORMAT, comparison_policy=listwise)
        with self.assertRaisesRegex(ValueError, 'calibration belongs to another'):
            comparison.run_arm(None, None, config, arm='native_listwise', output='unused',
                physical_batch=2, workers=2, epochs=40, identity={}, budget={})
        config['u_bridge_runtime']['batch_calibration'] = dict(format=comparison.FORMAT, comparison_policy=listwise)
        with self.assertRaisesRegex(ValueError, 'Identity comparison policy differs'):
            comparison.run_arm(None, None, config, arm='native_listwise', output='unused',
                physical_batch=2, workers=2, epochs=40, identity={'comparison_policy': native}, budget={})

    def test_cpu_training_fallback_remains_rejected(self):
        policy = comparison.arm_policy('native_fixed')
        config = {'u_bridge_runtime': {'batch_calibration': dict(format=comparison.FORMAT, comparison_policy=policy)}}
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU training fallback'):
                comparison.run_arm(None, None, config, arm='native_fixed', output='unused',
                    physical_batch=2, workers=2, epochs=40, identity={}, budget={})


class ReceiptUNIT(unittest.TestCase):
    def test_fixed_coverage_saved_returned_and_resume_verification_preserved(self):
        original = dict(trained_comparisons=dict(counts_by_source={'3': 7, '19': 7},
                            keys_by_source={'3': [f'U:{i}' for i in range(7)], '19': [f'U:{i}' for i in range(7)]},
                            expected_unique_U_per_source=128, complete_expected_coverage=False),
                        resume=dict(completed_checkpoint_no_repeat_verified=True,
                                    invocation_optimizer_updates=0, invocation_backward_attempts=0))
        policy = comparison.arm_policy('native_fixed')
        before = copy.deepcopy(original)
        report = comparison._normalize_report(original, policy)
        self.assertEqual(report['trained_comparisons']['expected_unique_U_per_source'], 7)
        self.assertIs(report['trained_comparisons']['complete_expected_coverage'], True)
        self.assertEqual(report['resume'], original['resume'])
        self.assertEqual(original, before)
        original['trained_comparisons']['counts_by_source']['19'] = 6
        self.assertIs(comparison._normalize_report(original, policy)['trained_comparisons']['complete_expected_coverage'], False)
        original['trained_comparisons']['counts_by_source']['19'] = 7
        functions = comparison._engine_functions(policy)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / 'training_complete.json'
            functions['run_arm'].__globals__['_write_new'](path, original)
            self.assertEqual(json.loads(path.read_text())['trained_comparisons'], report['trained_comparisons'])

    def test_listwise_execution_and_update_receipts_name_actual_objective(self):
        policy = comparison.arm_policy('native_listwise')
        namespace = comparison._engine_functions(policy)['run_arm'].__globals__
        original = {'loss': 'old pairwise description', 'format': comparison.FORMAT}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / 'execution_contract.json'
            namespace['_write_new'](path, original)
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt['loss'], policy['loss'])
            self.assertEqual(receipt['ranking_term'], 'listwise_cross_entropy')
            self.assertEqual(receipt['expected_unique_U_per_source'], 128)
            update = {'ranking': 2.1, 'loss': 2.2, 'candidate_keys': [['P', *(f'U:{i}' for i in range(7))]]}
            log = Path(directory) / 'updates.jsonl'; namespace['_append'](log, update)
            row = json.loads(log.read_text())
            self.assertEqual(row['ranking_term'], 'listwise_cross_entropy')
            self.assertEqual(row['comparison_policy_sha256'], policy['sha256'])
            self.assertNotIn('objective_kind', update)
        self.assertEqual(original, {'loss': 'old pairwise description', 'format': comparison.FORMAT})

    def test_checkpoint_policy_addition_reseals_exact_engine_payload_without_state_change(self):
        policy = comparison.arm_policy('native_fixed')
        namespace = comparison._engine_functions(policy)['run_arm'].__globals__
        original = dict(format=comparison.FORMAT, identity_sha256='unit-only',
            model={'weight': torch.tensor([1., 2.])}, optimizer={'state': {1: {'step': torch.tensor(3.)}}},
            state={'phase': 'training', 'updates': 3, 'position': 6}, shuffle_generator=torch.arange(5))
        original['content_sha256'] = engine.digest(original)
        before = engine.digest(original)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / 'checkpoint_latest.pt'
            namespace['atomic_save'](path, original)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            expected = saved.pop('content_sha256')
            self.assertEqual(expected, engine.digest(saved))
            self.assertEqual(saved['comparison_policy'], policy)
            self.assertEqual(saved['state'], original['state'])
            self.assertEqual(engine.digest(saved['model']), engine.digest(original['model']))
            self.assertEqual(engine.digest(saved['optimizer']), engine.digest(original['optimizer']))
        self.assertEqual(engine.digest(original), before)


if __name__ == '__main__': unittest.main()
