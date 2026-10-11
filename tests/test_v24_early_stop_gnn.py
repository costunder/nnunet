"""CPU UNIT only: tiny six-state tensors, no GNN/CUDA training or signals."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x import v24_early_stop_gnn as early
from hiercp_v1x import v24_nnunet_cp as original
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.u_bridge_training import digest
from tests import test_v24_current_gnn_nnunet_cp as fixtures

write_json = fixtures.write_json


class EarlyStopAdmission(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CurrentOwnArmAdmission(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        self.output, self.code = fixture.output, fixture.code
        self.authorization = fixture.root / 'CPU_UNIT_stop_authorization.json'
        self.target = 27
        history = fixture.latest['state']['history'][:self.target]
        for row in history:
            row['full_validation']['cases'] = [dict(case_id=case) for case in fixture.owner['binding']['val_cases']]
            row['stage_validation'] = dict(active_u=103)
            row['train_probe'] = dict(active_u=103)
        state = fixture.latest['state']
        state.update(history=history, epoch=28, phase='training', status='PAUSED', updates=459, attempts=459)
        state['curriculum'].update(last_completed_epoch=27, active_u=103)
        fixture.latest['scheduler']['last_epoch'] = 27
        fixture.latest['optimizer']['state'] = {index: dict(step=torch.tensor(459.),
            exp_avg=torch.zeros(1), exp_avg_sq=torch.ones(1)) for index in range(981)}
        fixture.best['state']['history'] = copy.deepcopy(history[:16])
        fixture.best['state']['pending_epoch_completion']['full_validation'] = copy.deepcopy(history[16]['full_validation'])
        fixture.best['optimizer']['state'] = {index: dict(step=torch.tensor(289.),
            exp_avg=torch.zeros(1), exp_avg_sq=torch.ones(1)) for index in range(981)}
        fixture.invocation.update(status='PAUSED', full_training=False,
            completed_epochs=27, optimizer_updates=459, active_u=103)
        fixture.persist()
        fixtures.interrupted_metadata_fixture(fixture.root, self.output, self.code, fixture.request, fixture.owner, 5)
        self.actor = dict(pid=99999999, create_time=1., command=['CPU_UNIT_never_executed'])
        request_path = self.output.parent / 'direct_training_request.json'
        write_json(request_path, dict(format='v24_existing_direct_GPU_exact_continuation_request_v1',
            GPU_arm=5, output=str(self.output), source_code=str(self.code)))
        write_json(self.output.parent / 'direct_training_registration.json',
            dict(format='v24_existing_direct_GPU_exact_continuation_registration_v1',
                GPU_arm=5, output=str(self.output), request_sha256=original.sha(request_path), **self.actor))
        self.auth = dict(format=early.AUTHORIZATION_FORMAT, GPU_arm=5, source_output=str(self.output),
            source_code=str(self.code), target_epoch=27, original_planned_epochs=40, user_authorized=True,
            stop_policy=early.STOP_POLICY, created_at=2., reason='CPU UNIT explicit approval fixture',
            original_actor=self.actor)
        write_json(self.authorization, self.auth)
        (self.output / 'training/STOP_AFTER_BATCH').write_bytes(b'CPU UNIT stop marker; never controls a process')

    def inspect(self):
        with patch.object(early, '_actor_absent', return_value=True):
            return early.inspect_early_stop(self.output, self.code, self.fixture.inventory,
                input_cache=self.fixture.cache, target_epoch=self.target,
                authorization_path=self.authorization)

    def persist(self):
        self.fixture.persist()

    def test_actual_paused_history_own_best_and_six_states_are_truthful(self):
        pin = self.inspect()
        self.assertEqual((pin['completed_epochs'], pin['optimizer_updates']), (27, 459))
        self.assertEqual(pin['format'], early.EARLY_PIN_FORMAT)
        self.assertFalse(pin['full_training'])
        self.assertEqual(pin['selected']['best_record']['epoch'], 17)
        self.assertEqual(pin['selected']['best_model_sha256'], digest(self.fixture.best['model']))
        self.assertEqual(set(pin['final_checkpoint_proof']['numerical_state_sha256']), set(early.NUMERICAL))
        self.assertEqual(pin['training_U_exposure']['actual_full128_training_epochs'], 0)
        self.assertEqual(pin['training_U_exposure']['full128_validation_epochs'], 27)
        self.assertEqual(early.validate_early_stop_pin(pin), pin)
        with self.assertRaises(ValueError):
            original.validate_current_pin(pin)

    def test_wrong_boundary_and_unauthorized_request_are_rejected(self):
        self.fixture.latest['state']['phase'] = 'full_validation'; self.persist()
        with self.assertRaisesRegex(ValueError, 'requested whole epoch'):
            self.inspect()
        self.fixture.latest['state']['phase'] = 'training'; self.persist()
        self.auth['user_authorized'] = False; write_json(self.authorization, self.auth)
        with self.assertRaisesRegex(ValueError, 'user-authorized'):
            self.inspect()

    def test_extra_next_epoch_update_cannot_be_relabelled_boundary(self):
        self.fixture.latest['state'].update(updates=460, attempts=460)
        for state in self.fixture.latest['optimizer']['state'].values(): state['step'] = torch.tensor(460.)
        self.persist()
        with self.assertRaisesRegex(ValueError, 'requested whole epoch'):
            self.inspect()

    def test_missing_full128_epoch_or_recipient_gt_exposure_rejected(self):
        self.fixture.latest['state']['history'][22]['full_validation']['active_u'] = 103
        self.persist()
        with self.assertRaisesRegex(ValueError, 'complete fixed full128'):
            self.inspect()

    def test_full_validation_population_must_be_complete(self):
        self.fixture.latest['state']['history'][22]['full_validation']['cases'].pop()
        self.persist()
        with self.assertRaisesRegex(ValueError, 'complete fixed full128'):
            self.inspect()

    def test_best_must_maximize_all_completed_full128_reports(self):
        self.fixture.latest['state']['history'][22]['full_validation']['metrics']['per_P_patient_mrr'] = .9
        self.persist()
        with self.assertRaisesRegex(ValueError, 'own full128 BEST'):
            self.inspect()

    def test_nonfinite_tensor_and_broken_optimizer_history_rejected(self):
        self.fixture.latest['model']['UNIT.weight'][0] = float('nan'); self.persist()
        with self.assertRaises(FloatingPointError):
            self.inspect()
        self.fixture.latest['model']['UNIT.weight'][0] = 2.
        self.fixture.latest['optimizer']['state'][8]['step'] = torch.tensor(458.)
        self.persist()
        with self.assertRaisesRegex(ValueError, 'AdamW update'):
            self.inspect()

    def test_paused_invocation_and_partial_jsonl_required(self):
        self.fixture.invocation['status'] = 'COMPLETE'; self.persist()
        with self.assertRaisesRegex(ValueError, 'PAUSED CUDA invocation'):
            self.inspect()
        self.fixture.invocation['status'] = 'PAUSED'; self.persist()
        path = self.output / 'training/invocations.jsonl'
        path.write_bytes(path.read_bytes().rstrip(b'\n'))
        with self.assertRaisesRegex(ValueError, 'complete PAUSED invocation'):
            self.inspect()

    def test_authorization_actor_and_scientific_source_cannot_change(self):
        self.auth['original_actor'] = dict(self.actor, pid=88888888); write_json(self.authorization, self.auth)
        with self.assertRaisesRegex(ValueError, 'registration'):
            self.inspect()
        self.auth['original_actor'] = self.actor; write_json(self.authorization, self.auth)
        target = self.code / 'hiercp_v1x/v24_training.py'
        target.write_bytes(target.read_bytes() + b'\n# CPU UNIT mutation\n')
        with self.assertRaisesRegex(ValueError, 'scientific source'):
            self.inspect()

    def test_original_interrupted_six_state_lineage_is_not_bypassed(self):
        receipt = self.output / original.INTERRUPTED_RECEIPT
        document = json.loads(receipt.read_text())
        document['six_numerical_states_unchanged'] = False
        write_json(receipt, document)
        with self.assertRaisesRegex(ValueError, 'saved-state lineage'):
            self.inspect()

    def test_pin_publishing_is_fresh_and_readmission_detects_changes(self):
        path = self.fixture.root / 'CPU_UNIT_early_pin.json'
        with patch.object(early, '_actor_absent', return_value=True):
            early.pin_early_stopped(source_output=self.output, source_code=self.code,
                inventory_path=self.fixture.inventory, input_cache=self.fixture.cache,
                target_epoch=27, authorization_path=self.authorization, output=path)
            admitted, request = early.admit_early_stop_pin(path, self.fixture.inventory)
            self.assertEqual(request, self.fixture.request)
            self.assertFalse(admitted['full_training'])
            with self.assertRaises(FileExistsError):
                early.pin_early_stopped(source_output=self.output, source_code=self.code,
                    inventory_path=self.fixture.inventory, input_cache=self.fixture.cache,
                    target_epoch=27, authorization_path=self.authorization, output=path)
            self.auth['reason'] += ' changed'; write_json(self.authorization, self.auth)
            with self.assertRaisesRegex(ValueError, 'proof changed'):
                early.admit_early_stop_pin(path, self.fixture.inventory)

    def test_current_completed_format_and_invalid_epoch_cannot_pass(self):
        pin = self.inspect()
        for changes in (dict(format=original.CURRENT_PIN_FORMAT), dict(full_training=True),
                        dict(completed_epochs=40, optimizer_updates=680), dict(physical_GPU=4)):
            altered = copy.deepcopy(pin); altered.update(changes)
            altered['pin_sha256'] = canonical_hash({k: v for k, v in altered.items() if k != 'pin_sha256'})
            with self.assertRaisesRegex(ValueError, 'truthful GPU5'):
                early.validate_early_stop_pin(altered)

    def test_post_scoring_cpu_reinspection_is_allowed_but_initial_pin_is_pre_cuda(self):
        before = self.inspect()
        with patch.object(torch.cuda, 'is_initialized', return_value=True):
            self.assertEqual(self.inspect(), before)
            with self.assertRaisesRegex(RuntimeError, 'must precede CUDA'):
                early.pin_early_stopped(source_output=self.output, source_code=self.code,
                    inventory_path=self.fixture.inventory, input_cache=self.fixture.cache,
                    target_epoch=27, authorization_path=self.authorization,
                    output=self.fixture.root / 'CPU_UNIT_never_published.json')

    def test_signed_metadata_cannot_invent_full128_training_or_complete_status(self):
        pin = self.inspect()
        for field in ('exposure', 'status'):
            altered = copy.deepcopy(pin)
            if field == 'exposure':
                altered['training_U_exposure']['actual_full128_training_epochs'] = 27
            else:
                altered['final_checkpoint_proof']['status'] = 'COMPLETE'
            altered['pin_sha256'] = canonical_hash({k: v for k, v in altered.items() if k != 'pin_sha256'})
            with self.assertRaisesRegex(ValueError, 'Truthful paused boundary'):
                early.validate_early_stop_pin(altered)


class ActorIdentity(unittest.TestCase):
    def test_live_registered_actor_and_reused_pid_are_rejected(self):
        import psutil
        actor = dict(pid=9876543, create_time=1., command=['CPU_UNIT_actor'])
        process = SimpleNamespace(create_time=lambda: 1., status=lambda: psutil.STATUS_SLEEPING,
            cmdline=lambda: actor['command'], uids=lambda: SimpleNamespace(real=0))
        with patch.object(psutil, 'Process', return_value=process), patch.object(early.os, 'getuid', return_value=0, create=True):
            with self.assertRaisesRegex(ValueError, 'still live'):
                early._actor_absent(actor)
            process.create_time = lambda: 2.
            with self.assertRaisesRegex(ValueError, 'PID reuse'):
                early._actor_absent(actor)


if __name__ == '__main__':
    unittest.main()
