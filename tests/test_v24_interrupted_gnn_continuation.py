"""Explicit CPU DEBUG metadata/state tests; no production data or GPU work."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from hiercp_v1x import v24_interrupted_gnn_continuation as runtime
from hiercp_v1x import v24_training_continuation as original
from hiercp_v1x.u_bridge_training import digest
from tools import run_v24_all_p as cli

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def save(path, value):
    value = copy.deepcopy(value)
    value['content_sha256'] = digest({key: row for key, row in value.items() if key != 'content_sha256'})
    torch.save(value, path)


def unit_main(argv=None):
    from hiercp_v1x.v24_factory import V24NativeInputs, build_runtime
    return build_runtime('EXPLICIT_CPU_DEBUG_NO_FORWARD', argv)


class InterruptedGNNDebug(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'outputs' / ('interrupted_gnn_CPU_DEBUG_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.source = self.root / 'old'; self.source.mkdir()
        self.job = self.root / 'job'; self.job.mkdir()
        self.output = self.root / 'new'
        self.inventory = self.root / 'DEBUG_inventory.json'; write_json(self.inventory, {'DEBUG': True})
        self.config = ROOT / 'config/v24_gpu5_GT_blind.json'
        self.args = SimpleNamespace(mode='train', gpu=5, resume=None, config=self.config,
            native_experiment=self.root / 'DEBUG_native', inventory=self.inventory,
            input_cache=self.root / 'DEBUG_cache', output=self.output, cpu_affinity=[0, 1, 2, 3], stunet_checkpoint=None)
        self.request = cli.request(self.args, runtime.read(self.config))
        self.train = ['DEBUG_train_' + str(i) for i in range(65)]
        self.val = ['DEBUG_val_' + str(i) for i in range(21)]
        self.contract = {'DEBUG': 'tiny_cpu_state_fixture_not_a_real_model'}
        self.calibration = dict(format='DEBUG_metadata_only_NOT_actual_calibration',
            request_sha256=self.request['request_sha256'], physical_GPU=5, initial_state_sha256='DEBUG_initial',
            model_contract=self.contract, selected_physical_patient_batch=4, selected_physical_candidate_batch=32,
            world_size=1, debug=False, original_model_and_RNG_preserved=True, measured_full_P_U128_backward=True)
        config = copy.deepcopy(self.request['config']); config['v24_runtime']['batch_calibration'] = self.calibration
        self.binding = dict(identity=self.request, config=config, train_cases=self.train, val_cases=self.val,
            physical_patient_batch=4, physical_candidate_batch=32, epochs=40, workers=4, debug=False,
            initial_state_sha256='DEBUG_initial', model_contract=self.contract)
        self.owner = dict(identity_sha256=digest(self.binding), binding=self.binding)
        history = [dict(epoch=epoch, DEBUG=True) for epoch in range(1, 25)]
        winner = dict(epoch=12, updates=204, selection_key=[.4, .3, -.5], selected_by='fixed_full128_validation_only')
        state = dict(epoch=25, phase='training', status='RUNNING', history=history, updates=408, attempts=408,
            overflows=0, best=winner, connected=[], train_order=None, train_position=0, train_rows=[],
            evaluation_position=0, evaluation_rows=[], epoch_start_updates=408,
            curriculum=dict(policy=config['v24_runtime']['curriculum'], active_u=103, stage_epochs=1))
        self.latest = dict(format='v24_GT_blind_all_P_patient_balanced_training_v1',
            identity_sha256=self.owner['identity_sha256'], model={'DEBUG.weight': torch.arange(3.)},
            optimizer={'state': {i: {'step': torch.tensor(408.), 'exp_avg': torch.ones(1), 'exp_avg_sq': torch.ones(1)} for i in range(981)}},
            scheduler={'last_epoch': 24}, scaler={'scale': 8.}, rank_rng=[{'DEBUG_CPU': torch.arange(3)}],
            shuffle_generator=torch.arange(4, dtype=torch.uint8), state=state)
        self.best = copy.deepcopy(self.latest)
        full = dict(epoch=12, phase='full_validation', active_u=128,
            metrics=dict(per_P_patient_mrr=.4, per_P_patient_top1=.3, patient_balanced_pair_loss=.5))
        self.best['state'].update(epoch=12, history=[dict(epoch=i, DEBUG=True) for i in range(1, 12)],
            phase='epoch_completion', updates=204, attempts=204, pending_epoch_completion={'full_validation': full})
        self.best['scheduler']['last_epoch'] = 11
        for row in self.best['optimizer']['state'].values():
            row['step'] = torch.tensor(204.)
        write_json(self.source / 'request.json', self.request)
        write_json(self.source / 'calibration.json', self.calibration)
        write_json(self.source / 'training/training_identity.json', self.owner)
        write_json(self.source / 'training/execution_contract.json', {'DEBUG': 'metadata fixture'})
        save(self.source / 'training/checkpoint_latest.pt', self.latest)
        save(self.source / 'training/checkpoint_best.pt', self.best)
        (self.source / 'training/curve.jsonl').write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in history))
        updates = [dict(status='OPTIMIZER_UPDATED', epoch=24, update=408),
                   dict(status='OPTIMIZER_UPDATED', epoch=25, update=409)]
        (self.source / 'training/update_timing.jsonl').write_bytes(b''.join(json.dumps(row).encode() + b'\n' for row in updates) + b'{"partial":')
        (self.source / 'training/failures.jsonl').write_bytes(b'{"DEBUG":"old_failure"}\n')
        (self.source / 'training/STOP_AFTER_BATCH').write_bytes(b'DEBUG preserve original stop')
        (self.source / 'training' / ('checkpoint_latest.pt.' + 'a' * 32 + '.tmp')).write_bytes(b'DEBUG partial checkpoint preserved')
        train_command = ['DEBUG_python', '-B', '-u', 'DEBUG_old_train.py', '--mode', 'train']
        pipeline = dict(GPU=5, production_output=str(self.source), code=str(ROOT),
                        stages=[dict(name='train', command=train_command)])
        self.worker = dict(pid=98765001, create_time=123., uid=41833,
            command=['DEBUG_python', '-B', '-u', 'DEBUG_old_worker.py', str(self.job / 'request.json')], absent=True)
        self.child = dict(pid=98765002, create_time=124., uid=41833, command=train_command, absent=True)
        status = dict(status='RUNNING', stage='train', request=pipeline,
            worker_pid=self.worker['pid'], worker_create_time=123., child_pid=self.child['pid'], child_create_time=124.)
        write_json(self.job / 'request.json', pipeline); write_json(self.job / 'status.json', status)
        self.context = dict(uid=41833, host='ece-a6gpu6', pid_exists=lambda pid: False)
        self.witness = self.make_witness()
        self.witness_path = self.root / 'DEBUG_interruption.json'; write_json(self.witness_path, self.witness)

    def make_witness(self):
        now = runtime.time.time_ns(); checkpoint = runtime.file_proof(self.source / 'training/checkpoint_latest.pt')
        return dict(format=runtime.WITNESS_FORMAT, GPU_arm=5, uid=41833, server_host='ece-a6gpu6',
            source_output=str(self.source), source_code=str(ROOT), source_job=str(self.job),
            status_file=runtime.file_proof(self.job / 'status.json'), pipeline_request_file=runtime.file_proof(self.job / 'request.json'),
            worker=self.worker, child=self.child, checkpoint=checkpoint, training_inventory=runtime.training_inventory(self.source),
            observations=[dict(time_ns=now - 2_000_000_000, worker_absent=True, child_absent=True, checkpoint=checkpoint),
                          dict(time_ns=now - 1_000_000_000, worker_absent=True, child_absent=True, checkpoint=checkpoint)], detected_ns=now)

    def prepare(self):
        with patch.object(runtime, '_context', return_value=self.context), patch.object(runtime, '_memory_selection', return_value=None):
            return runtime.prepare(self.args, source_output=self.source, source_code=ROOT,
                interruption_proof=self.witness_path, cli=cli, continuation=original)

    def admit(self):
        with patch.object(runtime, '_context', return_value=self.context), patch.object(runtime, '_memory_selection', return_value=None):
            return runtime.admit(self.args, source_output=self.source, source_code=ROOT,
                interruption_proof=self.witness_path, cli=cli, continuation=original)

    def test_arbitrary_epoch25_and_best_six_states_preserved_append_only(self):
        source_before = runtime.training_inventory(self.source)
        proof = self.prepare(); admitted = self.admit()
        self.assertEqual(proof, admitted)
        self.assertEqual(proof['source']['latest']['epoch'], 25)
        self.assertEqual(proof['source']['latest']['updates'], 408)
        self.assertEqual(set(proof['source']['latest']['numerical_state_sha256']), set(runtime.NUMERICAL))
        self.assertEqual((self.output / 'training/checkpoint_latest.pt').read_bytes(), (self.source / 'training/checkpoint_latest.pt').read_bytes())
        self.assertEqual((self.output / 'training/checkpoint_best.pt').read_bytes(), (self.source / 'training/checkpoint_best.pt').read_bytes())
        self.assertFalse((self.output / 'training/STOP_AFTER_BATCH').exists())
        self.assertEqual(source_before, runtime.training_inventory(self.source))
        self.assertEqual(len(proof['archived_only']), 2)
        rows = [json.loads(row) for row in (self.output / 'training/update_timing.jsonl').read_bytes().splitlines()]
        self.assertEqual([row['update'] for row in rows], [408])
        self.assertEqual((self.output / 'source_lineage/training/update_timing.jsonl').read_bytes(),
                         (self.source / 'training/update_timing.jsonl').read_bytes())
        self.assertEqual((self.output / 'training/failures.jsonl').read_bytes(), b'')

    def test_actual_partial_validation_case_ids_no_nonexistent_patients_key(self):
        state = dict(epoch=31, phase='train_probe', updates=527, train_position=65,
            evaluation_position=8, evaluation_rows=[{'case_id':case} for case in self.train[:8]], overflows=0, history=[])
        rows = [dict(epoch=31, phase='train_probe', case_ids=self.train[start:start + 4]) for start in (0, 4, 8)]
        raw = b''.join(json.dumps(row).encode() + b'\n' for row in rows)
        active, proof = runtime.derive_checkpoint_log(raw, 'validation_timing.jsonl', state)
        self.assertEqual([json.loads(row)['case_ids'] for row in active.splitlines()], [self.train[:4], self.train[4:8]])
        self.assertEqual(proof['rows_archived_only'], 1)
        rows[0]['case_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'case order'):
            runtime.derive_checkpoint_log(b''.join(json.dumps(row).encode() + b'\n' for row in rows), 'validation_timing.jsonl', state)

    def test_gpu4_full537_optimizer_arbitrary_epoch31_partial_probe(self):
        checkpoint = self.root / 'DEBUG_GPU4_state.pt'
        owner = copy.deepcopy(self.owner); owner['binding']['physical_candidate_batch'] = 64
        owner['identity_sha256'] = digest(owner['binding'])
        saved = copy.deepcopy(self.latest); saved['identity_sha256'] = owner['identity_sha256']
        saved['state'].update(epoch=31, history=[dict(epoch=i, DEBUG=True) for i in range(1, 31)],
            phase='train_probe', updates=527, attempts=527, train_order=self.train,
            train_rows=[{'case_id':case} for case in self.train], train_position=65,
            evaluation_position=8, evaluation_rows=[{'case_id':case} for case in self.train[:8]])
        saved['state']['curriculum']['active_u'] = 128
        saved['optimizer']['state'] = {i: {'step':torch.tensor(527.), 'exp_avg':torch.ones(1), 'exp_avg_sq':torch.ones(1)} for i in range(537)}
        saved['scheduler']['last_epoch'] = 30
        save(checkpoint, saved)
        _, proof = original._checkpoint(checkpoint, owner)
        self.assertEqual((proof['epoch'], proof['updates'], proof['optimizer_parameter_tensors'], proof['evaluation_position']), (31, 527, 537, 8))

    def test_amp_retry_saved_at_current_cursor_retained_only_saved_overflow_count(self):
        state = dict(epoch=25, phase='training', updates=408, train_position=0, evaluation_position=0, overflows=1)
        row = dict(status='AMP_OVERFLOW_RETRY_SAME_INPUT', epoch=25, cursor=0)
        active, proof = runtime.derive_checkpoint_log((json.dumps(row) + '\n').encode() * 2, 'update_timing.jsonl', state)
        self.assertEqual(len(active.splitlines()), 1); self.assertEqual(proof['rows_archived_only'], 1)

    def test_report_ahead_of_checkpoint_archived_but_durable_report_matches_state(self):
        report = dict(epoch=25, phase='train_probe', DEBUG='actual_saved_report')
        state = dict(epoch=25, phase='stage_validation', history=[], train_probe=report)
        self.assertTrue(runtime._report_durable('train_probe_epoch_025.json', json.dumps(report).encode(), state))
        self.assertFalse(runtime._report_durable('stage_validation_epoch_025.json', b'{partial', state))
        with self.assertRaisesRegex(ValueError, 'differs'):
            runtime._report_durable('train_probe_epoch_025.json', b'{"epoch":25,"phase":"train_probe"}', state)

    def test_changed_old_or_new_checkpoint_or_lineage_is_refused(self):
        self.prepare()
        path = self.output / 'source_lineage/training/failures.jsonl'; path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'lineage'):
            self.admit()

    def test_pid_reuse_full_argv_or_inventory_change_refused_without_signal(self):
        with patch.object(runtime, '_context', return_value={**self.context, 'pid_exists': lambda pid: True}), self.assertRaisesRegex(ValueError, 'still exists'):
            runtime.validate_interruption(self.witness_path, self.source, ROOT,
                dict(path=self.witness['checkpoint']['path'], source_stat=self.witness['checkpoint']['stat'], raw_sha256=self.witness['checkpoint']['sha256'], status='RUNNING'), 5)
        bad = copy.deepcopy(self.witness); bad['child']['command'].append('--wrong')
        path = self.root / 'DEBUG_bad_witness.json'; write_json(path, bad)
        with patch.object(runtime, '_context', return_value=self.context), self.assertRaisesRegex(ValueError, 'full argv'):
            runtime.validate_interruption(path, self.source, ROOT,
                dict(path=bad['checkpoint']['path'], source_stat=bad['checkpoint']['stat'], raw_sha256=bad['checkpoint']['sha256'], status='RUNNING'), 5)
        (self.source / 'training/DEBUG_unexpected.json').write_bytes(b'{}')
        with patch.object(runtime, '_context', return_value=self.context), self.assertRaisesRegex(ValueError, 'inventory'):
            runtime.validate_interruption(self.witness_path, self.source, ROOT,
                dict(path=self.witness['checkpoint']['path'], source_stat=self.witness['checkpoint']['stat'], raw_sha256=self.witness['checkpoint']['sha256'], status='RUNNING'), 5)

    def test_single_observation_and_short_interval_refused(self):
        bad = copy.deepcopy(self.witness); bad['observations'][1]['time_ns'] = bad['observations'][0]['time_ns'] + 10
        path = self.root / 'DEBUG_bad_interval.json'; write_json(path, bad)
        with patch.object(runtime, '_context', return_value=self.context), self.assertRaisesRegex(ValueError, 'Two stable'):
            runtime.validate_interruption(path, self.source, ROOT,
                dict(path=bad['checkpoint']['path'], source_stat=bad['checkpoint']['stat'], raw_sha256=bad['checkpoint']['sha256'], status='RUNNING'), 5)

    def test_existing_or_overlapping_destination_never_overwritten(self):
        self.output.mkdir(); marker = self.output / 'old'; marker.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_bytes(), b'preserve')
        self.args.output = self.source / 'nested'
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_original_cli_code_and_explicit_resume_argv_preserved(self):
        builder = Mock(return_value='CPU_DEBUG_ONLY'); clone = runtime.clone_main(unit_main, lambda old: builder, runtime.publish)
        self.assertIs(clone.__code__, unit_main.__code__)
        self.assertEqual(clone(['DEBUG_args']), 'CPU_DEBUG_ONLY')
        builder.assert_called_once_with('EXPLICIT_CPU_DEBUG_NO_FORWARD', ['DEBUG_args'])
        argv = original._argv(self.args, self.output / 'training/checkpoint_latest.pt')
        restored = cli.parse(argv)
        self.assertEqual(restored.resume, self.output / 'training/checkpoint_latest.pt')
        self.assertEqual(restored.gpu, 5)

    def test_no_gpu_execution_without_explicit_allocation_guard(self):
        with self.assertRaisesRegex(ValueError, 'allocation guard'):
            runtime.train(self.args, source_output=self.source, source_code=ROOT,
                interruption_proof=self.witness_path, allocation_guard=None, allocation_binding={})
        guard = Mock(side_effect=ValueError('DEBUG allocation rejected'))
        with self.assertRaisesRegex(ValueError, 'allocation rejected'):
            runtime.train(self.args, source_output=self.source, source_code=ROOT,
                interruption_proof=self.witness_path, allocation_guard=guard, allocation_binding={})
        self.assertFalse(self.output.exists()); guard.assert_called_once()

    def test_allocation_module_sha_and_live_guard_each_call(self):
        module = self.root / 'DEBUG_guard.py'
        module.write_text('def make_guard(proof_path,gpu,CPU_affinity):\n    def check():\n        assert gpu == 5 and len(CPU_affinity) == 4\n    return check\n', encoding='utf8')
        proof = self.root / 'DEBUG_allocation.json'; write_json(proof, {'DEBUG': 'not an actual scheduler allocation'})
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            runtime.load_allocation_guard(module, '0' * 64, proof, runtime.sha(proof), 5, [0, 1, 2, 3])
        check, binding = runtime.load_allocation_guard(module, runtime.sha(module), proof, runtime.sha(proof), 5, [0, 1, 2, 3])
        check(); proof.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            check()

    def test_source_manifest_gpu4_extended_and_legacy_gpu5_requires_reviewed_sha(self):
        self.assertEqual(runtime.source_names(cli, 4), cli.FILES + cli.GPU4_EXECUTION_FILES)
        old = SimpleNamespace(FILES=('DEBUG_science.py',), __file__=self.config)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            runtime.source_names(old, 4)
        with patch.object(runtime, 'sha', return_value=runtime.LEGACY_GPU5_CLI_SHA256):
            self.assertEqual(runtime.source_names(old, 5), old.FILES)

    def test_final_epoch_transition_resumes_without_reducing_or_repeating_updates(self):
        proof = dict(epoch=41, phase='training', optimizer_parameter_tensors=981)
        runtime._validate_unfinished_latest(proof, 5)
        for changed in (dict(proof, phase='complete'), dict(proof, epoch=42), dict(proof, phase='full_validation')):
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                runtime._validate_unfinished_latest(changed, 5)

    def test_actual_integer_recall_keys_use_unchanged_original_JSON_publisher(self):
        row = dict(epoch=1, train=dict(cases=[dict(recalled_observed_P={1:2,5:4,10:7})]),
                   train_probe=dict(DEBUG=True), full_validation=dict(DEBUG=True))
        state = dict(epoch=2, updates=17, train_position=0, evaluation_position=0,
                     overflows=0, history=[row])
        original_state = copy.deepcopy(state)
        raw = (json.dumps(row) + '\n').encode()
        active, proof = runtime.derive_checkpoint_log(raw, 'curve.jsonl', state)
        self.assertEqual(active, raw)
        self.assertEqual(state, original_state)
        self.assertTrue(proof['original_publication_code_preserved'])
        self.assertTrue(proof['full_saved_history_compared_via_original_JSON_publication'])
        report = dict(epoch=1, phase='full_validation', cases=[dict(recalled_observed_P={1:2,5:4,10:7})])
        state['history'][0]['full_validation'] = report
        self.assertTrue(runtime._report_durable('full_validation_epoch_001.json', json.dumps(report).encode(), state))
        wrong = copy.deepcopy(row); wrong['train']['cases'][0]['recalled_observed_P'][1] = 999
        with self.assertRaisesRegex(ValueError, 'Durable curve differs'):
            runtime.derive_checkpoint_log((json.dumps(wrong) + '\n').encode(), 'curve.jsonl', original_state)


if __name__ == '__main__':
    unittest.main()
