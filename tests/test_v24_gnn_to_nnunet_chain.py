"""CPU-only DEBUG metadata fixtures; these are not training results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import watch_v24_gnn_to_nnunet as chain


class CompletedArmFixture(unittest.TestCase):
    def setUp(self):
        temporary_parent = Path(__file__).absolute().parents[1] / 'outputs'
        temporary_parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='DEBUG_gnn_chain_', dir=temporary_parent)
        self.root = Path(self.temp.name)
        self.code = self.root / 'candidate'; self.code.mkdir()
        self.source_code = self.root / 'source_code'; self.source_code.mkdir()
        for code in (self.code, self.source_code):
            (code / 'tools').mkdir()
            for name in ('watch_v24_gnn_to_nnunet.py', 'run_v24_nnunet_cp.py', 'run_v24_gpu4_all_u.py'):
                (code / 'tools' / name).write_text('# CPU-only DEBUG source witness fixture\n')
        self.source = self.root / 'gnn'; (self.source / 'training').mkdir(parents=True)
        self.job = self.root / 'owned_source_job'; self.job.mkdir()
        self.target = self.root / 'new_chain'
        self.python = self.root / 'python'; self.python.write_text('DEBUG interpreter witness; never executed')
        self.inventory = self.root / 'inventory.json'; self.inventory.write_text('{}')
        self.preprocessed = self.root / 'baseline'; self.preprocessed.mkdir()
        self.cache = self.root / 'cache'; self.cache.mkdir()
        self.claim_root = self.root / 'gnn_native_chain_claims_20261010'; self.claim_root.mkdir()
        self.stunet = self.root / 'stunet.model'; self.stunet.write_text('DEBUG input witness')
        self.request = self.make_request(5)
        self.write_completion(5)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, path, value):
        Path(path).write_text(json.dumps(value), encoding='utf8')

    def make_request(self, gpu):
        python = str(self.python)
        entry = str(self.code / 'tools/run_v24_nnunet_cp.py')
        target = str(self.target)
        base = dict(format=chain.FORMAT, GPU_arm=gpu, chain_root=target,
                    code=str(self.code), commit='a' * 40, code_files={
                        'tools/' + p.name: chain.sha(p) for p in (self.code / 'tools').iterdir()},
                    source_output=str(self.source), source_code=str(self.source_code), source_commit='b' * 40,
                    source_files={'tools/run_v24_nnunet_cp.py': chain.sha(self.source_code / 'tools/run_v24_nnunet_cp.py')},
                    CPU_affinity=[12, 13, 14, 15] if gpu == 4 else chain.AFFINITIES[gpu], scoring_RAM_GiB=64, native_RAM_GiB=48,
                    min_free_disk_GiB=160, minimum_runtime_free_disk_GiB=10,
                    GPU_UUID='GPU-' + str(gpu), python=python, inventory=str(self.inventory),
                    baseline_preprocessed=str(self.preprocessed), input_cache=str(self.cache),
                    stunet_checkpoint=str(self.stunet) if gpu in (4, 6) else None,
                    scheduler=None if gpu in (4, 5) else dict(job_id='129443.ECE-util1', state='R',
                        expected_owner='aicompetition06', expected_node='ece-a6gpu6', minimum_remaining_seconds=28800,
                        native_minimum_remaining_seconds=25200, expires_at_unix=1791702055))
        if gpu == 4:
            base['stunet_checkpoint_sha256'] = chain.sha(self.stunet)
            config_name = 'config/v24_gpu4_STUNetS_all_U_GT_blind.json'
            config_path = self.source_code / config_name; config_path.parent.mkdir(exist_ok=True)
            actual_config = chain.read(Path(chain.__file__).absolute().parents[1] / config_name)
            self.write_json(config_path, actual_config)
            base['source_files'].update({config_name: chain.sha(config_path),
                'tools/run_v24_gpu4_all_u.py': chain.sha(self.source_code / 'tools/run_v24_gpu4_all_u.py')})
        base['source_claim_path'] = str(self.claim_root / (chain.hashlib.sha256(str(self.source).encode()).hexdigest() + '.json'))
        pairs = [dict(source_output=str(self.source), source_code=str(self.source_code), inventory=str(self.inventory),
                      input_cache=str(self.cache), output=str(self.target / 'pin.json')),
                 dict(pin=str(self.target / 'pin.json'), inventory=str(self.inventory), baseline_preprocessed=str(self.preprocessed),
                      input_cache=str(self.cache), output=str(self.target / 'bank')),
                 dict(bank=str(self.target / 'bank/index.json'), output=str(self.target / 'native')),
                 dict(native=str(self.target / 'native/native.json')), dict(native=str(self.target / 'native/native.json'))]
        if gpu in (4, 6):
            for options in pairs[:2]:
                options['stunet_checkpoint'] = str(self.stunet)
        stages = []
        for action, options in zip(chain.ACTIONS, pairs):
            args = [python, '-B', '-u', entry, action, '--gpu', str(gpu)]
            for name, value in options.items():
                args += ['--' + name.replace('_', '-'), value]
            stages.append(dict(name=action.replace('-', '_'), action=action, command=args))
        base['stages'] = stages
        worker_command = [python, '-B', '-u', str(self.job / 'pipeline_worker.py'), str(self.job / 'request.json')]
        pipeline = dict(GPU=gpu, GPU_uuid=base['GPU_UUID'], code=base['source_code'], commit=base['source_commit'],
                        production_output=base['source_output'], CPU_affinity=base['CPU_affinity'], RAM_GiB=64,
                        stages=[dict(name='train', command=['DEBUG_owned_train_command'])])
        if gpu in (4, 5):
            pipeline.update(epochs=40, physical_patient_batch=4, candidate_chunk=32 if gpu == 5 else 64)
        else:
            pipeline.update(original_total_epochs=40, original_patient_batch=4, original_candidate_chunk=64)
        if gpu == 4:
            pipeline.update(source_config=str(config_path), source_config_sha256=chain.sha(config_path), gnn_config=actual_config)
            pipeline['stages'] = []
            for name, mode in (('prepare_inputs', 'prepare'), ('calibrate', 'calibrate'), ('train', 'train')):
                pipeline['stages'].append(dict(name=name, command=[python, '-B', '-u',
                    str(self.source_code / 'tools/run_v24_gpu4_all_u.py'), '--mode', mode, '--config', str(config_path),
                    '--native-experiment', str(self.preprocessed), '--inventory', str(self.inventory),
                    '--input-cache', str(self.cache), '--output', str(self.source), '--gpu', '4',
                    '--stunet-checkpoint', str(self.stunet), '--cpu-affinity', ','.join(map(str, base['CPU_affinity']))]))
        self.write_json(self.job / 'request.json', pipeline)
        status = dict(request=pipeline, worker_pid=123, worker_create_time=10., actual_CPU_affinity=base['CPU_affinity'],
                      child_pid=456, child_create_time=11., stage='train', stages_completed=['prepare', 'train'], status='COMPLETE')
        self.write_json(self.job / 'status.json', status)
        base['source_job'] = dict(status=str(self.job / 'status.json'), worker_pid=123, worker_create_time=10.,
                                 worker_command=worker_command, training_stage='train',
                                 pipeline_request_sha256=chain.sha(self.job / 'request.json'))
        return base

    def write_completion(self, gpu):
        count = chain.PARAMETER_TENSORS[gpu]
        self.receipt = dict(status='COMPLETE', full_training=True, debug=False, actual_CUDA=True,
                            completed_epochs=40, optimizer_updates=680, connected_parameter_tensors=count,
                            expected_parameter_tensors=count, recipient_GT_used_in_forward=False,
                            checkpoint=str(self.source / 'training/checkpoint_latest.pt'),
                            best=dict(epoch=7, updates=119, selected_by='fixed_full128_validation_only'))
        scientific = dict(format='v24_full_native_GT_free_execution_request_v1', physical_GPU=gpu,
                          config=dict(epochs=40, v24_runtime=dict(debug=False, hidden_subset=False, workers=4, torch_threads=1)))
        if gpu == 4:
            scientific['STU_checkpoint_sha256'] = chain.sha(self.stunet)
            scientific['config'] = chain.read(self.source_code / 'config/v24_gpu4_STUNetS_all_U_GT_blind.json')
        scientific['request_sha256'] = chain.hashlib.sha256(json.dumps(scientific, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        calibration = dict(request_sha256=scientific['request_sha256'], physical_GPU=gpu,
                           selected_physical_patient_batch=4, selected_physical_candidate_batch=32 if gpu == 5 else 64)
        binding_config = copy.deepcopy(scientific['config']); binding_config['v24_runtime']['batch_calibration'] = calibration
        owner = dict(identity_sha256='d' * 64, binding=dict(identity=scientific, epochs=40, debug=False, config=binding_config))
        self.write_json(self.source / 'request.json', scientific)
        self.write_json(self.source / 'calibration.json', calibration)
        self.write_json(self.source / 'training/training_identity.json', owner)
        self.write_json(self.source / 'training/invocations.jsonl', self.receipt)

    def write_claim(self, request_path, request=None):
        request = request or self.request
        self.write_json(request['source_claim_path'], dict(format='v24_owned_gnn_to_native_source_claim_v1',
            source_output=request['source_output'], GPU_arm=request['GPU_arm'], chain_root=request['chain_root'],
            request_sha256=chain.sha(request_path)))

    def completion(self, status=None, worker=None, child=None):
        return chain.source_completion(self.request, status or chain.read(self.job / 'status.json'), worker, child)

    def test_both_original_full_arm_requests_are_admitted(self):
        for gpu in (5, 6):
            with self.subTest(gpu=gpu):
                chain.validate_request(self.make_request(gpu))

    def test_gpu4_full128_stunet_has_explicit_own_cpu_and_native_commands(self):
        self.request = self.make_request(4); self.write_completion(4)
        chain.validate_request(self.request)
        with patch.object(chain, 'process_witness', return_value=None):
            self.assertIsNotNone(chain.inspect_source(self.request))
        for stage in self.request['stages']:
            self.assertEqual(stage['command'][stage['command'].index('--gpu') + 1], '4')
        for stage in self.request['stages'][:2]:
            self.assertEqual(stage['command'][stage['command'].index('--stunet-checkpoint') + 1], str(self.stunet))
        self.assertEqual(self.request['CPU_affinity'], [12, 13, 14, 15])
        self.assertEqual(self.completion()['invocation']['expected_parameter_tensors'], 537)

    def test_gpu4_cannot_borrow_gpu6_pbs_or_invent_cpu_default(self):
        request = self.make_request(4)
        for cores in (None, [12, 12, 14, 15], [12, 13, 14], [15, 14, 13, 12], [12, 13, 14, True]):
            with self.subTest(cores=cores), self.assertRaises(ValueError):
                chain.validate_request(dict(request, CPU_affinity=cores))
        reserved = self.make_request(6)['scheduler']
        with self.assertRaisesRegex(ValueError, 'distinct owned PBS'):
            chain.validate_request(dict(request, scheduler=reserved))
        reserved = dict(reserved, job_id='DEBUG_GPU4.ECE-util1')
        chain.validate_request(dict(request, scheduler=reserved))

    def test_gpu4_all_u_and_official_asset_metadata_are_required_before_handoff(self):
        self.request = self.make_request(4); self.write_completion(4)
        scientific = chain.read(self.source / 'request.json')
        owner = chain.read(self.source / 'training/training_identity.json')
        for field, value in (('all_U_from_epoch_one', False), ('physical_GPU', 6), ('other_P_as_negative', True)):
            changed = copy.deepcopy(scientific); changed['config']['v24_runtime'][field] = value
            changed['request_sha256'] = chain.hashlib.sha256(json.dumps(
                {k: v for k, v in changed.items() if k != 'request_sha256'}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            changed_owner = copy.deepcopy(owner); changed_owner['binding']['identity'] = changed
            calibration = chain.read(self.source / 'calibration.json'); calibration['request_sha256'] = changed['request_sha256']
            changed_owner['binding']['config'] = copy.deepcopy(changed['config'])
            changed_owner['binding']['config']['v24_runtime']['batch_calibration'] = calibration
            self.write_json(self.source / 'request.json', changed)
            self.write_json(self.source / 'calibration.json', calibration)
            self.write_json(self.source / 'training/training_identity.json', changed_owner)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'full128U from epoch one'):
                chain.validate_source_metadata(self.request)
        self.write_completion(4)
        self.stunet.write_text('mutated DEBUG asset')
        with self.assertRaisesRegex(ValueError, 'official STU checkpoint SHA'):
            chain.validate_request(self.request)
        with self.assertRaisesRegex(ValueError, 'original STU asset'):
            chain.validate_source_metadata(self.request)

    def test_gpu4_actual_source_cpu_assignment_must_match_its_new_explicit_request(self):
        self.request = self.make_request(4); self.write_completion(4)
        pipeline = chain.read(self.job / 'request.json'); pipeline['CPU_affinity'] = [16, 17, 18, 19]
        self.write_json(self.job / 'request.json', pipeline)
        status = chain.read(self.job / 'status.json'); status['request'] = pipeline
        self.write_json(self.job / 'status.json', status)
        self.request['source_job']['pipeline_request_sha256'] = chain.sha(self.job / 'request.json')
        with self.assertRaisesRegex(ValueError, 'own-arm contract'):
            chain.inspect_source(self.request)

    def gpu4_live_status(self, stage):
        status = chain.read(self.job / 'status.json')
        status.update(status='RUNNING', stage=stage, stages_completed=[])
        self.write_json(self.job / 'status.json', status)

    def inspect_live_gpu4(self, worker_status='running'):
        with patch.object(chain, 'process_witness', side_effect=[
                dict(pid=123, status=worker_status), dict(pid=456, status='running')]) as witness, \
                patch('psutil.Process') as process:
            process.return_value.ppid.return_value = 123
            value = chain.inspect_source(self.request)
        return value, witness

    def test_gpu4_fresh_prepare_and_calibration_wait_for_unpublished_real_metadata(self):
        self.request = self.make_request(4); self.write_completion(4)
        for name in ('request.json', 'calibration.json', 'training/training_identity.json'):
            (self.source / name).unlink()
        for stage in ('prepare_inputs', 'calibrate'):
            self.gpu4_live_status(stage)
            value, witness = self.inspect_live_gpu4()
            self.assertIsNone(value)
            pipeline = chain.read(self.job / 'request.json')
            declared = next(row for row in pipeline['stages'] if row['name'] == stage)
            self.assertEqual(witness.call_args_list[-1].args, (456, 11., declared['command']))
        self.assertFalse((self.target / 'pin.json').exists())

    def test_gpu4_train_identity_publish_may_wait_only_for_exact_live_source(self):
        self.request = self.make_request(4); self.write_completion(4)
        (self.source / 'training/training_identity.json').unlink()
        self.gpu4_live_status('train')
        self.assertIsNone(self.inspect_live_gpu4()[0])
        with self.assertRaisesRegex(RuntimeError, 'exact live RUNNING'):
            self.inspect_live_gpu4(worker_status='zombie')
        with patch.object(chain, 'process_witness', return_value=None), self.assertRaisesRegex(RuntimeError, 'disappeared'):
            chain.inspect_source(self.request)
        status = chain.read(self.job / 'status.json'); status.update(status='COMPLETE', stages_completed=['train'])
        self.write_json(self.job / 'status.json', status)
        with patch.object(chain, 'process_witness', return_value=None), self.assertRaisesRegex(FileNotFoundError, 'required real metadata'):
            chain.inspect_source(self.request)

    def test_gpu4_pending_metadata_never_hides_existing_malformed_json(self):
        self.request = self.make_request(4); self.write_completion(4)
        (self.source / 'training/training_identity.json').unlink()
        (self.source / 'calibration.json').write_text('{corrupt CPU UNIT metadata', encoding='utf8')
        self.gpu4_live_status('calibrate')
        with self.assertRaises(json.JSONDecodeError):
            self.inspect_live_gpu4()
        self.write_json(self.source / 'calibration.json', {})
        with self.assertRaisesRegex(ValueError, 'calibration lacks its actual full'):
            self.inspect_live_gpu4()

    def test_gpu4_fresh_pending_source_rejects_initial_u7_or_wrong_stage_gpu(self):
        self.request = self.make_request(4); self.write_completion(4)
        (self.source / 'training/training_identity.json').unlink()
        self.gpu4_live_status('prepare_inputs')
        pipeline = chain.read(self.job / 'request.json')
        command = pipeline['stages'][0]['command']; command[command.index('--gpu') + 1] = '6'
        self.write_json(self.job / 'request.json', pipeline)
        status = chain.read(self.job / 'status.json'); status['request'] = pipeline
        self.write_json(self.job / 'status.json', status)
        self.request['source_job']['pipeline_request_sha256'] = chain.sha(self.job / 'request.json')
        with self.assertRaisesRegex(ValueError, 'fresh stage changed'):
            chain.inspect_source(self.request)
        self.request = self.make_request(4); self.gpu4_live_status('prepare_inputs')
        pipeline = chain.read(self.job / 'request.json')
        pipeline['gnn_config']['v24_runtime']['curriculum']['initial_u'] = 7
        self.write_json(pipeline['source_config'], pipeline['gnn_config'])
        pipeline['source_config_sha256'] = chain.sha(pipeline['source_config'])
        self.request['source_files']['config/v24_gpu4_STUNetS_all_U_GT_blind.json'] = pipeline['source_config_sha256']
        self.write_json(self.job / 'request.json', pipeline)
        status = chain.read(self.job / 'status.json'); status['request'] = pipeline
        self.write_json(self.job / 'status.json', status)
        self.request['source_job']['pipeline_request_sha256'] = chain.sha(self.job / 'request.json')
        with self.assertRaisesRegex(ValueError, 'native128U from epoch one'):
            chain.inspect_source(self.request)

    def test_gpu4_native_child_routes_physical_device_and_retains_ram_and_full_calibration(self):
        self.request = self.make_request(4); self.target.mkdir(); (self.target / 'native').mkdir()
        request_path = self.root / 'registered_request.json'; self.write_json(request_path, self.request)
        stage = self.request['stages'][-1]
        status = dict(stages_completed=[], stage_proofs=[])
        child = SimpleNamespace(pid=789, poll=lambda: 0, wait=lambda: 0)
        with patch.object(chain.subprocess, 'Popen', return_value=child) as launch, \
                patch('psutil.Process') as process, patch.object(chain, 'stage_proof', return_value={'CPU_UNIT_ONLY': True}):
            process.return_value.create_time.return_value = 12.
            chain.run_stage(self.request, stage, status, request_path, chain.sha(request_path))
        self.assertEqual(launch.call_args.args[0], stage['command'])
        env = launch.call_args.kwargs['env']
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4')
        self.assertEqual(env['V24_ARM_RSS_GIB'], '48')
        self.assertNotIn('shell', launch.call_args.kwargs)
        calibration = dict(debug=True, production_updates=0, production_epochs=250,
                           production_physical_batch=2, production_cp_probability=.5)
        self.write_json(self.target / 'native/calibration.json', calibration)
        chain.stage_proof(self.request, 'calibrate-native')
        calibration['production_physical_batch'] = 1
        self.write_json(self.target / 'native/calibration.json', calibration)
        with self.assertRaisesRegex(ValueError, 'full production contract'):
            chain.stage_proof(self.request, 'calibrate-native')

    def test_waits_for_running_worker_without_loading_checkpoint(self):
        status = chain.read(self.job / 'status.json'); status['status'] = 'RUNNING'
        self.assertIsNone(self.completion(status, dict(status='running')))
        self.assertFalse((self.target / 'pin.json').exists())

    def test_complete_live_child_is_not_an_automatic_handoff(self):
        self.assertIsNone(self.completion(child=dict(status='running')))

    def test_exact_completed_zombies_are_finished_without_signals(self):
        value = self.completion(worker=dict(status='zombie'), child=dict(status='zombie'))
        self.assertTrue(value['worker_and_child_finished'])
        self.assertEqual(value['checkpoint_validation_delegated_to'], 'pin-current-gnn')

    def test_failed_paused_and_disappeared_running_source_do_not_launch(self):
        for outcome in ('FAILED', 'PAUSED', 'PAUSED_BETWEEN_STAGES', 'RUNNING'):
            with self.subTest(outcome=outcome):
                status = chain.read(self.job / 'status.json'); status['status'] = outcome
                with self.assertRaises(RuntimeError):
                    self.completion(status)

    def test_pause_marker_prevents_downstream_even_with_complete_receipt(self):
        (self.source / 'training/STOP_AFTER_BATCH').write_text('DEBUG pause marker')
        with self.assertRaisesRegex(RuntimeError, 'pause marker'):
            self.completion()

    def test_full_completion_requires_40_epochs_680_updates_cuda_and_all_gradients(self):
        changes = dict(completed_epochs=39, optimizer_updates=679, actual_CUDA=False, debug=True,
                       full_training=False, connected_parameter_tensors=980, expected_parameter_tensors=980,
                       recipient_GT_used_in_forward=True, status='DEBUG_COMPLETE')
        for name, value in changes.items():
            with self.subTest(name=name):
                receipt = dict(self.receipt, **{name: value})
                self.write_json(self.source / 'training/invocations.jsonl', receipt)
                with self.assertRaises(ValueError):
                    self.completion()
        self.write_completion(5)

    def test_full_complete_keeps_own_best_delegation(self):
        value = self.completion()
        self.assertEqual(value['invocation']['best']['epoch'], 7)
        self.assertEqual(value['invocation']['optimizer_updates'], 680)
        self.assertEqual(value['checkpoint_validation_delegated_to'], 'pin-current-gnn')

    def test_source_gpu_cannot_be_crossed(self):
        self.write_json(self.source / 'request.json', dict(physical_GPU=6))
        with self.assertRaisesRegex(ValueError, 'another GPU'):
            self.completion()

    def test_final_train_stage_must_have_completed(self):
        status = chain.read(self.job / 'status.json'); status['stages_completed'] = ['prepare']
        with self.assertRaisesRegex(ValueError, 'final GNN train'):
            self.completion(status)

    def test_source_worker_request_change_is_detected_before_process_inspection(self):
        self.write_json(self.job / 'request.json', {'GPU': 6})
        with self.assertRaisesRegex(ValueError, 'request changed'):
            chain.inspect_source(self.request)

    def test_child_stage_argv_is_bound_to_the_owned_pipeline(self):
        with patch.object(chain, 'process_witness', return_value=None) as witness:
            chain.inspect_source(self.request)
        self.assertEqual(witness.call_args_list[-1].args, (456, 11., ['DEBUG_owned_train_command']))

    def test_actual_gpu6_original_key_schema_is_admitted_without_optional_fallback(self):
        self.request = self.make_request(6); self.write_completion(6)
        with patch.object(chain, 'process_witness', return_value=None):
            self.assertIsNotNone(chain.inspect_source(self.request))
        pipeline = chain.read(self.job / 'request.json')
        pipeline['epochs'] = pipeline.pop('original_total_epochs')
        self.write_json(self.job / 'request.json', pipeline)
        status = chain.read(self.job / 'status.json'); status['request'] = pipeline
        self.write_json(self.job / 'status.json', status)
        self.request['source_job']['pipeline_request_sha256'] = chain.sha(self.job / 'request.json')
        with patch.object(chain, 'process_witness', return_value=None), self.assertRaisesRegex(ValueError, 'own-arm contract'):
            chain.inspect_source(self.request)

    def test_mutated_scientific_source_metadata_fails_checksum_before_launch(self):
        scientific = chain.read(self.source / 'request.json'); scientific['config']['epochs'] = 1
        self.write_json(self.source / 'request.json', scientific)
        with self.assertRaisesRegex(ValueError, 'immutable checksum'):
            chain.inspect_source(self.request)

    def test_lease_rejects_duplicate_registration_without_overwriting(self):
        chain.claim(self.target, 'a' * 64)
        original = (self.target / 'launch.lease.json').read_bytes()
        with self.assertRaises(FileExistsError):
            chain.claim(self.target, 'b' * 64)
        self.assertEqual((self.target / 'launch.lease.json').read_bytes(), original)
        with self.assertRaises(FileExistsError):
            chain.validate_request(self.request)

    def test_full_contract_and_commands_cannot_shrink_or_change_gpu(self):
        for mutation in ('batch', 'gpu', 'output', 'stage'):
            request = copy.deepcopy(self.request)
            if mutation == 'batch':
                request['stages'][-1]['command'] += ['--debug-epochs', '1']
            elif mutation == 'gpu':
                command = request['stages'][-1]['command']; command[command.index('--gpu') + 1] = '6'
            elif mutation == 'output':
                command = request['stages'][0]['command']; command[command.index('--output') + 1] = str(self.source / 'pin.json')
            else:
                request['stages'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                chain.validate_request(request)

    def test_preexisting_native_outputs_are_rejected(self):
        self.target.mkdir(); (self.target / 'native').mkdir()
        with self.assertRaisesRegex(FileExistsError, 'no replay'):
            chain.validate_request(self.request)

    def test_actual_bank_embeds_this_own_arm_pin(self):
        self.target.mkdir(); (self.target / 'bank').mkdir()
        pin = dict(physical_GPU=5, selected=dict(gpu=5, best_record=dict(epoch=7)))
        self.write_json(self.target / 'pin.json', pin)
        bank = dict(pipeline_version=chain.CURRENT_FORMAT, physical_GPU=5, complete=True, pin=pin)
        self.write_json(self.target / 'bank/index.json', bank)
        self.assertEqual(chain.stage_proof(self.request, 'prepare-current-bank')['action'], 'prepare-current-bank')
        bank['pin']['selected']['gpu'] = 6
        self.write_json(self.target / 'bank/index.json', bank)
        with self.assertRaises(ValueError):
            chain.stage_proof(self.request, 'prepare-current-bank')

    def test_native_complete_receipt_requires_actual_checkpoint_checksum(self):
        self.target.mkdir(); native = self.target / 'native'; native.mkdir()
        checkpoint = native / 'checkpoint_final.pth'; checkpoint.write_bytes(b'DEBUG checksum fixture, not a model')
        receipt = dict(epochs=250, checkpoint=str(checkpoint), checkpoint_sha256=chain.sha(checkpoint))
        self.write_json(native / 'training_complete_DEBUG.json', receipt)
        self.assertEqual(chain.stage_proof(self.request, 'train')['action'], 'train')
        checkpoint.write_bytes(b'changed DEBUG file')
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            chain.stage_proof(self.request, 'train')

    def test_native_child_failure_stops_before_all_downstream_stages(self):
        request_path = self.root / 'registered_request.json'; self.write_json(request_path, self.request)
        self.write_claim(request_path)
        events = []
        def failed_stage(request, stage, *args):
            events.append(stage['action'])
            raise RuntimeError('DEBUG stage failure; no actual child')
        with patch.object(chain, 'verify_code'), patch.object(chain, 'inspect_source', return_value={'full': True}), \
                patch.object(chain, 'resources', return_value={}), patch.object(chain, 'run_stage', side_effect=failed_stage), \
                patch('psutil.Process') as process:
            process.return_value.cpu_affinity.return_value = self.request['CPU_affinity']
            process.return_value.create_time.return_value = 100.
            with self.assertRaisesRegex(RuntimeError, 'DEBUG stage failure'):
                chain.run(request_path)
        self.assertEqual(events, ['pin-current-gnn'])
        status = chain.read(self.target / 'status.json')
        self.assertEqual(status['status'], 'FAILED')
        self.assertEqual(status['stages_completed'], [])
        self.assertTrue((self.target / 'launch.lease.json').is_file())

    def test_waiting_allocation_creates_no_child_until_adequate(self):
        request_path = self.root / 'registered_request.json'; self.write_json(request_path, self.request)
        self.write_claim(request_path)
        events = []
        def resource_check(*args, **kwargs):
            events.append(('resources', kwargs['stage_index']))
            if len(events) == 1:
                raise chain.AllocationUnavailable('DEBUG insufficient original walltime')
            return {}
        def completed_stage(request, stage, status, *args):
            events.append(('stage', stage['action']))
            status['stages_completed'].append(stage['name'])
        with patch.object(chain, 'verify_code'), patch.object(chain, 'inspect_source', return_value={'full': True}), \
                patch.object(chain, 'resources', side_effect=resource_check), patch.object(chain, 'run_stage', side_effect=completed_stage), \
                patch.object(chain.time, 'sleep'), patch('psutil.Process') as process:
            process.return_value.cpu_affinity.return_value = self.request['CPU_affinity']
            process.return_value.create_time.return_value = 100.
            chain.run(request_path)
        self.assertEqual(events[:3], [('resources', 0), ('resources', 0), ('stage', 'pin-current-gnn')])
        self.assertEqual(chain.read(self.target / 'status.json')['status'], 'COMPLETE')

    def test_scheduler_uses_wallclock_expiry_despite_lagging_counters(self):
        request = self.make_request(6)
        start = time.mktime(time.strptime('Sat Oct 10 07:00:55 2026', '%a %b %d %H:%M:%S %Y'))
        expiry = start + 86400; request['scheduler']['expires_at_unix'] = expiry
        output = '\n'.join(['Job Id: 129443.ECE-util1', '    job_state = R',
            '    Job_Owner = aicompetition06@ece-util1', '    exec_host = ece-a6gpu6/0*6',
            '    Resource_List.ncpus = 6', '    Resource_List.ngpus = 1', '    Resource_List.mem = 128gb',
            '    Resource_List.walltime = 24:00:00', '    resources_used.walltime = 01:00:00',
            '    stime = Sat Oct 10 07:00:55 2026'])
        with patch.object(chain, '_run', return_value=output), patch.object(chain.time, 'time', return_value=expiry - 27000):
            with self.assertRaises(chain.AllocationUnavailable):
                chain.scheduler_proof(request, stage_index=0)
            proof = chain.scheduler_proof(request, stage_index=4)
        self.assertEqual(proof['remaining_seconds'], 27000)
        self.assertEqual(proof['required_remaining_seconds'], 25200)

    def test_code_sha_tampering_fails_closed(self):
        (self.code / 'tools/run_v24_nnunet_cp.py').write_text('changed DEBUG source')
        with patch.object(chain.subprocess, 'check_output', return_value='a' * 40 + '\n'):
            with self.assertRaisesRegex(ValueError, 'Actual source changed'):
                chain.verify_code(self.request)

    def test_second_fresh_chain_cannot_reuse_one_source_claim(self):
        request_path = self.root / 'registered_request.json'; self.write_json(request_path, self.request)
        self.write_claim(request_path)
        self.assertEqual(chain.verify_source_claim(self.request, chain.sha(request_path)), chain.sha(self.request['source_claim_path']))
        other = dict(self.request, chain_root=str(self.root / 'second_chain'))
        with self.assertRaisesRegex(ValueError, 'another native chain'):
            chain.verify_source_claim(other, chain.sha(request_path))

    def test_alternate_global_claim_parent_is_rejected(self):
        request = dict(self.request, source_claim_path=str(self.root / 'another_claims/source.json'))
        with self.assertRaisesRegex(ValueError, 'globally derived'):
            chain.validate_request(request)


if __name__ == '__main__':
    unittest.main()
