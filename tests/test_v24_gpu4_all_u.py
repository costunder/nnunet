"""CPU UNIT full-bank policy and signed metadata tests; no native GPU training."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x.v23_data import V23Population
from hiercp_v1x.v24_targets import new_curriculum, finish_curriculum_epoch, validate_policy
from tools.run_v24_all_p import FILES, execution_source_files, parse, validate_config
from tests.test_v23_data import unit_inventory
from tools import run_v24_gpu4_all_u as wrapper
from tools import run_v24_gpu4_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]
GPU4 = ROOT / 'config/v24_gpu4_STUNetS_all_U_GT_blind.json'
GPU6 = ROOT / 'config/v24_gpu6_STUNetS_GT_blind.json'


def metadata_report(epoch, phase, cases, *, mrr, top1):
    return dict(epoch=epoch, active_u=128, phase=phase,
        evaluation_view_epoch=29, all_P_scored=True,
        cases=[dict(case_id=case) for case in cases],
        metrics=dict(per_P_patient_mrr=mrr, per_P_patient_top1=top1,
                     patient_balanced_pair_loss=.2))


class GPU4FullNativeBankUnit(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(GPU4.read_text(encoding='utf8'))
        self.policy = self.config['v24_runtime']['curriculum']

    def test_forty_complete_epochs_keep_all_native_U_despite_all_gate_outcomes(self):
        # Actual controller transitions, including success, failure and plateau,
        # cannot select seven U or defer admission in the GPU4 experiment.
        cases = ['CPU_UNIT_patient_A', 'CPU_UNIT_patient_B']
        for outcome in ('mastered', 'never_mastered', 'alternating'):
            with self.subTest(outcome=outcome):
                state = new_curriculum(self.policy)
                trained = []
                for epoch in range(1, 41):
                    trained.append(state['active_u'])
                    mrr, top1 = ((.95, .9) if outcome == 'mastered' or
                        (outcome == 'alternating' and epoch % 2) else (.3, .1))
                    state, receipt = finish_curriculum_epoch(state,
                        metadata_report(epoch, 'train_probe', cases, mrr=.96, top1=.95),
                        metadata_report(epoch, 'stage_validation', cases, mrr=mrr, top1=top1),
                        epoch=epoch, actual_train_cases=cases,
                        expected_train_cases=cases, updates_in_epoch=1)
                    self.assertEqual(receipt['previous_active_u'], 128)
                    self.assertEqual(receipt['next_active_u'], 128)
                    self.assertFalse(receipt['expanded'])
                    self.assertEqual(receipt['reason'], 'full_bank_replay')
                self.assertEqual(trained, [128] * 40)
                self.assertEqual(state['last_completed_epoch'], 40)
                self.assertEqual(len(state['history']), 40)
        proof = validate_policy(self.policy)
        self.assertEqual(proof['latest_first_full_bank_epoch'], 1)
        self.assertEqual(proof['guaranteed_full_bank_epochs'], 40)
        self.assertEqual(proof['stages'], 1)

    def test_signed_population_retains_every_P_and_U_in_original_query_order(self):
        # Explicit synthetic metadata fixture exercises the actual production
        # population selector without inventing CT data or model predictions.
        meta = unit_inventory()
        population = V23Population(meta, debug=True)
        for partition in ('inner_train', 'inner_val'):
            for case in population.partition_cases(partition):
                plan = population.case(case, self.policy['initial_u'])
                original = [row for row in meta['records'] if row['case_id'] == case]
                self.assertEqual(plan.record_ids, tuple(row['id'] for row in original))
                self.assertEqual(plan.active_u_indices, tuple(range(128)))
                self.assertEqual(len(plan.unobserved_indices), 128)
                self.assertEqual(plan.observed_P, sum(row['target'] == 1 for row in original))
                self.assertTrue(all('target' not in row and 'component' not in row
                                    for row in plan.query_rows))

    def test_model_data_resource_batch_and_epoch_contract_match_GPU6(self):
        original = json.loads(GPU6.read_text(encoding='utf8'))
        expected = copy.deepcopy(original)
        expected['experiment'] = self.config['experiment']
        expected['v24_runtime'].update(physical_GPU=4,
            U_order=self.config['v24_runtime']['U_order'], all_U_from_epoch_one=True)
        expected['v24_runtime']['curriculum']['initial_u'] = 128
        self.assertEqual(self.config, expected)
        self.assertEqual(self.config['encoder'], 'official_pretrained_STU_Net_S')
        self.assertEqual(self.config['v24_runtime']['physical_patient_batch_candidates'], [4, 8])
        self.assertEqual(self.config['v24_runtime']['physical_candidate_batch_candidates'], [32, 64, 128])
        self.assertIs(validate_config(self.config, 4, Path('CPU_UNIT_official_STU_asset')), self.config)

    def test_GPU4_CLI_accepts_explicit_official_checkpoint_and_full_population(self):
        args = parse(['--mode', 'train', '--config', str(GPU4),
            '--native-experiment', 'CPU_UNIT_native', '--inventory', 'CPU_UNIT_inventory',
            '--input-cache', 'CPU_UNIT_fresh_cache', '--output', 'CPU_UNIT_fresh_output',
            '--gpu', '4', '--stunet-checkpoint', 'CPU_UNIT_official_STU_asset'])
        self.assertEqual(args.gpu, 4)
        self.assertIsNone(args.resume)
        self.assertIs(validate_config(self.config, args.gpu, args.stunet_checkpoint), self.config)

    def test_missing_official_asset_or_curriculum_start_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint must be supplied'):
            validate_config(self.config, 4)
        for field, value in (('initial_u', 7), ('total_u', 119), ('total_epochs', 39)):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.config)
                bad['v24_runtime']['curriculum'][field] = value
                with self.assertRaisesRegex(ValueError, 'every native128U'):
                    validate_config(bad, 4, Path('CPU_UNIT_official_STU_asset'))
        bad = copy.deepcopy(self.config)
        bad['v24_runtime']['all_U_from_epoch_one'] = False
        with self.assertRaisesRegex(ValueError, 'every native128U'):
            validate_config(bad, 4, Path('CPU_UNIT_official_STU_asset'))

    def test_existing_GPU5_and_GPU6_configs_keep_their_original_contract(self):
        for gpu, filename in ((5, 'v24_gpu5_GT_blind.json'),
                              (6, 'v24_gpu6_STUNetS_GT_blind.json')):
            with self.subTest(gpu=gpu):
                config = json.loads((ROOT / 'config' / filename).read_text(encoding='utf8'))
                checkpoint = Path('CPU_UNIT_official_STU_asset') if gpu == 6 else None
                self.assertIs(validate_config(config, gpu, checkpoint), config)
                self.assertEqual(new_curriculum(config['v24_runtime']['curriculum'])['active_u'], 7)
        bad = copy.deepcopy(self.config)
        bad['encoder'] = 'original_CNN_GAT'
        with self.assertRaisesRegex(ValueError, 'official pretrained STU-Net-S'):
            validate_config(bad, 4, Path('CPU_UNIT_official_STU_asset'))

    def test_GPU4_optimizer_input_source_binding_does_not_redefine_existing_arms(self):
        self.assertEqual(execution_source_files(5), FILES)
        self.assertEqual(execution_source_files(6), FILES)
        gpu4 = execution_source_files(4)
        self.assertEqual(gpu4[:len(FILES)], FILES)
        self.assertEqual(len(gpu4), len(set(gpu4)))
        self.assertEqual(set(gpu4) - set(FILES), {
            'tools/run_v24_gpu4_all_u.py', 'hiercp_v1x/v24_memory_runtime.py',
            'hiercp_v1x/v24_hash_runtime.py', 'hiercp_v1x/v24_prefetch_runtime.py',
            'hiercp_v1x/v24_input_runtime.py'})
        with self.assertRaisesRegex(ValueError, 'Explicit physical GPU4,5or6'):
            execution_source_files(0)


def private_factory_CPU_UNIT_main():
    """Explicit metadata-only import fixture; never constructs a model."""
    from hiercp_v1x.v24_factory import V24NativeInputs, build_runtime
    return V24NativeInputs, build_runtime


class GPU4FreshRunnerUnit(unittest.TestCase):
    def metadata_calibration(self):
        return dict(CPU_UNIT=True,
            selected_physical_patient_batch=8, selected_physical_candidate_batch=128,
            trials=[dict(physical_patient_batch=batch, physical_candidate_batch=chunk,
                         accepted=True, CPU_UNIT=True)
                    for batch in (4, 8) for chunk in (32, 64, 128)])

    def test_fair_comparison_batch_retains_all_measured_trials_and_throughput_winner(self):
        original = self.metadata_calibration()
        before = copy.deepcopy(original['trials'])
        wrapped = wrapper.select_comparison_batch(lambda: original)
        result = wrapped()
        self.assertEqual(result['selected_physical_patient_batch'], 4)
        self.assertEqual(result['selected_physical_candidate_batch'], 64)
        self.assertEqual(result['unconstrained_throughput_winner'],
            dict(physical_patient_batch=8, physical_candidate_batch=128))
        self.assertEqual(result['trials'], before)

    def test_missing_rejected_or_duplicate_comparison_trial_has_no_fallback(self):
        for problem in ('missing', 'rejected', 'duplicate'):
            with self.subTest(problem=problem):
                report = self.metadata_calibration()
                target = next(row for row in report['trials'] if
                    (row['physical_patient_batch'], row['physical_candidate_batch']) == (4, 64))
                if problem == 'missing':
                    report['trials'].remove(target)
                elif problem == 'rejected':
                    target['accepted'] = False
                else:
                    report['trials'].append(copy.deepcopy(target))
                with self.assertRaisesRegex(MemoryError, 'no smaller batch or model fallback'):
                    wrapper.select_comparison_batch(lambda: report)()

    def test_private_factory_import_preserves_original_code_and_global_factory(self):
        from hiercp_v1x import v24_factory
        original = v24_factory.build_runtime
        marker = object()
        cloned = wrapper.clone_main(private_factory_CPU_UNIT_main, lambda build: marker)
        self.assertIs(cloned.__code__, private_factory_CPU_UNIT_main.__code__)
        constructor, injected = cloned()
        self.assertIs(constructor, v24_factory.V24NativeInputs)
        self.assertIs(injected, marker)
        self.assertIs(v24_factory.build_runtime, original)

    def test_fresh_runner_rejects_other_GPU_or_any_trained_resume_before_installation(self):
        common = ['--mode', 'train', '--config', str(GPU4), '--native-experiment', 'CPU_UNIT_native',
                  '--inventory', 'CPU_UNIT_inventory', '--input-cache', 'CPU_UNIT_cache',
                  '--output', 'CPU_UNIT_output', '--stunet-checkpoint', 'CPU_UNIT_official_asset']
        for extra in (['--gpu', '6'], ['--gpu', '4', '--resume', 'CPU_UNIT_GPU6_latest.pt']):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, 'no GPU6 trained GNN or optimizer state'):
                    wrapper.main(common + extra)

    def test_actual_adapter_installation_precedes_CUDA_in_isolated_CPU_UNIT_process(self):
        # Run actual installation in a fresh process so patches cannot escape
        # into other tests. Replace only the preparation entry body; no CT data,
        # STU checkpoint, model, optimizer or CUDA work is performed.
        code = r'''
import json
from unittest.mock import patch
from tools import run_v24_all_p as cli, run_v24_gpu4_all_u as wrapper
def CPU_UNIT_prepare(argv=None):
    import os, torch
    from hiercp_v1x import v24_factory, v24_memory_runtime as memory
    from hiercp_v1x import v24_hash_runtime as hashing, v24_inputs
    from hiercp_v1x import v24_prefetch_runtime as prefetch, v24_input_runtime as inputs
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    assert v24_factory.V24InputCoordinator is memory.MemorySafeCoordinator
    assert v24_factory.V24InputProvider is memory.MemorySafeInputProvider
    assert v24_inputs.tensor_digest is hashing.tensor_digest
    assert memory.MemorySafeCoordinator.trim is prefetch._trim
    assert memory._GET.__globals__['materialize_pair'] is inputs.materialize_pair
    assert memory._GET.__globals__['collate'] is inputs.collate
    return 'CPU_UNIT_pre_CUDA_installation_verified'
with patch.object(cli, 'main', CPU_UNIT_prepare):
    result = wrapper.main(['--mode', 'prepare', '--config', 'config/v24_gpu4_STUNetS_all_U_GT_blind.json',
        '--native-experiment', 'CPU_UNIT_no_data', '--inventory', 'CPU_UNIT_no_data',
        '--input-cache', 'CPU_UNIT_no_cache', '--output', 'CPU_UNIT_no_output', '--gpu', '4',
        '--stunet-checkpoint', 'CPU_UNIT_no_checkpoint'])
print(json.dumps(dict(result=result, actual_GPU_work=False, actual_model_created=False)))
'''
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', code],
            cwd=ROOT, env=environment, text=True, capture_output=True, timeout=45, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt['result'], 'CPU_UNIT_pre_CUDA_installation_verified')
        self.assertFalse(receipt['actual_GPU_work'])
        self.assertFalse(receipt['actual_model_created'])


class GPU4AtomicMetadataAndAdmissionUnit(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='GPU4_atomic_CPU_UNIT_', dir=ROOT/'outputs')
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to((ROOT/'outputs').resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_JSON_is_visible_only_at_atomic_publication(self):
        path = self.directory/'CPU_UNIT_metadata.json'
        value = dict(CPU_UNIT=True, observed_records=[dict(index=index) for index in range(128)])
        real_link = os.link
        observations = []
        def verify_link(source, destination):
            self.assertEqual(Path(destination), path)
            self.assertFalse(path.exists())
            observations.append(json.loads(Path(source).read_text(encoding='utf8')))
            return real_link(source, destination)
        with patch.object(wrapper.os, 'link', side_effect=verify_link):
            wrapper.atomic_new_json(path, value)
        self.assertEqual(observations, [value])
        self.assertEqual(json.loads(path.read_text(encoding='utf8')), value)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_existing_results_are_preserved_for_equal_different_and_new_publications(self):
        path = self.directory/'CPU_UNIT_existing.json'
        original = b'{"CPU_UNIT": true, "value": 7}\n'
        path.write_bytes(original)
        wrapper.atomic_publish(path, dict(CPU_UNIT=True, value=7))
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(FileExistsError, 'old results preserved'):
            wrapper.atomic_publish(path, dict(CPU_UNIT=True, value=8))
        with self.assertRaises(FileExistsError):
            wrapper.atomic_new_json(path, dict(CPU_UNIT=True, value=9))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_JSON_serialization_failure_never_publishes_or_leaves_a_partial_record(self):
        path = self.directory/'CPU_UNIT_failed.json'
        with self.assertRaises(ValueError):
            wrapper.atomic_new_json(path, dict(CPU_UNIT=True, invalid_metric=float('nan')))
        self.assertFalse(path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_exact_GPU_UUID_and_free_memory_are_required_before_CUDA_stage(self):
        request = dict(GPU_uuid='GPU-CPU-UNIT-assigned-4')
        valid = 'GPU-CPU-UNIT-assigned-4, NVIDIA RTX A6000, 46060, 1'
        with patch.object(pipeline.subprocess, 'check_output', side_effect=[valid, '']) as query:
            self.assertEqual(pipeline.assigned_GPU(request), valid)
            self.assertEqual(query.call_args_list[0].args[0][1], '--id=4')
            self.assertEqual(query.call_count, 2)
        for description in ('GPU-CPU-UNIT-other, NVIDIA RTX A6000, 46060, 1',
                            'GPU-CPU-UNIT-assigned-4, NVIDIA RTX A5000, 46060, 1',
                            'GPU-CPU-UNIT-assigned-4, NVIDIA RTX A6000, 40960, 1',
                            'GPU-CPU-UNIT-assigned-4, NVIDIA RTX A6000, 46060, 10'):
            with self.subTest(description=description):
                with patch.object(pipeline.subprocess, 'check_output', return_value=description) as query:
                    with self.assertRaisesRegex(RuntimeError, 'UUID/free40GiB admission failed'):
                        pipeline.assigned_GPU(request)
                    self.assertEqual(query.call_count, 1)

    def test_other_compute_application_prevents_launch_without_affecting_any_process(self):
        request = dict(GPU_uuid='GPU-CPU-UNIT-assigned-4')
        valid = 'GPU-CPU-UNIT-assigned-4, NVIDIA RTX A6000, 46060, 1'
        apps = '123456, GPU-CPU-UNIT-other\n654321, GPU-CPU-UNIT-assigned-4\n'
        with patch.object(pipeline.subprocess, 'check_output', side_effect=[valid, apps]):
            with self.assertRaisesRegex(RuntimeError, 'another compute application'):
                pipeline.assigned_GPU(request)


if __name__ == '__main__':
    unittest.main()
