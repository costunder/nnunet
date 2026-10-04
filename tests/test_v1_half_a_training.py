"""CPU orchestration/metadata regression tests; no medical/GPU training claim.

Synthetic UNIT checkpoint metadata follows the actual archived publication
format. Real model initialization and learning are verified separately.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

from hiercp_v1x import half_a_entry as entry, half_a_training as training
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.experiment import digest
from hiercp_v1x.half_a_model import model_contract
from tools import run_v1_half_a as runner


ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'ab' * 32
NATIVE_ARCHITECTURE = 'hiercp_source_content_population_metric_v5'


def archived_config():
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as bundle:
        config = json.loads(bundle.read('config/train.json'))
    config['graph']['adaptive_roi_margin_mm'] = 10.0
    config['graph']['context_outer_radius_mm'] = 10.0
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def reseal(receipt):
    result = deepcopy(receipt)
    result['contract_sha256'] = canonical_hash({key: value for key, value in result.items()
                                                if key != 'contract_sha256'})
    return result


def synthetic_unit_baseline(directory):
    """Owned tiny evidence fixture, never represented as a real trained model."""
    baseline = Path(directory).resolve() / 'UNIT_baseline'
    baseline.mkdir()
    config = archived_config()
    manifest = dict(config=config, margin_mm=10.0, contract_sha256='cd' * 32,
                    source_experiment=str(Path(directory).resolve() / 'UNIT_native'),
                    split=dict(train=[f'UNIT_train_{i}' for i in range(84)],
                               val=[f'UNIT_val_{i}' for i in range(21)],
                               outer_validation_excluded=[f'UNIT_outer_{i}' for i in range(26)]))
    write_json(baseline / 'manifest.json', manifest)
    write_json(baseline / 'config.json', config)
    result = baseline / 'results/v1.0'
    result.mkdir(parents=True)
    write_json(result / 'initial_validation.json', dict(
        format='hiercp_v1_bounded_initial_validation_v1', epoch=0,
        scope_contract_sha256=SCOPE, optimizer_steps=0, checkpoint_created=False,
        full_validation=True, metrics=dict(mrr=.3, acc=.1, margin=-.01)))
    for name in ('config.json', 'index.json', 'complete.json'):
        write_json(baseline / 'shared/cache' / name, {'UNIT_metadata_only': name})
    write_json(baseline / 'shared/cache/index.json', dict(UNIT_metadata_only=True, entries=[
        dict(path='UNIT_train_0__000.pt', case_id='UNIT_train_0', sample_index=0, split='train'),
        dict(path='UNIT_val_0__000.pt', case_id='UNIT_val_0', sample_index=0, split='val')]))
    for name in ('manifest.csv',):
        (baseline / 'shared/cache' / name).write_text('UNIT_metadata_only\n', encoding='utf8')
    for name in ('split.json', 'metadata.json'):
        write_json(baseline / 'shared' / name, {'UNIT_metadata_only': name})
    for name in ('prototype_bank.pt', 'manifest.csv'):
        (baseline / 'shared' / name).write_bytes(b'UNIT metadata only, no real bank')
    publication = {name + '_sha256': digest(baseline / 'shared/cache' / (name + '.json'))
                   for name in ('config', 'index', 'complete')}
    fingerprint = dict(selected_device='cuda', gpu_devices=[{'UNIT_model': 'measured_gpu'}],
                       cuda_visible_devices_env='3', cpu_affinity_cores=16)
    measured = dict(physical_batch=16, workers=8, resource_fingerprint=fingerprint)
    preflight = dict(format='hiercp_preflight_calibration_v2',
                     selected_batch_size=16, selected_num_workers=8,
                     resource_fingerprint=fingerprint,
                     identity=dict(seed=42, run_mode='production',
                         cache_dir=str(baseline / 'shared/cache'),
                         checkpoint_path=str(result / 'checkpoint_best.pt'),
                         graph_config=config['graph'], model_kwargs=config['model'],
                         cache_publication=publication,
                         train_cache_files=['UNIT_train_0__000.pt'], val_cache_files=['UNIT_val_0__000.pt']))
    signature = dict(format='hiercp_training_signature_v1', run_mode='production',
        target_epochs=40, seed=42, batch_setting='auto', batch_size=16,
        worker_setting='auto', num_workers=8, gradient_accumulation_setting=1,
        gradient_accumulation_steps=1, target_effective_batch_size=None,
        resolved_effective_batch_size=16, edge_attention_execution='hiercp_exact_edge_streaming_v1',
        calibration_resource_fingerprint=fingerprint, consistency_weight=config['training']['consistency_weight'],
        optimizer=dict(name='AdamW', lr=config['training']['lr'],
                       weight_decay=config['training']['weight_decay'], fused=True),
        scheduler=dict(name='CosineAnnealingLR', t_max=40), amp=True,
        grad_clip=config['training']['grad_clip'], deterministic=True, allow_tf32=False,
        curriculum={key: config['training'][key] for key in (
            'easy_epochs', 'inter_epochs', 'intra_epochs', 'model_mine_start_epoch',
            'semi_hard_low_percentile', 'semi_hard_high_percentile',
            'cross_entropy_weight', 'pairwise_weight', 'ordinal_weight', 'mined_weight')},
        train_cache_files=['UNIT_train_0__000.pt'], val_cache_files=['UNIT_val_0__000.pt'])
    selection = dict(mrr=1., acc=1., margin=14., ranking=.01, consistency=.02)
    validation_policy = dict(format='hiercp_fixed_validation_v1', epoch=29,
                             checkpoint_order=['mrr', 'acc', 'margin', '-ranking', '-consistency'],
                             metric_precision=8)
    last = dict(format='hiercp_training_state_v1', epoch=40, target_epochs=40,
        training_complete=True, architecture_version=NATIVE_ARCHITECTURE + '|bounded_scope_' + SCOPE,
        method='hiercp-full', framework='torch_geometric', geometry_contract=config['graph']['geometry_contract'],
        upper_feature_policy='source_content_observed_ct_population_v4', model_kwargs=config['model'],
        graph_config=config['graph'], ct_clip=tuple(config['ct_clip']), cache_publication=publication,
        state_dict={'v1x_bounded_scope_digest': torch.tensor(list(bytes.fromhex(SCOPE)), dtype=torch.uint8)},
        training_signature=signature, preflight_calibration=preflight,
        gradient_connectivity=dict(format='hiercp_gradient_connectivity_v1', verified=True,
                                   expected_parameter_count=0, connected_parameter_count=0,
                                   connected_parameters=[], missing_parameters=[]),
        best_epoch=11, best_mrr=1., best_selection=selection, validation_policy=validation_policy,
        best_checkpoint=str(result / 'checkpoint_best.pt'), optimizer_state_dict={},
        scheduler_state_dict={}, scaler_state_dict={}, rng_state={},
        train_shuffle_generator_state=torch.tensor([1], dtype=torch.uint8),
        train_worker_generator_state=torch.tensor([2], dtype=torch.uint8),
        val_worker_generator_state=torch.tensor([3], dtype=torch.uint8))
    best = deepcopy(last)
    best.pop('format')
    best.update(epoch=11, completed_epoch=40, selection=selection, selection_mrr=1.)
    write_json(result / 'checkpoint_best.pt.preflight.json', preflight)
    torch.save(last, result / 'checkpoint_best.last.pt')
    torch.save(best, result / 'checkpoint_best.pt')
    return baseline, manifest, measured, publication, last, best


class BaselineAndIsolationUnits(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='half_A_training_UNIT_', dir=ROOT / 'work')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.baseline, self.native, self.execution, self.publication,
         self.last, self.best) = synthetic_unit_baseline(self.directory)
        self.original_verifier = patch('tools.run_v1_bounded_training.verify_bound_experiment',
                                       return_value=deepcopy(self.native))
        self.original_verifier.start()
        self.addCleanup(self.original_verifier.stop)

    def test_actual_completed_publication_format_is_read_without_writing_baseline(self):
        before = {str(path): digest(path) for path in self.baseline.rglob('*') if path.is_file()}
        native, proof = training.baseline_proof(self.baseline)
        self.assertEqual(native, self.native)
        self.assertEqual(proof['execution'], self.execution)
        self.assertEqual(proof['neural_baseline']['completed_epoch'], 40)
        self.assertEqual(proof['neural_baseline']['best_epoch'], 11)
        self.assertTrue(proof['cache_reused_without_preparation'])
        self.assertEqual(before, {str(path): digest(path) for path in self.baseline.rglob('*') if path.is_file()})

    def test_incomplete_or_differently_bound_last_state_is_rejected(self):
        changes = [('format', 'old'), ('epoch', 39), ('target_epochs', 39),
                   ('training_complete', False), ('cache_publication', {}),
                   ('model_kwargs', {}), ('architecture_version', 'another_model|bounded_scope_' + SCOPE)]
        for key, value in changes:
            with self.subTest(key=key):
                changed = deepcopy(self.last)
                changed[key] = value
                with self.assertRaises(ValueError):
                    training.validate_completed_state(changed, self.baseline, self.native['config'],
                                                       self.execution, SCOPE, self.publication)

    def test_loss_optimizer_curriculum_recipe_changes_cannot_claim_original_baseline(self):
        for key, value in [('consistency_weight', 0.), ('scheduler', {'name': 'Other', 't_max': 40}),
                           ('optimizer', {'name': 'AdamW', 'lr': .1, 'weight_decay': 0., 'fused': True}),
                           ('curriculum', {}), ('train_cache_files', ['another_training_file.pt'])]:
            with self.subTest(key=key):
                changed = deepcopy(self.last)
                changed['training_signature'][key] = value
                with self.assertRaises(ValueError):
                    training.validate_completed_state(changed, self.baseline, self.native['config'],
                                                       self.execution, SCOPE, self.publication)

    def test_best_requires_full_run_completion_and_exact_uint8_scope_marker(self):
        changes = [('training_complete', False), ('completed_epoch', 39),
                   ('target_epochs', 39), ('epoch', 12), ('graph_config', {})]
        for key, value in changes:
            with self.subTest(key=key):
                changed = deepcopy(self.best)
                changed[key] = value
                torch.save(changed, self.baseline / 'results/v1.0/checkpoint_best.pt')
                with self.assertRaises(ValueError):
                    training.baseline_proof(self.baseline)
        changed = deepcopy(self.best)
        changed['state_dict']['v1x_bounded_scope_digest'] = changed['state_dict']['v1x_bounded_scope_digest'].float()
        torch.save(changed, self.baseline / 'results/v1.0/checkpoint_best.pt')
        with self.assertRaises(ValueError):
            training.baseline_proof(self.baseline)

    def test_calibration_from_another_checkpoint_cache_seed_or_invalid_measured_value_rejected(self):
        base = json.loads((self.baseline / 'results/v1.0/checkpoint_best.pt.preflight.json').read_text())
        for category, key, value in [
            ('identity', 'checkpoint_path', str(self.directory / 'other.pt')),
            ('identity', 'cache_dir', str(self.directory / 'other_cache')),
            ('identity', 'seed', 41), ('identity', 'run_mode', 'debug'),
            ('top', 'selected_batch_size', True), ('top', 'selected_batch_size', 0),
            ('top', 'selected_num_workers', -1), ('top', 'resource_fingerprint', {})]:
            with self.subTest(key=key):
                changed = deepcopy(base)
                (changed['identity'] if category == 'identity' else changed)[key] = value
                with self.assertRaises(ValueError):
                    training.measured_execution(changed, self.baseline)

    def test_baseline_race_during_checkpoint_read_is_detected(self):
        original_load = torch.load
        def changed_during_read(path, *args, **kwargs):
            result = original_load(path, *args, **kwargs)
            if Path(path).name == 'checkpoint_best.pt':
                (self.baseline / 'shared/cache/manifest.csv').write_text('UNIT changed during read\n')
            return result
        with patch.object(torch, 'load', side_effect=changed_during_read):
            with self.assertRaisesRegex(ValueError, 'changed while reading'):
                training.baseline_proof(self.baseline)

    def initialize(self):
        return training.initialize(self.baseline, self.directory / 'UNIT_half_A', 3, 40., 192.)

    def test_new_arm_changes_only_measured_execution_and_never_baseline_files(self):
        before = {str(path): digest(path) for path in self.baseline.rglob('*') if path.is_file()}
        root, receipt = self.initialize()
        expected = deepcopy(self.native['config'])
        expected['training'].update(batch_size=16, num_workers=8)
        self.assertEqual(receipt['config'], expected)
        self.assertEqual(training.verify(root), receipt)
        self.assertEqual(receipt['source'], str(self.baseline / 'source/v1.0'))
        self.assertFalse(any(receipt[key] for key in ('preparation_started', 'baseline_retraining',
                         'B_automatic_start', 'combined_automatic_start', 'target_supervision_changed')))
        self.assertEqual(before, {str(path): digest(path) for path in self.baseline.rglob('*') if path.is_file()})

    def test_existing_unbound_output_and_overlapping_output_are_preserved(self):
        unbound = self.directory / 'UNIT_existing'
        unbound.mkdir()
        note = unbound / 'preserved.txt'
        note.write_text('UNIT existing evidence')
        with self.assertRaises(FileExistsError):
            training.initialize(self.baseline, unbound, 3, 40., 192.)
        self.assertEqual(note.read_text(), 'UNIT existing evidence')
        for path in (self.baseline, self.baseline / 'nested', self.baseline.parent):
            with self.subTest(path=path), self.assertRaises(ValueError):
                training.initialize(self.baseline, path, 3, 40., 192.)

    def test_resealed_recipe_source_and_missing_implementation_cannot_silently_migrate(self):
        root, original = self.initialize()
        changes = [('B_automatic_start', True), ('combined_automatic_start', True),
                   ('preparation_started', True), ('learning_target', 'other_GT'),
                   ('source', str(self.directory / 'another_source')),
                   ('implementation', {})]
        for key, value in changes:
            with self.subTest(key=key):
                changed = deepcopy(original)
                changed[key] = value
                write_json(root / 'manifest.json', reseal(changed))
                with self.assertRaises(ValueError):
                    training.verify(root)
        changed = deepcopy(original)
        changed['config']['training']['epochs'] = 2
        changed = reseal(changed)
        write_json(root / 'config.json', changed['config'])
        write_json(root / 'manifest.json', changed)
        with self.assertRaises(ValueError):
            training.verify(root)


def actual_archived_prerun_payload():
    """Evaluate only the original report dict AST, never the training function."""
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as bundle:
        source = bundle.read('hiercp/pipeline.py').decode()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_train')
    assignment = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'pre_run_report'
                              for target in node.targets))
    config = archived_config()
    values = dict(run_mode='production', model=SimpleNamespace(), model_kwargs=config['model'],
        parameter_count=123, trainable_parameter_count=112,
        graph_statistics={'UNIT': 'original_statistics'}, input_inventory={'UNIT': 'original_inventory'},
        EDGE_ATTENTION_EXECUTION_VERSION='hiercp_exact_edge_streaming_v1',
        EDGE_ATTENTION_WORKSPACE_BYTES=64 * 1024**2,
        graph_config=SimpleNamespace(patch_size=48, sample_hops=config['graph']['sample_hops']),
        UNAVAILABLE='UNIT_unavailable', cache_usage={'UNIT': 'original_usage', 'subset_active': False},
        batch_size=16, accumulation_steps=1, data_parallel_workers=1, epochs=40,
        train_loader=range(7), math=math, use_amp=True, runtime=config.get('runtime', {}),
        workers=8, training=config['training'], prefetch_factor=1, pin_memory=True, cuda_prefetch=False,
        args=SimpleNamespace(epochs=None, batch_size=None, num_workers=None),
        resource_report={'cuda_visible_device_count': 1})
    return eval(compile(ast.Expression(assignment.value), '<archived-Prerun-UNIT>', 'eval'), values)


class ReportingResourceAndDispatchUnits(unittest.TestCase):
    def test_materialized_cohort_preserves_original_index_and_reports_configured_absences(self):
        files = ['UNIT_case1__000.pt', 'UNIT_case1__001.pt', 'UNIT_case2__000.pt']
        cases = ['UNIT_case1','UNIT_case2','UNIT_case3']
        index = dict(entries=[dict(case_id='UNIT_case1',sample_index=i,split='val',path=files[i]) for i in (0,1)]
                     + [dict(case_id='UNIT_case2',sample_index=0,split='val',path=files[2])])
        row = training.validation_cohort(files, cases, index)
        self.assertEqual(row['case_ids'], cases[:2])
        self.assertEqual(row['samples'],3)
        self.assertEqual(row['configured_but_not_materialized_cases'],cases[2:])
        self.assertEqual(row['case_sample_counts'],{'UNIT_case1':2,'UNIT_case2':1})
        for bad_files,bad_index in ((files[:-1],index),(files+files[:1],index),
                (files,dict(entries=[dict(x,case_id='UNIT_case3') for x in index['entries']]))):
            with self.assertRaises(ValueError):
                training.validation_cohort(bad_files,cases,bad_index)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='half_A_dispatch_UNIT_', dir=ROOT / 'work')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.receipt = dict(experiment=str(self.root), baseline_experiment=str(self.root.parent / 'UNIT_reference'),
                            contract_sha256='ef' * 32,
                            config={'training': {'gradient_accumulation_steps': 1}},
                            baseline_proof={'execution': dict(physical_batch=16, workers=8,
                                resource_fingerprint={'UNIT_allocation': 'same'})})

    def test_prerun_adapter_consumes_actual_archived_keys_without_mutating_original_report(self):
        original = actual_archived_prerun_payload()
        before = deepcopy(original)
        reports = []
        pipeline = SimpleNamespace(_print_report=lambda name, payload: reports.append((name, payload)))
        entry.report_actual_encoder(pipeline, self.receipt)
        pipeline._print_report('PreRun', original)
        name, actual = reports[-1]
        self.assertEqual(name, 'PreRun')
        self.assertEqual(original, before)
        self.assertEqual(actual['model']['local_layers'], 0)
        self.assertEqual(actual['model']['configuration']['local_layers_argument'], 3)
        self.assertNotIn('local_layers', actual['model']['configuration'])
        self.assertEqual(actual['model']['local_encoder'], model_contract())
        self.assertFalse(actual['graph_and_input']['local_message_passing'])
        for section in ('optimization', 'dataset', 'loader', 'flags', 'device_resources', 'parallelism'):
            self.assertEqual(actual[section], before[section])
        pipeline._print_report('EpochPostRun', original)
        self.assertIs(reports[-1][1], original)

    def test_execution_lock_records_inherited_measurement_and_repeat_preserves_bytes(self):
        fingerprint = deepcopy(self.receipt['baseline_proof']['execution']['resource_fingerprint'])
        pipeline = SimpleNamespace(_calibration_resource_fingerprint=lambda report, *, device: deepcopy(report))
        entry.bind_execution_resources(pipeline, self.receipt)
        self.assertEqual(pipeline._calibration_resource_fingerprint(fingerprint, device='cuda'), fingerprint)
        path = self.root / 'execution_lock.json'
        before = path.read_bytes()
        lock = json.loads(before)
        self.assertFalse(lock['newly_calibrated'])
        self.assertEqual((lock['physical_batch'], lock['workers'], lock['effective_batch']), (16, 8, 16))
        self.assertEqual(lock['half_A_contract_sha256'], self.receipt['contract_sha256'])
        pipeline._calibration_resource_fingerprint(fingerprint, device='cuda')
        self.assertEqual(path.read_bytes(), before)

    def test_allocation_change_rejected_before_publishing_execution_lock(self):
        pipeline = SimpleNamespace(_calibration_resource_fingerprint=lambda report, *, device: report)
        entry.bind_execution_resources(pipeline, self.receipt)
        with self.assertRaisesRegex(ValueError, 'allocation differs'):
            pipeline._calibration_resource_fingerprint({'UNIT_allocation': 'different'}, device='cuda')
        self.assertFalse((self.root / 'execution_lock.json').exists())

    def test_changed_existing_execution_lock_is_not_overwritten(self):
        path = self.root / 'execution_lock.json'
        write_json(path, {'UNIT_existing_lock': 'preserved'})
        before = path.read_bytes()
        pipeline = SimpleNamespace(_calibration_resource_fingerprint=lambda report, *, device: report)
        entry.bind_execution_resources(pipeline, self.receipt)
        with self.assertRaisesRegex(ValueError, 'lock changed'):
            pipeline._calibration_resource_fingerprint({'UNIT_allocation': 'same'}, device='cuda')
        self.assertEqual(path.read_bytes(), before)

    def test_controller_dispatches_only_one_half_a_child_on_success_and_failure(self):
        for code in (0, 7):
            with self.subTest(returncode=code):
                process = SimpleNamespace(wait=lambda: code, pid=123)
                with patch('hiercp_v1x.half_a_training.verify', return_value=self.receipt), \
                     patch('hiercp_v1x.experiment.owned_lock', return_value=nullcontext()), \
                     patch('hiercp_v1x.snapshot_inventory.isolated_snapshot_bytecode_env',
                           return_value={'PYTHONPATH': 'UNIT_old', 'HIERCP_V1X_SAMPLING_CONTRACT': 'UNIT_old', 'KEEP': 'yes'}), \
                     patch.object(runner.subprocess, 'Popen', return_value=process) as launched:
                    if code:
                        with self.assertRaises(RuntimeError):
                            runner.run(self.root, self.receipt)
                    else:
                        runner.run(self.root, self.receipt)
                self.assertEqual(launched.call_count, 1)
                command = launched.call_args.args[0]
                self.assertEqual(command[:4], [sys.executable, '-u', '-m', 'hiercp_v1x.half_a_entry'])
                self.assertEqual(command[4], '--request')
                request = json.loads(Path(command[5]).read_text(encoding='utf8'))
                self.assertEqual(request, dict(experiment=str(self.root), contract_sha256=self.receipt['contract_sha256']))
                self.assertNotIn('PYTHONPATH', launched.call_args.kwargs['env'])
                self.assertNotIn('HIERCP_V1X_SAMPLING_CONTRACT', launched.call_args.kwargs['env'])

    def test_entry_exposes_no_prepare_complement_combined_or_scale_override_dispatch(self):
        source = Path(entry.__file__).read_text(encoding='utf8')
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        commands = [node.value for node in ast.walk(main) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                            and target.value.id == 'sys' and target.attr == 'argv' for target in node.targets)
                    and isinstance(node.value, ast.List)]
        self.assertEqual(len(commands), 1)
        self.assertEqual([element.value for element in commands[0].elts[:2]], ['hiercp.pipeline', 'train'])
        literals = {node.value for node in ast.walk(commands[0]) if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        self.assertTrue({'--config', '--cache-dir', '--prototype-bank', '--checkpoint', '--device'} <= literals)
        self.assertFalse({'prepare', 'generate', '--epochs', '--batch-size', '--num-workers',
                          '--max-cases', '--overwrite', 'half_B', 'combined'} & literals)


if __name__ == '__main__':
    unittest.main()
