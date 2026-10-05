"""Metadata/contract UNIT tests only; no CT, model quality, or neural execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from hiercp_v1x import native_transition_receipt as receipt


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')


class NativeReceiptMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='native_receipt_UNIT_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source_checkout'
        module = self.source / 'l0_regions/donor_learning.py'
        module.parent.mkdir(parents=True)
        module.write_text('# UNIT metadata fixture; never executed\n', encoding='utf8')
        self.source_identity = {}
        for group, names in receipt.required_source_files().items():
            self.source_identity[group] = {}
            for name in names:
                path = self.source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# UNIT metadata file fixture; never executed\n', encoding='utf8')
                self.source_identity[group][name] = receipt.sha_file(path)
        self.digest = receipt.sha_file(module)
        write(self.source / 'versions/v1/manifest.json', dict(files={}))
        with ZipFile(self.source / 'versions/v1/pipeline_v1_source.zip', 'w'):
            pass
        for name, path in (('V1_ARCHIVE_SHA256', self.source / 'versions/v1/pipeline_v1_source.zip'),
                           ('V1_MANIFEST_SHA256', self.source / 'versions/v1/manifest.json')):
            pinned = patch.object(receipt, name, receipt.sha_file(path))
            pinned.start()
            self.addCleanup(pinned.stop)
        self.roots = {name: self.root / name for name in ('native', 'baseline', 'A', 'B')}
        actual_repo = Path(__file__).resolve().parents[1]
        self.inventory = dict(format='native_local_cnn_inventory_v1', complete=True, debug=False,
                              config=json.loads((actual_repo / 'config/prompt_graph_v222_v1_l0.json').read_text()),
                              base=json.loads((actual_repo / 'config/train.json').read_text()),
                              local_cnn=json.loads((actual_repo / 'config/v22_local_cnn.json').read_text()),
                              learning_policy='same_donor_live_v1', original_inventory_sha256='1' * 64,
                              identities={'cases': {}}, donor_pool=[], raw_records=[],
                              source_identity=self.source_identity,
                              split={'inner_train': [f'case_{i}' for i in range(84)],
                                     'inner_val': [f'case_{i}' for i in range(84, 105)]}, records=[])
        for case in range(105):
            for position in range(128 + 6 + (case < 32)):
                self.inventory['records'].append(dict(id=f'{case}:{position}', case_id=f'case_{case}',
                                                      target=int(position >= 128)))
        self.native = self.roots['native']
        write(self.native / 'inventory/index.json', self.inventory)
        self.identity = dict(format='fixed_region_sage_training_v1', debug=False, activation_storage='retained',
                             graph_representation=self.inventory['local_cnn']['architecture'],
                             resident_budget_bytes=128 * 2**30, profile_policy='strict',
                             execution_pipeline=dict(mode='overlapped', device_cache_bytes=8 * 2**30,
                                 sage_workspace_bytes=64 * 2**20,
                                 checkpoints='every update; immutable packed CPU snapshot; one ordered writer; flush on pause/phase/return',
                                 gradient_check='one finite decision per update'),
                             source=self.source_identity, config=self.inventory['config'], base=self.inventory['base'],
                             cache_sha256=receipt.sha_file(self.native / 'inventory/index.json'),
                             local_cnn=self.inventory['local_cnn'], learning_policy=self.inventory['learning_policy'],
                             ranking={'normalization': 'epoch pair mean'}, epochs=40, workers=16,
                             candidates=[32], resource_limits={'cuda_bytes': 40 * 2**30, 'rss_bytes': 192 * 2**30},
                             precision='FP32', support_training=dict(policy='patient_episode_v1', patients=16,
                                 observations='all eligible rows of selected patients',
                                 exclusion='query patient on both recipient and donor sides',
                                 selection='class-covered least-used patients; seeded epoch tie order',
                                 refresh='one selection and frozen cluster plan per query-patient episode',
                                 evaluation='full eligible inner-train support, once per query patient'))
        self.settings = dict(workers=16, cuda_gib=40, rss_gib=192, resident_gib=128,
                             device_cache_gib=8, batch_candidates=[32], support_patients=16, debug=False)
        self.manifest = dict(format='local_cnn_experiment_v1', request={
            'source': self.source_identity, 'local_cnn': self.inventory['local_cnn'],
            'original_inventory_sha256': '1' * 64, 'settings': self.settings},
            attempts=[{'output': 'attempts/0001', 'resume_from': None}])
        write(self.native / 'experiment.json', self.manifest)
        self.checkpoint = self.native / 'attempts/0001/checkpoint_latest.pt'
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'UNIT metadata reader stand-in, not model weights')
        self.execution = dict(self.identity, physical_batch=32, effective_batch=32, accumulation=1,
                              optimization_steps=533 * 40, train_samples=11279, val_samples=2823)
        write(self.checkpoint.parent / 'execution_contract.json', self.execution)
        self.schedule = dict(policy='same_donor_live_v1', unique_observations=11279, physical_batch=32,
                             optimization_steps=533, actual_batch_sizes=[32] * 533)
        write(self.checkpoint.parent / 'learning_schedule.json', self.schedule)
        self.state = dict(epoch=22, step=11872, phase='optimization', next_batch=146, batch=32,
                          memory={'record_ids': ['UNIT metadata memory']}, memory_parts=[], memory_done=0,
                          plan=None, last_group=None, best=None, selected_epoch=None,
                          initial_validation=None, validation_history=[])
        for name in ('baseline', 'A', 'B'):
            formats = dict(baseline='v1_bounded_roi_full_learning_v1', A='hiercp_v14_m10_half_A_learning_v1',
                           B='hiercp_v14_m10_half_B_learning_v1')
            helper_key = 'helpers' if name == 'baseline' else 'implementation'
            write(self.roots[name] / 'manifest.json', dict(format=formats[name], stages={},
                  source=str(self.source), source_hashes=self.source_identity['runtime'],
                  **{helper_key: {'l0_regions/donor_learning.py': self.digest}}))
        self.patch_root = patch.object(receipt, 'ROOT', self.source)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def reader(self, path):
        return dict(path=str(path), sha256=receipt.sha_file(path), bytes=path.stat().st_size,
                    payload=dict(format='fixed_region_sage_training_v1', identity=self.identity,
                                 state=self.state, optimizer={}, rng={}, content_sha256='UNIT only; content hash is not recomputed',
                                 model={'weight': {'kind': 'tensor_metadata_only', 'shape': [128, 128],
                                                   'values_exported': False}}))

    def export(self, **kwargs):
        return receipt.export_receipt(self.native, self.roots['baseline'], self.roots['A'], self.roots['B'],
                                      self.root / 'receipt_out', checkpoint_reader=self.reader, **kwargs)

    def test_bound_metadata_preserves_entire_inventory_and_never_claims_neural_quality(self):
        result, bundle = self.export()
        self.assertTrue(result['exact_native_recipe_bound'])
        self.assertEqual(result['inventory']['records'], 14102)
        self.assertEqual(len(result['inventory']['per_case']), 105)
        self.assertTrue(all(row['U'] == 128 for row in result['inventory']['per_case']))
        self.assertFalse(result['training_started'])
        self.assertFalse(result['quality_verified'])
        self.assertFalse(result['source_execution_validated'])
        self.assertEqual(receipt.validate_receipt(bundle), result)
        self.assertEqual(receipt.validate_receipt(bundle.parent), result)
        with ZipFile(bundle) as zipped:
            copied = json.loads(zipped.read('native/inventory/index.json'))
            self.assertEqual(copied, self.inventory)
            self.assertFalse(any(name.endswith('.pt') or name.endswith('.nii.gz') for name in zipped.namelist()))

    def test_all_epoch_invocations_and_case_scores_are_preserved(self):
        results = self.roots['B'] / 'results/half_B'
        for name in ('epoch_telemetry_01.jsonl', 'epoch_telemetry_02.jsonl',
                     'case_scores_01.jsonl', 'checkpoint_best.pt.preflight.json'):
            path = results / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n', encoding='utf8')
        _, bundle = self.export()
        with ZipFile(bundle) as zipped:
            for name in ('epoch_telemetry_01.jsonl', 'epoch_telemetry_02.jsonl',
                         'case_scores_01.jsonl', 'checkpoint_best.pt.preflight.json'):
                self.assertIn('comparisons/half_B/results/half_B/' + name, zipped.namelist())

    def test_no_newest_checkpoint_search_or_unbound_foreign_attempt(self):
        foreign = self.native / 'attempts/9999/checkpoint_latest.pt'
        foreign.parent.mkdir()
        foreign.write_bytes(b'newer foreign checkpoint')
        self.assertEqual(receipt.select_checkpoint(self.native, self.manifest), self.checkpoint)

    def test_missing_current_saved_checkpoint_does_not_revert_to_older(self):
        self.manifest['attempts'].append({'output': 'attempts/0002', 'resume_from': 'attempts/0001/checkpoint_latest.pt'})
        write(self.native / 'attempts/0002/training_complete.json', {})
        with self.assertRaises(FileNotFoundError):
            receipt.select_checkpoint(self.native, self.manifest)

    def test_explicit_resume_from_is_used_for_unsaved_bound_attempt(self):
        self.manifest['attempts'].append({'output': 'attempts/0002', 'resume_from': 'attempts/0001/checkpoint_latest.pt'})
        self.assertEqual(receipt.select_checkpoint(self.native, self.manifest), self.checkpoint)

    def test_unsafe_manifest_path_is_rejected(self):
        for value in ('../foreign', '/absolute', 'C:/drive', 'attempts\\bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                receipt.select_checkpoint(self.native, dict(self.manifest, attempts=[{'output': value}]))

    def test_missing_native_evidence_is_a_diagnostic_receipt(self):
        self.checkpoint.unlink()
        result, bundle = self.export()
        self.assertFalse(result['exact_native_recipe_bound'])
        self.assertIn('native.checkpoint', result['required_fields_missing'])
        self.assertIsNone(result['native_checkpoint'])
        self.assertFalse(receipt.validate_receipt(bundle)['production_ready'])

    def test_mismatched_source_is_not_silently_admitted(self):
        (self.source / 'l0_regions/donor_learning.py').write_text('# altered UNIT source\n', encoding='utf8')
        result, _ = self.export()
        self.assertFalse(result['exact_native_recipe_bound'])
        self.assertIn('native.source_root_matching_checkpoint', result['required_fields_missing'])
        self.assertEqual(len(result['sources'][0]['mismatches']), 1)

    def test_missing_core_and_runtime_inventory_cannot_be_marked_matched(self):
        for group in ('core', 'runtime'):
            identity = {name: dict(files) for name, files in self.source_identity.items()}
            key = next(iter(identity[group]))
            del identity[group][key]
            with self.subTest(group=group):
                collector = receipt.Collector()
                checks = receipt.collect_sources(collector, identity, [self.source])
                self.assertFalse(checks[0]['matches_checkpoint_source'])
                self.assertIn(group + '/' + key, checks[0]['required_source_identity_files_missing'])
        identity = {'runtime': {'l0_regions/donor_learning.py': self.digest}}
        checks = receipt.collect_sources(receipt.Collector(), identity, [self.source])
        self.assertFalse(checks[0]['matches_checkpoint_source'])
        self.assertIn('core', checks[0]['required_source_identity_files_missing'])

    def test_native_inventory_required_metadata_cannot_be_synthesized(self):
        for name in ('complete', 'split', 'identities', 'donor_pool', 'raw_records', 'source_identity'):
            value = dict(self.inventory)
            del value[name]
            with self.subTest(field=name):
                self.assertIn('native.inventory.' + name, receipt.native_inventory_contract(value))
        self.assertIn('native.inventory.format_contract', receipt.native_inventory_contract(dict(self.inventory, format='wrong')))
        self.assertIn('native.inventory.complete_contract', receipt.native_inventory_contract(dict(self.inventory, complete=False)))

    def test_comparison_manifest_missing_declared_helpers_is_unknown(self):
        for role, result, key in (('baseline', 'v1.0', 'helpers'), ('A', 'half_A', 'implementation'), ('B', 'half_B', 'implementation')):
            path = self.roots[role] / 'manifest.json'
            value = json.loads(path.read_text())
            del value[key]
            write(path, value)
            with self.subTest(role=role):
                collected = receipt.collect_comparison(receipt.Collector(), role, self.roots[role], result, [self.source])
                self.assertIn(key, collected['required_source_inventories_missing'])

    def test_inventory_binding_mismatch_is_reported_without_relabeling(self):
        self.identity['cache_sha256'] = '0' * 64
        result, _ = self.export()
        self.assertIn('binding.checkpoint_inventory_sha256_matches', result['required_fields_missing'])
        self.assertEqual(result['inventory']['observed_positive'], 662)

    def test_output_overwrite_and_nested_input_are_rejected(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()
        for output in (self.native / 'output', self.root):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                receipt.export_receipt(self.native, self.roots['baseline'], self.roots['A'], self.roots['B'],
                                       output, checkpoint_reader=self.reader)

    def test_manifest_rejects_modified_or_unlisted_bytes(self):
        _, bundle = self.export()
        directory = bundle.parent
        path = directory / 'native/inventory/index.json'
        path.write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            receipt.validate_receipt(directory)
        path.write_text(json.dumps(self.inventory, sort_keys=True), encoding='utf8')
        (directory / 'extra.json').write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            receipt.validate_receipt(directory)

    def test_sha_resealed_quality_promotion_is_still_rejected(self):
        _, bundle = self.export()
        directory = bundle.parent
        report = json.loads((directory / 'receipt.json').read_text())
        report['quality_verified'] = True
        raw = json.dumps(report).encode('utf8')
        (directory / 'receipt.json').write_bytes(raw)
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['files']['receipt.json'] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        write(directory / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'quality_verified'):
            receipt.validate_receipt(directory)

    def test_duplicate_and_traversal_zip_members_are_rejected(self):
        for name, members in (('duplicate', ['receipt.json', 'receipt.json']), ('traversal', ['../escape.json'])):
            path = self.root / (name + '.zip')
            with ZipFile(path, 'w') as zipped:
                for member in members:
                    zipped.writestr(member, '{}')
            with self.subTest(name=name), self.assertRaises(ValueError):
                receipt.validate_receipt(path)

    def test_original_input_bytes_are_preserved(self):
        before = {path: receipt.sha_file(path) for root in self.roots.values() for path in root.rglob('*') if path.is_file()}
        self.export()
        self.assertEqual(before, {path: receipt.sha_file(path) for path in before})

    def test_native_checkpoint_changed_after_metadata_read_is_unknown(self):
        def changing_reader(path):
            value = self.reader(path)
            path.write_bytes(b'UNIT simulated atomic checkpoint update')
            return value
        result, _ = receipt.export_receipt(self.native, self.roots['baseline'], self.roots['A'], self.roots['B'],
                                          self.root / 'receipt_out', checkpoint_reader=changing_reader)
        self.assertFalse(result['exact_native_recipe_bound'])
        self.assertFalse(result['checkpoint_artifact_checks'][0]['unchanged'])
        self.assertIn('checkpoint_changed_before_receipt_seal.native.checkpoint_latest.pt', result['required_fields_missing'])

    def test_every_wrapper_setting_mismatch_is_not_exact_bound(self):
        changed = dict(workers=8, cuda_gib=24, rss_gib=64, resident_gib=24, device_cache_gib=0,
                       batch_candidates=[48], support_patients=8, debug=True)
        for name, value in changed.items():
            with self.subTest(setting=name):
                current = dict(self.manifest, request=dict(self.manifest['request'], settings=dict(self.settings, **{name: value})))
                checks, missing = receipt.native_execution_bindings(current, self.inventory, self.reader(self.checkpoint)['payload'],
                                                                   self.execution, self.schedule)
                self.assertFalse(checks['request_' + name + '_matches_checkpoint'])
                self.assertIn('binding.request_' + name + '_matches_checkpoint', missing)

    def test_missing_settings_and_saved_state_fields_are_not_defaulted(self):
        payload = self.reader(self.checkpoint)['payload']
        for name in ('batch', 'epoch', 'step', 'phase', 'next_batch'):
            state = dict(self.state)
            del state[name]
            with self.subTest(state=name):
                _, missing = receipt.native_execution_bindings(self.manifest, self.inventory, dict(payload, state=state),
                                                              self.execution, self.schedule)
                self.assertIn('native.checkpoint.state.' + name, missing)
        for name in ('activation_storage', 'resident_budget_bytes', 'debug', 'execution_pipeline'):
            identity = dict(self.identity)
            del identity[name]
            with self.subTest(identity=name):
                _, missing = receipt.native_execution_bindings(self.manifest, self.inventory, dict(payload, identity=identity),
                                                              self.execution, self.schedule)
                self.assertIn('native.checkpoint.identity.' + name, missing)
        current = dict(self.manifest, request={k:v for k,v in self.manifest['request'].items() if k != 'settings'})
        _, missing = receipt.native_execution_bindings(current, self.inventory, payload, self.execution, self.schedule)
        self.assertIn('native.request.settings', missing)

    def test_physical_batch_and_saved_training_cursor_mismatch_rejected(self):
        payload = self.reader(self.checkpoint)['payload']
        for state in (dict(self.state, batch=48), dict(self.state, epoch=21), dict(self.state, step=11873),
                      dict(self.state, next_batch=147), dict(self.state, phase='unknown')):
            with self.subTest(state=state):
                _, missing = receipt.native_execution_bindings(self.manifest, self.inventory, dict(payload, state=state),
                                                              self.execution, self.schedule)
                self.assertTrue(missing)
        self.state['batch'] = 48
        result, _ = self.export()
        self.assertFalse(result['exact_native_recipe_bound'])
        self.assertIn('binding.resolved_batch_is_declared_candidate', result['required_fields_missing'])

    def test_execution_and_schedule_resolved_batch_mismatch_rejected(self):
        payload = self.reader(self.checkpoint)['payload']
        for name in ('physical_batch', 'effective_batch', 'accumulation', 'optimization_steps', 'train_samples'):
            changed = dict(self.execution, **{name: self.execution[name] + 1})
            with self.subTest(execution=name):
                _, missing = receipt.native_execution_bindings(self.manifest, self.inventory, payload, changed, self.schedule)
                self.assertTrue(missing)
        _, missing = receipt.native_execution_bindings(self.manifest, self.inventory, payload, self.execution,
                                                      dict(self.schedule, physical_batch=48))
        self.assertIn('binding.resolved_batch_matches_schedule', missing)


class FakeCheckpointMetadataUnitTests(unittest.TestCase):
    def test_real_torch_serialization_exports_shapes_without_tensor_values(self):
        import torch
        with tempfile.TemporaryDirectory(prefix='fake_checkpoint_metadata_UNIT_') as folder:
            path = Path(folder) / 'own_UNIT_checkpoint.pt'
            torch.save(dict(identity={'local_cnn': {'margin_mm': 10}}, state={'step': 11872},
                            model={'weight': torch.full((32, 128), 0.31415926535)},
                            optimizer={'state': {0: {'exp_avg': torch.ones(32, 128)}}}), path)
            result = receipt.checkpoint_metadata(path)
            weight = result['payload']['model']['weight']
            self.assertEqual(weight['shape'], [32, 128])
            self.assertEqual(weight['numel'], 4096)
            self.assertFalse(weight['values_exported'])
            self.assertEqual(result['payload']['state']['step'], 11872)
            self.assertNotIn('0.314159', json.dumps(result))
            self.assertEqual(result['sha256'], receipt.sha_file(path))


if __name__ == '__main__':
    unittest.main()
