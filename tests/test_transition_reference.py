"""UNIT metadata contracts only; synthetic receipts are not research results."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import native_transition_receipt as reader
from hiercp_v1x.transition_reference import ReferenceContractError, validate_reference_metadata


def unit_fixture():
    """Explicitly synthetic UNIT endpoint with actual native field names."""
    source = {group: {name: 'a' * 64 for name in sorted(names)}
              for group, names in reader.required_source_files().items()}
    cfg = dict(gnn_epochs=40, seed=42, candidate_count=128, task_layers=2, alignment_layers=2)
    base = dict(training=dict(lr=0.0001, weight_decay=0.0001, fused_optimizer=True))
    cnn = dict(architecture='paired_native_local_cnn_v1', margin_mm=10.0,
               channels=[12, 24, 32], convolutions=[2, 3, 3], hidden_dim=128,
               learning_policy='same_donor_live_v1')
    inventory = dict(format='native_local_cnn_inventory_v1', complete=True, debug=False,
                     config=cfg, base=base, local_cnn=cnn, learning_policy='same_donor_live_v1',
                     original_inventory_sha256='b' * 64)
    identity = dict(format='fixed_region_sage_training_v1', debug=False, cache_sha256='c' * 64,
                    source=source, epochs=40, config=cfg, base=base, precision='FP32', workers=16,
                    candidates=[32], ranking=dict(temperature=0.2), resident_budget_bytes=128 * 2**30,
                    profile_policy='strict', activation_storage='retained',
                    resource_limits=dict(cuda_bytes=40 * 2**30, rss_bytes=192 * 2**30),
                    local_cnn=cnn, graph_representation='paired_native_local_cnn_v1',
                    learning_policy='same_donor_live_v1', support_training=dict(policy='patient_episode_v1', patients=16),
                    execution_pipeline=dict(mode='overlapped', device_cache_bytes=8 * 2**30,
                                            sage_workspace_bytes=64 * 2**20,
                                            checkpoints='UNIT recorded publication contract', gradient_check='UNIT finite decision'))
    state = dict(batch=32, epoch=39, step=21320, phase='refresh_memory', next_batch=533,
                 memory=None, memory_parts=[], memory_done=0, plan=None, last_group=None,
                 best=None, selected_epoch=None, initial_validation={}, validation_history=[])
    payload = dict(format='fixed_region_sage_training_v1', identity=identity, state=state,
                   model={}, optimizer={}, rng={}, content_sha256='d' * 64)
    receipt = dict(path='UNIT/checkpoint_latest.pt', sha256='e' * 64, bytes=1024,
                   loading='FakeTensorMode; CPU metadata only; no neural execution; mmap=False',
                   tensor_values_exported=False, payload=payload)
    settings = dict(workers=16, cuda_gib=40, rss_gib=192, resident_gib=128, device_cache_gib=8,
                    batch_candidates=[32], support_patients=16, debug=False)
    manifest = dict(format='local_cnn_experiment_v1', request=dict(source=deepcopy(source), local_cnn=deepcopy(cnn),
                    original_inventory_sha256='b' * 64, settings=settings), attempts=[dict(output='attempts/0001')])
    execution = deepcopy(identity)
    execution.update(train_samples=11279, val_samples=2823, physical_batch=32, effective_batch=32,
                     accumulation=1, optimization_steps=21320)
    schedule = dict(policy='same_donor_live_v1', unique_observations=11279, physical_batch=32,
                    actual_batch_sizes=[32] * 533, optimization_steps=533)
    # Keep the four actual JSON structures independent so tampering one endpoint
    # cannot accidentally change the expected recipe in another fixture mapping.
    return {name: deepcopy(value) for name, value in dict(manifest=manifest, inventory=inventory, receipt=receipt,
                                                         execution=execution, schedule=schedule).items()}


class ReferenceMetadataUnit(unittest.TestCase):
    def setUp(self):
        self.fixture = unit_fixture()

    def admit(self):
        return validate_reference_metadata(**self.fixture, inventory_sha256='c' * 64)

    def identity(self):
        return self.fixture['receipt']['payload']['identity']

    def state(self):
        return self.fixture['receipt']['payload']['state']

    def test_actual_server_cursor_is_partial_after40_optimizer_epochs(self):
        receipt = self.admit()
        self.assertTrue(receipt['admitted'])
        self.assertEqual(receipt['saved_cursor'], dict(epoch=39, step=21320, phase='refresh_memory', next_batch=533, batch=32))
        self.assertEqual(receipt['optimizer_epochs_completed'], 40)
        self.assertEqual(receipt['validated_epochs_completed'], 39)
        self.assertFalse(receipt['native_training_complete'])
        self.assertFalse(receipt['weights_transferred'])
        self.assertFalse(receipt['neural_execution'])
        self.assertFalse(receipt['checkpoint_content_hash_recomputed'])
        self.assertNotIn('model', receipt)
        json.dumps(receipt, allow_nan=False)

    def test_saved_complete_phase_is_distinct_from_final_memory(self):
        for phase, complete in (('final_memory', False), ('complete', True)):
            with self.subTest(phase=phase):
                self.state().update(epoch=40, next_batch=0, step=21320, phase=phase)
                result = self.admit()
                self.assertEqual(result['native_training_complete'], complete)
                self.assertEqual(result['validated_epochs_completed'], 40)

    def test_recipe_binding_rejects_each_changed_checkpoint_field(self):
        for key in ('local_cnn', 'config', 'base', 'learning_policy', 'cache_sha256'):
            with self.subTest(key=key):
                self.fixture = unit_fixture()
                value = self.identity()[key]
                self.identity()[key] = dict(value, UNIT_tamper=True) if isinstance(value, dict) else 'tampered'
                with self.assertRaises(ReferenceContractError) as caught:
                    self.admit()
                self.assertTrue(any(key in name or 'inventory_sha256' in name for name in caught.exception.failed_bindings))

    def test_missing_checkpoint_recipe_is_never_invented(self):
        for key in ('cache_sha256', 'config', 'base', 'local_cnn', 'source', 'epochs', 'learning_policy'):
            with self.subTest(key=key):
                self.fixture = unit_fixture()
                del self.identity()[key]
                with self.assertRaises(ReferenceContractError) as caught:
                    self.admit()
                self.assertIn('checkpoint.identity.' + key, caught.exception.required_fields_missing)

    def test_request_source_must_equal_saved_checkpoint_source(self):
        name = next(iter(self.fixture['manifest']['request']['source']['core']))
        self.fixture['manifest']['request']['source']['core'][name] = 'f' * 64
        with self.assertRaises(ReferenceContractError) as caught:
            self.admit()
        self.assertIn('request_source_matches_checkpoint', caught.exception.failed_bindings)

    def test_request_inventory_and_localcnn_must_be_bound(self):
        for key in ('original_inventory_sha256', 'local_cnn'):
            with self.subTest(key=key):
                self.fixture = unit_fixture()
                self.fixture['manifest']['request'][key] = 'UNIT wrong'
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_source_omission_from_both_endpoints_is_explicitly_missing(self):
        name = next(iter(self.identity()['source']['runtime']))
        del self.identity()['source']['runtime'][name]
        del self.fixture['manifest']['request']['source']['runtime'][name]
        with self.assertRaises(ReferenceContractError) as caught:
            self.admit()
        self.assertIn('checkpoint.identity.source.runtime.' + name, caught.exception.required_fields_missing)

    def test_unsafe_or_nonhex_source_is_rejected(self):
        for name, digest in (('../outside.py', 'a' * 64), ('UNIT.py', 'z' * 64)):
            with self.subTest(name=name):
                self.fixture = unit_fixture()
                self.identity()['source']['runtime'][name] = digest
                self.fixture['manifest']['request']['source'] = deepcopy(self.identity()['source'])
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_request_settings_each_match_actual_execution(self):
        settings = self.fixture['manifest']['request']['settings']
        for key in tuple(settings):
            with self.subTest(key=key):
                self.fixture = unit_fixture()
                actual = self.fixture['manifest']['request']['settings']
                actual[key] = [16] if key == 'batch_candidates' else True if key == 'debug' else actual[key] + 1
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_missing_setting_and_cursor_are_explicit(self):
        del self.fixture['manifest']['request']['settings']['device_cache_gib']
        del self.state()['next_batch']
        with self.assertRaises(ReferenceContractError) as caught:
            self.admit()
        self.assertIn('native.request.settings.device_cache_gib', caught.exception.required_fields_missing)
        self.assertIn('checkpoint.state.next_batch', caught.exception.required_fields_missing)

    def test_no_debug_or_short_training_reference(self):
        for field in ('debug', 'epochs'):
            with self.subTest(field=field):
                self.fixture = unit_fixture()
                if field == 'debug':
                    self.fixture['inventory']['debug'] = True
                    self.identity()['debug'] = True
                else:
                    self.fixture['inventory']['config']['gnn_epochs'] = 1
                    self.identity()['epochs'] = 1
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_wrong_format_or_learning_policy_is_rejected(self):
        for where, key, value in (('manifest', 'format', 'unknown'), ('inventory', 'format', 'old_graph_inventory'),
                                  ('inventory', 'complete', False), ('schedule', 'policy', 'old_policy')):
            with self.subTest(where=where, key=key):
                self.fixture = unit_fixture()
                self.fixture[where][key] = value
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_epoch_step_and_phase_tampering_rejected(self):
        for update in (dict(step=21319), dict(next_batch=532), dict(phase='unknown'),
                       dict(epoch=True), dict(epoch=40, phase='optimization', next_batch=0, step=21320),
                       dict(epoch=39, phase='complete'), dict(epoch=-1)):
            with self.subTest(update=update):
                self.fixture = unit_fixture()
                self.state().update(update)
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_initial_state_needs_zero_updates(self):
        self.state().update(epoch=0, phase='initial_validation', step=0, next_batch=0)
        result = self.admit()
        self.assertEqual(result['optimizer_epochs_completed'], 0)
        self.state()['step'] = 1
        with self.assertRaises(ReferenceContractError):
            self.admit()

    def test_batch_schedule_and_accumulation_are_not_guessed(self):
        for update in (dict(physical_batch=16), dict(effective_batch=64), dict(accumulation=2),
                       dict(optimization_steps=21321), dict(accumulation=True)):
            with self.subTest(update=update):
                self.fixture = unit_fixture()
                self.fixture['execution'].update(update)
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_actual_schedule_sizes_and_source_coverage_required(self):
        self.fixture['schedule']['actual_batch_sizes'] = [32] * 532
        with self.assertRaises(ReferenceContractError):
            self.admit()
        self.fixture = unit_fixture()
        self.fixture['schedule']['unique_observations'] = 1
        with self.assertRaises(ReferenceContractError):
            self.admit()

    def test_tensor_value_reader_is_never_admitted(self):
        for key, value in (('tensor_values_exported', True), ('loading', 'ordinary tensor load')):
            with self.subTest(key=key):
                self.fixture = unit_fixture()
                self.fixture['receipt'][key] = value
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_invalid_file_identity_is_rejected(self):
        for key, value in (('sha256', 'not SHA'), ('bytes', 0), ('bytes', True), ('path', '')):
            with self.subTest(key=key):
                self.fixture = unit_fixture()
                self.fixture['receipt'][key] = value
                with self.assertRaises(ReferenceContractError):
                    self.admit()

    def test_reject_bool_disguised_as_numeric_recipe_or_execution(self):
        # Python equality considers True==1. Canonical recipe/identity checks must
        # reject that substitution even when the rest of execution is identical.
        self.identity()['config']['UNIT_integer'] = 1
        self.fixture['inventory']['config']['UNIT_integer'] = True
        with self.assertRaises(ReferenceContractError):
            self.admit()
        self.fixture = unit_fixture()
        self.fixture['execution']['workers'] = 16.0
        with self.assertRaises(ReferenceContractError):
            self.admit()

    def test_admission_is_pure_and_has_no_weight_payload(self):
        before = deepcopy(self.fixture)
        result = self.admit()
        self.assertEqual(self.fixture, before)
        self.assertEqual(result['required_fields_missing'], [])
        self.assertNotIn('optimizer', result)
        self.assertNotIn('rng', result)
        self.assertNotIn('payload', result)


class ActualCpuMetadataReaderUnit(unittest.TestCase):
    def test_real_fake_tensor_load_exports_shapes_without_values(self):
        # CPU serialization and FakeTensorMode exercise the real reader without
        # creating filesystem fixtures, CUDA contexts or a neural forward pass.
        import torch
        stream = io.BytesIO()
        torch.save(dict(format='UNIT_ONLY', model={'UNIT_tensor': torch.arange(4).reshape(2, 2)}), stream)
        size = stream.tell()
        original_load = torch.load

        def load_cpu_metadata(_path, **kwargs):
            self.assertEqual(kwargs, dict(map_location='cpu', weights_only=False, mmap=False))
            stream.seek(0)
            return original_load(stream, **kwargs)

        fake_path = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=size, st_mtime_ns=1), resolve=lambda: 'UNIT.pt')
        with patch.object(reader, 'Path', return_value=fake_path), patch.object(reader, 'sha_file', return_value='a' * 64), \
                patch.object(torch, 'load', side_effect=load_cpu_metadata):
            receipt = reader.checkpoint_metadata('UNIT.pt')
        tensor = receipt['payload']['model']['UNIT_tensor']
        self.assertEqual(tensor['kind'], 'tensor_metadata_only')
        self.assertEqual(tensor['shape'], [2, 2])
        self.assertEqual(tensor['numel'], 4)
        self.assertFalse(tensor['values_exported'])
        self.assertFalse(receipt['tensor_values_exported'])
        self.assertNotIn('data', tensor)


if __name__ == '__main__':
    unittest.main()
