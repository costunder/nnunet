"""CPU UNIT admission checks for the intermediate comparison evaluation loader.

Synthetic UNIT tensors exercise identity and saved-state validation only.
These are not CT data, model quality results, a CUDA load, or training.
"""
from __future__ import annotations

import copy
import json
import unittest

import torch

from hiercp_v1x import comparison_native_checkpoint as loader
from hiercp_v1x import u_bridge_training as engine
from hiercp_v1x.comparison_training import arm_policy, FORMAT as V19_FORMAT
from hiercp_v1x.scope_probe_support import state_digest
from hiercp_v1x.u_bridge_data import UBridgeData, _centers


def json_roundtrip(value):
    """Use the same lossy tuple-to-array serialization as _write_new."""
    return json.loads(json.dumps(value, allow_nan=False))


def original_provider_examples():
    """CPU UNIT identities with the actual UBridgeData constructor's types."""
    provider = UBridgeData.__new__(UBridgeData)
    provider._examples = []
    for index, partition in enumerate(('train', 'train', 'val')):
        centers = _centers([(0, 1, 2)] + [(i + 1, 1, 2) for i in range(7)], 8, 'UNIT selected')
        native = _centers([(i + 1, 10, 20) for i in range(128)], 128, 'UNIT native U')
        provider._examples.append(dict(index=index, id=f'UNITcase{index}:0',
            case_id=f'UNITcase{index}', sample_index=0, source_component=1,
            split=partition, partition=partition, positive_center=centers[0],
            selected_centers=centers[1:], native_centers=native,
            original_sample_sha256=f'UNITsample{index}'))
    return engine._examples(provider, 'train'), engine._examples(provider, 'val')


def fixture(arm):
    family = 'u_bridge' if arm in ('selected', 'native') else 'comparison'
    model = {'UNIT.weight': torch.arange(4, dtype=torch.float32)}
    initial = dict(model=model, model_sha256=state_digest(model), contract_sha256='UNITmanifest')
    model_hash = engine.digest(model)
    calibration = dict(physical_batch=2, contract_sha256='UNITmanifest', reports={arm: dict(
        arm=arm, initial_state_sha256=model_hash, original_model_and_RNG_preserved=True,
        reports=[dict(accepted=True, physical_batch=2)])})
    train, validation = original_provider_examples()
    manifest = json_roundtrip(dict(sha256='UNITmanifest', config={'UNIT': True}, workers=2,
        validation_local_chunk=7, samples=train + validation))
    arm_calibration = copy.deepcopy(calibration['reports'][arm])
    arm_calibration['selected_physical_batch'] = 2
    config = dict(UNIT=True, u_bridge_runtime=dict(batch_calibration=arm_calibration,
        validation_local_chunk_size=7, expected_parameters=10434532))
    identity = dict(contract_sha256='UNITmanifest', initial_neural_sha256=initial['model_sha256'],
                    initial_state_sha256=model_hash)
    policy, training_format = None, engine.FORMAT
    if family == 'comparison':
        policy, training_format = arm_policy(arm), V19_FORMAT
        config['comparison_policy'] = copy.deepcopy(policy)
        identity['comparison_policy'] = copy.deepcopy(policy)
    binding = dict(format=training_format, identity=identity, config=config, arm=arm, debug=False,
        epochs=40, physical_batch=2, workers=2, train_examples=train,
        val_examples=validation, initial_state_sha256=model_hash)
    # Original run_arm hashes first, then _write_new JSON-serializes the binding.
    ownership = json_roundtrip(dict(binding=binding, identity_sha256=engine.digest(binding)))
    report = dict(metrics=dict(mrr=.8, top1=.7, pair_loss=.2))
    state = dict(epoch=2, phase='training', position=0, updates=1, validation_position=0,
        best=dict(epoch=1, update=1, selection_key=[.8, .7, -.2]),
        history=[dict(epoch=1, update=1, validation129=report)])
    saved = dict(format=training_format, identity_sha256=ownership['identity_sha256'],
                 model=copy.deepcopy(model), state=state)
    if policy is not None:
        saved['comparison_policy'] = copy.deepcopy(policy)
    seal(saved)
    return dict(arm=arm, family=family, manifest=manifest, ownership=ownership,
                calibration=calibration, initial=initial, policy=policy,
                training_format=training_format, saved=saved, original_binding=binding)


def seal(saved):
    saved.pop('content_sha256', None)
    saved['content_sha256'] = engine.digest(saved)


def validate_binding(value):
    return loader._validate_binding(**{key: value[key] for key in
        ('arm', 'family', 'manifest', 'ownership', 'calibration', 'initial')})


def metadata(value, selection):
    return loader._checkpoint_metadata(value['saved'], selection=selection,
        **{key: value[key] for key in ('ownership', 'training_format', 'arm', 'policy')})


class ComparisonNativeCheckpointUnit(unittest.TestCase):
    def test_all_four_original_bindings_and_best_latest_metadata(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                original = value['original_binding']['train_examples'][0]
                persisted = value['ownership']['binding']['train_examples'][0]
                self.assertIs(type(original['positive_center']), tuple)
                for key in ('selected_centers', 'native_centers'):
                    self.assertIs(type(original[key]), tuple)
                    self.assertTrue(all(type(center) is tuple for center in original[key]))
                    self.assertIs(type(persisted[key]), list)
                    self.assertTrue(all(type(center) is list for center in persisted[key]))
                self.assertIs(type(persisted['positive_center']), list)
                self.assertNotEqual(engine.digest(value['ownership']['binding']),
                                    value['ownership']['identity_sha256'])
                before = engine.digest(value)
                self.assertEqual(validate_binding(value), (value['training_format'], value['policy']))
                for selection, epoch in (('best', 1), ('latest', 2)):
                    result = metadata(value, selection)
                    self.assertEqual(result['selected_epoch'], epoch)
                    self.assertEqual(result['comparison_arm'], arm)
                    self.assertEqual(result['completed_epochs'], 1)
                    self.assertFalse(result['full_training_complete'])
                    self.assertFalse(result['optimizer_imported'])
                    self.assertFalse(result['training_rng_restored'])
                    self.assertEqual(result['optimizer_updates'], 0)
                self.assertEqual(engine.digest(value), before, 'Read-only validation mutated evidence')

    def test_resigned_wrong_arm_binding_is_rejected(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                value['original_binding']['arm'] = next(x for x in loader.ARMS if x != arm)
                binding = value['original_binding']
                value['ownership'] = json_roundtrip(dict(binding=binding, identity_sha256=engine.digest(binding)))
                with self.assertRaisesRegex(ValueError, 'original complete comparison binding'):
                    validate_binding(value)

    def test_hash_of_json_binding_is_not_accepted_as_original_identity(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                value['ownership']['identity_sha256'] = engine.digest(value['ownership']['binding'])
                with self.assertRaisesRegex(ValueError, 'original complete comparison binding'):
                    validate_binding(value)

    def test_persisted_binding_content_corruption_is_rejected(self):
        for arm in loader.ARMS:
            for field in ('positive_center', 'selected_centers', 'native_centers'):
                with self.subTest(arm=arm, field=field):
                    value = fixture(arm)
                    coordinates = value['ownership']['binding']['train_examples'][0][field]
                    target = coordinates if field == 'positive_center' else coordinates[0]
                    target[0] += 1
                    with self.assertRaisesRegex(ValueError, 'original complete comparison binding'):
                        validate_binding(value)

    def test_missing_or_extra_persisted_binding_field_is_rejected(self):
        for arm in loader.ARMS:
            for missing in (False, True):
                with self.subTest(arm=arm, missing=missing):
                    value = fixture(arm)
                    binding = value['ownership']['binding']
                    if missing:
                        binding['train_examples'][0].pop('original_sample_sha256')
                    else:
                        binding['train_examples'][0]['extra'] = True
                    with self.assertRaisesRegex(ValueError, 'original complete comparison binding'):
                        validate_binding(value)

    def test_persisted_source_order_is_part_of_the_original_identity(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                value['ownership']['binding']['train_examples'].reverse()
                with self.assertRaisesRegex(ValueError, 'original complete comparison binding'):
                    validate_binding(value)

    def test_saved_tensor_corruption_is_rejected(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                value['saved']['model']['UNIT.weight'][0] = 99
                with self.assertRaisesRegex(ValueError, 'content/arm/policy identity'):
                    metadata(value, 'best')

    def test_resigned_stale_best_update_is_rejected_but_latest_cursor_is_explicit(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                value['saved']['state'].update(updates=2, position=1)
                seal(value['saved'])
                with self.assertRaisesRegex(ValueError, 'BEST file is not the model'):
                    metadata(value, 'best')
                result = metadata(value, 'latest')
                self.assertEqual(result['selected_epoch'], 2)
                self.assertEqual(result['cursor']['position'], 1)
                self.assertEqual(result['completed_epochs'], 1)
                self.assertIn('partial epoch', result['selected_epoch_interpretation'])

    def test_resigned_stale_best_selection_loses_to_history_is_rejected(self):
        for arm in loader.ARMS:
            with self.subTest(arm=arm):
                value = fixture(arm)
                state = value['saved']['state']
                state['history'].append(dict(epoch=2, update=2,
                    validation129=dict(metrics=dict(mrr=.9, top1=.8, pair_loss=.1))))
                state.update(epoch=3, updates=2)
                seal(value['saved'])
                with self.assertRaisesRegex(ValueError, 'BEST selection history differs'):
                    metadata(value, 'latest')


if __name__ == '__main__':
    unittest.main()
