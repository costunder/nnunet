"""CPU UNIT admission checks for the intermediate comparison evaluation loader.

Synthetic UNIT tensors exercise identity and saved-state validation only.
These are not CT data, model quality results, a CUDA load, or training.
"""
from __future__ import annotations

import copy
import unittest

import torch

from hiercp_v1x import comparison_native_checkpoint as loader
from hiercp_v1x import u_bridge_training as engine
from hiercp_v1x.comparison_training import arm_policy, FORMAT as V19_FORMAT
from hiercp_v1x.scope_probe_support import state_digest


def fixture(arm):
    family = 'u_bridge' if arm in ('selected', 'native') else 'comparison'
    model = {'UNIT.weight': torch.arange(4, dtype=torch.float32)}
    initial = dict(model=model, model_sha256=state_digest(model), contract_sha256='UNITmanifest')
    model_hash = engine.digest(model)
    calibration = dict(physical_batch=2, contract_sha256='UNITmanifest', reports={arm: dict(
        arm=arm, initial_state_sha256=model_hash, original_model_and_RNG_preserved=True,
        reports=[dict(accepted=True, physical_batch=2)])})
    manifest = dict(sha256='UNITmanifest', config={'UNIT': True}, workers=2,
        validation_local_chunk=7, samples=[dict(id='a', index=0, partition='train'),
                                          dict(id='b', index=1, partition='val')])
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
        epochs=40, physical_batch=2, workers=2, train_examples=manifest['samples'][:1],
        val_examples=manifest['samples'][1:], initial_state_sha256=model_hash)
    ownership = dict(binding=binding, identity_sha256=engine.digest(binding))
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
                training_format=training_format, saved=saved)


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
                value['ownership']['binding']['arm'] = next(x for x in loader.ARMS if x != arm)
                value['ownership']['identity_sha256'] = engine.digest(value['ownership']['binding'])
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
        value = fixture('selected')
        state = value['saved']['state']
        state['history'].append(dict(epoch=2, update=2,
            validation129=dict(metrics=dict(mrr=.9, top1=.8, pair_loss=.1))))
        state.update(epoch=3, updates=2)
        seal(value['saved'])
        with self.assertRaisesRegex(ValueError, 'BEST selection history differs'):
            metadata(value, 'latest')


if __name__ == '__main__':
    unittest.main()
