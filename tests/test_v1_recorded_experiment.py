"""Recorded-arm METADATA UNIT tests; no CT decode, CUDA or model execution.

The full 84/21/26 cohort consists only of empty, NIfTI-named filename fixtures.
They exercise inventory/split contracts and are never preparation inputs. The
pinned real source archive is checked byte for byte. Subprocess launching is
blocked, and no fixture is represented as a trained checkpoint or metric.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
from zipfile import ZipFile

from hiercp_v1x.contracts import (
    ContractError, LOCAL_SAMPLING_ROLES, STAGES, TARGET_CONTRACT, canonical_hash,
    verify_archive,
)
from hiercp_v1x.experiment import (
    EPOCH_RECORDING, EPOCH_RECORDING_HELPERS, ROOT, command_plan, digest,
    execute_stage, execution_reference, freeze_execution, initialize, load_suite,
    preparation_root, read, sampling_overlay,
)


class RecordedExperimentMetadataUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Default mkdir mode avoids Python 3.13's Windows tempfile ACL path.
        cls.owned = (ROOT / 'work' / ('v1_recorded_metadata_UNIT_' + uuid.uuid4().hex)).resolve()
        cls.owned.mkdir(parents=True, exist_ok=False)
        cls.addClassCleanup(cls._cleanup_owned)
        cls.medical = cls.owned / 'medical'
        cls.cohort = dict(
            seed=42,
            train=[f'UNIT_train_{i:03d}' for i in range(84)],
            val=[f'UNIT_val_{i:03d}' for i in range(21)],
            outer_validation_excluded=[f'UNIT_outer_{i:03d}' for i in range(26)],
        )
        cases = (cls.cohort['train'] + cls.cohort['val']
                 + cls.cohort['outer_validation_excluded'])
        for kind in ('image', 'labels'):
            directory = cls.medical / 'Data' / kind
            directory.mkdir(parents=True)
            for case in cases:
                suffix = '_0000.nii.gz' if kind == 'image' else '.nii.gz'
                (directory / (case + suffix)).write_bytes(b'')
        cls.split = cls.owned / 'split.json'
        cls.split.write_text(json.dumps(cls.cohort), encoding='utf-8')
        cls.profile = dict(zip(LOCAL_SAMPLING_ROLES, (64, 32, 96, 64, 96, 64)))
        cls.native = cls.owned / 'native_recorded'
        cls.nested = cls.owned / 'nested416_recorded'
        cls.unrecorded = cls.owned / 'native_unrecorded'
        cls.nm = initialize(cls.native, cls.medical, cls.split,
                            local_sampling='native', record_epochs=True)
        cls.sm = initialize(cls.nested, cls.medical, cls.split,
                            local_sampling='strict_nested',
                            sampling_profile=cls.profile,
                            reference_experiment=cls.native, record_epochs=True)
        cls.um = initialize(cls.unrecorded, cls.medical, cls.split,
                            local_sampling='native')
        # Calibration JSON is metadata only, not a hardware measurement claim.
        calibration = dict(
            format='hiercp_preflight_calibration_v2', selected_batch_size=32,
            selected_num_workers=8,
            resource_fingerprint={'UNIT_metadata_only_not_measured': True},
            identity=dict(seed=42, cache_dir=str(cls.native / 'shared/cache'),
                          checkpoint_path=str(cls.native / 'results/v1.0/checkpoint_best.pt')),
        )
        calibration_path = cls.native / 'results/v1.0/checkpoint_best.pt.preflight.json'
        calibration_path.parent.mkdir(parents=True)
        calibration_path.write_text(json.dumps(calibration), encoding='utf-8')
        cls.lock = freeze_execution(cls.native)

    @classmethod
    def _cleanup_owned(cls):
        # Check every resolved target, then remove only this test's own tree.
        if (cls.owned.parent != (ROOT / 'work').resolve()
                or not cls.owned.name.startswith('v1_recorded_metadata_UNIT_')
                or cls.owned.is_symlink()):
            raise AssertionError('Refusing cleanup outside the exact UNIT root')
        paths = sorted(cls.owned.rglob('*'), key=lambda p: len(p.parts), reverse=True)
        for path in paths:
            if path.is_symlink() or cls.owned not in path.resolve().parents:
                raise AssertionError('Refusing cleanup of a redirected UNIT path')
        for path in paths:
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        cls.owned.rmdir()

    def setUp(self):
        blocker = patch('hiercp_v1x.experiment.subprocess.Popen',
                        side_effect=AssertionError('METADATA UNIT must not launch training'))
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_recorded_arms_bind_same_policy_full_cohort_seed_and_exact_configuration(self):
        self.assertEqual(self.nm['epoch_recording'], EPOCH_RECORDING)
        self.assertEqual(self.sm['epoch_recording'], EPOCH_RECORDING)
        self.assertEqual(self.nm['split'], self.sm['split'])
        self.assertEqual(self.nm['split']['seed'], 42)
        for key, count in (('train', 84), ('val', 21), ('outer_validation_excluded', 26)):
            self.assertEqual(len(self.nm['split'][key]), count)
        self.assertEqual(self.sm['sampling_contract']['profile'], self.profile)
        self.assertEqual(sum(self.profile.values()), 416)
        self.assertEqual(self.nm['target_contract'], TARGET_CONTRACT)
        self.assertEqual(self.sm['target_contract'], TARGET_CONTRACT)
        proof = verify_archive(ROOT)
        for experiment in (self.native, self.nested, self.unrecorded):
            self.assertEqual(read(experiment / 'configs/v1.0.json'), proof['base_config'])
        self.assertEqual(proof['base_config']['training']['epochs'], 40)
        for manifest in (self.nm, self.sm):
            self.assertFalse(manifest['long_training_started'])
            self.assertFalse(manifest['historical_v1_accuracy_reproduced'])

    def test_native_original_archive_and_nested_only_sampling_append_hooks_preserved(self):
        proof = verify_archive(ROOT)
        self.assertEqual(proof['verified_files'], 202)
        with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
            for name in archive.namelist():
                with self.subTest(source=name):
                    original = archive.read(name)
                    self.assertEqual((self.native / 'source/v1.0' / name).read_bytes(), original)
                    self.assertEqual((self.nested / 'source/v1.0' / name).read_bytes(),
                                     original + sampling_overlay(name))
                    self.assertEqual((self.unrecorded / 'source/v1.0' / name).read_bytes(), original)
        for name in ('hiercp/pipeline.py', 'hiercp/loss.py'):
            self.assertEqual(sampling_overlay(name), b'')
        self.assertTrue(self.nm['stages']['v1.0']['source_original_exact'])
        self.assertFalse(self.sm['stages']['v1.0']['source_original_exact'])

    def test_recording_helpers_are_exact_snapshots_and_part_of_sampler_source_identity(self):
        self.assertEqual(EPOCH_RECORDING_HELPERS,
                         ('hiercp_v1x/epoch_telemetry.py', 'hiercp_v1x/telemetry_entry.py'))
        for experiment, manifest in ((self.native, self.nm), (self.nested, self.sm)):
            for stage in STAGES:
                for name in EPOCH_RECORDING_HELPERS:
                    with self.subTest(experiment=experiment.name, stage=stage, helper=name):
                        snapshot = experiment / 'source' / stage / name
                        self.assertEqual(snapshot.read_bytes(), (ROOT / name).read_bytes())
                        sha = digest(snapshot)
                        self.assertEqual(manifest['stages'][stage]['source_hashes'][name], sha)
                        self.assertEqual(manifest['sampling_contracts'][stage]
                                         ['source_identity']['files'][name], sha)
        self.assertNotIn('epoch_recording', self.um)
        for name in EPOCH_RECORDING_HELPERS:
            self.assertFalse((self.unrecorded / 'source/v1.0' / name).exists())

    def test_recorded_train_uses_telemetry_entry_and_request_generate_keeps_sampling_entry(self):
        for experiment in (self.native, self.nested):
            cwd, commands = command_plan(experiment, 'v1.0', 'train', python='UNIT_python')
            self.assertEqual(cwd, experiment / 'source/v1.0')
            self.assertEqual(len(commands), 1)
            train = commands[0]
            self.assertEqual(train[:4], ['UNIT_python', '-u', '-m', 'hiercp_v1x.telemetry_entry'])
            self.assertEqual(train[train.index('--contract') + 1], str(experiment / 'sampling/v1.0.json'))
            self.assertEqual(train[train.index('--request') + 1],
                             str(experiment / 'results/v1.0/telemetry_request.json'))
            self.assertEqual(train[train.index('--') + 1], 'train')
            self.assertEqual(train[train.index('--device') + 1], 'cuda')
            _, commands = command_plan(experiment, 'v1.0', 'generate', python='UNIT_python')
            generate = commands[0]
            self.assertEqual(generate[:4], ['UNIT_python', '-u', '-m', 'hiercp_v1x.sampling_entry'])
            self.assertNotIn('--request', generate)
            self.assertEqual(generate[generate.index('--') + 1], 'generate')
        _, commands = command_plan(self.unrecorded, 'v1.0', 'train', python='UNIT_python')
        self.assertEqual(commands[0][:4], ['UNIT_python', '-u', '-m', 'hiercp_v1x.sampling_entry'])
        self.assertNotIn('--request', commands[0])

    def test_shared_native_prototype_cache_prepare_commands_and_physical_worker_lock(self):
        self.assertEqual(preparation_root(self.nested), self.native / 'shared')
        self.assertEqual(execution_reference(self.nested), self.native)
        self.assertEqual(freeze_execution(self.nested), self.lock)
        self.assertEqual(self.lock['selected_batch_size'], 32)
        self.assertEqual(self.lock['selected_num_workers'], 8)
        self.assertFalse((self.nested / 'shared/execution_lock.json').exists())
        for experiment in (self.native, self.nested):
            cwd, commands = command_plan(experiment, 'v1.0', 'prepare', python='UNIT_python')
            self.assertEqual(cwd, self.native / 'source/v1.0')
            self.assertEqual(commands[0][:4], ['UNIT_python', '-u', '-m', 'hiercp.pipeline'])
            self.assertEqual([argv[4] for argv in commands], ['prepare-prototypes', 'prepare'])
            self.assertEqual(commands[1][commands[1].index('--prototype-bank') + 1],
                             str(self.native / 'shared/prototype_bank.pt'))
            for target in ('train', 'generate'):
                _, commands = command_plan(experiment, 'v1.0', target, python='UNIT_python')
                argv = commands[0]
                self.assertEqual(argv[argv.index('--prototype-bank') + 1],
                                 str(self.native / 'shared/prototype_bank.pt'))
                if target == 'train':
                    self.assertEqual(argv[argv.index('--cache-dir') + 1], str(self.native / 'shared/cache'))
            self.assertFalse((experiment / 'results/v1.0/checkpoint_best.pt').exists())
        self.assertFalse((self.nested / 'shared/cache').exists())
        self.assertFalse((self.native / 'shared/prototype_bank.pt').exists())

    def test_epoch_request_is_native_bound_and_created_before_blocked_subprocess(self):
        for experiment, mode in ((self.native, 'native'), (self.nested, 'strict_nested')):
            with self.subTest(mode=mode), \
                    patch('hiercp_v1x.experiment.bind_launch') as binding, \
                    patch('hiercp_v1x.experiment.subprocess.Popen',
                          side_effect=RuntimeError('UNIT blocked before any training process')) as popen, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'UNIT blocked before any training process'):
                    execute_stage(experiment, 'v1.0', 'train')
                binding.assert_called_once()
                popen.assert_called_once()
                self.assertEqual(popen.call_args.args[0][3], 'hiercp_v1x.telemetry_entry')
            request = read(experiment / 'results/v1.0/telemetry_request.json')
            self.assertEqual(request, dict(
                format='hiercp_v1_epoch_telemetry_request_v1',
                output_dir=str(experiment / 'results/v1.0'), stage='v1.0', mode=mode,
                native_reference=str(self.native),
                baseline_manifest_sha256=self.nm['manifest_sha256'],
                baseline_sampling_contract_sha256=self.nm['sampling_contracts']['v1.0']['contract_sha256'],
            ))
            self.assertFalse((experiment / 'results/v1.0/run.lock').exists())
            self.assertFalse((experiment / 'results/v1.0/checkpoint_best.pt').exists())
        resolved = read(self.nested / 'results/v1.0/resolved_config.json')
        self.assertEqual(resolved['training']['batch_size'], self.lock['selected_batch_size'])
        self.assertEqual(resolved['training']['num_workers'], self.lock['selected_num_workers'])
        self.assertEqual(resolved['training']['epochs'], 40)

    def test_recording_requires_explicit_sampling_boolean_and_matched_reference_policy(self):
        with self.assertRaisesRegex(ContractError, 'requires explicit'):
            initialize(self.owned / 'UNIT_legacy_recorded_rejected', self.medical, self.split,
                       record_epochs=True)
        for invalid in (1, 'true', None):
            with self.subTest(record_epochs=invalid), self.assertRaises(ContractError):
                initialize(self.owned / 'UNIT_nonboolean_rejected', self.medical, self.split,
                           local_sampling='native', record_epochs=invalid)
        for reference, recorded in ((self.native, False), (self.unrecorded, True)):
            destination = self.owned / f'UNIT_policy_mismatch_{recorded}_rejected'
            with self.assertRaisesRegex(ContractError, 'same epoch recording policy'):
                initialize(destination, self.medical, self.split, local_sampling='strict_nested',
                           sampling_profile=self.profile, reference_experiment=reference,
                           record_epochs=recorded)
            self.assertFalse(destination.exists())

    def test_reinitialization_keeps_exact_recording_policy_and_preserves_existing_results(self):
        for experiment, manifest, recorded in ((self.native, self.nm, True),
                                                (self.unrecorded, self.um, False)):
            marker = experiment / 'results/v1.0/UNIT_existing_metadata.txt'
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b'UNIT owned metadata result only')
            original = (experiment / 'manifest.json').read_bytes()
            repeated = initialize(experiment, self.medical, self.split,
                                  local_sampling='native', record_epochs=recorded)
            self.assertEqual(repeated, manifest)
            with self.assertRaisesRegex(ContractError, 'different data/split contract'):
                initialize(experiment, self.medical, self.split,
                           local_sampling='native', record_epochs=not recorded)
            self.assertEqual((experiment / 'manifest.json').read_bytes(), original)
            self.assertEqual(marker.read_bytes(), b'UNIT owned metadata result only')

    def test_changed_recording_policy_rejected_even_if_manifest_is_rehashed(self):
        path = self.nested / 'manifest.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['epoch_recording']['format'] = 'UNIT_changed_policy'
            changed['manifest_sha256'] = canonical_hash(
                {k: v for k, v in changed.items() if k != 'manifest_sha256'})
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ContractError, 'Epoch recording policy changed'):
                load_suite(self.nested)
        finally:
            path.write_bytes(original)

    def test_recording_helper_mutation_stops_before_any_launch(self):
        for experiment in (self.native, self.nested):
            for name in EPOCH_RECORDING_HELPERS:
                path = experiment / 'source/v1.0' / name
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b'\n# UNIT deliberate snapshot mutation\n')
                    with self.subTest(experiment=experiment.name, helper=name), \
                            self.assertRaisesRegex(ContractError, 'Snapshot changed'):
                        command_plan(experiment, 'v1.0', 'train')
                finally:
                    path.write_bytes(original)

    def test_telemetry_request_mutation_rejected_before_subprocess_and_preserved(self):
        with patch('hiercp_v1x.experiment.bind_launch'), \
                patch('hiercp_v1x.experiment.subprocess.Popen',
                      side_effect=RuntimeError('UNIT blocked before training')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'UNIT blocked before training'):
                execute_stage(self.native, 'v1.0', 'train')
        path = self.native / 'results/v1.0/telemetry_request.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['native_reference'] = str(self.unrecorded)
            path.write_text(json.dumps(changed), encoding='utf-8')
            with patch('hiercp_v1x.experiment.bind_launch'), \
                    patch('hiercp_v1x.experiment.subprocess.Popen') as popen:
                with self.assertRaisesRegex(ContractError, 'telemetry request changed'):
                    execute_stage(self.native, 'v1.0', 'train')
                popen.assert_not_called()
            self.assertEqual(read(path), changed)
            self.assertFalse((self.native / 'results/v1.0/run.lock').exists())
        finally:
            path.write_bytes(original)

    def test_cli_record_epochs_flag_reaches_idempotent_initialization(self):
        from tools.run_v1x_experiment import main
        argv = ['run_v1x_experiment.py', 'init', '--experiment', str(self.native),
                '--medical-root', str(self.medical), '--split', str(self.split),
                '--local-sampling', 'native', '--record-epochs']
        output = io.StringIO()
        with patch('sys.argv', argv), contextlib.redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['epoch_recording'], EPOCH_RECORDING)
        self.assertEqual(result['train_cases'], 84)
        self.assertEqual(result['validation_cases'], 21)
        self.assertEqual(result['excluded_outer_cases'], 26)
        self.assertFalse(result['training_started'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
