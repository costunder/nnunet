"""Full bounded-arm controller contracts; no CT/GPU/training claim.

UNIT prototype/reference bytes here exercise immutable copy and request
contracts only. They are never loaded as a neural checkpoint or CT cache.
"""
from __future__ import annotations

from tests.artifacts import unit_artifact_root
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
from zipfile import ZipFile

from tools import run_v1_bounded_training as controller
from hiercp_v1x import scope_training_entry as entry


ROOT = Path(__file__).resolve().parents[1]


def original_config():
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
        return json.loads(archive.read('config/train.json'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


class BoundedConfigUnits(unittest.TestCase):
    def test_only_two_extent_fields_change_and_complete_native_config_retained(self):
        for margin in (10., 20.):
            native = original_config()
            before = copy.deepcopy(native)
            bounded = controller.bounded_config(native, margin)
            self.assertEqual(native, before)
            expected = copy.deepcopy(native)
            expected['graph']['adaptive_roi_margin_mm'] = margin
            expected['graph']['context_outer_radius_mm'] = margin
            self.assertEqual(bounded, expected)
            self.assertEqual(bounded['training']['epochs'], 40)
            self.assertEqual(bounded['cache']['total_candidates'], 8)
            self.assertEqual(bounded['cache']['candidate_pool_size'], 128)
            self.assertEqual(bounded['seed'], 42)
            for section in ('model', 'training', 'cache', 'generation', 'runtime', 'labels', 'ct_clip'):
                self.assertEqual(bounded[section], native[section])

    def test_invalid_native_epoch_seed_and_candidate_contract_rejected(self):
        for section, key, value in (('training', 'epochs', 2), ('cache', 'total_candidates', 7),
                                   ('cache', 'candidate_pool_size', 64)):
            config = original_config(); config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                controller.bounded_config(config, 10.)
        config = original_config(); config['seed'] = 43
        with self.assertRaises(ValueError): controller.bounded_config(config, 10.)

    def test_requested_scope_must_be_explicit_supported_scalar(self):
        for margin in (None, True, [], [10., 20.], 9., 40., float('nan'), float('inf')):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                controller.bounded_config(original_config(), margin)


class ControllerFixture(unittest.TestCase):
    def setUp(self):
        self.root = unit_artifact_root() / ('v1_bounded_controller_UNIT_' + uuid.uuid4().hex)
        self.source = self.root / 'native'
        self.destination = self.root / 'bounded10'
        self.shared = self.source / 'shared'
        self.source.mkdir(parents=True)
        self.split = dict(train=[f'UNIT_train_{i}' for i in range(84)],
            val=[f'UNIT_val_{i}' for i in range(21)],
            outer_validation_excluded=[f'UNIT_outer_{i}' for i in range(26)])
        self.old = dict(sampling_contract={'mode': 'native'}, split=self.split,
            manifest_sha256='a' * 64, medical_root=str(self.root / 'UNIT_MEDICAL'))
        write(self.source / 'configs/v1.0.json', original_config())
        write(self.shared / 'split.json', self.split)
        (self.shared / 'prototype_bank.pt').write_bytes(b'UNIT immutable prototype identity only, no neural run')
        (self.shared / 'manifest.csv').write_text('UNIT,reference\nonly,copy\n', encoding='utf8')
        write(self.shared / 'metadata.json', dict(UNIT=True, state='ready',
            prototype_sha256=controller.digest(self.shared / 'prototype_bank.pt'),
            manifest_sha256=controller.digest(self.shared / 'manifest.csv'),
            training_cases=self.split['train']))
        write(self.shared / 'cache/config.json', {'UNIT': True})
        (self.shared / 'cache/manifest.csv').write_text('UNIT,no_CT_loaded\n', encoding='utf8')
        write(self.shared / 'regions/UNIT_regions.json', {'UNIT': 'copy identity only'})
        self.source_hashes = {str(path): controller.digest(path)
                              for path in self.source.rglob('*') if path.is_file()}
        self.suite = patch.object(controller, 'load_suite', side_effect=lambda *_: copy.deepcopy(self.old))
        self.preparation = patch.object(controller, 'preparation_root', return_value=self.shared)
        self.suite.start(); self.preparation.start()
        self.addCleanup(self.suite.stop); self.addCleanup(self.preparation.stop)

    def initialize(self, destination=None):
        return controller.initialize(self.source, destination or self.destination,
                                     10., 3, 40., 192.)

    def test_full84_21_arm_original_prototype_and_snapshot_preserved(self):
        root, manifest = self.initialize()
        self.assertEqual(manifest['split'], self.split)
        self.assertEqual(manifest['epochs'], 40)
        self.assertEqual(manifest['candidate_pool'], 128)
        self.assertEqual(manifest['candidates_per_sample'], 8)
        self.assertEqual(manifest['views'], 2)
        self.assertFalse(manifest['native_or_30mm_rerun'])
        self.assertTrue(manifest['original_model'])
        self.assertTrue(manifest['original_curriculum'])
        self.assertFalse(manifest['quality_verified'])
        self.assertFalse(manifest['production_CP_started'])
        self.assertFalse(manifest['nnunet_started'])
        self.assertEqual(root, self.destination)
        with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    self.assertEqual((root / 'source/v1.0' / info.filename).read_bytes(), archive.read(info))
        for name in ('split.json', 'prototype_bank.pt', 'metadata.json', 'manifest.csv',
                     'regions/UNIT_regions.json'):
            self.assertEqual((root / 'shared' / name).read_bytes(), (self.shared / name).read_bytes())
        self.assertFalse((root / 'shared/cache').exists())
        self.assertFalse((root / 'results').exists())
        for path, digest in self.source_hashes.items():
            self.assertEqual(controller.digest(path), digest)

    def test_each_invocation_requests_only_its_explicit_phase_and_owned_outputs(self):
        root, manifest = self.initialize()
        for phase in ('prepare', 'train'):
            request = controller.request_for(root, manifest, phase)
            self.assertEqual(request['phase'], phase)
            self.assertEqual(request['gpu'], 3)
            self.assertEqual(request['margin_mm'], 10.)
            self.assertEqual(request['cuda_gib'], 40.)
            self.assertEqual(request['rss_gib'], 192.)
            self.assertEqual(request['contract_sha256'], manifest['contract_sha256'])
            self.assertEqual(request['native_cache'], str(self.shared / 'cache'))
            self.assertEqual(request['source_experiment'], str(self.source))
            for name in ('source', 'config', 'prototype_bank', 'regions', 'cache', 'split'):
                self.assertTrue(Path(request[name]).is_relative_to(root))
            self.assertEqual(entry.validate_request(request), request)

    def test_subset_cohort_and_nested_input_rejected_before_output_creation(self):
        for field in ('train', 'val', 'outer_validation_excluded'):
            previous = copy.deepcopy(self.old)
            self.old['split'][field] = self.old['split'][field][:-1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.initialize()
            self.assertFalse(self.destination.exists())
            self.old = previous
        self.old['sampling_contract']['mode'] = 'strict_nested'
        with self.assertRaises(ValueError): self.initialize()
        self.assertFalse(self.destination.exists())

    def test_source_or_source_parent_or_child_are_never_experiment_outputs(self):
        for destination in (self.source, self.source / 'bounded', self.root):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.initialize(destination)

    def test_unbound_existing_directory_is_preserved(self):
        self.destination.mkdir()
        marker = self.destination / 'UNIT_existing.txt'
        marker.write_text('preserve', encoding='utf8')
        with self.assertRaises(FileExistsError): self.initialize()
        self.assertEqual(marker.read_text(), 'preserve')

    def test_immutable_copy_never_overwrites_changed_destination(self):
        destination = self.root / 'UNIT_copy.pt'
        source = self.shared / 'prototype_bank.pt'
        expected = controller.copy_verified(source, destination)
        self.assertEqual(expected, controller.digest(source))
        self.assertEqual(controller.copy_verified(source, destination), expected)
        destination.write_bytes(b'UNIT existing changed bytes')
        with self.assertRaises(ValueError): controller.copy_verified(source, destination)
        self.assertEqual(destination.read_bytes(), b'UNIT existing changed bytes')

    def test_same_owned_contract_resume_preserves_artifacts(self):
        root, manifest = self.initialize()
        before = {str(path): controller.digest(path) for path in root.rglob('*') if path.is_file()}
        again, again_manifest = self.initialize()
        self.assertEqual(again, root)
        self.assertEqual(again_manifest, manifest)
        for path, digest in before.items():
            self.assertEqual(controller.digest(path), digest)

    def test_changed_gpu_resource_or_margin_cannot_silently_resume(self):
        self.initialize()
        for margin, gpu, cuda, rss in ((20., 3, 40., 192.), (10., 2, 40., 192.),
                                     (10., 3, 39., 192.), (10., 3, 40., 191.)):
            with self.subTest(margin=margin, gpu=gpu, cuda=cuda, rss=rss), self.assertRaises(ValueError):
                controller.initialize(self.source, self.destination, margin, gpu, cuda, rss)

    def test_owned_config_split_bank_metadata_and_snapshot_tampering_rejected(self):
        root, manifest = self.initialize()
        files = ('config.json', 'shared/split.json', 'shared/metadata.json',
                 'shared/prototype_bank.pt', 'shared/manifest.csv',
                 'shared/regions/UNIT_regions.json', 'source/v1.0/hiercp/model.py')
        for name in files:
            path = root / name
            before = path.read_bytes()
            try:
                if name == 'config.json':
                    config = json.loads(before)
                    config['runtime']['allow_tf32'] = not config['runtime']['allow_tf32']
                    write(path, config)
                else:
                    path.write_bytes(before + b'\nUNIT tamper')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    controller.verify_bound_experiment(root, manifest)
                with self.subTest(resume_name=name), self.assertRaises(ValueError):
                    self.initialize()
            finally:
                path.write_bytes(before)

    def test_unlisted_region_file_and_source_file_rejected(self):
        root, manifest = self.initialize()
        for name in ('shared/regions/UNIT_unlisted.json', 'source/v1.0/UNIT_unlisted.txt'):
            path = root / name
            path.write_bytes(b'UNIT unlisted member')
            with self.subTest(name=name), self.assertRaises(ValueError):
                controller.verify_bound_experiment(root, manifest)
            path.unlink()  # Only this test's exact created UNIT file.

    def test_contract_hash_wrong_root_phase_and_reference_tampering_rejected(self):
        root, manifest = self.initialize()
        changed = copy.deepcopy(manifest); changed['gpu'] = 2
        with self.assertRaises(ValueError): controller.verify_bound_experiment(root, changed)
        wrong = self.root / 'UNIT_other_output'; wrong.mkdir()
        with self.assertRaises(ValueError): controller.request_for(wrong, manifest, 'train')
        for phase in ('native', '30mm', 'generate', 'nnunet', ''):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                controller.request_for(root, manifest, phase)
        native = self.shared / 'cache/config.json'
        native.write_bytes(native.read_bytes() + b'\nUNIT reference tamper')
        with self.assertRaises(ValueError): controller.verify_bound_experiment(root, manifest)

    def test_controller_launches_only_prepare_and_train_for_same_bounded_arm(self):
        args = ['run_v1_bounded_training.py', '--gpu', '3', '--margin-mm', '10',
                '--source-experiment', str(self.source), '--experiment', str(self.destination),
                '--cuda-gib', '40', '--rss-gib', '192']
        # Mock subprocess alone: initialization/request hash checks remain real.
        with patch.object(controller.sys, 'argv', args), \
                patch.object(controller.subprocess, 'Popen') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.wait.return_value = 0
            controller.main()
        self.assertEqual(launch.call_count, 2)
        phases = []
        for call in launch.call_args_list:
            command = call.args[0]
            self.assertEqual(command[2:4], ['-m', 'hiercp_v1x.scope_training_entry'])
            self.assertEqual(command[4], '--request')
            request = json.loads(Path(command[5]).read_text(encoding='utf8'))
            phases.append(request['phase'])
            self.assertEqual(request['experiment'], str(self.destination))
            self.assertEqual(request['margin_mm'], 10.)
            self.assertEqual(request['gpu'], 3)
            self.assertEqual(request['contract_sha256'],
                             json.loads((self.destination/'manifest.json').read_text())['contract_sha256'])
            self.assertEqual(call.kwargs['cwd'], ROOT)
            self.assertEqual(call.kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertNotIn('HIERCP_V1X_SAMPLING_CONTRACT', call.kwargs['env'])
        self.assertEqual(phases, ['prepare', 'train'])
        self.assertFalse((self.destination / 'run.lock').exists())

    def test_failed_prepare_does_not_launch_train_or_native_reference(self):
        args = ['run_v1_bounded_training.py', '--gpu', '3', '--margin-mm', '10',
                '--source-experiment', str(self.source), '--experiment', str(self.destination),
                '--cuda-gib', '40', '--rss-gib', '192']
        with patch.object(controller.sys, 'argv', args), \
                patch.object(controller.subprocess, 'Popen') as launch, redirect_stdout(io.StringIO()):
            launch.return_value.wait.return_value = 1
            with self.assertRaises(RuntimeError): controller.main()
        self.assertEqual(launch.call_count, 1)
        request_path = Path(launch.call_args.args[0][-1])
        self.assertEqual(json.loads(request_path.read_text())['phase'], 'prepare')
        self.assertFalse((self.destination / 'run.lock').exists())


if __name__ == '__main__':
    unittest.main()
