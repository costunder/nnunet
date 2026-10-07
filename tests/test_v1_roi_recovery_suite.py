"""UNIT tests for explicit resource-only fresh-suite recovery; no CT/model run."""
from tests.artifacts import unit_artifact_root
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from hiercp_v1x.contracts import (
    ContractError, canonical_hash, make_preparation_admission, verify_archive,
)
from hiercp_v1x.experiment import initialize, load_suite, command_plan, ROOT
from hiercp_v1x.budget_recovery_entry import install_recovery
from tools.run_v1x_experiment import admission_from_probe


class RoiRecoverySuiteUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = unit_artifact_root()/('roi_recovery_suite_UNIT_'+uuid.uuid4().hex)
        cls.medical = cls.root/'medical'
        for kind in ('image', 'labels'):
            directory = cls.medical/'Data'/kind
            directory.mkdir(parents=True)
            for case in ('unit_train', 'unit_val', 'unit_outer'):
                suffix = '_0000.nii.gz' if kind == 'image' else '.nii.gz'
                (directory/(case+suffix)).write_bytes(b'UNIT filename-only, never decoded')
        cls.split = cls.root/'split.json'
        cls.split.write_text(json.dumps(dict(seed=42, train=['unit_train'], val=['unit_val'],
            outer_validation_excluded=['unit_outer'])), encoding='utf-8')
        cls.profile = make_preparation_admission(12_000_000, 'a'*64)

    def destination(self, name):
        return self.root/(name+'_'+uuid.uuid4().hex)

    def create(self, destination, **kwargs):
        return initialize(destination, self.medical, self.split,
                          local_sampling='native', record_epochs=True, **kwargs)

    def test_default_stays_exact_eight_million(self):
        target = self.destination('default')
        m = self.create(target)
        self.assertNotIn('preparation_admission', m)
        cfg = json.loads((target/'configs/v1.0.json').read_text())
        self.assertEqual(cfg, verify_archive()['base_config'])

    def test_explicit_guard_and_native_source_preserved(self):
        target = self.destination('admission')
        m = self.create(target, preparation_admission=self.profile)
        cfg = json.loads((target/'configs/v1.0.json').read_text())
        expected = verify_archive()['base_config']
        expected['graph']['adaptive_roi_max_voxels'] = 12_000_000
        self.assertEqual(cfg, expected)
        self.assertEqual(load_suite(target), m)
        for name in ('hiercp/pipeline.py', 'hiercp/cache.py', 'hiercp/spatial.py', 'hiercp/sample.py'):
            self.assertEqual(m['stages']['v1.0']['source_hashes'][name],
                             verify_archive()['file_hashes'][name])

    def test_same_profile_required_for_nested_arm(self):
        native, nested = self.destination('native'), self.destination('nested')
        self.create(native, preparation_admission=self.profile)
        options = dict(local_sampling='strict_nested',
            sampling_profile=dict(zip(('tumor_surface','tumor_interior','source_context',
                                      'source_liver_surface','target_context','target_liver_surface'),
                                     (64,32,96,64,96,64))),
            reference_experiment=native, record_epochs=True)
        with self.assertRaisesRegex(ContractError, 'same explicit ROI'):
            initialize(nested, self.medical, self.split, **options)
        m = initialize(nested, self.medical, self.split,
                       preparation_admission=self.profile, **options)
        self.assertEqual(load_suite(nested), m)

    def test_existing_suite_guard_not_mutated(self):
        target = self.destination('old')
        self.create(target)
        before = (target/'manifest.json').read_bytes()
        with self.assertRaises(ContractError):
            self.create(target, preparation_admission=self.profile)
        self.assertEqual((target/'manifest.json').read_bytes(), before)

    def test_recovery_plan_uses_only_bound_entry(self):
        source, target = self.destination('failed'), self.destination('recovered')
        self.create(source)
        cache = source/'shared/cache'
        cache.mkdir()
        (cache/'config.json').write_text('{"state":"failed"}', encoding='utf-8')
        (cache/'manifest.csv').write_text('UNIT metadata placeholder; never passed to cache API', encoding='utf-8')
        self.create(target, preparation_admission=self.profile, recover_preparation_from=source)
        cwd, commands = command_plan(target, 'v1.0', 'prepare')
        self.assertEqual(cwd, target/'source/v1.0')
        self.assertEqual(len(commands), 1)
        self.assertIn('hiercp_v1x.budget_recovery_entry', commands[0])
        self.assertNotIn('train', commands[0])
        (cache/'manifest.csv').write_text('UNIT source changed', encoding='utf-8')
        with self.assertRaisesRegex(ContractError, 'source changed'):
            load_suite(target)

    def test_original_ready_cache_cannot_be_recovery_source(self):
        source = self.destination('published')
        self.create(source)
        cache = source/'shared/cache'
        cache.mkdir()
        (cache/'config.json').write_text('{"state":"failed"}', encoding='utf-8')
        (cache/'index.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ContractError, 'failed and unpublished'):
            self.create(self.destination('target'), preparation_admission=self.profile,
                        recover_preparation_from=source)

    def test_probe_must_pass_without_quality_promotion(self):
        path = self.root/('probe_UNIT_'+uuid.uuid4().hex+'.json')
        report = dict(format='v1_roi_budget_probe_v1', scope='debug_geometry_only',
            archived_source_sha256=verify_archive()['archive_sha256'],
            original_roi_max_voxels=8_000_000, candidate_roi_max_voxels=12_000_000,
            completed=True, originals_preserved=True, training_started=False,
            cache_publication_created=False, failed_sample_requests=1,
            measurements=[dict(status='PASS', geometry_equal=True, original_guard_reproduced=True)])
        path.write_text(json.dumps(report), encoding='utf-8')
        value = admission_from_probe(12_000_000, path)
        self.assertFalse(value['quality_verified'])
        for field in ('completed','originals_preserved'):
            bad = copy.deepcopy(report)
            bad[field] = False
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaises(ValueError):
                admission_from_probe(12_000_000, path)

    def test_recovery_wrapper_preserves_prepare_kwargs_and_checks_both_sides(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        target = self.destination('wrapper')
        target.mkdir()
        cache = target/'shared/cache'
        cache.mkdir(parents=True)
        pipeline = SimpleNamespace(prepare_hierarchical_cache=Mock(return_value=['original']))
        original = pipeline.prepare_hierarchical_cache
        with patch('hiercp_v1x.cache_budget_recovery.validate_failed_current_cache') as validate:
            install_recovery(pipeline, self.root/'unused', target)
            kwargs = dict(cache_dir=str(cache), UNIT_no_real_data=True)
            self.assertEqual(pipeline.prepare_hierarchical_cache(**kwargs), ['original'])
            original.assert_called_once_with(**kwargs)
            self.assertEqual(validate.call_count, 2)


if __name__ == '__main__':
    unittest.main()
