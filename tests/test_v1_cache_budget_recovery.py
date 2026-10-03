"""Current-format recovery UNIT fixtures; no clinical inference or training.

The reusable fixture contains tiny synthetic NIfTI arrays and structural cache
payloads. They exercise integrity/copy contracts only, never a neural model.
"""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from hiercp import cache
from hiercp.schema import GraphBuildConfig
from hiercp_v1x.cache_budget_recovery import (
    _semantic_sha, migrate_failed_current_cache, validate_failed_current_cache,
)


fixture_path = Path(__file__).with_name('test_cache_recovery_debug.py')
fixture_spec = importlib.util.spec_from_file_location('_current_recovery_UNIT_fixture', fixture_path)
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)


class CurrentBudgetRecoveryUnitTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture_module.CacheMigrationDebugTests(
            methodName='test_migration_preserves_sources_and_only_relabels_verified_artifact_metadata')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        # Rebind the legacy fixture as a current donor-aware failed cache.
        contract = {key: value for key, value in f.legacy.items()
                    if key not in {'config_fingerprint', 'state', 'data_dir',
                                   'prototype_bank', 'region_cache_dir', 'progress_format'}}
        contract['donor_eligibility'] = copy.deepcopy(f.donor)
        contract['donor_contract_sha256'] = f.donor['contract_sha256']
        fingerprint = cache._cache_config_fingerprint(contract)
        f.legacy.update(contract, config_fingerprint=fingerprint)
        f.write_json(f.old / 'config.json', f.legacy)
        for key, row in f.records.items():
            row['config_fingerprint'] = fingerprint
            if key[1] is None and key[0] == 'absent':
                row['status'] = 'donor_ineligible'
            if key[1] is not None and row['status'] == 'ok':
                path = f.old / row['path']
                payload = cache._torch_load_cpu(path)
                payload['config_fingerprint'] = fingerprint
                cache._atomic_torch_save(payload, path, overwrite=True)
                row.update(artifact_sha256=cache._sha256_file(path),
                           file_size=str(path.stat().st_size))
        cache._atomic_progress_manifest_save(f.records, f.old / 'manifest.csv')
        f.kwargs['graph_config'] = GraphBuildConfig(**{
            **f.graph.to_dict(), 'adaptive_roi_max_voxels': f.graph.adaptive_roi_max_voxels + 1})
        self.before = {str(path): cache._sha256_file(path)
                       for path in f.old.parent.rglob('*') if path.is_file()}
        # This UNIT path copies every provided artifact; the real scheduler's
        # parallel resource calibration remains covered by its own tests.
        def jobs(*, tasks, function, commit, **kwargs):
            for task in tasks:
                commit(function(task))
        self.jobs = patch('hiercp.preparation_runtime.run_case_jobs', side_effect=jobs)
        self.jobs.start()
        self.addCleanup(self.jobs.stop)

    def migrate(self):
        return migrate_failed_current_cache(source_cache_dir=self.f.old,
            destination_cache_dir=self.f.new, prepare_kwargs=self.f.kwargs)

    def validate(self):
        return validate_failed_current_cache(source_cache_dir=self.f.old,
            destination_cache_dir=self.f.new, prepare_kwargs=self.f.kwargs)

    def test_success_preserves_donor_tensors_geometry_and_original_bytes(self):
        result = self.migrate()
        self.assertEqual(result['state'], 'ready_for_prepare')
        self.assertEqual(result['allowed_changes'], ['increased_adaptive_roi_max_voxels'])
        self.assertEqual(result['copied_count'], 3)
        config = cache._load_json_object(self.f.new / 'config.json')
        self.assertEqual(config['donor_eligibility'], self.f.donor)
        self.assertEqual(config['donor_contract_sha256'], self.f.donor['contract_sha256'])
        self.assertFalse((self.f.new / 'index.json').exists())
        self.assertFalse((self.f.new / 'complete.json').exists())
        self.assertFalse((self.f.new / 'migration.json').exists())
        for path in self.f.old.glob('*.pt'):
            self.assertEqual(_semantic_sha(cache._torch_load_cpu(path)),
                             _semantic_sha(cache._torch_load_cpu(self.f.new / path.name)))
        self.assertEqual(self.before, {str(path): cache._sha256_file(path)
                          for path in self.f.old.parent.rglob('*') if path.is_file()})
        self.validate()

    def test_same_or_lower_guard_and_semantic_change_rejected_before_copy(self):
        for config in (self.f.graph,
                GraphBuildConfig(**{**self.f.graph.to_dict(), 'adaptive_roi_max_voxels': 1}),
                GraphBuildConfig(**{**self.f.graph.to_dict(), 'adaptive_roi_max_voxels': 9000000,
                                    'context_radius_mm': self.f.graph.context_radius_mm + 1})):
            with self.subTest(config=config):
                self.f.kwargs['graph_config'] = config
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.f.new.exists())

    def test_changed_donor_contract_rejected(self):
        self.f.kwargs['donor_eligibility'] = copy.deepcopy(self.f.donor)
        self.f.kwargs['donor_eligibility']['eligible_case_ids'].clear()
        with self.assertRaisesRegex(ValueError, 'donor eligibility exactly'):
            self.migrate()
        self.assertFalse(self.f.new.exists())

    def test_changed_training_semantics_rejected(self):
        self.f.kwargs['seed'] = 43
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.f.new.exists())

    def test_source_corruption_rejected_before_destination_created(self):
        (self.f.old / 'donor__000.pt').write_bytes(b'UNIT deliberate corruption')
        with self.assertRaises((ValueError, FileExistsError)):
            self.migrate()
        self.assertFalse(self.f.new.exists())

    def test_existing_destination_preserved(self):
        self.f.new.mkdir()
        marker = self.f.new / 'UNIT_preserved.txt'
        marker.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            self.migrate()
        self.assertEqual(marker.read_bytes(), b'keep')

    def test_target_content_tampering_rejected(self):
        self.migrate()
        (self.f.new / 'donor__000.pt').write_bytes(b'UNIT deliberate corruption')
        with self.assertRaises(ValueError):
            self.validate()

    def test_source_after_copy_tampering_rejected(self):
        self.migrate()
        (self.f.old / 'donor__000.pt').write_bytes(b'UNIT deliberate corruption')
        with self.assertRaises(ValueError):
            self.validate()

    def test_certificate_tampering_rejected(self):
        self.migrate()
        certificate = cache._load_json_object(self.f.new / 'budget_recovery.json')
        certificate['copied_count'] = 999
        self.f.write_json(self.f.new / 'budget_recovery.json', certificate)
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
