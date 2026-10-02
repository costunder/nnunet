"""Sampling-runner METADATA UNITs; no CT decode, neural forward or learning.

NIfTI-named empty files only exercise full cohort accounting. All publications
belong to a fresh task-owned TemporaryDirectory and are never production data.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from hiercp_v1x.contracts import (ContractError, LOCAL_SAMPLING_ROLES,
    TARGET_CONTRACT, V1_ARCHIVE_SHA256, canonical_hash, compare_reports,
    local_sampling_spec, make_run_contract, resolve_execution_config, validate_resume)
from hiercp_v1x.experiment import (ROOT, SAMPLING_FORMAT, command_plan, digest,
    execution_is_resolved, execution_reference, freeze_execution, initialize,
    load_suite, preparation_root, sampler_contract, sampling_overlay)


class SamplingExperimentMetadataUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='v1x_sampling_metadata_UNIT_', dir=ROOT / 'work')
        cls.owned = Path(cls.temporary.name).resolve()
        cls.medical = cls.owned / 'medical'
        for kind in ('image', 'labels'):
            directory = cls.medical / 'Data' / kind
            directory.mkdir(parents=True)
            for case in ('UNIT_train', 'UNIT_val', 'UNIT_outer'):
                suffix = '_0000.nii.gz' if kind == 'image' else '.nii.gz'
                (directory / (case + suffix)).write_bytes(b'')
        cls.split = cls.owned / 'split.json'
        cls.split.write_text(json.dumps(dict(seed=42, train=['UNIT_train'], val=['UNIT_val'],
                                            outer_validation_excluded=['UNIT_outer'])), encoding='utf-8')
        cls.native = cls.owned / 'native'
        cls.nested = cls.owned / 'nested'
        cls.profile = dict(zip(LOCAL_SAMPLING_ROLES, (16, 8, 24, 16, 24, 16)))
        cls.nm = initialize(cls.native, cls.medical, cls.split, local_sampling='native')
        cls.sm = initialize(cls.nested, cls.medical, cls.split, local_sampling='strict_nested',
                            sampling_profile=cls.profile, reference_experiment=cls.native)

    @classmethod
    def tearDownClass(cls):
        # TemporaryDirectory removes only the exact task-owned UNIT root.
        cls.temporary.cleanup()

    def test_all_six_role_budgets_are_mandatory_no_arbitrary_profile(self):
        for profile in (None, {}, {'tumor_surface': 16}, {**self.profile, 'other': 1},
                        {**self.profile, 'target_context': True},
                        {**self.profile, 'target_context': 0}):
            with self.subTest(profile=profile), self.assertRaises(ContractError):
                local_sampling_spec('strict_nested', profile)
        with self.assertRaises(ContractError):
            local_sampling_spec('native', self.profile)
        self.assertEqual(local_sampling_spec('strict_nested', self.profile)['profile'], self.profile)

    def test_native_core_bytes_and_nested_only_declared_append_hooks(self):
        self.assertEqual(self.nm['format'], SAMPLING_FORMAT)
        with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
            for name in archive.namelist():
                original = archive.read(name)
                self.assertEqual((self.native / 'source/v1.0' / name).read_bytes(), original)
                self.assertEqual((self.nested / 'source/v1.0' / name).read_bytes(), original + sampling_overlay(name))
        self.assertEqual(sampling_overlay('hiercp/tensor.py'), b'')
        self.assertFalse(self.sm['stages']['v1.0']['source_original_exact'])

    def test_nested_reuses_exact_native_preparation_and_calibration_root(self):
        self.assertEqual(preparation_root(self.nested), self.native / 'shared')
        self.assertEqual(execution_reference(self.nested), self.native)
        self.assertFalse((self.nested / 'shared/cache').exists())
        self.assertTrue(execution_is_resolved(self.sm, 'v1.0'))
        self.assertFalse(execution_is_resolved(self.nm, 'v1.0'))
        calibration = dict(format='hiercp_preflight_calibration_v2', selected_batch_size=32,
            selected_num_workers=8, resource_fingerprint={'UNIT_metadata_only': True},
            identity=dict(seed=42, cache_dir=str(self.native / 'shared/cache'),
                          checkpoint_path=str(self.native / 'results/v1.0/checkpoint_best.pt')))
        target = self.native / 'results/v1.0/checkpoint_best.pt.preflight.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(calibration), encoding='utf-8')
        native_lock = freeze_execution(self.native)
        self.assertEqual(freeze_execution(self.nested), native_lock)
        self.assertEqual(native_lock['baseline_sampling_contract_sha256'],
                         self.nm['sampling_contracts']['v1.0']['contract_sha256'])
        self.assertFalse((self.nested / 'shared/execution_lock.json').exists())

    def test_train_and_generation_select_explicit_frozen_contract_prepare_is_native(self):
        for experiment, manifest in ((self.native, self.nm), (self.nested, self.sm)):
            for target in ('train', 'generate'):
                cwd, commands = command_plan(experiment, 'v1.0', target, python='UNIT_python')
                self.assertEqual(cwd, experiment / 'source/v1.0')
                argv = commands[0]
                self.assertEqual(argv[:4], ['UNIT_python', '-u', '-m', 'hiercp_v1x.sampling_entry'])
                self.assertEqual(argv[argv.index('--contract') + 1], str(experiment / 'sampling/v1.0.json'))
                self.assertIn(target, argv)
            cwd, commands = command_plan(experiment, 'v1.0', 'prepare', python='UNIT_python')
            self.assertEqual(cwd, self.native / 'source/v1.0')
            self.assertEqual(commands[0][:4], ['UNIT_python', '-u', '-m', 'hiercp.pipeline'])
            self.assertNotIn('--contract', commands[0])
        _, commands = command_plan(self.nested, 'v1.0', 'train', python='UNIT_python')
        self.assertEqual(commands[0][commands[0].index('--config') + 1],
                         str(self.nested / 'results/v1.0/resolved_config.json'))
        self.assertEqual(commands[0][commands[0].index('--cache-dir') + 1], str(self.native / 'shared/cache'))

    def test_nested_requires_explicit_native_reference_and_same_full_cohort(self):
        with self.assertRaisesRegex(ContractError, 'reference experiment'):
            initialize(self.owned / 'UNIT_missing_reference', self.medical, self.split,
                       local_sampling='strict_nested', sampling_profile=self.profile)
        with self.assertRaisesRegex(ContractError, 'explicit native'):
            initialize(self.owned / 'UNIT_nested_reference', self.medical, self.split,
                       local_sampling='strict_nested', sampling_profile=self.profile,
                       reference_experiment=self.nested)
        with self.assertRaisesRegex(ContractError, 'different data/split'):
            initialize(self.nested, self.medical, self.split, local_sampling='strict_nested',
                       sampling_profile={role: count + 1 for role, count in self.profile.items()},
                       reference_experiment=self.native)

    def test_frozen_contract_file_and_native_reference_identity_tampering_rejected(self):
        path = self.nested / 'sampling/v1.0.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['profile']['target_context'] += 1
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ContractError, 'Sampling contract changed'):
                load_suite(self.nested)
        finally:
            path.write_bytes(original)
        manifest_path = self.native / 'manifest.json'
        original = manifest_path.read_bytes()
        try:
            changed = json.loads(original)
            changed['UNIT_reference_identity_change'] = True
            changed['manifest_sha256'] = canonical_hash({k:v for k,v in changed.items() if k != 'manifest_sha256'})
            manifest_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ContractError, 'Native reference identity'):
                load_suite(self.nested)
        finally:
            manifest_path.write_bytes(original)

    def test_sampling_contract_is_in_exact_resume_identity_without_changing_v1_config(self):
        config = json.loads((self.native / 'configs/v1.0.json').read_text(encoding='utf-8'))
        self.assertEqual(config, json.loads((self.nested / 'configs/v1.0.json').read_text(encoding='utf-8')))
        lock = dict(selected_batch_size=32, selected_num_workers=8)
        resolved = resolve_execution_config(config, lock)
        common = dict(cache_identity={'UNIT_cache': 'unchanged'}, source_identity={'UNIT_source': 'unchanged'},
                      evaluation_identity={'UNIT_target': 'unchanged'}, physical_batch_size=32,
                      debug=False, execution_lock=lock)
        native = make_run_contract('v1.0', resolved, **common,
                                  sampling_contract=sampler_contract(self.native, 'v1.0'))
        nested = make_run_contract('v1.0', resolved, **common,
                                  sampling_contract=sampler_contract(self.nested, 'v1.0'))
        validate_resume(native, copy.deepcopy(native))
        with self.assertRaisesRegex(ContractError, 'resume contract mismatch'):
            validate_resume(native, nested)
        self.assertEqual(native['target_contract'], nested['target_contract'])
        self.assertEqual(native['physical_batch_size'], nested['physical_batch_size'])

    def test_stage_zero_sampling_compare_is_explicit_and_never_a_stage_upgrade(self):
        evaluation = dict(target_contract=TARGET_CONTRACT, candidate_contract='UNIT_same_candidate_policy',
            split_sha256='a' * 64, candidate_ids_sha256='b' * 64, mask_contract='UNIT_same_mask_policy',
            source_archive_sha256=V1_ARCHIVE_SHA256, evaluation_source_sha256='c' * 64,
            physical_batch_size=32, effective_batch_size=32, target_epochs=40,
            denominator={'validation_samples': 2, 'validation_cases': 1, 'candidates_per_sample': 8})
        base = dict(stage='v1.0', complete=True, debug=True, evaluation=evaluation,
                    metrics={'mrr': 0.5}, sampling_contract=sampler_contract(self.native, 'v1.0'))
        candidate = {**base, 'metrics': {'mrr': 0.4},
                     'sampling_contract': sampler_contract(self.nested, 'v1.0')}
        result = compare_reports(base, candidate)
        self.assertEqual(result['format'], 'hiercp_v1x_sampling_comparison_v1')
        self.assertAlmostEqual(result['metric_deltas']['baseline']['mrr'], -0.1)
        self.assertFalse(result['graph_quality_passed'])
        self.assertFalse(result['automatic_promotion'])
        with self.assertRaisesRegex(ContractError, 'strict_nested sampling'):
            compare_reports(base, copy.deepcopy(base))
        with self.assertRaisesRegex(ContractError, 'incomparable'):
            compare_reports(base, {**candidate, 'evaluation': {**evaluation, 'physical_batch_size': 16}})
        bound_base = {**base, 'native_reference_experiment': str(self.native)}
        bound_candidate = {**candidate, 'native_reference_experiment': str(self.native)}
        self.assertEqual(compare_reports(bound_base, bound_candidate)['format'], result['format'])
        for foreign in (candidate, {**candidate, 'native_reference_experiment': str(self.nested)}):
            with self.assertRaisesRegex(ContractError, 'another native reference'):
                compare_reports(bound_base, foreign)


if __name__ == '__main__':
    unittest.main(verbosity=2)
