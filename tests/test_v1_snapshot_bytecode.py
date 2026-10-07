"""Frozen-source/bytecode METADATA UNIT regressions; no CT or model execution.

Every fixture belongs to a new work/cache/tests/v1_snapshot_bytecode_UNIT_<UUID> root and
is preserved. Cache payloads in inventory tests are metadata-only bytes. The
subprocess regression executes only a two-line UNIT module and proves an old
timestamp-valid cache cannot override the verified source with isolated lookup.
"""
from __future__ import annotations

from tests.artifacts import unit_artifact_root
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
from zipfile import ZipFile

from hiercp_v1x import experiment
from hiercp_v1x.contracts import ContractError, STAGES, TARGET_CONTRACT, canonical_hash
from hiercp_v1x.snapshot_inventory import (
    checked_snapshot_inventory, isolated_snapshot_bytecode_env,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotBytecodeMetadataUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owned = unit_artifact_root() / (
            'v1_snapshot_bytecode_UNIT_' + uuid.uuid4().hex
        )
        cls.owned.mkdir(parents=True, exist_ok=False)

    def setUp(self):
        self.root = self.owned / self._testMethodName
        self.root.mkdir(exist_ok=False)

    def write_new(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
        return path

    def inventory_fixture(self, *, omit=()):
        source = self.root / 'source' / 'v1.0'
        source.mkdir(parents=True)
        payloads = {
            'hiercp/__init__.py': b'# UNIT inventory fixture; never imported\n',
            'hiercp/spatial.py': b'# UNIT tracked spatial source; never imported\n',
        }
        for name, payload in payloads.items():
            if name not in omit:
                self.write_new(source / name, payload)
        return source, set(payloads)

    def cache_for(self, source, source_name='hiercp/spatial.py', *, optimization=''):
        return Path(importlib.util.cache_from_source(
            str(source / source_name), optimization=optimization,
        ))

    def assert_rejected_name(self, source, expected, name):
        with self.assertRaises(ContractError) as caught:
            checked_snapshot_inventory(source, expected)
        self.assertIn(name, str(caught.exception))
        self.assertIn(str(source), str(caught.exception))

    def test_generated_cache_is_tolerated_and_all_bytes_preserved(self):
        source, expected = self.inventory_fixture()
        caches = [self.cache_for(source), self.cache_for(source, optimization='1')]
        for path in caches:
            self.write_new(path, b'UNIT cache metadata; never executed\n')
        before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        result = checked_snapshot_inventory(source, expected)
        self.assertEqual(result['files'], expected)
        self.assertEqual(set(result['ignored_bytecode']), {
            p.relative_to(source).as_posix() for p in caches
        })
        self.assertEqual(before, {p: p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_missing_listed_source_is_rejected_despite_existing_cache(self):
        missing = 'hiercp/spatial.py'
        source, expected = self.inventory_fixture(omit={missing})
        self.write_new(self.cache_for(source), b'UNIT metadata-only old cache')
        self.assert_rejected_name(source, expected, missing)

    def test_unlisted_python_source_is_rejected(self):
        source, expected = self.inventory_fixture()
        name = 'hiercp/UNIT_unlisted.py'
        self.write_new(source / name, b'# UNIT unlisted source; never imported\n')
        self.assert_rejected_name(source, expected, name)

    def test_unlisted_python_source_inside_pycache_is_rejected(self):
        source, expected = self.inventory_fixture()
        name = 'hiercp/__pycache__/UNIT_unlisted.py'
        self.write_new(source / name, b'# UNIT source injection; never imported\n')
        self.assert_rejected_name(source, expected, name)

    def test_bytecode_outside_direct_pycache_is_rejected(self):
        source, expected = self.inventory_fixture()
        name = 'hiercp/spatial.pyc'
        self.write_new(source / name, b'UNIT metadata-only standalone bytecode')
        self.assert_rejected_name(source, expected, name)

    def test_bytecode_nested_below_pycache_is_rejected(self):
        source, expected = self.inventory_fixture()
        path = source / 'hiercp/__pycache__/nested' / self.cache_for(source).name
        self.write_new(path, b'UNIT metadata-only nested bytecode')
        self.assert_rejected_name(source, expected, path.relative_to(source).as_posix())

    def test_cache_mapping_to_unlisted_source_is_rejected(self):
        source, expected = self.inventory_fixture()
        path = self.cache_for(source, 'hiercp/UNIT_unlisted.py')
        self.write_new(path, b'UNIT metadata-only unlisted module cache')
        self.assert_rejected_name(source, expected, path.relative_to(source).as_posix())

    def test_malformed_cache_name_is_rejected(self):
        source, expected = self.inventory_fixture()
        name = 'hiercp/__pycache__/spatial.pyc'
        self.write_new(source / name, b'UNIT metadata-only malformed cache')
        self.assert_rejected_name(source, expected, name)

    def test_non_bytecode_extra_inside_pycache_is_rejected(self):
        source, expected = self.inventory_fixture()
        name = 'hiercp/__pycache__/UNIT_extra.txt'
        self.write_new(source / name, b'UNIT metadata-only extra')
        self.assert_rejected_name(source, expected, name)

    def test_listed_bytecode_remains_a_required_inventory_member(self):
        source, expected = self.inventory_fixture()
        path = self.cache_for(source)
        self.write_new(path, b'UNIT explicitly listed metadata bytes')
        name = path.relative_to(source).as_posix()
        result = checked_snapshot_inventory(source, expected | {name})
        self.assertEqual(result['files'], expected | {name})
        self.assertEqual(result['ignored_bytecode'], [])

    def test_symlink_cannot_be_ignored_as_generated_cache(self):
        source, expected = self.inventory_fixture()
        outside = self.write_new(self.root / 'UNIT_external_cache', b'UNIT not bytecode')
        link = self.cache_for(source)
        link.parent.mkdir()
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'UNIT symlink creation unavailable: {error}')
        with self.assertRaises(ContractError):
            checked_snapshot_inventory(source, expected)
        self.assertTrue(link.is_symlink())
        self.assertEqual(outside.read_bytes(), b'UNIT not bytecode')

    def test_symlink_directory_in_snapshot_is_rejected(self):
        source, expected = self.inventory_fixture()
        outside = self.root / 'UNIT_external_directory'
        outside.mkdir()
        link = source / 'UNIT_symlink_directory'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'UNIT directory symlink creation unavailable: {error}')
        with self.assertRaises(ContractError):
            checked_snapshot_inventory(source, expected)

    def test_isolated_lookup_environment_is_absolute_and_preserves_parent(self):
        lookup = self.root / 'UNIT_new_lookup'
        before = dict(os.environ)
        with patch.dict(os.environ, {'UNIT_SNAPSHOT_PARENT_VARIABLE': 'preserved'}):
            result = isolated_snapshot_bytecode_env(lookup)
            self.assertEqual(result['UNIT_SNAPSHOT_PARENT_VARIABLE'], 'preserved')
            self.assertEqual(result['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertEqual(result['PYTHONPYCACHEPREFIX'], str(lookup.resolve()))
            self.assertNotEqual(os.environ.get('PYTHONPYCACHEPREFIX'), str(lookup.resolve()))
        self.assertEqual(dict(os.environ), before)
        self.assertFalse(lookup.exists())

    def test_isolated_lookup_rejects_existing_directory_or_file(self):
        directory = self.root / 'UNIT_existing_lookup'
        directory.mkdir()
        existing_file = self.write_new(self.root / 'UNIT_existing_lookup_file', b'preserved')
        for path in (directory, existing_file):
            with self.subTest(path=path), self.assertRaises(ContractError):
                isolated_snapshot_bytecode_env(path)
        self.assertEqual(existing_file.read_bytes(), b'preserved')

    def test_isolated_lookup_rejects_broken_symlink(self):
        link = self.root / 'UNIT_lookup_symlink'
        try:
            link.symlink_to(self.root / 'UNIT_nonexistent_target', target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'UNIT lookup symlink creation unavailable: {error}')
        with self.assertRaises(ContractError):
            isolated_snapshot_bytecode_env(link)
        self.assertTrue(link.is_symlink())

    def test_isolated_subprocess_executes_source_and_preserves_forged_cache(self):
        source = self.root / 'source'
        module = self.write_new(source / 'UNIT_verified_source.py',
                                b"VALUE = 'verified_source'\n")
        cache = Path(importlib.util.cache_from_source(str(module), optimization=''))
        metadata = module.stat()
        forged_code = compile("VALUE = 'different_cached_code'\n", str(module), 'exec')
        payload = (importlib.util.MAGIC_NUMBER + struct.pack(
            '<III', 0, int(metadata.st_mtime) & 0xffffffff, metadata.st_size & 0xffffffff,
        ) + marshal.dumps(forged_code))
        self.write_new(cache, payload)
        original_files = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        command = [sys.executable, '-B', '-c',
                   'import UNIT_verified_source; print(UNIT_verified_source.VALUE)']
        baseline_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        baseline_env.pop('PYTHONPYCACHEPREFIX', None)
        # Establish that this exact forged cache can override normal source import.
        baseline = subprocess.run(command, cwd=source, env=baseline_env, text=True,
                                  capture_output=True, check=True, timeout=20)
        self.assertEqual(baseline.stdout.strip(), 'different_cached_code')
        lookup = self.root / 'UNIT_isolated_cache_lookup'
        isolated = subprocess.run(command, cwd=source,
                                  env=isolated_snapshot_bytecode_env(lookup), text=True,
                                  capture_output=True, check=True, timeout=20)
        self.assertEqual(isolated.stdout.strip(), 'verified_source')
        self.assertEqual(original_files, {
            p: p.read_bytes() for p in source.rglob('*') if p.is_file()
        })
        self.assertFalse(lookup.exists())

    def suite_fixture(self, *, omitted_source=None, changed_source=None, rebind=False):
        """Tiny coherent four-stage fixture; archive/config admission alone is mocked.

        Its sources are metadata-only comments and never imported. Real stage
        overlays, manifest binding, inventory checks, SHA checks, and byte-exact
        overlay validation all execute in load_suite.
        """
        suite, repo = self.root / 'suite', self.root / 'UNIT_archive_repository'
        archive_path = repo / 'versions/v1/pipeline_v1_source.zip'
        archive_path.parent.mkdir(parents=True)
        archive_sources = {
            name: f'# UNIT archived metadata for {name}; never imported\n'.encode()
            for name in ('hiercp/model.py', 'hiercp/contracts.py', 'hiercp/sample.py')
        }
        with ZipFile(archive_path, 'x') as archive:
            for name, payload in archive_sources.items():
                archive.writestr(name, payload)
        proof = dict(archive_sha256=sha(archive_path.read_bytes()),
                     file_hashes={name: sha(payload) for name, payload in archive_sources.items()})
        split = dict(seed=42, train=['UNIT_train'], val=['UNIT_val'])
        self.write_new(suite / 'shared/split.json', json.dumps(split).encode())
        helpers = {'hiercp_v1x/__init__.py': b'',
                   'hiercp_v1x/models.py': b'# UNIT helper metadata; never imported\n'}
        config = {'UNIT_metadata_only': True}
        stages = {}
        for stage in STAGES:
            hashes = {}
            for name, original in {**archive_sources, **helpers}.items():
                expected_payload = original + experiment.overlay(stage, name)
                actual_payload = expected_payload
                if stage == 'v1.0' and name == changed_source:
                    actual_payload += b'# UNIT unauthorized source change\n'
                if not (stage == 'v1.0' and name == omitted_source):
                    self.write_new(suite / 'source' / stage / name, actual_payload)
                hashes[name] = sha(actual_payload if rebind else expected_payload)
            self.write_new(suite / 'configs' / f'{stage}.json', json.dumps(config).encode())
            stages[stage] = dict(spec=STAGES[stage].to_dict(), source_hashes=hashes,
                                 config_sha256=canonical_hash(config))
        manifest = dict(format=experiment.FORMAT, archive_sha256=proof['archive_sha256'],
                        target_contract=TARGET_CONTRACT, split=split, stages=stages)
        manifest['manifest_sha256'] = canonical_hash(manifest)
        self.write_new(suite / 'manifest.json', json.dumps(manifest).encode())
        return suite, repo, proof, manifest

    def load_metadata_suite(self, suite, repo, proof):
        with patch.object(experiment, 'verify_archive', return_value=proof), \
                patch.object(experiment, 'validate_stage_config'):
            return experiment.load_suite(suite, repo=repo)

    def test_load_suite_accepts_cache_and_preserves_manifest_and_source_bytes(self):
        suite, repo, proof, manifest = self.suite_fixture()
        source = suite / 'source/v1.0'
        cache = self.cache_for(source, 'hiercp/model.py')
        self.write_new(cache, b'UNIT metadata-only generated cache')
        before = {p: p.read_bytes() for p in suite.rglob('*') if p.is_file()}
        self.assertEqual(self.load_metadata_suite(suite, repo, proof), manifest)
        self.assertEqual(before, {p: p.read_bytes() for p in suite.rglob('*') if p.is_file()})

    def test_load_suite_rejects_missing_source_even_with_mapped_cache(self):
        name = 'hiercp/model.py'
        suite, repo, proof, _ = self.suite_fixture(omitted_source=name)
        self.write_new(self.cache_for(suite / 'source/v1.0', name), b'UNIT old cache')
        with self.assertRaises(ContractError) as caught:
            self.load_metadata_suite(suite, repo, proof)
        self.assertIn(name, str(caught.exception))
        self.assertIn('v1.0', str(caught.exception))

    def test_load_suite_cache_tolerance_does_not_bypass_source_hash(self):
        suite, repo, proof, _ = self.suite_fixture(changed_source='hiercp/model.py')
        self.write_new(self.cache_for(suite / 'source/v1.0', 'hiercp/model.py'), b'UNIT old cache')
        with self.assertRaisesRegex(ContractError, 'Snapshot changed'):
            self.load_metadata_suite(suite, repo, proof)

    def test_load_suite_cache_tolerance_does_not_bypass_exact_overlay(self):
        suite, repo, proof, _ = self.suite_fixture(
            changed_source='hiercp/model.py', rebind=True,
        )
        self.write_new(self.cache_for(suite / 'source/v1.0', 'hiercp/model.py'), b'UNIT old cache')
        with self.assertRaisesRegex(ContractError, 'Unspecified source overlay'):
            self.load_metadata_suite(suite, repo, proof)


if __name__ == '__main__':
    unittest.main()
