"""CPU UNIT integrity races using real files, hashes and hardlinks, no training.

Windows exposes creation time as st_ctime. An extra ctime generation emulates
the Linux inode change counter while keeping real inode/size/mtime and bytes.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from hiercp_v1x import comparison_sample_cache as cache


class SampleReferenceIntegrityUnit(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix='UNIT_sample_integrity_',
                                            dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'canonical.pt'
        self.path.write_bytes(b'UNIT immutable canonical content')
        self.expected = cache._sha(self.path)
        klass = cache.sample_cache_provider(object)
        self.provider = klass.__new__(klass)
        self.provider._layout_verify_lock = threading.RLock()
        self.provider._layout_verified_files = {}
        self.provider._layout_stats = dict(raw_verifications=0)
        self.sha = cache._sha
        self.real_stat = cache._stat
        self.generation = 0
        def inode_token(path):
            value = self.real_stat(path)
            return (*value[:4], value[4] + self.generation)
        self.stat_patch = patch.object(cache, '_stat', side_effect=inode_token)
        self.stat_patch.start()
        self.addCleanup(self.stat_patch.stop)

    def mutate_during_first_hash(self, action):
        calls = []
        def hashing(path):
            result = self.sha(path)
            calls.append(result)
            if len(calls) == 1:
                action()
                self.generation += 1
            return result
        return calls, hashing

    def test_link_only_ctime_race_rehashes_and_then_memoizes_final_token(self):
        calls, hashing = self.mutate_during_first_hash(
            lambda: os.link(self.path, self.root / 'other_arm.pt'))
        with patch.object(cache, '_sha', side_effect=hashing):
            receipt = self.provider._layout_file(self.path, self.expected)
            self.assertEqual(receipt['sha256'], self.expected)
            self.assertEqual(len(calls), 2)
            self.assertEqual(self.path.stat().st_nlink, 2)
            self.assertEqual(self.provider._layout_file(self.path, self.expected), receipt)
            self.assertEqual(len(calls), 2)
        self.assertEqual(self.provider._layout_verified_files[str(self.path)][0], cache._stat(self.path))

    def test_same_size_tamper_with_restored_mtime_during_hash_is_rejected(self):
        def tamper():
            before = self.path.stat()
            data = self.path.read_bytes()
            self.path.write_bytes(b'X' + data[1:])
            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        calls, hashing = self.mutate_during_first_hash(tamper)
        with patch.object(cache, '_sha', side_effect=hashing), self.assertRaisesRegex(ValueError, 'changed while hashing'):
            self.provider._layout_file(self.path)
        self.assertEqual(len(calls), 2)
        self.assertNotIn(str(self.path), self.provider._layout_verified_files)

    def test_expected_digest_still_rejects_stable_corruption_after_ctime_reproof(self):
        self.path.write_bytes(b'X' + self.path.read_bytes()[1:])
        calls, hashing = self.mutate_during_first_hash(
            lambda: os.link(self.path, self.root / 'other_arm.pt'))
        with patch.object(cache, '_sha', side_effect=hashing), self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            self.provider._layout_file(self.path, self.expected)
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.provider._layout_verified_files)

    def test_inode_replacement_during_hash_rejects_even_identical_bytes(self):
        def replace():
            before = self.path.stat()
            replacement = self.root / 'replacement.pt'
            replacement.write_bytes(self.path.read_bytes())
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(replacement, self.path)
        _, hashing = self.mutate_during_first_hash(replace)
        with patch.object(cache, '_sha', side_effect=hashing), self.assertRaisesRegex(ValueError, 'changed while hashing'):
            self.provider._layout_file(self.path, self.expected)
        self.assertFalse(self.provider._layout_verified_files)

    def test_ctime_is_still_a_memoization_witness_and_mtime_restored_tamper_fails(self):
        self.provider._layout_file(self.path, self.expected)
        before = self.path.stat()
        self.path.write_bytes(b'X' + self.path.read_bytes()[1:])
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.generation += 1
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            self.provider._layout_file(self.path)

    def test_second_hash_mutation_is_not_treated_as_safe_metadata_reproof(self):
        count = 0
        def hashing(path):
            nonlocal count
            value = self.sha(path)
            count += 1
            if count == 1:
                os.link(path, self.root / 'other_arm.pt')
            else:
                before = path.stat()
                path.write_bytes(b'X' + path.read_bytes()[1:])
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.generation += 1
            return value
        with patch.object(cache, '_sha', side_effect=hashing), self.assertRaisesRegex(ValueError, 'changed while hashing'):
            self.provider._layout_file(self.path, self.expected)
        self.assertFalse(self.provider._layout_verified_files)

    def test_prior_verified_inode_cannot_be_replaced_by_identical_content(self):
        self.provider._layout_file(self.path, self.expected)
        replacement = self.root / 'replacement.pt'
        replacement.write_bytes(self.path.read_bytes())
        os.replace(replacement, self.path)
        with self.assertRaisesRegex(ValueError, 'inode changed'):
            self.provider._layout_file(self.path, self.expected)


if __name__ == '__main__':
    unittest.main()
