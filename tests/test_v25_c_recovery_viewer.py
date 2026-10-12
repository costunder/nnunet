"""Explicit debug-only tests for separate C recovery observation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import watch_v25_c_recovery as viewer


class CRecoveryViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.group = Path(self.temporary.name).absolute()
        self.monitor = self.group/'C/monitor/C_recovery_debug_owned/status.json'
        self.monitor.parent.mkdir(parents=True)
        self.seal = self.group/'C_recovery_seal_debug.json'
        self.value = dict(format='v25_exact_C_standalone_continuation_v1', selected_arms=['C'],
            group_root=str(self.group), scientific_configuration_unchanged=True,
            no_process_signaled=True, slots={'C':dict(arm='C', gpu=1, monitor_path=str(self.monitor))})
        self.write()

    def write(self):
        body = json.dumps(self.value).encode()
        self.seal.write_bytes(body)
        self.digest = hashlib.sha256(body).hexdigest()

    def bind(self):
        return viewer.monitor_binding(self.group, self.seal, self.digest)

    def test_separate_monitor_is_selected(self):
        self.assertEqual(self.bind(), self.monitor)

    def test_seal_checksum_change_fails(self):
        self.seal.write_bytes(self.seal.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.bind()

    def test_canonical_monitor_is_rejected(self):
        self.value['slots']['C']['monitor_path'] = str(self.group/'C/monitor/status.json')
        self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_other_gpu_is_rejected(self):
        self.value['slots']['C']['gpu'] = 6
        self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_other_arm_is_rejected(self):
        self.value['slots']['C']['arm'] = 'A'
        self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_foreign_group_is_rejected(self):
        self.value['group_root'] = str(self.group/'other')
        self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_only_c_monitor_reads_are_routed(self):
        canonical = self.group/'C/monitor/status.json'
        canonical_request = canonical.parent/'request.json'
        other = self.group/'A/monitor/status.json'
        saved = viewer.original.read
        calls = []
        def read(path, *, required=False):calls.append((path, required));return {'path':str(path)}
        def main(args):
            viewer.original.read(canonical, required=True)
            viewer.original.read(canonical_request)
            viewer.original.read(other)
            self.assertEqual(args[-1], '--once')
        with patch.object(viewer.original, 'read', side_effect=read) as mocked_read:
            with patch.object(viewer.original, 'main', side_effect=main):
                viewer.main(['--group', str(self.group), '--recovery-seal', str(self.seal),
                    '--recovery-seal-sha256', self.digest, '--once'])
            self.assertIs(viewer.original.read, mocked_read)
        self.assertIs(viewer.original.read, saved)
        self.assertEqual(calls, [(self.monitor, True), (self.monitor.parent/'request.json', False), (other, False)])

    def test_reader_is_restored_after_error(self):
        saved = viewer.original.read
        with patch.object(viewer.original, 'main', side_effect=RuntimeError('explicit observer error')):
            with self.assertRaises(RuntimeError):
                viewer.main(['--group', str(self.group), '--recovery-seal', str(self.seal),
                    '--recovery-seal-sha256', self.digest, '--once'])
        self.assertIs(viewer.original.read, saved)


if __name__ == '__main__':unittest.main()
