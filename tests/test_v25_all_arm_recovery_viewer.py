"""Debug-only verification that old FAILED records remain untouched."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import watch_v25_all_arm_recovery as viewer


class AllRecoveryViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.group = Path(self.tmp.name).absolute()
        self.seal = self.group/'ALL_recovery_seal_debug.json'
        self.value = dict(format=viewer.FORMAT, selected_arms=list(viewer.GPUS),
            group_root=str(self.group), scientific_configuration_unchanged=True,
            no_process_signaled=True, slots={})
        for arm,gpu in viewer.GPUS.items():
            monitor = self.group/arm/'monitor/ALL_recovery_debug/status.json'
            monitor.parent.mkdir(parents=True)
            old = monitor.parent.parent/'status.json';old.write_text('{"status":"FAILED"}')
            self.value['slots'][arm] = dict(arm=arm,gpu=gpu,monitor_path=str(monitor))
        self.write()

    def write(self):
        body = json.dumps(self.value).encode();self.seal.write_bytes(body)
        self.digest = hashlib.sha256(body).hexdigest()

    def bind(self):return viewer.monitor_bindings(self.group,self.seal,self.digest)

    def test_all_original_arm_gpu_bindings(self):
        bindings = self.bind();self.assertEqual(set(bindings),set(viewer.GPUS))

    def test_checksum_change_fails(self):
        self.seal.write_bytes(self.seal.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.bind()

    def test_canonical_failed_monitor_cannot_be_selected(self):
        self.value['slots']['A']['monitor_path'] = str(self.group/'A/monitor/status.json');self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_missing_arm_is_rejected(self):
        del self.value['slots']['C'];self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_foreign_arm_monitor_is_rejected(self):
        self.value['slots']['A']['monitor_path'] = self.value['slots']['B']['monitor_path'];self.write()
        with self.assertRaises(ValueError):self.bind()

    def test_reader_routes_monitors_and_preserves_old_files(self):
        normal = viewer.original.read;calls=[]
        def read(path, *, required=False):calls.append(path);return {}
        def main(args):
            for arm in viewer.GPUS:
                viewer.original.read(self.group/arm/'monitor/status.json')
            self.assertEqual(args[-1], '--once')
        with patch.object(viewer.original,'read',side_effect=read):
            with patch.object(viewer.original,'main',side_effect=main):
                viewer.main(['--group',str(self.group),'--recovery-seal',str(self.seal),
                    '--recovery-seal-sha256',self.digest,'--arm','C','--once'])
        self.assertIs(viewer.original.read, normal)
        self.assertEqual(calls,[Path(self.value['slots'][arm]['monitor_path'])for arm in viewer.GPUS])
        for arm in viewer.GPUS:self.assertEqual((self.group/arm/'monitor/status.json').read_text(),'{"status":"FAILED"}')

    def test_reader_restored_on_observer_failure(self):
        normal = viewer.original.read
        with patch.object(viewer.original,'main',side_effect=RuntimeError('observer error')):
            with self.assertRaises(RuntimeError):
                viewer.main(['--group',str(self.group),'--recovery-seal',str(self.seal),
                    '--recovery-seal-sha256',self.digest,'--arm','D','--once'])
        self.assertIs(viewer.original.read,normal)


if __name__ == '__main__':unittest.main()
