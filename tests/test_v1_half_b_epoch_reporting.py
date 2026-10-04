"""Whole-epoch resource bookkeeping UNIT checks; no neural/model execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from hiercp_v1x.half_b_entry import install_support_epoch_reporting


class HalfBWholeEpochReporting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = []
        self.session = SimpleNamespace(begin_epoch=lambda e: None,
            report=lambda original, name, payload: self.saved.append(payload))
        self.manager = SimpleNamespace(output=self.root, _generation=None, manifest_sha256='ab'*32)
        self.calls = 0
        def refresh(epoch, seconds, peak):
            self.calls += 1
            self.manager._generation = str(self.calls)
            value = dict(format='hiercp_half_B_full_training_support_refresh_v1',
                half_B_contract_sha256=self.manager.manifest_sha256, generation=str(self.calls),
                full_signed_training_cache=True, epoch=epoch, wall_seconds=seconds,
                cuda_peak_allocated_bytes=peak)
            (self.root/f'support_refresh_{self.calls}.json').write_text(json.dumps(value))
        self.manager.refresh = refresh
        install_support_epoch_reporting(self.session, self.manager)

    def tearDown(self):
        self.temp.cleanup()

    def report(self, epoch):
        self.session.report(None, 'EpochPostRun', dict(epoch=epoch,
            train={'cuda_peak_allocated_bytes':20}, validation={'cuda_peak_allocated_bytes':10}))
        return self.saved[-1]['half_B_epoch_resources']

    def test_initial_refresh_not_counted_again_in_epoch(self):
        self.manager.refresh(0, 100, 999)
        self.session.begin_epoch(1)
        self.manager.refresh(1, 3, 30)
        result = self.report(1)
        self.assertEqual(result['support_refresh_seconds'],3)
        self.assertEqual(result['whole_epoch_peak_vram_allocated_bytes'],30)
        self.assertEqual(result['support_refresh_count'],1)

    def test_resume_reconstruction_and_end_refresh_both_count(self):
        self.session.begin_epoch(4)
        self.manager.refresh(3, 7, 40)
        self.manager.refresh(4, 9, 30)
        result = self.report(4)
        self.assertEqual(result['support_refresh_seconds'],16)
        self.assertEqual(result['support_refresh_count'],2)
        self.assertEqual(result['whole_epoch_peak_vram_allocated_bytes'],40)

    def test_incomplete_epoch_cannot_report_complete_resources(self):
        self.session.begin_epoch(1)
        with self.assertRaises(ValueError):
            self.report(1)
        self.manager.refresh(0, 1, 30)
        with self.assertRaises(ValueError):
            self.report(1)


if __name__ == '__main__':
    unittest.main()
