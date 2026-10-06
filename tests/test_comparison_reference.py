"""UNIT: synthetic execution metadata only; no CT, neural run, or checkpoint."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.v19_reference_execution import reference_execution


class ReferenceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v19_reference_UNIT_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = dict(format='v18_u_bridge_matched_experiment_v1',
                               debug=False, epochs=40, workers=16,
                               unit_scope='synthetic metadata, not medical/model evidence')
        self.calibration = dict(physical_batch=2, accepted=True, arm='matched_both',
                                explicit_candidates=[1, 2, 4], accepted_common=[1, 2], reports={})
        for arm in ('selected', 'native'):
            self.calibration['reports'][arm] = dict(arm=arm, original_model_and_RNG_preserved=True,
                initial_state_sha256='a' * 64, reports=[dict(physical_batch=2, accepted=True,
                    actual_sample_indices=[5, 3], samples_per_second=2.5)])
        self.publish()

    def publish(self):
        content = {key: value for key, value in self.experiment.items() if key != 'sha256'}
        self.experiment['sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.calibration['contract_sha256'] = self.experiment['sha256']
        self.write('experiment.json', self.experiment)
        self.write('calibration.json', self.calibration)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding='utf8')

    def invalid(self):
        self.write('calibration.json', self.calibration)
        with self.assertRaises(ValueError):
            reference_execution(self.root)

    def test_valid_read_is_immutable_even_when_active(self):
        (self.root / '.pipeline.lock').write_text('{"active": true}', encoding='utf8')
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.iterdir()}
        receipt = reference_execution(self.root)
        self.assertEqual((receipt['physical_batch'], receipt['workers']), (2, 16))
        self.assertEqual(receipt['calibration_initial_state_sha256'], 'a' * 64)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.iterdir()}
        self.assertEqual(before, after)

    def test_cli_only_prints_batch_and_workers(self):
        script = Path(__file__).resolve().parents[1] / 'tools/v19_reference_execution.py'
        command = [sys.executable, '-B', str(script), '--reference', str(self.root)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, '2 16\n')
        self.assertEqual(result.stderr, '')
        receipt = json.loads(subprocess.run(command + ['--json'], check=True,
            capture_output=True, text=True).stdout)
        self.assertEqual(receipt['contract_sha256'], self.experiment['sha256'])

    def test_tampered_contract_rejected(self):
        changed = dict(self.experiment, workers=8)
        self.write('experiment.json', changed)
        with self.assertRaises(ValueError):
            reference_execution(self.root)

    def test_wrong_experiment_format_rejected(self):
        self.experiment['format'] = 'other'
        self.publish()
        with self.assertRaises(ValueError):
            reference_execution(self.root)

    def test_debug_or_short_epoch_reference_rejected(self):
        for field, value in (('debug', True), ('epochs', 2), ('epochs', True), ('workers', 1), ('workers', True)):
            with self.subTest(field=field, value=value):
                old = copy.deepcopy(self.experiment)
                self.experiment[field] = value
                self.publish()
                with self.assertRaises(ValueError):
                    reference_execution(self.root)
                self.experiment = old

    def test_wrong_calibration_binding_rejected(self):
        self.calibration['contract_sha256'] = 'b' * 64
        self.invalid()

    def test_unsupported_batch_or_acceptance_rejected(self):
        for field, value in (('physical_batch', 8), ('physical_batch', True),
                             ('accepted_common', [1]), ('explicit_candidates', [1]),
                             ('accepted', False), ('arm', 'selected')):
            with self.subTest(field=field, value=value):
                old = copy.deepcopy(self.calibration)
                self.calibration[field] = value
                self.invalid()
                self.calibration = old

    def test_missing_or_unaccepted_actual_measurement_rejected(self):
        for mutation in ('missing', 'rejected', 'duplicate'):
            with self.subTest(mutation=mutation):
                old = copy.deepcopy(self.calibration)
                report = self.calibration['reports']['native']
                if mutation == 'missing': report['reports'] = []
                elif mutation == 'rejected': report['reports'][0]['accepted'] = False
                else: report['reports'].append(copy.deepcopy(report['reports'][0]))
                self.invalid()
                self.calibration = old

    def test_source_count_duplicates_or_bad_index_rejected(self):
        for ids in ([5], [5, 5], [5, True], [5, -1]):
            with self.subTest(ids=ids):
                self.calibration['reports']['native']['reports'][0]['actual_sample_indices'] = ids
                self.invalid()

    def test_measured_batch_is_an_integer(self):
        self.calibration['reports']['native']['reports'][0]['physical_batch'] = 2.0
        self.invalid()

    def test_nonfinite_or_missing_throughput_rejected(self):
        for value in (0, -1, True, None, float('nan'), float('inf')):
            with self.subTest(value=value):
                self.calibration['reports']['native']['reports'][0]['samples_per_second'] = value
                self.invalid()

    def test_state_mismatch_rejected(self):
        self.calibration['reports']['native']['initial_state_sha256'] = 'b' * 64
        self.invalid()

    def test_state_hash_provenance_and_arm_names_rejected(self):
        for key, value in (('initial_state_sha256', 'invalid'), ('arm', 'selected'),
                           ('original_model_and_RNG_preserved', False)):
            with self.subTest(key=key):
                old = copy.deepcopy(self.calibration)
                self.calibration['reports']['native'][key] = value
                self.invalid()
                self.calibration = old

    def test_matched_source_ids_required(self):
        self.calibration['reports']['native']['reports'][0]['actual_sample_indices'] = [3, 5]
        self.invalid()

    def test_missing_file_is_not_silently_defaulted(self):
        (self.root / 'calibration.json').unlink()
        with self.assertRaises(FileNotFoundError):
            reference_execution(self.root)


if __name__ == '__main__':
    unittest.main()
