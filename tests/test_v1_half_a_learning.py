"""CPU-only checks of recorded actual-CT/CUDA Half-A replay contracts.

No model, medical fixture or synthetic predictions are generated in these tests.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from hiercp_v1x.contracts import canonical_hash
from tools.verify_v1_half_a_learning import compare_curves, verify_bounded_reference


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'validation/v14_matched_learning_20261004'


class HalfALearningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded = {
            'bounded': json.loads((EVIDENCE/'matched10mm/report.json').read_text(encoding='utf8')),
            'contract': json.loads((EVIDENCE/'matched10mm/execution_contract.json').read_text(encoding='utf8')),
            'native': json.loads((EVIDENCE/'native_reference/report.json').read_text(encoding='utf8')),
            'native_contract': json.loads((EVIDENCE/'native_reference/execution_contract.json').read_text(encoding='utf8')),
            'manifest': json.loads((EVIDENCE/'native_fixture/fixture_manifest.json').read_text(encoding='utf8')),
        }

    def setUp(self):
        self.inputs = deepcopy(self.recorded)

    def verify(self):
        return verify_bounded_reference(self.inputs['bounded'], self.inputs['contract'],
            self.inputs['native'], self.inputs['native_contract'], self.inputs['manifest'])

    def test_recorded_exact_protocol_accepted_without_mutation(self):
        before = deepcopy(self.inputs)
        protocol = self.verify()
        self.assertEqual(protocol['reference_updates'], 2)
        self.assertEqual(protocol['optimization_curriculum_epoch'], 29)
        self.assertEqual(protocol['physical_sample_batch'], 2)
        self.assertEqual(self.inputs, before)

    def test_unverified_or_quality_claiming_reference_rejected(self):
        for field in ('debug', 'actual_CT', 'actual_CUDA', 'finite_complete_updates',
                      'original_fixture_preserved', 'original_reference_preserved', 'original_source_preserved'):
            with self.subTest(field=field):
                self.inputs = deepcopy(self.recorded)
                self.inputs['bounded'][field] = False
                with self.assertRaises(ValueError):
                    self.verify()
        for field in ('full_training', 'full_evaluation', 'quality_verified', 'production_ready',
                      'checkpoint_written', 'native_retrained'):
            with self.subTest(field=field):
                self.inputs = deepcopy(self.recorded)
                self.inputs['bounded'][field] = True
                with self.assertRaises(ValueError):
                    self.verify()

    def test_configuration_gt_fixture_and_state_identity_preserved(self):
        changes = (
            ('margin_mm', 20.0), ('fixture_sha256', '0'*64),
            ('initial_neural_sha256', '0'*64), ('protocol', {}),
        )
        for field, value in changes:
            with self.subTest(field=field):
                self.inputs = deepcopy(self.recorded)
                self.inputs['bounded'][field] = value
                with self.assertRaises(ValueError):
                    self.verify()
        self.inputs = deepcopy(self.recorded)
        self.inputs['bounded']['configuration']['training']['epochs'] = 8
        with self.assertRaises(ValueError):
            self.verify()

    def test_contract_tampering_rejected(self):
        self.inputs['contract']['arguments']['updates'] = 8
        with self.assertRaises(ValueError):
            self.verify()
        # Even a self-consistent edit must agree with completed run evidence.
        self.inputs['contract']['identity_sha256'] = canonical_hash(
            {key:value for key,value in self.inputs['contract'].items() if key != 'identity_sha256'})
        with self.assertRaises(ValueError):
            self.verify()

    def test_only_complete_successful_update_evidence_accepted(self):
        changes = ('wrong_count', 'nonfinite_loss', 'skipped_as_update', 'missing_initial_curve')
        for change in changes:
            with self.subTest(change=change):
                self.inputs = deepcopy(self.recorded)
                bounded = self.inputs['bounded']
                if change == 'wrong_count':
                    bounded['completed_optimizer_updates'] -= 1
                elif change == 'nonfinite_loss':
                    bounded['updates'][0]['loss'] = float('inf')
                elif change == 'skipped_as_update':
                    bounded['updates'][0]['gradient_observation']['finite'] = False
                else:
                    bounded['curve'] = bounded['curve'][1:]
                with self.assertRaises(ValueError):
                    self.verify()

    def test_nonfinite_or_missing_eight_candidate_scores_rejected(self):
        for invalid in ('score', 'margin', 'count'):
            with self.subTest(invalid=invalid):
                self.inputs = deepcopy(self.recorded)
                metric = self.inputs['bounded']['curve'][0]['validation']
                if invalid == 'score':
                    metric['scores'][0][0] = float('nan')
                elif invalid == 'margin':
                    metric['mean_margin'] = float('inf')
                else:
                    metric['scores'][0].pop()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_eight_update_run_compares_matching_steps_only(self):
        old = self.recorded['bounded']['curve']
        short = [deepcopy(row) for row in old if row['step'] <= 8]
        before = deepcopy(short)
        comparisons = compare_curves(short, old)
        self.assertEqual([row['step'] for row in comparisons], [0, 2, 4, 8])
        self.assertNotIn(16, [row['step'] for row in comparisons])
        for row in comparisons:
            self.assertFalse(row['loss_comparison_available'])
            self.assertTrue(all(value == 0 for split in row['difference'].values() for value in split.values()))
        self.assertEqual(short, before)


if __name__ == '__main__':
    unittest.main()
