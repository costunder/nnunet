"""CPU metadata/synthetic-score UNITs for exact held-out case recording.

Scores here are explicit UNIT values, not model predictions or CT evidence.
No neural module, CUDA context, loss, optimizer or quality metric is executed.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x.half_a_case_metrics import (
    CaseScoreRecorder, FORMAT, install_case_score_recorder,
)


ROOT = Path(__file__).resolve().parents[1]


def batch(cases):
    return SimpleNamespace(case_ids=list(cases), counts=[8]*len(cases))


def scores(count, *, dtype=torch.float32):
    return SimpleNamespace(scores=[torch.arange(8, dtype=dtype)+index for index in range(count)])


class CaseScoreRecorderUnit(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='half_a_case_scores_CPU_UNIT_', dir=ROOT/'work')
        self.root = Path(self.directory.name)
        self.session = SimpleNamespace(_epoch=None)
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            if hasattr(recorder, 'restore'):
                recorder.restore()
            recorder.close()
        self.directory.cleanup()

    def recorder(self, expected=('case_A_UNIT', 'case_B_UNIT'), count=2):
        recorder = CaseScoreRecorder(self.root, self.session, expected_case_ids=list(expected),
            expected_samples=count, invocation_id='CPU_UNIT_invocation')
        self.recorders.append(recorder)
        return recorder

    @staticmethod
    def read(recorder):
        return [json.loads(line) for line in recorder.path.read_text(encoding='utf8').splitlines()]

    def test_exact_original_scores_rows_and_pass_totals_without_mutation(self):
        recorder = self.recorder(count=4)
        output = scores(4)
        before = [value.clone() for value in output.scores]
        output.scores[0].requires_grad_()
        self.assertIsNone(recorder.record(batch(['case_A_UNIT', 'case_B_UNIT']*2), output))
        rows = self.read(recorder)
        self.assertEqual(len(rows), 4)
        self.assertEqual(FORMAT, 'hiercp_half_a_validation_case_scores_v1')
        for row, original, case in zip(rows, before, ['case_A_UNIT', 'case_B_UNIT']*2):
            self.assertEqual(row['format'], FORMAT)
            self.assertEqual(row['invocation_id'], 'CPU_UNIT_invocation')
            self.assertEqual(row['epoch'], 0)
            self.assertEqual(row['case_id'], case)
            self.assertEqual(row['gt_index'], 0)
            self.assertEqual(row['candidate_count'], 8)
            self.assertEqual(row['scores'], original.tolist())
        self.assertEqual([row['sample_ordinal_in_case_epoch'] for row in rows], [1, 1, 2, 2])
        self.assertTrue(all(torch.equal(value, old) for value, old in zip(output.scores, before)))
        self.assertTrue(output.scores[0].requires_grad)
        totals = recorder.totals()
        self.assertEqual(totals['expected_case_ids'], ['case_A_UNIT', 'case_B_UNIT'])
        self.assertEqual(totals['expected_samples'], 4)
        self.assertEqual(totals['epochs']['0'], dict(samples=4,
            cases=['case_A_UNIT', 'case_B_UNIT'],
            case_sample_counts={'case_A_UNIT':2, 'case_B_UNIT':2}, complete=True))

    def test_partial_pass_blocks_epoch_change_then_completed_epochs_reset_ordinals(self):
        recorder = self.recorder()
        recorder.record(batch(['case_A_UNIT']), scores(1))
        partial = recorder.totals()
        self.assertFalse(partial['epochs']['0']['complete'])
        self.session._epoch = 1
        with self.assertRaises(ValueError):
            recorder.record(batch(['case_B_UNIT']), scores(1))
        self.assertEqual(recorder.totals(), partial)
        self.session._epoch = None
        recorder.record(batch(['case_B_UNIT']), scores(1))
        self.session._epoch = 1
        recorder.record(batch(['case_B_UNIT', 'case_A_UNIT']), scores(2))
        totals = recorder.totals()
        self.assertEqual(totals['epochs']['0']['samples'], 2)
        self.assertTrue(totals['epochs']['0']['complete'])
        self.assertTrue(totals['epochs']['1']['complete'])
        self.assertEqual([row['epoch'] for row in self.read(recorder)], [0, 0, 1, 1])
        self.assertEqual([row['sample_ordinal_in_case_epoch'] for row in self.read(recorder)], [1, 1, 1, 1])

    def test_extra_sample_and_wrong_final_cohort_are_rejected(self):
        recorder = self.recorder()
        recorder.record(batch(['case_A_UNIT']), scores(1))
        before = recorder.path.read_bytes()
        with self.assertRaises(ValueError):
            recorder.record(batch(['case_A_UNIT']), scores(1))
        self.assertEqual(recorder.path.read_bytes(), before)
        recorder.record(batch(['case_B_UNIT']), scores(1))
        complete = recorder.path.read_bytes()
        with self.assertRaises(ValueError):
            recorder.record(batch(['case_A_UNIT']), scores(1))
        self.assertEqual(recorder.path.read_bytes(), complete)
        self.assertEqual(recorder.totals()['epochs']['0']['samples'], 2)

    def test_bad_batch_metadata_is_rejected_before_score_publication(self):
        recorder = self.recorder()
        invalid = (
            SimpleNamespace(case_ids=['case_A_UNIT'], counts=[7]),
            SimpleNamespace(case_ids=['case_A_UNIT'], counts=[8, 8]),
            SimpleNamespace(case_ids=['other_case_UNIT'], counts=[8]),
        )
        for value in invalid:
            with self.subTest(batch=value), self.assertRaises(ValueError):
                recorder.record(value, scores(1))
        self.assertEqual(self.read(recorder), [])

    def test_wrong_candidate_shape_nonfinite_and_dtype_are_rejected(self):
        recorder = self.recorder()
        invalid = (
            SimpleNamespace(scores=[]),
            SimpleNamespace(scores=[torch.zeros(7)]),
            SimpleNamespace(scores=[torch.zeros(1, 8)]),
            SimpleNamespace(scores=[torch.zeros(8, dtype=torch.long)]),
            SimpleNamespace(scores=[torch.tensor([float('nan')]+[0.]*7)]),
            SimpleNamespace(scores=[torch.tensor([float('inf')]+[0.]*7)]),
        )
        for value in invalid:
            with self.subTest(output=value), self.assertRaises(ValueError):
                recorder.record(batch(['case_A_UNIT']), value)
        self.assertEqual(self.read(recorder), [])

    def test_mixed_score_device_or_dtype_is_rejected_without_cuda(self):
        recorder = self.recorder()
        for invalid in (
            SimpleNamespace(scores=[torch.zeros(8), torch.zeros(8, dtype=torch.float64)]),
            SimpleNamespace(scores=[torch.zeros(8), torch.empty(8, device='meta')]),
        ):
            with self.subTest(output=invalid), self.assertRaises(ValueError):
                recorder.record(batch(['case_A_UNIT', 'case_B_UNIT']), invalid)
        self.assertEqual(self.read(recorder), [])

    def test_bad_expected_cohort_or_count_never_defaults(self):
        for cases, count in (([], 2), (['case_A_UNIT', 'case_A_UNIT'], 2),
                             (['case_A_UNIT'], 0), (['case_A_UNIT'], True)):
            with self.subTest(cases=cases, count=count), self.assertRaises(ValueError):
                CaseScoreRecorder(self.root, self.session, expected_case_ids=cases, expected_samples=count)

    def test_epoch_metadata_and_closed_recorder_are_explicit_errors(self):
        recorder = self.recorder()
        for value in (0, -1, True, '1', 1.0):
            self.session._epoch = value
            with self.subTest(epoch=value), self.assertRaises(ValueError):
                recorder.record(batch(['case_A_UNIT']), scores(1))
        self.session._epoch = None
        recorder.record(batch(['case_A_UNIT']), scores(1))
        recorder.close()
        self.assertFalse(recorder.totals()['epochs']['0']['complete'])
        with self.assertRaises(ValueError):
            recorder.record(batch(['case_B_UNIT']), scores(1))

    def test_one_detached_stacked_cpu_transfer_per_batch(self):
        recorder = self.recorder()
        output = scores(2)
        shapes = []
        original = torch.Tensor.cpu
        def copied(value, *args, **kwargs):
            shapes.append(tuple(value.shape))
            self.assertFalse(value.requires_grad)
            return original(value, *args, **kwargs)
        output.scores[0].requires_grad_()
        with patch.object(torch.Tensor, 'cpu', copied):
            recorder.record(batch(['case_A_UNIT', 'case_B_UNIT']), output)
        self.assertEqual(shapes, [(2, 8)])

    def test_forward_wrapper_calls_original_once_and_records_only_eval(self):
        session = self.session
        value = scores(2)
        calls = []
        class MetadataModel:
            def __init__(self):
                self.training = True
            def forward(self, payload, *args, **kwargs):
                calls.append((payload, args, kwargs))
                return value
        recorder = install_case_score_recorder(MetadataModel, session, self.root,
            expected_case_ids=['case_A_UNIT', 'case_B_UNIT'], expected_samples=2)
        self.recorders.append(recorder)
        model = MetadataModel()
        payload = batch(['case_A_UNIT', 'case_B_UNIT'])
        self.assertIs(model.forward(payload, 'argument_UNIT', keyword='UNIT'), value)
        self.assertEqual(calls, [(payload, ('argument_UNIT',), {'keyword':'UNIT'})])
        self.assertEqual(self.read(recorder), [])
        model.training = False
        self.assertIs(model.forward(payload), value)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1], (payload, (), {}))
        self.assertEqual(len(self.read(recorder)), 2)
        self.assertTrue(recorder.totals()['epochs']['0']['complete'])
        with self.assertRaises(ValueError):
            install_case_score_recorder(MetadataModel, session, self.root,
                expected_case_ids=['case_A_UNIT', 'case_B_UNIT'], expected_samples=2)


if __name__ == '__main__':
    unittest.main()
