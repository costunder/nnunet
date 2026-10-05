"""Synthetic receipt UNIT checks; no neural/CT quality claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_competition_analysis import unit_report
from tools.analyze_full128_competition import analyze_summary, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def fixture(root):
    request = dict(format='historical_V1_A_B_C_full128_readonly_evaluation_v1',
        debug=True, training_started=False, UNIT_synthetic_receipts=True)
    write(root/'request.json', request)
    summary = dict(format=request['format'], complete=True, debug=True,
        full_evaluation=False, training_started=False, same_native_query_cohort=True, arms={})
    for arm in ('V1', 'A', 'B', 'C'):
        report = unit_report()
        report.update(arm=arm, actual_CUDA=True, execution_contract_bound=True,
            old_evidence_preserved=True, training_started=False, optimizer_updates=0,
            checkpoint={'UNIT_only': True}, evaluation_request_sha256=canonical(request))
        report['cohort']['cohort_sha256'] = 'a'*64
        path = root/arm/'report.json'
        write(path, report)
        write(path.with_name('complete.json'), dict(arm=arm,
            request_sha256=canonical(request), report_sha256=sha(path)))
        summary['arms'][arm] = dict(report=str(path), checkpoint=report['checkpoint'],
            metrics=report['metrics'], denominators=report['denominators'])
    write(root/'summary.json', summary)
    return root/'summary.json'


class CompleteSavedReceiptUnit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.summary = fixture(self.root)

    def test_valid_receipts_are_read_only(self):
        before = {str(p):sha(p) for p in self.root.rglob('*.json')}
        report = analyze_summary(self.summary, debug=True)
        self.assertEqual(report['original_files_sha256'], before)
        self.assertEqual(report['arms']['V1']['curve'], report['arms']['C']['curve'])
        self.assertFalse(report['neural_forward_executed'])
        self.assertFalse(report['quality_verified'])
        self.assertEqual({str(p):sha(p) for p in self.root.rglob('*.json')}, before)

    def test_rejects_missing_completion_seal(self):
        (self.root/'A/complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            analyze_summary(self.summary, debug=True)

    def test_rejects_tampered_report(self):
        path = self.root/'B/report.json'
        report = json.loads(path.read_text())
        report['cases'][0]['case_scores'][0]['score'] = 100.0
        write(path, report)
        with self.assertRaisesRegex(ValueError, 'receipt differ'):
            analyze_summary(self.summary, debug=True)

    def test_rejects_different_original_request(self):
        path = self.root/'request.json'
        value = json.loads(path.read_text())
        value['changed_after_evaluation'] = True
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'receipt differ'):
            analyze_summary(self.summary, debug=True)

    def test_debug_cannot_be_silently_promoted(self):
        with self.assertRaisesRegex(ValueError, 'DEBUG identity'):
            analyze_summary(self.summary)

    def test_rejects_outside_report_path(self):
        value = json.loads(self.summary.read_text())
        value['arms']['V1']['report'] = str(self.root/'A/report.json')
        write(self.summary, value)
        with self.assertRaisesRegex(ValueError, 'outside'):
            analyze_summary(self.summary, debug=True)

    def test_rejects_stale_summary_denominator(self):
        value = json.loads(self.summary.read_text())
        value['arms']['A']['denominators']['cases'] = 21
        write(self.summary, value)
        with self.assertRaisesRegex(ValueError, 'receipt differ'):
            analyze_summary(self.summary, debug=True)


if __name__ == '__main__':
    unittest.main()
