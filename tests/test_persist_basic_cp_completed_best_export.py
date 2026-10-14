"""Independent CPU DEBUG provenance tests for an already completed BEST26.

Real temporary file bytes and SHA admissions are used. Tiny checkpoint decoder
adapters are explicitly DEBUG; these tests do not perform clinical inference.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import persist_basic_cp_seven as durable
from tests import test_persist_basic_cp_code_exports as code_fixture
from tests.test_persist_basic_cp_seven import debug_torch


class CompletedBestOriginalExportTests(unittest.TestCase):
    def setUp(self):
        code_fixture.ControllerCodeExportDEBUGTests.setUp(self)
        self.validator = 'hiercp_v1x/v24_native_best_eval_runtime.py'
        actual = Path(durable.__file__).resolve().parents[1] / self.validator
        (self.export / self.validator).write_bytes(actual.read_bytes())
        self.current = self.root / 'new_group/code'
        (self.current / 'hiercp_v1x').mkdir(parents=True)
        (self.current / self.validator).write_bytes(actual.read_bytes())
        pins = {p.relative_to(self.export).as_posix(): durable.digest(p)
                for p in self.export.rglob('*.py')}
        self.proof.write_text(json.dumps(dict(code=str(self.export), commit='a'*40,
            code_sha256=pins)), encoding='utf8')
        self.native = dict(root=str(self.arm), native_root=str(self.root/'original_native'),
            trainer='DEBUG_NATIVE', baseline=dict(dataset_name='DEBUG_DATA', plans_name='DEBUG_PLANS'),
            bank_sha256='b'*64, split={'outer_val':['DEBUG_CASE_'+str(i) for i in range(26)]})
        self.goal = durable.fold(self.native)
        self.goal.mkdir(parents=True)
        self.receipt = self.goal / 'best_validation_complete.json'
        self.receipt.write_text(json.dumps({'DEBUG_ORIGINAL_RECEIPT': True}), encoding='utf8')
        self.native['RAM512_completed_best_validator'] = dict(code=str(self.export),
            export=dict(path=str(self.proof), sha256=durable.digest(self.proof)),
            receipt=dict(path=str(self.receipt), sha256=durable.digest(self.receipt)))

    def tearDown(self):
        code_fixture.ControllerCodeExportDEBUGTests.tearDown(self)

    def test_same_scientific_bytes_reuse_original_export_and_receipt_unchanged(self):
        before = self.receipt.read_bytes()
        self.assertEqual(durable.completed_best_code(self.native, self.current), self.export)
        self.assertEqual(self.receipt.read_bytes(), before)
        self.assertNotEqual(self.current, self.export)

    def test_changed_pinned_receipt_is_fatal(self):
        self.receipt.write_bytes(b'{"DEBUG_MUTATED_RECEIPT": true}')
        with self.assertRaisesRegex(ValueError, 'evidence checksum changed'):
            durable.completed_best_code(self.native, self.current)

    def test_changed_pinned_export_is_fatal(self):
        self.proof.write_bytes(b'{"DEBUG_CHANGED_EXPORT": true}')
        with self.assertRaisesRegex(ValueError, 'evidence checksum changed'):
            durable.completed_best_code(self.native, self.current)

    def test_secondary_source_mutation_is_fatal_despite_unchanged_descriptor(self):
        (self.export / 'hiercp_v1x/DEBUG_state.py').write_bytes(b'DEBUG_ONLY = False\n')
        with self.assertRaisesRegex(ValueError, 'source export changed'):
            durable.completed_best_code(self.native, self.current)

    def test_new_validator_scientific_change_is_fatal(self):
        (self.current / self.validator).write_bytes(b'# DEBUG different science\n')
        with self.assertRaisesRegex(ValueError, 'scientific validator changed'):
            durable.completed_best_code(self.native, self.current)

    def test_sealed_receipt_in_another_fold_is_not_admitted(self):
        other = self.root / 'DEBUG_OTHER_FOLD'
        other.mkdir()
        receipt = other / self.receipt.name
        receipt.write_bytes(self.receipt.read_bytes())
        native = copy.deepcopy(self.native)
        native['RAM512_completed_best_validator']['receipt'] = dict(path=str(receipt),
            sha256=durable.digest(receipt))
        with self.assertRaisesRegex(ValueError, 'different original fold'):
            durable.completed_best_code(native, self.current)

    def test_unpinned_or_misnamed_original_export_is_rejected(self):
        for change in ('missing_validator', 'wrong_code', 'parent_path'):
            proof = durable.read(self.proof)
            if change == 'missing_validator': proof['code_sha256'].pop(self.validator)
            elif change == 'wrong_code': proof['code'] = str(self.current)
            else: proof['code_sha256']['../export.json'] = durable.digest(self.proof)
            native = copy.deepcopy(self.native)
            path = self.root / ('DEBUG_'+change+'_export.json')
            path.write_text(json.dumps(proof), encoding='utf8')
            native['RAM512_completed_best_validator']['export'] = dict(path=str(path),
                sha256=durable.digest(path))
            with self.subTest(change=change), self.assertRaises(ValueError):
                durable.completed_best_code(native, self.current)

    def test_descriptor_requires_current_scientific_export_without_fallback(self):
        with self.assertRaisesRegex(ValueError, 'Named original'):
            durable.completed_best_code(self.native, None)
        native = copy.deepcopy(self.native)
        native['RAM512_completed_best_validator']['unverified_fallback'] = True
        with self.assertRaisesRegex(ValueError, 'Named original'):
            durable.completed_best_code(native, self.current)

    def test_completed_checkpoint_selection_uses_original_validator_and_pinned_best(self):
        best = self.goal/'checkpoint_best.pth'; best.write_bytes(b'DEBUG_WHOLE_BEST_STATE')
        final = self.goal/'checkpoint_final.pth'; final.write_bytes(b'DEBUG_FULL250_STATE')
        proof = dict(epochs=250, checkpoint='checkpoint_best.pth', fold=str(self.goal),
            evaluation_checkpoint_is_final=False, best_sha256=durable.digest(best),
            final_training_witness_sha256=durable.digest(final))
        (self.arm/'training_and_BEST26_complete.json').write_text(json.dumps(proof), encoding='utf8')
        validation = {'checkpoint_sha256':durable.digest(best)}
        with patch.dict('sys.modules', torch=debug_torch(epoch=250)), \
                patch.object(durable, 'best_validator', return_value=lambda *a, **kw: validation) as dispatch:
            source, role, details = durable.select_checkpoint(self.native, unfinished='best', code=self.current)
        dispatch.assert_called_once_with(self.export)
        self.assertEqual(source, best)
        self.assertEqual(role, 'BEST_EVALUATED26')
        self.assertEqual(details['expected_sha256'], durable.digest(best))
        self.assertEqual(details['evaluated_cases'], 26)
        self.assertFalse(details['checkpoint_is_training_resume'])
        self.assertEqual(final.read_bytes(), b'DEBUG_FULL250_STATE')


if __name__ == '__main__':
    unittest.main()
