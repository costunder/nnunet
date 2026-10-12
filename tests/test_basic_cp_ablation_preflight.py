"""CPU UNIT provenance fixtures; no image geometry, CUDA or training results."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_ablation_preflight as reuse
from tests.test_basic_cp_ablation_trainer import ROOT, source_text


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf8')


class CompletePreflightReuseCPUUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp'); cls.folder = Path(cls.temp.name)
        cls.oldroot = cls.folder / 'old'; cls.oldroot.mkdir()
        cls.bankroot = cls.folder / 'bank'; cls.bankroot.mkdir(); (cls.bankroot / 'entries').mkdir()
        cls.package = cls.oldroot / 'native/runtime/nnunetv2/training/nnUNetTrainer'; cls.package.mkdir(parents=True)
        cls.current = cls.folder / 'new_trainer.py'
        cls.current.write_bytes((ROOT / reuse.TRAINER_RELATIVE).read_bytes())
        current_tree = ast.parse(cls.current.read_text(encoding='utf8'))
        owner = next(n for n in current_tree.body if isinstance(n, ast.ClassDef) and n.name == '_BasicCPAblation')
        ctor = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        ctor.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], vararg=ast.arg(arg='args'),
                                 kwonlyargs=[], kw_defaults=[], kwarg=ast.arg(arg='kwargs'), defaults=[])
        supercall = next(n for n in ast.walk(ctor) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'super().__init__')
        supercall.args = [ast.Starred(value=ast.Name(id='args', ctx=ast.Load()), ctx=ast.Load())]
        supercall.keywords = [ast.keyword(arg=None, value=ast.Name(id='kwargs', ctx=ast.Load()))]
        cls.oldtrainer = cls.package / 'nnUNetTrainer_BasicCPAblations.py'
        cls.oldtrainer.write_text(ast.unparse(current_tree), encoding='utf8')
        (cls.package / 'nnUNetTrainer_OnlinePairedCP.py').write_bytes(source_text().encode('utf8'))
        cls.ids = [f'CPU_UNIT_case_{n:03d}' for n in range(105)]
        cls.source_cases = {}; cls.checksums = {}
        for case_index, case in enumerate(cls.ids[:81]):
            names = []
            for index in range(7 if case_index < 6 else 8):
                name = f'entries/{case}_{index}.npz'; names.append(name)
                path = cls.bankroot / name
                path.write_bytes(f'CPU_UNIT_HASH_ONLY_NOT_CT_{case}_{index}'.encode())
                cls.checksums[name] = reuse.sha(path)
            cls.source_cases[case] = names
        cls.bankpath = cls.bankroot / 'index.json'
        write_json(cls.bankpath, {'entries_by_case': cls.source_cases, 'candidate_count': 128,
                                 'total_candidates': 82176})
        cls.banksha = reuse.sha(cls.bankpath)
        cls.admission = cls.folder / 'readonly_admission.json'
        write_json(cls.admission, {'CPU_UNIT_provenance_only': True})
        cls.request = {'arm': 'multi3', 'gpu': 1, 'epochs': 250, 'train_cases': 105, 'validation_cases': 26,
            'physical_batch': 2, 'checkpoint': 'checkpoint_best.pth', 'sources': 642,
            'candidates_per_source': 128, 'copies_per_CP_event': 3, 'blend_border_preprocessed_voxels': 0,
            'readonly_admission': str(cls.admission), 'readonly_admission_sha256': reuse.sha(cls.admission),
            'code_sha256': {reuse.TRAINER_RELATIVE: reuse.sha(cls.oldtrainer)}}
        cls.native = {'arm': 'multi3', 'private_runtime': str(cls.package.parents[2]),
            'bank': str(cls.bankpath), 'bank_sha256': cls.banksha,
            'split': {'outer_train': cls.ids, 'outer_val': [f'CPU_UNIT_val_{n}' for n in range(26)]}}
        cls.result = {'format': reuse.FORMAT, 'arm': 'multi3', 'complete': True, 'index_sha256': cls.banksha,
            'historical_trainer_sha256': reuse.HISTORICAL_SHA, 'historical_bank_sha256': cls.banksha,
            'source_entries': 642, 'eligible_cases': 81, 'full_training_cases': 105,
            'full_validation_cases': 26, 'candidate_count': 128, 'validated_positions': 82176,
            'three_paste_positions': 82176, 'CP_probability': .5, 'copies_per_event': 3,
            'blend_border_preprocessed_voxels': 0, 'original_gate_primary_RNG_schedule_preserved': True,
            'GNN_scores_used': False, 'physical_batch_size': 2, 'seed': 42, 'epochs': 250,
            'network_patch_size': [128] * 3, 'training_executed': False,
            'actual_training_identifiers': cls.ids, 'entries_sha256': cls.checksums,
            'case_receipts': [{'case_id': case, 'source_entries': len(names),
                'validated_positions': len(names) * 128, 'three_paste_positions': len(names) * 128,
                'entries_sha256': {name: cls.checksums[name] for name in names}}
                              for case, names in cls.source_cases.items()]}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.current.write_bytes((ROOT / reuse.TRAINER_RELATIVE).read_bytes())
        write_json(self.oldroot / 'request.json', self.request)
        write_json(self.oldroot / 'native.json', self.native)
        write_json(self.oldroot / 'preflight.json', self.result)
        self.bankpatch = patch.object(reuse, 'BANK_SHA', self.banksha); self.bankpatch.start()

    def tearDown(self):
        self.bankpatch.stop()

    def verify(self, **kwargs):
        args = dict(oldroot=self.oldroot, oldrequestSHA=reuse.sha(self.oldroot / 'request.json'),
            currenttrainerpath=self.current, bankpath=self.bankpath, bankSHA=self.banksha,
            arm='multi3', expectedtrainids105=self.ids)
        args.update(kwargs)
        return reuse.verify_preflight_reuse(**args)

    def test_complete642_source_SHA_case_receipts_and_constructor_only_change_reuse(self):
        before = {p: reuse.sha(p) for p in self.oldroot.rglob('*') if p.is_file()}
        result = self.verify(readonly_admission=self.admission, readonly_admission_sha256=reuse.sha(self.admission))
        proof = result.pop('reuse_proof')
        self.assertEqual(result, self.result)
        self.assertEqual(proof['full_source_SHA_reverified'], 642)
        self.assertFalse(proof['geometry_recomputed']); self.assertFalse(proof['source_files_written'])
        self.assertEqual(before, {p: reuse.sha(p) for p in self.oldroot.rglob('*') if p.is_file()})

    def test_wrong_request_SHA_and_installed_old_source_SHA_rejected(self):
        with self.assertRaisesRegex(ValueError, 'request SHA'):
            self.verify(oldrequestSHA='0' * 64)
        changed = copy.deepcopy(self.request); changed['code_sha256'][reuse.TRAINER_RELATIVE] = '0' * 64
        write_json(self.oldroot / 'request.json', changed)
        with self.assertRaisesRegex(ValueError, 'installed wrapper SHA'):
            self.verify()

    def test_change_outside_constructor_or_inside_scientific_constructor_body_rejected(self):
        original = self.current.read_text(encoding='utf8')
        for changed in [original.replace("ARMS = ('multi3', 'candidates256', 'blend')", "ARMS = ('blend',)"),
                        original.replace('self._ablation_pasted_lesions = 0', 'self._ablation_pasted_lesions = 1')]:
            self.current.write_text(changed, encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'Scientific module changed'):
                self.verify()

    def test_partial_overall_or_patient_triple_receipt_rejected(self):
        for field, wrong in [('complete', False), ('validated_positions', 82175), ('three_paste_positions', 82175),
                             ('original_gate_primary_RNG_schedule_preserved', False), ('GNN_scores_used', True)]:
            changed = copy.deepcopy(self.result); changed[field] = wrong
            write_json(self.oldroot / 'preflight.json', changed)
            with self.assertRaises(ValueError):
                self.verify()
        changed = copy.deepcopy(self.result); changed['case_receipts'][0]['three_paste_positions'] -= 1
        write_json(self.oldroot / 'preflight.json', changed)
        with self.assertRaisesRegex(ValueError, 'Patient receipt'):
            self.verify()

    def test_missing_source_or_partial_patient_population_rejected(self):
        changed = copy.deepcopy(self.result); changed['entries_sha256'].pop(next(iter(changed['entries_sha256'])))
        write_json(self.oldroot / 'preflight.json', changed)
        with self.assertRaisesRegex(ValueError, 'Every actual642'):
            self.verify()
        changed = copy.deepcopy(self.result); changed['case_receipts'] = changed['case_receipts'][:-1]
        write_json(self.oldroot / 'preflight.json', changed)
        with self.assertRaisesRegex(ValueError, 'All81'):
            self.verify()

    def test_actual_NPZ_file_bytes_changed_after_preflight_rejected(self):
        name = next(iter(self.checksums)); path = self.bankroot / name; original = path.read_bytes()
        try:
            path.write_bytes(original + b'CHANGED')
            with self.assertRaisesRegex(ValueError, 'Actual source bytes changed'):
                self.verify()
        finally:
            path.write_bytes(original)

    def test_current105_population_or_readonly_admission_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Full original105/26'):
            self.verify(expectedtrainids105=self.ids[:-1] + ['FOREIGN_PATIENT'])
        with self.assertRaisesRegex(ValueError, 'All105'):
            self.verify(expectedtrainids105=self.ids[:-1])
        with self.assertRaisesRegex(ValueError, 'admissions differ'):
            self.verify(readonly_admission=self.admission, readonly_admission_sha256='0' * 64)


if __name__ == '__main__':
    unittest.main()
