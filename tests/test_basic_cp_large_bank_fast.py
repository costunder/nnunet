"""CPU DEBUG preservation/ordering fixtures; no clinical bank or train results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from hiercp_v1x import basic_cp_large_bank as base
from hiercp_v1x import basic_cp_large_bank_fast as fast
from tests.test_basic_cp_large_bank import source_fixture
from tests import test_basic_cp_expanded as expanded_fixture

ROOT = Path(__file__).resolve().parents[1]


class FastPartialPrefixTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / 'old'; self.destination = self.root / 'fresh'
        self.destination.mkdir(); self.original = source_fixture(128); self.admitted = source_fixture(256)
        self.relative = 'entries/DEBUG_case__component_1.npz'; self.partial = {'root': str(self.old)}

    def tearDown(self):
        self.temp.cleanup()

    def create(self, count, payload=None):
        payload = source_fixture(count) if payload is None else payload
        path = self.old / str(count) / self.relative; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: np.savez(stream, **payload)
        receipt = self.old / str(count) / 'entry_receipts' / Path(self.relative).with_suffix('.json')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        proof = base.historical.verify_preservation(self.original, payload, original_count=128, candidate_count=count)
        proof.update(admitted_prefix_count=256, all_admitted_prefix_centers_byte_preserved=True)
        row = dict(entry=self.relative, case_id='DEBUG', new_entry_sha256=base.file_sha(path), proof=proof)
        base.write_json_exclusive(receipt, row)
        return path, receipt

    def reuse(self, count):
        return fast._reuse_entry(self.partial, self.relative, count, self.original, self.admitted, self.destination)

    def test_all_three_partial_scales_copy_exact_npz_bytes_and_preserve_old_receipts(self):
        for count in base.COUNTS:
            path, receipt = self.create(count)
            before, receipt_before = path.read_bytes(), receipt.read_bytes()
            self.assertTrue(self.reuse(count))
            target = self.destination / str(count) / self.relative
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(path.read_bytes(), before); self.assertEqual(receipt.read_bytes(), receipt_before)
            result = base.read_entry(target)
            base.historical.verify_preservation(self.original, result, original_count=128, candidate_count=count)
            self.assertEqual(result['scores'].shape, (128,))
            np.testing.assert_array_equal(result['candidate_centers'][:256], self.admitted['candidate_centers'])
            row = base.read_json(self.destination / str(count) / 'entry_receipts' / Path(self.relative).with_suffix('.json'))
            self.assertEqual(row['partial_reuse']['path'], str(path))
            self.assertEqual(row['partial_reuse']['receipt_sha256'], base.file_sha(receipt))
            with self.assertRaises(FileExistsError): self.reuse(count)

    def test_missing_receipt_is_not_admitted_as_a_completed_partial(self):
        self.assertFalse(self.reuse(512))
        self.assertFalse((self.destination / '512' / self.relative).exists())

    def test_partial_npz_sha_tampering_fails_before_any_destination_copy(self):
        path, _ = self.create(512)
        with path.open('ab') as stream: stream.write(b'DEBUG_TAMPER')
        with self.assertRaisesRegex(base.BankContractError, 'partial receipt bytes'):
            self.reuse(512)
        self.assertFalse((self.destination / '512' / self.relative).exists())

    def test_changed_source_array_cannot_be_admitted_even_with_updated_partial_sha(self):
        path, receipt = self.create(512); payload = base.read_entry(path)
        payload['source_data'][0, 0, 0, 0] += 10
        with path.open('wb') as stream: np.savez(stream, **payload)
        row = base.read_json(receipt); row['new_entry_sha256'] = base.file_sha(path)
        receipt.write_text(json.dumps(row), encoding='utf8')
        with self.assertRaisesRegex(base.BankContractError, 'Original field/prefix bytes changed: source_data'):
            self.reuse(512)
        self.assertFalse((self.destination / '512' / self.relative).exists())

    def test_changed_completed256_prefix_fails_before_copy(self):
        self.create(512); self.admitted['candidate_centers'][200, 0] += 10000
        with self.assertRaisesRegex(base.BankContractError, 'completed256 prefix changed'):
            self.reuse(512)
        self.assertFalse((self.destination / '512' / self.relative).exists())

    def test_reused1024_requires_exact_current512_generation_prefix(self):
        self.create(1024)
        original_root = self.root / 'original'; original_path = original_root / self.relative
        original_path.parent.mkdir(parents=True)
        with original_path.open('xb') as stream: np.savez(stream, **self.original)
        prefix_root = self.root / 'prepared256'; prefix_path = prefix_root / self.relative
        prefix_path.parent.mkdir(parents=True)
        with prefix_path.open('xb') as stream: np.savez(stream, **self.admitted)
        current512 = source_fixture(512); current512['candidate_centers'][400, 0] += 20000
        new_prefix = self.destination / '512' / self.relative; new_prefix.parent.mkdir(parents=True)
        with new_prefix.open('xb') as stream: np.savez(stream, **current512)
        binding = {'bank_index': str(original_root / 'index.json')}
        provider = types.SimpleNamespace(load_patient=lambda case: 'DEBUG_NO_CT',
            verify_patient_inputs=lambda: None)
        with (patch.object(fast, '_PROVIDER', provider), patch.object(fast, '_BINDING', binding),
              patch.object(fast, 'verify_resources'), patch.object(base.historical, 'preprocessed_valid', return_value=True)):
            with self.assertRaisesRegex(base.BankContractError, 'current nested prefix'):
                fast._build_stage_patient(binding, 'DEBUG', [self.relative], str(self.destination),
                                          str(prefix_root), 1024, self.partial)


class StagedScopeAndGuardTests(unittest.TestCase):
    def test_all81_patients_finish_before_next_stage_is_admitted(self):
        # Full population/control-flow fixture: no synthetic NPZ is published
        # as a real bank and the geometry worker is explicitly not executed.
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            folder = Path(temporary).resolve(); original = folder / 'original'; original.mkdir()
            groups = {f'DEBUG_case_{i}': [] for i in range(81)}
            for i in range(642): groups[f'DEBUG_case_{i % 81}'].append(f'entries/DEBUG_{i}.npz')
            index = dict(format='hiercp_online_bank_v2', candidate_count=128, source_entries=642,
                eligible_cases=81, total_candidates=82176, entries_by_case=groups, minimum_liver_coverage=.85,
                liver_label=1, tumor_label=2, draw_count=256, attempts=4, cp_probability=.5)
            index_path = original / 'index.json'; base.write_json_exclusive(index_path, index)
            binding = {'bank_index': str(index_path)}; events = []

            class DebugFuture:
                def __init__(self, count, case, entries): self.count, self.case, self.entries = count, case, entries
                def result(self):
                    events.append(('finished', self.count, self.case, tuple(self.entries)))
                    return {'case_id': self.case, 'source_entries': self.entries}
                def cancel(self): raise AssertionError('No fixture failure should cancel a worker')

            class DebugExecutor:
                def __init__(self, **kwargs): self.kwargs = kwargs
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def submit(self, fn, binding, case, entries, destination, prefix256, count, partial):
                    self.last = fn; events.append(('submitted', count, case)); return DebugFuture(count, case, entries)

            def publish(root, count, *args, **kwargs):
                events.append(('published', count, 642)); return {'DEBUG_CONTROL_FLOW_ONLY': True, 'count': count}

            with (patch.object(fast, 'verify_resources'), patch.object(fast, '_ROOT', None),
                  patch.object(fast, 'ProcessPoolExecutor', DebugExecutor),
                  patch.object(fast, 'wait', side_effect=lambda futures, **kwargs: (set(futures), set())),
                  patch.object(base.historical, 'BANK_SHA256', base.file_sha(index_path)),
                  patch.object(base.historical, 'cpu_affinity', return_value=list(range(64))),
                  patch.object(base.historical, 'process_tree_rss', return_value=2**30),
                  patch.object(base.historical, 'verify_prepared_bank', return_value={'index': str(folder / 'DEBUG256/index.json')}),
                  patch.object(fast, '_publish_checked_prefix', side_effect=publish)):
                result = fast.build_staged_banks(binding, folder / 'fresh', workers=64, rss_limit_gib=288,
                    prepared256_witness={'DEBUG_ONLY': True}, binding_path=folder / 'DEBUG_binding.json', binding_sha256='DEBUG_SHA')
            self.assertEqual(set(result['prefixes']), {'512', '1024', '2048'})
            for count in base.COUNTS:
                published_at = events.index(('published', count, 642))
                finished = [row for row in events[:published_at] if row[0] == 'finished' and row[1] == count]
                self.assertEqual(len({row[2] for row in finished}), 81)
                all_names = [name for row in finished for name in row[3]]
                self.assertEqual(len(all_names), 642)
                self.assertEqual(len(set(all_names)), 642)
                self.assertEqual(set(all_names), {name for group in groups.values() for name in group})
            for first, second in ((512, 1024), (1024, 2048)):
                publish_at = events.index(('published', first, 642))
                second_at = next(i for i, row in enumerate(events) if row[0] == 'submitted' and row[1] == second)
                self.assertLess(publish_at, second_at)

    def test_fast_proposals_and_finite_sampler_both_bind_the_shared_guard_privately(self):
        guard = fast.FastGeometryProvider.proposals.__globals__['verify_resources']
        self.assertIs(guard, fast.verify_resources)
        catalogue_class = fast.FastGeometryProvider.proposals.__globals__['FiniteLiverCenters']
        self.assertIs(catalogue_class.sampler.__globals__['verify_resources'], fast.verify_resources)
        self.assertIs(base.FiniteLiverCenters.sampler.__globals__['verify_resources'], base.verify_resources)

    def test_actual_published_metadata_format_matches_trainer_for_every_large_count(self):
        expanded_fixture.ExpandedBasicCPTests.setUpClass()
        try:
            module = expanded_fixture.ExpandedBasicCPTests.module
            receipt = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011/HISTORICAL_BASIC_CP_CONTRACT_READONLY_1791835410056009100.json'
            original = json.loads(receipt.read_text(encoding='utf8'))['bank']
            for count in base.COUNTS:
                data = copy.deepcopy(original)
                data.update(candidate_count=count, total_candidates=642 * count,
                            ablation=base._ablation(Path('/DEBUG_original/index.json'), count, Path('/DEBUG_shared')))
                module.validate_metadata(data, f'candidates{count}')
                data['ablation']['format'] = 'basic_cp_expanded_candidates_preserved128_v1'
                with self.assertRaises(module.OnlineCPError): module.validate_metadata(data, f'candidates{count}')
        finally:
            expanded_fixture.ExpandedBasicCPTests.tearDownClass()


if __name__ == '__main__': unittest.main()
