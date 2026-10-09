"""CPU UNIT immutable geometry boundaries with actual PyG tensor graphs.

Analytic UNIT tensors and mocked clinical admission are not CT validation,
full inference, training, or evidence of model quality. The actual historical
builder remains unchanged and is never replaced in production preparation.
"""
from __future__ import annotations

import copy
import json
import mmap
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, HeteroData

from hiercp_v1x import comparison_native_upper_cache as cache
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.historical_evaluation import sha, write_new


def _graph(rows, *, prototype=False):
    graph = HeteroData()
    name = 'prototype' if prototype else 'candidate'
    count = 5 if prototype else len(rows)
    base = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
    graph[name].raw_x = base[:, ::2]  # Serialization must preserve noncontiguous strides.
    graph[name].pos = base[:, 1:4] / 7
    graph[name].num_nodes = count
    graph[(name, 'UNIT_relation', name)].edge_index = torch.stack((torch.arange(count), torch.arange(count)))
    graph.UNIT_scope = 'UNIT exact all-fields roundtrip'
    return graph


def fixture(directory):
    """Twenty-one whole 128-U UNIT cases, including every zero-P case."""
    directory = Path(directory)
    root = directory / 'cache'; root.mkdir()
    case_ids = [f'UNITrecipient{i:02}' for i in range(21)]
    rows = {case: [dict(id=f'{case}:U:{i}', case_id=case, patient_group=case,
                       center=[i, 1, 2], donor_case_id='UNITdonor', donor_component=1,
                       donor_group='UNITdonor') for i in range(128)] for case in case_ids}
    raw = {}
    for case in [*case_ids, 'UNITdonor']:
        image = directory / (case + '.UNITimage'); image.write_bytes(case.encode() + b' image')
        label = directory / (case + '.UNITlabel'); label.write_bytes(case.encode() + b' label')
        raw[case] = dict(image=str(image), label=str(label), image_sha256=sha(image), label_sha256=sha(label))
    metadata = dict(config=dict(donor_max_diameter_mm=99), split=dict(inner_train=['UNITdonor']),
                    raw_records=[dict(case_id=case, **item) for case, item in raw.items()])
    inventory_path = directory / 'UNITinventory.json'; write_new(inventory_path, metadata)
    bank_path = directory / 'UNITbank.pt'; torch.save({'UNIT': torch.arange(4)}, bank_path)
    source = directory / 'UNITsource'; source.mkdir()
    (source / 'hiercp').mkdir()
    (source / 'hiercp/common.py').write_text('# explicit analytic UNIT source fixture\n', encoding='utf8')
    native_config = directory / 'UNITnative.json'; write_new(native_config, {'UNIT': True})
    source_proof = dict(source=str(source), verified_files={'hiercp/common.py': 'a' * 64})
    baseline_proof = dict(UNIT='actual clinical admission explicitly mocked in CPU UNIT')
    cohort = dict(debug=False, case_ids=case_ids, by_case=rows,
                  rows=[row for case in case_ids for row in rows[case]], records=21 * 128,
                  observed_P=0, unobserved_U=21 * 128)
    signature = dict(format=cache.FORMAT, native_inventory_path=str(inventory_path),
        native_inventory_sha256=sha(inventory_path), raw_inputs=raw,
        execution_code_sha256=cache._code(), graph={'UNIT': True}, ct_clip=[-200., 250.], source_pad=2,
        scope=dict(contract_sha256='c' * 64), source_proof=source_proof,
        baseline=str(directory), bank_path=str(bank_path), bank_sha256=sha(bank_path),
        bank_fingerprint='UNITbank', bank_training_case_ids=['UNITdonor'], training_case_ids=['UNITdonor'],
        baseline_proof_sha256=canonical_hash(baseline_proof), donor_max_diameter_mm=99,
        native_region_config_path=str(native_config), native_region_config_sha256=sha(native_config))
    bank = SimpleNamespace(fingerprint=lambda: 'UNITbank', training_case_ids=['UNITdonor'])
    bundle = SimpleNamespace(receipt=dict(debug=False, baseline_proof=baseline_proof),
        config=dict(graph={'UNIT': True}, ct_clip=[-200., 250.], cache=dict(source_pad=2)),
        scope=signature['scope'], source=source, baseline=directory, prototype_bank=bank)
    regions = {}
    for case in raw:
        region = directory / 'UNITregions' / case; region.mkdir(parents=True)
        write_new(region / 'metadata.json', dict(UNIT_case=case))
        (region / 'UNITpayload').write_bytes(case.encode())
        regions[case] = dict(path=str(region), files=cache._region_files(region),
                             metadata=cache._read(region / 'metadata.json'))
    write_new(root / 'request.json', cache._sealed(dict(signature=signature)))
    entries = []
    for case in case_ids:
        audit = dict(debug=False, candidate_count=128, recipient_case_id=case, donor_case_id='UNITdonor',
            bank_fingerprint='UNITbank', prototype_training_case_ids=['UNITdonor'],
            P_U_labels_in_forward=False, original_learned_operators_unchanged=True,
            original_source_module_sha256={'common': 'a' * 64}, UNIT_analytic=True)
        entries.append(cache._save_case(root, rows[case], _graph(rows[case]), _graph(rows[case], prototype=True),
                                       audit, signature, regions, dict(UNIT=True)))
    index = cache._sealed(dict(format=cache.FORMAT, complete=True, actual_CT=True, debug=False,
        signature=signature, case_count=21, records=21 * 128, observed_P=0, unobserved_U=21 * 128,
        cases=entries, region_bindings=regions, request_sha256=sha(root / 'request.json')))
    write_new(root / 'index.json', index)
    return SimpleNamespace(root=root, rows=rows, metadata=metadata, bundle=bundle,
        inventory_path=inventory_path, cohort=cohort, signature=signature, source_proof=source_proof,
        provider=SimpleNamespace(ds=SimpleNamespace(meta=metadata)), index=index)


class NativeUpperCacheCPUUnit(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(dir=cache.ROOT / 'tmp', prefix='native-upper-UNIT-')
        self.value = fixture(self.directory.name)
        self.patches = [patch.object(cache, 'validate_cohort', return_value=self.value.cohort),
                        patch.object(cache, '_signature', return_value=self.value.signature),
                        patch.object(cache, '_source_proof', return_value=self.value.source_proof)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        # These are positively owned temporary UNIT artifacts, never experiments.
        for path in Path(self.directory.name).rglob('*.pt'):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.directory.cleanup()

    def open(self):
        return cache.UpperGeometryCache(self.value.root, self.value.bundle, self.value.inventory_path)

    def obtain(self, instance, rows=None):
        query = rows or self.value.rows['UNITrecipient00']
        return instance.upper_graphs(self.value.bundle, self.value.provider, query,
                                     region_output=self.value.root / 'unused')

    def resign_index(self, value):
        value.pop('content_sha256', None)
        (self.value.root / 'index.json').write_text(json.dumps(cache._sealed(value)), encoding='utf8')

    def test_exact_pyg_roundtrip_every_case_including_zero_p(self):
        instance = self.open()
        with patch('hiercp_v1x.historical_evaluation.upper_graphs', side_effect=AssertionError('No rebuild')):
            for case, rows in self.value.rows.items():
                graph, prototype, audit = self.obtain(instance, rows)
                self.assertTrue(cache._exact(graph.to_dict(), _graph(rows).to_dict()))
                self.assertTrue(cache._exact(prototype.to_dict(), _graph(rows, prototype=True).to_dict()))
                self.assertEqual(audit['recipient_case_id'], case)
                self.assertFalse(audit['P_U_labels_in_forward'])
        self.assertEqual(instance.receipt['cache_hits'], 21)
        self.assertTrue(instance.finish()['final_full_input_SHA256_verified'])
        self.assertTrue(all(entry['measurement']['exact_serialization_roundtrip_verified']
                            for entry in instance.index['cases']))

    def test_private_mmap_and_pyg_batch_cannot_dirty_disk_or_next_load(self):
        instance = self.open()
        entry = instance.entries['UNITrecipient00']; path = self.value.root / entry['path']
        graph, prototype, audit = self.obtain(instance)
        batch = Batch.from_data_list([graph])
        batch['candidate'].raw_x.fill_(-9)
        graph['candidate'].raw_x.fill_(123)
        prototype['prototype'].raw_x.fill_(456)
        audit['candidate_count'] = 0
        self.assertEqual(sha(path), entry['payload_sha256'])
        other, other_prototype, other_audit = self.obtain(instance)
        self.assertTrue(cache._exact(other.to_dict(), _graph(self.value.rows['UNITrecipient00']).to_dict()))
        self.assertTrue(cache._exact(other_prototype.to_dict(),
                                    _graph(self.value.rows['UNITrecipient00'], prototype=True).to_dict()))
        self.assertEqual(other_audit['candidate_count'], 128)

    def test_query_order_missing_query_wrong_donor_and_target_are_rejected(self):
        instance = self.open()
        original = self.value.rows['UNITrecipient00']
        changed = copy.deepcopy(original); changed[0]['donor_component'] = 2
        target = copy.deepcopy(original); target[0]['target'] = 0
        for rows in (original[::-1], original[:-1], changed, target):
            with self.subTest(kind=len(rows)), self.assertRaises(ValueError):
                self.obtain(instance, rows)

    def test_loaded_bundle_bank_config_scope_source_and_baseline_are_bound(self):
        instance = self.open()
        mutations = [lambda b: b.config['graph'].update(UNIT=False),
                     lambda b: setattr(b.prototype_bank, 'fingerprint', lambda: 'other'),
                     lambda b: b.scope.update(contract_sha256='d' * 64),
                     lambda b: setattr(b, 'source', self.value.root),
                     lambda b: setattr(b, 'baseline', self.value.root)]
        for change in mutations:
            old = self.value.bundle
            self.value.bundle = copy.deepcopy(old)
            change(self.value.bundle)
            with self.assertRaisesRegex(ValueError, 'bundle bank/config/source/scope'):
                self.obtain(instance)
            self.value.bundle = old

    def test_payload_corruption_fails_before_unpickling(self):
        entry = self.value.index['cases'][0]; path = self.value.root / entry['path']
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        with path.open('ab') as stream:
            stream.write(b'UNIT corruption')
        with patch.object(torch, 'load', side_effect=AssertionError('No untrusted load')):
            with self.assertRaisesRegex(ValueError, 'payload size/path/SHA256'):
                self.open()

    def test_manifest_partial_extra_case_or_signature_corruption_is_rejected(self):
        for change in (lambda x: x.update(complete=False), lambda x: x['cases'].pop(),
                       lambda x: x['signature'].update(bank_sha256='f' * 64)):
            value = copy.deepcopy(self.value.index); change(value); self.resign_index(value)
            with self.assertRaisesRegex(ValueError, 'signature/cohort/config/source/bank'):
                self.open()
        self.resign_index(copy.deepcopy(self.value.index))

    def test_resigned_query_receipt_corruption_is_rejected(self):
        value = copy.deepcopy(self.value.index)
        receipt = value['cases'][0]['receipt']; receipt.pop('content_sha256')
        receipt['query_rows'][0]['center'][0] += 1
        value['cases'][0]['receipt'] = cache._sealed(receipt)
        self.resign_index(value)
        with self.assertRaisesRegex(ValueError, 'exact query order/input/bank/config'):
            self.open()

    def test_raw_and_region_changes_are_rejected(self):
        instance = self.open()
        raw = self.value.signature['raw_inputs']['UNITrecipient00']
        Path(raw['image']).write_bytes(b'UNIT changed CT')
        with self.assertRaisesRegex(ValueError, 'raw CT/annotation/region/source input changed'):
            self.obtain(instance)
        Path(raw['image']).write_bytes(b'UNITrecipient00 image')
        region = Path(instance.index['region_bindings']['UNITrecipient00']['path']) / 'UNITpayload'
        region.write_bytes(b'UNIT changed region')
        # A restored mtime is insufficient: the post-SHA ctime identity changes.
        # The first mutation already invalidates this entire admitted reader.
        with self.assertRaisesRegex(ValueError, 'raw CT/annotation/region/source input changed'):
            self.obtain(instance)

    def test_manifest_inventory_and_code_changes_after_open_are_rejected(self):
        instance = self.open()
        with patch.object(cache, '_code', return_value={}):
            with self.assertRaisesRegex(ValueError, 'inventory/code/manifest changed'):
                self.obtain(instance)
        with self.value.inventory_path.open('a', encoding='utf8') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'inventory/code/manifest changed'):
            self.obtain(instance)

    def test_unsafe_or_symlink_payload_path_is_rejected(self):
        for name in ('../escape.pt', '/absolute.pt', 'cases\\wrong.pt', 'C:drive.pt'):
            with self.subTest(path=name), self.assertRaises(ValueError):
                cache._relative(self.value.root, name)

    def test_shared_mmap_is_rejected_and_existing_publication_not_overwritten(self):
        instance = self.open()
        with patch.object(cache, '_private_mmap', return_value=False):
            with self.assertRaisesRegex(ValueError, 'private copy-on-write'):
                self.obtain(instance)
        first = self.value.index['cases'][0]
        rows = self.value.rows[first['case_id']]
        with self.assertRaises(FileExistsError):
            cache._save_case(self.value.root, rows, _graph(rows), _graph(rows, prototype=True),
                self.obtain(instance)[2], self.value.signature, self.value.index['region_bindings'], {})
        self.assertEqual(sha(self.value.root / first['path']), first['payload_sha256'])


if __name__ == '__main__':
    unittest.main()
