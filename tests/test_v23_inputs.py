"""Mechanical CPU input UNIT tests; no clinical/model-quality result.

Only the canonical loader and original view materializer are analytic UNIT
fixtures. The actual native collator, storage ledger, worker/lifetime and cache
admission machinery execute here with explicitly marked synthetic records.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

from hiercp_v1x import transition_v1_data as original
from hiercp_v1x import transition_v1_local as local
from hiercp_v1x.v23_inputs import V23InputMemoryCoordinator, V23InputProvider


def _dataset(root, partition='inner_val'):
    dataset = object.__new__(original.NativeObservationDataset)
    dataset.rows = [dict(id=f'UNIT-{partition}-{index}', case_id='UNIT-recipient',
                         donor_case_id='UNIT-donor') for index in range(2)]
    dataset.path = root / (partition+'.json')
    dataset.path.write_text(json.dumps(dataset.rows), encoding='utf8')
    dataset.index_sha256 = original._sha(dataset.path)
    dataset.partition, dataset.debug = partition, True
    dataset.meta = {}; dataset.base = {}; dataset.assignment_sha256 = 'a'*64
    dataset.scope_contract = 'b'*64
    return dataset


def _records(dataset, source=None):
    source = torch.ones(5, 48, 48, 48, dtype=torch.float16) if source is None else source
    records = {}
    for index, row in enumerate(dataset.rows):
        records[row['id']] = dict(UNIT=True, UNIT_index=index,
            content_binding=dict(graph_sha256=str(index).zfill(64)),
            input_provenance=dict(observation_id=row['id']),
            source_patch=source, target_patch=torch.ones_like(source)*(index+2),
            canonical_nodes=torch.arange(20), canonical_edges=torch.arange(40))
    return records


def _pair(record, *, epoch):
    views = []
    for view in (0, 1):
        graph = HeteroData()
        graph['UNIT'].x = torch.tensor([[record['UNIT_index'], epoch, view]], dtype=torch.float32)
        graph['UNIT', 'UNIT_link', 'UNIT'].edge_index = torch.tensor([[0], [0]])
        graph.transition_record_sha256 = record['content_binding']['graph_sha256']
        graph.transition_observation_id = record['input_provenance']['observation_id']
        graph.transition_recipient_case = 'UNIT-recipient'
        graph.transition_donor_case = 'UNIT-donor'
        graph.transition_view_id = torch.tensor([view])
        graph.transition_epoch = torch.tensor([epoch])
        views.append(graph)
    return tuple(views), record['source_patch'], record['target_patch']


class _Provider(V23InputProvider):
    def __init__(self, dataset, records, **kwargs):
        super().__init__(dataset, workers=2, resident_bytes=64*1024**2,
                         rss_bytes=10**13, **kwargs)
        self.unit_records = records
        self._canonical = {key:{} for key in records}
        self.load_threads = []
    def _load(self, row):
        self.load_threads.append(threading.get_ident())
        return self.unit_records[row['id']]


class NativeInputReuseUNIT(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='v23_inputs_UNIT_', dir=Path.cwd())
        self.root = Path(self.directory.name)
        self.validation = patch.object(local, 'validate_record', side_effect=lambda record: None)
        self.validation.start()
        self.materialized = []
        def materialize(record, *, epoch):
            self.materialized.append((record['input_provenance']['observation_id'], epoch,
                                      threading.get_ident()))
            return _pair(record, epoch=epoch)
        self.materializer = patch.object(local, 'materialize_pair', side_effect=materialize)
        self.materializer.start()
        self.providers = []
    def tearDown(self):
        for provider in self.providers: provider.close()
        self.materializer.stop(); self.validation.stop(); self.directory.cleanup()
    def provider(self, partition='inner_val', **kwargs):
        dataset = _dataset(self.root, partition)
        provider = _Provider(dataset, _records(dataset), **kwargs)
        self.providers.append(provider)
        return provider

    def test_fixed_validation_views_reuse_actual_epoch_and_preserve_every_tensor(self):
        provider = self.provider()
        before_rng = torch.random.get_rng_state().clone()
        first = provider.get([0, 1], epoch=29)
        second = provider.get([0, 1], epoch=29)
        self.assertEqual(len(self.materialized), 2)
        torch.testing.assert_close(first.graph['UNIT'].x, second.graph['UNIT'].x, rtol=0, atol=0)
        torch.testing.assert_close(first.source_patches, second.source_patches, rtol=0, atol=0)
        torch.testing.assert_close(first.target_patches, second.target_patches, rtol=0, atol=0)
        torch.testing.assert_close(first.graph_observation_index, second.graph_observation_index)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        profile = provider.profile()
        self.assertEqual(profile['sampled_pair_hits'], 2)
        self.assertEqual(profile['sampled_pairs_materialized'], 2)
        self.assertFalse(profile['neural_features_cached'])

    def test_training_views_always_use_original_epoch_and_are_never_memoized(self):
        provider = self.provider('inner_train')
        first = provider.get([0], epoch=8)
        second = provider.get([0], epoch=9)
        third = provider.get([0], epoch=8)
        self.assertEqual([epoch for _, epoch, _ in self.materialized], [8, 9, 8])
        self.assertFalse(torch.equal(first.graph['UNIT'].x, second.graph['UNIT'].x))
        torch.testing.assert_close(first.graph['UNIT'].x, third.graph['UNIT'].x)
        self.assertEqual(provider.profile()['sampled_pair_hits'], 0)
        self.assertEqual(provider.profile()['sampled_view_entries'], 0)
        self.assertFalse(provider.profile()['CPU_sampled_view_cache'])

    def test_returned_batch_cannot_mutate_cached_views_or_original_patch_owners(self):
        provider = self.provider()
        first = provider.get([0, 1], epoch=29)
        reference = first.graph['UNIT'].x.clone()
        first.graph['UNIT'].x.add_(100)
        first.source_patches.fill_(55); first.target_patches.fill_(66)
        second = provider.get([0, 1], epoch=29)
        torch.testing.assert_close(second.graph['UNIT'].x, reference)
        self.assertTrue(bool((second.source_patches == 1).all()))
        self.assertTrue(bool((second.target_patches[0] == 2).all()))

    def test_internal_cached_tensor_mutation_is_rejected_instead_of_silent_reuse(self):
        provider = self.provider(); provider.get([0], epoch=29)
        entry = next(iter(provider._views.values()))
        entry['payload'][0][0]['UNIT'].x.add_(1)
        with self.assertRaisesRegex(ValueError, 'sampled CPU view changed'):
            provider.get([0], epoch=29)

    def test_record_binding_sha_and_epoch_form_distinct_cache_identity(self):
        provider = self.provider(); provider.get([0], epoch=29)
        provider.unit_records[provider.ds.rows[0]['id']]['content_binding']['graph_sha256'] = 'f'*64
        provider.get([0], epoch=29)
        self.assertEqual(len(self.materialized), 2)
        self.assertEqual(provider.profile()['sampled_view_entries'], 2)
        with self.assertRaisesRegex(ValueError, 'fixed epoch'):
            provider.get([0], epoch=30)

    def test_worker_executor_reused_and_close_drains_and_rejects_further_batches(self):
        provider = self.provider(); executor = provider._executor
        provider.get([0], epoch=29); provider.get([1], epoch=29)
        self.assertIs(provider._executor, executor)
        self.assertTrue(all(worker != threading.get_ident() for worker in provider.load_threads))
        provider.close(); provider.close()
        self.assertEqual(provider.profile()['sampled_view_entries'], 0)
        with self.assertRaisesRegex(RuntimeError, 'Closed'):
            provider.get([0], epoch=29)
        self.assertFalse(any(thread.name.startswith('v23_cpu_native') for thread in threading.enumerate()))

    def test_baseline_flags_delegate_original_get_without_view_or_persistent_pool(self):
        provider = self.provider(persistent_cpu_workers=False, cache_sampled_views=False)
        first = provider.get([0], epoch=29); second = provider.get([0], epoch=29)
        self.assertIsNone(provider._executor)
        self.assertEqual(provider.profile()['sampled_view_entries'], 0)
        self.assertEqual(len(self.materialized), 2)
        torch.testing.assert_close(first.graph['UNIT'].x, second.graph['UNIT'].x)

    def test_alias_ledger_counts_shared_patches_once_and_eviction_keeps_complete_returned_batch(self):
        provider = self.provider(); batch = provider.get([0, 1], epoch=29)
        owners = [record for record, _ in provider._records.values()]
        owners += [tuple([graph.to_dict() for graph in entry['payload'][0]]) + entry['payload'][1:]
                   for entry in provider._views.values()]
        self.assertEqual(provider._resident.bytes('all'), original._bytes(owners))
        preserved = batch.graph['UNIT'].x.clone()
        while provider._evict_one('view'): pass
        self.assertEqual(provider._resident.bytes('view'), 0)
        torch.testing.assert_close(batch.graph['UNIT'].x, preserved)
        reloaded = provider.get([0, 1], epoch=29)
        torch.testing.assert_close(batch.graph['UNIT'].x, reloaded.graph['UNIT'].x)
        self.assertEqual(len(self.materialized), 4)

    def test_shared_process_pressure_evicts_inactive_partition_cache_first(self):
        coordinator = V23InputMemoryCoordinator(10**13)
        train = self.provider('inner_train', memory_coordinator=coordinator)
        validation = self.provider(memory_coordinator=coordinator)
        train.get([0], epoch=8); validation.get([0], epoch=29)
        def memory():
            # Mechanical pressure oracle: actual implementation reads psutil
            # RSS; this UNIT oracle exposes only owned-cache release behavior.
            cached = len(train._views)+len(validation._views)
            return SimpleNamespace(rss=10**13+1 if cached else 0)
        with patch('hiercp_v1x.v23_inputs.psutil.Process', return_value=SimpleNamespace(memory_info=memory)):
            coordinator.trim()
        self.assertEqual(len(train._views)+len(validation._views), 0)
        self.assertGreater(train.profile()['sampled_view_evictions']+validation.profile()['sampled_view_evictions'], 0)

    def test_active_complete_chunk_pressure_is_explicit_and_never_shrinks_input(self):
        provider = self.provider(); provider.get([0], epoch=29)
        provider._protected_ids = {provider.ds.rows[0]['id']}
        provider._protected_view_keys = set(provider._views)
        with patch('hiercp_v1x.v23_inputs.psutil.Process', return_value=SimpleNamespace(
                memory_info=lambda:SimpleNamespace(rss=provider.rss_bytes+1))):
            with self.assertRaisesRegex(MemoryError, 'Complete active native input'):
                provider.memory_coordinator.trim()
        self.assertEqual(len(provider._views), 1)
        provider._protected_ids.clear(); provider._protected_view_keys.clear()

    def test_materialization_failure_propagates_and_owned_workers_are_drained(self):
        provider = self.provider()
        with patch.object(local, 'materialize_pair', side_effect=ValueError('UNIT original view failure')):
            with self.assertRaisesRegex(ValueError, 'original view failure'):
                provider.get([0, 1], epoch=29)
        provider.close()
        self.assertFalse(any(thread.name.startswith('v23_cpu_native') for thread in threading.enumerate()))


if __name__ == '__main__': unittest.main()
