"""Actual original CPU graph/storage UNIT fixture; no clinical/model claim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

import torch

from tests import test_v24_gt_blind as _gt_fixture
from hiercp_v1x.v24_provider import V24InputProvider,V24InputCoordinator,FORMAT
from hiercp_v1x.v24_inputs import query_inputs,tensor_digest

ROOT=Path(__file__).resolve().parents[1]


class V24CPUProviderDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _gt_fixture.RecipientGTBlindDebug.setUpClass()
        cls.fixture=_gt_fixture.RecipientGTBlindDebug
        cls.root=ROOT/'outputs'/('v24_CPU_provider_UNIT_'+uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        from hiercp_v22.storage import GraphWriter
        writer=GraphWriter(cls.root,minimum_free_bytes=0);records=[]
        for row,record in zip(cls.fixture.rows,cls.fixture.records):
            entry=writer.write('records/'+str(len(records))+'.pt.gz',record,('UNIT_same_donor',1))
            records.append(dict(id=row['id'],query=query_inputs([row])[0],tensor_sha256=record['tensor_sha256'],**entry))
        cls.index=cls.root/'index.json'
        cls.index.write_text(json.dumps(dict(format=FORMAT,complete=True,debug=False,
            recipient_GT_used_in_forward=False,records=records)),encoding='utf8')

    @classmethod
    def tearDownClass(cls):_gt_fixture.RecipientGTBlindDebug.tearDownClass()

    def provider(self,partition='inner_val'):
        import psutil
        # Explicit CPU UNIT headroom avoids machine-dependent production192GiB
        # budgets; graph/node/model equations stay the original operators.
        rss=psutil.Process().memory_info().rss+2**30
        ds=SimpleNamespace(rows=copy.deepcopy(self.fixture.rows),meta=dict(records=self.fixture.rows),partition=partition)
        return V24InputProvider(ds,self.index,workers=12,resident_bytes=2**29,coordinator=V24InputCoordinator(rss))

    @staticmethod
    def signature(batch):
        return tensor_digest((batch.graph.to_dict(),batch.source_patches,batch.target_patches,
            batch.source_index,batch.graph_observation_index,batch.indices))

    def test_fixed29_memo_is_exact_complete_and_private(self):
        provider=self.provider()
        try:
            first=provider.get([0,1,2],epoch=29);expected=self.signature(first)
            first.target_patches.add_(100)
            first.graph['tumor_surface'].x.add_(100)
            second=provider.get([0,1,2],epoch=29)
            self.assertEqual(expected,self.signature(second))
            stats=provider.profile();self.assertEqual(stats['sampled_pair_hits'],3)
            self.assertEqual(stats['sampled_pairs_materialized'],3)
            self.assertEqual(stats['sampled_view_entries'],3)
            self.assertEqual(second.graph.num_graphs,6)
            self.assertEqual(second.indices.tolist(),[0,1,2])
        finally:provider.close()

    def test_training_never_memoizes_epoch_views_or_changes_rng(self):
        provider=self.provider('inner_train')
        try:
            before=torch.get_rng_state().clone()
            first=provider.get([0,1,2],epoch=1);second=provider.get([0,1,2],epoch=2)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertFalse(torch.equal(first.graph.transition_epoch,second.graph.transition_epoch))
            self.assertEqual(provider.profile()['sampled_view_entries'],0)
            self.assertEqual(provider.profile()['sampled_pairs_materialized'],6)
        finally:provider.close()

    def test_warm_canonical_file_replacement_detected(self):
        provider=self.provider()
        try:
            provider.get([0],epoch=29)
            file=provider.root/provider.records[self.fixture.rows[0]['id']]['path']
            old=file.stat().st_mtime_ns
            import os
            os.utime(file,ns=(file.stat().st_atime_ns,old+1000000))
            with self.assertRaisesRegex(ValueError,'replaced or changed'):provider.get([0],epoch=29)
        finally:
            if 'old'in locals():os.utime(file,ns=(file.stat().st_atime_ns,old))
            provider.close()

    def test_recipient_supervision_not_a_neural_query_or_cache_key(self):
        provider=self.provider()
        try:
            first=provider.get([0,1,2],epoch=29)
            for row in provider.ds.rows:row['target']=1-row['target'];row['component']=999
            second=provider.get([0,1,2],epoch=29)
            self.assertEqual(self.signature(first),self.signature(second))
            self.assertEqual(provider.profile()['sampled_pair_hits'],3)
        finally:provider.close()

    def test_close_finishes_workers_and_rejects_further_input(self):
        provider=self.provider();provider.get([0],epoch=29);provider.close();provider.close()
        self.assertEqual(provider.coordinator.providers,[])
        with self.assertRaisesRegex(RuntimeError,'Closed'):provider.get([0],epoch=29)

    def test_real_validation_epoch_and_unique_full_indices_required(self):
        provider=self.provider()
        try:
            with self.assertRaisesRegex(ValueError,'fixed29'):provider.get([0],epoch=1)
            with self.assertRaisesRegex(ValueError,'unique'):provider.get([0,0],epoch=29)
        finally:provider.close()

    def test_coordinator_evictions_always_own_shared_RLock(self):
        from unittest.mock import patch
        coordinator=V24InputCoordinator(1024);ownership=[]
        def evict(category):ownership.append(coordinator.lock._is_owned());return False
        coordinator.providers=[SimpleNamespace(evict=evict)]
        with patch('hiercp_v1x.v24_provider.psutil.Process',return_value=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=2048))):
            coordinator.trim(strict=False)
        self.assertEqual(ownership,[True,True])


if __name__=='__main__':unittest.main()
