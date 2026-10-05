"""Inference batching/guard regressions, never training-quality evidence."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hiercp_v1x.historical_evaluation import (FIELDS,assert_new_destination,
    calibration_candidates,encode_original_fields,unpack_fields,write_new,
    verify_inventory_request,sha)


class HistoricalEvaluationRegression(unittest.TestCase):
    def test_inventory_change_between_workers_cannot_rebind_parent_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json';write_new(path,{'assignment':'first'})
            request={'inventory_sha256':sha(path)}
            verify_inventory_request(path,request)
            path.write_text('{"assignment":"different"}',encoding='utf8')
            with self.assertRaises(ValueError):verify_inventory_request(path,request)

    def test_existing_artifact_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'; write_new(path,{'old':True})
            with self.assertRaises(FileExistsError): write_new(path,{'old':False})
            self.assertIn('true',path.read_text())

    def test_disjoint_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            old=Path(directory)/'old';old.mkdir()
            assert_new_destination(Path(directory)/'new',(old,))
            for path in (old,old/'child',Path(directory)):
                with self.assertRaises(ValueError):assert_new_destination(path,(old,))

    def test_explicit_physical_batches(self):
        calibration_candidates([8,16,32],128)
        for value in ([],[0],[16,8],[8,8],[129]):
            with self.assertRaises(ValueError):calibration_candidates(value,128)

    def test_fields_cannot_be_replaced_by_fused_only(self):
        with self.assertRaises(ValueError):unpack_fields(torch.zeros(4,128))
        value=torch.arange(4*len(FIELDS)*128).reshape(4,-1)
        fields=unpack_fields(value)
        self.assertEqual(tuple(fields),FIELDS)
        self.assertTrue(torch.equal(torch.cat(list(fields.values()),1),value))

    def test_two_views_are_batched_and_meaned_in_correct_order(self):
        class Core:
            calls=0
            def encode_dense_maps(self,source,indices,target):
                return source.index_select(0,indices),target
            def forward_graph(self,graph,source,target):
                self.calls+=1
                self.assert_source=source[:,0].tolist();self.assert_target=target[:,0].tolist()
                return {name:torch.arange(6).float()[:,None].expand(-1,128)+i*10
                        for i,name in enumerate(FIELDS)}
        class Batch:
            graph=SimpleNamespace(num_graphs=6)
            source_patches=torch.tensor([[100.],[200.]])
            source_index=torch.tensor([0,1,0]);target_patches=torch.tensor([[1.],[2.],[3.]])
            graph_observation_index=torch.tensor([0,0,1,1,2,2])
            def __len__(self):return 3
        core=Core(); result=unpack_fields(encode_original_fields(SimpleNamespace(local_encoder=core),Batch()))
        self.assertEqual(core.calls,1)
        self.assertEqual(core.assert_source,[100,100,200,200,100,100])
        self.assertEqual(core.assert_target,[1,1,2,2,3,3])
        self.assertEqual(result['tumor'][:,0].tolist(),[.5,2.5,4.5])
        self.assertEqual(result['fused'][:,0].tolist(),[110.5,112.5,114.5])

    def test_missing_second_view_fails(self):
        class Batch:
            graph=SimpleNamespace(num_graphs=3)
            def __len__(self):return 3
        with self.assertRaises(ValueError):encode_original_fields(None,Batch())


if __name__=='__main__':unittest.main()
