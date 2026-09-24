"""Checkpoint safety and CPU batch ownership tests (not training evidence)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch_geometric.data import HeteroData,Batch
from hiercp_v222.v1_local import LocalBatch
from hiercp_v222.v1_execution import atomic_torch,tree_to

TEST_ROOT=Path(__file__).resolve().parents[1]/'work'

class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):TEST_ROOT.mkdir(parents=True,exist_ok=True)
    def test_transfer_preserves_cpu_stores(self):
        graph=HeteroData();graph['node'].x=torch.arange(12).reshape(4,3)
        graph['node','near','node'].edge_index=torch.tensor([[0,1],[1,2]])
        payload=LocalBatch(Batch.from_data_list([graph]),torch.ones(1,1,2,2,2),
            torch.ones(1,1,2,2,2),torch.tensor([0]),torch.tensor([0]))
        moved=payload.to('meta')
        self.assertEqual(payload.graph['node'].x.device.type,'cpu')
        self.assertEqual(moved.graph['node'].x.device.type,'meta')
        self.assertIsNot(payload.graph['node'],moved.graph['node'])
    def test_failed_checkpoint_retains_last_good(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(TEST_ROOT.resolve()))
            path=Path(temp)/'latest.pt';atomic_torch(path,{'step':7})
            with patch('torch.save',side_effect=OSError('simulated disk write failure')):
                with self.assertRaises(OSError):atomic_torch(path,{'step':8})
            self.assertEqual(torch.load(path,weights_only=True),{'step':7})
            self.assertEqual(list(Path(temp).iterdir()),[path])
    def test_new_checkpoint_replaces_owned_previous(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(TEST_ROOT.resolve()))
            path=Path(temp)/'latest.pt';atomic_torch(path,{'step':7});atomic_torch(path,{'step':8})
            self.assertEqual(torch.load(path,weights_only=True),{'step':8})
    def test_nested_state_preserves_containers(self):
        value={'plan':(torch.tensor([1]),[torch.tensor([2])]),'group':'case'}
        copied=tree_to(value,'cpu')
        self.assertIsInstance(copied['plan'],tuple)
        self.assertTrue(torch.equal(copied['plan'][1][0],torch.tensor([2])))

if __name__=='__main__':unittest.main()
