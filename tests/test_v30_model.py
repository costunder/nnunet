"""Explicit synthetic CPU UNIT checks for the complete v3.0 vessel path.

These fixtures test connectivity, ownership, gradients and numerical behavior.
They are not real CT inputs, a smoke-training result, or an evaluation result.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v1x.v30_model import (PhysicalVesselAttention, VesselAugmentedScoreHead,
                                 install_vessel_branch, vessel_score_context)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "work/pretrained_l0_latency_DEBUG_20261008/assets/small_ep4k.model"


def vessel_fixture_UNIT():
    """Two patients, five genuine UNIT components including isolated nodes."""
    positions = torch.tensor([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.],
                              [8.,1.,0.], [9.,1.,0.], [100.,3.,2.],
                              [0.,20.,0.], [1.,20.,0.], [70.,20.,0.]])
    owner = torch.tensor([0,0,0,0,0,0,1,1,1])
    components = torch.tensor([0,0,0,1,1,2,3,3,4])
    forward = torch.tensor([[0,1,3,6], [1,2,4,7]])
    edge_index = torch.cat((forward, forward.flip(0)), 1)
    features = torch.linspace(-.8, 1.1, 9 * 12).reshape(9,12)
    features[:,3] = torch.linspace(.6, 2.8, 9)
    delta = positions[edge_index[1]] - positions[edge_index[0]]
    length = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    radius_delta = features[edge_index[1],3:4] - features[edge_index[0],3:4]
    edge_attr = torch.cat((delta, length, delta / length, radius_delta), 1)
    return SimpleNamespace(node_features=features, pos_mm=positions, edge_index=edge_index,
        edge_attr=edge_attr, node_patient_index=owner, component_index=components,
        component_patient_index=torch.tensor([0,0,0,1,1]), case_ids=("UNIT_patient_A", "UNIT_patient_B"))


def head_fixture_UNIT():
    original = nn.Sequential(nn.Linear(1546,512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(0),
        nn.Linear(512,256), nn.SiLU(), nn.Dropout(0), nn.Linear(256,1,bias=False))
    return VesselAugmentedScoreHead(original, dropout=0, query_chunk_size=2, node_chunk_size=3)


class CompleteVesselModelUNIT(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.head = head_fixture_UNIT().eval()
        self.model = SimpleNamespace(score_head=self.head)
        self.batch = vessel_fixture_UNIT()
        self.pair = torch.randn(4,1546)
        self.pos = torch.tensor([[.5,0.,0.], [.5,20.,0.], [7.,1.,0.], [50.,20.,0.]])
        self.owner = torch.tensor([0,1,0,1])

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def score(self, batch=None, pair=None, pos=None, owner=None):
        with vessel_score_context(self.model, self.batch if batch is None else batch,
                                  self.pos if pos is None else pos, self.owner if owner is None else owner):
            return self.head(self.pair if pair is None else pair)

    def test_original_full_width_weights_retained_and_no_runtime_state_cached(self):
        self.assertEqual(self.head[0].in_features,1546)
        self.assertEqual(self.head[0].out_features,512)
        self.assertEqual(self.head[4].out_features,256)
        self.assertEqual(len(self.head.vessel_encoder.blocks),2)
        self.assertTrue(all(block.heads == 4 and block.ffn[0].out_features == 512
                            for block in self.head.vessel_encoder.blocks))
        original = copy.deepcopy(self.head.original)
        with torch.no_grad():
            self.head.vessel_fusion.weight.zero_()
            torch.testing.assert_close(self.score(), original(self.pair), rtol=0, atol=0)
        self.assertIsNone(self.head._context)
        clone = copy.deepcopy(self.head)
        self.assertIsNone(clone._context)
        self.assertFalse(any("context" in name for name in clone.state_dict()))
        clone.vessel_fusion.weight.data.add_(1)
        self.assertFalse(torch.equal(clone.vessel_fusion.weight, self.head.vessel_fusion.weight))

    def test_complete_branch_and_original_trunk_receive_finite_gradients_and_optimizer_updates(self):
        self.head.train()
        self.batch.node_features.requires_grad_()
        before = {name:value.detach().clone() for name,value in self.head.named_parameters()}
        optimizer = torch.optim.AdamW(self.head.parameters(),lr=1e-3,weight_decay=0)
        score = self.score().flatten()
        loss = F.softplus(score[2]-score[0]) + F.softplus(score[3]-score[1])
        loss.backward()
        missing = [name for name,p in self.head.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [name for name,p in self.head.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        self.assertEqual(missing,[])
        self.assertEqual(nonfinite,[])
        self.assertTrue((self.batch.node_features.grad.abs().sum(1) > 0).all())
        optimizer.step()
        for module in ("original", "vessel_encoder", "vessel_readout.nodes", "vessel_readout.components",
                       "vessel_readout.query", "vessel_readout.fuse", "vessel_fusion"):
            self.assertTrue(any(not torch.equal(before[name],p.detach()) for name,p in self.head.named_parameters()
                                if name.startswith(module + ".")), module)
        self.assertIsNone(self.head._context)

    def test_candidate_permutation_preserves_exact_ownership_and_scoring_order(self):
        baseline = self.score()
        permutation = torch.tensor([3,0,2,1])
        changed = self.score(pair=self.pair[permutation],pos=self.pos[permutation],owner=self.owner[permutation])
        torch.testing.assert_close(changed,baseline[permutation],rtol=1e-5,atol=1e-6)

    def test_remote_disconnected_component_changes_own_patient_scores_only(self):
        baseline = self.score()
        changed = copy.deepcopy(self.batch)
        changed.node_features[5,8] += 12
        changed.node_features[5,3] += 3
        alternate = self.score(batch=changed)
        self.assertGreater((alternate[[0,2]]-baseline[[0,2]]).abs().max().item(),1e-7)
        torch.testing.assert_close(alternate[[1,3]],baseline[[1,3]],rtol=0,atol=0)
        self.assertEqual(self.batch.edge_index.shape[1],8)
        self.assertEqual(self.batch.component_patient_index.numel(),5)

    def test_physical_candidate_geometry_and_source_conditioning_change_scores(self):
        baseline = self.score()
        changed = self.pos.clone(); changed[0] += torch.tensor([21.,11.,4.])
        self.assertGreater((self.score(pos=changed)[0]-baseline[0]).abs().item(),1e-8)
        changed_pair = self.pair.clone(); changed_pair[0,:128] += 2
        self.assertGreater((self.score(pair=changed_pair)[0]-baseline[0]).abs().item(),1e-7)

    def test_missing_context_and_foreign_component_edges_fail_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError,"explicit complete vessels"):
            self.head(self.pair)
        bad = copy.deepcopy(self.batch)
        bad.edge_index[:,0] = torch.tensor([0,5])
        with self.assertRaisesRegex(RuntimeError,"disconnected native components"):
            self.score(batch=bad)
        self.assertIsNone(self.head._context)
        bad = copy.deepcopy(self.batch)
        bad.edge_index[:,0] = torch.tensor([0,6])
        with self.assertRaisesRegex(RuntimeError,"patient boundaries"):
            self.score(batch=bad)
        self.assertIsNone(self.head._context)

    def test_context_nested_restore_and_exception_cleanup(self):
        with vessel_score_context(self.model,self.batch,self.pos,self.owner):
            outer = self.head._context
            with self.assertRaisesRegex(ValueError,"UNIT deliberate failure"):
                with vessel_score_context(self.model,self.batch,self.pos+1,self.owner):
                    raise ValueError("UNIT deliberate failure")
            self.assertIs(self.head._context,outer)
        self.assertIsNone(self.head._context)

    def test_tiled_all_node_attention_matches_untiled_softmax_and_gradients(self):
        dense = PhysicalVesselAttention(query_chunk_size=4,node_chunk_size=9).eval()
        tiled = PhysicalVesselAttention(query_chunk_size=1,node_chunk_size=1).eval()
        tiled.load_state_dict(dense.state_dict(),strict=True)
        queries = torch.randn(4,128,requires_grad=True)
        nodes = torch.randn(9,128,requires_grad=True)
        args = (queries,nodes,self.pos,self.batch.pos_mm,self.batch.node_features[:,3],self.owner,self.batch.node_patient_index)
        actual = tiled(*args)
        expected = dense(*args)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        left = torch.autograd.grad(actual.square().sum(),(queries,nodes),retain_graph=True)
        right = torch.autograd.grad(expected.square().sum(),(queries,nodes))
        for one,two in zip(left,right):
            torch.testing.assert_close(one,two,rtol=1e-4,atol=1e-5)


@unittest.skipUnless(CHECKPOINT.is_file(),"Pinned official full STU-Net-S checkpoint UNIT fixture unavailable")
class FullSTUHierarchyVesselUNIT(unittest.TestCase):
    """Actual original L0/L1/L2 plus new branch, using explicit synthetic UNIT CT."""

    @classmethod
    def setUpClass(cls):
        import tests.test_v24_gt_blind as fixtures
        from hiercp_v1x.v24_model import build_gt_free_model
        from hiercp_v1x.v24_stunet import install_stunet_l0
        fixtures.RecipientGTBlindDebug.setUpClass()
        cls.fixture = fixtures.RecipientGTBlindDebug
        torch.manual_seed(42)
        cls.net = build_gt_free_model(cls.fixture.model_config,original_snapshot_root=cls.fixture.snapshot)
        install_stunet_l0(cls.net,CHECKPOINT)
        original = cls.net.score_head
        original_weights = {name:value.detach().clone() for name,value in original.named_parameters()}
        install_vessel_branch(cls.net,query_chunk_size=2,node_chunk_size=3)
        cls.original_preserved = original is cls.net.score_head.original and all(
            torch.equal(original_weights[name],value) for name,value in original.named_parameters())

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_full_real_module_path_loss_backward_optimizer_and_parameter_contract(self):
        from hiercp_v1x.v24_inputs import collate,materialize_pair
        from torch_geometric.data import Batch
        batch = collate([(materialize_pair(record,epoch=29),i) for i,record in enumerate(self.fixture.records)])
        self.net.eval(); self.net.zero_grad(set_to_none=True)
        local = self.net.local_encoder
        source,target = local.encode_dense_maps(batch.source_patches,batch.source_index,batch.target_patches)
        fields = local.forward_graph(batch.graph,source.index_select(0,batch.graph_observation_index),
                                     target.index_select(0,batch.graph_observation_index))
        views = tuple({key:value[view::2] for key,value in fields.items()} for view in (0,1))
        merged = {key:(views[0][key]+views[1][key])/2 for key in fields}
        upper = SimpleNamespace(patient_batch=Batch.from_data_list([self.fixture.graph.clone()]),
            prototype_batch=Batch.from_data_list([self.fixture.prototype.clone()]),counts=(len(self.fixture.rows),))
        vessels = vessel_fixture_UNIT()
        vessels = SimpleNamespace(node_features=vessels.node_features[:6],pos_mm=vessels.pos_mm[:6],
            edge_index=vessels.edge_index[:,torch.tensor([0,1,2,4,5,6])],
            edge_attr=vessels.edge_attr[torch.tensor([0,1,2,4,5,6])],
            node_patient_index=vessels.node_patient_index[:6],component_index=vessels.component_index[:6],
            component_patient_index=vessels.component_patient_index[:3],case_ids=("UNIT_val_recipient",))
        pos = torch.tensor([row["center"] for row in self.fixture.rows],dtype=torch.float32) * 1.5
        with vessel_score_context(self.net,vessels,pos,torch.zeros(len(pos),dtype=torch.long)):
            scores = self.net._score_upper(upper,merged)[0]
        loss = F.softplus(scores[1:]-scores[0]).mean() + .1 * self.net._view_consistency(*views)
        loss.backward()
        self.assertTrue(self.original_preserved)
        missing = [name for name,p in self.net.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [name for name,p in self.net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        self.assertEqual(missing,[])
        self.assertEqual(nonfinite,[])
        count = sum(p.numel() for p in self.net.parameters())
        self.assertEqual(count,self.net.v24_model_contract["parameters"])
        self.assertEqual(count,self.net.v30_model_contract["parameters"])
        self.assertEqual(len(self.net.local_encoder.dense_encoder.conv_blocks_context),6)
        self.assertEqual(len(self.net.patient_encoder.blocks),2)
        self.assertEqual(len(self.net.prototype_encoder.blocks),2)
        optimizer = torch.optim.AdamW(self.net.parameters(),lr=1e-4,weight_decay=0)
        before = self.net.score_head.vessel_fusion.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before,self.net.score_head.vessel_fusion.weight))
        self.assertIsNone(self.net.score_head._context)


if __name__ == "__main__":
    unittest.main()
