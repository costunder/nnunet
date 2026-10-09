"""Official pretrained STU-S/full GT-blind hierarchy CPU UNIT, not CT results."""
from pathlib import Path
from types import SimpleNamespace
import copy
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'work/pretrained_l0_latency_DEBUG_20261008/assets/small_ep4k.model'


@unittest.skipUnless(CHECKPOINT.is_file(),'Pinned official STU-S checkpoint fixture unavailable')
class STUCompleteHierarchyDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tests.test_v24_gt_blind as fixtures
        from hiercp_v1x.v24_model import build_gt_free_model
        from hiercp_v1x.v24_stunet import install_stunet_l0
        from hiercp_v1x.v24_inputs import materialize_pair,collate
        fixtures.RecipientGTBlindDebug.setUpClass()
        cls.fixture=fixtures.RecipientGTBlindDebug
        torch.manual_seed(42)
        cls.net=build_gt_free_model(cls.fixture.model_config,original_snapshot_root=cls.fixture.snapshot)
        cls.audit=install_stunet_l0(cls.net,CHECKPOINT)
        cls.batch=collate([(materialize_pair(record,epoch=29),i) for i,record in enumerate(cls.fixture.records)])

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_full_pretrained_six_scales_to_L1_L2_all_gradients_optimizer_and_GT_invariance(self):
        from torch_geometric.data import Batch
        from hiercp_v1x.v24_inputs import recipient_context,tensor_digest
        from hiercp_v1x.v24_geometry import build_recipient_regions,build_upper_graphs
        self.net.eval();self.net.zero_grad(set_to_none=True)
        local=self.net.local_encoder; batch=self.batch; fixture=self.fixture
        self.assertEqual(self.audit['loaded_encoder_tensors'],60)
        self.assertEqual(self.audit['loaded_parameter_numel'],7185744)
        self.assertEqual(self.audit['tensor_coverage'],1.)
        self.assertFalse(any(name.startswith('blocks.') or 'seg_outputs' in name or 'localization' in name
                             for name,_ in local.named_parameters()))
        source,target=local.encode_dense_maps(batch.source_patches,batch.source_index,batch.target_patches)
        self.assertEqual(len(source.maps),6);self.assertEqual(len(target.maps),6)
        fields=local.forward_graph(batch.graph,source.index_select(0,batch.graph_observation_index),
                                  target.index_select(0,batch.graph_observation_index))
        self.assertEqual(len(fields),12)
        self.assertTrue(all(value.shape==(2*len(fixture.rows),128) for value in fields.values()))
        views=tuple({key:value[view::2] for key,value in fields.items()} for view in (0,1))
        merged={key:(views[0][key]+views[1][key])/2 for key in fields}
        upper=SimpleNamespace(patient_batch=Batch.from_data_list([fixture.graph.clone()]),
            prototype_batch=Batch.from_data_list([fixture.prototype.clone()]),counts=(len(fixture.rows),))
        scores=self.net._score_upper(upper,merged)[0]
        # Supervision is independently changed; all actual neural inputs and
        # original sampled coordinates remain those of the explicit CT/organ.
        rows=copy.deepcopy(fixture.rows)
        for row in rows:row['target']=1-row['target'];row['component']=12345
        context=recipient_context(fixture.recipient.case_id,fixture.CT,fixture.organ,fixture.spacing,
                                  fixture.recipient.image_affine)
        regions=build_recipient_regions(context,config=fixture.config,seed=42,ct_clip=fixture.clip)
        graph,prototype,audit=build_upper_graphs(context,fixture.donor,fixture.source,regions,
            fixture.donor_regions,rows,fixture.bank,config=fixture.config,ct_clip=fixture.clip,
            training_case_ids=(fixture.donor.paths.case_id,))
        self.assertEqual(tensor_digest((graph.to_dict(),prototype.to_dict())),
                         tensor_digest((fixture.graph.to_dict(),fixture.prototype.to_dict())))
        alternate=SimpleNamespace(patient_batch=Batch.from_data_list([graph]),
            prototype_batch=Batch.from_data_list([prototype]),counts=(len(rows),))
        torch.testing.assert_close(scores,self.net._score_upper(alternate,merged)[0],rtol=0,atol=0)
        self.assertFalse(audit['recipient_GT_used_in_forward'])
        loss=torch.nn.functional.softplus(scores[1:]-scores[0]).mean()+.1*self.net._view_consistency(*views)
        loss.backward()
        parameters=dict(self.net.named_parameters())
        missing=[name for name,p in parameters.items() if p.requires_grad and p.grad is None]
        nonfinite=[name for name,p in parameters.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
        self.assertEqual(missing,[]);self.assertEqual(nonfinite,[])
        for index,stage in enumerate(local.dense_encoder.conv_blocks_context):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in stage.parameters()),0.,str(index))
        original=[stage[0].conv1.weight.detach().clone() for stage in local.dense_encoder.conv_blocks_context]
        optimizer=torch.optim.AdamW(parameters.values(),lr=1e-4,weight_decay=1e-4)
        optimizer.step()
        for index,stage in enumerate(local.dense_encoder.conv_blocks_context):
            self.assertFalse(torch.equal(original[index],stage[0].conv1.weight),str(index))
        print('STU_HIERARCHY_CPU_UNIT',dict(parameters=sum(p.numel() for p in parameters.values()),
            trainable_tensors=len(parameters),pretrained_encoder_tensors=60,pretrained_encoder_numel=7185744,
            input_shape=list(batch.target_patches.shape),local_nodes=sum(s.num_nodes for s in batch.graph.node_stores),
            local_edges=sum(s.num_edges for s in batch.graph.edge_stores),GT_invariance=True,
            all_gradients_finite_and_present=True,all_six_encoder_stages_updated=True),flush=True)

    def test_training_mode_nested_stage_and_outer_checkpoints_reach_every_parameter(self):
        """Actual complete hierarchy CPU UNIT; CUDA AMP/full4/8 still need calibration."""
        from torch_geometric.data import Batch
        from torch.utils.checkpoint import checkpoint
        from hiercp_v1x.v23_training import V23Scorer, FIELDS, patient_balanced_objective
        net=copy.deepcopy(self.net).train();net.zero_grad(set_to_none=True)
        provider=SimpleNamespace(ds=SimpleNamespace(rows=[dict(id=str(i)) for i in range(len(self.fixture.rows))]))
        scorer=V23Scorer(net,dict(inner_train=provider,inner_val=provider),lambda *a:None,
            physical_candidate_batch=32,checkpoint_local_chunks=True,amp=False,budget=None)
        upper=SimpleNamespace(patient_batch=Batch.from_data_list([self.fixture.graph.clone()]),
            prototype_batch=Batch.from_data_list([self.fixture.prototype.clone()]),counts=(len(self.fixture.rows),))
        stage_weights=[stage[0].conv1.weight.detach().clone() for stage in net.local_encoder.dense_encoder.conv_blocks_context]
        with patch('hiercp_v1x.v24_stunet.activation_checkpoint',wraps=checkpoint) as stages, \
             patch('hiercp_v1x.v23_training.activation_checkpoint',wraps=checkpoint) as outer:
            packed,consistency=scorer._encode(self.batch,training=True)
            fields=dict(zip(FIELDS,packed.split(128,dim=1)))
            scores=tuple(net._score_upper(upper,fields))
            loss,_=patient_balanced_objective(scores,[(0,)],consistency)
            loss.backward()
        self.assertEqual(outer.call_count,1)
        self.assertIs(outer.call_args.args[1],next(net.parameters()))
        self.assertFalse(outer.call_args.kwargs['use_reentrant'])
        invoked={id(call.args[0]) for call in stages.call_args_list}
        self.assertEqual(invoked,{id(stage) for stage in net.local_encoder.dense_encoder.conv_blocks_context})
        self.assertTrue(all(call.args[1].device.type=='cpu' and not call.kwargs['use_reentrant']
                            for call in stages.call_args_list))
        parameters=list(net.named_parameters())
        self.assertEqual(len(parameters),537)
        self.assertEqual([name for name,p in parameters if p.grad is None],[])
        self.assertTrue(all(torch.isfinite(p.grad).all() for _,p in parameters))
        optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
        optimizer.step()
        for initial,stage in zip(stage_weights,net.local_encoder.dense_encoder.conv_blocks_context):
            self.assertFalse(torch.equal(initial,stage[0].conv1.weight))
        print('STU_NESTED_CHECKPOINT_CPU_UNIT',dict(trainable_tensors=537,encoder_stages=6,
            outer_real_parameter_input=True,stage_checkpoint_calls=stages.call_count,
            all_gradients_finite_and_present=True,all_six_encoder_stages_updated=True,
            CUDA_AMP_or_full_native_physical_batch_verified=False),flush=True)


if __name__=='__main__':unittest.main()
