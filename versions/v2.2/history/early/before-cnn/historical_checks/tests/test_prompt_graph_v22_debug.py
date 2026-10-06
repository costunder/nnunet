"""DEBUG: CNN-free inputs, unchanged geometry/targets, actual gradient path."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from hiercp_v22 import PIPELINE_VERSION
from hiercp_v22.contracts import load_config,verify_v1,validate_encoder_config
from hiercp_v22.data import LocalBatch,materialize,collate
from hiercp_v22.model import PromptGraphModel,context_descriptor,prompt_loss
from hiercp_v22.migration import convert_record
from hiercp_v22.storage import GraphWriter,load_record
from hiercp_v2.data import LocalBatch as OldBatch, materialize as old_materialize
from hiercp_v2.model import context_descriptor as old_descriptor
from test_hierarchy_model_debug import debug_batch
from test_prompt_graph_v22_alignment_debug import memory_fixture
from hiercp_v22.features import WIDTH


def fixtures():
    f=debug_batch(counts=(2,2),patch_size=48)
    graphs=f.local_batch.to_data_list()
    for graph in graphs:
        for kind in graph.node_types:
            # Explicit synthetic DEBUG wiring fixture; production features come from CT.
            graph[kind].observed=torch.randn(graph[kind].num_nodes,WIDTH)
    return None,collate([(g,i) for i,g in enumerate(graphs)])


class RawGraphV22DebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(4)
        cls.cfg,cls.base=load_config()
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_preserved_versions_and_scale(self):
        verify_v1()
        self.assertEqual(self.base['model']['hidden_dim'],128)
        self.assertEqual(self.base['model']['local_layers'],3)
        self.assertEqual((self.cfg['gnn_epochs'],self.cfg['candidate_count'],self.cfg['nnunet_epochs']),(40,128,250))
        bad=dict(self.cfg,local_encoder='dense_cnn')
        with self.assertRaises(ValueError):validate_encoder_config(bad)

    def test_descriptor_uses_only_active_observations(self):
        _,new=fixtures()
        original=context_descriptor(new).clone()
        self.assertEqual(original.shape,(4,144))
        self.assertFalse(hasattr(new,'source_patches'))
        self.assertFalse(hasattr(new,'patch_moments'))
        for kind in new.graph.node_types:new.graph[kind].x[:,:13]+=100
        torch.testing.assert_close(original,context_descriptor(new),rtol=0,atol=0)
        new.to('cpu')
        torch.testing.assert_close(original,context_descriptor(new))

    def test_CNN_absent_raw_observation_intervention_and_batching(self):
        _,batch=fixtures();torch.manual_seed(7)
        model=PromptGraphModel(self.cfg,self.base,patient_ids=['A','B','C']).eval()
        self.assertFalse(any(isinstance(m,nn.Conv3d) for m in model.modules()))
        self.assertFalse(any('dense_encoder' in n for n,_ in model.named_parameters()))
        with torch.no_grad():before=model.encode_local(batch)
        graph=copy.deepcopy(batch.graph)
        for kind in graph.node_types:graph[kind].observed[:,0]+=1
        altered=LocalBatch(graph,batch.indices)
        with torch.no_grad():after=model.encode_local(altered)
        self.assertFalse(torch.allclose(before,after))
        for edge in graph.edge_types:torch.testing.assert_close(graph[edge].edge_index,batch.graph[edge].edge_index)
        singles=[]
        with torch.no_grad():
            for i,g in enumerate(batch.graph.to_data_list()):
                singles.append(model.encode_local(collate([(g,i)])))
        torch.testing.assert_close(before,torch.cat(singles),rtol=3e-5,atol=3e-6)

    def test_no_CNN_full_width_backward_and_update(self):
        _,batch=fixtures();model=PromptGraphModel(self.cfg,self.base,patient_ids=['A','B','C']).train()
        memory=memory_fixture();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        output=model.forward_tasks(memory,model.encode_local(batch),torch.tensor([0,1,1,2]))
        loss,_=prompt_loss(output,context_descriptor(batch),torch.tensor([1,-1,0,1]),self.cfg['loss_weights'],self.cfg['temperature'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([n for n,p in model.named_parameters() if p.grad is None],[])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        prefixes=['local.project.','local.blocks.','data_projection.','patient_labels','local_decoder.','label_decoder.']
        prefixes += [f'task.to_label.{i}.' for i in range(2)]+[f'task.to_data.{i}.' for i in range(2)]+[f'alignment.layers.{i}.' for i in range(2)]
        for prefix in prefixes:
            self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n.startswith(prefix)),prefix)

    def test_fresh_geometry_writer_loader_equivalence_to_v21(self):
        import numpy as np
        from hiercp.common import LoadedCase,CasePaths
        from hiercp_v2.data import local_record as old_record
        from hiercp_v22.data import local_record,sources,LocalDataset
        from hiercp_v22.training import full_inventory
        grid=np.indices((64,64,64));organ=((grid-32)**2).sum(0)<28**2
        tumor=((grid-np.array([25,30,32])[:,None,None,None])**2).sum(0)<4**2
        labels=np.zeros(organ.shape,np.uint8);labels[organ]=1;labels[tumor]=2
        image=(40+grid[0]+grid[1]*.3-grid[2]*.1).astype(np.float32)
        case=LoadedCase(CasePaths('DEBUG_GEOMETRY',Path('unused'),Path('unused')),image,labels,
                        np.eye(4),np.eye(4),None,None,np.ones(3),{}, {})
        source,_=sources(case,self.base['cache']['source_pad'],20)[0]
        before=old_record(case,source,(38,32,32),self.base)
        fresh=local_record(case,source,(38,32,32),self.base)
        def same(a,b):
            if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
            elif isinstance(a,dict):
                self.assertEqual(set(a),set(b))
                for key in a:same(a[key],b[key])
            else:self.assertEqual(a,b)
        for branch in ('source_local','target_local'):
            same(before[branch]['counts'],fresh[branch]['counts'])
            same(before[branch]['edges'],fresh[branch]['edges'])
            for kind in fresh[branch]['nodes']:
                a,b=before[branch]['nodes'][kind],fresh[branch]['nodes'][kind]
                same(a['pos_mm'],b['pos_mm'])
                self.assertEqual(b['observed'].shape[1],WIDTH)
        with self.assertRaisesRegex(ValueError,'Raw CT rebuild'):convert_record(before)
        with tempfile.TemporaryDirectory(prefix='raw_v22_DEBUG_') as name:
            root=Path(name);writer=GraphWriter(root,minimum_free_bytes=0)
            saved=writer.write('a.pt.gz',fresh,('A',1));saved2=writer.write('b.pt.gz',fresh,('A',1))
            self.assertEqual(saved['shared_source'],saved2['shared_source'])
            same(fresh,load_record(root,saved['path']))
            ds=LocalDataset(root,[dict(id='debug',**saved)])
            new=collate([ds[0],ds[0]])
            self.assertEqual(context_descriptor(new).shape,(2,144))
            self.assertNotIn('source_patch',fresh)
            same(materialize(fresh).edge_index_dict,new.graph.to_data_list()[0].edge_index_dict)
            self.assertEqual(full_inventory(root,[dict(id='a',**saved)])[0]['bytes'],saved['bounds']['bytes'])
            shared=root/saved['shared_source']['path']
            with shared.open('ab') as f:f.write(b'DEBUG corruption')
            with self.assertRaisesRegex(ValueError,'SHA'):load_record(root,saved['path'])

    def test_partial_cache_and_old_weights_rejected(self):
        from hiercp_v22.migration import convert_cache
        from hiercp_v22.contracts import write_new,validate_checkpoint
        with tempfile.TemporaryDirectory(prefix='v22_DEBUG_partial_') as name:
            root=Path(name);write_new(root/'index.json',{'complete':False})
            with self.assertRaisesRegex(ValueError,'Raw CT rebuild'):
                convert_cache(root/'index.json',root/'converted',self.cfg,self.base)
            self.assertFalse((root/'converted').exists())
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            validate_checkpoint({'format':'hiercp_prompt_graph_v2_cross_patient_r1','complete':True})


if __name__=='__main__':unittest.main()
