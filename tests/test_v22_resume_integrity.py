"""Explicit CPU fixtures for B01--B03; not a training/efficacy benchmark."""
import copy
import random
import unittest
import numpy as np
import torch
from tools.v22_artifacts import best_snapshot,tree_hash
from tools.v22_resume_integrity import (rng_contract,rng_hash,validate_rng,generation,
    start_memory,seal_resume,validate_integrity,validate_plan,plan_generation)


def rng():
    return dict(torch=torch.get_rng_state(),cuda=[],numpy=np.random.get_state(),python=random.getstate())


class IntegrityTests(unittest.TestCase):
    def fixture(self):
        net=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
        identity=dict(config={'gnn_epochs':2},run_id='fixture')
        best=best_snapshot(net,1,.3,identity)
        state=dict(phase='final_memory',epoch=2,step=4,memory_next=0,memory_work=None,
            memory={'embeddings':torch.ones(4,128)},plan=None,plan_generation=None)
        state['memory_generation']=dict(run_id='fixture',phase='refresh_memory',epoch=1,step=4,
            model_sha256='a'*64)  # Deliberately older/worse final-epoch model.
        value=dict(state=state,model=copy.deepcopy(best['state_dict']),best_snapshot=best,
                   run_id='fixture',rng=rng(),rng_contract=rng_contract('cpu'))
        seal_resume(value)
        return value

    def test_earlier_best_start_partial_and_completed_prefix(self):
        value=self.fixture();validate_integrity(value,[])
        for count in (2,4):
            start_memory(value['state'],value['run_id'],value['model'])
            value['state'].update(memory_work=torch.ones(count,128),memory_next=count)
            seal_resume(value);validate_integrity(value,[])
            start_memory(value['state'],value['run_id'],value['model'])

    def test_finite_final_weight_buffer_and_prefix_mutations_rejected(self):
        for change in ('weight','buffer','prefix','generation','best_replacement'):
            value=self.fixture();s=value['state'];start_memory(s,value['run_id'],value['model'])
            s.update(memory_work=torch.ones(2,128),memory_next=2);seal_resume(value)
            if change=='weight':value['model']['0.weight'][0,0]+=.1
            elif change=='buffer':value['model']['1.running_mean'][0]+=.1
            elif change=='prefix':s['memory_work'][0,0]+=.1
            elif change=='generation':s['memory_work_generation']['step']-=1;seal_resume(value)
            else:value['best_snapshot']['model_sha256']='f'*64
            with self.subTest(change=change),self.assertRaises(ValueError):validate_integrity(value,[])

    def test_optimization_weights_need_not_equal_best_or_reference(self):
        value=self.fixture();value['state']['phase']='optimization'
        value['model']['0.weight']+=1
        seal_resume(value);validate_integrity(value,[])

    def plan(self):
        from hiercp_v222.clustering import fit_prototypes
        torch.manual_seed(4)
        owners=torch.tensor([0,0,1,1,2,2]);classes=torch.tensor([0,1,0,1,0,1])
        p=fit_prototypes(torch.randn(3,2,128),owners,classes)
        p['support_embeddings']=torch.randn(6,128)
        return p

    def test_valid_frozen_cluster_plan(self):validate_plan(self.plan())

    def test_same_group_cursor_and_next_group_teacher_generation(self):
        from types import SimpleNamespace
        from hiercp_v222.v1_training import groups
        from hiercp_v222.clustering import fit_prototypes
        from tools.v222_review_contracts import grouped_support
        rows=[dict(patient_group=g,bounds={'edges':j}) for g in 'ABCD' for j in range(4)]
        memory=dict(embeddings=torch.randn(16,128),owners=torch.arange(4).repeat_interleave(4),
            classes=torch.tensor([0,1]*8),case_ids=list('abcd'),patient_groups=list('ABCD'),donor_groups=['external']*16)
        value=self.fixture();s=value['state'];value['config']={'seed':42};s.update(epoch=0,step=0,phase='optimization',
            memory=memory,batch=2,next_batch=0)
        s['memory_generation']=dict(run_id='fixture',phase='initial_memory',epoch=0,step=0,model_sha256='a'*64)
        order=list(groups(SimpleNamespace(rows=rows),2,42,0))
        def new_plan(cursor):
            group=rows[order[cursor][0]]['patient_group'];emb,owners,classes=grouped_support(memory,group)
            p=fit_prototypes(torch.randn(3,2,128),owners,classes);p['support_embeddings']=emb
            s.update(plan=p,last_group=group)
            s['plan_generation']=plan_generation(s,value['run_id'],value['model'],group)
        new_plan(0)
        for cursor in (1,2):
            s.update(step=cursor,next_batch=cursor);seal_resume(value);validate_integrity(value,rows)
        self.assertEqual(s['plan_generation']['teacher_step'],0)
        new_plan(2);s.update(step=3,next_batch=3);seal_resume(value);validate_integrity(value,rows)
        self.assertEqual(s['plan_generation']['teacher_step'],2)
        s['plan_generation']['teacher_step']=3;seal_resume(value)
        with self.assertRaises(ValueError):validate_integrity(value,rows)

    def test_all_cluster_structure_corruption(self):
        original=self.plan()
        for key in ('active','assignment','membership','mass','prototype_classes','centers','log_prior','alignment_row_weights','support_embeddings','owners','classes'):
            p=copy.deepcopy(original)
            if key=='prototype_classes':p[key]=1-p[key]
            elif key in ('active','assignment','owners','classes'):p[key][0]=-1
            elif key=='support_embeddings':p[key]=p[key].double()
            else:p[key].flatten()[0]+=.3
            with self.subTest(key=key),self.assertRaises(ValueError):validate_plan(p)

    def test_finite_center_swap_and_teacher_metadata_hash_rejected(self):
        value=self.fixture();value['state'].update(phase='optimization',plan=self.plan(),plan_generation={'fixture':True})
        seal_resume(value)
        for key in ('centers','prototype_classes','support_embeddings'):
            bad=copy.deepcopy(value);bad['state']['plan'][key]=bad['state']['plan'][key].flip(0)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_integrity(bad,[])
        value['state']['plan_generation']['fixture']=False
        with self.assertRaises(ValueError):validate_integrity(value,[])


class RNGTests(unittest.TestCase):
    def test_cpu_only_legal_and_validation_does_not_consume_global_rng(self):
        state=rng();contract=rng_contract('cpu');digest=rng_hash(state)
        validate_rng(state,contract,digest)
        self.assertEqual(rng_hash(rng()),digest)

    def test_cuda_declared_empty_missing_truncated_and_bad_bytes(self):
        state=rng();state['cuda']=[torch.arange(16,dtype=torch.uint8)]
        contract=rng_contract('cpu');contract.update(training_device='cuda',cuda_device_count=1,
            devices=[dict(logical_index=0,name='explicit CPU fixture',uuid='fixture',total_memory=1024,state_shape=[16])])
        digest=rng_hash(state);validate_rng(state,contract,digest)
        for change in ('empty','missing','shape','dtype','bytes','mapping'):
            r=copy.deepcopy(state);c=copy.deepcopy(contract)
            if change=='empty':r['cuda']=[]
            elif change=='missing':del r['cuda']
            elif change=='shape':r['cuda'][0]=r['cuda'][0][:-1]
            elif change=='dtype':r['cuda'][0]=r['cuda'][0].long()
            elif change=='bytes':r['cuda'][0][0]+=1
            else:c['devices'][0]['logical_index']=1
            with self.subTest(change=change),self.assertRaises(ValueError):validate_rng(r,c,digest)

    def test_torch_numpy_python_malformed_or_finite_changed(self):
        state=rng();contract=rng_contract('cpu');digest=rng_hash(state)
        for change in ('torch_shape','torch_dtype','numpy_shape','numpy_bytes','python_shape','python_bytes'):
            r=copy.deepcopy(state)
            if change=='torch_shape':r['torch']=r['torch'][:-1]
            elif change=='torch_dtype':r['torch']=r['torch'].float()
            elif change=='numpy_shape':r['numpy']=(r['numpy'][0],r['numpy'][1][:-1],*r['numpy'][2:])
            elif change=='numpy_bytes':r['numpy'][1][0]+=np.uint32(1)
            elif change=='python_shape':r['python']=(3,(),None)
            else:r['python']=(r['python'][0],r['python'][1],.2)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_rng(r,contract,digest)


if __name__=='__main__':unittest.main()
