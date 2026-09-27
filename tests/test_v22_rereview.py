"""CPU admission counterexamples; fixtures are not CT/GNN performance evidence."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.v22_artifacts import tree_hash,best_snapshot
from tools.v22_resume_state import optimizer_contract,validate_adam,validate_resume_state
from tools.v22_calibration import validate_calibration
from hiercp_v222.placement import PlacementSpec
from hiercp_v222.record_binding import bind_record
from tools.v22_rank_recommendation import recommend


class AdamTests(unittest.TestCase):
    def fixture(self,steps=2):
        torch.manual_seed(5);net=torch.nn.Linear(2,1);opt=torch.optim.AdamW(net.parameters(),lr=.01)
        contract=optimizer_contract(net,opt)
        for _ in range(steps):opt.zero_grad();net(torch.ones(3,2)).square().mean().backward();opt.step()
        return net,opt,contract

    def test_initial_and_updated_admission(self):
        for step in (0,1,2):
            _,opt,contract=self.fixture(step);state=opt.state_dict()
            validate_adam(state,contract,step,tree_hash(state))

    def test_moments_groups_and_hash_corruption(self):
        _,opt,contract=self.fixture();original=copy.deepcopy(opt.state_dict())
        for change in ('empty','missing','nan','inf','shape','dtype','step','order','hyper','hash'):
            with self.subTest(change=change):
                state=copy.deepcopy(original)
                if change=='empty':state['state']={}
                elif change=='missing':del state['state'][0]['exp_avg']
                elif change in ('nan','inf'):state['state'][0]['exp_avg'].flatten()[0]=float(change)
                elif change=='shape':state['state'][0]['exp_avg']=torch.zeros(1)
                elif change=='dtype':state['state'][0]['exp_avg']=state['state'][0]['exp_avg'].double()
                elif change=='step':state['state'][0]['step']+=1
                elif change=='order':state['param_groups'][0]['params'].reverse()
                elif change=='hyper':state['param_groups'][0]['lr']=.1
                digest='wrong' if change=='hash' else tree_hash(state)
                with self.assertRaises(ValueError):validate_adam(state,contract,2,digest)

    def test_real_adam_next_update_matches_after_resume(self):
        net,opt,contract=self.fixture();state=copy.deepcopy(opt.state_dict());weights=copy.deepcopy(net.state_dict())
        other=torch.nn.Linear(2,1);other.load_state_dict(weights);resumed=torch.optim.AdamW(other.parameters(),lr=.01)
        validate_adam(state,contract,2,tree_hash(state));resumed.load_state_dict(state)
        for model,optimizer in ((net,opt),(other,resumed)):
            optimizer.zero_grad();model(torch.ones(3,2)).square().mean().backward();optimizer.step()
        self.assertEqual(tree_hash(net.state_dict()),tree_hash(other.state_dict()))
        self.assertEqual(tree_hash(opt.state_dict()),tree_hash(resumed.state_dict()))


class CursorTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(patient_group='A',bounds={'edges':i+1}) for i in range(4)]
        best=best_snapshot(torch.nn.Linear(2,1),1,.4,{'config':{'gnn_epochs':2}})
        state=dict(epoch=1,step=2,next_batch=0,phase='optimization',batch=2,memory_batch=2,workers=0,
            memory={'fixture':True},memory_next=0,memory_work=None,seen=[],losses=[],alignment=[],plan=None,last_group=None,
            best=.4,best_path='audit-only',best_snapshot_meta={k:best[k] for k in ('epoch','metric','model_sha256')})
        return dict(config={'seed':42,'gnn_epochs':2},state=state,best_snapshot=best),rows

    def test_best_metric_future_epoch_and_terminal_contradictions(self):
        valid,rows=self.fixture();validate_resume_state(valid,rows)
        for field in ('best','meta','embedded','future','terminal_validation','cursor','seen','prefix','selected'):
            bad=copy.deepcopy(valid);s=bad['state']
            if field=='best':s['best']=.1
            elif field=='meta':s['best_snapshot_meta']['metric']=.1
            elif field=='embedded':bad['best_snapshot']['metric']=.1
            elif field=='future':bad['best_snapshot']['epoch']=2;s['best_snapshot_meta']['epoch']=2
            elif field=='terminal_validation':s.update(epoch=2,step=4,phase='validation')
            elif field=='cursor':s['next_batch']=9999
            elif field=='seen':s['seen']=[0]
            elif field=='prefix':s.update(memory_next=2,memory_work=None)
            elif field=='selected':s.update(epoch=2,step=4,phase='final_memory',selected_epoch=2)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_resume_state(bad,rows)

    def test_legal_terminal_transition_and_partial_final_memory(self):
        value,rows=self.fixture();s=value['state'];s.update(epoch=2,step=4)
        validate_resume_state(value,rows)
        s.update(phase='final_memory',selected_epoch=1,memory_next=2,memory_work=torch.ones(2,128))
        validate_resume_state(value,rows)
        s['memory_work'][0,0]=float('nan')
        with self.assertRaises(ValueError):validate_resume_state(value,rows)


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        identity={k:'fixture' for k in ('artifact_contract','geometry_contract','source_identity','cache_sha256','ranking_contract','training_objective','feature_coordinates','support_task_contract','config','base','debug')}
        old=dict(identity,calibration_runtime_sha256={'runtime':'hash'},parameters=10,train_samples=8,val_samples=2,gpu='fixture',
                 physical_batch=2,memory_physical_batch=2,workers=0)
        row=dict(physical_batch=2,accepted=True,executed=True,graphs_per_second=3.,peak_vram_bytes=10,budget_bytes=20,nodes=3,edges=4)
        reports={'memory_batch_calibration.json':[dict(row,training=False)],'training_batch_calibration.json':[dict(row,training=True)],
            'loader_calibration.json':dict(selected=0,reports=[dict(workers=0,seconds=1.,warm_seconds=.5,graphs_per_batch=8,batches_per_pass=4,producers=1)])}
        args=dict(identity=identity,parameters=10,train_samples=8,val_samples=2,gpu='fixture',runtime={'runtime':'hash'})
        return old,reports,args

    def test_valid_and_corrupted_reuse(self):
        old,reports,args=self.fixture();validate_calibration(old,reports,**args)
        for field in ('training_objective','feature_coordinates','calibration_runtime_sha256','physical_batch','dict','empty','nan','unexecuted','workers'):
            bad=copy.deepcopy(old);r=copy.deepcopy(reports)
            if field in ('training_objective','feature_coordinates','calibration_runtime_sha256'):bad[field]='wrong'
            elif field=='physical_batch':bad[field]=32
            elif field=='dict':r['training_batch_calibration.json']={}
            elif field=='empty':r['memory_batch_calibration.json']=[]
            elif field=='nan':r['training_batch_calibration.json'][0]['graphs_per_second']=float('nan')
            elif field=='unexecuted':r['training_batch_calibration.json'][0]['executed']=False
            elif field=='workers':bad[field]=8
            with self.subTest(field=field),self.assertRaises(ValueError):validate_calibration(bad,r,**args)


class RecommendationTests(unittest.TestCase):
    def fixture(self):
        case=SimpleNamespace(paths=SimpleNamespace(case_id='r'),image=np.zeros((9,9,9),np.float32),label=np.ones((9,9,9),np.uint8),
                             spacing=np.ones(3),image_affine=np.eye(4),label_affine=np.eye(4))
        specs=[PlacementSpec('r','d',1,(i,4,4),(1.,1.,1.),tuple(map(tuple,np.eye(4))),np.ones((3,3,3)),np.ones((3,3,3),bool),(1,1,1),tuple(map(tuple,np.eye(3)))) for i in (3,5)]
        records=[bind_record(dict(case_id='r',donor_case_id='d',component_id=1,center=list(s.center),placement=s.metadata(),target_patch=torch.zeros(1,2,2,2)),case) for s in specs]
        net=torch.nn.Linear(1,1).eval();net.ranking_identities={'cases':{'r':{'patient_group':'R'},'d':{'patient_group':'D'}}};net.ranking_split={'inner_train':['d']}
        net.prepare_support=lambda *args:{}
        net.local=lambda payload:torch.ones(len(payload.items),1)
        net.predict_embeddings=lambda x,state:{'logits':torch.cat([x*0,x],1)}
        opts=dict(query_group='R',batch_size=2,workers=0,recipient_case=case,placements=specs,min_liver_coverage=.9)
        return net,records,opts

    def run_api(self,net,records,opts):
        def batch(items):
            payload=SimpleNamespace(items=items);payload.to=lambda device:payload;return payload
        with patch('tools.v22_rank_recommendation.grouped_support',return_value=(None,None,None)), \
             patch('tools.v22_rank_recommendation.materialize',side_effect=lambda x:x),patch('tools.v22_rank_recommendation.collate',side_effect=batch):
            return recommend(net,records,{},**opts)

    def test_workers_empty_and_GT_control(self):
        net,records,opts=self.fixture();reference=None
        for workers in (0,1,2,4,8):
            result=self.run_api(net,records,dict(opts,workers=workers))
            if reference is None:reference=result
            self.assertEqual(result,reference)
        opts['recipient_case'].label[3,4,4]=2
        changed=self.run_api(net,records,opts)
        self.assertEqual([r['model_score'] for r in changed['ranked_candidates']],[r['model_score'] for r in reference['ranked_candidates']])
        self.assertTrue(self.run_api(net,[],dict(opts,placements=[]))['keep_original'])
        for bad in (dict(query_group='wrong'),dict(recipient_case=None),dict(min_liver_coverage=float('nan')),dict(workers=-1)):
            with self.assertRaises(ValueError):self.run_api(net,[],dict(opts,placements=[],**bad))

    def test_second_placement_event_content_and_frame(self):
        for change in ('affine','spacing','transform','donor_ct','target_patch','recipient_ct','organ'):
            for reverse in (False,True):
                net,records,opts=self.fixture();case=opts['recipient_case'];specs=opts['placements']
                if change=='affine':
                    affine=np.eye(4);affine[0,3]=100;specs[1]=replace(specs[1],affine=tuple(map(tuple,affine)))
                elif change=='spacing':specs[1]=replace(specs[1],spacing=(2.,1.,1.))
                elif change=='transform':specs[1]=replace(specs[1],transform=tuple(map(tuple,np.diag([-1.,1.,1.]))))
                elif change=='donor_ct':specs[1]=replace(specs[1],image=np.zeros((3,3,3)))
                elif change=='target_patch':records[1]['target_patch'][0,0,0,0]=1
                elif change=='recipient_ct':case.image[0,0,0]=1
                elif change=='organ':case.label[0,0,0]=0
                if change in ('affine','spacing','transform','donor_ct'):
                    records[1]['placement']=specs[1].metadata();bind_record(records[1],case)
                if reverse:records.reverse();specs.reverse()
                with self.subTest(change=change,reverse=reverse),self.assertRaises(ValueError):self.run_api(net,records,opts)


if __name__=='__main__':unittest.main()
