"""Actual CT DEBUG: continuous/resumed gradients, final boundaries and corrupt admission."""
from pathlib import Path
import sys,json,copy,random,shutil,argparse
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch,numpy as np
    from tools.run_v222_process_runtime import main as run
    from tools.v22_artifacts import tree_hash,validate_artifact
    from tools.v22_resume_integrity import rng_hash
    from tools.v222_runtime_execution import AsyncSaver
    from tools import v22_ranking_training as training
    from tools.v22_debug_profile import RepairDataset
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path);parser.add_argument('root',type=Path)
    parser.add_argument('--all-resume-phases',action='store_true',help='C01: capture/admit/resume all five production phases in isolated DEBUG')
    args=parser.parse_args();cache=args.cache;root=args.root;paused=root/'paused'
    saved=torch.load(paused/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    if not saved['debug']:raise ValueError('DEBUG checkpoint required')
    original_save=AsyncSaver.save;original_step=training.optimizer_step
    original_clip=torch.nn.utils.clip_grad_norm_
    captured={};records=[]
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        # Recorded by named parameters in the step wrapper before clipping.
        captured['gradients']={name:tree_hash(p.grad) for name,p in captured['net'].named_parameters()}
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise AssertionError('Missing/nonfinite gradient')
        return original_clip(parameters,*args,**kwargs)
    def step(net,opt,*args,**kwargs):
        captured['net']=net
        result=original_step(net,opt,*args,**kwargs)
        records.append(dict(loss=result[0],gradients=captured.pop('gradients'),model=tree_hash(net.state_dict()),adam=tree_hash(opt.state_dict())))
        captured.pop('net');return result
    def save(self,state):
        receipt=original_save(self,state)
        if args.all_resume_phases:
            phase_path=self.root/f'phase_{state["phase"]}_DEBUG.pt'
            if not phase_path.exists():self.flush();shutil.copyfile(self.root/'checkpoint_latest.pt',phase_path)
        if state['phase']=='final_memory':
            name='start' if state['memory_work'] is None else 'partial' if state['memory_next']<8 else 'complete'
            path=self.root/f'final_{name}_DEBUG.pt'
            if not path.exists():self.flush();shutil.copyfile(self.root/'checkpoint_latest.pt',path)
        return receipt
    def execute(output,extra,capture=False):
        argv=sys.argv;sys.argv=['process','--debug-profile','review_repair','--cache',str(cache),
            '--output',str(output),'--workspace-mib','256','--release-unused',*extra]
        try:
            if capture:
                with patch.object(AsyncSaver,'save',save),patch.object(training,'optimizer_step',step),patch.object(torch.nn.utils,'clip_grad_norm_',clip):run()
            else:run()
        finally:sys.argv=argv
    execute(root/'resumed',['--resume',str(paused/'checkpoint_latest.pt')],True)
    resumed=records.copy();records.clear()
    np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    execute(root/'uninterrupted',['--calibration',str(paused)],True)
    continuous=records.copy()
    if resumed!=continuous[1:]:raise AssertionError('Next-update loss/gradients/model/Adam differ')
    a=torch.load(root/'resumed/checkpoint_latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(root/'uninterrupted/checkpoint_latest.pt',map_location='cpu',weights_only=False)
    checks={k:tree_hash(a[k])==tree_hash(b[k]) for k in ('model','optimizer')}
    checks.update(rng=rng_hash(a['rng'])==rng_hash(b['rng']),support=tree_hash(a['state']['memory'])==tree_hash(b['state']['memory']))
    if not all(checks.values()):raise AssertionError(checks)
    dataset=RepairDataset(cache,'inner_train')
    def admit(v):
        identity=(v.get('best_snapshot') or {}).get('identity')
        if identity is None:
            keys=('training_objective','ranking_contract','feature_coordinates','artifact_contract','geometry_contract','run_id',
                'support_task_contract','config','base','source_identity','cache_sha256','debug','optimizer_contract','rng_contract')
            identity={k:v[k] for k in keys}
        validate_artifact(v,'resume',allow_debug=True,identity=identity,resume_rows=dataset.rows)
    admit(saved);admit(a)
    rejections=[]
    mutations={
        'CUDA_empty':lambda v:v['rng'].__setitem__('cuda',[]),
        'CUDA_missing':lambda v:v['rng'].pop('cuda'),
        'CUDA_truncated':lambda v:v['rng']['cuda'].__setitem__(0,v['rng']['cuda'][0][:-1]),
        'CUDA_dtype':lambda v:v['rng']['cuda'].__setitem__(0,v['rng']['cuda'][0].long()),
        'prototype_class_swap':lambda v:v['state']['plan'].__setitem__('prototype_classes',1-v['state']['plan']['prototype_classes']),
        'finite_center_swap':lambda v:v['state']['plan'].__setitem__('centers',v['state']['plan']['centers'].flip(0)),
        'teacher_step':lambda v:v['state']['plan_generation'].__setitem__('teacher_step',99),
    }
    for name,mutate in mutations.items():
        value=copy.deepcopy(saved);mutate(value)
        try:admit(value)
        except ValueError as error:rejections.append(dict(case=name,error=str(error)))
        else:raise AssertionError('Corrupt admission: '+name)
    for name in ('final_weight','final_prefix','final_generation'):
        value=copy.deepcopy(a)
        if name=='final_weight':next(t for t in value['model'].values() if t.is_floating_point()).flatten()[0]+=.1
        elif name=='final_prefix':value['state']['memory_work'][0,0]+=.1
        else:value['state']['memory_work_generation']['model_sha256']='f'*64
        try:admit(value)
        except ValueError as error:rejections.append(dict(case=name,error=str(error)))
        else:raise AssertionError('Corrupt admission: '+name)
    boundaries={}
    reference=torch.load(root/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
    for boundary in ('start','partial','complete'):
        source=root/'resumed'/f'final_{boundary}_DEBUG.pt'
        execute(root/f'boundary_{boundary}',['--resume',str(source)])
        actual=torch.load(root/f'boundary_{boundary}/checkpoint.pt',map_location='cpu',weights_only=False)
        boundaries[boundary]=all(tree_hash(actual[k])==tree_hash(reference[k]) for k in ('state_dict','memory'))
    if not all(boundaries.values()):raise AssertionError(boundaries)
    phase_resumes={};model_rejections=[]
    if args.all_resume_phases:
        for phase in ('initial_memory','optimization','refresh_memory','validation','final_memory'):
            source=root/'uninterrupted'/f'phase_{phase}_DEBUG.pt'
            good=torch.load(source,map_location='cpu',weights_only=False);admit(good)
            for mutation in ('finite_parameter','missing_model_hash'):
                bad=copy.deepcopy(good)
                if mutation=='finite_parameter':next(t for t in bad['model'].values() if t.is_floating_point()).flatten()[0]+=.125
                else:del bad['model_sha256']
                try:admit(bad)
                except ValueError as error:model_rejections.append(dict(phase=phase,mutation=mutation,error=str(error)))
                else:raise AssertionError('C01 corrupt model admitted: '+phase)
            execute(root/f'phase_resume_{phase}',['--resume',str(source)])
            actual=torch.load(root/f'phase_resume_{phase}/checkpoint.pt',map_location='cpu',weights_only=False)
            phase_resumes[phase]=all(tree_hash(actual[k])==tree_hash(reference[k]) for k in ('state_dict','memory'))
        if not all(phase_resumes.values()):raise AssertionError(phase_resumes)
    result=dict(debug=True,actual_CT=True,full_model=True,full_training=False,updates=4,
        next_group_loss_gradients_L0_L1_L2_model_Adam_bitwise=True,continuous_vs_resumed=checks,
        final_memory_boundary_resume_bitwise=boundaries,rejections=rejections,
        all_phase_resume_bitwise=phase_resumes,current_model_rejections=model_rejections,
        earlier_best_test='separate CPU fixture; this actual CT run has one epoch',
        same_group_probe='separate numerical probe; this lifecycle has one batch per group')
    (root/'integrity_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/'gradient_comparison_DEBUG.json').write_text(json.dumps(dict(resumed=resumed,continuous=continuous),indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
