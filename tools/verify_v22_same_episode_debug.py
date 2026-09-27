"""Actual CT numerical probe: a saved teacher remains frozen after an update.

Repeats one real batch explicitly to exercise the same-episode branch; this is
not an epoch coverage test, new experiment data, or a production training run.
"""
from pathlib import Path
import sys,json,io,copy
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch,psutil
    from hiercp_v222 import v1_execution as ex
    from tools.v22_artifacts import tree_hash,validate_artifact
    from tools.v22_resume_integrity import rng_hash
    from tools.v22_debug_profile import RepairDataset
    from tools.v222_runtime_cache import CachedPairLoader
    from tools.v222_review_contracts import installed,grouped_support
    from tools.v22_rank_objective import RankingContext,configuration
    from tools.v22_ranking_steps import optimizer_step
    import hiercp.model as implementation
    cache=Path(sys.argv[1]);checkpoint=Path(sys.argv[2]);output=Path(sys.argv[3])
    output.mkdir(parents=True,exist_ok=False)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not saved['debug'] or saved['state']['phase']!='optimization':raise ValueError('Mid-optimization DEBUG required')
    ds=RepairDataset(cache,'inner_train');cfg=saved['config'];base=saved['base']
    identity_keys=('training_objective','ranking_contract','feature_coordinates','artifact_contract','geometry_contract','run_id',
        'support_task_contract','config','base','source_identity','cache_sha256','debug','optimizer_contract','rng_contract')
    validate_artifact(saved,'resume',allow_debug=True,identity={k:saved[k] for k in identity_keys},resume_rows=ds.rows)
    ex.configure_runtime(base,cfg['seed']);torch.set_num_threads(psutil.cpu_count(logical=False))
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=256*1024**2
    with installed('stride4'):
        net=ex.model(cfg,base).cuda();net.local.dense_batch_size=saved['state']['batch'];net.train()
        opt=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
        state=ex.tree_to(saved['state'],'cuda');support=grouped_support(state['memory'],state['last_group'])
        ids=state['seen'][-state['batch']:]
        if any(ds.rows[i]['patient_group']!=state['last_group'] for i in ids):raise ValueError('Same group required')
        loader=CachedPairLoader(ds,saved['state']['workers'])
        try:payload=loader.make(ids,state['epoch'])
        finally:loader.close()
        context=RankingContext(ds,state['memory']);targets=torch.tensor([ds.rows[i]['target'] for i in ids],device='cuda')
        counts=torch.bincount(state['memory']['classes'],minlength=2).float();weights=counts.sum()/(2*counts)
        original_clip=torch.nn.utils.clip_grad_norm_;gradients={}
        def clip(parameters,*args,**kwargs):
            gradients.clear();gradients.update({n:tree_hash(p.grad) for n,p in net.named_parameters()})
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise AssertionError('Gradient missing/nonfinite')
            return original_clip(parameters,*args,**kwargs)
        def update(snapshot):
            net.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);ex.restore_rng(snapshot['rng'])
            plan=ex.tree_to(snapshot['state']['plan'],'cuda')
            before=tree_hash(plan)
            with patch.object(net,'fit_support_clusters',side_effect=AssertionError('Frozen teacher was regenerated')),patch.object(torch.nn.utils,'clip_grad_norm_',clip):
                result=optimizer_step(net,opt,payload,support,plan,targets,weights,base['training']['grad_clip'],context=context,settings=configuration(),check_gradients=True)
            if tree_hash(plan)!=before:raise AssertionError('Teacher changed during update')
            return dict(loss=result[0],gradients=gradients.copy(),model=tree_hash(net.state_dict()),adam=tree_hash(opt.state_dict()),rng=rng_hash(ex.rng_state()))
        first=update(saved)
        stream=io.BytesIO();torch.save(saved,stream);stream.seek(0)
        roundtrip=torch.load(stream,map_location='cpu',weights_only=False)
        second=update(roundtrip)
        if first!=second:raise AssertionError('Same-episode next update differs after serialization')
    result=dict(debug=True,actual_CT=True,full_model=True,fixture_scope='explicit repeat of one real query batch; not epoch coverage',
        physical_batch=len(ids),teacher_step=saved['state']['plan_generation']['teacher_step'],current_step=saved['state']['step'],
        teacher_regeneration_forbidden=True,loss_gradients_L0_L1_L2_Adam_model_RNG_bitwise_equal=True,
        gradient_parameter_count=len(first['gradients']),checks=first)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':main()
