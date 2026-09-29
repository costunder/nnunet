"""Short same-query full-support/patient-episode CUDA comparison; disposable updates only."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from l0_regions import training as t
from l0_regions.training_data import RegionDataset,Budget,source_identity,sha,write_new
from l0_regions.support_episodes import PatientEpisodes,contract,verify_migration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('workers','support-patients'):p.add_argument('--'+name,type=int,required=True)
    for name in ('cuda-gib','rss-gib','resident-gib'):p.add_argument('--'+name,type=float,required=True)
    a=p.parse_args()
    if a.workers<1 or min(a.cuda_gib,a.rss_gib,a.resident_gib)<=0:raise ValueError('Explicit positive resource limits required')
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('Leave device headroom')
    previous_fraction=torch.cuda.get_per_process_memory_fraction()
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total);torch.set_num_threads(a.workers)
    before=sha(a.checkpoint);cache_hash=sha(a.cache)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    digest=saved.pop('content_sha256',None)
    if saved.get('format')!=t.FORMAT or digest!=t.hash_state(saved):raise ValueError('Checkpoint integrity/format mismatch')
    identity=saved['identity'];current=dict(identity,source=source_identity(),support_training=contract(a.support_patients))
    if 'support_training' in identity:
        if identity!=current:raise ValueError('Episodic checkpoint source/configuration differs')
    else:verify_migration(identity,current)
    if identity['cache_sha256']!=cache_hash or saved['state']['phase']!='optimization':raise ValueError('Matching cache and paused optimization checkpoint required')
    ds=RegionDataset(a.cache,'inner_train',identity['debug'],identity['profile_policy'])
    if identity['config']!=ds.meta['config'] or identity['base']!=ds.meta['base']:
        raise ValueError('Checkpoint configuration differs from its cache')
    net=t.make_model(ds,budget,identity['debug'],identity['activation_storage'])
    state=t.tree_to(saved['state'],'cuda');net.local.dense_batch_size=state['batch']
    schedule=list(t.groups(ds,state['batch'],ds.meta['config']['seed'],state['epoch']))
    if state['next_batch']>=len(schedule):raise ValueError('No next query batch in saved epoch')
    ids=schedule[state['next_batch']];memory=state['memory'];group=ds.rows[ids[0]]['patient_group']
    episodes=PatientEpisodes(ds.rows,schedule,a.support_patients,ds.meta['config']['seed'],state['epoch']).bind(memory)
    loader=t.Loader(ds,a.workers,int(a.resident_gib*2**30))
    start=time.perf_counter();cpu=loader.get(ids);query=cpu.to('cuda');torch.cuda.synchronize()
    load_transfer=time.perf_counter()-start
    weights=torch.bincount(memory['classes'],minlength=2).float();weights=weights.sum()/(2*weights)
    context=t.RankingContext(ds,memory);rows=[]
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        for arm in ('full_support','patient_episode'):
            support=t.support_for_recipient(memory,group) if arm=='full_support' else episodes.support(group)
            net.load_state_dict(saved['model'],strict=True);net.eval()
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():plan=net.fit_support_clusters(*support)
            torch.cuda.synchronize();plan_time=time.perf_counter()-start
            for repeat in range(4):
                net.load_state_dict(saved['model'],strict=True);net.train()
                optimizer=torch.optim.AdamW(net.parameters(),lr=ds.meta['base']['training']['lr'],
                    weight_decay=ds.meta['base']['training']['weight_decay'],fused=ds.meta['base']['training']['fused_optimizer'])
                optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));optimizer.zero_grad(set_to_none=True)
                t.restore_rng(saved['rng']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                loss,_=t.forward_loss(net,query,support,plan,memory['classes'][ids],weights,context,t.rank_config(),indices=ids)
                torch.cuda.synchronize();forward=time.perf_counter()-start;back_start=time.perf_counter()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite diagnostic loss')
                loss.backward();torch.cuda.synchronize();backward=time.perf_counter()-back_start
                t.gradient_check(net)
                torch.nn.utils.clip_grad_norm_(net.parameters(),ds.meta['base']['training']['grad_clip'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start;budget.check()
                row=dict(arm=arm,warmup=repeat==0,query_records=len(ids),physical_batch=state['batch'],
                    support_records=len(support[0]),support_patients=int(support[1].max())+1,
                    forward_seconds=forward,backward_seconds=backward,update_seconds=elapsed,
                    plan_seconds=plan_time,peak_bytes=torch.cuda.max_memory_allocated(),loss=float(loss.detach()),
                    all_parameter_gradients_present_and_finite=True)
                print(json.dumps(row),flush=True);rows.append(row)
                del loss,optimizer
            del support,plan
        if before!=sha(a.checkpoint) or cache_hash!=sha(a.cache):raise RuntimeError('Source artifacts changed during diagnostic')
        means={arm:{key:statistics.mean(r[key] for r in rows if r['arm']==arm and not r['warmup'])
                    for key in ('forward_seconds','backward_seconds','update_seconds','peak_bytes')}
               for arm in ('full_support','patient_episode')}
        write_new(a.output/'report.json',dict(debug=True,full_training=False,production_ready=False,
            gpu=torch.cuda.get_device_name(),source=source_identity(),checkpoint_sha256=before,cache_sha256=cache_hash,
            saved_step=state['step'],query_indices=ids,load_transfer_seconds=load_transfer,rows=rows,means=means,
            schedule=episodes.audit,scope='Same next query and saved weights/Adam/RNG; different support training objective. One warmup + three disposable repeats per arm.',
            excluded='epoch refresh, validation, checkpoint copy/hash/disk, loader steady-state; no epoch speed claim',
            checkpoints_written=False,source_artifacts_preserved=True))
    finally:torch.cuda.set_per_process_memory_fraction(previous_fraction)

if __name__=='__main__':main()
