"""Three disposable updates: warmup, baseline, attributed backward.
Use an already PAUSED retained checkpoint and an otherwise idle assigned GPU.
No prepare, support refresh, production checkpoint, parameter changes or training.
"""
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('cuda-gib','rss-gib','resident-gib'):p.add_argument('--'+name,type=float,required=True)
    p.add_argument('--workers',type=int,required=True)
    a=p.parse_args()
    import torch
    import psutil
    from l0_regions import training as t
    from l0_regions.training_data import RegionDataset,Budget,sha,source_identity
    from l0_regions.execution_upgrade import verify_upgrade
    from tools.region_backward_attribution import Attribution
    from tools.compare_region_memory_debug import assert_state_close
    if a.output.exists():raise FileExistsError('New diagnostic output directory required')
    if a.workers<1 or min(a.cuda_gib,a.rss_gib,a.resident_gib)<=0:raise ValueError('Positive explicit resources required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('Leave device headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(a.workers)
    before=sha(a.checkpoint);cache_hash=sha(a.cache)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    digest=saved.pop('content_sha256',None)
    if t.hash_state(saved)!=digest or saved.get('format')!=t.FORMAT:raise ValueError('Checkpoint integrity/format mismatch')
    identity=saved['identity']
    if 'support_training' in identity:raise ValueError('This diagnostic assumes full support; use compare_region_support_debug.py for patient episodes')
    if identity.get('activation_storage')!='retained':raise ValueError('This backward attribution requires the current retained execution; no silent policy change')
    if identity['cache_sha256']!=cache_hash or saved['state']['phase']!='optimization':raise ValueError('Matching cache and optimization checkpoint required')
    current={**identity,'source':source_identity()}
    verified_revision=verify_upgrade(identity,current)
    ds=RegionDataset(a.cache,'inner_train',identity['debug'],identity['profile_policy'])
    if identity['base']!=ds.meta['base'] or identity['config']!=ds.meta['config']:raise ValueError('Checkpoint/cache config mismatch')
    net=t.make_model(ds,budget,identity['debug'],identity['activation_storage'])
    state=t.tree_to(saved['state'],'cuda');batch=state['batch'];net.local.dense_batch_size=batch
    schedule=list(t.groups(ds,batch,ds.meta['config']['seed'],state['epoch']))
    if state['next_batch']>=len(schedule):raise ValueError('No next optimization batch; no automatic epoch advancement')
    ids=schedule[state['next_batch']];memory=state['memory']
    loader=t.Loader(ds,a.workers,int(a.resident_gib*2**30))
    start=time.perf_counter();cpu=loader.get(ids);query=cpu.to('cuda');torch.cuda.synchronize()
    preparation={'load_and_transfer_seconds':time.perf_counter()-start}
    context=t.RankingContext(ds,memory);counts=torch.bincount(memory['classes'],minlength=2).float()
    if bool((counts==0).any()):raise ValueError('Both observed classes required')
    weights=counts.sum()/(2*counts);group=ds.rows[ids[0]]['patient_group']
    support=t.support_for_recipient(memory,group)
    net.load_state_dict(saved['model'],strict=True)
    start=time.perf_counter()
    if state['last_group']==group:plan=state['plan']
    else:
        net.eval()
        with torch.no_grad():plan=net.fit_support_clusters(*support)
    torch.cuda.synchronize();preparation['support_plan_seconds']=time.perf_counter()-start
    a.output.mkdir(parents=True,exist_ok=False)
    print(f'DEBUG only | saved step {state["step"]} | next step {state["step"]+1} | physical batch {batch} | actual {len(ids)} | full support {len(support[0])} | retained | 3 disposable updates',flush=True)
    rows=[];reference=None;breakdown=None;parity={}
    for arm in ('warmup','baseline','attributed'):
        net.load_state_dict(saved['model'],strict=True);net.train()
        optimizer=torch.optim.AdamW(net.parameters(),lr=ds.meta['base']['training']['lr'],weight_decay=ds.meta['base']['training']['weight_decay'],fused=ds.meta['base']['training']['fused_optimizer'])
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));optimizer.zero_grad(set_to_none=True)
        t.restore_rng(saved['rng']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times={};attribution=Attribution()
        def timed(label,fn):
            torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
            times[label]=time.perf_counter()-start;return out
        def forward():return t.forward_loss(net,query,support,plan,memory['classes'][ids],weights,context,t.rank_config(),indices=ids)
        try:
            if arm=='attributed':
                with attribution.ranges(net):loss,terms=timed('forward',forward)
                attribution.attach(loss)
            else:loss,terms=timed('forward',forward)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
            timed('backward',loss.backward)
            if arm=='attributed':breakdown=attribution.report()
            raw_grads={k:v.grad.detach().cpu().clone() for k,v in net.named_parameters() if v.requires_grad and v.grad is not None}
            def check_clip():
                t.gradient_check(net)
                torch.nn.utils.clip_grad_norm_(net.parameters(),ds.meta['base']['training']['grad_clip'],error_if_nonfinite=True)
            timed('check_clip',check_clip);timed('optimizer',optimizer.step);budget.check()
            current=dict(loss=loss.detach().cpu(),terms=t.tree_to(terms,'cpu'),unclipped_gradients=raw_grads,
                         clipped_gradients={k:v.grad.detach().cpu().clone() for k,v in net.named_parameters() if v.requires_grad and v.grad is not None},
                         model=t.tree_to(net.state_dict(),'cpu'),optimizer=t.tree_to(optimizer.state_dict(),'cpu'))
            rng_hash=t.hash_state(t.rng_state())
            if reference is None:reference=current;reference_rng=rng_hash
            else:
                assert_state_close(current,reference)
                if rng_hash!=reference_rng:raise ValueError('Instrumentation changed RNG advancement')
                parity[arm]=True
            rows.append(dict(arm=arm,times=times,compute_update_seconds=sum(times.values()),peak_cuda_bytes=torch.cuda.max_memory_allocated(),rss_bytes=psutil.Process().memory_info().rss))
            print(f'{arm}: backward={times["backward"]:.3f}s compute_update={sum(times.values()):.3f}s peak={rows[-1]["peak_cuda_bytes"]/2**30:.3f}GiB',flush=True)
        finally:attribution.close()
        del optimizer,loss,terms,current,raw_grads
        net.zero_grad(set_to_none=True);gc.collect()
    if before!=sha(a.checkpoint) or cache_hash!=sha(a.cache):raise ValueError('Source files changed; use a paused checkpoint')
    baseline=rows[1]['times']['backward'];instrumented=rows[2]['times']['backward']
    result=dict(debug=True,full_training=False,production_ready=False,production_writes=False,cache_rebuilt=False,
        checkpoint_unchanged=True,status='PASS',activation_storage=identity['activation_storage'],gpu=torch.cuda.get_device_name(),
        saved_step=state['step'],next_step=state['step']+1,configured_batch=batch,actual_batch=len(ids),support_records=len(support[0]),
        total_memory_records=len(memory['record_ids']),records=[ds.rows[i]['id'] for i in ids],
        checkpoint_sha256=before,cache_sha256=cache_hash,verified_source=verified_revision,runtime_source=source_identity(),
        parity=parity,parity_rtol=2e-5,parity_atol=2e-6,rng_exact=True,rows=rows,preparation=preparation,backward_attribution=breakdown,
        backward_instrumentation_ratio=instrumented/baseline,
        attribution_warning='Hook/event overhead is included. Do not transfer shares to production seconds or use this as an epoch estimate.',
        scope='Same saved next batch and full saved support reset three times. Compute times exclude loader, plan, parity copies and disk checkpoints. No model or data reduction.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('BACKWARD GROUPS (instrumented; not production timings):',flush=True)
    for row in breakdown['groups']:print(f'{row["group"]}: {row["seconds"]:.4f}s',flush=True)
    print(f'Instrumentation backward ratio: {instrumented/baseline:.3f}; do not scale these shares to production seconds',flush=True)
    print('PARITY: PASS (loss, unclipped/clipped gradients, model, Adam within tolerance; RNG exact)',flush=True)
    print('TOP BACKWARD OPERATIONS:',flush=True)
    for row in breakdown['top_ops'][:5]:print(f'{row["group"]}/{row["operation"]}: {row["seconds"]:.4f}s',flush=True)
    print('REPORT:',a.output/'report.json',flush=True)


if __name__=='__main__':main()
