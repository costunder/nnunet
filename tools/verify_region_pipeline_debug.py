"""Actual CT synchronous/overlapped parity, timing and fresh-process resume."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training import hash_state,load_checkpoint,Loader,make_model
from l0_regions.training_data import RegionDataset,Budget,source_identity,sha,write_new
from l0_regions.execution_pipeline import DeviceBatchCache,packed_cpu_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','legacy-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();old=torch.load(a.legacy_checkpoint,map_location='cpu',weights_only=False)
    identity=old['identity']
    if not identity['debug'] or 'support_training' not in identity:raise ValueError('Existing patient-episode DEBUG checkpoint required')
    a.output.mkdir(parents=True,exist_ok=False);limits=identity['resource_limits']
    common=['--debug','--profile-policy',identity['profile_policy'],'--cache',str(a.cache),
        '--workers',str(identity['workers']),'--cuda-gib',str(limits['cuda_bytes']/2**30),
        '--rss-gib',str(limits['rss_bytes']/2**30),'--resident-gib',str(identity['resident_budget_bytes']/2**30),
        '--batch-candidates',*map(str,identity['candidates']),'--activation-storage',identity['activation_storage'],
        '--support-patients',str(identity['support_training']['patients'])]
    def run(name,mode,*extra):
        args=[] if mode=='synchronous' else ['--execution-pipeline','overlapped','--device-cache-gib','1','--sage-workspace-mib','256']
        subprocess.run([sys.executable,'-B','-u','tools/run_fixed_regions.py','train',*common,*args,
            '--output',str(a.output/name),*map(str,extra)],cwd=ROOT,check=True)
    run('synchronous','synchronous')
    run('overlapped','overlapped')
    run('paused','overlapped','--debug-pause-step',1)
    run('resumed','overlapped','--resume',a.output/'paused/checkpoint_latest.pt')
    run('upgraded','overlapped','--resume',a.legacy_checkpoint,'--resume-execution-upgrade',
        '--debug-pause-step',old['state']['step']+1)
    def read(name):return torch.load(a.output/name/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    baseline,overlap,resumed=read('synchronous'),read('overlapped'),read('resumed')
    parity={arm:{key:hash_state(value[key])==hash_state(overlap[key]) for key in ('model','optimizer','state','rng')}
            for arm,value in [('synchronous_vs_overlapped',baseline),('resume_vs_uninterrupted',resumed)]}
    if not all(all(v.values()) for v in parity.values()):raise AssertionError(parity)
    upgraded=read('upgraded')
    if upgraded['state']['step']!=old['state']['step']+1:raise AssertionError('Upgrade cursor changed')
    policy=overlap['identity']['execution_pipeline']
    transitioned=load_checkpoint(a.legacy_checkpoint,dict(identity,source=source_identity(),execution_pipeline=policy),allow_execution_upgrade=True)
    preservation={k:hash_state(transitioned[k])==hash_state(old[k]) for k in ('model','optimizer','state','rng')}
    if not all(preservation.values()):raise AssertionError(preservation)
    ds=RegionDataset(a.cache,'inner_train',True,identity['profile_policy']);loader=Loader(ds,identity['workers'],identity['resident_budget_bytes'])
    cpu=loader.get([0,1]);cache=DeviceBatchCache(2**30,Budget(**limits))
    torch.cuda.synchronize();start=__import__('time').perf_counter();gpu=cache.get(cpu);torch.cuda.synchronize();cold=__import__('time').perf_counter()-start
    start=__import__('time').perf_counter();again=cache.get(cpu);torch.cuda.synchronize();warm=__import__('time').perf_counter()-start
    if again is not gpu or cache.hits!=1:raise AssertionError('GPU cache not reused')
    from l0_regions.sparse import workspace
    net=make_model(ds,Budget(**limits),True,'retained');net.load_state_dict(overlap['model']);net.eval()
    net.local.dense_batch_size=overlap['state']['batch']
    def local_update(batch):
        net.local.zero_grad(set_to_none=True)
        with workspace(256*2**20):value=net.local(batch);value.square().sum().backward()
        return value.detach().cpu(),[p.grad.detach().cpu().clone() for p in net.local.parameters()]
    value1,grad1=local_update(gpu);cache.remember(net,cpu)
    other=loader.get([2,3]);other_gpu=cache.get(other);local_update(other_gpu);cache.remember(net,other)
    gpu=cache.get(cpu);cache.activate(net,cpu);builds=[c.builds for c in net.local.adjacencies]
    value2,grad2=local_update(gpu)
    topology_parity=torch.equal(value1,value2) and all(torch.equal(x,y) for x,y in zip(grad1,grad2))
    topology_reused=builds==[c.builds for c in net.local.adjacencies]
    if not topology_parity or not topology_reused:raise AssertionError('Cached SAGE topology changes output/gradient or rebuilds')
    gpu.target_patches.add_(1)
    try:cache.get(cpu)
    except ValueError:mutation_rejected=True
    else:raise AssertionError('Mutated GPU input accepted')
    # CUDA/CPU packed snapshot equality, including strided values and Adam scalars.
    original={'x':torch.arange(24,device='cuda').view(4,6)[:,::2],'y':torch.randn(5,device='cuda'),'step':torch.tensor(9.)}
    snap=packed_cpu_snapshot(original)
    if hash_state(snap)!=hash_state(original):raise AssertionError('Packed D2H changed values')
    means={}
    for arm in ('synchronous','overlapped'):
        rows=[json.loads(s) for s in (a.output/arm/'update_timing.jsonl').read_text().splitlines()]
        means[arm]={key:statistics.mean(r[key] for r in rows) for key in ('step_seconds','loader_wait_seconds','checkpoint_seconds','transfer_seconds','forward_seconds','backward_seconds','check_clip_optimizer_seconds')}
        # Include the final writer drain, which is outside the last step's enqueue timing.
        means[arm]['timing_scope']='all four DEBUG updates; enqueue step times omit final drain, see checkpoint_timing rows for hash/fsync'
        phases=[json.loads(s) for s in (a.output/arm/'phase_timing.jsonl').read_text().splitlines()]
        means[arm]['phase_seconds']={r['phase']:r['seconds'] for r in phases}
    write_new(a.output/'report.json',dict(debug=True,full_training=False,production_ready=False,
        source=source_identity(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        parity=parity,upgrade_preservation=preservation,source_checkpoint_sha256=sha(a.legacy_checkpoint),
        means=means,device_cache=dict(cold_seconds=cold,hit_seconds=warm,bytes=cache.bytes,mutation_rejected=mutation_rejected),
        cached_topology_output_gradient_equal=topology_parity,cached_topology_not_rebuilt=topology_reused,
        packed_cuda_snapshot_equal=True,train_records=len(ds),actual_query_batch=2,
        scope='Actual CT DEBUG entire supplied train/val; same math and updates; not a server epoch speed estimate'))
    print('REPORT:',a.output/'report.json',flush=True)

if __name__=='__main__':main()
