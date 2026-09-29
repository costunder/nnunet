"""Explicit short real-pair experiment, fixed checkpoint CNN partitions.

Preparation/storage/reload measured separately. Rejected initial profiles stay
rejected; --allow-unvalidated-profile permits diagnostic compute only.
No production checkpoint, whole cache build, training run or ready marker.
"""
import argparse,copy,gc,json,statistics,sys,zipfile,subprocess,importlib.metadata
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,psutil
from tools.v22_artifacts import tree_hash
from tools.profile_l0_ezsp_debug import timed,sha
from tools.diagnose_l0_ezsp_pairs import update_loss,tensor_hash
from tools.v22_debug_profile import RepairDataset
from tools.v222_review_contracts import installed,grouped_support
from hiercp_v222.v1_cache import configuration,provenance
from hiercp_v222.v1_local import V1LocalEncoder
from hiercp_v222.model import PromptGraphModel
from l0_regions.materialization import load_pairs
from l0_ezsp.config import load_profile
from l0_ezsp.diagnostic import ResourceBudget
from l0_ezsp.identity import load_cnn_only
from l0_sage.encoder import GraphSAGEEncoder
from l0_regions.preparation import binding,prepare,fixed_profile
from l0_regions.data import save_new,load,collate
from l0_regions.encoder import RegionSAGEEncoder
from l0_regions.resident import RegionResidentCache,storage_bytes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','partition-checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('physical-batch','workers','repeats','warmup','view-epoch'):p.add_argument('--'+key,type=int,required=True)
    for key in ('reg-scale1','reg-scale2','cuda-gib','rss-gib','seconds'):p.add_argument('--'+key,type=float,required=True)
    p.add_argument('--allow-unvalidated-profile',action='store_true')
    p.add_argument('--schedule-passes',type=int,required=True,help='Explicit >=2 short full DEBUG schedule passes, cold then RAM hit')
    a=p.parse_args()
    if min(a.physical_batch,a.workers,a.repeats,a.warmup)<1 or a.view_epoch<0:raise ValueError('Explicit positive diagnostic settings')
    if a.schedule_passes<2:raise ValueError('At least two explicit schedule passes required')
    a.output.mkdir(parents=True,exist_ok=False)
    budget=ResourceBudget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30),a.seconds)
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('Device headroom required')
    previous=torch.cuda.get_per_process_memory_fraction();torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    report=dict(debug=True,production_admitted=False,training_ready=False,full_training=False,
        architecture='fixed view + frozen CNN partitions at both scales; live CNN pooling + SAGE2/1',
        physical_batch=a.physical_batch,effective_batch=a.physical_batch,gradient_accumulation=1,workers=a.workers,precision='FP32',
        resources=dict(gpu=torch.cuda.get_device_name(),free_vram=torch.cuda.mem_get_info()[0],total_vram=total,
            cpu_physical=psutil.cpu_count(logical=False),ram_available=psutil.virtual_memory().available,
            cuda_budget=budget.cuda_bytes,rss_budget=budget.rss_bytes,seconds_budget=budget.seconds),
        unchanged=['Basic CP','L1/L2','rank/aux/alignment loss','candidate128','original masks','3 layers/128D'],
        exclusions=['CP accuracy','semantic quality of partitions','server/MIG timing','production full support','production ready'],
        timing_scope='same GPU batch with CSR prepared by refresh; update excludes preparation, loader/H2D, refresh/plan and disk checkpoints; each starts cloned model and fresh Adam',
        checkpoint_path=str(a.partition_checkpoint),checkpoint_sha256=sha(a.partition_checkpoint),core_identity=provenance(),runs=[])
    paths=sorted(set([*ROOT.glob('*.py'),*[p for d in ('hiercp','hiercp_v22','hiercp_v221','hiercp_v222','l0_ezsp','l0_sage','l0_regions','tools','tests') for p in (ROOT/d).rglob('*.py')],*(ROOT/'config').glob('*.json')]))
    frozen_sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    with zipfile.ZipFile(a.output/'measured_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:archive.write(path,path.relative_to(ROOT))
    report['measured_source_sha256']=sha(a.output/'measured_source.zip')
    report['source_snapshot_manifest']=frozen_sources
    report['environment']={k:importlib.metadata.version(k) for k in ('torch','torch-geometric','numpy','scipy','psutil')}
    report['gpu_snapshot_before']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True,check=True).stdout
    try:
        torch.set_num_threads(a.workers)
        ds=RepairDataset(a.cache,'inner_train')
        if len(ds.rows)!=a.physical_batch:raise ValueError('Use all explicit DEBUG inner_train records; no slicing')
        checkpoint=torch.load(a.partition_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if sha(a.partition_checkpoint)!=report['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed while reading; retry against a stable saved snapshot')
        state=checkpoint.get('state',{})
        if state.get('step',0)<=0 or 'model' not in checkpoint:raise ValueError('Existing optimized CNN checkpoint required; no random fallback')
        report['checkpoint_state']={k:state.get(k) for k in ('step','epoch','phase')}
        report['checkpoint_debug']=checkpoint.get('debug')
        cfg,base=configuration();torch.manual_seed(42)
        with installed('stride4'):reference=V1LocalEncoder(base)
        load_cnn_only(reference,checkpoint['model']);del checkpoint
        report['frozen_cnn_sha256']=tree_hash(reference.dense_encoder.state_dict())
        report['weights']='CNN from specified existing optimized checkpoint; common remaining parameters seed42; new model, not exact resume'
        profile=fixed_profile(load_profile(reg_scale1=a.reg_scale1,reg_scale2=a.reg_scale2))
        report['profile']=profile
        identities=[binding(record=dict(r,donor_component=ds.record(i)['component_id']),dataset_index=i,cache_sha256=sha(a.cache),
            frozen_cnn_sha256=report['frozen_cnn_sha256'],profile=profile,view_epoch=a.view_epoch,view_index=0,
            feature_evidence='checkpoint_partition_quality_unverified') for i,r in enumerate(ds.rows)]
        start=perf_counter();fine_cpu=load_pairs(ds,list(range(len(ds.rows))),workers=a.workers,epoch=a.view_epoch,cache_path=a.cache)
        report['fine_loader_seconds']=perf_counter()-start
        fine,report['fine_h2d_seconds']=timed(lambda:fine_cpu.to('cuda'))
        report['fine_nodes']=sum(fine.graph[k].num_nodes for k in fine.graph.node_types)
        report['fine_edges']=sum(fine.graph[e].edge_index.shape[1] for e in fine.graph.edge_types)
        frozen=copy.deepcopy(reference).eval().requires_grad_(False).cuda()
        (items,audit),report['prepare_seconds']=timed(lambda:prepare(fine,frozen,profile,identities,budget,allow_unvalidated_profile=a.allow_unvalidated_profile))
        report['partition_audit']=audit
        del frozen
        folder=a.output/'region_cache';folder.mkdir()
        start=perf_counter()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            sizes=list(pool.map(lambda i:save_new(folder/f'pair_{i:03d}.pt',items[i]),range(len(items))))
        report['cache_write_seconds']=perf_counter()-start;report['cache_serialized_pt_bytes']=sum(sizes)
        start=perf_counter()
        resident=RegionResidentCache(max_tensor_bytes=budget.rss_bytes,workers=a.workers)
        paths=[folder/f'pair_{i:03d}.pt' for i in range(len(items))];indices=list(range(len(items)))
        region_cpu=resident.get(paths,identities,indices)
        report['cache_read_validate_collate_seconds']=perf_counter()-start
        report['resident_tensor_storage_bytes']=storage_bytes(list(resident.entries.values()))
        report['process_rss_after_resident_load_bytes']=psutil.Process().memory_info().rss
        report['resident_memory_scope']='Retained unique tensor storages; excludes Python metadata and transient item/collate allocations. RSS snapshot is not peak RSS.'
        report['pinned_tensor_storage_bytes']=0
        report['pinned_memory_scope']='No pin_memory in this diagnostic loader; transient load/collate peak is not measured.'
        start=perf_counter()
        for _ in range(a.repeats):
            if resident.get(paths,identities,indices) is not region_cpu:raise RuntimeError('Resident batch not reused')
        report['resident_warm_get_seconds']=(perf_counter()-start)/a.repeats
        report['resident_hits']=resident.hits;report['resident_misses']=resident.misses
        region,report['region_h2d_seconds']=timed(lambda:region_cpu.to('cuda'))
        source_hashes=lambda t:[tree_hash(x) for x in t]
        fine_sources=source_hashes(fine_cpu.source_patches);region_sources=source_hashes(region_cpu.source_patches)
        report['source_dedup']=dict(fine_storage_count=len(fine_sources),fine_content_count=len(set(fine_sources)),
            region_storage_count=len(region_sources),region_content_count=len(set(region_sources)),
            same_unique_contents=set(fine_sources)==set(region_sources),
            cnn_source_work_equal=len(fine_sources)==len(region_sources))
        report['coarse_nodes']=[sum(g[k].num_nodes for k in g.node_types) for g in region.graphs]
        report['coarse_edges']=[sum(g[e].edge_index.shape[1] for e in g.edge_types) for g in region.graphs]
        report['profile_exceeded']=region.profile_exceeded
        report['fine_edges_stored_or_transferred_in_region_batch']=0
        common=PromptGraphModel(cfg,base,{},local_encoder=reference)
        masters={}
        for name in ('fine_sage','fixed_region_sage'):
            net=copy.deepcopy(common)
            net.local=GraphSAGEEncoder(reference,seed=42) if name=='fine_sage' else RegionSAGEEncoder(reference,seed=42,resource_budget=budget,allow_unvalidated_profile=a.allow_unvalidated_profile)
            masters[name]=net
        other=dict(masters['fixed_region_sage'].named_parameters())
        for name,value in masters['fine_sage'].named_parameters():
            mapped=name.replace('local.encoder.','local.core.',1)
            if not torch.equal(value,other[mapped]):raise ValueError('Comparison parameter mismatch: '+name)
        hashes={k:tensor_hash(v) for k,v in masters.items()};report['master_hashes']=hashes
        report['parameters']={k:sum(p.numel() for p in v.parameters()) for k,v in masters.items()}
        groups=sorted({r['patient_group'] for r in ds.rows});contexts={g:[i for i,r in enumerate(ds.rows) if r['patient_group']==g] for g in groups}
        targets=torch.tensor([r['target'] for r in ds.rows],device='cuda');counts=torch.bincount(targets,minlength=2).float()
        if bool((counts==0).any()):raise ValueError('Both observed classes required')
        weights=counts.sum()/(2*counts)
        for trial in range(a.warmup+a.repeats):
            for name in (('fine_sage','fixed_region_sage') if trial%2==0 else ('fixed_region_sage','fine_sage')):
                budget.check();net=copy.deepcopy(masters[name]).cuda().eval();batch=fine if name=='fine_sage' else region
                with torch.no_grad():embedding,refresh=timed(lambda:net.local(batch))
                memory=dict(embeddings=embedding.detach(),record_ids=[r['id'] for r in ds.rows],patient_groups=groups,
                    donor_groups=[r['donor_group'] for r in ds.rows],owners=torch.tensor([groups.index(r['patient_group']) for r in ds.rows],device='cuda'),classes=targets)
                plans,plan_seconds=timed(lambda:{g:net.fit_support_clusters(*grouped_support(memory,g)) for g in groups})
                optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
                net.train();torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();start=perf_counter()
                (loss,terms),forward=timed(lambda:update_loss(net,batch,ds,memory,contexts,plans,weights))
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite rank loss')
                _,backward=timed(loss.backward)
                _,clip=timed(lambda:torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True))
                _,step=timed(optimizer.step);whole=perf_counter()-start
                missing=[n for n,p in net.named_parameters() if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all()))]
                if missing:raise ValueError('Missing/nonfinite gradients: '+str(missing))
                prefix='local.encoder.' if name=='fine_sage' else 'local.core.'
                for component in (prefix+'dense_encoder',prefix+'blocks.0.conv',prefix+'blocks.1.conv',prefix+'blocks.2.conv','l1','l2'):
                    if not any(p.grad is not None and bool((p.grad!=0).any()) for n,p in net.named_parameters() if n.startswith(component)):raise ValueError('No live gradient: '+component)
                row=dict(arm=name,trial=trial,warmup=trial<a.warmup,update_seconds=whole,forward_seconds=forward,backward_seconds=backward,
                    local_forward_seconds=terms['local_forward_seconds'],l1_l2_objective_seconds=terms['l1_l2_objective_seconds'],
                    clip_seconds=clip,optimizer_seconds=step,refresh_seconds=refresh,plan_seconds=plan_seconds,loss=float(loss.detach()),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_trainable_gradients_finite=True)
                report['runs'].append(row);print(name,trial,round(whole,4),'seconds',flush=True)
                del net,optimizer,loss,memory,embedding,plans;gc.collect();torch.cuda.empty_cache()
        report['means']={name:{k:statistics.mean(r[k] for r in report['runs'] if r['arm']==name and not r['warmup']) for k in ('update_seconds','refresh_seconds','forward_seconds','backward_seconds','peak_allocated_bytes','local_forward_seconds','l1_l2_objective_seconds')} for name in masters}
        from tools.profile_region_schedule_debug import profile_schedule
        report['consecutive_schedule']=profile_schedule(masters=masters,dataset=ds,cache_path=a.cache,paths=paths,
            bindings=identities,full_batches=dict(fine_sage=fine,fixed_region_sage=region),
            physical_batch=a.physical_batch,workers=a.workers,view_epoch=a.view_epoch,
            passes=a.schedule_passes,base=base,budget=budget)
        for name,net in masters.items():
            if tensor_hash(net)!=hashes[name]:raise RuntimeError('Master mutated')
        report['status']='DIAGNOSTIC_COMPLETE_PARTITION_QUALITY_UNVERIFIED'
    except Exception as exc:
        report['status']='ERROR';report['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        torch.cuda.set_per_process_memory_fraction(previous)
        report['source_sha256']={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),*(ROOT/'l0_regions').glob('*.py')]}
        report['measured_sources_unchanged']=all(sha(ROOT/name)==digest for name,digest in frozen_sources.items())
        report['cache_total_bytes']=sum(path.stat().st_size for path in (a.output/'region_cache').glob('*'))
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
