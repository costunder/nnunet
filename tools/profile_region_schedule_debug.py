"""Full supplied DEBUG schedule, alternating arms, no production writes.

Uses the existing patient/size scheduler, including every natural tail. Reports
actual cardinality rather than calling a tail a full configured physical batch.
"""
import copy
import gc
from time import perf_counter
from unittest.mock import patch

import psutil
import torch
from hiercp_v222.v1_training import groups
from l0_regions.materialization import load_pairs
from l0_regions.resident import RegionResidentCache, storage_bytes
from tools.profile_l0_ezsp_debug import timed
from tools.v222_review_contracts import grouped_support
from tools.v22_rank_objective import RankingContext, configuration, forward_loss


def profile_schedule(*, masters, dataset, cache_path, paths, bindings, full_batches,
                     physical_batch, workers, view_epoch, passes, base, budget):
    if passes < 2:
        raise ValueError('At least two explicit DEBUG passes: cold and resident hit')
    order=list(groups(dataset,physical_batch,42,view_epoch))
    flat=[i for ids in order for i in ids]
    if sorted(flat)!=list(range(len(dataset.rows))) or len(flat)!=len(set(flat)):
        raise ValueError('Schedule must cover the entire supplied DEBUG dataset once')
    if len(order)<2:raise ValueError('Consecutive distinct scheduled batches required')
    report=dict(debug=True,production_admitted=False,training_ready=False,
        configured_physical_batch=physical_batch,actual_batch_sizes=[len(ids) for ids in order],
        schedule=order,seed=42,fixed_view_epoch=view_epoch,passes=passes,
        scope='Complete local DEBUG schedule, not full-cohort/server epoch; each pass starts a cloned model/new optimizer; optimizer persists across batches within pass',
        step_scope='CPU batch retrieval + H2D/validation + CSR construction + forward/loss/backward/clip/optimizer; excludes support/plan refresh, offline preparation and production checkpoint writes',
        pinned_tensor_storage_bytes=0,pinned_memory_scope='Unpinned loader; no pin_memory calls',
        transient_memory_scope='RSS snapshots only; transient load/collate peak not sampled',
        runs=[],support=[],fine_cold_load=[])
    resident=RegionResidentCache(max_tensor_bytes=budget.rss_bytes,workers=workers)
    fine_resident={}
    # Fine arm also gets a CPU resident cache, so both include CPU lookup rather
    # than timing repeated disk I/O only on one arm. Cold cost is kept separately.
    for ids in order:
        budget.check();start=perf_counter()
        fine_resident[tuple(ids)]=load_pairs(dataset,ids,workers=workers,epoch=view_epoch,cache_path=cache_path)
        report['fine_cold_load'].append(dict(indices=ids,seconds=perf_counter()-start))
    all_groups=sorted({r['patient_group'] for r in dataset.rows})
    targets=torch.tensor([r['target'] for r in dataset.rows],device='cuda')
    counts=torch.bincount(targets,minlength=2).float()
    if bool((counts==0).any()):raise ValueError('Both observation classes required')
    weights=counts.sum()/(2*counts)
    for trial in range(passes):
        arms=('fine_sage','fixed_region_sage') if trial%2==0 else ('fixed_region_sage','fine_sage')
        for arm in arms:
            budget.check();net=copy.deepcopy(masters[arm]).cuda().eval()
            with torch.no_grad():embedding,refresh=timed(lambda:net.local(full_batches[arm]))
            memory=dict(embeddings=embedding.detach(),record_ids=[r['id'] for r in dataset.rows],patient_groups=all_groups,
                donor_groups=[r['donor_group'] for r in dataset.rows],
                owners=torch.tensor([all_groups.index(r['patient_group']) for r in dataset.rows],device='cuda'),classes=targets)
            plans,plan_time=timed(lambda:{g:net.fit_support_clusters(*grouped_support(memory,g)) for g in all_groups})
            context=RankingContext(dataset,memory)
            report['support'].append(dict(arm=arm,trial=trial,records=len(dataset.rows),refresh_seconds=refresh,plan_seconds=plan_time))
            optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
            net.train();net.local.dense_batch_size=physical_batch
            for position,ids in enumerate(order):
                budget.check();torch.manual_seed(42+position)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter()
                before_hits=resident.hits
                cpu=(fine_resident[tuple(ids)] if arm=='fine_sage' else
                     resident.get([paths[i] for i in ids],[bindings[i] for i in ids],ids))
                lookup=perf_counter()-start
                batch,transfer=timed(lambda:cpu.to('cuda'))
                caches=[net.local.adjacency] if arm=='fine_sage' else net.local.adjacencies
                graphs=[batch.graph] if arm=='fine_sage' else batch.graphs
                builds_before=sum(c.builds for c in caches)
                _,csr=timed(lambda:[c.prepare(g) for c,g in zip(caches,graphs)])
                group=dataset.rows[ids[0]]['patient_group']
                if any(dataset.rows[i]['patient_group']!=group for i in ids):raise ValueError('Patient schedule mismatch')
                support=grouped_support(memory,group)
                selected=targets[torch.tensor(ids,device='cuda')]
                local_times=[];original=net.local.forward
                def local(value):
                    result,seconds=timed(lambda:original(value));local_times.append(seconds);return result
                with patch.object(net.local,'forward',local):
                    (loss,terms),forward=timed(lambda:forward_loss(net,batch,support,plans[group],selected,weights,context,configuration(),indices=ids))
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite scheduled loss')
                    _,backward=timed(loss.backward)
                _,clip=timed(lambda:torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True))
                _,step=timed(optimizer.step)
                whole=perf_counter()-start
                missing=[n for n,p in net.named_parameters() if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all()))]
                if missing:raise ValueError('Scheduled gradient failure: '+str(missing))
                report['runs'].append(dict(arm=arm,trial=trial,position=position,indices=ids,actual_physical_batch=len(ids),
                    region_resident_hit=resident.hits>before_hits if arm=='fixed_region_sage' else None,
                    lookup_seconds=lookup,h2d_validate_seconds=transfer,csr_seconds=csr,
                    csr_builds=sum(c.builds for c in caches)-builds_before,
                    forward_seconds=forward,local_forward_seconds=sum(local_times),backward_seconds=backward,
                    clip_seconds=clip,optimizer_seconds=step,whole_step_seconds=whole,
                    loss=float(loss.detach()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    process_rss_bytes=psutil.Process().memory_info().rss,
                    resident_tensor_storage_bytes=storage_bytes(list(resident.entries.values())),
                    all_trainable_gradients_finite=True))
                print('schedule',arm,trial,position,'actual_batch',len(ids),round(whole,4),'s',flush=True)
                del batch,loss,terms,support,selected
            del optimizer,net,memory,embedding,plans,context
            gc.collect();torch.cuda.empty_cache()
    report['resident_tensor_storage_bytes']=storage_bytes(list(resident.entries.values()))
    report['fine_cpu_tensor_storage_bytes']=storage_bytes(fine_resident)
    report['resident_entries']=len(resident.entries)
    report['resident_hits']=resident.hits;report['resident_misses']=resident.misses
    return report
