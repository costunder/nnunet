"""Full paired-cohort L0/L1/L2 training with measured disjoint graph batches."""
import csv
import gc
import math
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from scipy.stats import rankdata
from .contracts import write_new,sha
from .v1_cache import PairDataset,PairLoader,configuration,provenance,emit,save_torch_new
from .v1_local import model,support_for_recipient
from .model import supervised_loss
from .training import configure_runtime,require_device,batch_admission,execution_budget

CHECKPOINT_FORMAT='v222_v1_paired_trained_v1'

def groups(dataset,batch,seed=None,epoch=0):
    grouped={}
    for i,row in enumerate(dataset.rows):grouped.setdefault(row['patient_group'],[]).append(i)
    rng=np.random.default_rng(seed+epoch) if seed is not None else None
    keys=list(grouped)
    if rng is not None:rng.shuffle(keys)
    for key in keys:
        # Similar graph sizes share a physical batch. Randomize bucket order,
        # not cohort membership; the last partial batch is never dropped.
        ids=sorted(grouped[key],key=lambda i:dataset.rows[i]['bounds']['edges'])
        chunks=[ids[j:j+batch] for j in range(0,len(ids),batch)]
        if rng is not None:rng.shuffle(chunks)
        yield from chunks

def contiguous(dataset,batch):
    for j in range(0,len(dataset),batch):yield list(range(j,min(j+batch,len(dataset))))

@torch.no_grad()
def memory_encode(net,dataset,batch,workers):
    net.eval();embeddings=torch.empty((len(dataset),128),device='cuda',dtype=torch.float32)
    loader=PairLoader(dataset,workers);seen=set();start=last=time.perf_counter()
    try:
        for payload in loader.batches(contiguous(dataset,batch),epoch=0):
            ids=payload.indices.tolist()
            with torch.autocast('cuda',dtype=torch.bfloat16):values=net.local(payload.cuda(non_blocking=True))
            embeddings[ids]=values.float();seen.update(ids)
            now=time.perf_counter()
            if now-last>=30 or len(seen)==len(dataset):
                emit(stage='support_memory',completed=len(seen),total=len(dataset),physical_batch=batch,
                     graphs_per_second=len(seen)/(now-start));last=now
    finally:loader.close()
    if seen!=set(range(len(dataset))):raise RuntimeError('Incomplete support memory')
    cases=sorted(set(r['case_id'] for r in dataset.rows));lookup={c:i for i,c in enumerate(cases)}
    if not set(cases)<=set(dataset.meta['split']['inner_train']):raise ValueError('Validation entered support memory')
    return dict(embeddings=embeddings,owners=torch.tensor([lookup[r['case_id']] for r in dataset.rows],device='cuda'),
        classes=torch.tensor([r['target'] for r in dataset.rows],device='cuda'),case_ids=cases,
        patient_groups=[dataset.meta['identities']['cases'][c]['patient_group'] for c in cases],
        donor_groups=[r['donor_group'] for r in dataset.rows],
        record_ids=[r['id'] for r in dataset.rows])

def gradient_check(net):
    missing=[n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]
    invalid=[n for n,p in net.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    if missing or invalid:raise RuntimeError(f'Gradient path failure: missing={missing}, nonfinite={invalid}')
    return dict(all_parameter_gradients_present=True,all_parameter_gradients_finite=True)

def calibrate_workers(dataset,indices,root):
    reports=[]
    # Same real full-graph workload, including collation and pinning, at each width.
    for workers in (0,2,4,8):
        loader=PairLoader(dataset,workers)
        try:
            start=time.perf_counter();payload=loader.make(indices,0)
            report=dict(workers=workers,graphs=len(indices),seconds=time.perf_counter()-start,
                nodes=int(payload.graph.num_nodes),edges=int(payload.graph.num_edges),debug_calibration=True)
            del payload
        finally:loader.close()
        reports.append(report);emit(stage='paired_loader_calibration',**report)
    # Cold cache cost of the first probe must not determine the winner.
    for report in reports:
        loader=PairLoader(dataset,report['workers'])
        try:
            start=time.perf_counter();payload=loader.make(indices,0)
            report['warm_seconds']=time.perf_counter()-start;del payload
        finally:loader.close()
    selected=min(reports,key=lambda r:r['warm_seconds'])['workers']
    write_new(root/'loader_calibration.json',dict(reports=reports,selected=selected,
        shared_cache_bytes=dataset.budget,worker_kind='threads; shared canonical cache; one batch CPU prefetch'))
    return selected

def calibrate(net,dataset,cfg,workers,root,memory=None):
    training=memory is not None
    counts={}
    for row in dataset.rows:counts[row['patient_group']]=counts.get(row['patient_group'],0)+1
    maximum=max(counts.values())
    if training:
        # Use the case with highest canonical graph mass, not a uniform-graph assumption.
        group=max(counts,key=lambda g:sum(r['bounds']['edges'] for r in dataset.rows if r['patient_group']==g))
        indices=sorted([i for i,r in enumerate(dataset.rows) if r['patient_group']==group],
                       key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)
        support=support_for_recipient(memory,group)
        with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*support)
    else:
        indices=sorted(range(len(dataset)),key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)[:maximum]
    candidates=[2]
    while candidates[-1]<len(indices):candidates.append(min(2*candidates[-1],len(indices)))
    reports=[];loader=PairLoader(dataset,workers)
    net.train(training)
    try:
        for count in candidates:
            net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            baseline,budget=execution_budget(cfg)
            admission=batch_admission(count,reports,baseline,budget)
            if not admission['admitted']:
                reports.append(dict(physical_batch=count,accepted=False,executed=False,**admission));break
            cpu=loader.make(indices[:count],0);query=loss=output=None
            # Execution chunk only: CNN sees the full measured physical batch.
            net.local.dense_batch_size=count
            try:
                query=cpu.cuda(non_blocking=True)
                labels=torch.tensor([dataset.rows[i]['target'] for i in indices[:count]],device='cuda')
                def step():
                    with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
                        if training:
                            value=supervised_loss(net(query,*support,cluster_plan=plan),labels)
                        else:value=net.local(query)
                    if training:value.backward()
                    return value
                output=step();del output;output=None;net.zero_grad(set_to_none=True);torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(cfg['batch_calibration_repeats']):
                    net.zero_grad(set_to_none=True);output=step();del output;output=None
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if training:gradient_check(net)
                peak=torch.cuda.max_memory_allocated()
                report=dict(physical_batch=count,graphs_per_second=count*cfg['batch_calibration_repeats']/elapsed,
                    peak_vram_bytes=peak,accepted=peak<budget,executed=True,budget_bytes=budget,
                    nodes=int(cpu.graph.num_nodes),edges=int(cpu.graph.num_edges),training=training)
                reports.append(report)
            except torch.cuda.OutOfMemoryError as error:
                reports.append(dict(physical_batch=count,accepted=False,error=str(error),
                    policy='execution batch admission only; full graph/model/cohort retained'))
            finally:
                query=loss=output=None;del cpu;net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            emit(stage='paired_training_calibration' if training else 'paired_memory_calibration',**reports[-1])
            if not reports[-1]['accepted']:break
    finally:loader.close()
    name='training_batch' if training else 'memory_batch'
    write_new(root/f'{name}_calibration.json',reports)
    accepted=[r for r in reports if r['accepted']]
    if not accepted:raise MemoryError('Full paired graph batch does not fit; inspect memory report before changing execution')
    selected=max(accepted,key=lambda r:r['graphs_per_second'])['physical_batch']
    net.local.dense_batch_size=selected
    return selected,reports

@torch.no_grad()
def evaluate(net,dataset,memory,batch,workers,batch_complete=None):
    net.eval();loader=PairLoader(dataset,workers);last_group=None;pred=[];truth=[];seen=set()
    try:
        for payload in loader.batches(groups(dataset,batch),epoch=0):
            ids=payload.indices.tolist();group=dataset.rows[ids[0]]['patient_group']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if group!=last_group:
                    state=net.prepare_support(*support_for_recipient(memory,group));last_group=group
                result=net.predict_embeddings(net.local(payload.cuda(non_blocking=True)),state)
            pred.append(result['logits'].float().cpu());truth.extend(dataset.rows[i]['target'] for i in ids);seen.update(ids)
            del result
            if batch_complete is not None:batch_complete()
    finally:loader.close()
    if seen!=set(range(len(dataset))):raise RuntimeError('Incomplete inner validation')
    logits=torch.cat(pred);y=torch.tensor(truth);prob=logits.softmax(-1)[:,1].numpy()
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:raise ValueError('Both validation observation classes required')
    return dict(observation_cross_entropy=float(torch.nn.functional.cross_entropy(logits,y)),
        observation_accuracy=float((logits.argmax(-1)==y).float().mean()),
        observation_auroc=float((rankdata(prob)[y.numpy()==1].sum()-pos*(pos+1)/2)/(pos*neg)),
        samples=len(y),positives=pos,negatives=neg,
        scope='observation task only; not segmentation Dice or validated CP placement efficacy')


def train(cache,output,**kwargs):
    from .v1_execution import train as execute
    return execute(cache,output,**kwargs)
