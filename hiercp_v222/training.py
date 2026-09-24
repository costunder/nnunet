"""Measured physical batches, complete-epoch query coverage and frozen support provenance."""
from pathlib import Path
import time
import math
import os
import json
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from hiercp.preparation_runtime import snapshot
from .contracts import write_new, safe_new_root, source_identity, sha
from .data import ContextDataset, support_for_query
from .model import PromptGraphModel, supervised_loss
from . import PIPELINE_VERSION

def require_device(device):
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no automatic CPU fallback')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('This measured runner targets the single-GPU project host; multi-GPU execution must be configured explicitly')

def configure_runtime(base, seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = base['runtime']['allow_tf32']
    torch.backends.cudnn.allow_tf32 = base['runtime']['allow_tf32']
    torch.backends.cudnn.benchmark = base['runtime']['cudnn_benchmark']
    torch.use_deterministic_algorithms(base['runtime']['deterministic'])

def batch_admission(count, reports, baseline_bytes, budget_bytes, *, nonlinear_sampler=False):
    """Bound the next physical batch before CUDA/WDDM can page device memory.

    Extrapolate measured incremental memory instead of multiplying fixed support
    overhead by the batch size. Reserve 20% on the predicted increase; the device
    budget separately retains its configured reserve. This is execution only.
    """
    recent=[r for r in reports if r.get('accepted') and 'peak_vram_bytes' in r][-2:]
    if not recent:return dict(admitted=True,estimated_peak_bytes=None,budget_bytes=budget_bytes)
    last=recent[-1]
    if len(recent)==2 and last['physical_batch']>recent[0]['physical_batch']:
        previous=recent[0]
        slope=max(0,last['peak_vram_bytes']-previous['peak_vram_bytes'])/(last['physical_batch']-previous['physical_batch'])
        # Never extrapolate a flat/decreasing pair as zero incremental memory.
        slope=max(slope,max(0,last['peak_vram_bytes']-baseline_bytes)/last['physical_batch']*.1)
        predicted=int(max(baseline_bytes,last['peak_vram_bytes'])+1.2*slope*max(0,count-last['physical_batch']))
    else:
        per_sample=max(0,last['peak_vram_bytes']-baseline_bytes)/last['physical_batch']
        predicted=int(baseline_bytes+per_sample*count)
    if nonlinear_sampler:
        # PPR/A* scratch and deterministic backward do not follow the small
        # incremental slope of the tiled GNN. On this host the affine estimate
        # admitted batch128 after64 and WDDM stalled at full device residency.
        # A conservative per-sample live-memory envelope blocks that measured
        # failure without a hard physical-batch or graph-size cap.
        ratio=max(1.,count/last['physical_batch'])
        envelope=baseline_bytes+1.25*max(0,last['peak_vram_bytes']-baseline_bytes)*ratio
        predicted=max(predicted,int(envelope))
    return dict(admitted=predicted<=budget_bytes,estimated_peak_bytes=predicted,budget_bytes=budget_bytes)

def execution_budget(cfg):
    free,total=torch.cuda.mem_get_info()
    baseline=torch.cuda.memory_allocated()
    return baseline,int(min(total*cfg['max_vram_fraction'],baseline+free*cfg['max_vram_fraction']))

def release_probe_memory(model):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def graph_statistics(audit):
    """Called at report boundaries, never a per-node/device synchronization."""
    result={}
    for name in ('node_counts','edge_counts','ppr_core_nodes','path_nodes','halo_nodes','retained_mass'):
        value=audit[name].detach().float()
        result[name]=dict(min=float(value.min()),max=float(value.max()),mean=float(value.mean()))
    result.update(candidate_nodes=audit['candidate_nodes'],candidate_edges=audit['candidate_edges'],
                  outer_targets=audit['outer_targets'],ppr=audit['ppr'],astar_iterations=audit['astar_iterations'])
    return result

def loader(dataset, batch, workers, *, indices=None):
    data = dataset if indices is None else Subset(dataset, indices)
    options = dict(batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=1)
    return DataLoader(data, **options)

@torch.no_grad()
def build_memory(model, dataset, batch, workers):
    training = model.training; model.eval()
    values = []; completed = 0; began = last_log = time.perf_counter()
    for patches, indices in loader(dataset, batch, workers):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values.append(model.local(patches.cuda(non_blocking=True)).float().cpu())
        completed += len(patches)
        now = time.perf_counter()
        if now-last_log >= 30 or completed == len(dataset):
            print(json.dumps(dict(stage='support_memory',completed=completed,total=len(dataset),
                physical_batch=batch,workers=workers,graphs_per_second=completed/(now-began))),flush=True)
            last_log = now
    cases = sorted(set(r['case_id'] for r in dataset.rows)); lookup = {c: i for i, c in enumerate(cases)}
    memory = dict(embeddings=torch.cat(values).cuda(),
                  owners=torch.tensor([lookup[r['case_id']] for r in dataset.rows], device='cuda'),
                  classes=torch.tensor([r['target'] for r in dataset.rows], device='cuda'),
                  case_ids=cases,
                  patient_groups=[dataset.meta['identities']['cases'][c]['patient_group'] for c in cases])
    model.train(training)
    return memory


def calibrate_memory(model,dataset,cfg,output):
    """Measure full-topology L0 inference before encoding the entire support set."""
    maximum=max(len(ids) for _,ids in grouped_indices(dataset))
    patches=torch.stack([dataset[i][0] for i in range(maximum)])
    candidates=[1]
    while candidates[-1]<maximum:candidates.append(min(candidates[-1]*2,maximum))
    reports=[];training=model.training;model.eval()
    for count in candidates:
        query=result=None
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        baseline,budget=execution_budget(cfg)
        admission=batch_admission(count,reports,baseline,budget,nonlinear_sampler=True)
        if not admission['admitted']:
            reports.append(dict(physical_batch=count,accepted=False,executed=False,
                reason='predicted_peak_exceeds_VRAM_reserve',**admission))
            print(json.dumps(dict(stage='memory_batch_calibration',**reports[-1])),flush=True)
            break
        try:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                query=patches[:count].cuda()
                result=model.local(query);torch.cuda.synchronize()  # warm-up
                del result;result=None
                started=time.perf_counter()
                for _ in range(cfg['batch_calibration_repeats']):
                    result=model.local(query)
                    del result;result=None
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
            peak=torch.cuda.max_memory_allocated()
            reports.append(dict(physical_batch=count,graphs_per_second=count*cfg['batch_calibration_repeats']/elapsed,
                peak_vram_bytes=peak,accepted=peak<budget,executed=True,budget_bytes=budget))
            del query;query=None
        except torch.cuda.OutOfMemoryError as error:
            reports.append(dict(physical_batch=count,accepted=False,error=str(error),
                policy='Inference batch only; complete graph, model and cohort retained'))
            query=result=None;torch.cuda.empty_cache()
            print(json.dumps(dict(stage='memory_batch_calibration',**reports[-1])),flush=True)
            break
        print(json.dumps(dict(stage='memory_batch_calibration',**reports[-1])),flush=True)
        release_probe_memory(model)
        if not reports[-1]['accepted']:break
    write_new(Path(output)/'memory_batch_calibration.json',reports)
    valid=[r for r in reports if r['accepted']]
    if not valid:raise MemoryError('Full-graph L0 inference does not fit; no graph/model fallback')
    batch=max(valid,key=lambda r:r['graphs_per_second'])['physical_batch']
    # Recheck the selected batch after all larger probes and cache release.
    release_probe_memory(model)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        query=patches[:batch].cuda();result=model.local(query);del result
        torch.cuda.synchronize();started=time.perf_counter()
        for _ in range(cfg['batch_calibration_repeats']):
            result=model.local(query);del result
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        del query
    speed=batch*cfg['batch_calibration_repeats']/seconds
    reference=next(r['graphs_per_second'] for r in valid if r['physical_batch']==batch)
    write_new(Path(output)/'memory_selected_recheck.json',dict(physical_batch=batch,
        graphs_per_second=speed,calibration_graphs_per_second=reference,
        allocated_after=torch.cuda.memory_allocated(),reserved_after=torch.cuda.memory_reserved(),
        passed=speed>=reference*.5))
    if speed<reference*.5:raise RuntimeError('Selected inference batch lost over half its measured throughput after calibration; inspect paging before full support encoding')
    release_probe_memory(model)
    worker_reports=[]
    for workers in cfg['worker_candidates']:
        started=time.perf_counter();count=0
        for value,_ in loader(dataset,batch,workers):count+=len(value)
        row=dict(workers=workers,samples=count,seconds=time.perf_counter()-started)
        worker_reports.append(row)
        print(json.dumps(dict(stage='memory_worker_calibration',**row)),flush=True)
    write_new(Path(output)/'memory_worker_calibration.json',worker_reports)
    workers=min(worker_reports,key=lambda r:r['seconds'])['workers']
    model.train(training)
    return batch,workers

def calibrate(model, dataset, memory, cfg, output):
    """Real full-graph forward/backward probes; no weights updated or data truncated."""
    by_case = {}
    for i, row in enumerate(dataset.rows): by_case.setdefault(row['case_id'], []).append(i)
    # All patches have the exact same graph cardinality. Use the largest query case.
    indices = max(by_case.values(), key=len)
    patches = torch.stack([dataset[i][0] for i in indices])
    group = dataset.rows[indices[0]]['patient_group']
    support = support_for_query(memory, group)
    with torch.autocast('cuda', dtype=torch.bfloat16): cluster_plan = model.fit_support_clusters(*support)
    reports = []; candidates = [1]
    while candidates[-1] < len(indices): candidates.append(min(2*candidates[-1], len(indices)))
    model.train()
    for count in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        query = targets = loss = None
        baseline,budget=execution_budget(cfg)
        admission=batch_admission(count,reports,baseline,budget,nonlinear_sampler=True)
        if not admission['admitted']:
            reports.append(dict(physical_batch=count,accepted=False,executed=False,
                reason='predicted_peak_exceeds_VRAM_reserve',**admission))
            print(json.dumps(dict(stage='training_batch_calibration',**reports[-1])),flush=True)
            break
        try:
            query = patches[:count].cuda()
            targets = torch.tensor([dataset.rows[i]['target'] for i in indices[:count]], device='cuda')
            start = time.perf_counter()
            for _ in range(cfg['batch_calibration_repeats']):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = supervised_loss(model(query, *support, cluster_plan=cluster_plan), targets)
                loss.backward()
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated()
            reports.append(dict(physical_batch=count, graphs_per_second=count*cfg['batch_calibration_repeats']/elapsed,
                                peak_vram_bytes=peak, seconds=elapsed,
                                accepted=peak < budget,executed=True,budget_bytes=budget))
            print(json.dumps(dict(stage='training_batch_calibration',**reports[-1])),flush=True)
            del loss, query, targets
            release_probe_memory(model)
            if not reports[-1]['accepted']:break
        except torch.cuda.OutOfMemoryError as error:
            reports.append(dict(physical_batch=count, accepted=False, error=str(error),
                                policy='Only measured physical batch changes; architecture and full graph unchanged'))
            query = targets = loss = None
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            break
    model.zero_grad(set_to_none=True)
    write_new(Path(output)/'batch_calibration.json', reports)
    valid = [r for r in reports if r['accepted']]
    if not valid: raise MemoryError('No full-graph training batch fits; inspect calibration, do not reduce model')
    batch = max(valid, key=lambda r: r['graphs_per_second'])['physical_batch']
    worker_reports = []
    for workers in cfg['worker_candidates']:
        start = time.perf_counter(); count = 0
        for value, _ in loader(dataset, batch, workers): count += len(value)
        worker_reports.append(dict(workers=workers, samples=count, seconds=time.perf_counter()-start))
        print(json.dumps(dict(stage='training_worker_calibration',**worker_reports[-1])),flush=True)
    write_new(Path(output)/'worker_calibration.json', worker_reports)
    workers = min(worker_reports, key=lambda r: r['seconds'])['workers']
    return batch, workers, reports

def grouped_indices(dataset, rng=None):
    groups = {}
    for i, row in enumerate(dataset.rows): groups.setdefault(row['patient_group'], []).append(i)
    names = list(groups)
    if rng is not None: rng.shuffle(names)
    for group in names:
        indices = np.asarray(groups[group])
        if rng is not None: rng.shuffle(indices)
        yield group, indices.tolist()

class PatientBatchSampler:
    """One persistent loader; all batches retain a single query patient group."""
    def __init__(self, dataset, batch, seed=None):
        self.dataset, self.batch, self.seed, self.epoch = dataset, batch, seed, 0
    def __iter__(self):
        rng = None if self.seed is None else np.random.default_rng(self.seed+self.epoch)
        for _, indices in grouped_indices(self.dataset, rng):
            for begin in range(0, len(indices), self.batch):
                yield indices[begin:begin+self.batch]
    def __len__(self):
        return sum(math.ceil(len(ids)/self.batch) for _, ids in grouped_indices(self.dataset))

def grouped_loader(dataset, batch, workers, seed=None):
    sampler = PatientBatchSampler(dataset, batch, seed)
    kwargs = dict(batch_sampler=sampler, num_workers=workers, pin_memory=True)
    if workers: kwargs.update(persistent_workers=True, prefetch_factor=1)
    return DataLoader(dataset, **kwargs), sampler

@torch.no_grad()
def evaluate(model, dataset, memory, batch, workers):
    model.eval(); logits = []; targets = []
    iterator, _ = grouped_loader(dataset, batch, workers)
    last_group = None
    for patches, ids in iterator:
        group = dataset.rows[int(ids[0])]['patient_group']
        if group != last_group:
            support = support_for_query(memory, group)
            with torch.autocast('cuda', dtype=torch.bfloat16): state = model.prepare_support(*support)
            last_group = group
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model.predict_embeddings(model.local(patches.cuda(non_blocking=True)), state)
        logits.append(output['logits'].cpu())
        targets.extend(dataset.rows[i]['target'] for i in ids.tolist())
    logits = torch.cat(logits); targets = torch.tensor(targets)
    prob = logits.softmax(-1)[:, 1].numpy(); y = targets.numpy()
    from scipy.stats import rankdata
    positive = int(y.sum()); negative = len(y)-positive
    if not positive or not negative: raise ValueError('Both validation observation classes required')
    auc = float((rankdata(prob)[y == 1].sum()-positive*(positive+1)/2)/(positive*negative))
    return dict(observation_cross_entropy=float(torch.nn.functional.cross_entropy(logits, targets)),
                observation_accuracy=float((logits.argmax(-1) == targets).float().mean()),
                observation_auroc=auc, samples=len(y), positives=positive, negatives=negative,
                scope='constructed observation task; not tumor incidence probability or segmentation performance')

def train(cache, output, cfg, base):
    from . import TRAINING_READY, DESIGN_BLOCK_REASON
    if not TRAINING_READY:
        raise RuntimeError(DESIGN_BLOCK_REASON)
    configure_runtime(base, cfg['seed']); require_device('cuda')
    root = safe_new_root(output)
    trainset = ContextDataset(cache, 'inner_train'); valset = ContextDataset(cache, 'inner_val')
    if trainset.meta['config'] != cfg or trainset.meta['base'] != base:
        raise ValueError('Cache/model configuration differs')
    contract = trainset.meta['input_contract']
    model = PromptGraphModel(cfg, base, contract).cuda()
    write_new(root/'initialization.json',dict(format=PIPELINE_VERSION,seed=cfg['seed'],
        config=cfg,base=base,input_contract=contract,identities=trainset.meta['identities'],
        source_identity=source_identity(),train_samples=len(trainset),validation_samples=len(valset),
        parameters=sum(p.numel() for p in model.parameters()),candidate_nodes_per_graph=len(model.local.grid),
        candidate_edges_per_graph=model.local.edge.shape[1],graph_sampling=cfg['graph_sampling'],resources=snapshot(),
        gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),debug=False,subset=False))
    memory_batch,memory_workers=calibrate_memory(model,trainset,cfg,root)
    memory = build_memory(model, trainset, memory_batch, memory_workers)
    batch, workers, measurements = calibrate(model, trainset, memory, cfg, root)
    counts = torch.bincount(memory['classes'], minlength=2).float()
    if bool((counts == 0).any()): raise ValueError('Both training classes required')
    weights = counts.sum()/(2*counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base['training']['lr'], weight_decay=base['training']['weight_decay'])
    epochs = cfg['gnn_epochs']; best = math.inf; best_path = None
    total_steps = epochs*sum(math.ceil(len(ids)/batch) for _, ids in grouped_indices(trainset))
    report = dict(format=PIPELINE_VERSION, config=cfg, base=base, input_contract=contract,
                  parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  input_shape=[batch, 1, 48, 48, 48], candidate_nodes_per_graph=len(model.local.grid),
                  candidate_edges_per_graph=model.local.edge.shape[1],graph_sampling=cfg['graph_sampling'],
                  measured_sampled_graphs=graph_statistics(model.local.sampler.last_audit),physical_batch=batch, gradient_accumulation=1,
                  effective_batch=batch, epochs=epochs, optimization_steps=total_steps,
                  memory_physical_batch=memory_batch,memory_workers=memory_workers,
                  train_samples=len(trainset), used_train_samples=len(trainset), train_usage_ratio=1.0,
                  val_samples=len(valset), workers=workers, persistent_workers=workers > 0,
                  prefetch_factor=1 if workers else None, pin_memory=True, precision='bfloat16_autocast_FP32_loss',
                  gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(), resources=snapshot(),
                  support_memory='complete inner_train, detached embeddings refreshed every epoch; all queries train L0',
                  debug=False, subset=False, peak_vram=max(r.get('peak_vram_bytes', 0) for r in measurements))
    write_new(root/'training_started.json', report)
    step = 0; previous_clusters = {}
    query_loader, sampler = grouped_loader(trainset, batch, workers, cfg['seed'])
    for epoch in range(epochs):
        # Calibration never updates weights. The preceding validation memory is
        # already current at the next epoch start; do not encode it twice.
        model.train(); seen = set(); begin = time.perf_counter(); losses = []; alignment_losses = []
        last_progress=begin
        cluster_audits = {}; epoch_nodes=[]; epoch_edges=[]
        sampler.epoch = epoch; last_group = None
        for patches, ids in query_loader:
            group = trainset.rows[int(ids[0])]['patient_group']
            if group != last_group:
                support = support_for_query(memory, group); last_group = group
                with torch.autocast('cuda', dtype=torch.bfloat16): cluster_plan = model.fit_support_clusters(*support)
                # Epoch-to-epoch stability is a diagnostic, never a query-label
                # criterion for selecting prototypes or cluster count.
                assignment = cluster_plan['assignment'].cpu().numpy()
                active_classes = (cluster_plan['active']%2).cpu().numpy()
                stability = None
                if group in previous_clusters:
                    from sklearn.metrics import adjusted_rand_score
                    old = previous_clusters[group]
                    stability = [float(adjusted_rand_score(old[active_classes==c],assignment[active_classes==c])) for c in (0,1)]
                previous_clusters[group] = assignment.copy()
                cluster_audits[group] = dict(**cluster_plan['audit'],
                    support_patient_groups=sorted(set(memory['patient_groups'])-{group}),
                    excluded_query_patient_group=group, classwise_previous_epoch_ARI=stability)
            targets = torch.tensor([trainset.rows[i]['target'] for i in ids.tolist()], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output_ = model(patches.cuda(non_blocking=True), *support, cluster_plan=cluster_plan)
                loss = supervised_loss(output_, targets, weights)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base['training']['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; seen.update(ids.tolist()); losses.append(loss.detach())
            epoch_nodes.append(model.local.sampler.last_audit['node_counts'].detach())
            epoch_edges.append(model.local.sampler.last_audit['edge_counts'].detach())
            alignment_losses.append(output_['alignment_loss'].detach())
            now=time.perf_counter()
            if step==1 or now-last_progress>=30:
                print(json.dumps(dict(stage='optimization',epoch=epoch+1,step=step,total_steps=total_steps,
                    visited_queries=len(seen),total_queries=len(trainset),loss=float(loss.detach()),
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    sampled_graphs=graph_statistics(model.local.sampler.last_audit))),flush=True)
                last_progress=now
        if seen != set(range(len(trainset))): raise RuntimeError('Incomplete epoch coverage')
        memory = build_memory(model, trainset, memory_batch, memory_workers)
        metrics = evaluate(model, valset, memory, batch, workers)
        elapsed = time.perf_counter()-begin
        row = dict(epoch=epoch+1, optimization_steps=step, train_loss=float(torch.stack(losses).mean()),
                   alignment_loss=float(torch.stack(alignment_losses).mean()), cluster_episodes=cluster_audits,
                   validation=metrics, all_train_queries_visited=len(seen), seconds=elapsed,
                   query_graphs_per_second=len(seen)/elapsed, peak_vram=torch.cuda.max_memory_allocated(), resources=snapshot(),
                   sampled_nodes=dict(min=int(torch.cat(epoch_nodes).min()),max=int(torch.cat(epoch_nodes).max()),mean=float(torch.cat(epoch_nodes).float().mean())),
                   sampled_edges=dict(min=int(torch.cat(epoch_edges).min()),max=int(torch.cat(epoch_edges).max()),mean=float(torch.cat(epoch_edges).float().mean())))
        write_new(root/f'epoch_{epoch+1:03d}.json', row); print(row, flush=True)
        if metrics['observation_cross_entropy'] < best:
            best = metrics['observation_cross_entropy']; best_path = root/f'weights_epoch_{epoch+1:03d}.pt'
            with best_path.open('xb') as stream: torch.save(dict(state_dict=model.state_dict(), epoch=epoch+1), stream)
    best_state = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(best_state['state_dict']); memory = build_memory(model, trainset, memory_batch, memory_workers)
    memory = {k: v.cpu() if torch.is_tensor(v) else v for k, v in memory.items()}
    meta = trainset.meta
    payload = dict(format=PIPELINE_VERSION, debug=False, completed_epochs=epochs, selected_epoch=best_state['epoch'],
                   state_dict={k: v.cpu() for k, v in model.state_dict().items()}, config=cfg, base=base,
                   input_contract=contract, split=meta['split'], identities=meta['identities'], memory=memory,
                   donor_pool=meta['donor_pool'], raw_records=meta['raw_records'], physical_batch=batch, workers=workers,
                   source_identity=source_identity(), cache_sha256=sha(cache), optimization_steps=step)
    with (root/'checkpoint.pt').open('xb') as stream: torch.save(payload, stream)
    write_new(root/'training_complete.json', dict(format=PIPELINE_VERSION, epochs=epochs, selected_epoch=best_state['epoch'],
              checkpoint_sha256=sha(root/'checkpoint.pt'), segmentation_training_executed=False))
    return root/'checkpoint.pt'
