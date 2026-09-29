"""Read metadata/logs only: no torch, graph loading, training or checkpoint writes.

One run directory per invocation avoids double-counting resumed/repeated steps.
Unknown epoch phases stay unknown. No local-to-MIG speedup extrapolation.
"""
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def schedule(meta, physical_batch, memory_batch):
    if min(physical_batch, memory_batch) < 1:
        raise ValueError('Positive explicit physical and memory batches required')
    result = {}
    for partition in ('inner_train', 'inner_val'):
        allowed = set(meta['split'][partition])
        rows = [r for r in meta['records'] if r['case_id'] in allowed]
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate observation IDs')
        groups = Counter(r['patient_group'] for r in rows)
        result[partition] = dict(records=len(rows), recipient_groups=len(groups),
            missing_cases=sorted(allowed-set(r['case_id'] for r in rows)),
            grouped_query_batches=sum(math.ceil(n/physical_batch) for n in groups.values()),
            contiguous_memory_batches=math.ceil(len(rows)/memory_batch))
    result['cache_debug'] = meta.get('debug')
    result['schedule_scope'] = 'metadata counts; actual tensors/provenance/admission not verified'
    return result


def read_jsonl(path):
    if not path.exists():
        return None
    # Never silently drop a malformed/partially written log row.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def summarize(rows, events, *, epoch, first_step=None, last_step=None):
    chosen = [] if rows is None else [r for r in rows if r.get('stage')=='optimization'
        and r.get('epoch')==epoch and (first_step is None or r['step']>=first_step)
        and (last_step is None or r['step']<=last_step)]
    steps = [r['step'] for r in chosen]
    if len(set(steps)) != len(steps):
        raise ValueError('Duplicate optimization steps: separate run segments before summation')
    result = dict(epoch=epoch, step_log_present=rows is not None, measured_updates=len(chosen),
        first_step=min(steps) if steps else None, last_step=max(steps) if steps else None,
        means={}, measured_step_seconds=None, complete_epoch_seconds=None,
        runtime_phase_seconds=None, epoch_status='UNKNOWN_NO_COMPLETE_EVENT')
    for key in ('step_seconds','loader_wait_seconds','support_plan_seconds','H2D_seconds',
                'forward_seconds','backward_seconds','optimizer_seconds','checkpoint_seconds'):
        values = [r[key] for r in chosen if key in r]
        if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('Invalid timing: '+key)
        result['means'][key] = dict(count=len(values), mean=statistics.mean(values) if values else None)
    if chosen and result['means']['step_seconds']['count']==len(chosen):
        result['measured_step_seconds'] = sum(r['step_seconds'] for r in chosen)
    completed = [r for r in (events or []) if r.get('stage')=='epoch_complete' and r.get('epoch')==epoch]
    if len(completed)>1:
        raise ValueError('Duplicate epoch completion events')
    if completed:
        event=completed[0]
        parts=event.get('runtime_phase_seconds')
        if isinstance(parts,dict) and set(parts)=={'optimization','refresh_memory','validation'}:
            if any(not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in parts.values()):
                raise ValueError('Invalid phase timings')
            if not math.isclose(sum(parts.values()),event['seconds'],rel_tol=1e-7,abs_tol=1e-5):
                raise ValueError('Phase sum disagrees with epoch time')
            result.update(complete_epoch_seconds=event['seconds'],runtime_phase_seconds=parts,
                epoch_status='REPORTED_ACTIVE_PHASE_TIME',time_scope=event.get('time_scope'))
    result['boundary_notes'] = [
        'Runtime epoch time excludes initial/final memory and paused downtime.',
        'Legacy epoch_NNN.json seconds is optimization-only; not used as whole epoch.',
        'Step checkpoint_seconds is submission/previous-writer wait, not necessarily current durable write.',
        'Do not add loader/save to runtime phase seconds again: they are already within phase intervals.',
        'No GraphSAGE epoch estimate without matched hardware/batch/full-support timings.']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True,help='training directory containing JSONL logs')
    for key in ('physical-batch','memory-batch','epoch'):
        parser.add_argument('--'+key,type=int,required=True)
    for key in ('first-step','last-step'):
        parser.add_argument('--'+key,type=int)
    args=parser.parse_args()
    if args.epoch<1 or (args.first_step is not None and args.last_step is not None and args.first_step>args.last_step):
        raise ValueError('Invalid epoch/step range')
    paths=[args.cache,args.run/'step_timings.jsonl',args.run/'runtime_events.jsonl']
    contract_path=args.run/'training_started.json'
    contract=json.loads(contract_path.read_text(encoding='utf-8')) if contract_path.exists() else None
    if contract is not None:
        for key,requested in (('physical_batch',args.physical_batch),('memory_physical_batch',args.memory_batch)):
            if contract.get(key) is not None and contract[key]!=requested:
                raise ValueError(f'{key} argument disagrees with training_started.json')
        paths.append(contract_path)
    report=dict(readonly=True,training_started=False,production_admitted=False,
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in paths},
        supplied_batches=dict(query=args.physical_batch,memory=args.memory_batch),
        saved_execution=None if contract is None else {k:contract.get(k) for k in
            ('physical_batch','memory_physical_batch','precision','dense_execution_chunk','optimization_steps','epochs','gpu','workers')},
        schedule=schedule(json.loads(args.cache.read_text(encoding='utf-8')),args.physical_batch,args.memory_batch),
        timing=summarize(read_jsonl(paths[1]),read_jsonl(paths[2]),epoch=args.epoch,
            first_step=args.first_step,last_step=args.last_step))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
