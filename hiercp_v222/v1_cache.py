"""Lossless paired cache for v1-style L0; complete existing observation cohort."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import io
import json
from pathlib import Path
import threading
import time
import numpy as np
import psutil
import torch
from hiercp.common import CasePaths, load_case, organ_depth_mm, stable_case_seed
from hiercp_v22.data import sources
from hiercp_v22.storage import GraphWriter, load_record
from hiercp_v22.volumes import volume_memory_bound
from .contracts import ROOT, read_json, write_new, sha, validate_identities, verify_v1
from .v1_local import prepare_donor, pair_record, materialize, collate, ENCODER_ID

CACHE_FORMAT='v222_v1_paired_cohort_v1'

def configuration():
    cfg=read_json(ROOT/'config/prompt_graph_v222_v1_l0.json')
    base=read_json(ROOT/cfg['base_config'])
    expected=dict(local_encoder=ENCODER_ID,seed=42,gnn_epochs=40,nnunet_epochs=250,
                  cp_probability=.8,candidate_count=128,comparison_centers_per_patient=128,
                  task_layers=2,alignment_layers=2,gradient_accumulation=1)
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Approved paired model/training contract changed')
    verify_v1()
    return cfg,base

def provenance():
    paths=[ROOT/'run_v222_v1_l0.py',ROOT/'config/prompt_graph_v222_v1_l0.json',ROOT/'config/train.json']
    for directory in ('hiercp','hiercp_v22','hiercp_v222'):
        paths.extend((ROOT/directory).glob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def emit(**value):
    print(json.dumps(value,ensure_ascii=False),flush=True)

def assign_pairs(meta, cfg):
    """One seed-fixed uniform training-pool donor per observation, independent of class.

    The full existing recipient observation set is retained. This is not the
    Cartesian product of 14k centers and 527 donors. Online CP still draws per event.
    """
    validate_identities(meta['identities'],meta['split'])
    pool=meta['donor_pool']
    if not pool or any(d['case_id'] not in meta['split']['inner_train'] for d in pool):
        raise ValueError('Donor pool must be nonempty and exclusively inner train')
    if len({(d['case_id'],d['component_id']) for d in pool})!=len(pool):
        raise ValueError('Duplicate donor entries')
    rows=[]
    for row in meta['records']:
        group=meta['identities']['cases'][row['case_id']]['patient_group']
        allowed=[d for d in pool if meta['identities']['cases'][d['case_id']]['patient_group']!=group]
        if not allowed:raise ValueError('No independent training donor')
        rng=np.random.default_rng(stable_case_seed(cfg['seed'],row['id'],'v1-pair-donor'))
        donor=allowed[int(rng.integers(len(allowed)))]
        rows.append({k:row[k] for k in ('id','case_id','patient_group','component','center','target')}
                    |dict(donor_case_id=donor['case_id'],donor_component=donor['component_id'],
                          donor_group=meta['identities']['cases'][donor['case_id']]['patient_group']))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate observations')
    for c in meta['split']['outer_train']:
        actual=[r for r in rows if r['case_id']==c]
        raw=next(r for r in meta['raw_records'] if r['case_id']==c)
        if sum(r['target']==0 for r in actual)!=128 or sum(r['target']==1 for r in actual)!=len(raw['positives']):
            raise ValueError(f'Incomplete full observation coverage: {c}')
    if set(r['case_id'] for r in rows)!=set(meta['split']['outer_train']):
        raise ValueError('Outer-validation input or missing outer-training case')
    return rows

def load_verified(info):
    paths=CasePaths(info['case_id'],Path(info['image']),Path(info['label']))
    if sha(paths.image_path)!=info['image_sha256'] or sha(paths.label_path)!=info['label_sha256']:
        raise ValueError(f'Original CT/mask changed: {info["case_id"]}')
    case=load_case(paths)
    organ=np.isin(case.label,[1,2])
    return case,organ,organ_depth_mm(organ,case.spacing)

def save_torch_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(value,stream)

def prepare(index, output, *, reuse=None):
    cfg,base=configuration(); meta=read_json(index)
    rows=assign_pairs(meta,cfg); root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    raw={r['case_id']:r for r in meta['raw_records']}
    previous=Path(reuse).resolve() if reuse else None
    if previous is not None:
        from .v1_recovery import validate_reuse_sources
        previous_meta=read_json(previous/'started.json')
        if previous_meta['config']!=cfg or previous_meta['base']!=base or previous_meta['source_index_sha256']!=sha(index):
            raise ValueError('Previous preparation used different data/config')
        changed=validate_reuse_sources(previous_meta['source_identity'],provenance())
        write_new(root/'reuse_source_audit.json',dict(previous=str(previous),changed_files=changed,
            nonempty_graph_semantics_verified=True,empty_context_is_explicit_recipient_only=True))
    write_new(root/'started.json',dict(format=CACHE_FORMAT,config=cfg,base=base,
        source_index=str(Path(index).resolve()),source_index_sha256=sha(index),
        reused='split, identities, observation centers and raw hashes only; no old CT tensors',
        total_observations=len(rows),donors=len(meta['donor_pool']),source_identity=provenance(),
        pairing='seed-fixed inner-train draw; label-blind redraw only when recipient-grid mask is empty; all observations and global donor pool retained',
        debug=False,subset=False,raw_volumes_duplicated=False))
    torch.set_num_threads(1)
    bounds={c:volume_memory_bound(r['image']) for c,r in raw.items()}
    # Full volume EDT peak admitted before concurrency. This limits workers only.
    safe=min(psutil.cpu_count(logical=False),int(psutil.virtual_memory().available*.40/max(bounds.values())))
    if safe<1:raise MemoryError('Full case exceeds RAM reserve; no graph/model reduction')
    donor_cases=sorted({d['case_id'] for d in meta['donor_pool']})
    donor_files={}
    def build_donors(c):
        case,organ,depth=load_verified(raw[c]); results={}
        for source,diameter in sources(case,base['cache']['source_pad'],cfg['donor_max_diameter_mm']):
            key=f'{c}:{source.component_id}'
            prepared=prepare_donor(case,source,organ,depth,base)
            # Transport payload has explicit patch-local coordinates, not a
            # retained full-volume mask per component.
            origin=np.array([s.start for s in source.patch_slices])
            transport=replace(source,full_mask=source.patch_mask,
                anchor_center=tuple(np.array(source.anchor_center)-origin),
                centroid=tuple(np.array(source.centroid)-origin),
                patch_slices=tuple(slice(0,n) for n in source.patch_mask.shape))
            file=root/'donors'/f'{c}_{source.component_id}.pt'
            save_torch_new(file,dict(source=transport,prepared=prepared,spacing=case.spacing,diameter=diameter))
            results[key]=dict(path=file.relative_to(root).as_posix(),sha256=sha(file))
        emit(stage='paired_donors',case=c,donors=len(results),workers=safe)
        return results
    if previous is None:
        with ThreadPoolExecutor(max_workers=safe) as pool:
            for result in pool.map(build_donors,donor_cases):donor_files.update(result)
    else:
        import os
        donor_files=read_json(previous/'donors.json')
        for entry in donor_files.values():
            source=(previous/entry['path']).resolve();destination=(root/entry['path']).resolve()
            if not source.is_relative_to(previous) or not destination.is_relative_to(root):raise ValueError('Unsafe donor path')
            if sha(source)!=entry['sha256']:raise ValueError('Existing donor hash mismatch')
            destination.parent.mkdir(parents=True,exist_ok=True);os.link(source,destination)
        emit(stage='verified_donor_reuse',donors=len(donor_files))
    expected={f'{d["case_id"]}:{d["component_id"]}' for d in meta['donor_pool']}
    if set(donor_files)!=expected:raise ValueError('Actual donor inventory differs from complete pool')
    write_new(root/'donors.json',donor_files)
    donor_lock=threading.Lock(); donor_cache={}
    def get_donor(row):
        key=f'{row["donor_case_id"]}:{row["donor_component"]}'
        with donor_lock:
            if key not in donor_cache:
                donor_cache[key]=torch.load(root/donor_files[key]['path'],weights_only=False,map_location='cpu',mmap=True)
            return donor_cache[key]
    from .v1_recovery import transport_preflight,recover_files
    rows,preflight=transport_preflight(rows,meta,cfg,get_donor)
    write_new(root/'transport_preflight.json',preflight)
    write_new(root/'pair_assignment.json',rows)
    emit(stage='all_pair_transport_preflight_complete',observations=len(rows),rejected_draws=len(preflight['rejected_draws']),dropped_observations=0)
    by_case={c:[r for r in rows if r['case_id']==c] for c in meta['split']['outer_train']}
    writer=GraphWriter(root,minimum_free_bytes=cfg['minimum_free_gb']*1024**3)
    prepared_rows=[]
    if previous is not None:
        prepared_rows=recover_files(previous,root,rows,read_json(previous/'pair_assignment.json'))
        for row in prepared_rows:writer.sources[f'{row["donor_case_id"]}:{row["donor_component"]}']=row['shared_source']
        write_new(root/'reused_graphs.json',prepared_rows)
        emit(stage='verified_graph_reuse_complete',graphs=len(prepared_rows))
    reused_ids={r['id'] for r in prepared_rows}
    # Measure identical full-geometry pair workload at safe concurrency levels.
    probe_case=max(by_case,key=lambda c:len(by_case[c]))
    case,organ,depth=load_verified(raw[probe_case])
    # Pair workers share both immutable source tables and one full recipient.
    # Do not incorrectly charge a full CT volume per independent graph worker.
    pair_bound=(base['graph']['adaptive_roi_max_voxels']*80+
                base['graph']['canonical_relation_edge_limit']*13*8)
    graph_safe=min(psutil.cpu_count(logical=False),int(psutil.virtual_memory().available*.40/pair_bound))
    if graph_safe<1:raise MemoryError('Full graph preparation exceeds RAM reserve')
    probe_rows=by_case[probe_case][:2*graph_safe]
    def probe(row):
        d=get_donor(row)
        rec=pair_record(case,d['source'],d['spacing'],d['prepared'],row['center'],organ,depth,base,donor_id=row['donor_case_id'])
        return sum(rec[b]['counts'][k] for b in ('source_local','target_local') for k in rec[b]['counts'])
    widths=sorted(set([1]+[n for n in (2,4,8) if n<=graph_safe]+[graph_safe]))
    reports=[]
    for width in widths:
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=width) as pool:list(pool.map(probe,probe_rows))
        report=dict(workers=width,pairs=len(probe_rows),seconds=time.perf_counter()-start,debug_calibration=True)
        reports.append(report);emit(stage='pair_preparation_calibration',**report)
    # Within-case independent graph builds use measured worker count. Full CT
    # loading is prefetched while current-case graph workers execute.
    workers=min(reports,key=lambda r:r['seconds'])['workers']
    write_new(root/'preparation_workers.json',dict(measurements=reports,selected=workers,
        case_prefetch=1,reason='same full-pair geometry workload; full-volume RAM admission',ram=psutil.virtual_memory()._asdict()))
    del case,organ,depth
    completed=len(prepared_rows); start=time.perf_counter();initial_completed=completed
    def build_one(row,loaded):
        case,organ,depth=loaded; d=get_donor(row)
        if row['target']==0 and case.label[tuple(row['center'])]!=1:
            raise ValueError('Comparison center no longer annotation liver')
        rec=pair_record(case,d['source'],d['spacing'],d['prepared'],row['center'],organ,depth,base,donor_id=row['donor_case_id'])
        relative=f'graphs/{row["case_id"]}/{row["id"].split(":")[-1]}.pt.gz'
        stored=writer.write(relative,rec,f'{row["donor_case_id"]}:{row["donor_component"]}')
        return row|stored
    cases=[c for c in by_case if any(r['id'] not in reused_ids for r in by_case[c])]
    if not cases:raise ValueError('Recovery requires at least one incomplete observation')
    with ThreadPoolExecutor(max_workers=1) as prefetch,ThreadPoolExecutor(max_workers=workers) as executor:
        future=prefetch.submit(load_verified,raw[cases[0]])
        for i,c in enumerate(cases):
            loaded=future.result()
            if i+1<len(cases):future=prefetch.submit(load_verified,raw[cases[i+1]])
            futures=[executor.submit(build_one,row,loaded) for row in by_case[c] if row['id'] not in reused_ids]
            case_rows=[]
            for result in futures:
                case_rows.append(result.result());completed+=1
                if completed%32==0:emit(stage='paired_cache',completed=completed,total=len(rows),case=c,
                    graphs_per_second=(completed-initial_completed)/(time.perf_counter()-start),workers=workers,rss=psutil.Process().memory_info().rss)
            write_new(root/'cases'/f'{c}.json',[r for r in prepared_rows if r['case_id']==c]+case_rows)
            prepared_rows.extend(case_rows);del loaded,futures
    if len(prepared_rows)!=len(rows):raise RuntimeError('Incomplete paired cohort')
    output_meta=dict(format=CACHE_FORMAT,complete=True,debug=False,config=cfg,base=base,
        split=meta['split'],identities=meta['identities'],records=sorted(prepared_rows,key=lambda r:r['id']),
        donor_pool=meta['donor_pool'],raw_records=meta['raw_records'],source_identity=provenance(),
        source_index_sha256=sha(index),donor_files=donor_files,preparation_workers=workers,
        transport_preflight=preflight,reused_graphs=len(reused_ids))
    write_new(root/'index.json',output_meta)
    emit(stage='paired_cache_complete',observations=len(rows),index=str(root/'index.json'))
    return root/'index.json'

class PairDataset:
    def __init__(self,index,partition):
        self.path=Path(index).resolve();self.root=self.path.parent;self.meta=read_json(index)
        if self.meta.get('format')!=CACHE_FORMAT or not self.meta.get('complete') or self.meta.get('debug'):
            raise ValueError('Complete paired production cache required')
        validate_identities(self.meta['identities'],self.meta['split'])
        if self.meta['source_identity']!=provenance():raise ValueError('Paired implementation changed after preparation')
        if partition not in ('inner_train','inner_val'):raise ValueError('No outer evaluation inputs')
        allowed=set(self.meta['split'][partition]);self.rows=[r for r in self.meta['records'] if r['case_id'] in allowed]
        if set(r['case_id'] for r in self.rows)!=allowed:raise ValueError('Missing cohort cases')
        self.cache=OrderedDict();self.lock=threading.Lock();self.bytes=0
        self.budget=int(psutil.virtual_memory().available*.20)
        self.verified=set()
    def __len__(self):return len(self.rows)
    def record(self,i):
        row=self.rows[i]
        with self.lock:
            if i in self.cache:
                self.cache.move_to_end(i);return self.cache[i][0]
        path=self.root/row['path']
        if i not in self.verified:
            if sha(path)!=row['sha256']:raise ValueError('Paired graph hash mismatch')
        value=load_record(self.root,row['path']);size=row['bounds']['bytes']
        with self.lock:
            self.verified.add(i)
            if i not in self.cache:
                while self.cache and self.bytes+size>self.budget:
                    _,(_,old)=self.cache.popitem(last=False);self.bytes-=old
                if size<=self.budget:self.cache[i]=(value,size);self.bytes+=size
        return value
    def item(self,i,epoch=0):return materialize(self.record(i),epoch=epoch),i

class PairLoader:
    """Persistent threaded decoding/sampling and one-batch pinned prefetch.

    Shares the canonical RAM cache rather than replicating it in Windows workers.
    GPU receives one disjoint-union batch, never one graph at a time.
    """
    def __init__(self,dataset,workers):
        self.dataset=dataset;self.workers=workers
        self.pool=ThreadPoolExecutor(max_workers=workers) if workers else None
        self.prefetch=ThreadPoolExecutor(max_workers=1)
    def make(self,indices,epoch):
        args=[(int(i),epoch) for i in indices]
        values=list(self.pool.map(lambda a:self.dataset.item(*a),args)) if self.pool else [self.dataset.item(*a) for a in args]
        return collate(values).pin_memory()
    def batches(self,groups,epoch=0):
        it=iter(groups);first=next(it,None)
        if first is None:return
        future=self.prefetch.submit(self.make,first,epoch)
        for ids in it:
            batch=future.result();future=self.prefetch.submit(self.make,ids,epoch);yield batch
        yield future.result()
    def close(self):
        self.prefetch.shutdown(wait=True)
        if self.pool:self.pool.shutdown(wait=True)
