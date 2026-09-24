"""Append-only recovery of paired preparation after an unrepresentable donor draw.

Recipient observations and the global donor pool remain complete. A rejected
GNN pair gets another seed-determined donor; masks are never fabricated or grown.
This does not change the native CP event donor schedule or resampling kernel.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import io
import os
import time
import numpy as np
import torch
from hiercp.common import stable_case_seed
from hiercp_v22.data import donor_in_target_spacing
from hiercp_v22.storage import load_record
from hiercp_v22.resources import _local_bound
from .contracts import read_json,write_new,sha
from .v1_cache import configuration,assign_pairs,provenance,emit
from .v1_local import RECORD_FORMAT

def valid_transport(donor,spacing):
    try:
        donor_in_target_spacing(donor['source'],donor['spacing'],np.asarray(spacing))
        return True
    except ValueError as error:
        if str(error)!='Real donor disappears at target spacing; no fabricated footprint':raise
        return False

def transport_preflight(rows,meta,cfg,donor_loader):
    """Check every assigned geometry before generating any new expensive graph."""
    raw={r['case_id']:r for r in meta['raw_records']}
    cache={};rejected=[];output=[]
    for row in rows:
        row=dict(row)
        def compatible(d):
            key=(d['case_id'],int(d['component_id']),tuple(raw[row['case_id']]['spacing']))
            if key not in cache:
                value=donor_loader(dict(donor_case_id=key[0],donor_component=key[1]))
                cache[key]=valid_transport(value,key[2])
            return cache[key]
        original=dict(case_id=row['donor_case_id'],component_id=row['donor_component'])
        if not compatible(original):
            allowed=[d for d in meta['donor_pool'] if meta['identities']['cases'][d['case_id']]['patient_group']!=row['patient_group']]
            rng=np.random.default_rng(stable_case_seed(cfg['seed'],row['id'],'v1-pair-donor'))
            draw=int(rng.integers(len(allowed)))
            if allowed[draw]!=original:raise ValueError('Original seeded donor draw changed')
            candidates=rng.permutation(len(allowed))
            chosen=next((allowed[int(i)] for i in candidates if compatible(allowed[int(i)])),None)
            if chosen is None:raise ValueError(f'No representable donor for observation {row["id"]}; no record dropped')
            row.update(donor_case_id=chosen['case_id'],donor_component=chosen['component_id'],
                       donor_group=meta['identities']['cases'][chosen['case_id']]['patient_group'])
            rejected.append(dict(observation=row['id'],original=original,replacement=chosen,
                reason='nearest_neighbor_mask_empty_at_recipient_spacing',observation_retained=True,
                label_used_for_selection=False))
        output.append(row)
    return output,dict(observations=len(output),unique_geometry_checks=len(cache),rejected_draws=rejected,
        dropped_observations=0,global_donor_pool_unchanged=True,interpolation_unchanged=True,
        policy='seeded_uniform_initial_draw_then_label_blind_permutation_of_representable_donors',
        scope='GNN observation pair construction only; native CP event scheduling unchanged')

def validate_reuse_sources(old,current):
    allowed={'run_v222_v1_l0.py','hiercp_v222/v1_cache.py','hiercp_v222/v1_recovery.py'}
    changed=[k for k in set(old)|set(current) if old.get(k)!=current.get(k)]
    geometry_changes=set(changed)-allowed
    if geometry_changes:
        # Exact reviewed revision pair, never a general cache-version bypass.
        from .contracts import ROOT
        receipt=read_json(ROOT/'work/v222_v1_empty_context_check_20260924/result.json')
        verified={'hiercp_v22/spatial.py','hiercp_v22/sample.py','hiercp_v222/v1_local.py'}
        if (geometry_changes-verified or not receipt['nonempty_graph_CT_edges_exact']
            or receipt['nonempty_epoch_views_exact']!=[0,1,39]
            or not receipt['all_parameter_gradients_finite']
            or any(old.get(k)!=receipt['old_source_identity'].get(k)
                or current.get(k)!=receipt['current_source_identity'].get(k) for k in geometry_changes)):
            raise ValueError(f'Unsafe old graph reuse; unverified construction revision: {changed}')
    return changed

def recover_files(previous,root,rows,old_rows):
    """Verify and hard-link immutable existing files into a NEW run directory."""
    previous=Path(previous).resolve();root=Path(root).resolve()
    by_id={r['id']:r for r in old_rows};receipts={}
    for file in (previous/'cases').glob('*.json'):
        for row in read_json(file):receipts[row['id']]=row
    def link(source,target):
        source=source.resolve();target=target.resolve()
        if not source.is_relative_to(previous) or not target.is_relative_to(root):raise ValueError('Recovery path escapes run roots')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(source,target)
    shared={};reused=[]
    for row in rows:
        old=by_id[row['id']]
        if any(row[k]!=old[k] for k in row):continue
        relative=f'graphs/{row["case_id"]}/{row["id"].split(":")[-1]}.pt.gz'
        file=previous/relative
        if not file.exists():continue
        digest=sha(file)
        if row['id'] in receipts and digest!=receipts[row['id']]['sha256']:
            raise ValueError('Existing graph differs from case receipt')
        # Validate also files completed after the failure but before executor drain.
        with gzip.open(file,'rb') as stream:
            payload=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=False)
        reference=payload['shared_source']
        if reference['path'] not in shared:
            if sha(previous/reference['path'])!=reference['sha256']:raise ValueError('Existing source digest mismatch')
            link(previous/reference['path'],root/reference['path']);shared[reference['path']]=reference
        record=load_record(previous,relative)
        if (record['format']!=RECORD_FORMAT or record['center_masking'] is not False
            or record['case_id']!=row['case_id'] or record['donor_case_id']!=row['donor_case_id']
            or record['component_id']!=row['donor_component'] or record['center']!=row['center']):
            raise ValueError('Partial graph input identity mismatch')
        a,b=(_local_bound(record[k]) for k in ('source_local','target_local'))
        bounds=dict(nodes=a[0]+b[0],edges=a[1]+b[1],bytes=a[2]+b[2]+record['source_patch'].numel()*4+record['target_patch'].numel()*4)
        link(file,root/relative)
        reused.append(row|dict(path=relative,sha256=digest,bounds=bounds,shared_source=reference))
        if len(reused)%256==0:emit(stage='verified_graph_reuse',graphs=len(reused))
    return reused
