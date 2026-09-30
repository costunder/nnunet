"""Rebuild donor-dependent geometry. Never relabel an old graph's donor."""
import copy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from tqdm import tqdm
from hiercp.common import stable_case_seed
from hiercp_v222.v1_cache import configuration,provenance,load_verified,save_torch_new
from hiercp_v222.v1_local import prepare_donor,pair_record
from hiercp_v222.v1_cache import sources,GraphWriter
from .training_data import sha,write_new,RegionDataset
from .fine_graph import FineDataset,FineLoader,OriginalFineDataset,verified,MODE
from .materialization import load_pairs
from .resident import check_verified
from .donor_learning import POLICY,validate_rows

FORMAT='same_donor_paired_cache_v1'

def assignment(meta,seed):
    pool=sorted(meta['donor_pool'],key=lambda d:(d['case_id'],d['component_id']))
    known=meta['identities']['cases'];train=set(meta['split']['inner_train'])
    if not pool or any(d['case_id'] not in train for d in pool):raise ValueError('Train-only full donor pool required')
    donors={}
    for case in sorted({r['case_id'] for r in meta['records']}):
        group=known[case]['patient_group'];eligible=[d for d in pool if known[d['case_id']]['patient_group']!=group]
        if not eligible:raise ValueError('No independent donor')
        rng=np.random.default_rng(stable_case_seed(seed,case,POLICY))
        donors[case]=eligible[int(rng.integers(len(eligible)))]
    rows=[]
    for r in meta['records']:
        d=donors[r['case_id']]
        rows.append({k:r[k] for k in ('id','case_id','patient_group','component','center','target')} |
            dict(donor_case_id=d['case_id'],donor_component=d['component_id'],donor_group=known[d['case_id']]['patient_group']))
    validate_rows(rows);return rows

def prepare(index,initialization,output,*,workers,budget,debug=False):
    from hiercp_v222.contracts import validate_identities
    from hiercp_v222.placement import GEOMETRY_CONTRACT
    cfg,base=configuration();index=Path(index).resolve();initialization=Path(initialization).resolve()
    meta=json.loads(index.read_text(encoding='utf8'));init=json.loads(initialization.read_text(encoding='utf8'))
    if bool(meta.get('debug'))!=debug or bool(init.get('debug'))!=debug:raise ValueError('Explicit matching DEBUG/production inputs required')
    if init['original_cache_sha256']!=sha(index):raise ValueError('Initialization and observation inventory differ')
    expected=copy.deepcopy(cfg)
    if debug:expected.update(gnn_epochs=1,batch_calibration_repeats=1)
    if meta['config']!=expected or meta['base']!=base:raise ValueError('Configuration differs')
    cfg=meta['config']
    # Read the existing verified initialization metadata, never its learned SAGE.
    RegionDataset(initialization,'inner_train',debug,init['profile_policy'])
    validate_identities(meta['identities'],meta['split'])
    if workers<1:raise ValueError('Explicit parallel worker count required')
    if not debug:
        for c in meta['split']['outer_train']:
            rs=[r for r in meta['records'] if r['case_id']==c]
            raw=next(r for r in meta['raw_records'] if r['case_id']==c)
            if sum(r['target']==0 for r in rs)!=128 or sum(r['target']==1 for r in rs)!=len(raw['positives']):raise ValueError('Full observation coverage required')
    rows=assignment(meta,cfg['seed']);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    write_new(root/'assignment.json',dict(policy=POLICY,rows=rows,debug=debug,source_sha256=sha(index)))
    raw={r['case_id']:r for r in meta['raw_records']};donors={}
    # Selected donors only are materialized; the complete eligible pool remains
    # recorded. No repeated full recipient volume per graph worker.
    for case in tqdm(sorted({r['donor_case_id'] for r in rows}),desc='same-donor sources'):
        ct,organ,depth=load_verified(raw[case]);wanted={r['donor_component'] for r in rows if r['donor_case_id']==case}
        for source,diameter in sources(ct,base['cache']['source_pad'],cfg['donor_max_diameter_mm']):
            if source.component_id in wanted:
                prepared=prepare_donor(ct,source,organ,depth,base)
                # Reuse the source transport convention of the original preparer.
                from dataclasses import replace
                origin=np.array([s.start for s in source.patch_slices])
                source=replace(source,full_mask=source.patch_mask,anchor_center=tuple(np.array(source.anchor_center)-origin),
                    centroid=tuple(np.array(source.centroid)-origin),patch_slices=tuple(slice(0,n) for n in source.patch_mask.shape))
                donors[(case,source.component_id)]=(source,ct.spacing,prepared)
        if {k[1] for k in donors if k[0]==case}!=wanted:raise ValueError('Selected donor missing from real CT')
        del ct,organ,depth;budget.check()
    writer=GraphWriter(root,minimum_free_bytes=cfg['minimum_free_gb']*2**30);results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for case in tqdm(sorted({r['case_id'] for r in rows}),desc='same-donor recipient cases'):
            ct,organ,depth=load_verified(raw[case]);selected=[r for r in rows if r['case_id']==case]
            def build(row):
                if row['target']==0 and ct.label[tuple(row['center'])]!=1:raise ValueError('Comparison center changed')
                source,spacing,prepared=donors[(row['donor_case_id'],row['donor_component'])]
                rec=pair_record(ct,source,spacing,prepared,row['center'],organ,depth,base,donor_id=row['donor_case_id'])
                stored=writer.write(f'graphs/{case}/{row["id"].split(":")[-1]}.pt.gz',rec,f'{row["donor_case_id"]}:{row["donor_component"]}')
                return row|stored
            for result in pool.map(build,selected):results.append(result);budget.check()
            del ct,organ,depth
    if {r['id'] for r in results}!={r['id'] for r in rows}:raise ValueError('Incomplete regenerated cohort')
    out={k:copy.deepcopy(meta[k]) for k in ('config','base','split','identities','donor_pool','raw_records')}
    out.update(format=FORMAT,complete=True,debug=debug,learning_policy=POLICY,records=sorted(results,key=lambda r:r['id']),
        source_identity=provenance(),geometry_contract=GEOMETRY_CONTRACT,initialization_sha256=sha(initialization),
        source_index_sha256=sha(index),workers=workers,donor_geometry_rebuilt=True)
    write_new(root/'index.json',out);return root/'index.json'

class DonorDataset:
    def __init__(self,index,partition,debug,profile_policy,original_index):
        original=RegionDataset(index,partition,debug,profile_policy);self.__dict__.update(original.__dict__)
        self.original_path=Path(original_index).resolve();m=json.loads(self.original_path.read_text(encoding='utf8'))
        if m.get('format')!=FORMAT or m.get('learning_policy')!=POLICY or m.get('complete') is not True:raise ValueError('Regenerated same-donor cache required')
        if m['debug']!=debug or m['initialization_sha256']!=sha(index):raise ValueError('Same-donor initialization mismatch')
        if m['source_index_sha256']!=self.meta['original_cache_sha256']:raise ValueError('Original observation provenance changed')
        if m['source_identity']!=provenance():raise ValueError('Same-donor geometry source changed')
        from hiercp_v222.contracts import validate_identities
        validate_identities(m['identities'],m['split'])
        expected_pairs={r['id']:r for r in assignment(m,m['config']['seed'])}
        for r in m['records']:
            if any(r[k]!=expected_pairs[r['id']][k] for k in ('donor_case_id','donor_component','donor_group')):raise ValueError('Donor assignment contract changed')
        expected=copy.deepcopy(self.meta['config'])
        if debug:expected.update(gnn_epochs=1,batch_calibration_repeats=1)
        if m['config']!=expected or m['base']!=self.meta['base']:raise ValueError('Configuration changed')
        allowed=set(m['split'][partition]);self.rows=[r for r in m['records'] if r['case_id'] in allowed];validate_rows(self.rows)
        if {r['id'] for r in self.rows}!={r['id'] for r in original.rows}:raise ValueError('Observation coverage changed')
        old={r['id']:r for r in original.rows}
        for r in self.rows:
            if any(r[k]!=old[r['id']][k] for k in ('case_id','patient_group','component','center','target')):raise ValueError('Observation identity changed')
        self.raw=object.__new__(OriginalFineDataset);self.raw.path=self.original_path;self.raw.root=self.original_path.parent
        self.raw.meta=m;self.raw.rows=self.rows;self.raw.store=None
        self.meta=copy.deepcopy(self.meta);self.meta.update(graph_representation=MODE,learning_policy=POLICY,
            original_cache=str(self.original_path),original_cache_sha256=sha(self.original_path))
        self.entries=[dict(row=r,fine_nodes=r['bounds']['nodes']) for r in self.rows]
    def __len__(self):return len(self.rows)

class DonorLoader(FineLoader):
    def get(self,ids):
        # Original file/hash reads remain canonical, but no old-donor materialized
        # batch or partition is reused. No CT/edge checks are bypassed.
        self.guard()
        identities=[]
        for i in ids:
            row=self.ds.rows[i];identities.append(self.store.path_key(row['path'],row['sha256'])[1])
            ref=row.get('shared_source')
            if ref:identities.append(self.store.path_key(ref['path'],ref['sha256'])[1])
        key=('same_donor_bound_batch',str(self.ds.original_path),tuple(ids),self.ds.meta['view_epoch'],
             tuple(identities))
        def build():return verified(load_pairs(self.ds.raw,ids,workers=self.workers,epoch=self.ds.meta['view_epoch'],cache_path=self.ds.original_path))
        batch=self.store.cache.get(key,build) if self.retain_batches else build()
        check_verified(batch);return batch
