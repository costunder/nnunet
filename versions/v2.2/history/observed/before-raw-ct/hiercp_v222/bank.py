"""New v2 scores + the unchanged raw-target/native CP transport contract."""
from pathlib import Path
import numpy as np
import torch
from hiercp.common import discover_cases,load_case,organ_depth_mm,build_candidate_pool,stable_case_seed
from hiercp.schema import graph_config_from_dict
from hiercp.preparation_runtime import run_case_jobs
from tools.online_raw_bank_preparation import prepare_raw_case,prepare_source_candidates
from . import PIPELINE_VERSION
from .contracts import read_json,write_new,safe_new_root,sha,validate_native
from .data import sources
from .scoring import Scorer

def map_center(point,plans,properties,shape):
    xyz=np.asarray(point,dtype=np.float64)[::-1][plans['transpose_forward']]
    crop=xyz-np.asarray(properties['bbox_used_for_cropping'])[:,0]
    return np.rint((crop+.5)*np.asarray(shape)/np.asarray(properties['shape_after_cropping_and_before_resampling'])-.5).astype(np.int32)

def build(native_path,checkpoint,output):
    from .contracts import validate_checkpoint,source_identity
    from hiercp_v22.donors import POLICY,validate_pool,reject_cross_split_duplicates
    native=validate_native(read_json(native_path))
    payload=validate_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=False))
    if native['split']!=payload['split']:raise ValueError('Bank/native/GNN patient split differs')
    if 'identities' in native and native['identities']!=payload['identities']:
        raise ValueError('Native/GNN patient group provenance differs')
    if set(native['planning_patient_ids'])!=set(native['split']['outer_train']):
        raise ValueError('Native normalization/planning must be fitted on outer_train only')
    raw={r['case_id']:r for r in native['raw_records']}
    reject_cross_split_duplicates(native['raw_records'],native['split'])
    for case in native['split']['outer_train']:
        for key in ('image','label'):
            if sha(raw[case][key])!=raw[case][key+'_sha256']:raise ValueError('Original training CT/GT changed')
    for row in payload['raw_records']:
        for key in ('image','label'):
            if row[key+'_sha256']!=raw[row['case_id']][key+'_sha256']:raise ValueError('GNN/native raw content differs')
    pool=payload['donor_pool']; validate_pool(pool,native['split'])
    root=safe_new_root(output)
    entries={c:[entry_name(c,i) for i in range(len(pool))] for c in native['split']['outer_train']}
    inventory={c:list(range(1,len(pool)+1)) for c in entries}
    slots={c:[{'source_component':i+1,'status':'ok','entry':name} for i,name in enumerate(names)] for c,names in entries.items()}
    metadata={'format':'hiercp_online_bank_v2','pipeline_version':PIPELINE_VERSION,'complete':True,
        'completion_scope':'validated donor/recipient catalog; candidate entries materialized only when a CP event selects them',
        'materialization':'on_demand_complete_128_pool_v1','donor_policy':POLICY,'donor_pool':pool,
        'paste_contract':'onlinecp_raw_target_paste_v1','entry_storage':'v21_scores128_selected_raw_target_v1',
        'source_mapping_policy':'online_cp_raw_target_resampling_v2','candidate_count':128,'hier_top_k':1,
        'tumor_label':2,'liver_label':1,'cp_probability':payload['config']['cp_probability'],'intensity_scale_range':[.95,1.05],
        'intensity_shift_range_hu':[-5.,5.],'entries_by_case':entries,'entry_sha256':{},
        'no_placement_policy':'retain_original','source_schedule_format':'onlinecp_all_source_slots_v1',
        'source_slots_by_case':slots,'eligible_sources_by_case':inventory,'eligible_source_slots':sum(map(len,slots.values())),
        'no_placement_sources':0,'split':native['split'],'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':sha(checkpoint),
        'native_preparation':str(Path(native_path).resolve()),'native_sha256':sha(native_path),
        'selection':payload['config']['selection'],'label_semantics':payload['config']['label_definition'],
        'alignment_semantics':payload['config']['alignment_target'],'source_identity':source_identity(),
        'identities':payload['identities'],'input_contract':payload['input_contract']}
    validate_catalog(metadata)
    write_new(root/'index.json',metadata)
    return root/'index.json'


def entry_name(recipient,index):return f'entries/{recipient}__donor_{index+1:06d}.npz'


def validate_catalog(meta):
    from hiercp_v22.donors import POLICY,validate_pool
    from .contracts import validate_identities
    if meta.get('pipeline_version')!=PIPELINE_VERSION or meta.get('selection')!='aligned_observation_context_argmax':
        raise ValueError('Only v2.22 observation-context scores may enter this CP bank')
    validate_identities(meta['identities'],meta['split'])
    if meta.get('donor_policy')!=POLICY or meta.get('materialization')!='on_demand_complete_128_pool_v1':
        raise ValueError('Shared per-event donor catalog required; old within-patient bank rejected')
    if meta.get('entry_storage')!='v21_scores128_selected_raw_target_v1':
        raise ValueError('Full 128-score / selected-raw-paste entry format required')
    validate_pool(meta['donor_pool'],meta['split'])
    recipients=set(meta['split']['outer_train']); count=len(meta['donor_pool'])
    if set(meta['entries_by_case'])!=recipients or set(meta['source_slots_by_case'])!=recipients or set(meta['eligible_sources_by_case'])!=recipients:
        raise ValueError('Donor catalog must cover all and only outer-training recipients')
    for case in recipients:
        expected=[entry_name(case,i) for i in range(count)]
        if meta['entries_by_case'][case]!=expected or meta['eligible_sources_by_case'][case]!=list(range(1,count+1)):
            raise ValueError('Every recipient must have the same complete donor pool')
        if meta['source_slots_by_case'][case]!=[dict(source_component=i+1,status='ok',entry=name) for i,name in enumerate(expected)]:
            raise ValueError('Donor slot availability/order changed')
    if meta['candidate_count']!=128 or meta['cp_probability']!=.8:raise ValueError('CP80 comparison scale changed')
    return meta


class EntryBuilder:
    """Produce a selected donor/recipient pair once; never enumerate all pairs."""
    def __init__(self,index_path,gpu_lock):
        import threading
        from .contracts import source_identity
        from hiercp_v22.volumes import VolumeCache
        from custom_trainers.onlinecp_raw_bank import RawBankStore
        self.path=Path(index_path).resolve(); self.root=self.path.parent
        self.meta=validate_catalog(read_json(self.path))
        if self.meta['source_identity']!=source_identity():raise ValueError('Catalog source identity changed')
        if sha(self.meta['checkpoint'])!=self.meta['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if sha(self.meta['native_preparation'])!=self.meta['native_sha256']:raise ValueError('Native preparation changed')
        self.native=validate_native(read_json(self.meta['native_preparation']))
        with gpu_lock:self.scorer=Scorer(self.meta['checkpoint'])
        if (self.meta['split']!=self.native['split'] or self.meta['split']!=self.scorer.payload['split']
                or self.meta['donor_pool']!=self.scorer.payload['donor_pool']):
            raise ValueError('Catalog donor/split differs from frozen training provenance')
        self.cfg,self.base=self.scorer.cfg,self.scorer.base; self.gpu_lock=gpu_lock
        all_paths={p.case_id:p for p in discover_cases(Path(self.native['medical'])/'Data')}
        self.paths={c:all_paths[c] for c in self.meta['split']['outer_train']}
        raw={r['case_id']:r for r in self.native['raw_records']}
        for c,p in self.paths.items():
            if sha(p.image_path)!=raw[c]['image_sha256'] or sha(p.label_path)!=raw[c]['label_sha256']:
                raise ValueError('Original CT/GT changed after native preparation')
        self.volumes=VolumeCache(self.paths,self.base,self.cfg)
        self.plans=read_json(self.native['plans']); self.plan=self.plans['configurations']['3d_fullres']
        self.locks={c:threading.RLock() for c in self.paths}; self.entry_locks={}; self.lock=threading.Lock()
        self.store=RawBankStore(self.root); self.raw_ready={}; self.native_lock=threading.Lock()

    def _native_case(self,case_id,raw):
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from custom_trainers.onlinecp_raw_bank import save_case
        with self.native_lock,self.locks[case_id]:
            if case_id not in self.raw_ready:
                receipt=self.root/f'raw_receipts/{case_id}.json'
                if receipt.exists():
                    saved=read_json(receipt)
                    if saved['native_sha256']!=self.meta['native_sha256']:raise ValueError('Stale raw case receipt')
                    ref,digest=saved['reference'],saved['sha256']
                    prep_ref,prep_sha=saved['preparation_reference'],saved['preparation_sha256']
                else:
                    directory=Path(self.native['preprocessed'])/self.plan['data_identifier']
                    ds=infer_dataset_class(str(directory))(str(directory),[case_id])
                    pre,seg,_,props=ds.load_case(case_id)
                    native_case,ref,digest=prepare_raw_case(self.root,case_id,raw.image,raw.label,props,self.plans,pre,seg,
                        configuration_name='3d_fullres',raw_spacing=raw.spacing,raw_spatial_unit=raw.image_header.get_xyzt_units()[0],
                        minimum_free_bytes=int(self.cfg['minimum_free_gb']*1024**3))
                    # Runtime case storage intentionally excludes preparation
                    # volumes. Keep that full subtree separately, once per
                    # recipient, so later foreign donors use the verified grid.
                    prep_ref=f'raw_preparation/{case_id}.json'
                    prep_sha=save_case(self.root,prep_ref,native_case['preparation'])
                    del native_case,pre,seg
                    write_new(receipt,dict(reference=ref,sha256=digest,native_sha256=self.meta['native_sha256'],
                        preparation_reference=prep_ref,preparation_sha256=prep_sha,
                        properties={key:np.asarray(props[key]).tolist() for key in ('bbox_used_for_cropping','shape_after_cropping_and_before_resampling')}))
                self.raw_ready[case_id]=(ref,digest,prep_ref,prep_sha,read_json(receipt)['properties'])
            ref,digest,prep_ref,prep_sha,props=self.raw_ready[case_id]
            # This reader is accessed only under native_lock; array references
            # remain valid in callers when an LRU mapping entry is evicted.
            case=dict(self.store.load_case(ref,digest))
            case['preparation']=self.store.load_case(prep_ref,prep_sha)
            return case,ref,digest,props

    def materialize(self,recipient,donor_index):
        import threading
        from dataclasses import replace
        from hiercp_v22.donors import select_donor
        from .data import donor_in_target_spacing,candidate_pool
        from hiercp_v22.parallel import run_jobs
        count=len(self.meta['donor_pool'])
        if type(donor_index) is not int or not 0<=donor_index<count:raise ValueError('Invalid donor index')
        select_donor(self.meta['donor_pool'],self.meta['split'],recipient,(donor_index+.5)/count)
        name=entry_name(recipient,donor_index); receipt=self.root/(name+'.receipt.json')
        with self.lock:lock=self.entry_locks.setdefault(name,threading.Lock())
        with lock:
            if receipt.exists():
                ready=read_json(receipt)
                if ready['catalog_sha256']!=sha(self.path) or sha(self.root/name)!=ready['entry_sha256']:
                    raise ValueError('Materialized CP entry provenance changed')
                return {'relative':name,'sha256':ready['entry_sha256']}
            donor_row=self.meta['donor_pool'][donor_index]; donor_id=donor_row['case_id']
            with self.volumes.pair(recipient,donor_id) as (target,donor):
                raw,dc=target['case'],donor['case']
                collection=donor['sources']; pos=[c for c,_ in collection.entries].index(donor_row['component_id'])
                original,diameter=collection[pos]
                source,_=donor_in_target_spacing(original,dc.spacing,raw.spacing)
                pool,audit=candidate_pool(raw,source,self.cfg,self.base,target['depth'],donor_case_id=donor_id)
                centers=np.asarray([p.center for p in pool],dtype=np.int32)
                records={}
                def graph(item):
                    i,center=item
                    from .inputs import context_patch
                    record={'patch':context_patch(raw.image,raw.spacing,center,self.scorer.payload['input_contract'])}
                    return i,record
                run_jobs(list(enumerate(centers)),graph,lambda row:records.__setitem__(*row),self.cfg['preparation_workers'],
                         self.root/f'resources/{recipient}__{donor_index+1:06d}.json')
                with self.gpu_lock:
                    scores,resources=self.scorer.score_records([records[i] for i in range(128)],recipient)
                del records
                selected=int(np.argmax(scores))
                native_case,ref,digest,props=self._native_case(recipient,raw)
                mapped=np.stack([map_center(c,self.plans,props,native_case['metadata']['preprocessed_shape']) for c in centers])
                anchor=np.asarray(source.anchor_center)-np.asarray([s.start for s in source.patch_slices])
                refs,digests,transport=prepare_source_candidates(self.root,recipient,donor_index+1,native_case,digest,
                    source.patch_image,source.patch_mask,anchor,centers[selected:selected+1],self.plan['patch_size'])
                file=self.root/name; file.parent.mkdir(parents=True,exist_ok=True)
                with file.open('xb') as stream:
                    np.savez(stream,paste_contract=np.asarray(['onlinecp_raw_target_paste_v1']),case_id=np.asarray([recipient]),
                        donor_case_id=np.asarray([donor_id]),donor_component_id=np.asarray([donor_row['component_id']]),
                        selected_candidate=np.asarray([selected]),selected_payload=refs,selected_payload_sha256=digests,raw_case_reference=np.asarray([ref]),
                        raw_case_reference_sha256=np.asarray([digest]),candidate_centers=mapped,candidate_raw_centers=centers,
                        scores=scores,source_component=np.asarray([donor_index+1]),source_diameter_mm=np.asarray([diameter]))
                write_new(receipt,dict(catalog_sha256=sha(self.path),recipient=recipient,donor=donor_row,entry_sha256=sha(file),
                    candidate_audit=audit,scoring_resources=resources,transport_audit=transport,
                    scored_candidates=128,raw_materialized_candidates=1,selection='exact_argmax'))
                return {'relative':name,'sha256':sha(file)}
