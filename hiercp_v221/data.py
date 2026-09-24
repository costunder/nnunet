"""Full-cohort, source-resolved local graph caches; no v1 upper graphs."""
from __future__ import annotations
from dataclasses import dataclass,replace
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from scipy import ndimage as ndi
from hiercp.common import (discover_cases,load_case,SourceTumor,bbox_of_mask,organ_depth_mm,stable_case_seed,build_candidate_pool)
from hiercp.curriculum import CandidateSpec
from .geometry import prepare_local_source,build_local_graph
from .sample import build_local_view
from hiercp.schema import graph_config_from_dict
from hiercp.preparation_runtime import run_case_jobs
from .contracts import read_json,write_new,sha,safe_new_root,validate_split
from . import PIPELINE_VERSION
from .storage import GraphWriter,load_record

class SourceCollection:
    """Repeatable source sequence; retain one component map, not N full masks."""
    def __init__(self,case,pad,maximum):
        self.case,self.pad=case,pad
        self.components,count=ndi.label(case.label==2,structure=ndi.generate_binary_structure(3,1))
        self.sizes=np.bincount(self.components.ravel(),minlength=count+1)
        self.entries=[]
        for component in range(1,count+1):
            diameter=2*(3*int(self.sizes[component])*float(np.prod(case.spacing))/(4*math.pi))**(1/3)
            if diameter<=maximum:self.entries.append((component,diameter))
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        component,diameter=self.entries[index]; case=self.case
        mask=self.components==component; box=bbox_of_mask(mask,pad=self.pad)
        # extract_centered / Copy-Paste use shape//2, including even-size boxes.
        # The lower midpoint displaced even boxes by one voxel in v2.1/r1.
        center=tuple(s.start+(s.stop-s.start)//2 for s in box)
        return SourceTumor(component,mask,mask[box].copy(),case.image[box].copy(),box,center,
                        tuple(np.argwhere(mask).mean(0)),int(self.sizes[component])),diameter
    def __iter__(self):
        for index in range(len(self)):yield self[index]

def sources(case,pad,maximum):return SourceCollection(case,pad,maximum)

def candidate_spec(source,center,coverage=1.):
    # These legacy transport fields never enter an upper graph in v2.
    return CandidateSpec(tuple(map(int,center)),0,0,-1,-1,float(coverage),0.,0.,0.,0.)

def local_record(case,source,center,base,*,prepared=None,depth=None,organ=None):
    gc=graph_config_from_dict(base['graph']); organ=np.isin(case.label,[1,2]) if organ is None else organ
    depth=organ_depth_mm(organ,case.spacing) if depth is None else depth
    rng=np.random.default_rng(stable_case_seed(base['seed'],case.paths.case_id,str(source.component_id)))
    prepared=prepared or prepare_local_source(case,source,full_organ_mask=organ,organ_depth=depth,config=gc,rng=rng,ct_clip=tuple(base['ct_clip']))
    built=build_local_graph(case,source,candidate_spec(source,center),full_organ_mask=organ,organ_depth=depth,
                            config=gc,rng=rng,ct_clip=tuple(base['ct_clip']),prepared_source=prepared)
    return {'format':PIPELINE_VERSION,'case_id':case.paths.case_id,'component_id':source.component_id,
            'seed':base['seed'],
            'source_patch':torch.from_numpy(built.source_patch.astype(np.float16)),
            'target_patch':torch.from_numpy(built.target_patch.astype(np.float16)),
            'source_local':built.source_local,'target_local':built.target_local,'graph_config':gc.to_dict()}

def candidate_pool(case,source,cfg,base,depth,*,donor_case_id=None):
    settings=base['generation']; organ=np.isin(case.label,[1,2]); diagnostics={}
    pool,_=build_candidate_pool(case,source,placement_mask=case.label==1,full_organ_mask=organ,
        occupied_mask=case.label==2,organ_distance=depth,
        rng=np.random.default_rng(stable_case_seed(cfg['seed'],case.paths.case_id,f'{donor_case_id or case.paths.case_id}:{source.component_id}')),
        num_candidates=cfg['candidate_count'],max_draws=settings['max_draws'],
        min_liver_coverage=settings['min_liver_coverage'],occupied_clearance_vox=settings['occupied_clearance_vox'],
        min_center_separation_mm=settings['min_center_separation_mm'],
        min_center_separation_vox=settings['min_center_separation_vox'],diagnostics=diagnostics)
    if len(pool)!=cfg['candidate_count']: raise ValueError('Complete candidate pool required; no source silently dropped')
    return pool,diagnostics

def donor_in_target_spacing(source,donor_spacing,target_spacing):
    """Real donor regridded about its exact anchor for context-only foreign queries.

    No patient GT is changed. Source node physical coordinates remain from the
    donor; only the virtual target footprint uses target-native spacing.
    """
    anchor=np.asarray(source.anchor_center)-np.asarray([s.start for s in source.patch_slices])
    shape=np.asarray(source.patch_mask.shape)
    before=np.ceil(anchor*donor_spacing/target_spacing).astype(int)
    after=np.ceil((shape-1-anchor)*donor_spacing/target_spacing).astype(int)
    new_shape=before+after+1
    axes=np.indices(tuple(new_shape),dtype=np.float64)
    coordinates=(axes-before[:,None,None,None])*target_spacing[:,None,None,None]/donor_spacing[:,None,None,None]+anchor[:,None,None,None]
    mask=ndi.map_coordinates(source.patch_mask.astype(np.uint8),coordinates,order=0,mode='constant',cval=0)>0
    image=ndi.map_coordinates(source.patch_image,coordinates,order=1,mode='nearest').astype(np.float32)
    if not mask.any(): raise ValueError('Real donor disappears at target spacing; no fabricated footprint')
    # build_candidate_pool only consumes the real resampled patch fields.
    return replace(source,patch_mask=mask,patch_image=image,voxel_count=int(mask.sum()),
                   anchor_center=tuple(map(int,before)),patch_slices=tuple(slice(0,int(s)) for s in new_shape)),mask

def prepare(medical,split_path,output,cfg,base):
    from .donors import donor_pool,context_round,validate_training_rows,reject_cross_split_duplicates,CONTEXT
    from .parallel import run_jobs
    split=read_json(split_path); validate_split(split)
    paths={c.case_id:c for c in discover_cases(Path(medical)/'Data')}
    if set(split['outer_train']+split['outer_val'])!=set(paths):raise ValueError('Split must match complete cohort')
    root=safe_new_root(output); storage=GraphWriter(root); records=[]; cases=[]; inventory={}
    write_new(root/'split.json',split); write_new(root/'config.json',cfg); write_new(root/'base_config.json',base)
    # Hash audit includes held-out file identities, never held-out CT/GT tensors.
    raw_records=[{'case_id':c,'image':str(paths[c].image_path.resolve()),'image_sha256':sha(paths[c].image_path),
                  'label':str(paths[c].label_path.resolve()),'label_sha256':sha(paths[c].label_path)}
                 for c in split['outer_train']+split['outer_val']]
    reject_cross_split_duplicates(raw_records,split)
    def scan(case_id):
        case=load_case(paths[case_id]); collection=sources(case,base['cache']['source_pad'],cfg['donor_max_diameter_mm'])
        return case_id,[component for component,_ in collection.entries]
    from .volumes import volume_memory_bound
    scan_bound=max(volume_memory_bound(paths[c].image_path) for c in split['outer_train'])
    run_jobs(split['outer_train'],scan,lambda item:inventory.__setitem__(*item),cfg['preparation_workers'],root/'inventory_resources.json',memory_per_job=scan_bound)
    pool=donor_pool(inventory,split)
    train_events=context_round(pool,split['inner_train'],cfg['seed'])
    val_events=context_round(pool,split['inner_val'],cfg['seed']+1)
    schedule=train_events+val_events
    # Keep EVERY observed small-tumor anchor. Uniform query policy is independent
    # of recipient tumor count; each donor and each patient is covered per round.
    jobs=[dict(kind='T',recipient=c,donor_case=c,component=component,event=f'T_{c}_{component}')
          for c in split['outer_train'] for component in inventory[c]]
    jobs += [dict(kind='U',recipient=e['recipient'],donor_case=pool[e['donor_index']]['case_id'],
                  component=pool[e['donor_index']]['component_id'],event=f'U_{i:06d}') for i,e in enumerate(schedule)]
    write_new(root/'source_inventory.json',{'sources':inventory,'donor_pool':pool,'schedule':schedule,'schedule_policy':CONTEXT,
              'total_graphs':sum(1 if j['kind']=='T' else cfg['candidate_count'] for j in jobs)})
    # Volumes/depth are reused with a bounded byte-accounted cache; eviction never drops data.
    from .volumes import VolumeCache
    volumes=VolumeCache({c:paths[c] for c in split['outer_train']},base,cfg)
    def one(job):
        with volumes.pair(job['recipient'],job['donor_case']) as (target,donor):
            case,organ,depth=target['case'],target['organ'],target['depth']
            dc=donor['case']; source_index=[c for c,_ in donor['sources'].entries].index(job['component'])
            original,diameter=donor['sources'][source_index]
            prepared=prepare_local_source(dc,original,full_organ_mask=donor['organ'],organ_depth=donor['depth'],
                config=graph_config_from_dict(base['graph']),rng=np.random.default_rng(cfg['seed']),ct_clip=tuple(base['ct_clip']))
            source=original
            if job['recipient']!=job['donor_case']:
                from .geometry import exact_source_footprint
                source,_=donor_in_target_spacing(original,dc.spacing,case.spacing)
                prepared=replace(prepared,source_footprint=exact_source_footprint(source))
            if job['kind']=='T':points=[original.anchor_center]; audit={'observed_original_tumor':True}
            else:
                candidates_,audit=candidate_pool(case,source,cfg,base,depth,donor_case_id=job['donor_case']); points=[p.center for p in candidates_]
            rows=[]
            for i,center in enumerate(points):
                record=local_record(case,source,center,base,prepared=prepared,depth=depth,organ=organ)
                record.update(donor_case_id=job['donor_case'],evidence=1 if job['kind']=='T' else -1,center=list(center))
                record['source_local']['footprint_voxels']=int(original.voxel_count)
                record['target_virtual_footprint_voxels']=int(prepared.source_footprint.sum())
                record['donor_spacing']=dc.spacing.tolist(); record['target_spacing']=case.spacing.tolist()
                saved=storage.write(f"graphs/{job['event']}/{i:04d}.pt.gz",record,(job['recipient'],job['donor_case'],job['component']))
                rows.append(dict(id=f"{job['event']}:{i}",case_id=job['recipient'],donor_case_id=job['donor_case'],
                                 component_id=job['component'],evidence=record['evidence'],diameter_mm=diameter,center=list(map(int,center)),**saved))
            return rows,dict(job,graphs=len(rows),candidate_audit=audit)
    audits=[]
    def commit(result):
        rows,audit=result; records.extend(rows); audits.append(audit)
        print({'stage':'context_event_complete','event':audit['event'],'graphs':len(rows),'completed_events':len(audits),'total_events':len(jobs)},flush=True)
    run_jobs(jobs,one,commit,cfg['preparation_workers'],root/'resources.json')
    validate_training_rows(records,split,pool)
    write_new(root/'index.json',{'format':PIPELINE_VERSION,'complete':True,'split':split,'donor_pool':pool,
              'records':sorted(records,key=lambda r:r['id']),'cases':[r for r in raw_records if r['case_id'] in split['outer_train']],
              'all_raw_identities':raw_records,'schedule_policy':CONTEXT,'event_audits':audits,
              'config':cfg,'base_config':base,'evidence_contract':cfg['evidence_contract'],
              'geometry_filter_policy':'invalid candidates excluded separately; never creates a false data-label relation',
              'relation_annotation_status':'not_defined_by_tumor_mask'})
    return root/'index.json'

class LocalDataset(Dataset):
    def __init__(self,root,rows,*,view=0,epoch=0,verify=True):
        self.root=Path(root); self.rows=list(rows); self.view=view; self.epoch=epoch
        if verify:
            for row in rows:
                if sha(self.root/row['path'])!=row['sha256']: raise ValueError('Local cache SHA mismatch')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        record=load_record(self.root,row['path'])
        return materialize(record,view=self.view,epoch=self.epoch),index

class MemoryLocalDataset(Dataset):
    """All 128 real candidate graphs in RAM; no discarded candidates/files."""
    def __init__(self,records,case_id):
        from .resources import _local_bound
        self.records=list(records);self.root=Path('.');self.rows=[]
        for i,r in enumerate(self.records):
            a,b=_local_bound(r['source_local']),_local_bound(r['target_local'])
            self.rows.append(dict(id=f'{case_id}:{i}',case_id=case_id,bounds=dict(nodes=a[0]+b[0],edges=a[1]+b[1],
                bytes=a[2]+b[2]+r['source_patch'].numel()*4+r['target_patch'].numel()*4)))
    def __len__(self):return len(self.records)
    def __getitem__(self,index):return materialize(self.records[index]),index

def materialize(record,view=0,epoch=0):
    if record['format']!=PIPELINE_VERSION: raise ValueError('Wrong local cache version')
    seed=stable_case_seed(record['seed'],record['case_id'],f"{record['component_id']}:{view}:{epoch}")
    graph=build_local_view(record['source_local'],record['target_local'],graph_config_from_dict(record['graph_config']),seed=seed)
    # Canonical coordinates remain CPU geometry metadata. Only sampling grids
    # and structural shell/type membership reach the learned encoder.
    for kind in graph.node_types:
        del graph[kind].pos_mm
    patches=tuple(record[name] for name in ('source_patch','target_patch'))
    for patch in patches:
        if patch.ndim!=4 or patch.shape[0]!=1 or tuple(patch.shape[1:])!=(record['graph_config']['patch_size'],)*3:
            raise ValueError('Rebuild CNN-only cache from raw CT: wrong patch shape')
        if not torch.isfinite(patch).all():raise ValueError('Nonfinite CT patch')
    return graph,*patches

@dataclass
class LocalBatch:
    graph: Batch
    source_patches: torch.Tensor
    target_patches: torch.Tensor
    source_index: torch.Tensor
    indices: torch.Tensor
    def to(self,device):
        self.graph=self.graph.to(device,non_blocking=True)
        for key in ('source_patches','target_patches','source_index','indices'):
            setattr(self,key,getattr(self,key).to(device,non_blocking=True))
        return self
    def pin_memory(self):
        self.graph=self.graph.pin_memory()
        for key in ('source_patches','target_patches','source_index','indices'):
            setattr(self,key,getattr(self,key).pin_memory())
        return self

def collate(items):
    payload,indices=zip(*items)
    graphs,source,target=zip(*payload)
    return LocalBatch(Batch.from_data_list(list(graphs)),torch.stack(source),torch.stack(target),
                      torch.arange(len(items)),torch.tensor(indices))
