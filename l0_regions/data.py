"""Compact cache records: fine sampling coordinates, assignments, coarse edges.

No fine edge list is stored or transferred by this module.
"""
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import torch
from torch_geometric.data import HeteroData
from hiercp_v22.schema import LOCAL_NODE_TYPES as NT, LOCAL_EDGE_TYPES as ET
from l0_ezsp.ops import compact, reduce_bounds, quotient
from tools.v22_artifacts import tree_hash


def unique_edge_count(edge,source_count,target_count):
    """Exact duplicate check using injective int64 keys after endpoint checks.

    unique(dim=0) sorts/compares two-element rows at substantial CPU cost.
    Preserve its result, including the general overflow-safe path. Never truncate
    endpoints or hash them into a collision-prone fixed-size table.
    """
    if edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2:raise ValueError('Invalid edge shape/type')
    if source_count<0 or target_count<0:raise ValueError('Invalid node counts')
    if edge.numel() and bool((edge<0).any()|(edge[0]>=source_count).any()|(edge[1]>=target_count).any()):
        raise ValueError('Edge endpoint outside range')
    if not edge.numel():return 0
    if source_count*target_count>torch.iinfo(torch.long).max:
        return len(torch.unique(edge.T,dim=0))
    return len(torch.unique(edge[0]*target_count+edge[1]))


class VerifiedItem(dict):
    """Loader-verified immutable-by-contract view, guarded by tensor versions."""
    def __init__(self,item):
        validate_item(dict(item))
        super().__init__(item)
        from .resident import signature
        self._receipt_signature=signature(self)


def scale_count(profile):
    n=profile.get('region_scales',2)
    if type(n)!=int or n not in (1,2):raise ValueError('Invalid region scale count')
    return n


def profile_status(item):
    limits=item['binding']['profile']['diagnostic_profile_from_original_NOT_VALIDATED']
    exceeded=False
    for level in range(scale_count(item['binding']['profile'])):
        rs=item['scales'][level];audit=item['audit'][f'scale{level+1}']
        exceeded |= sum(len(r['mass']) for r in rs.values())>limits['nodes_total'][level]
        exceeded |= sum(e.shape[1] for e in item['edges'][level].values())>limits['directed_edges_total'][level]
        for role,r in rs.items():
            bound=limits['bbox_context_diagonal_mm' if 'context' in role else 'bbox_surface_diagonal_mm'][level]
            exceeded |= bool(((r['upper']-r['lower']).norm(dim=1)>bound).any())
            cap=limits['caps_by_role'][role+'_per_shell' if 'context' in role else role][level]
            exceeded |= bool((torch.unique(r['shell'],return_counts=True)[1]>cap).any())
            groups=[g for g in audit['roles'][role]['group_diagnostics'] if g['owner']==item['preparation_pair_index']]
            if sum(g['clusters'] for g in groups)!=len(r['mass']):raise ValueError('Admission receipt cluster coverage')
            exceeded |= any(g['variance_excess']['clusters']>0 for g in groups)
    return bool(exceeded)


def seal_item(item):
    # Integrity receipt, not a cryptographic signature or authority to fabricate provenance.
    item['preparation_receipt_sha256']=tree_hash({k:v for k,v in item.items() if k!='preparation_receipt_sha256'})


def verify_receipt(item):
    expected=tree_hash({k:v for k,v in item.items() if k!='preparation_receipt_sha256'})
    if item.get('preparation_receipt_sha256')!=expected:raise ValueError('Preparation/admission receipt changed')
    b=item['binding'];r=item['materialization_receipt']
    for key in ('dataset_index','record_id','record_sha256','cache_sha256','shared_source','center','donor_case_id','donor_component'):
        if b[key]!=r[key]:raise ValueError('Record receipt mismatch')
    if b['view']['epoch']!=r['view_epoch'] or b['view']['index']!=r['view_index']:raise ValueError('View receipt mismatch')
    evidence=item['fine_graph_evidence']
    if not evidence['stage1_quotient_verified']:raise ValueError('Missing quotient evidence')
    if scale_count(b['profile'])==2:
        if not evidence['stage2_quotient_verified']:raise ValueError('Missing quotient evidence')
    elif evidence.get('stage2_status')!='REMOVED_BY_DESIGN':raise ValueError('Single-scale evidence required')
    if item['profile_exceeded']!=profile_status(item):raise ValueError('Incorrect profile flag')


def validate_item(item):
    if isinstance(item,VerifiedItem):
        from .resident import signature
        if item._receipt_signature!=signature(item):raise ValueError('Verified region item mutated')
        return
    verify_receipt(item)
    if item['diagnostic_only'] is not True or item['training_ready'] is not False:raise ValueError('Diagnostic cache only')
    levels=scale_count(item['binding']['profile'])
    if set(item['fine'])!=set(NT) or len(item['scales'])!=levels or len(item['edges'])!=levels:raise ValueError('Invalid region schema')
    for key in ('source_patch','target_patch'):
        t=item[key]
        if t.shape!=(1,48,48,48) or not bool(torch.isfinite(t).all()):raise ValueError('Invalid raw CT')
    for role,fine in item['fine'].items():
        if set(fine)!= {'grid','pos_mm','shell','stable_id','parent'}:raise ValueError('Unexpected fine fields/edges')
        n=len(fine['grid'])
        if fine['grid'].shape!=(n,3) or fine['pos_mm'].shape!=(n,3):raise ValueError('Coordinate shape')
        if not bool(torch.isfinite(fine['grid']).all() & torch.isfinite(fine['pos_mm']).all()) or bool((fine['grid'].abs()>1.001).any()):raise ValueError('Invalid coordinates')
        if role!='target_context' and n==0:raise ValueError('Required role missing')
        for key in ('shell','stable_id'):
            if fine[key].shape!=(n,) or fine[key].dtype!=torch.long:raise ValueError('Discrete identity shape/type')
        if len(torch.unique(fine['stable_id']))!=n or bool((fine['stable_id']<0).any()):raise ValueError('Canonical coverage')
        if 'context' in role:
            if bool(((fine['shell']<0)|(fine['shell']>2)).any()):raise ValueError('Shell identity')
        elif bool((fine['shell']!=-1).any()):raise ValueError('Surface shell identity')
        previous=fine; mass=torch.ones(n,device=fine['grid'].device)
        lower=upper=fine['pos_mm']
        for level in range(levels):
            r=item['scales'][level][role];p=previous['parent'];k=compact(p,len(mass))
            required={'mass','shell','lower','upper','stable_id'} | ({'parent'} if level+1<levels else set())
            if set(r)!=required:raise ValueError('Unexpected region fields or unused parent mapping')
            if len(r['mass'])!=k or r['mass'].shape!=(k,) or r['shell'].shape!=(k,) or r['shell'].dtype!=torch.long:raise ValueError('Coarse coverage')
            if not torch.equal(r['shell'][p],previous['shell']):raise ValueError('Mixed shell')
            expected=torch.zeros(k,device=mass.device).index_add(0,p,mass)
            if not torch.equal(expected,r['mass']) or not bool(torch.isfinite(r['mass']).all() & (r['mass']>0).all()):raise ValueError('Fine mass lost')
            for key,values,reduce in (('lower',lower,'amin'),('upper',upper,'amax')):
                if r[key].shape!=(k,3) or not torch.equal(r[key],reduce_bounds(values,p,k,reduce)):raise ValueError('Physical bounds lost')
            if r['stable_id'].shape!=(k,) or r['stable_id'].dtype!=torch.long or bool((r['stable_id']<0).any()):raise ValueError('Coarse identity')
            previous=r;mass=r['mass'];lower=r['lower'];upper=r['upper']
    for level,edges in enumerate(item['edges']):
        if set(edges)!=set(ET):raise ValueError('Typed relations lost')
        for rel,e in edges.items():
            if e.dtype!=torch.long or e.ndim!=2 or e.shape[0]!=2:raise ValueError('Invalid coarse edge')
            if e.numel() and bool((e<0).any()|(e[0]>=len(item['scales'][level][rel[0]]['mass'])).any()|(e[1]>=len(item['scales'][level][rel[2]]['mass'])).any()):raise ValueError('Edge outside region')
            if unique_edge_count(e,len(item['scales'][level][rel[0]]['mass']),len(item['scales'][level][rel[2]]['mass']))!=e.shape[1]:raise ValueError('Duplicate quotient edge')
            if rel[0]==rel[2] and bool((e[0]==e[1]).any()):raise ValueError('Collapsed internal edge')
    for rel in (ET if levels==2 else ()):
        expected=quotient(item['edges'][0][rel],item['scales'][0][rel[0]]['parent'],
            item['scales'][0][rel[2]]['parent'],same_type=rel[0]==rel[2])[0]
        if not torch.equal(expected,item['edges'][1][rel]):raise ValueError('Stage2 is not exact typed quotient')


def save_new(path,item):
    validate_item(item);path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    with path.open('xb') as stream:torch.save(dict(item),stream)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.with_suffix('.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(sha256=digest,binding=item['binding'],diagnostic_only=True),stream,indent=2)
    return path.stat().st_size


def load(path,expected_binding):
    path=Path(path);manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if manifest['binding']!=expected_binding:raise ValueError('Different record/view/geometry/partition model')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Corrupt region cache')
    item=torch.load(path,map_location='cpu',weights_only=True)
    if item['binding']!=expected_binding:raise ValueError('Cache identity mismatch')
    return VerifiedItem(item)


@dataclass
class RegionBatch:
    source_patches:torch.Tensor
    target_patches:torch.Tensor
    source_index:torch.Tensor
    indices:torch.Tensor
    fine:dict
    regions:list
    graphs:list
    bindings:list
    profile_exceeded:bool
    admission_flags:tuple

    def __len__(self):return len(self.target_patches)

    def to(self,device,non_blocking=True):
        from .resident import check_verified,mark_verified
        check_verified(self)
        move=lambda t:t.to(device,non_blocking=non_blocking)
        return mark_verified(RegionBatch(*(move(getattr(self,k)) for k in ('source_patches','target_patches','source_index','indices')),
            {k:{name:move(t) for name,t in v.items()} for k,v in self.fine.items()},
            [{k:SimpleNamespace(**{name:move(t) for name,t in vars(v).items()}) for k,v in layer.items()} for layer in self.regions],
            [copy.copy(g).to(device,non_blocking=non_blocking) for g in self.graphs],copy.deepcopy(self.bindings),self.profile_exceeded,self.admission_flags))


def collate(items,indices,*,compatible_sources=None):
    if len(items)!=len(indices) or len(set(indices))!=len(indices) or not items:raise ValueError('Exact batch coverage')
    for item in items:validate_item(item)
    if len({it['binding']['record_id'] for it in items})!=len(items):raise ValueError('Duplicate actual record')
    if any(it['binding']['dataset_index']!=i for it,i in zip(items,indices)):raise ValueError('Wrong dataset index for record')
    # Mixed partition models/views/profiles must not silently form one experiment.
    for key in ('cache_sha256','frozen_cnn_sha256','profile','view','feature_coordinates','feature_evidence'):
        if any(item['binding'][key]!=items[0]['binding'][key] for item in items):raise ValueError('Mixed partition contract: '+key)
    origins={item['binding']['preparation_source_sha256'] for item in items}
    if compatible_sources is not None:
        if not origins<=compatible_sources:raise ValueError('Unreviewed batch preparation source')
    elif len(origins)>1:
        raise ValueError('Mixed partition contract: unverified preparation sources')
    levels=scale_count(items[0]['binding']['profile'])
    offsets=[{k:[0] for k in NT} for _ in range(levels)]
    for item in items:
        for level in range(levels):
            for k in NT:offsets[level][k].append(offsets[level][k][-1]+len(item['scales'][level][k]['mass']))
    fine={};regions=[{} for _ in range(levels)];graphs=[HeteroData() for _ in range(levels)]
    for k in NT:
        fine[k]={name:torch.cat([item['fine'][k][name] for item in items]) for name in ('grid','shell','stable_id','pos_mm')}
        fine[k]['owner']=torch.cat([torch.full((len(item['fine'][k]['grid']),),i,dtype=torch.long) for i,item in enumerate(items)])
        fine[k]['parent']=torch.cat([item['fine'][k]['parent']+offsets[0][k][i] for i,item in enumerate(items)])
        for level in range(levels):
            fields={name:torch.cat([item['scales'][level][k][name] for item in items]) for name in ('shell','mass','lower','upper','stable_id')}
            fields['owner']=torch.cat([torch.full((len(item['scales'][level][k]['mass']),),i,dtype=torch.long) for i,item in enumerate(items)])
            if level+1<levels:fields['parent']=torch.cat([item['scales'][0][k]['parent']+offsets[1][k][i] for i,item in enumerate(items)])
            regions[level][k]=SimpleNamespace(**fields)
            graphs[level][k].num_nodes=len(fields['mass'])
    for level in range(levels):
        for rel in ET:
            graphs[level][rel].edge_index=torch.cat([item['edges'][level][rel]+torch.tensor([[offsets[level][rel[0]][i]],[offsets[level][rel[2]][i]]]) for i,item in enumerate(items)],1)
    sources=[];source_lookup={};source_indices=[]
    for item in items:
        source=item['source_patch']
        key=hashlib.sha256(source.contiguous().numpy().tobytes()).hexdigest()
        if key not in source_lookup:source_lookup[key]=len(sources);sources.append(source)
        source_indices.append(source_lookup[key])
    from .resident import mark_verified
    return mark_verified(RegionBatch(torch.stack(sources),torch.stack([it['target_patch'] for it in items]),
        torch.tensor(source_indices),torch.tensor(indices),fine,regions,graphs,[copy.deepcopy(it['binding']) for it in items],
        any(it['profile_exceeded'] for it in items),tuple(it['profile_exceeded'] for it in items)))
