"""Offline-only official partition; never imported by the training encoder."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from hiercp_v22.schema import LOCAL_NODE_TYPES as NT, LOCAL_EDGE_TYPES as ET, SOURCE_LOCAL_NODE_TYPES
from hiercp_v222.deterministic_sampling import sample_nodes
from tools.v222_review_contracts import cnn_lattice, input_grid_to_feature_grid
from tools.v22_artifacts import tree_hash
from l0_ezsp.validation import validate_batch
from l0_ezsp.config import validate,load_profile
from l0_ezsp.encoder import EZSPEncoder
from l0_ezsp.partition import Region, CoarseningConstraintError
from l0_ezsp.ops import undirected, quotient, mass_mean
from .materialization import verify_materialization

PREPARATION_FILES=('l0_regions/preparation.py','l0_regions/admission_report.py','l0_regions/profile_policy.py','l0_regions/data.py','l0_regions/materialization.py',
    'l0_ezsp/partition.py','l0_ezsp/ops.py','l0_ezsp/data.py','l0_ezsp/backend.py',
    'l0_ezsp/encoder.py','tools/v222_review_contracts.py','tools/v22_artifacts.py',
    'l0_ezsp/config.py','l0_ezsp/validation.py','config/l0_ezsp_unresolved.json')


def preparation_fingerprint(root, core_source):
    source=dict(core_source)
    for name in PREPARATION_FILES:source[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()


def fixed_profile(original):
    result=validate(original)
    return _fixed_profile(result)


def _fixed_profile(result):
    result['document_type']='fixed_partition_diagnostic_profile_not_training_config'
    result['encoder_id']='fixed_region_sage_2_1_debug_v1'
    result['blockers']=['partition quality from the chosen checkpoint is unverified',
        'both explicit reg values and admission bounds remain uncalibrated',
        'production full-cohort/MIG cost and CP utility unverified']
    result['partition']['features']['scale2']='frozen_CNN32_mass_mean'
    result['graph_encoder']={'sage_layers':[2,1],'hidden':128,'readout':'mass-aware attention; equal two-scale mix','output_dim':128}
    return result


def single_profile(reg_scale1,sharding=None):
    import math
    from l0_ezsp.config import CONTRACT_PATH
    if isinstance(reg_scale1,bool) or not isinstance(reg_scale1,(int,float)) or not math.isfinite(reg_scale1) or reg_scale1<0:
        raise ValueError('Explicit finite nonnegative scale1 reg required')
    if sharding is not None and (type(sharding)!=int or sharding<=1):raise ValueError('Invalid sharding')
    result=_fixed_profile(json.loads(CONTRACT_PATH.read_text(encoding='utf-8')))
    result['partition'].update(reg_scale1=float(reg_scale1),reg_scale2=None)
    result['partition']['features']['scale2']='REMOVED'
    result['partition']['reg_policy']='Only reg_scale1 is used; scale2 removed'
    result['backend']['options']['sharding']=sharding
    result['encoder_id']='fixed_region_sage_single_scale_v1'
    result['region_scales']=1
    result['blockers']=['partition quality from the chosen checkpoint is unverified',
        'explicit scale1 reg and admission bounds remain uncalibrated',
        'production full-cohort cost and CP utility unverified']
    result['graph_encoder'].update(sage_layers=[3],readout='single-scale mass-aware attention')
    return result


def binding(*, record, dataset_index, cache_sha256, frozen_cnn_sha256, profile, view_epoch, view_index, feature_evidence):
    if type(dataset_index)!=int or dataset_index<0:raise ValueError('Dataset index required')
    if type(view_epoch)!=int or type(view_index)!=int or min(view_epoch,view_index)<0:
        raise ValueError('Explicit fixed nonnegative view epoch/index required')
    if feature_evidence not in ('untrained_plumbing_only','checkpoint_partition_quality_unverified'):
        raise ValueError('No implicit semantic feature approval')
    from hiercp_v222.v1_cache import provenance
    root=Path(__file__).resolve().parents[1]
    source_sha=preparation_fingerprint(root,provenance())
    return dict(format='fixed_region_cache_receipt_v2', dataset_index=dataset_index, cache_sha256=cache_sha256,preparation_source_sha256=source_sha,
        record_id=record['id'], record_sha256=record['sha256'],
        shared_source=record['shared_source'], center=record['center'],
        donor_case_id=record['donor_case_id'], donor_component=record['donor_component'],
        frozen_cnn_sha256=frozen_cnn_sha256, profile=profile,
        view=dict(epoch=view_epoch,index=view_index,policy='fixed materialized view; never overlaid onto epoch-resampled fine nodes'),
        feature_coordinates='stride4', feature_evidence=feature_evidence,
        scale2_features=('REMOVED' if profile.get('region_scales')==1 else 'original-mass mean of frozen CNN32; not current SAGE128'),
        training_ready=False)


class OfflinePartition:
    # Reuse the preserved official partition/quotient/hard integrity logic only.
    _coarsen=EZSPEncoder._coarsen
    timed=EZSPEncoder.timed

    def __init__(self, profile, resource_budget, *, allow_unvalidated_profile):
        if profile.get('region_scales')==1:
            expected=single_profile(profile['partition']['reg_scale1'],profile['backend']['options']['sharding'])
        else:
            expected=fixed_profile(load_profile(reg_scale1=profile['partition']['reg_scale1'],reg_scale2=profile['partition']['reg_scale2'],
                sharding=profile['backend']['options']['sharding']))
        if expected!=profile:raise ValueError('Unknown fixed partition contract')
        self.profile=copy.deepcopy(profile)
        self.resource_budget=resource_budget
        self.allow_unvalidated_profile=allow_unvalidated_profile
        self.timing_enabled=True; self.detailed_diagnostics=True
        self.last_audit={}; self.last_topology={}

    def _admission(self, stats, budget, violations):
        self.resource_budget.check()
        stats['diagnostic_only']=True
        stats['profile_exceeded']=bool(budget or violations)
        if (budget or violations) and not self.allow_unvalidated_profile:
            raise CoarseningConstraintError('Fixed partition initial profile rejected; no repair/skip/fallback',stats)


@torch.no_grad()
def prepare(batch, frozen_reference, profile, bindings, budget, *, allow_unvalidated_profile):
    """One disjoint batch on GPU; CPU splitting happens only during preparation.

The caller must supply frozen weights explicitly. Random weights are permitted
only with the explicit untrained plumbing label; no production artifacts here.
"""
    budget.check();validate_batch(batch)
    if len(bindings)!=len(batch):raise ValueError('Exact record bindings required')
    verify_materialization(batch,bindings)
    cnn_hash=tree_hash(frozen_reference.dense_encoder.state_dict())
    if any(b['frozen_cnn_sha256']!=cnn_hash or b['profile']!=profile or b['training_ready'] is not False for b in bindings):
        raise ValueError('Frozen feature/profile identity mismatch')
    if frozen_reference.training or any(p.requires_grad for p in frozen_reference.parameters()):
        raise ValueError('Partition encoder must be explicitly frozen and eval')
    source,target=frozen_reference.encode_dense_maps(batch.source_patches,batch.source_index,batch.target_patches)
    shape,jump,origin=cnn_lattice(frozen_reference.dense_encoder)
    features={};regions={}
    for role in NT:
        node=batch.graph[role];meta=batch.sidecar[role]
        feature=source if role in SOURCE_LOCAL_NODE_TYPES else target
        grid=input_grid_to_feature_grid(node.grid,feature_shape=shape,stride=jump,origin=origin)
        features[role]=sample_nodes(feature,grid,node.batch)
        shell=node.shell_id.long() if 'context' in role else torch.full_like(node.batch,-1)
        same=next(e for e in ET if e[0]==role and e[2]==role)
        adj,w=undirected(batch.graph[same].edge_index,node.batch*4+shell+1)
        pos=meta['pos_mm']
        regions[role]=Region(node.batch,shell,torch.ones(len(node.batch),device=pos.device),pos,pos,meta['stable_id'],adj,w)
    runner=OfflinePartition(profile,budget,allow_unvalidated_profile=allow_unvalidated_profile)
    scales=profile.get('region_scales',2)
    level=1
    try:
        first,r1,e1=runner._coarsen(features,regions,batch.graph.edge_index_dict,1,len(batch))
        # No trainable GNN features enter either partition decision.
        if scales==2:
            level=2
            _,r2,e2=runner._coarsen(first,r1,e1,2,len(batch))
    except CoarseningConstraintError as exc:
        # Structural failures retain their original classification and payload.
        if str(exc)=='Fixed partition initial profile rejected; no repair/skip/fallback':
            from .admission_report import rejection_report
            exc.admission_report=rejection_report(runner,level,bindings,profile)
        raise
    if scales==1:
        return _single_items(batch,bindings,runner,regions,r1,e1,cnn_hash,frozen_reference,budget)
    runner.last_audit['scale2_feature_basis']='frozen CNN32 pooled by original-node mass'
    # Independent quotient verification while fine edges are still available.
    for level,edges,parents,inputs in ((1,e1,runner.last_topology[1]['parents'],batch.graph.edge_index_dict),
                                      (2,e2,runner.last_topology[2]['parents'],e1)):
        for rel in ET:
            expected=quotient(inputs[rel],parents[rel[0]],parents[rel[2]],same_type=rel[0]==rel[2])[0]
            if not torch.equal(expected,edges[rel]):raise ValueError('Preparation quotient mismatch')
    # Cumulative original-feature dispersion is diagnostic, not a new merge/admission rule.
    cumulative={}
    for role,x in features.items():
        u=torch.nn.functional.normalize(x.float(),dim=1,eps=profile['partition']['features']['eps'])
        p1=runner.last_topology[1]['parents'][role];p2=runner.last_topology[2]['parents'][role][p1]
        means,mass=mass_mean(u,p2,torch.ones(len(u),device=u.device))
        variance=torch.zeros_like(mass).index_add_(0,p2,(u-means[p2]).square().sum(1))/mass
        cumulative[role]=[dict(owner=i,clusters=int((r2[role].owner==i).sum()),
            variance=variance[r2[role].owner==i].tolist(),
            original_mass=r2[role].mass[r2[role].owner==i].tolist()) for i in range(len(batch))]
    runner.last_audit['scale2_cumulative_original_normalized_feature_variance']=cumulative
    runner.last_audit['scale2']['variance_basis']='normalized stage1 centroids, weighted by original fine mass'
    cpu=lambda t:t.detach().cpu()
    # Compact per-pair records are independently loadable/rebatchable; there is
    # no fine edge field in this schema.
    items=[]
    for i,identity in enumerate(bindings):
        item=dict(binding=copy.deepcopy(identity),diagnostic_only=True,training_ready=False,
            source_patch=cpu(batch.source_patches[batch.source_index[i]]),target_patch=cpu(batch.target_patches[i]),
            fine={},scales=[{},{}],edges=[{},{}],audit=copy.deepcopy(runner.last_audit))
        indices=[{},{}];lookup=[{},{}]
        for level,rs in enumerate((r1,r2)):
            for role,r in rs.items():
                ids=torch.where(r.owner==i)[0];indices[level][role]=ids
                remap=torch.full_like(r.owner,-1);remap[ids]=torch.arange(len(ids),device=ids.device)
                lookup[level][role]=remap
                item['scales'][level][role]={name:cpu(getattr(r,name)[ids]) for name in ('shell','mass','lower','upper','stable_id')}
        for role,r in regions.items():
            mask=r.owner==i
            p1=runner.last_topology[1]['parents'][role]
            p2=runner.last_topology[2]['parents'][role]
            item['fine'][role]=dict(grid=cpu(batch.graph[role].grid[mask]),pos_mm=cpu(r.lower[mask]),
                shell=cpu(r.shell[mask]),stable_id=cpu(r.stable_id[mask]),parent=cpu(lookup[0][role][p1[mask]]))
            item['scales'][0][role]['parent']=cpu(lookup[1][role][p2[indices[0][role]]])
        for level,(rs,edges) in enumerate(((r1,e1),(r2,e2))):
            for rel,e in edges.items():
                keep=rs[rel[0]].owner[e[0]]==i
                item['edges'][level][rel]=cpu(torch.stack((lookup[level][rel[0]][e[0,keep]],lookup[level][rel[2]][e[1,keep]])))
        item['profile_exceeded']=any(runner.last_audit[f'scale{s}']['profile_exceeded'] for s in (1,2))
        # This batch flag is conservative: never call an individually rejected
        # pair admitted merely because another pair passed.
        item['admission_scope']='whole preparation batch; inspect per-pair counts and role/shell diagnostics'
        item['materialization_receipt']=copy.deepcopy(batch.receipts[i])
        item['preparation_pair_index']=i
        item['fine_graph_evidence']=dict(materialized_batch_sha256=batch.materialized_sha256,
            fine_edges_sha256=tree_hash(batch.graph.edge_index_dict),
            stage1_quotient_verified=True,stage2_quotient_verified=True)
        from .data import seal_item,profile_status
        item['preparation_batch_profile_exceeded']=item['profile_exceeded']
        item['profile_exceeded']=profile_status(item)
        item['admission_scope']='per pair; separate conservative preparation batch status retained'
        seal_item(item)
        items.append(item)
    if tree_hash(frozen_reference.dense_encoder.state_dict())!=cnn_hash:raise RuntimeError('Frozen CNN mutated')
    budget.check()
    return items,runner.last_audit


def _single_items(batch,bindings,runner,regions,r1,e1,cnn_hash,frozen_reference,budget):
    parents=runner.last_topology[1]['parents']
    for rel in ET:
        expected=quotient(batch.graph[rel].edge_index,parents[rel[0]],parents[rel[2]],same_type=rel[0]==rel[2])[0]
        if not torch.equal(expected,e1[rel]):raise ValueError('Preparation quotient mismatch')
    runner.last_audit['scale2']={'status':'REMOVED_BY_DESIGN'}
    cpu=lambda t:t.detach().cpu()
    items=[]
    from .data import seal_item,profile_status
    for i,identity in enumerate(bindings):
        item=dict(binding=copy.deepcopy(identity),diagnostic_only=True,training_ready=False,
            source_patch=cpu(batch.source_patches[batch.source_index[i]]),target_patch=cpu(batch.target_patches[i]),
            fine={},scales=[{}],edges=[{}],audit=copy.deepcopy(runner.last_audit))
        lookup={}
        for role,r in r1.items():
            ids=torch.where(r.owner==i)[0]
            remap=torch.full_like(r.owner,-1);remap[ids]=torch.arange(len(ids),device=ids.device);lookup[role]=remap
            item['scales'][0][role]={name:cpu(getattr(r,name)[ids]) for name in ('shell','mass','lower','upper','stable_id')}
            fine=regions[role];mask=fine.owner==i
            item['fine'][role]=dict(grid=cpu(batch.graph[role].grid[mask]),pos_mm=cpu(fine.lower[mask]),
                shell=cpu(fine.shell[mask]),stable_id=cpu(fine.stable_id[mask]),parent=cpu(remap[parents[role][mask]]))
        for rel,e in e1.items():
            keep=r1[rel[0]].owner[e[0]]==i
            item['edges'][0][rel]=cpu(torch.stack((lookup[rel[0]][e[0,keep]],lookup[rel[2]][e[1,keep]])))
        item['materialization_receipt']=copy.deepcopy(batch.receipts[i]);item['preparation_pair_index']=i
        item['fine_graph_evidence']=dict(materialized_batch_sha256=batch.materialized_sha256,
            fine_edges_sha256=tree_hash(batch.graph.edge_index_dict),stage1_quotient_verified=True,stage2_status='REMOVED_BY_DESIGN')
        item['preparation_batch_profile_exceeded']=runner.last_audit['scale1']['profile_exceeded']
        item['profile_exceeded']=profile_status(item);item['admission_scope']='per pair; single-scale profile'
        seal_item(item);items.append(item)
    if tree_hash(frozen_reference.dense_encoder.state_dict())!=cnn_hash:raise RuntimeError('Frozen CNN mutated')
    budget.check()
    return items,runner.last_audit
