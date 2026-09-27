"""Score first, then exclude annotated tumor overlap using the paste footprint.

This is the paired ranking/selection API. Native nnU-Net event integration is
still separate; no synthetic score or annotation-based score override exists.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import numpy as np
import torch
from tools.v222_review_contracts import grouped_support
from hiercp_v222.v1_local import materialize,collate


def load_checkpoint(path,*,device='cuda',allow_debug=False):
    """Load a complete ranking artifact with matching features and train support."""
    from hiercp_v222.v1_cache import provenance
    from hiercp_v222.v1_local import model
    from hiercp_v222.v1_execution import tree_to
    from tools.v222_review_contracts import installed,resolve_feature_contract
    from tools.v22_rank_objective import resolve_objective,OBJECTIVE
    value=torch.load(path,map_location='cpu',weights_only=False)
    from tools.v22_artifacts import validate_artifact
    validate_artifact(value,'final',allow_debug=allow_debug)
    if resolve_objective(value)!=OBJECTIVE:raise ValueError('A ranking checkpoint is required')
    if value.get('debug',True) and not allow_debug:raise ValueError('DEBUG weights are not production weights')
    if value.get('source_identity')!=provenance():raise ValueError('Model/cache source identity changed')
    if resolve_feature_contract(value)!='stride4':raise ValueError('Corrected ranking coordinates required')
    if not {'memory','state_dict','config','base','split'}<=value.keys():raise ValueError('Complete final ranking artifact required')
    if not set(value['memory']['case_ids'])<=set(value['split']['inner_train']):raise ValueError('Held-out case entered support')
    with installed('stride4'):network=model(value['config'],value['base']).to(device)
    network.load_state_dict(value['state_dict']);network.eval()
    network.local.dense_batch_size=value['physical_batch']
    network.ranking_identities=value['identities']
    network.ranking_split=value['split']
    return network,tree_to(value['memory'],device),value


def rank_then_filter(scores,centers,footprint,anchor,recipient_label,*,min_liver_coverage,candidate_keys=None):
    from tools.v22_candidate_order import candidate_order,candidate_key,TIE_POLICY
    scores=np.asarray(scores,dtype=np.float64);centers=np.asarray(centers)
    mask=np.asarray(footprint);anchor=np.asarray(anchor);label=np.asarray(recipient_label)
    if scores.ndim!=1 or not np.isfinite(scores).all():raise ValueError('Finite model scores required')
    if not len(scores):
        if centers.size:raise ValueError('Empty scores with nonempty centers')
        return dict(selected_index=None,keep_original=True,ranked_candidates=[],score_override=False,
                    filter_after_model_scoring=True,tie_policy=TIE_POLICY)
    if centers.shape!=(len(scores),3) or not np.issubdtype(centers.dtype,np.integer):raise ValueError('Integer recipient voxel centers required')
    if len(np.unique(centers,axis=0))!=len(centers):raise ValueError('Duplicate candidate positions')
    if mask.ndim!=3 or mask.dtype!=bool or not mask.any() or label.ndim!=3:raise ValueError('Actual 3D boolean paste footprint and recipient label required')
    if anchor.shape!=(3,) or not np.issubdtype(anchor.dtype,np.integer) or ((anchor<0)|(anchor>=mask.shape)).any():raise ValueError('Valid paste anchor required')
    if not 0<min_liver_coverage<=1:raise ValueError('Explicit existing liver coverage rule required')
    points=centers[:,None,:]+np.argwhere(mask)[None,:,:]-anchor
    inside=((points>=0)&(points<np.asarray(label.shape))).all(-1)
    safe=np.minimum(np.maximum(points,0),np.asarray(label.shape)-1)
    labels=label[safe[...,0],safe[...,1],safe[...,2]]
    overlap=((labels==2)&inside).sum(1)
    coverage=(np.isin(labels,[1,2])&inside).mean(1)
    eligible=inside.all(1)&(overlap==0)&(coverage>=min_liver_coverage)
    keys=candidate_keys if candidate_keys is not None else [candidate_key('recipient','donor',1,c) for c in centers]
    order=candidate_order(scores,keys);eligible_order=[int(i) for i in order if eligible[i]]
    rows=[]
    for rank,i in enumerate(order,1):
        center_inside=bool(((centers[i]>=0)&(centers[i]<label.shape)).all())
        observed_center=bool(label[tuple(centers[i])]==2) if center_inside else False
        reasons=[]
        if not inside[i].all():reasons.append('paste_out_of_bounds')
        if overlap[i]:reasons.append('observed_tumor_overlap')
        if coverage[i]<min_liver_coverage:reasons.append('insufficient_liver_coverage')
        rows.append(dict(index=int(i),center=centers[i].tolist(),model_score=float(scores[i]),raw_rank=rank,
            observed_tumor_at_center=observed_center,tumor_overlap_voxels=int(overlap[i]),liver_coverage=float(coverage[i]),
            eligible=bool(eligible[i]),excluded_reasons=reasons,
            recommendation_rank=eligible_order.index(int(i))+1 if eligible[i] else None))
    return dict(selected_index=eligible_order[0] if eligible_order else None,
                keep_original=not bool(eligible_order),ranked_candidates=rows,
                tie_policy=TIE_POLICY,
                score_override=False,filter_after_model_scoring=True)


@torch.no_grad()
def recommend(network,records,memory,*,query_group,batch_size,workers,recipient_case,
              placements,min_liver_coverage):
    """All supplied candidates use one actual donor and one recipient CT.

    Footprint must be the exact mask at recipient spacing after any transform,
    with the identical anchor used by actual CT/label paste. No dilation or
    arbitrary tumor-distance threshold is introduced.
    """
    from tools.v22_candidate_order import record_key
    from hiercp_v222.placement import PlacementSpec,validate_grid
    if network.training or type(batch_size)!=int or batch_size<1 or type(workers)!=int or workers<0:
        raise ValueError('Eval model and measured nonnegative batch/worker policy required')
    identities=getattr(network,'ranking_identities',None)
    split=getattr(network,'ranking_split',None)
    if identities is None or split is None:raise ValueError('Verified ranking model identities/split required')
    if recipient_case is None:raise ValueError('Explicit recipient CT required even for empty proposals')
    validate_grid(recipient_case)
    recipient=recipient_case.paths.case_id
    if recipient not in identities['cases'] or query_group!=identities['cases'][recipient]['patient_group']:
        raise ValueError('Recipient/query group mismatch')
    if not np.isfinite(min_liver_coverage) or not 0<min_liver_coverage<=1:
        raise ValueError('Finite liver coverage in (0,1] required')
    if not records:
        if placements:raise ValueError('Placement/record count mismatch')
        return dict(selected_index=None,keep_original=True,ranked_candidates=[],score_override=False,filter_after_model_scoring=True)
    if len(placements)!=len(records) or not all(isinstance(p,PlacementSpec) for p in placements):
        raise ValueError('One verified PlacementSpec per graph required')
    first=placements[0]
    from hiercp_v222.record_binding import recipient_binding,validate_record
    binding=recipient_binding(recipient_case)
    event={k:v for k,v in first.metadata().items() if k!='center'}
    for record,placement in zip(records,placements):
        if placement.recipient!=recipient or tuple(recipient_case.spacing)!=placement.spacing or not np.array_equal(recipient_case.image_affine,np.asarray(placement.affine)):
            raise ValueError('Recipient native frame differs from graph/paste placement')
        if {k:v for k,v in placement.metadata().items() if k!='center'}!=event:
            raise ValueError('Fixed CP event donor/transform/CT/mask/anchor/frame differs')
        validate_record(record,binding)
        if record.get('placement')!=placement.metadata():raise ValueError('Scored graph and paste placement differ')
        if (record['case_id']!=placement.recipient or record['donor_case_id']!=placement.donor or
            record['component_id']!=placement.component or tuple(record['center'])!=placement.center):
            raise ValueError('Record/placement identity mismatch')
        if placement.donor not in split['inner_train']:raise ValueError('Held-out donor in recommendation')
        if identities['cases'][placement.donor]['patient_group']==query_group:raise ValueError('Self-patient CP donor')
        if not np.array_equal(first.mask,placement.mask) or first.anchor!=placement.anchor:
            raise ValueError('One CP event requires the same transformed donor footprint')
    if len({r['case_id'] for r in records})!=1 or len({(r['donor_case_id'],r['component_id']) for r in records})!=1:
        raise ValueError('One CP event must use a fixed recipient and donor component')
    support=grouped_support(memory,query_group);device=next(network.parameters()).device
    # Labels and paste-overlap flags are deliberately not passed to this scorer.
    with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):
        state=network.prepare_support(*support)
        values=[]
        # Training workers=0 means no decode threads. Keep graph GPU batching;
        # only CPU materialization is serial for this explicit measured policy.
        with (ThreadPoolExecutor(max_workers=workers) if workers else nullcontext()) as pool:
            for start in range(0,len(records),batch_size):
                part=records[start:start+batch_size]
                mapped=pool.map(materialize,part) if pool is not None else map(materialize,part)
                payload=collate(list(zip(mapped,range(start,start+len(part))))).to(device)
                logits=network.predict_embeddings(network.local(payload),state)['logits'].float()
                values.append(logits[:,1]-logits[:,0])
        scores=torch.cat(values).cpu().numpy()
    if len(scores)!=len(records):raise RuntimeError('Incomplete candidate scoring')
    return rank_then_filter(scores,[r['center'] for r in records],first.mask,first.anchor,
                            recipient_case.label,min_liver_coverage=min_liver_coverage,
                            candidate_keys=[record_key(r) for r in records])
