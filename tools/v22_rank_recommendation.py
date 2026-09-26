"""Score first, then exclude annotated tumor overlap using the paste footprint.

This is the paired ranking/selection API. Native nnU-Net event integration is
still separate; no synthetic score or annotation-based score override exists.
"""
from concurrent.futures import ThreadPoolExecutor
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
    if resolve_objective(value)!=OBJECTIVE:raise ValueError('A ranking checkpoint is required')
    if value.get('debug',True) and not allow_debug:raise ValueError('DEBUG weights are not production weights')
    if value.get('source_identity')!=provenance():raise ValueError('Model/cache source identity changed')
    if resolve_feature_contract(value)!='stride4':raise ValueError('Corrected ranking coordinates required')
    if not {'memory','state_dict','config','base','split'}<=value.keys():raise ValueError('Complete final ranking artifact required')
    if not set(value['memory']['case_ids'])<=set(value['split']['inner_train']):raise ValueError('Held-out case entered support')
    with installed('stride4'):network=model(value['config'],value['base']).to(device)
    network.load_state_dict(value['state_dict']);network.eval()
    network.local.dense_batch_size=value['physical_batch']
    return network,tree_to(value['memory'],device),value


def rank_then_filter(scores,centers,footprint,anchor,recipient_label,*,min_liver_coverage):
    scores=np.asarray(scores,dtype=np.float64);centers=np.asarray(centers)
    mask=np.asarray(footprint);anchor=np.asarray(anchor);label=np.asarray(recipient_label)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():raise ValueError('Nonempty finite model scores required')
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
    order=np.argsort(-scores,kind='stable');eligible_order=[int(i) for i in order if eligible[i]]
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
                score_override=False,filter_after_model_scoring=True)


@torch.no_grad()
def recommend(network,records,memory,*,query_group,batch_size,workers,recipient_label,
              target_spacing_footprint,paste_anchor,min_liver_coverage):
    """All supplied candidates use one actual donor and one recipient CT.

    Footprint must be the exact mask at recipient spacing after any transform,
    with the identical anchor used by actual CT/label paste. No dilation or
    arbitrary tumor-distance threshold is introduced.
    """
    if network.training or batch_size<1 or workers<1 or not records:raise ValueError('Eval model and measured batch/workers required')
    if len({r['case_id'] for r in records})!=1 or len({(r['donor_case_id'],r['component_id']) for r in records})!=1:
        raise ValueError('One CP event must use a fixed recipient and donor component')
    support=grouped_support(memory,query_group);device=next(network.parameters()).device
    # Labels and paste-overlap flags are deliberately not passed to this scorer.
    with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):
        state=network.prepare_support(*support)
        values=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0,len(records),batch_size):
                part=records[start:start+batch_size]
                payload=collate(list(zip(pool.map(materialize,part),range(start,start+len(part))))).to(device)
                logits=network.predict_embeddings(network.local(payload),state)['logits'].float()
                values.append(logits[:,1]-logits[:,0])
        scores=torch.cat(values).cpu().numpy()
    if len(scores)!=len(records):raise RuntimeError('Incomplete candidate scoring')
    return rank_then_filter(scores,[r['center'] for r in records],target_spacing_footprint,paste_anchor,
                            recipient_label,min_liver_coverage=min_liver_coverage)
