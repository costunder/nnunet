"""Native CNN CP scoring; preserve exact original placement and rank/filter rules."""
import numpy as np
import torch
from .data import CropStore

@torch.no_grad()
def recommend(network,memory,payload,recipient_case,donor_case,source,placements,*,batch_size,workers):
    from hiercp_v222.placement import validate_grid,placement_spec,array_hash
    from hiercp_v22.data import donor_in_target_spacing
    from tools.v222_review_contracts import grouped_support
    from tools.v22_rank_recommendation import rank_then_filter
    from tools.v22_candidate_order import record_key
    validate_grid(recipient_case);validate_grid(donor_case)
    recipient=recipient_case.paths.case_id;donor=donor_case.paths.case_id
    if network.training or batch_size<1 or workers<2 or not placements:raise ValueError('Eval mode, nonempty candidates and parallel loader required')
    known=payload['identities']['cases'];group=known[recipient]['patient_group']
    if donor not in payload['split']['inner_train'] or known[donor]['patient_group']==group:raise ValueError('Held-out/self patient donor')
    transformed,_=donor_in_target_spacing(source,donor_case.spacing,recipient_case.spacing)
    for p in placements:
        expected=placement_spec(recipient_case,transformed,p.center,donor)
        if p.metadata()!=expected.metadata():raise ValueError('Native CNN placement differs from original donor mask/CT/frame')
    rows=[dict(id=f'{recipient}:cnn:{i}',case_id=recipient,donor_case_id=donor,donor_component=source.component_id,
        component_id=source.component_id,patient_group=group,donor_group=known[donor]['patient_group'],center=list(p.center)) for i,p in enumerate(placements)]
    store=getattr(network,'local_crop_store',None)
    if store is None:
        store=CropStore(payload,workers,payload['resident_budget_bytes'],payload['resource_limits']['rss_bytes']);network.local_crop_store=store
    store.raw.preload(rows)
    for case in (recipient_case,donor_case):
        raw=store.raw.cache[case.paths.case_id]
        if not np.array_equal(raw['ct'],case.image) or not np.array_equal(raw['organ'],np.isin(case.label,[1,2])):
            raise ValueError('Scored native CT/organ differs from actual paste event')
    state=network.prepare_support(*grouped_support(memory,group));scores=[]
    with torch.autocast('cuda',enabled=False):
        for start in range(0,len(rows),batch_size):
            part=rows[start:start+batch_size];q=store.batch(part,list(range(start,start+len(part)))).to('cuda')
            logits=network.predict_embeddings(network.local(q),state)['logits'].float();scores.append(logits[:,1]-logits[:,0])
    score=torch.cat(scores).cpu().numpy();first=placements[0]
    return rank_then_filter(score,[p.center for p in placements],first.mask,first.anchor,recipient_case.label,
        min_liver_coverage=payload['base']['generation']['min_liver_coverage'],candidate_keys=[record_key(r) for r in rows])
