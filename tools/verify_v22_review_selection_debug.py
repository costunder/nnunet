"""Actual fixed-donor graphs → scoring → eligibility → raw paste DEBUG."""
from pathlib import Path
import sys,json
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(cache=None,checkpoint=None,output=None):
    import numpy as np
    import torch
    from hiercp_v222.v1_cache import load_verified,sources,configuration
    from hiercp_v222.v1_local import pair_record,prepare_donor
    from hiercp_v22.data import donor_in_target_spacing
    from hiercp_v222.placement import placement_spec
    from tools.v22_rank_recommendation import load_checkpoint,recommend
    from hiercp_v222.training import configure_runtime
    if cache is None:cache,checkpoint,output=sys.argv[1:4]
    cache,checkpoint,output=map(Path,(cache,checkpoint,output));output.mkdir(parents=True,exist_ok=False)
    meta=json.loads(cache.read_text());raw={r['case_id']:r for r in meta['raw_records']}
    rows=[r for r in meta['records'] if r['case_id'] in meta['split']['inner_val']]
    if len(rows)!=2 or len({r['case_id'] for r in rows})!=1:raise ValueError('Two actual validation candidates required')
    # Fix the donor from the observed DEBUG row for both candidate positions.
    # Selection of this diagnostic fixture does not change production donor draws.
    donor_row=next(r for r in rows if r['target']==1)
    recipient=rows[0]['case_id'];donor_id=donor_row['donor_case_id'];component=donor_row['donor_component']
    cfg,base=configuration();configure_runtime(base,cfg['seed']);torch.set_num_threads(8)
    target,organ,depth=load_verified(raw[recipient]);donor,dorgan,ddepth=load_verified(raw[donor_id])
    source=next(s for s,_ in sources(donor,base['cache']['source_pad'],20) if s.component_id==component)
    prepared=prepare_donor(donor,source,dorgan,ddepth,base)
    transformed,_=donor_in_target_spacing(source,donor.spacing,target.spacing)
    # Additional explicit DEBUG positive-paste control. This is not a production
    # proposal rule or a donor/position resampling policy.
    interior=tuple(map(int,np.unravel_index(np.argmax(depth),depth.shape)))
    centers=[tuple(r['center']) for r in rows]+[interior]
    if len(set(centers))!=len(centers):raise ValueError('Distinct DEBUG paste-control center required')
    placements=[placement_spec(target,transformed,c,donor_id) for c in centers]
    def graph(p):return pair_record(target,source,donor.spacing,prepared,p.center,organ,depth,base,donor_id=donor_id,placement=p)
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(graph,placements))
    network,memory,saved=load_checkpoint(checkpoint,allow_debug=True)
    group=saved['identities']['cases'][recipient]['patient_group']
    options=dict(query_group=group,batch_size=saved['physical_batch'],workers=saved['workers'],
                 placements=placements,min_liver_coverage=base['generation']['min_liver_coverage'])
    original=recommend(network,records,memory,recipient_case=target,**options)
    worker_results={}
    for workers in (0,1,2,4,8):
        report=recommend(network,records,memory,recipient_case=target,**dict(options,workers=workers))
        if report!=original:raise AssertionError(f'Recommendation differs with workers={workers}')
        worker_results[str(workers)]=True
    repeated=recommend(network,records,memory,recipient_case=target,**options)
    changed=replace(target,label=np.where(target.label==2,1,target.label).astype(target.label.dtype))
    no_tumors=recommend(network,records,memory,recipient_case=changed,**options)
    left=[(r['index'],r['model_score'],r['raw_rank']) for r in original['ranked_candidates']]
    control=[(r['index'],r['model_score'],r['raw_rank']) for r in repeated['ranked_candidates']]
    right=[(r['index'],r['model_score'],r['raw_rank']) for r in no_tumors['ranked_candidates']]
    if left!=control:raise AssertionError(f'Unchanged-input inference is not repeatable: {left} vs {control}')
    if left!=right:raise AssertionError(f'Recipient tumor annotation changed raw score/rank: {left} vs {right}')
    for row in original['ranked_candidates']:
        points=placements[row['index']].coordinates()
        inside=((points>=0)&(points<np.asarray(target.label.shape))).all(1)
        overlap=int((target.label[tuple(points[inside].T)]==2).sum())
        if overlap!=row['tumor_overlap_voxels']:raise AssertionError('Whole footprint overlap mismatch')
    paste_audit=None
    selected=original['selected_index']
    if selected is not None:
        image,label=placements[selected].paste(target.image,target.label)
        expected=np.zeros(label.shape,bool);expected[tuple(placements[selected].coordinates().T)]=True
        if not np.array_equal((label==2)&(target.label!=2),expected):raise AssertionError('Paste attribution mismatch')
        indices=tuple(placements[selected].coordinates().T)
        if not np.array_equal(image[indices],placements[selected].image[placements[selected].mask]):raise AssertionError('Pasted CT values mismatch')
        if not np.array_equal(image[~expected],target.image[~expected]) or not np.array_equal(label[~expected],target.label[~expected]):
            raise AssertionError('Paste modified voxels outside selected footprint')
        paste_audit=dict(actual_pasted_voxels=int(expected.sum()),exact_placement_voxels=True,exact_CT_values=True,unchanged_outside_footprint=True)
    else:
        raise AssertionError('Explicit DEBUG valid paste control failed; no successful paste was tested')
    try:recommend(network,records,memory,recipient_case=target,**dict(options,query_group='wrong'))
    except ValueError:wrong_group_rejected=True
    else:raise AssertionError('Wrong query group accepted')
    try:load_checkpoint(checkpoint)
    except ValueError:production_rejects_debug=True
    else:raise AssertionError('DEBUG artifact admitted for production')
    result=dict(debug=True,actual_CT=True,full_model=True,full_training=False,nnunet_preprocessing_tested=False,
        recipient=recipient,donor=donor_id,component=component,candidates=len(records),support=len(memory['record_ids']),
        workers_bitwise_identical=worker_results,extra_interior_center_is_DEBUG_control_only=True,
        footprint_voxels=int(placements[0].mask.sum()),
        annotation_score_rank_invariant=True,unchanged_input_bitwise_repeatable=True,
        deterministic_execution=base['runtime']['deterministic'],
        wrong_group_rejected=wrong_group_rejected,production_rejects_debug=production_rejects_debug,
        original=original,annotation_removed=no_tumors,paste=paste_audit)
    with (output/'result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
