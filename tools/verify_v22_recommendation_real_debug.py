"""Actual cached fixed-donor candidates + original recipient annotation DEBUG."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch,numpy as np,nibabel as nib
    from hiercp_v222.v1_cache import configuration,sha
    from hiercp_v222.v1_local import model
    from hiercp_v222.v1_execution import tree_to
    from hiercp_v22.data import donor_in_target_spacing
    from tools.v222_runtime_cache import CachedPairDataset
    from tools.v222_review_contracts import installed
    from tools.v22_rank_recommendation import recommend
    checkpoint=Path(sys.argv[1]);root=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();torch.set_num_threads(8)
    data=CachedPairDataset(ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json','inner_train')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not saved['debug']:raise ValueError('This diagnostic requires the explicit DEBUG checkpoint')
    pairs={}
    for i,row in enumerate(data.rows):
        if row['case_id']=='liver_108':pairs.setdefault((row['donor_case_id'],row['donor_component']),[]).append(i)
    donor,ids=next((key,ids) for key,ids in sorted(pairs.items()) if {data.rows[i]['target'] for i in ids}=={0,1})
    reference=data.meta['donor_files'][f'{donor[0]}:{donor[1]}'];path=data.root/reference['path']
    if sha(path)!=reference['sha256']:raise ValueError('Donor transport hash mismatch')
    transport=torch.load(path,map_location='cpu',weights_only=False)
    raw=next(r for r in data.meta['raw_records'] if r['case_id']=='liver_108')
    if sha(raw['label'])!=raw['label_sha256']:raise ValueError('Recipient annotation changed')
    image=nib.load(raw['image']);annotation=nib.load(raw['label'])
    if not np.allclose(image.affine,annotation.affine):raise ValueError('Recipient annotation coordinates differ')
    label=np.asarray(annotation.dataobj,dtype=np.int16)
    source,mask=donor_in_target_spacing(transport['source'],np.asarray(transport['spacing']),np.asarray(image.header.get_zooms()[:3]))
    records=[data.record(i) for i in ids]
    with installed('stride4'):
        net=model(cfg,base).cuda();net.load_state_dict(saved['model']);net.eval();net.local.dense_batch_size=32
        memory=tree_to(saved['state']['memory'],'cuda')
        report=recommend(net,records,memory,query_group='case:liver_108',batch_size=32,workers=8,
            recipient_label=label,target_spacing_footprint=mask,paste_anchor=source.anchor_center,
            min_liver_coverage=base['generation']['min_liver_coverage'])
    annotated=[row for row in report['ranked_candidates'] if row['observed_tumor_at_center']]
    assert annotated and all(not row['eligible'] and row['tumor_overlap_voxels']>0 for row in annotated)
    if report['selected_index'] is not None:
        selected=next(row for row in report['ranked_candidates'] if row['index']==report['selected_index'])
        assert selected['tumor_overlap_voxels']==0 and selected['eligible']
    result=dict(debug=True,real_CT=True,actual_annotation=True,full_training=False,
        case='liver_108',fixed_donor=donor,records=[data.rows[i]['id'] for i in ids],
        footprint_voxels=int(mask.sum()),scope='selection path test with one-update DEBUG weights, not placement accuracy',**report)
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
