"""Regenerate ten actual CT graphs for explicit repair DEBUG; full graph rules."""
from pathlib import Path
import sys,json
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from hiercp_v222.v1_cache import configuration,provenance,load_verified,sources
    from hiercp_v222.v1_local import pair_record,prepare_donor
    from hiercp_v222.placement import GEOMETRY_CONTRACT
    from hiercp_v22.storage import GraphWriter
    from hiercp_v22.volumes import VolumeCache
    from hiercp.common import CasePaths
    import numpy as np
    old=Path(sys.argv[1]);root=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=False)
    meta=json.loads(old.read_text(encoding='utf-8'));cfg,base=configuration();torch.set_num_threads(1)
    raw={r['case_id']:r for r in meta['raw_records']}
    size=lambda case:int(np.prod(raw[case]['shape']))
    selected=[]
    train_cases=[]
    for partition,needed in [('inner_train',4),('inner_val',1)]:
        found=0;chosen=[]
        for case in sorted(meta['split'][partition],key=size):
            rows=[r for r in meta['records'] if r['case_id']==case]
            if {r['target'] for r in rows}!={0,1}:continue
            chosen.append(case);found+=1
            if found==needed:break
        if found!=needed:raise ValueError('Actual DEBUG cohort unavailable')
        if partition=='inner_train':train_cases=chosen
        for case in chosen:
            rows=[r for r in meta['records'] if r['case_id']==case and r['donor_case_id'] not in train_cases]
            if {r['target'] for r in rows}!={0,1}:raise ValueError('DEBUG support cannot retain both classes after two-sided exclusion')
            selected.extend(min((r for r in rows if r['target']==label),key=lambda r:size(r['donor_case_id'])) for label in (0,1))
    paths={c:CasePaths(c,Path(r['image']),Path(r['label'])) for c,r in raw.items()}
    volumes=VolumeCache(paths,base,cfg)
    writer=GraphWriter(root,minimum_free_bytes=cfg['minimum_free_gb']*1024**3)
    def build(row):
        from hiercp_v222.placement import validate_grid
        from hiercp_v222.contracts import sha
        for case_id in (row['case_id'],row['donor_case_id']):
            for field in ('image','label'):
                if sha(raw[case_id][field])!=raw[case_id][field+'_sha256']:raise ValueError('Actual CT source hash changed')
        with volumes.pair(row['case_id'],row['donor_case_id']) as (target_data,donor_data):
            target=validate_grid(target_data['case']);donor=validate_grid(donor_data['case'])
            source=next(s for s,_ in donor_data['sources'] if s.component_id==row['donor_component'])
            prepared=prepare_donor(donor,source,donor_data['organ'],donor_data['depth'],base)
            record=pair_record(target,source,donor.spacing,prepared,row['center'],target_data['organ'],target_data['depth'],base,donor_id=row['donor_case_id'])
        entry={k:v for k,v in row.items() if k not in ('path','bounds','sha256','shared_source')}
        entry.update(writer.write(f"graphs/{row['id'].replace(':','_')}.pt.gz",record,f"{row['donor_case_id']}:{row['donor_component']}"))
        print(json.dumps(dict(debug=True,rebuilt=row['id'],nodes=entry['bounds']['nodes'],edges=entry['bounds']['edges'])),flush=True)
        return entry
    # Two independent full-volume+EDT pairs; RAM is checked by load/preparation.
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(build,selected))
    value=dict(meta,format='v22_review_repair_DEBUG_cache',debug=True,complete=False,
        config=dict(cfg,gnn_epochs=1,batch_calibration_repeats=1),records=records,
        source_identity=provenance(),geometry_contract=GEOMETRY_CONTRACT,
        debug_scope='8 train / 2 validation actual CT observations; full architecture and graph rules')
    with (root/'index.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    print(str(root/'index.json'),flush=True)


if __name__=='__main__':main()
