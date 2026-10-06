"""Read-only full-cohort input audit; never authorizes patient identity or trains."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from hiercp.common import discover_cases
    from hiercp.preparation_runtime import snapshot
    from hiercp_v22.parallel import run_jobs
    from hiercp_v22.volumes import volume_memory_bound
    from hiercp_v222.contracts import load_config,read_json,write_new,source_identity,validate_split
    from hiercp_v222.data import inspect_case,fit_contract,assert_no_duplicate_ct
    from hiercp_v222.inputs import topology
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();dataset=Path(args.dataset);out=Path(args.output)
    if not out.is_dir() or any(out.iterdir()):raise ValueError('Empty audit output directory required')
    cfg,base=load_config();split=read_json(dataset/'split.json');validate_split(split)
    ready=read_json(dataset/'dataset_ready.json');expected={r['case_id']:r for r in ready['patients']}
    paths={p.case_id:p for p in discover_cases(dataset/'medical/Data')}
    if set(paths)!=set(split['outer_train']+split['outer_val']) or set(paths)!=set(expected):
        raise ValueError('Full published cohort/verified receipt/split disagree')
    write_new(out/'started.json',dict(utc=datetime.now(timezone.utc).isoformat(),config=cfg,base=base,
        source_identity=source_identity(),resources=snapshot(),gpu=torch.cuda.get_device_name(),
        gpu_count=torch.cuda.device_count(),vram_free_total=torch.cuda.mem_get_info(),
        cases=len(paths),training_started=False,scope='Full-data read-only audit; no identity assertions'))
    inventory={}
    def scan(c):
        row=inspect_case(paths[c],cfg['donor_max_diameter_mm'])
        for key in ('image_sha256','label_sha256'):
            if row[key]!=expected[c][key]:raise ValueError(f'Raw file changed: {c} {key}')
        return c,row
    def commit(pair):
        case,row=pair;inventory[case]=row
        write_new(out/f'{case}.json',row)
        print(f'VERIFIED {len(inventory)}/{len(paths)} {case} eligible={len(row["positives"])}',flush=True)
    run_jobs(sorted(paths),scan,commit,cfg['preparation_workers'],out/'resources.json',
             memory_per_job=max(volume_memory_bound(p.image_path) for p in paths.values()))
    write_new(out/'inventory.json',inventory)
    assert_no_duplicate_ct(inventory,split)
    contract=fit_contract(inventory,split,base)
    violations=[dict(case=c,**row) for c in split['inner_val'] for row in inventory[c]['positives']
                if row['extent_mm']>contract['blind_radius_mm']-contract['blind_margin_mm']]
    grid,edge=topology(contract)
    summary=dict(format=cfg['format'],complete=True,training_started=False,
        verified_raw_cases=len(inventory),duplicate_decoded_CT=False,
        partitions={k:dict(cases=len(split[k]),eligible_lesions=sum(len(inventory[c]['positives']) for c in split[k]),
            all_tumor_components=sum(inventory[c]['all_tumor_components'] for c in split[k]))
            for k in ('inner_train','inner_val','outer_val')},
        input_contract=contract,inner_val_blind_violations=violations,
        nodes_per_graph=len(grid),edges_per_graph=edge.shape[1],
        identity_manifest_status='Not supplied; no independently verified patient mapping in local source',
        annotation_scope='Published labels; uncertain small lesions may be omitted per LiTS annotation protocol',
        publication='https://arxiv.org/html/1901.04056',resources=snapshot())
    write_new(out/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
