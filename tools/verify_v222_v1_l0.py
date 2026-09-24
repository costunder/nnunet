"""Real-CT DEBUG of v1-style L0 attached to unchanged v2.22 L1/L2.

Three explicitly named inner-training recipients and two observations each
are diagnostics, not a production subset. One optimizer step is a connectivity
check, never a trained checkpoint. Optional CP check scores 128 actual sites.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import psutil
    import torch
    from hiercp.common import discover_cases, load_case, organ_depth_mm
    from hiercp_v22.data import sources, candidate_pool, donor_in_target_spacing
    from hiercp_v222.contracts import read_json, write_new, sha, verify_v1
    from hiercp_v222.model import supervised_loss
    from hiercp_v222.v1_local import (prepare_donor, pair_record, materialize, collate,
        model, score_candidates, support_for_recipient, observation_loss)
    from hiercp.preparation_runtime import snapshot
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-cp128', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required, no fallback')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = read_json(ROOT/'config/prompt_graph_v222_v1_l0.json')
    base = read_json(ROOT/cfg['base_config'])
    index = read_json(os.environ.get('HIERCP_TEST_OBSERVATION_INDEX', ROOT/'work/v222_raw_ct_r3_training_20260923/context/index_vram_affine.json'))
    cases = ['liver_5', 'liver_6', 'liver_108']
    donor_id = 'liver_1'
    if not set(cases+[donor_id]) <= set(index['split']['inner_train']):
        raise ValueError('DEBUG cases must all belong to inner_train')
    for c in cases:
        if {r['target'] for r in index['records'] if r['case_id']==c} != {0,1}:
            raise ValueError(f'DEBUG fixture requires both observed classes: {c}; no data loaded or silently skipped')
    raw = {r['case_id']:r for r in index['raw_records']}
    medical = Path(raw[donor_id]['image']).parents[2]
    paths = {r.case_id:r for r in discover_cases(medical/'Data')}
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(psutil.cpu_count(logical=False))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resource = snapshot()
    write_new(root/'started.json', dict(debug=True, config=cfg, base=base,
        cases=cases, donor=donor_id, records_per_case=2, cpu_workers=len(cases),
        source_scope='original CT and provided masks; case-only independence as previously authorized',
        gpu=torch.cuda.get_device_name(), vram=torch.cuda.mem_get_info(), resources=resource,
        v1_revision=verify_v1(), model_source_sha256=sha(ROOT/'hiercp_v222/v1_local.py')))

    def load(c):
        row=raw[c]
        if sha(paths[c].image_path)!=row['image_sha256'] or sha(paths[c].label_path)!=row['label_sha256']:
            raise ValueError(f'Original data changed: {c}')
        case=load_case(paths[c]); organ=np.isin(case.label, [1,2])
        return case,organ,organ_depth_mm(organ,case.spacing)
    donor,donor_organ,donor_depth=load(donor_id)
    donor_source,_=sources(donor,base['cache']['source_pad'],cfg['donor_max_diameter_mm'])[0]
    prepared=prepare_donor(donor,donor_source,donor_organ,donor_depth,base)
    print(json.dumps(dict(stage='donor_ready',nodes=prepared.canonical_counts)),flush=True)

    def build(c):
        start=time.perf_counter(); case,organ,depth=load(c)
        rows=[next(r for r in index['records'] if r['case_id']==c and r['target']==t) for t in (1,0)]
        records=[pair_record(case,donor_source,donor.spacing,prepared,r['center'],organ,depth,base,donor_id=donor_id) for r in rows]
        values=[materialize(r) for r in records]
        print(json.dumps(dict(stage='recipient_ready',case=c,seconds=time.perf_counter()-start,
            graphs=[v[0].num_nodes for v in values])),flush=True)
        return rows,records,values
    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        built=list(executor.map(build,cases))
    rows=[r for group,_,_ in built for r in group]
    records=[r for _,group,_ in built for r in group]
    values=[v for _,_,group in built for v in group]
    audits=[]
    for row,record,value in zip(rows,records,values):
        g=value[0]
        if 'tumor_interior' in g.node_types or any('x' in g[k] for k in g.node_types) or any('edge_attr' in g[e] for e in g.edge_types):
            raise AssertionError('Retired learned inputs reintroduced')
        audits.append(dict(case=row['case_id'],target=row['target'],center=row['center'],
            nodes={k:g[k].num_nodes for k in g.node_types},edges={'|'.join(e):g[e].edge_index.shape[1] for e in g.edge_types},
            canonical_nodes={b:record[b]['counts'] for b in ('source_local','target_local')}))
    write_new(root/'graphs.json',audits)
    # Save actual connectivity/CT for inspection without fabricating predictions.
    with (root/'actual_graphs_DEBUG.pt').open('xb') as f:
        torch.save(dict(values=values,rows=rows,debug=True,trained=False),f)
    net=model(cfg,base).cuda().eval()
    reports=[]
    for count in (2,4,6):
        batch=collate([(v,i) for i,v in enumerate(values[:count])]).to('cuda')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            net.local(batch); torch.cuda.synchronize(); begin=time.perf_counter()
            for _ in range(3): net.local(batch)
            torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        reports.append(dict(physical_batch=count,unique_sources=len(batch.source_patches),
            graphs_per_second=3*count/seconds,peak_vram_bytes=torch.cuda.max_memory_allocated()))
        del batch
    batch=collate([(v,i) for i,v in enumerate(values)]).to('cuda')
    with torch.no_grad(): embeddings=net.local(batch)
    memory=dict(embeddings=embeddings.detach(),owners=torch.tensor([0,0,1,1,2,2],device='cuda'),
        classes=torch.tensor([r['target'] for r in rows],device='cuda'),case_ids=cases,
        patient_groups=[index['identities']['cases'][c]['patient_group'] for c in cases],
        donor_groups=[index['identities']['cases'][donor_id]['patient_group']]*len(rows))
    support=support_for_recipient(memory,memory['patient_groups'][0])
    if set(support[1].tolist())!={0,1} or len(support[0])!=4:
        raise AssertionError('Whole query case must be excluded from support')
    query=collate([(values[i],i) for i in (0,1)]).to('cuda')
    with torch.no_grad():
        fields=net.local.forward_fields(query)
        clean=net(query,*support)['ranking_score']
        changed=query.to('cuda'); changed.target_patches=changed.target_patches.flip(-1)
        altered=net(changed,*support)['ranking_score']
        delta=float((clean-altered).abs().max())
        if delta<=1e-8: raise AssertionError('Recipient CT intervention did not reach ranking')
        changed=query.to('cuda'); changed.source_patches=changed.source_patches.flip(-1)
        source_delta=float((clean-net(changed,*support)['ranking_score']).abs().max())
        if source_delta<=1e-8: raise AssertionError('Donor CT intervention did not reach ranking')
    net.train(); optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=0)
    before={n:v.detach().clone() for n,v in net.named_parameters()}
    loss,_=observation_loss(net,query,memory['classes'][:2],memory,memory['patient_groups'][0]);loss.backward()
    missing=[n for n,v in net.named_parameters() if v.grad is None]
    if missing: raise AssertionError(f'Parameters disconnected from loss: {missing}')
    if not all(torch.isfinite(v.grad).all() for v in net.parameters()): raise FloatingPointError('Nonfinite gradient')
    optimizer.step()
    groups=['local.dense_encoder','local.project','local.blocks.0','local.blocks.1','local.blocks.2',
            'local.source_relation','local.target_relation','local.final_fuse','l1.0','l1.1','l2.0','l2.1','l2_updates.0','l2_updates.1']
    updated={g:any(not torch.equal(before[n],v) for n,v in net.named_parameters() if n.startswith(g)) for g in groups}
    if not all(updated.values()):raise AssertionError(f'No optimizer update: {updated}')
    net.eval(); cp_report=None
    if args.check_cp128:
        case,organ,depth=load(cases[0])
        target_source,_=donor_in_target_spacing(donor_source,donor.spacing,case.spacing)
        pool,_=candidate_pool(case,target_source,cfg,base,depth,donor_case_id=donor_id)
        def candidate(p):
            return pair_record(case,donor_source,donor.spacing,prepared,p.center,organ,depth,base,donor_id=donor_id)
        with ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
            candidates=list(executor.map(candidate,pool))
        batch_size=max(reports,key=lambda r:r['graphs_per_second'])['physical_batch']
        scores,selected=score_candidates(net,candidates,memory,
            query_group=memory['patient_groups'][0],batch_size=batch_size)
        cp_report=dict(candidates=128,scores=scores.cpu().tolist(),selected=int(selected),
            centers=[list(map(int,p.center)) for p in pool],debug_optimizer_steps=1,
            valid_learned_recommendation=False,paste_executed=False)
        write_new(root/'cp128_DEBUG.json',cp_report)
    result=dict(debug=True,actual_CT=True,implementation='v1_ct_only_context_pair',
        L0_parameters=sum(p.numel() for p in net.local.parameters()),
        total_parameters=sum(p.numel() for p in net.parameters()),benchmarks=reports,
        loss=float(loss.detach()),all_parameter_gradients_finite=True,updated_groups=updated,
        recipient_CT_score_delta=delta,donor_CT_score_delta=source_delta,
        local_fields={k:list(v.shape) for k,v in fields.items()},support_query_excluded=True,
        cp128_scored=cp_report is not None,whole_training=False,whole_evaluation=False,
        optimizer_steps=1,checkpoint_created=False,raw_inputs_changed=False,
        limitation='Numerical/context connectivity DEBUG; no learned CP benefit or segmentation result',
        v1_revision_after=verify_v1(),resources_after=snapshot())
    write_new(root/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
