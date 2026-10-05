"""Actual CT/CUDA DEBUG: certified absence, mixed batching and native D loss.

No production training or checkpoint is created. Native labels, original
full masks and10mm coordinates are used without alteration. The empty-only
case has no observed P, so its native ranking term is intentionally zero.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'inventory', 'old-cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import psutil
    import torch
    from tools.run_v17_crossed_training import prepare_source
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    torch.manual_seed(42)
    source, scope = prepare_source(args.source)
    from hiercp_v1x import transition_v1_local as local
    from hiercp_v1x.transition_v1_data import NativeObservationDataset, OriginalInputProvider, _verify_stored_files
    from hiercp_v1x.transition_model import CrossedModel, full_support_for_case, tensor_hash
    from hiercp_v1x.transition_runtime import native_D_loss, gradient_receipt
    from l0_regions.donor_learning import LiveContext
    from hiercp_v22.storage import load_record
    inventory_bytes = args.inventory.read_bytes()
    ds = NativeObservationDataset(args.inventory, 'inner_train', True)
    provider = OriginalInputProvider(ds, workers=2, resident_bytes=24*2**30, rss_bytes=32*2**30)
    case_rows = [r for r in ds.rows if r['case_id']=='liver_106']
    # Fixed explicit32-row DEBUG cost/mechanics probe, including exact failedrow.
    selected = case_rows[:31]
    failed = next(r for r in case_rows if r['id']=='liver_106:2')
    if failed not in selected:
        selected.append(failed)
    else:
        selected = case_rows[:32]
    print('ACTUAL CT DEBUG | build32 unchanged liver_106 rows including liver_106:2', flush=True)
    started = time.perf_counter()
    records = provider._records_for(selected)
    record_map = dict(zip((r['id'] for r in selected), records))
    geometry_seconds = time.perf_counter()-started
    empty_rows = [r for r in selected if records[selected.index(r)]['target_local']['nodes']['target_context']['x'].shape[0]==0]
    if not empty_rows:
        raise AssertionError('The actual failed recipient absence was not reproduced')
    print(f'CT DEBUG | built32, actual empty recipients={len(empty_rows)}; loading original completed support', flush=True)
    index = json.loads(args.old_cache.read_text(encoding='utf8'))
    raw = {r['id']:r for r in ds.rows}
    supported = [r for r in index['records'] if r['case_id'] in {'liver_1','liver_5','liver_6'}]
    support_records = []
    for row in supported:
        storage = _verify_stored_files(args.old_cache.parent, row)
        record = load_record(storage, row['path'])
        local.validate_record(record)
        support_records.append(record)
        record_map[row['id']] = record
    expected = [r for r in ds.rows if r['case_id'] in {'liver_1','liver_5','liver_6'}]
    if {r['id'] for r in supported} != {r['id'] for r in expected}:
        raise ValueError('DEBUG support must preserve complete selected cases P+128U')
    base = copy.deepcopy(ds.base)
    base['model'].update(checkpoint_dense_encoder=False, checkpoint_local_blocks=False)
    net = CrossedModel(local.PreservedV1LocalEncoder(base, scope_contract=scope['contract_sha256'],
        expected_snapshot_root=args.source), arm='D', scope_contract=scope['contract_sha256'], dropout=.1, debug=True).cuda()
    context = LiveContext(ds, 32)
    lookup = {r['id']:i for i,r in enumerate(ds.rows)}

    def batch_for(rows):
        return local.collate([(local.materialize_pair(record_map[r['id']], epoch=0), lookup[r['id']]) for r in rows]).to('cuda')

    net.eval()
    parts=[]
    with torch.no_grad():
        for start in range(0, len(supported), 32):
            rows=[raw[r['id']] for r in supported[start:start+32]]
            parts.append(net.local(batch_for(rows)).detach())
    names=sorted({raw[r['id']]['patient_group'] for r in supported})
    memory=dict(embeddings=torch.cat(parts), row_groups=[raw[r['id']]['patient_group'] for r in supported],
        donor_groups=[raw[r['id']]['donor_group'] for r in supported],
        owners=torch.tensor([names.index(raw[r['id']]['patient_group']) for r in supported],device='cuda'),
        classes=torch.tensor([raw[r['id']]['target'] for r in supported],device='cuda'))
    print(f'CUDA DEBUG | fresh original-model support={len(supported)} actual rows; testing mixed32 and empty-only', flush=True)
    control_case = [raw[r['id']] for r in supported if r['case_id']=='liver_1']
    positive=next(r for r in control_case if r['target']==1)
    unobserved=next(r for r in control_case if r['target']==0)
    groups=[('mixed_actual32', selected), ('all_target_context_empty',empty_rows), ('nonempty_P_U_control',[positive,unobserved])]
    trials=[]
    initial=tensor_hash(net.state_dict())
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    for name, rows in groups:
        support=full_support_for_case(memory,rows[0]['patient_group'])
        net.eval()
        with torch.no_grad():
            plan=net.fit_support_clusters(*support)
        query=batch_for(rows)
        before=tensor_hash(net.state_dict())
        net.train();optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        targets=torch.tensor([r['target'] for r in rows],device='cuda')
        loss,terms=native_D_loss(net,query,support,plan,targets,context,[lookup[r['id']] for r in rows])
        loss.backward()
        gradient=gradient_receipt(net)
        torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        after=tensor_hash(net.state_dict())
        receipt=dict(name=name,observations=[r['id'] for r in rows],physical_observations=len(rows),
            actual_local_graphs=2*len(rows),loss=float(loss.detach()),gradient=gradient,
            terms={key:float(value.detach()) if isinstance(value,torch.Tensor) else value for key,value in terms.items()},
            changed=before!=after,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),
            node_counts={role:int(query.graph[role].num_nodes) for role in local._ROLE_NAMES})
        if not receipt['changed']:
            raise AssertionError('Actual native loss optimizer did not change neural weights')
        trials.append(receipt)
        print(json.dumps(receipt,allow_nan=False),flush=True)
        del query,loss,terms,support,plan
    report=dict(format='v17_actual_empty_recipient_CUDA_DEBUG_v1',debug=True,source=source,scope=scope,
        local_identity=local.source_identity(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        device_total_bytes=torch.cuda.get_device_properties(0).total_memory,physical_probe32_is_production_batch_unit=True,
        production_server_40GiB_admission_not_claimed=True,parameters=sum(p.numel() for p in net.parameters()),
        initial_neural_sha256=initial,final_neural_sha256=tensor_hash(net.state_dict()),
        actual_geometry_seconds=geometry_seconds,full_DEBUG_support_records=len(supported),trials=trials,
        rss_bytes=psutil.Process().memory_info().rss,production_training_started=False,checkpoint_created=False,
        quality_verified=False,originals_modified=False,scope_changed=False,GT_or_donor_assignment_changed=False,
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest())
    if args.inventory.read_bytes()!=inventory_bytes:
        raise AssertionError('Read-only native inventory changed')
    with (args.output/'report.json').open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
