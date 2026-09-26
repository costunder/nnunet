"""Corrected-coordinate full model / physical32 smoke on actual cached graphs.

Six real support observations from three inner-train groups; one distinct heavy
query group. Not full support, a complete epoch, or a medical/CP efficacy score.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch,psutil
    import hiercp.model as implementation
    from hiercp_v222.v1_cache import configuration,provenance
    from hiercp_v222.v1_local import model
    from hiercp_v222.v1_execution import memory_metadata,optimizer_step
    from hiercp_v222.training import configure_runtime
    from hiercp_v222.deterministic_sampling import sample_nodes
    from tools.v222_review_contracts import installed,grouped_support
    from tools.v222_runtime_cache import CachedPairDataset
    from tools.v222_process_loader import ProcessPairLoader,close_producers
    from tools.run_v222_process_runtime import runtime_identity
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();configure_runtime(base,42);torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(9e9/torch.cuda.get_device_properties(0).total_memory)
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=256*1024**2
    dataset=CachedPairDataset(ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json','inner_train')
    by_group={}
    for i,row in enumerate(dataset.rows):by_group.setdefault(row['patient_group'],[]).append(i)
    support_ids=[]
    for group,ids in sorted(by_group.items()):
        if {dataset.rows[i]['target'] for i in ids}=={0,1}:
            support_ids.extend(next(i for i in ids if dataset.rows[i]['target']==target) for target in (0,1))
        if len(support_ids)==6:break # Explicit diagnostic support only.
    if len(support_ids)!=6:raise ValueError('Three observed groups with both classes required for DEBUG')
    excluded={dataset.rows[i][key] for i in support_ids for key in ('patient_group','donor_group')}
    candidates={group:sorted(ids,key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)
        for group,ids in by_group.items() if group not in excluded and len(ids)>=32}
    query_group=max(candidates,key=lambda g:sum(dataset.rows[i]['bounds']['edges'] for i in candidates[g][:32]))
    query_ids=candidates[query_group][:32] # Declared physical32 DEBUG capacity test.
    class SupportView:
        rows=[dataset.rows[i] for i in support_ids]
        meta=dataset.meta
    loader=ProcessPairLoader(dataset,8)
    try:
        support_cpu=loader.make(support_ids,0)
        query_cpu=loader.make(query_ids,0)
        with installed('stride4'):
            net=model(cfg,base).cuda();net.local.dense_batch_size=32;net.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                payload=support_cpu.cuda(non_blocking=True)
                corrected=net.local(payload).float()
                with patch.object(net.local,'_sample_dense_features',sample_nodes):legacy=net.local(payload).float()
            if torch.equal(corrected,legacy):raise AssertionError('Coordinate adapter did not affect actual L0 output')
            delta=float((corrected-legacy).abs().max());del payload,legacy
            memory=memory_metadata(SupportView,corrected)
            support=grouped_support(memory,query_group)
            with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*support)
            net.train();optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
            targets=torch.tensor([dataset.rows[i]['target'] for i in query_ids],device='cuda')
            all_targets=torch.tensor([r['target'] for r in dataset.rows],device='cuda')
            counts=torch.bincount(all_targets,minlength=2).float();weights=counts.sum()/(2*counts)
            values,timing,usage,gradients=optimizer_step(net,optimizer,query_cpu,support,plan,targets,weights,5,check_gradients=True)
            result=dict(debug=True,real_CT=True,full_training=False,full_support=False,
                scope='One full-model physical32 update; six actual support records, not an accuracy/throughput benchmark',
                feature_coordinates='stride4',support_task_contract='patient_group_v1',
                runtime_sha256=runtime_identity(),source_identity=provenance(),
                parameters=sum(p.numel() for p in net.parameters()),physical_batch=32,gradient_accumulation=1,
                precision='bf16',workspace_mib=256,allocator_cap_bytes=9000000000,
                nodes=int(query_cpu.graph.num_nodes),edges=int(query_cpu.graph.num_edges),
                support_record_ids=[dataset.rows[i]['id'] for i in support_ids],query_record_ids=[dataset.rows[i]['id'] for i in query_ids],
                query_group=query_group,recipient_and_donor_exclusion_verified=query_group not in excluded,
                actual_L0_legacy_corrected_max_absolute_difference=delta,
                cpu_logical=psutil.cpu_count(),available_ram=psutil.virtual_memory().available,
                gpu=torch.cuda.get_device_name(),**values,**timing,**usage,**gradients)
            (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result),flush=True)
    finally:
        loader.close();close_producers()


if __name__=='__main__':main()
