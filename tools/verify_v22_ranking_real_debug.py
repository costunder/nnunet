"""Actual CT, full-model ranking smoke and exact optimizer resume; DEBUG only.

One complete observed case plus six support observations from three other cases.
All nodes/edges retained; production cohort/config is never rewritten.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch,psutil
    import hiercp.model as implementation
    from hiercp_v222.v1_cache import configuration,provenance
    from hiercp_v222.v1_local import model
    from hiercp_v222.v1_execution import memory_metadata,tree_to,restore_rng
    from hiercp_v222.training import configure_runtime
    from tools.v222_review_contracts import installed,grouped_support
    from tools.v222_runtime_cache import CachedPairDataset
    from tools.v222_process_loader import ProcessPairLoader,close_producers
    from tools.v222_support_snapshot import AsyncSaver
    from tools.v22_rank_objective import RankingContext,configuration as rank_configuration,OBJECTIVE,resolve_objective
    from tools.v22_ranking_steps import optimizer_step
    from tools.verify_v1_resume import equal
    from tools.run_v222_process_runtime import runtime_identity
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--allocator-gb',type=float,default=9.)
    p.add_argument('--feature-coordinates',choices=('stride4',),default='stride4')
    p.add_argument('--workspace-mib',type=int,choices=(64,256,512),default=256)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();settings=rank_configuration();configure_runtime(base,42);torch.set_num_threads(8)
    if a.batch_size<2 or a.allocator_gb<=0:raise ValueError('Valid explicit DEBUG batch/allocator required')
    torch.cuda.set_per_process_memory_fraction(a.allocator_gb*1e9/torch.cuda.get_device_properties(0).total_memory)
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=a.workspace_mib*1024**2
    data=CachedPairDataset(a.cache,'inner_train');by_group={}
    for i,row in enumerate(data.rows):by_group.setdefault(row['patient_group'],[]).append(i)
    support_ids=[]
    for group,ids in sorted(by_group.items()):
        if {data.rows[i]['target'] for i in ids}=={0,1}:
            support_ids.extend(next(i for i in ids if data.rows[i]['target']==target) for target in (0,1))
        if len(support_ids)==6:break # Declared DEBUG support, never production.
    if len(support_ids)!=6:raise ValueError('Three actual support groups required for this DEBUG')
    excluded={data.rows[i][key] for i in support_ids for key in ('patient_group','donor_group')}
    candidates={g:ids for g,ids in by_group.items() if g not in excluded and any(data.rows[i]['target']==1 for i in ids)
                and sum(data.rows[i]['target']==0 for i in ids)>=a.batch_size}
    query_group=max(candidates,key=lambda g:sum(data.rows[i]['bounds']['edges'] for i in candidates[g]))
    selected=support_ids+candidates[query_group]
    class View:
        rows=[data.rows[i] for i in selected]
        meta=data.meta
        path=data.path
        def __len__(self):return len(self.rows)
    view=View();loader=ProcessPairLoader(view,8)
    try:
        with installed('stride4'):
            net=model(cfg,base).cuda();net.local.dense_batch_size=a.batch_size;net.eval();values=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for cpu in loader.batches([list(range(i,min(i+a.batch_size,len(view)))) for i in range(0,len(view),a.batch_size)]):
                    values.append(net.local(cpu.cuda(non_blocking=True)).float())
            memory=memory_metadata(view,torch.cat(values));del values
            context=RankingContext(view,memory);support=grouped_support(memory,query_group)
            with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*support)
            # All live queries are unobserved: observed anchors must come from
            # complete same-case references, not a convenient balanced minibatch.
            ids=sorted([i for i,r in enumerate(view.rows) if r['patient_group']==query_group and r['target']==0],
                       key=lambda i:view.rows[i]['bounds']['edges'],reverse=True)[:a.batch_size]
            cpu=loader.make(ids,0);targets=torch.tensor([view.rows[i]['target'] for i in ids],device='cuda')
            counts=torch.bincount(torch.tensor([r['target'] for r in data.rows],device='cuda'),minlength=2).float()
            weights=counts.sum()/(2*counts);net.train()
            optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
            args=(support,plan,targets,weights,base['training']['grad_clip'])
            first,timing,usage,gradients=optimizer_step(net,optimizer,cpu,*args,context=context,settings=settings,check_gradients=True)
            assert first['ranking_pairs']>0
            identity=dict(debug=True,training_objective=OBJECTIVE,ranking_contract=settings)
            AsyncSaver.execution_policy=dict(training_objective=OBJECTIVE,feature_coordinates='stride4',runtime_sha256=runtime_identity())
            saver=AsyncSaver(root,net,optimizer,identity)
            state=dict(phase='optimization',epoch=0,step=1,next_batch=1,memory=memory,plan=plan)
            saver.save(state);saver.flush()
            expected,_,_,_=optimizer_step(net,optimizer,cpu,*args,context=context,settings=settings)
            expected_model=tree_to(net.state_dict(),'cpu');expected_optimizer=tree_to(optimizer.state_dict(),'cpu')
            saver.close();del net,optimizer,saver,context,memory,support,plan,args;torch.cuda.empty_cache()
            saved=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            assert resolve_objective(saved)==OBJECTIVE
            net=model(cfg,base).cuda();net.local.dense_batch_size=a.batch_size;net.load_state_dict(saved['model']);net.train()
            optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
            optimizer.load_state_dict(saved['optimizer']);state=tree_to(saved['state'],'cuda')
            context=RankingContext(view,state['memory']);support=grouped_support(state['memory'],query_group)
            restore_rng(saved['rng'])
            actual,_,_,_=optimizer_step(net,optimizer,cpu,support,state['plan'],targets,weights,base['training']['grad_clip'],context=context,settings=settings)
            assert actual==expected and equal(expected_model,tree_to(net.state_dict(),'cpu')) and equal(expected_optimizer,tree_to(optimizer.state_dict(),'cpu'))
            result=dict(debug=True,real_CT=True,full_training=False,full_support=False,parameters=sum(p.numel() for p in net.parameters()),
                physical_batch=a.batch_size,gradient_accumulation=1,reference_observations=len(view),support_observations=6,
                query_case=view.rows[ids[0]]['case_id'],all_live_queries_unobserved=True,
                nodes=int(cpu.graph.num_nodes),edges=int(cpu.graph.num_edges),
                optimizer_resume_loss_model_adam_bitwise_equal=True,**first,**timing,**usage,**gradients,
                runtime_sha256=runtime_identity(),source_identity=provenance(),ranking_contract=settings,
                cpu_logical=psutil.cpu_count(),available_ram=psutil.virtual_memory().available,gpu=torch.cuda.get_device_name(),
                scope='full-model DEBUG update with complete one-case ranking references and six other-case support; not accuracy validation')
            (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result),flush=True)
    finally:loader.close();close_producers()


if __name__=='__main__':main()
