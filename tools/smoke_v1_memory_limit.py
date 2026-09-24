"""DEBUG: real full-size paired graphs under a strict CUDA allocator budget.

Tests execution feasibility, not MIG throughput or full-cohort training quality.
Each physical batch candidate has a separate process and fresh optimizer.
Uses an existing real initial-memory checkpoint prefix; no invented embeddings.
"""
from pathlib import Path
import argparse
import gc
import json
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def child(args,root):
    import torch
    import psutil
    from hiercp_v222.v1_cache import PairDataset,PairLoader,configuration,provenance
    from hiercp_v222.v1_local import model,support_for_recipient
    from hiercp_v222.v1_execution import optimizer_step,memory_metadata,Saver,tree_to,restore_rng
    from hiercp_v222.training import configure_runtime,require_device
    from tools.verify_v1_resume import equal
    require_device('cuda')
    total=torch.cuda.get_device_properties(0).total_memory
    cap=int(args.allocator_gb*10**9)
    if cap>=total:raise ValueError('This local emulation needs a cap smaller than physical VRAM')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    # Verify enforcement before loading any model; this must raise, not page.
    try:
        probe=torch.empty(cap+2**20,dtype=torch.uint8,device='cuda')
    except torch.cuda.OutOfMemoryError:
        enforced=True
    else:
        del probe
        raise AssertionError('Allocator budget was not enforced')
    torch.cuda.empty_cache()
    cfg,base=configuration();configure_runtime(base,42);torch.set_num_threads(8)
    dataset=PairDataset(args.cache,'inner_train')
    checkpoint=torch.load(args.support_checkpoint,weights_only=False,map_location='cpu')
    if checkpoint['source_identity']!=provenance() or checkpoint['config']!=cfg:
        raise ValueError('Support/model provenance mismatch')
    from hiercp_v222.contracts import sha
    if checkpoint['cache_sha256']!=sha(args.cache):raise ValueError('Support belongs to another cache')
    state=checkpoint['state']
    if state['phase']!='initial_memory' or state['step']!=0 or state['memory_next']<1:
        raise ValueError('Verified real initial support prefix required')
    prefix=state['memory_next'];embeddings=state['memory_work']
    if embeddings.shape!=(prefix,128) or not torch.isfinite(embeddings).all():raise ValueError('Invalid support prefix')
    class Prefix:
        rows=dataset.rows[:prefix]
        meta=dataset.meta
    memory=memory_metadata(Prefix,embeddings.cuda())
    net=model(cfg,base).cuda();net.load_state_dict(checkpoint['model']);net.local.dense_batch_size=args.batch
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],
        weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    del checkpoint,embeddings,state
    by_group={}
    for i,r in enumerate(dataset.rows):by_group.setdefault(r['patient_group'],[]).append(i)
    for ids in by_group.values():ids.sort(key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)
    ranked=sorted(by_group,key=lambda g:sum(dataset.rows[i]['bounds']['edges'] for i in by_group[g][:args.batch]),reverse=True)
    # Three heaviest recipient groups, two distinct batches each: explicit DEBUG.
    chosen=[(g,by_group[g][k*args.batch:(k+1)*args.batch]) for g in ranked[:3] for k in range(2)]
    if any(len(ids)!=args.batch for _,ids in chosen):raise ValueError('Not enough real full-size query batches')
    start=time.perf_counter();rows=[];loader=PairLoader(dataset,8)
    train_classes=torch.tensor([r['target'] for r in dataset.rows],device='cuda')
    counts=torch.bincount(train_classes,minlength=2).float();weights=counts.sum()/(2*counts)
    info=dict(debug=True,full_training=False,actual_CT=True,allocator_cap_bytes=cap,budget_enforced=enforced,
        gpu=torch.cuda.get_device_name(0),physical_vram=total,model_parameters=sum(p.numel() for p in net.parameters()),
        config=cfg,base_model=base['model'],source_identity=provenance(),physical_batch=args.batch,
        effective_batch=args.batch,gradient_accumulation=1,workers=8,prefetch=1,precision='bfloat16_autocast_FP32_loss',
        support_prefix=prefix,full_support_count=len(dataset),support_cases=len(memory['case_ids']),
        query_batches=[dict(group=g,ids=[dataset.rows[i]['id'] for i in ids]) for g,ids in chosen],
        cpu_logical=psutil.cpu_count(),cpu_physical=psutil.cpu_count(logical=False),ram_available=psutil.virtual_memory().available,
        limitations=['Allocator only: CUDA context/library allocations outside PyTorch are not capped.',
                    'RTX 5070 Ti is not an A100 MIG compute/bandwidth emulator.',
                    'Real support prefix and six query batches are DEBUG, not full training or clinical validation.'])
    write(root/'started.json',info)
    print(json.dumps(dict(stage='memory_limit_smoke',batch=args.batch,cap_bytes=cap,support=prefix,query_groups=[g for g,_ in chosen])),flush=True)
    net.train();before={n:p.detach().cpu().clone() for n,p in net.named_parameters()}
    try:
        for number,(group,ids) in enumerate(chosen):
            began=time.perf_counter();cpu=loader.make(ids,epoch=0)
            support=support_for_recipient(memory,group)
            with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*support)
            targets=train_classes[ids]
            values,timing,usage,gradients=optimizer_step(net,optimizer,cpu,support,plan,targets,weights,5,check_gradients=True)
            row=dict(step=number+1,group=group,batch=args.batch,nodes=int(cpu.graph.num_nodes),edges=int(cpu.graph.num_edges),
                     seconds=time.perf_counter()-began,**values,**timing,**usage,**gradients)
            if usage['peak_reserved']>cap:raise AssertionError('Allocator budget exceeded')
            rows.append(row);print(json.dumps(row),flush=True)
            with (root/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            torch.cuda.empty_cache()
            if number==0:
                updated={group:any(not torch.equal(before[n],p.detach().cpu()) for n,p in net.named_parameters() if n.startswith(group))
                         for group in ('local.','l1.','l2.','l2_updates.')}
                if not all(updated.values()):raise AssertionError(f'Missing module update: {updated}')
                del before
                saver=Saver(root,net,optimizer,dict(debug=True,allocator_cap_bytes=cap))
                receipt=saver.save(dict(phase='optimization',epoch=0,step=1,next_batch=1,memory=support,plan=plan))
                expected,_,_,_=optimizer_step(net,optimizer,cpu,support,plan,targets,weights,5)
                expected_model=tree_to(net.state_dict(),'cpu');expected_optimizer=tree_to(optimizer.state_dict(),'cpu')
                del saver,net,optimizer,plan,support;gc.collect();torch.cuda.empty_cache()
                saved=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
                net=model(cfg,base).cuda();net.local.dense_batch_size=args.batch;net.load_state_dict(saved['model']);net.train()
                optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
                optimizer.load_state_dict(saved['optimizer']);restored=tree_to(saved['state'],'cuda');restore_rng(saved['rng'])
                actual,_,_,_=optimizer_step(net,optimizer,cpu,restored['memory'],restored['plan'],targets,weights,5)
                if not equal(expected,actual) or not equal(expected_model,tree_to(net.state_dict(),'cpu')) or not equal(expected_optimizer,tree_to(optimizer.state_dict(),'cpu')):
                    raise AssertionError('Memory-limited resumed update changed')
                del expected_model,expected_optimizer,saved,restored;gc.collect();torch.cuda.empty_cache()
            del cpu,targets
        net.eval();cpu=loader.make(chosen[-1][1],epoch=0);support=support_for_recipient(memory,chosen[-1][0])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            evaluation=net.predict_embeddings(net.local(cpu.cuda()),net.prepare_support(*support))
        if not torch.isfinite(evaluation['logits']).all():raise FloatingPointError('Nonfinite eval logits')
        report=dict(status='passed',debug=True,full_training=False,physical_batch=args.batch,allocator_cap_bytes=cap,
            budget_enforced=enforced,support_prefix=prefix,query_batches=6,query_observations=6*args.batch,
            peak_allocated=max(r['peak_allocated'] for r in rows),peak_reserved=max(r['peak_reserved'] for r in rows),
            query_step_seconds=sum(r['seconds'] for r in rows),elapsed_seconds=time.perf_counter()-start,
            all_gradients_finite=True,updated_modules=updated,resume_model_optimizer_loss_bitwise_equal=True,
            eval_finite=True,checkpoint_bytes=receipt['bytes'],limitations=info['limitations'])
        write(root/'result.json',report);print(json.dumps(report),flush=True)
    finally:loader.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--support-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allocator-gb',type=float,default=9.0)
    p.add_argument('--batches',type=int,nargs='+',default=[16,32,64]);p.add_argument('--batch',type=int)
    args=p.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    if args.batch is not None:
        import torch
        try:child(args,root)
        except torch.cuda.OutOfMemoryError as e:
            write(root/'result.json',dict(status='oom',debug=True,full_training=False,physical_batch=args.batch,error=str(e)))
            print(json.dumps(dict(stage='expected_capacity_measurement',batch=args.batch,status='oom')),flush=True)
        except Exception:
            (root/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    else:
        reports=[]
        for batch in args.batches:
            output=root/f'batch{batch}_DEBUG'
            command=[sys.executable,str(Path(__file__).resolve()),'--cache',str(Path(args.cache).resolve()),
                '--support-checkpoint',str(Path(args.support_checkpoint).resolve()),'--output',str(output),
                '--allocator-gb',str(args.allocator_gb),'--batch',str(batch)]
            subprocess.run(command,check=True)
            reports.append(json.loads((output/'result.json').read_text(encoding='utf-8')))
        write(root/'summary.json',dict(debug=True,full_training=False,allocator_gb=args.allocator_gb,reports=reports))


if __name__=='__main__':main()
