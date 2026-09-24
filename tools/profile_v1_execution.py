"""DEBUG actual full-size, varying-graph batch timing. Never a trained result."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time,gc,os
import torch,psutil
from hiercp_v222.v1_cache import PairDataset,PairLoader,configuration
from hiercp_v222.v1_training import groups
from hiercp_v222.v1_local import model,collate
from hiercp_v222.training import configure_runtime
from hiercp_v222.model import supervised_loss

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache');parser.add_argument('output')
    parser.add_argument('policy',choices=('release_unused','default'))
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--allocator-gb',type=float,help='Optional PyTorch allocator cap in decimal GB; not MIG compute emulation')
    args=parser.parse_args()
    if args.batch_size<1:raise ValueError('Positive physical batch size required')
    if args.allocator_gb is not None:
        total=torch.cuda.get_device_properties(0).total_memory
        cap=int(args.allocator_gb*10**9)
        if not 0<cap<total:raise ValueError('Allocator cap must be positive and smaller than visible VRAM')
        torch.cuda.set_per_process_memory_fraction(cap/total)
    cache,output,policy=args.cache,args.output,args.policy
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();configure_runtime(base,42);torch.set_num_threads(8)
    dataset=PairDataset(cache,'inner_train');loader=PairLoader(dataset,8)
    net=model(cfg,base).cuda();net.local.dense_batch_size=args.batch_size
    fixture=torch.load(os.environ.get('HIERCP_TEST_FIXTURE',ROOT/'work/v222_v1_l0_20260924/debug2/actual_graphs_DEBUG.pt'),weights_only=False,map_location='cpu')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        net.eval();support=net.local(collate([(v,i) for i,v in enumerate(fixture['values'])]).cuda()).float()
    owners=torch.tensor([0,0,1,1,2,2],device='cuda');classes=torch.tensor([1,0,1,0,1,0],device='cuda')
    net.train();optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],fused=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(support,owners,classes)
    batches=list(groups(dataset,args.batch_size,42,0))[:10] # Explicit DEBUG profile, production cohort untouched.
    iterator=iter(loader.batches(batches));reports=[]
    try:
        for i in range(len(batches)):
            start=time.perf_counter();cpu=next(iterator);loaded=time.perf_counter()
            torch.cuda.reset_peak_memory_stats();events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record();query=cpu.cuda(non_blocking=True);events[1].record()
            target=torch.tensor([dataset.rows[j]['target'] for j in cpu.indices.tolist()],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=supervised_loss(net(query,support,owners,classes,cluster_plan=plan),target)
            events[2].record();loss.backward();events[3].record()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);optimizer.step();events[4].record()
            torch.cuda.synchronize();stats=torch.cuda.memory_stats()
            row=dict(debug=True,full_training=False,support_scope='six real CT regression observations, not full training memory',
                step=i+1,batch=len(cpu),nodes=int(cpu.graph.num_nodes),edges=int(cpu.graph.num_edges),
                loader_wait=loaded-start,H2D=events[0].elapsed_time(events[1])/1000,
                forward=events[1].elapsed_time(events[2])/1000,backward=events[2].elapsed_time(events[3])/1000,
                optimizer=events[3].elapsed_time(events[4])/1000,seconds=time.perf_counter()-start,
                peak_allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.memory_reserved(),
                inactive_split=stats.get('inactive_split_bytes.all.current',0),rss=psutil.Process().memory_info().rss,
                allocator_policy=policy,allocator_cap_gb=args.allocator_gb,requested_batch=args.batch_size,loss=float(loss.detach()))
            del query,loss,cpu;optimizer.zero_grad(set_to_none=True)
            if policy=='release_unused':torch.cuda.empty_cache()
            elif policy!='default':raise ValueError('Unknown policy')
            row['seconds_with_cleanup']=time.perf_counter()-start
            with (root/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            reports.append(row);print(json.dumps(row),flush=True)
    finally:loader.close()
    (root/'complete.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
if __name__=='__main__':main()
