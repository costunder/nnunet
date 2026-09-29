"""Same full fine graph, cloned states, unchanged observed-rank objective.

Short local DEBUG only. Alternating order, explicit warmup/repeats/resources.
"""
import argparse,copy,gc,json,sys,statistics
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,psutil
from l0_sage.encoder import GraphSAGEEncoder
from l0_ezsp.data import load_pairs
from l0_ezsp.validation import validate_batch
from l0_ezsp.diagnostic import ResourceBudget
from hiercp_v222.v1_cache import configuration,provenance
from hiercp_v222.v1_local import V1LocalEncoder
from hiercp_v222.model import PromptGraphModel
from tools.v222_review_contracts import installed,grouped_support
from tools.v22_debug_profile import RepairDataset
from tools.diagnose_l0_ezsp_pairs import update_loss,tensor_hash
from tools.profile_l0_ezsp_debug import timed,sha

def unchanged_parameters(gat,sage):
    candidate=dict(sage.named_parameters());checked=[]
    for name,p in gat.named_parameters():
        if name.startswith('local.blocks.') and '.conv.' in name:continue
        mapped=name.replace('local.','local.encoder.',1) if name.startswith('local.') else name
        if not torch.equal(p,candidate[mapped]):raise ValueError('Unrequested parameter change: '+name)
        checked.append(name)
    return checked

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('physical-batch','workers','repeats','warmup-updates'):p.add_argument('--'+name,type=int,required=True)
    for name in ('cuda-gib','rss-gib','seconds'):p.add_argument('--'+name,type=float,required=True)
    a=p.parse_args()
    if min(a.physical_batch,a.workers,a.repeats)<1 or a.warmup_updates<0:raise ValueError('Invalid explicit diagnostic sizes')
    a.output.mkdir(parents=True,exist_ok=False)
    budget=ResourceBudget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30),a.seconds)
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('Explicit device headroom required')
    previous=torch.cuda.get_per_process_memory_fraction();torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(a.workers)
    ds=RepairDataset(a.cache,'inner_train')
    if len(ds.rows)!=a.physical_batch:raise ValueError('Use all records from this DEBUG cache in one physical batch; no slicing')
    cfg,base=configuration();torch.manual_seed(42)
    with installed('stride4'):local=V1LocalEncoder(base)
    gat=PromptGraphModel(cfg,base,{},local_encoder=local).cpu()
    sage=copy.deepcopy(gat);sage.local=GraphSAGEEncoder(gat.local,seed=42)
    checked=unchanged_parameters(gat,sage)
    masters={'gat':gat,'sage_mean':sage};hashes={k:tensor_hash(v) for k,v in masters.items()}
    report={'debug':True,'production_admitted':False,'training_ready':False,'full_training':False,
        'scope':'complete DEBUG update; 8 real observations, patient-excluded support, original rank/auxiliary/alignment loss; no full production validation',
        'model_changes':['only local relation convolutions: custom gated GATv2 -> PyG GraphSAGE mean/root','SAGE attention heads not applicable; GAT retains 4','SAGE has no attention dropout; existing residual/FFN dropout preserved'],
        'preserved':{'layers':3,'hidden_dim':128,'output_dim':128,'cnn_channels':[12,24,32],'roles':5,'relations':13,'feature_coordinates':'stride4',
            'source_target_CT':[1,48,48,48],'fine_graph':'all original sampled-view nodes and edges','neighbor_sampling':False,'coarsening':False,
            'L1_L2_loss':True,'candidate_count':128,'Basic_CP_and_masks_unchanged':True,'shared_parameters_verified':len(checked)},
        'physical_batch':a.physical_batch,'effective_batch':a.physical_batch,'gradient_accumulation':1,'workers':a.workers,
        'warmup_updates_per_arm':a.warmup_updates,'measured_updates_per_arm':a.repeats,'precision':'FP32','seed':42,
        'weight_source':'same initialized master outside replaced convolutions; each update starts from a clone; no trained checkpoint',
        'resources':{'gpu':torch.cuda.get_device_name(),'gpu_count':torch.cuda.device_count(),'total_vram':total,'free_vram_start':torch.cuda.mem_get_info()[0],
            'cpu_physical':psutil.cpu_count(logical=False),'available_ram':psutil.virtual_memory().available,'cuda_allocator_limit_bytes':budget.cuda_bytes,'rss_limit_bytes':budget.rss_bytes,'seconds_limit':a.seconds},
        'core_identity':provenance(),'cache_sha256':sha(a.cache),'tool_sha256':sha(__file__),
        'sage_source':{str(f):sha(f) for f in (ROOT/'l0_sage').glob('*.py')},'initial_master_hashes':hashes,
        'modes':{k:{'total_parameters':sum(p.numel() for p in v.parameters()),'local_parameters':sum(p.numel() for p in v.local.parameters()),'runs':[]} for k,v in masters.items()},
        'unmeasured':['accuracy and CP utility','MIG10GB','production physical32/full observations/full support','epoch and validation','BF16/AMP','production checkpoints']}
    try:
        start=perf_counter();batch=load_pairs(ds,list(range(len(ds.rows))),workers=a.workers).pin_memory()
        report['loader_graph_seconds']=perf_counter()-start
        batch,report['h2d_seconds']=timed(lambda:batch.to('cuda'));validate_batch(batch)
        report['records']=[r['id'] for r in ds.rows]
        report['fine_nodes']=sum(batch.graph[k].num_nodes for k in batch.graph.node_types)
        report['fine_edges']=sum(batch.graph[e].edge_index.shape[1] for e in batch.graph.edge_types)
        groups=sorted({r['patient_group'] for r in ds.rows});contexts={g:[i for i,r in enumerate(ds.rows) if r['patient_group']==g] for g in groups}
        targets=torch.tensor([r['target'] for r in ds.rows],device='cuda');counts=torch.bincount(targets,minlength=2).float()
        if bool((counts==0).any()):raise ValueError('Both actual observation classes required')
        weights=counts.sum()/(2*counts)
        for trial in range(a.warmup_updates+a.repeats):
            for name in (('gat','sage_mean') if trial%2==0 else ('sage_mean','gat')):
                budget.check();net=copy.deepcopy(masters[name]).cuda().eval();torch.manual_seed(42)
                with torch.no_grad():embedding,memory_seconds=timed(lambda:net.local(batch))
                memory={'embeddings':embedding.detach(),'record_ids':[r['id'] for r in ds.rows],'patient_groups':groups,
                    'donor_groups':[r['donor_group'] for r in ds.rows],
                    'owners':torch.tensor([groups.index(r['patient_group']) for r in ds.rows],device='cuda'),'classes':targets}
                plans,plan_seconds=timed(lambda:{g:net.fit_support_clusters(*grouped_support(memory,g)) for g in groups})
                optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
                net.train();torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();budget.check();start=perf_counter()
                (loss,terms),forward=timed(lambda:update_loss(net,batch,ds,memory,contexts,plans,weights))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                budget.check();_,backward=timed(loss.backward)
                _,clip=timed(lambda:torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True))
                _,step=timed(optimizer.step);whole=perf_counter()-start
                # Validate real-path gradients after timing to avoid contaminating cost.
                prefixes=('local.encoder.dense_encoder','local.encoder.blocks.0.conv','local.encoder.blocks.1.conv','local.encoder.blocks.2.conv','l1','l2') if name=='sage_mean' else ('local.dense_encoder','local.blocks.0.conv','local.blocks.1.conv','local.blocks.2.conv','l1','l2')
                gradient_report={}
                for prefix in prefixes:
                    grads=[p.grad for n,p in net.named_parameters() if n.startswith(prefix)]
                    valid=bool(grads) and all(g is not None and bool(torch.isfinite(g).all()) for g in grads) and any(bool((g!=0).any()) for g in grads)
                    if not valid:raise RuntimeError('Gradient connection failed: '+prefix)
                    gradient_report[prefix]=True
                row={'trial':trial,'warmup':trial<a.warmup_updates,'whole_update_seconds':whole,'forward_seconds':forward,'backward_seconds':backward,
                    'clip_seconds':clip,'optimizer_seconds':step,'memory_refresh_seconds':memory_seconds,'plan_seconds':plan_seconds,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'loss':float(loss.detach()),'terms':terms,'gradients':gradient_report}
                if name=='sage_mean':row['adjacency_build_count']=net.local.adjacency.builds
                report['modes'][name]['runs'].append(row);budget.check()
                print(name,'warmup' if row['warmup'] else 'measured',f"{whole:.4f}s peak={row['peak_allocated_bytes']/2**30:.3f}GiB",flush=True)
                del net,optimizer,loss,embedding,memory,plans;gc.collect();torch.cuda.empty_cache()
        for name,master in masters.items():
            if tensor_hash(master)!=hashes[name]:raise RuntimeError('Master state mutated')
            rows=[r for r in report['modes'][name]['runs'] if not r['warmup']]
            report['modes'][name]['summary']={k:{'mean':statistics.mean(r[k] for r in rows),'median':statistics.median(r[k] for r in rows),'min':min(r[k] for r in rows),'max':max(r[k] for r in rows)} for k in ('whole_update_seconds','forward_seconds','backward_seconds','memory_refresh_seconds','peak_allocated_bytes')}
        report['status']='DEBUG_COMPARISON_COMPLETE_NOT_PRODUCTION_READY'
    except Exception as exc:
        report['status']='ERROR';report['error']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        torch.cuda.set_per_process_memory_fraction(previous)
        report['rss_bytes_end']=psutil.Process().memory_info().rss
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('REPORT:',a.output/'report.json')

if __name__=='__main__':main()
