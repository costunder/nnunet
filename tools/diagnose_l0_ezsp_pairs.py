"""Per-record strict audit plus resource-bounded cloned full-update diagnostic.

Writes JSON only. No production checkpoint, ready marker or long training.
"""
import argparse,copy,gc,hashlib,json,os,sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import psutil
from l0_ezsp.config import load_profile
from l0_ezsp.data import load_pairs
from l0_ezsp.encoder import EZSPEncoder
from l0_ezsp.diagnostic import DiagnosticEZSPEncoder,ResourceBudget
from l0_ezsp.partition import CoarseningConstraintError
from l0_ezsp.validation import validate_batch
from hiercp_v222.v1_cache import configuration
from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_local import V1LocalEncoder
from tools.v222_review_contracts import installed,grouped_support
from tools.v22_rank_objective import RankingContext,forward_loss,configuration as rank_config
from tools.profile_l0_ezsp_debug import timed,sha

def tensor_hash(module):
    h=hashlib.sha256()
    for name,t in module.named_parameters():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def update_loss(net,batch,dataset,memory,contexts,plans,weights):
    # One disjoint-union physical batch L0 call. Patient-excluded L1/L2 tasks
    # retain the existing objective, separately for each recipient group.
    embeddings,local_seconds=timed(lambda:net.local(batch))
    began=perf_counter()
    total=embeddings.new_zeros(());details=[]
    for group,ids in contexts.items():
        positions=torch.tensor(ids,device=embeddings.device)
        query=SimpleNamespace(target_patches=batch.target_patches[positions])
        facade=SimpleNamespace(local=lambda unused,positions=positions:embeddings[positions],
            prepare_support=net.prepare_support,predict_embeddings=net.predict_embeddings)
        support=grouped_support(memory,group)
        targets=torch.tensor([dataset.rows[i]['target'] for i in ids],device=embeddings.device)
        loss,terms=forward_loss(facade,query,support,plans[group],targets,weights,
            RankingContext(dataset,memory),rank_config(),indices=ids)
        total=total+loss*(len(ids)/len(batch))
        details.append({'group':group,'records':len(ids),'ranking_pairs':int(terms['ranking_pairs'])})
    torch.cuda.synchronize()
    return total,{'groups':details,'local_forward_seconds':local_seconds,'l1_l2_objective_seconds':perf_counter()-began}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reg-scale1',type=float,required=True);p.add_argument('--reg-scale2',type=float,required=True)
    p.add_argument('--physical-batch',type=int,required=True);p.add_argument('--workers',type=int,required=True)
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True);p.add_argument('--rss-gib',type=float,required=True);p.add_argument('--seconds',type=float,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.updates<1 or a.workers<1:raise ValueError('Explicit positive updates/workers required')
    budget=ResourceBudget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30),a.seconds)
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('CUDA budget must leave device headroom')
    previous_fraction=torch.cuda.get_per_process_memory_fraction()
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(min(a.workers,os.cpu_count() or a.workers))
    from tools.v22_debug_profile import RepairDataset
    ds=RepairDataset(a.cache,'inner_train')
    if len(ds.rows)!=a.physical_batch:raise ValueError('This diagnostic requires the entire supplied DEBUG inner_train in one physical batch; no slicing')
    cfg,base=configuration();profile=load_profile(reg_scale1=a.reg_scale1,reg_scale2=a.reg_scale2)
    report={'diagnostic_only':True,'production_admitted':False,'training_ready':False,
        'profile':profile,'weights':{'source':'fresh seed42 initialization, cloned for each update; no trained checkpoint',
        'cnn':'existing CNN architecture, initial weights','gat':'existing 3-block GAT initial weights; not trained'},
        'resource_limits':vars(budget),'limits_scope':'CUDA PyTorch allocator cap; RSS and time checked at phase boundaries, no process/session termination',
        'physical_batch':a.physical_batch,'workers':a.workers,'precision':'FP32','updates':a.updates,
        'cache_sha256':sha(a.cache),'records':[r['id'] for r in ds.rows],'strict_pairs':[], 'cost_only':{},
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cpu_physical':psutil.cpu_count(logical=False),'available_ram':psutil.virtual_memory().available,
        'scope':'complete DEBUG optimizer update: batch L0 plus recipient-excluded L1/L2 plus unchanged observed_rank_v1 loss, backward, clipping, AdamW; no production checkpoint',
        'not_measured':['full production support/observations','production physical32','epoch/validation/nnUNet','durable production checkpoint']}
    report['adapter_hashes']={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'l0_ezsp').rglob('*.py')}
    report['tool_sha256']=sha(__file__)
    report['initial_free_cuda_bytes']=torch.cuda.mem_get_info()[0]
    try:
        torch.manual_seed(42);strict=EZSPEncoder(base,profile).cuda().eval()
        strict.timing_enabled=True;strict.detailed_diagnostics=True
        report['weights']['cnn_sha256']=tensor_hash(strict.dense_encoder);report['weights']['gat_sha256']=tensor_hash(strict.blocks)
        for i,row in enumerate(ds.rows):
            budget.check();batch=load_pairs(ds,[i],workers=a.workers).to('cuda')
            entry={'record_id':row['id'],'case_id':row['case_id'],'donor_case_id':row['donor_case_id'],
                'fine_nodes':sum(batch.graph[k].num_nodes for k in batch.graph.node_types),
                'fine_edges':sum(batch.graph[e].edge_index.shape[1] for e in batch.graph.edge_types),
                'scale2':{'status':'NOT_RUN'}}
            start=perf_counter()
            try:
                with torch.no_grad():strict(batch)
                entry['status']='INITIAL_PROFILE_ADMITTED_NOT_PRODUCTION_READY'
            except CoarseningConstraintError as exc:
                entry['status']='INITIAL_PROFILE_REJECTED';entry['error']=str(exc)
            entry['audit']=copy.deepcopy(strict.last_audit);entry['seconds']=perf_counter()-start
            if 'scale2' in strict.last_audit:entry['scale2']={'status':'RUN','audit':strict.last_audit['scale2']}
            report['strict_pairs'].append(entry)
            (a.output/'report.json').write_text(json.dumps(report,indent=2))
            print(row['id'],entry['status'],f"{entry['seconds']:.3f}s",flush=True)
        del strict,batch;gc.collect();torch.cuda.empty_cache()
        start=perf_counter();batch=load_pairs(ds,list(range(len(ds.rows))),workers=a.workers).pin_memory()
        report['loader_seconds']=perf_counter()-start
        batch,report['h2d_seconds']=timed(lambda:batch.to('cuda'));validate_batch(batch)
        groups=sorted({r['patient_group'] for r in ds.rows});contexts={g:[i for i,r in enumerate(ds.rows) if r['patient_group']==g] for g in groups}
        targets=torch.tensor([r['target'] for r in ds.rows],device='cuda')
        counts=torch.bincount(targets,minlength=2).float();weights=(len(targets)/(2*counts)).to('cuda')
        if bool((counts==0).any()):raise ValueError('Both real observed classes required')
        for mode in ('fine_baseline','ezsp_cost_only'):
            budget.check();torch.manual_seed(42)
            if mode=='fine_baseline':
                with installed('stride4'):local=V1LocalEncoder(base)
            else:local=DiagnosticEZSPEncoder(base,profile,resource_budget=budget)
            master=PromptGraphModel(cfg,base,{},local_encoder=local).cpu()
            original_hash=tensor_hash(master)
            runs=[];report['cost_only'][mode]={'runs':runs,'initial_parameter_sha256':original_hash}
            for repeat in range(a.updates):
                budget.check();net=copy.deepcopy(master).cuda()
                if mode!='fine_baseline':net.local.resource_budget=budget;net.local.timing_enabled=True
                torch.manual_seed(42);net.eval()
                with torch.no_grad():
                    memory_embeddings,memory_seconds=timed(lambda:net.local(batch))
                memory={'embeddings':memory_embeddings.detach(),'record_ids':[r['id'] for r in ds.rows],
                    'patient_groups':groups,'donor_groups':[r['donor_group'] for r in ds.rows],
                    'owners':torch.tensor([groups.index(r['patient_group']) for r in ds.rows],device='cuda'),'classes':targets}
                plans,plan_seconds=timed(lambda:{g:net.fit_support_clusters(*grouped_support(memory,g)) for g in groups})
                optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
                net.train();torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();budget.check()
                start=perf_counter()
                (loss,terms),forward_seconds=timed(lambda:update_loss(net,batch,ds,memory,contexts,plans,weights))
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite diagnostic loss')
                budget.check();_,backward_seconds=timed(loss.backward)
                _,clip_seconds=timed(lambda:torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True))
                budget.check();_,optimizer_seconds=timed(optimizer.step)
                run={'repeat':repeat,'status':'DIAGNOSTIC_UPDATE_ONLY_NOT_ADMITTED','loss':float(loss.detach()),
                    'forward_seconds':forward_seconds,'backward_seconds':backward_seconds,'clip_seconds':clip_seconds,'optimizer_seconds':optimizer_seconds,
                    'whole_update_seconds':perf_counter()-start,'memory_refresh_seconds':memory_seconds,'plan_seconds':plan_seconds,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'terms':terms}
                if mode!='fine_baseline':
                    run['stages']=copy.deepcopy(net.local.last_audit)
                    run['counterfactual_only']='Scale2 here is cost-only, not a result for strict pairs whose scale2 remains NOT_RUN'
                    run['partition_only_seconds']=sum(r['official_merge_seconds'] for level in ('scale1','scale2') for r in run['stages'][level]['roles'].values())
                runs.append(run);budget.check();print(mode,repeat,run['whole_update_seconds'],flush=True)
                del net,optimizer,loss,memory_embeddings,memory,plans;gc.collect();torch.cuda.empty_cache()
            if tensor_hash(master)!=original_hash:raise RuntimeError('Cloned diagnostic mutated master parameters')
            del master,local;gc.collect();torch.cuda.empty_cache()
        report['status']='DIAGNOSTICS_COMPLETE_NOT_PRODUCTION_ADMITTED'
    except Exception as exc:
        report['status']='DIAGNOSTIC_ERROR';report['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        torch.cuda.set_per_process_memory_fraction(previous_fraction)
        report['source_unchanged_during_run']=report['adapter_hashes']=={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'l0_ezsp').rglob('*.py')}
        report['rss_bytes_at_end']=psutil.Process().memory_info().rss
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
