"""Read-only, GPU learning diagnosis. No optimizer, training or checkpoint writes.

Explicit diagnostic case count; all candidates in each selected case are retained.
Measures trained representations and support sensitivity, not a new trained model.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def read_history(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return {int(r['epoch']):r for r in csv.DictReader(f)}

def history_matches(path,reference):
    actual=read_history(path)
    fields={'ranking_pairwise_loss':4,'ranking_mrr':3,'ranking_recall_at_1':3,
            'ranking_recall_at_5':3,'ranking_recall_at_10':3}
    return all(e in actual and all(f'{float(actual[e][k]):.{d}f}'==f'{float(r[k]):.{d}f}'
               for k,d in fields.items()) for e,r in reference.items())

def run_root(directory):
    directory=Path(directory).resolve()
    for p in (directory,*directory.parents):
        if (p/'experiment.json').is_file():return p
        if (p/'inventory/index.json').is_file():return p
    raise ValueError(f'No bound local CNN inventory above {directory}')

def find_run(medical,reference):
    roots=[medical/'experiments']+[p/'work' for p in medical.glob('HierCP*') if p.is_dir()]
    found=set()
    for root in roots:
        if not root.is_dir():continue
        for directory,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in {'cache','paired_cache','graphs','datasets','.git',
                                                'inventory','raw','imagesTr','labelsTr'}]
            if 'validation_history.csv' in files and history_matches(Path(directory)/'validation_history.csv',reference):
                found.add(run_root(directory))
    if len(found)!=1:
        raise ValueError(f'Expected one experiment matching the provided epoch history, found {len(found)}: '
                         +', '.join(map(str,sorted(found))))
    return found.pop()

def checkpoint_for(root):
    manifest=root/'experiment.json'
    if manifest.is_file():
        from tools.run_local_cnn_experiment import select_checkpoint
        cp,_=select_checkpoint(root,json.loads(manifest.read_text()))
        if cp is None:raise ValueError('Experiment has no optimization checkpoint')
        return cp
    cp=root/'training/checkpoint_latest.pt'
    if not cp.is_file():raise FileNotFoundError(cp)
    return cp

def spread(x):
    import torch
    from torch.nn import functional as F
    x=x.detach().float();z=F.normalize(x,dim=1)
    return dict(centered_energy=float((x-x.mean(0)).square().sum(1).mean()),
                normalized_centered_energy=float((z-z.mean(0)).square().sum(1).mean()),
                mean_norm=float(x.norm(dim=1).mean()),finite=bool(torch.isfinite(x).all()))

def score_summary(scores,truth):
    import torch
    from torch.nn import functional as F
    delta=scores[truth==1,None]-scores[None,truth==0]
    if not delta.numel():raise ValueError('Diagnostic requires both observed and unobserved candidates')
    return dict(score_std=float(scores.std(unbiased=False)),score_min=float(scores.min()),
        score_max=float(scores.max()),mean_positive_minus_unobserved=float(delta.mean()),
        pair_win_rate=float((delta>0).float().mean()),exact_tie_rate=float((delta==0).float().mean()),
        mean_pairwise_loss=float(F.softplus(-delta).mean()),comparisons=delta.numel())

def trace_head(net,embeddings,support,batch):
    import torch
    from hiercp_v222.clustering import prototype_logits,live_prototypes
    state=net.prepare_support(*support)
    stages=[embeddings];q=embeddings
    for layer,labels in zip(net.l1,state['histories']):
        parts=[]
        for value in q.split(batch):
            src=torch.arange(len(labels),device=q.device).repeat(len(value))
            dst=torch.arange(len(value),device=q.device).repeat_interleave(len(labels))
            parts.append(layer.messages(labels,value,src,dst,value.new_zeros(len(src),2)))
        q=torch.cat(parts);stages.append(q)
    logits=prototype_logits(q,state['labels'],state['cluster_plan'],net.temperature)
    prototypes=live_prototypes(state['labels'],state['cluster_plan'])
    cls=state['cluster_plan']['prototype_classes']
    cross=prototypes[cls==1]@prototypes[cls==0].T
    report=dict(stages=[dict(stage='L0' if i==0 else f'L1_{i}',**spread(x)) for i,x in enumerate(stages)],
        prototype_cross_class_cosine_min=float(cross.min()),prototype_cross_class_cosine_max=float(cross.max()),
        prototype_cross_class_cosine_mean=float(cross.mean()),cluster_audit=state['cluster_plan']['audit'],
        local_labels=spread(state['local_labels'].flatten(0,1)),aligned_labels=spread(state['labels'].flatten(0,1)),
        support_records=len(support[0]),support_patients=int(support[1].max())+1)
    return logits[:,1]-logits[:,0],report

def gradient_probe(net,embedding,support,rows,indices,context,query=None):
    """Actual objective coefficients; optional real CNN backward, never an update."""
    import torch
    from torch.nn import functional as F
    from l0_regions.donor_learning import configuration
    net.eval();q=embedding.detach().requires_grad_(True) if query is None else net.local(query)
    state=net.prepare_support(*support);out=net.predict_embeddings(q,state)
    logits=out['logits'].float();score=logits[:,1]-logits[:,0]
    target=torch.tensor([rows[i]['target'] for i in indices],device=q.device)
    delta=score[target==1,None]-score[None,target==0]
    coefficients=logits.new_tensor([context.steps/(2*context.counts[rows[i]['target']]*context.uses[i]) for i in indices])
    settings=configuration()
    terms={'ranking':F.softplus(-delta).sum()*context.steps/context.pairs*settings['ranking_weight'],
           'observation_ce':(F.cross_entropy(logits,target,reduction='none')*coefficients).sum()*settings['observation_auxiliary_weight'],
           'alignment':out['alignment_loss']*out['alignment_loss_weight']}
    named=[(n,p) for n,p in net.named_parameters() if query is not None or not n.startswith('local.')]
    vectors={};reports={}
    for key,value in terms.items():
        gradients=torch.autograd.grad(value,[q]+[p for _,p in named],retain_graph=True,allow_unused=True)
        packed={'L0_output':gradients[0] if gradients[0] is not None else torch.zeros_like(q)}
        modules=[('L1',('l1.','label_seed')),('L2',('l2.','l2_updates.'))]
        if query is not None:modules=[('CNN',('local.cnn.',)),('readout_fusion',('local.project.','local.fuse.'))]+modules
        for label,prefixes in modules:
            packed[label]=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                for (n,p),g in zip(named,gradients[1:]) if n.startswith(prefixes)])
        vectors[key]=packed
        reports[key]=dict(weighted_loss=float(value.detach()),gradient_norm={k:float(v.norm()) for k,v in packed.items()})
    cos={}
    for a,b in [('ranking','observation_ce'),('ranking','alignment')]:
        cos[a+' vs '+b]={}
        for name in vectors[a]:
            x=vectors[a][name].flatten();y=vectors[b][name].flatten();den=x.norm()*y.norm()
            cos[a+' vs '+b][name]=float(torch.dot(x,y)/den) if den>0 else None
    return dict(losses=reports,gradient_cosines=cos,query_rows=len(indices),cnn_backward=query is not None,
        scope='eval-mode derivative with actual epoch loss coefficients; no optimizer',
        dropout_note='Dropout disabled to isolate deterministic gradients; not a replay of the stochastic training update')

def diagnose(a):
    import torch
    from tqdm import tqdm
    from l0_local_cnn.data import Dataset,Loader
    from l0_regions.training import make_model,hash_state,load_checkpoint,legacy_groups
    from l0_regions.training_data import source_identity,sha,Budget
    from l0_regions.donor_learning import LiveContext,groups
    from l0_regions.support_episodes import PatientEpisodes
    from hiercp_v222.v1_local import support_for_recipient
    from hiercp_v222.v1_execution import tree_to
    from tools.v22_rank_objective import ranking_metrics
    from tools.v22_candidate_order import record_key
    if a.output.exists():raise FileExistsError(a.output)
    root=run_root(a.run) if a.run else find_run(a.medical_root,read_history(a.match_history))
    cp=checkpoint_for(root);print(f'DIAGNOSTIC ONLY | run={root}\ncheckpoint={cp}',flush=True)
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    if saved['identity']['source']!=source_identity():raise ValueError('Checkpoint runtime source differs; use matching code for diagnosis')
    # Verify this single loaded snapshot, even when another process atomically replaces latest.
    from l0_regions.training import FORMAT
    if saved['format']!=FORMAT or hash_state({k:v for k,v in saved.items() if k!='content_sha256'})!=saved['content_sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    identity=saved['identity'];state=saved['state'];index=root/'inventory/index.json'
    if identity.get('local_cnn') is None or identity['cache_sha256']!=sha(index):raise ValueError('Checkpoint/input is not the bound local CNN experiment')
    if identity['debug'] and not a.allow_debug:raise ValueError('DEBUG checkpoint needs --allow-debug')
    ds=Dataset(index,'inner_train',identity['debug']);val=Dataset(index,'inner_val',identity['debug'])
    torch.set_num_threads(a.workers)
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    total=torch.cuda.get_device_properties(0).total_memory
    if not 0<budget.cuda_bytes<total:raise ValueError('CUDA budget must leave device headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    net=make_model(ds,budget,identity['debug'],'retained');net.load_state_dict(saved['model']);net.eval()
    memory=tree_to(state['memory'],'cuda');batch=state['batch']
    if memory['record_ids']!=[r['id'] for r in ds.rows]:raise ValueError('Saved memory order differs from train inventory')
    counts=identity.get('support_training',{}).get('patients')
    episodes=None
    if counts:
        episodes=PatientEpisodes(ds.rows,list(legacy_groups(ds,batch,ds.meta['config']['seed'],state['epoch'])),
                                 counts,ds.meta['config']['seed'],state['epoch']).bind(memory)
    context=LiveContext(ds,batch)
    loader=Loader(ds,a.workers,int(a.resident_gib*2**30),budget.rss_bytes)
    vl=Loader(val,a.workers,int(a.resident_gib*2**30),budget.rss_bytes,store=loader.store)
    result=dict(diagnostic_only=True,training_started=False,optimizer_updates=0,full_evaluation=False,
        checkpoint_content_sha256=saved['content_sha256'],run=str(root),epoch=state['epoch'],step=state['step'],phase=state['phase'],
        margin_mm=identity['local_cnn']['margin_mm'],gpu=torch.cuda.get_device_name(),physical_batch=batch,
        cases_per_split=a.cases_per_split,memory_scope='saved epoch reference; NOT re-encoded current full cohort',
        model_sha256=hash_state(saved['model']),cases=[])
    for split,data,reader in [('train',ds,loader),('validation',val,vl)]:
        names=sorted({r['case_id'] for r in data.rows if r['target']==1})
        if len(names)<a.cases_per_split:raise ValueError('Requested diagnostic cases exceed available positive cases')
        chosen=[names[(i*len(names))//a.cases_per_split] for i in range(a.cases_per_split)]
        for name in chosen:
            ids=[i for i,r in enumerate(data.rows) if r['case_id']==name];rows=[data.rows[i] for i in ids]
            group=rows[0]['patient_group'];parts=[]
            with torch.no_grad():
                for start in tqdm(range(0,len(ids),batch),desc=f'DIAGNOSTIC {split} {name}'):
                    q=reader.get(ids[start:start+batch]).to('cuda');parts.append(net.local(q).detach());del q
                embeddings=torch.cat(parts);truth=torch.tensor([r['target'] for r in rows],device='cuda')
                full=support_for_recipient(memory,group)
                scores,trace=trace_head(net,embeddings,full,batch)
                metrics,details=ranking_metrics(scores.cpu(),truth.cpu(),[name]*len(ids),candidate_keys=[record_key(r) for r in rows])
                item=dict(split=split,case_id=name,records=len(ids),positives=int(truth.sum()),all_case_candidates_retained=True,
                    full_support=dict(trace=trace,score=score_summary(scores,truth),metrics=metrics),observed_ranks=details[0]['observed_ranks'])
                if split=='train':
                    old=memory['embeddings'][ids]
                    item['memory_drift']=dict(mean_l2=float((old-embeddings).norm(dim=1).mean()),old=spread(old),current=spread(embeddings))
                    episodic=full if episodes is None else episodes.support(group)
                    es,et=trace_head(net,embeddings,episodic,batch)
                    item['training_support_eval']=dict(trace=et,score=score_summary(es,truth),
                        mean_abs_score_change_from_full=float((es-scores).abs().mean()))
            if split=='train':
                tile=next(tile for tile in groups(ds,batch) if data.rows[tile[0]]['case_id']==name
                          and {data.rows[i]['target'] for i in tile}=={0,1})
                position={i:j for j,i in enumerate(ids)}
                q=reader.get(tile).to('cuda')
                item['gradient_probe']=gradient_probe(net,embeddings[[position[i] for i in tile]],episodic,ds.rows,tile,context,query=q)
                del q
            result['cases'].append(item);budget.check()
            print(json.dumps(dict(split=split,records=len(ids),**item['full_support']['score'])),flush=True)
            t=item['full_support']['trace']
            print('Normalized candidate variance: '+', '.join(f"{s['stage']}={s['normalized_centered_energy']:.6g}" for s in t['stages'])
                  +f" | cross-class prototype cosine={t['prototype_cross_class_cosine_mean']:.6g}",flush=True)
            if 'training_support_eval' in item:
                short=item['training_support_eval']['score']
                print(f"Same query/support A-B: episode loss={short['mean_pairwise_loss']:.6g}, full loss={item['full_support']['score']['mean_pairwise_loss']:.6g}",flush=True)
                print('Per-loss gradients: '+json.dumps(item['gradient_probe']['losses']),flush=True)
    result['peak_cuda_gib']=torch.cuda.max_memory_allocated()/2**30
    result['weights_unchanged']=hash_state(net.state_dict())==result['model_sha256']
    if not result['weights_unchanged']:raise AssertionError('Diagnostic changed model weights')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(f'REPORT: {a.output}\nNo training/checkpoint changes. This sampled diagnosis is not full evaluation.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--medical-root',type=Path,default=Path('/home/aicompetition06/Medical'))
    p.add_argument('--match-history',type=Path,default=ROOT/'validation/local_cnn_learning_diagnosis_20261001/server_history.csv')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases-per-split',type=int,required=True,help='Explicit DIAGNOSTIC subset; retains every candidate of selected cases')
    p.add_argument('--workers',type=int,required=True)
    for key in ('cuda-gib','rss-gib','resident-gib'):p.add_argument('--'+key,type=float,required=True)
    p.add_argument('--allow-debug',action='store_true')
    a=p.parse_args()
    if a.cases_per_split<1 or a.workers<2 or not 0<a.resident_gib<a.rss_gib:raise ValueError('Explicit valid diagnostic resources required')
    from tools.local_cnn_device import select
    select(a.gpu)
    diagnose(a)

if __name__=='__main__':main()
