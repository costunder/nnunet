"""Local read-only learning diagnosis: full metadata and explicit actual-CT DEBUG.

No production updates, partitions, checkpoints or claims of trained quality.
"""
import argparse,copy,hashlib,json,math,sys,time,zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
import torch
from torch.nn import functional as F
from l0_regions.training import make_model,Loader,metadata,hash_state
from l0_regions.training_data import RegionDataset,Budget,write_new
from l0_regions.fine_graph import FineDataset,FineLoader
from hiercp_v222.v1_training import groups
from hiercp_v222.v1_local import support_for_recipient
from tools.v22_rank_objective import RankingContext,forward_loss,configuration


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_audit(index):
    m=json.loads(index.read_text(encoding='utf8'));result={}
    for part in ('inner_train','inner_val'):
        rows=[r for r in m['records'] if r['case_id'] in m['split'][part]]
        by={c:[r for r in rows if r['case_id']==c] for c in m['split'][part]}
        cases=[];expected_recall={k:0. for k in (1,5,10)};mrr=[];mismatches=total_pairs=0
        for c,rs in by.items():
            p=sum(r['target'] for r in rs);n=len(rs);q=1.;expected=0.
            if p:
                for rank in range(1,n-p+2):
                    probability=q*p/(n-rank+1);expected+=probability/rank
                    q*=max(0,n-p-rank+1)/(n-rank+1)
                mrr.append(expected)
            for k in expected_recall:expected_recall[k]+=p*min(k,n)/n
            pos=Counter((r['donor_case_id'],r['donor_component']) for r in rs if r['target'])
            neg=Counter((r['donor_case_id'],r['donor_component']) for r in rs if not r['target'])
            pairs=p*(n-p);same=sum(count*neg[d] for d,count in pos.items())
            mismatches+=pairs-same;total_pairs+=pairs
            cases.append(dict(case=c,n=n,positive=p,random_first_positive_mrr=expected if p else None,
                              donor_mismatched_pairs=pairs-same,ranking_pairs=pairs))
        order=list(groups(SimpleNamespace(rows=rows),32,42,0));batches=[]
        positive_total=sum(r['target'] for r in rows)
        effective_positive=effective_negative=0.
        weights={0:len(rows)/(2*(len(rows)-positive_total)),1:len(rows)/(2*positive_total)}
        for ids in order:
            positives=sum(rows[i]['target'] for i in ids);negative=len(ids)-positives
            denominator=weights[1]*positives+weights[0]*negative
            effective_positive+=weights[1]*positives/denominator
            effective_negative+=weights[0]*negative/denominator
            batches.append(dict(count=len(ids),positive=positives))
        result[part]=dict(records=len(rows),positives=positive_total,cases=cases,batches=batches,
            all_negative_batches=sum(b['positive']==0 for b in batches),batches_total=len(batches),
            zero_positive_cases=sum(c['positive']==0 for c in cases),
            weighted_CE_positive_mass_per_epoch=effective_positive,weighted_CE_negative_mass_per_epoch=effective_negative,
            weighting_scope='sum of per-batch weighted-mean coefficients, not measured model gradient magnitude',
            random_permutation_expectation=dict(mrr=sum(mrr)/len(mrr),**{f'recall_at_{k}':v/positive_total for k,v in expected_recall.items()}),
            donor_mismatch_ratio=mismatches/total_pairs,ranking_pairs=total_pairs)
    result['source']=dict(path=str(index),sha256=sha(index),scope='local full paired inventory; server bounds/weights were not downloaded')
    return result


def dispersion(x):
    x=x.detach().float();norm=F.normalize(x,dim=-1);similarity=norm@norm.T
    mask=~torch.eye(len(x),device=x.device,dtype=torch.bool)
    return dict(mean_off_diagonal_cosine=float(similarity[mask].mean()),
                centered_energy=float((x-x.mean(0)).square().sum()/len(x)))


def layer_probe(net,embedding,support):
    state=net.prepare_support(*support);q=embedding;out=[dict(stage='L0',**dispersion(q))]
    for i,(layer,labels) in enumerate(zip(net.l1,state['histories']),1):
        src=torch.arange(len(labels),device=q.device).repeat(len(q))
        dst=torch.arange(len(q),device=q.device).repeat_interleave(len(labels))
        q=layer.messages(labels,q,src,dst,q.new_zeros(len(src),2));out.append(dict(stage=f'L1_{i}',**dispersion(q)))
    prediction=net.predict_embeddings(embedding,state)
    return dict(stages=out,scores=(prediction['logits'][:,1]-prediction['logits'][:,0]).tolist(),
                alignment_loss=float(state['alignment_loss']),cluster_audit=state['cluster_plan']['audit'])


def gradients(net,query,support,mem,ds):
    ids=query.indices.tolist();context=RankingContext(ds,mem)
    classes=mem['classes'];counts=torch.bincount(classes,minlength=2).float();weights=counts.sum()/(2*counts)
    plan=net.fit_support_clusters(*support)
    _,terms=forward_loss(net,query,support,plan,classes[ids],weights,context,configuration(),indices=ids)
    names=[n for n,p in net.named_parameters()];params=tuple(net.parameters())
    rows={};vectors={};keys=('ranking_loss','observation_auxiliary_loss','alignment_loss')
    modules={'CNN':'local.core.dense_encoder','SAGE':'local.core.blocks','L1':'l1','L2':'l2'}
    for key in keys:
        gs=torch.autograd.grad(terms[key],params,retain_graph=True,allow_unused=True)
        vectors[key]={label:torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for n,p,g in zip(names,params,gs) if n.startswith(prefix)]) for label,prefix in modules.items()}
        rows[key]=dict(value=float(terms[key]),norm={k:float(v.norm()) for k,v in vectors[key].items()})
    cosines={}
    for left,right in ((keys[0],keys[1]),(keys[0],keys[2]),(keys[1],keys[2])):
        cosines[left+' vs '+right]={}
        for module in modules:
            a,b=vectors[left][module],vectors[right][module];den=a.norm()*b.norm()
            cosines[left+' vs '+right][module]=float(torch.dot(a,b)/den) if den>0 else None
    return dict(losses=rows,gradient_cosines=cosines,query_ids=ids,
        scope='eval-mode actual CT DEBUG two-row query; diagnostic gradients only, no optimizer update')


def feature_loss(net,fine,coarse):
    from hiercp_v22.schema import LOCAL_NODE_TYPES,SOURCE_LOCAL_NODE_TYPES
    from hiercp_v222.deterministic_sampling import sample_nodes
    from tools.v222_review_contracts import input_grid_to_feature_grid
    src,tgt=net.local.core.encode_dense_maps(fine.source_patches,fine.source_index,fine.target_patches)
    result=[]
    for role in LOCAL_NODE_TYPES:
        f=coarse.fine[role];node=fine.graph[role]
        if not torch.equal(f['grid'],node.grid) or not torch.equal(f['owner'],node.batch):raise ValueError('Fine/region node identity mismatch')
        shape,jump,origin=net.local.feature_lattice
        x=sample_nodes(src if role in SOURCE_LOCAL_NODE_TYPES else tgt,input_grid_to_feature_grid(node.grid,feature_shape=shape,stride=jump,origin=origin),node.batch)
        parent=f['parent'];k=len(coarse.regions[0][role].mass)
        mean=x.new_zeros((k,x.shape[1])).index_add_(0,parent,x)/coarse.regions[0][role].mass[:,None]
        for owner in range(len(fine)):
            mask=node.batch==owner;xx=x[mask];reconstructed=mean[parent[mask]]
            if not len(xx):continue
            total=(xx-xx.mean(0)).square().sum();lost=(xx-reconstructed).square().sum()
            result.append(dict(pair=owner,role=role,fine_nodes=len(xx),regions=int(parent[mask].unique().numel()),
                within_role_feature_variation_lost=float(lost/total) if total>0 else None))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('full-index','region-cache','fine-cache','debug-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);budget=Budget(8*2**30,16*2**30)
    report=dict(debug=True,training_started=False,weights_updated=False,server_weights_available=False,
        resources=dict(gpu=torch.cuda.get_device_name(),free_total_cuda=torch.cuda.mem_get_info(),cpu=psutil.cpu_count(),ram_available=psutil.virtual_memory().available,workers=8),
        inventory=metadata_audit(a.full_index))
    z=zipfile.ZipFile(ROOT/'versions/v1/pipeline_v1_source.zip');arch=json.loads(z.read('config/train.json'))
    report['v1_archive']=dict(sha256=sha(ROOT/'versions/v1/pipeline_v1_source.zip'),cache=arch['cache'],training=arch['training'])
    write_new(a.output/'metadata.json',report)
    rd=RegionDataset(a.region_cache,'inner_train',True,'research-report');fd=FineDataset(a.region_cache,'inner_train',True,'research-report',a.fine_cache)
    if len(rd)!=8:raise ValueError('Explicit preserved actual CT DEBUG eight-pair fixture required')
    rloader=Loader(rd,8,4*2**30);floader=FineLoader(fd,8,4*2**30)
    r=rloader.get(list(range(8))).to('cuda');f=floader.get(list(range(8))).to('cuda')
    netr=make_model(rd,budget,True,'retained');netf=make_model(fd,budget,True,'retained')
    checkpoint=torch.load(a.debug_checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['identity']['debug'] is not True or checkpoint['content_sha256']!=hash_state({k:v for k,v in checkpoint.items() if k!='content_sha256'}):raise ValueError('Verified DEBUG weights required')
    results={}
    for snapshot in ('same_initialization','four_update_DEBUG_weights'):
        if snapshot!='same_initialization':
            for net in (netr,netf):
                weights=copy.deepcopy(checkpoint['model']);weights['local._extra_state']=net.local.get_extra_state()
                net.load_state_dict(weights)
        for key,v in netr.state_dict().items():
            if torch.is_tensor(v) and not torch.equal(v,netf.state_dict()[key]):raise ValueError('A/B numeric weights differ')
        netr.eval();netf.eval();result={}
        for name,net,batch,ds,loader in (('coarse',netr,r,rd,rloader),('fine',netf,f,fd,floader)):
            began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                embed=net.local(batch);mem=metadata(ds,embed.detach())
                support=support_for_recipient(mem,ds.rows[0]['patient_group'])
                result[name]=layer_probe(net,embed[:2],support)
            query=loader.get([0,1]).to('cuda')
            result[name]['gradients']=gradients(net,query,support,mem,ds)
            torch.cuda.synchronize();budget.check()
            result[name]['seconds_including_extra_probes']=time.perf_counter()-began
            result[name]['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
            result[name]['l0_embeddings']=embed.detach().cpu().tolist()
        with torch.no_grad():result['feature_variation_removed']=feature_loss(netf,f,r)
        results[snapshot]=result;print('DIAGNOSTIC',snapshot,'complete',flush=True)
    report['gpu']=results
    report['counts']=dict(pairs=8,fine_nodes=int(f.graph.num_nodes),fine_edges=int(f.graph.num_edges),
                          coarse_nodes=int(r.graphs[0].num_nodes),coarse_edges=int(r.graphs[0].num_edges))
    report['debug_checkpoint']=dict(path=str(a.debug_checkpoint),sha256=sha(a.debug_checkpoint),step=checkpoint['state']['step'])
    # Exact demonstration of weighted mean cancellation on a pure-class batch.
    logits=torch.tensor([[.2,-.1],[.5,-.4]],device='cuda',requires_grad=True);labels=torch.zeros(2,device='cuda',dtype=torch.long)
    unweighted=F.cross_entropy(logits,labels);weighted=F.cross_entropy(logits,labels,weight=torch.tensor([.52,10.7],device='cuda'))
    g1=torch.autograd.grad(unweighted,logits,retain_graph=True)[0];g2=torch.autograd.grad(weighted,logits)[0]
    torch.testing.assert_close(g1,g2)
    report['CE_pure_negative_cancellation']=dict(synthetic_operator_probe=True,max_gradient_delta=float((g1-g2).abs().max()))
    write_new(a.output/'report.json',report)
    print(json.dumps(dict(output=str(a.output/'report.json'),counts=report['counts'],full_training=False),indent=2))


if __name__=='__main__':main()
