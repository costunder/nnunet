"""Read-only diagnosis of the saved r8 partitions. No merge, update or admission change."""
import argparse, base64, gzip, hashlib, json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import psutil
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from l0_regions.training_data import dataset, reference_from_checkpoint
from l0_regions.materialization import load_pairs
from l0_regions.data import validate_item
from tools.v22_artifacts import tree_hash
from hiercp_v22.schema import LOCAL_NODE_TYPES as NT, LOCAL_EDGE_TYPES as ET, SOURCE_LOCAL_NODE_TYPES
from hiercp_v222.deterministic_sampling import sample_nodes
from tools.v222_review_contracts import cnn_lattice,input_grid_to_feature_grid

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2)
def mean(x,p,k):
    counts=np.bincount(p,minlength=k); out=np.zeros((k,x.shape[1]),np.float64)
    np.add.at(out,p,x)
    return out/counts[:,None]
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def unique_edges(e,groups):
    keep=(e[0]!=e[1]) & (groups[e[0]]==groups[e[1]])
    return np.unique(np.sort(e[:,keep],axis=0).T,axis=0).T
def boundary(e,p):
    q=p[e];q=q[:,q[0]!=q[1]]
    pairs,counts=np.unique(np.sort(q,axis=0).T,axis=0,return_counts=True)
    return pairs.T,counts
def energy(x,m,e,w,reg):
    a,b=e
    return m[a]*m[b]/(m[a]+m[b])*np.square(x[a]-x[b]).sum(1)-reg*w
def numbers(x):
    return dict(count=len(x),min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None,
                admissible=int((x<=0).sum()),near_zero=int((np.abs(x)<1e-6).sum()))
def cc(n,e):
    return connected_components(coo_matrix((np.ones(e.shape[1]),(e[0],e[1])),shape=(n,n)).tocsr(),directed=False)
def arr(t):return t.detach().cpu().numpy()
def encoded(v):return base64.b64encode(gzip.compress(json.dumps(v,separators=(',',':'),ensure_ascii=False).encode(),mtime=0)).decode()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inline',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.cache.parent;m=json.loads(args.cache.read_text());before=sha(args.cache)
    entries=m['partitions']['inner_train'];assert len(entries)==8 and m['debug'] is True
    started=time.perf_counter();torch.set_num_threads(8)
    resources=dict(cpu_logical=psutil.cpu_count(),cpu_affinity=psutil.Process().cpu_affinity(),available_ram_bytes=psutil.virtual_memory().available,
        gpu=torch.cuda.get_device_name(0),gpu_free_total_bytes=torch.cuda.mem_get_info(),workers=8,physical_pair_batch=8,
        scope='read-only frozen CNN forward and saved partition analysis; no merge or optimizer')
    with ThreadPoolExecutor(max_workers=8) as pool:
        items=list(pool.map(lambda e:torch.load(root/e['file'],weights_only=False,map_location='cpu'),entries))
    for it in items:validate_item(it)
    raw=Path(m['original_cache']);ds=dataset(raw,'inner_train',True)
    fine=load_pairs(ds,list(range(8)),workers=8,epoch=m['view_epoch'],cache_path=raw)
    assert all(it['fine_graph_evidence']['materialized_batch_sha256']==fine.materialized_sha256 for it in items),'Exact saved fine view differs'
    assert all(it['fine_graph_evidence']['fine_edges_sha256']==tree_hash(fine.graph.edge_index_dict) for it in items)
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    net=reference_from_checkpoint(args.checkpoint,True).eval().requires_grad_(False).cuda()
    assert tree_hash(net.dense_encoder.state_dict())==m['cnn_sha256']
    shape,jump,origin=cnn_lattice(net.dense_encoder);gpu=fine.to('cuda');features={}
    with torch.no_grad():
        src,tgt=net.encode_dense_maps(gpu.source_patches,gpu.source_index,gpu.target_patches)
        for role in NT:
            node=gpu.graph[role]
            grid=input_grid_to_feature_grid(node.grid,feature_shape=shape,stride=jump,origin=origin)
            features[role]=arr(sample_nodes(src if role in SOURCE_LOCAL_NODE_TYPES else tgt,grid,node.batch))
    resources['peak_allocated_bytes']=torch.cuda.max_memory_allocated();del gpu,src,tgt,net
    report=dict(scope='Actual 8 saved DEBUG pairs; not full-cohort/clinical/production validation',resources=resources,
        region_index=str(args.cache.resolve()),region_index_sha256=before,raw_cache_sha256=sha(raw),exact_fine_materialization_verified=True,
        cnn_source=str(args.checkpoint.resolve()),cnn_checkpoint_sha256=sha(args.checkpoint),cnn_state_sha256=m['cnn_sha256'],
        cnn_evidence='existing DEBUG optimized snapshot; partition quality unvalidated',
        partition_gat_weights='NONE: both partitions use frozen CNN32, no GAT or SAGE output',
        configuration_unchanged=True,training_started=False,pairs=[])
    view=dict(records=[],summary=[],scope='r8 저장 partition · 실제 CT DEBUG 8pair · 병합 재실행 없음')
    for i,(entry,it) in enumerate(zip(entries,items)):
        row=dict(record=it['binding']['record_id'],donor=it['binding']['donor_case_id'],fine_nodes=entry['fine_nodes'],fine_edges=0,
            scale_nodes=entry['region_nodes'],scale_edges=entry['region_edges'],profile_exceeded=it['profile_exceeded'],roles={})
        vis=dict(id=row['record'],donor=row['donor'],roles={},counts=[],patches={})
        for key in ('source','target'):
            patch=arr(it[key+'_patch'][0]);lo,hi=np.percentile(patch,[1,99]);hi=max(hi,lo+1e-6)
            vis['patches'][key]=dict(voxels=np.rint(np.clip((patch-lo)/(hi-lo),0,1)*255).astype(np.uint8).ravel().tolist(),window=[float(lo),float(hi)])
        for rel in ET:
            e=fine.graph[rel].edge_index
            row['fine_edges']+=int((fine.graph[rel[0]].batch[e[0]]==i).sum())
        for role in NT:
            f=it['fine'][role];pos=arr(f['pos_mm']);p1=arr(f['parent']);p12=arr(it['scales'][0][role]['parent']);p2=p12[p1]
            owner=arr(fine.graph[role].batch);mask=owner==i
            assert np.array_equal(arr(fine.sidecar[role]['pos_mm'])[mask],pos)
            assert np.array_equal(arr(fine.graph[role].grid)[mask],arr(f['grid']))
            same=next(e for e in ET if e[0]==role and e[2]==role)
            e=arr(fine.graph[same].edge_index);e=e[:,owner[e[0]]==i]-int(np.flatnonzero(mask)[0]) if mask.any() else np.empty((2,0),int)
            groups=arr(f['shell']);ue=unique_edges(e,groups)
            k1=len(it['scales'][0][role]['mass']);k2=len(it['scales'][1][role]['mass'])
            x=features[role][mask].astype(np.float64);rawmean=mean(x,p1,k1);m1=np.bincount(p1,minlength=k1)
            ce,cw=boundary(ue,p1);u1=norm(rawmean)
            energies=energy(u1,m1,ce,cw,.02)
            uraw=norm(x);u_mean=mean(uraw,p1,k1)
            term1=energy(u_mean,m1,ce,cw,.02)
            details=dict(fine_nodes=len(pos),same_role_directed_edges=[e.shape[1],it['edges'][0][same].shape[1],it['edges'][1][same].shape[1]],
                stage2_start_energy_fp64_check=numbers(energies),stage1_return_energy_fp64_check=numbers(term1),
                fine_clusters_connected=True,scales=[],audit=[it['audit'][f'scale{s}']['roles'][role] for s in (1,2)])
            vr=dict(pos=pos.round(4).tolist(),grid=arr(f['grid']).round(7).tolist(),shell=groups.tolist(),parents=[p1.tolist(),p2.tolist()],
                nodes=[],edges=[e.T.tolist(),arr(it['edges'][0][same]).T.tolist(),arr(it['edges'][1][same]).T.tolist()],clusters=[])
            for level,p in enumerate((p1,p2)):
                k=len(it['scales'][level][role]['mass']);mass=np.bincount(p,minlength=k)
                assert np.array_equal(mass,arr(it['scales'][level][role]['mass']))
                internal=ue[:,p[ue[0]]==p[ue[1]]];ncc,labels=cc(len(pos),internal)
                assert ncc==k,'Disconnected saved cluster'
                lower=arr(it['scales'][level][role]['lower']);upper=arr(it['scales'][level][role]['upper'])
                diag=np.linalg.norm(upper-lower,axis=1);centers=mean(pos,p,k)
                cumulative_mean=mean(uraw,p,k);cumulative_var=np.bincount(p,weights=np.square(uraw-cumulative_mean[p]).sum(1),minlength=k)/mass
                limit=it['binding']['profile']['diagnostic_profile_from_original_NOT_VALIDATED']['bbox_context_diagonal_mm' if 'context' in role else 'bbox_surface_diagonal_mm'][level]
                clusters=[]
                for c in range(k):
                    selected=pos[p==c];assert np.allclose(selected.min(0),lower[c]) and np.allclose(selected.max(0),upper[c])
                    clusters.append(dict(id=c,mass=int(mass[c]),shell=int(groups[p==c][0]),bbox_mm=float(diag[c]),
                        extent_mm=(upper[c]-lower[c]).tolist(),cumulative_fine_variance=float(cumulative_var[c]),
                        lower=lower[c].tolist(),upper=upper[c].tolist(),bbox_exceeded=bool(diag[c]>limit),
                        stage1_children=1 if level==0 else int((p12==c).sum())))
                auditgroups=[g for g in details['audit'][level]['group_diagnostics'] if g['owner']==i]
                stats=dict(nodes=k,connected_clusters=ncc,bbox_excess_clusters=int((diag>limit).sum()),
                    bbox_excess_fine_fraction=float(mass[diag>limit].sum()/max(len(pos),1)),mass_quantiles=np.quantile(mass,[0,.25,.5,.75,1]).tolist() if len(mass) else [],
                    unchanged_s1_clusters=int((np.bincount(p12,minlength=k2)==1).sum()) if level==1 else None,
                    group_diagnostics=auditgroups,clusters=clusters)
                details['scales'].append(stats);vr['nodes'].append(centers.round(4).tolist());vr['clusters'].append(clusters)
            vr['energies']=numbers(energies);vis['roles'][role]=vr;row['roles'][role]=details
        vis['counts']=[[row['fine_nodes'],row['fine_edges']],*[list(v) for v in zip(row['scale_nodes'],row['scale_edges'])]]
        report['pairs'].append(row);view['records'].append(vis)
        view['summary'].append(dict(id=row['record'],counts=vis['counts']))
        print('verified',row['record'],vis['counts'],flush=True)
    assert sha(args.cache)==before
    report['elapsed_seconds']=time.perf_counter()-started
    report['aggregate']=dict(fine_nodes=sum(r['fine_nodes'] for r in report['pairs']),fine_edges=sum(r['fine_edges'] for r in report['pairs']),
        scale_nodes=[sum(r['scale_nodes'][j] for r in report['pairs']) for j in (0,1)],
        scale_edges=[sum(r['scale_edges'][j] for r in report['pairs']) for j in (0,1)],
        stage2_start_eligible_edges=sum(d['stage2_start_energy_fp64_check']['admissible'] for r in report['pairs'] for d in r['roles'].values()),
        stage2_start_total_edges=sum(d['stage2_start_energy_fp64_check']['count'] for r in report['pairs'] for d in r['roles'].values()))
    write(args.output/'report.json',report)
    # Display geometry is rounded only in the visual; validation used exact stored tensors.
    template=(ROOT/'tools/fixed-region-inspector.template.html').read_text(encoding='utf-8')
    full=template.replace('__PAYLOAD__',encoded(view))
    css=':root{color-scheme:light dark;--background:light-dark(#fff,#17191c);--foreground:light-dark(#202328,#edf0f4);--border:light-dark(#c5cbd1,#515961);--muted-foreground:light-dark(#535b64,#b4bdc6);--primary:light-dark(#292f36,#dde5ec);--primary-foreground:light-dark(#fff,#191c20);--viz-series-1:light-dark(#16727e,#58d1e1);--viz-series-2:light-dark(#b15900,#ffc477);--viz-series-3:light-dark(#723dae,#ba93ec)}body{font:14px system-ui;margin:24px;color:var(--foreground);background:var(--background)}select,input,button{font:inherit;color:inherit;background:var(--background);padding:6px}h3{font-weight:500}.viz-controls,.viz-row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}.form-label{display:grid;gap:5px}.text-small{font-size:12px}.btn{border:1px solid var(--border);padding:6px}.table{width:100%;border-collapse:collapse}.table td,.table th{text-align:left;padding:6px;border-bottom:1px solid var(--border)}'
    with (args.output/'inspector.html').open('x',encoding='utf-8') as f:f.write('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 고정 partition 검사</title><style>'+css+'</style>'+full+'</html>')
    # Inline focuses on the actual pair containing the widest recipient-context cluster.
    best=max(range(8),key=lambda i:max(c['bbox_mm'] for c in view['records'][i]['roles']['target_context']['clusters'][0]))
    rec=view['records'][best];rec=dict(rec,roles={'target_context':rec['roles']['target_context']},patches={'target':rec['patches']['target']})
    compact=dict(view,records=[rec],scope='실제 8pair 집계 · 아래 그래프는 최대 recipient-context bbox pair')
    fragment=template.replace('__PAYLOAD__',encoded(compact))
    if len(fragment.encode())>=1_000_000:raise ValueError('Inline exceeds size budget; no silent downsample')
    with args.inline.open('x',encoding='utf-8') as f:f.write(fragment)
    print(json.dumps(dict(aggregate=report['aggregate'],inline_bytes=len(fragment.encode()),full_html_bytes=(args.output/'inspector.html').stat().st_size,selected=rec['id']),ensure_ascii=False))

if __name__=='__main__':main()
