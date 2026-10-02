"""Native CT sparse-feature graph, full physical32 CUDA smoke and 3-D export.

No long training, production checkpoint, ready flag, relabeling, or production
source edit. The CNN is a verified saved *local DEBUG* snapshot, not the trained
server model. Graph/readout parameters are new; results are mechanical evidence.
"""
import argparse
import base64
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from torch import nn
from tqdm import tqdm
from skimage.measure import find_contours, approximate_polygon
from l0_local_cnn.model import LocalBatch
from l0_local_cnn.data import Dataset, CropStore
from l0_regions.resident import signature, check_verified
from l0_regions.donor_learning import LiveContext, forward_loss, configuration
from l0_regions.training import FORMAT, hash_state, make_model
from l0_regions.training_data import Budget, sha, source_identity
from tools.diagnose_local_cnn_learning import checkpoint_for
from tools.local_cnn_interaction_runtime import support_binding
from l0_sparse_feature.model import SparseFeatureProfile, SparseFeatureL0
from hiercp_v222.v1_execution import rng_state, restore_rng


class GeometryBatch(LocalBatch):
    """Explicit original anchors; neither padded midpoint nor shifted anchor."""
    def __init__(self, base, recipient_centers, donor_centers):
        check_verified(base)
        super().__init__(**{k:getattr(base,k) for k in
            ('images','organ','donor','recipient','indices','audit')})
        self.recipient_centers = recipient_centers
        self.donor_centers = donor_centers
        for centers in (recipient_centers, donor_centers):
            if centers.shape != (len(base),3) or centers.device != base.images.device:
                raise ValueError('Every original native anchor must be bound to its pair')
        self.validate()


class GraphLocalAdapter(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch):
        if not isinstance(batch, GeometryBatch):
            raise TypeError('Explicit verified original pair geometry required')
        check_verified(batch)
        return self.encoder(batch, recipient_centers_native=batch.recipient_centers,
            donor_centers_native=batch.donor_centers)


def bound_batch(store, rows, ids):
    cpu = store.batch(rows, ids)
    recipient = torch.tensor([r['center'] for r in rows], dtype=torch.float32, device='cuda')
    donor = torch.tensor(np.stack([store.donor_bounds(r)[0] for r in rows]),
                         dtype=torch.float32, device='cuda')
    return GeometryBatch(cpu.to('cuda'), recipient, donor)


def contours(mask, spacing, origin, step):
    lines=[]
    for z in range(0, mask.shape[2], step):
        if not mask[:,:,z].any():
            continue
        for line in find_contours(np.pad(mask[:,:,z],1).astype(float), .5):
            line=approximate_polygon(line, tolerance=.6)-1
            points=np.column_stack((line, np.full(len(line),z)))
            lines.append(np.round((points+origin)*spacing,2).tolist())
    return lines


def anatomy(raw):
    lo,hi=raw['lo'],raw['hi']
    sl=tuple(slice(a,b) for a,b in zip(lo,hi))
    return dict(organ_surface=contours(raw['organ'][sl],raw['spacing'],lo,3),
        tumor_surface=contours(raw['lab'][sl]==2,raw['spacing'],lo,1))


def graph_scene(graph, scene_index, batch, raw, case, cached_anatomy):
    """Export precisely the nodes and adjacency that went through message passing."""
    alive=graph['node_mask'][scene_index].detach().cpu().bool()
    ids=torch.where(alive)[0].tolist()
    positions=graph['xyz_mm'][scene_index].detach().cpu()
    native=graph['xyz_native'][scene_index].detach().cpu()
    roles=graph['role'][scene_index].detach().cpu().long()
    features=graph['features'][scene_index].detach().cpu().float()
    adjacency=graph['adjacency'][scene_index].detach().cpu().bool()
    kinds=graph['edge_kind'][scene_index].detach().cpu().long()
    lookup={old:new for new,old in enumerate(ids)}
    names=('query','near','mid','wide')
    center=positions[0]
    f=torch.nn.functional.normalize(features,dim=-1)
    nodes=[]
    for i in ids:
        abstract=int(roles[i])==0
        p=native[i].round().long().numpy()
        if not abstract and (not np.all((p>=0)&(p<np.array(raw['organ'].shape)))
                or not raw['organ'][tuple(p)]):
            raise AssertionError('A spatial graph node is outside the actual native organ')
        nodes.append(dict(xyz=np.round(positions[i].numpy(),3).tolist(),role=names[int(roles[i])],
            native_voxel=np.round(native[i].numpy(),3).tolist(),abstract=abstract,
            feature_norm=round(float(features[i].norm()),6),
            cosine_to_query=round(float((f[i]*f[0]).sum()),6),
            distance_mm=round(float((positions[i]-center).norm()),3),
            feature_channels=68,scale='CNN strides 1 / 2 / 4'))
    edges=[]
    edge_names={1:'spatial',2:'feature',3:'spatial + feature',4:'connectivity',
        5:'spatial + connectivity',6:'feature + connectivity',7:'spatial + feature + connectivity'}
    for u,v in torch.triu(adjacency,1).nonzero().tolist():
        if u in lookup and v in lookup:
            edges.append([lookup[u],lookup[v],edge_names[int(kinds[u,v])]])
    # Graph edges are message relations, not sampled voxel paths or vessels.
    reached={0}
    while True:
        previous=len(reached)
        for u,v,_ in edges:
            if u in reached or v in reached:
                reached.update((u,v))
        if len(reached)==previous:
            break
    if len(reached)!=len(ids):
        raise AssertionError('Exported model graph is disconnected')
    viewid=int(graph['crop_index'][scene_index])
    a=batch.audit[viewid]
    spacing=np.asarray(a['spacing']);origin=np.asarray(a['origin'])
    ranges=[]
    for name in ('near','mid','wide'):
        d=[n['distance_mm'] for n in nodes if n['role']==name]
        ranges.append(dict(role=name,min_mm=min(d),max_mm=max(d)))
    return dict(case=case,**cached_anatomy,nodes=nodes,edges=edges,center=center.tolist(),
        box=[((origin-.5)*spacing).tolist(),((origin+np.asarray(a['shape'])-.5)*spacing).tolist()],
        stats=dict(node_count=len(nodes),undirected_edges=len(edges),directed_edges=int(adjacency.sum()),
            role_counts={r:sum(n['role']==r for n in nodes) for r in names},components=1,
            max_radius_mm=max(n['distance_mm'] for n in nodes),role_radius_ranges=ranges))


def run(a):
    if a.output.exists() or a.visual.exists():
        raise FileExistsError('Existing evidence/visualization preserved; select fresh paths')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU model fallback')
    if a.workers<2 or not 0<a.resident_gib<a.rss_gib:
        raise ValueError('Parallel raw readers and explicit RAM headroom required')
    torch.set_num_threads(a.workers)
    total=torch.cuda.get_device_properties(0).total_memory
    free,_=torch.cuda.mem_get_info()
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    if not 0<budget.cuda_bytes<min(total,free):
        raise ValueError('Explicit DEBUG CUDA budget must leave current free headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    cp=checkpoint_for(a.run.resolve());cp_sha=sha(cp);assignment_sha=sha(a.assignment)
    receipt=json.loads(a.assignment_receipt.read_text(encoding='utf-8'))
    if receipt.get('assignment_sha256') != assignment_sha or receipt.get('case') != a.case:
        raise ValueError('Full native assignment differs from the existing verified complete-case receipt')
    source_before=source_identity()
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    if saved.get('format')!=FORMAT or not saved['identity'].get('debug'):
        raise ValueError('A matching local DEBUG checkpoint is required')
    if saved['identity']['source']!=source_before or hash_state({k:v for k,v in saved.items()
            if k!='content_sha256'})!=saved['content_sha256']:
        raise ValueError('Existing model/source integrity mismatch')
    ds=Dataset(a.run/'inventory/index.json','inner_train',True)
    if saved['identity']['cache_sha256']!=sha(ds.path):
        raise ValueError('Original native inventory binding differs')
    observed=[r for r in ds.rows if r['case_id']==a.case]
    if not observed:
        raise ValueError('Actual case must be in this explicit DEBUG cohort')
    donor={k:observed[0][k] for k in ('donor_case_id','donor_component','donor_group')}
    native=[r for r in json.loads(a.assignment.read_text(encoding='utf-8')) if r['case_id']==a.case]
    p=[r for r in native if r['target']==1];u=[r for r in native if r['target']==0]
    raw=next(r for r in ds.meta['raw_records'] if r['case_id']==a.case)
    if len(u)!=128 or len(p)!=len(raw['positives']) or not p:
        raise ValueError('All original positive anchors and all 128 U centers required')
    case_rows=[dict(r,**donor,bounds={'edges':i}) for i,r in enumerate(p+u)]
    rows=case_rows+[r for r in ds.rows if r['case_id']!=a.case]
    context=LiveContext(SimpleNamespace(rows=rows),32)
    tile=next(ids for ids in context.order if len(ids)==32
              and {rows[i]['case_id'] for i in ids}=={a.case}
              and 0<sum(rows[i]['target'] for i in ids)<32)
    net=make_model(ds,budget,True,'retained');net.load_state_dict(saved['model'],strict=True)
    original_state=hash_state(net.state_dict())
    reference=copy.deepcopy(net.local)
    profile=SparseFeatureProfile(context_nodes_per_band=a.nodes_per_band,
        query_radius_mm=a.query_radius_mm,near_radius_mm=a.near_radius_mm,mid_radius_mm=a.mid_radius_mm)
    sparse=SparseFeatureL0(reference,profile,budget=budget).cuda()
    net.local=GraphLocalAdapter(sparse).cuda();net.eval()
    reader=CropStore(ds.meta,a.workers,int(a.resident_gib*2**30),budget.rss_bytes)
    old_memory=saved['state']['memory'];memory=copy.deepcopy(old_memory)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    # Rebuild the entire verified DEBUG memory in the new representation. Do not
    # pair a graph query with the old mean-CNN support or reuse its cluster plan.
    with torch.no_grad():
        support_vectors=[]
        for start in range(0,len(ds.rows),32):
            ids=list(range(start,min(start+32,len(ds.rows))))
            query=bound_batch(reader,[ds.rows[i] for i in ids],ids)
            support_vectors.append(net.local(query).detach())
        memory['embeddings']=torch.cat(support_vectors)
        memory['owners']=memory['owners'].to('cuda');memory['classes']=memory['classes'].to('cuda')
        support,support_ids,_=support_binding(memory,ds.rows,case_rows[0]['patient_group'])
        plan=net.fit_support_clusters(*support)
    # Preserve actual forward graph for all original case candidates, displaying
    # every P and the first three U without using labels in node selection.
    display=set(range(len(p)))|set(range(len(p),len(p)+3))
    anatomy_cache={};candidates=[];donor_scene=None;all_stats=[]
    times=[];all_vectors=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(case_rows),32),desc='actual CT graph DEBUG',unit='batch'):
            ids=list(range(start,min(start+32,len(case_rows))))
            batch=bound_batch(reader,[case_rows[i] for i in ids],ids)
            torch.cuda.synchronize();begin=time.perf_counter()
            vectors,graph=sparse(batch,recipient_centers_native=batch.recipient_centers,
                donor_centers_native=batch.donor_centers,return_graph=True)
            torch.cuda.synchronize();times.append(time.perf_counter()-begin)
            all_vectors.append(vectors.detach())
            all_stats.append({k:v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v
                              for k,v in graph['statistics'].items()})
            # Scene order is donor scenes then recipient scenes (explicit API).
            for j,i in enumerate(ids):
                if i not in display:
                    continue
                row=case_rows[i]
                for case in (row['case_id'],row['donor_case_id']):
                    if case not in anatomy_cache:
                        anatomy_cache[case]=anatomy(reader.raw.cache[case])
                scene=graph_scene(graph,len(batch)+j,batch,reader.raw.cache[row['case_id']],
                    row['case_id'],anatomy_cache[row['case_id']])
                candidates.append(dict(id=row['id'],kind='P' if row['target'] else 'U',scene=scene))
                if donor_scene is None:
                    donor_scene=graph_scene(graph,j,batch,reader.raw.cache[row['donor_case_id']],
                        row['donor_case_id'],anatomy_cache[row['donor_case_id']])
    torch.cuda.synchronize();scope_encode_seconds=time.perf_counter()-started
    saved_step=saved['state'].get('step')
    # One actual current full-objective update. All modules are connected, but
    # this is neither training completion nor accuracy evidence.
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);net.train()
    query=bound_batch(reader,[rows[i] for i in tile],tile)
    targets=torch.tensor([rows[i]['target'] for i in tile],device='cuda',dtype=torch.long)
    optimizer=torch.optim.AdamW(net.parameters(),lr=ds.meta['base']['training']['lr'],
        weight_decay=ds.meta['base']['training']['weight_decay'])
    before={k:v.detach().clone() for k,v in net.named_parameters()}
    torch.cuda.synchronize();t=time.perf_counter()
    loss,terms=forward_loss(net,query,support,plan,targets,None,context,configuration(),indices=tile)
    torch.cuda.synchronize();forward_seconds=time.perf_counter()-t
    t=time.perf_counter();loss.backward();torch.cuda.synchronize();backward_seconds=time.perf_counter()-t
    grad={};missing=[]
    for name,param in net.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(param.grad).all()):
            raise FloatingPointError('Nonfinite gradient: '+name)
    if missing:
        raise AssertionError('Trainable parameters missing gradients: '+str(missing))
    for group,prefix in {'CNN':'local.encoder.cnn','node_projection':'local.encoder.node_project',
        'SAGE':'local.encoder.blocks','graph_readout':'local.encoder.scene_project','fusion':'local.encoder.fuse',
        'L1':'l1','L2':'l2'}.items():
        grads=[v.grad.float().square().sum() for k,v in net.named_parameters() if k.startswith(prefix)]
        if not grads:
            raise AssertionError('Expected module prefix missing: '+prefix)
        grad[group]=float(torch.stack(grads).sum().sqrt())
        if not grad[group]>0:
            raise AssertionError('No loss gradient reached '+group)
    torch.nn.utils.clip_grad_norm_(net.parameters(),ds.meta['base']['training']['grad_clip'])
    t=time.perf_counter();optimizer.step();torch.cuda.synchronize();optimizer_seconds=time.perf_counter()-t
    deltas={group:float(torch.stack([(v.detach()-before[k]).float().square().sum()
        for k,v in net.named_parameters() if k.startswith(prefix)]).sum().sqrt())
        for group,prefix in {'CNN':'local.encoder.cnn','SAGE':'local.encoder.blocks',
            'L1':'l1','L2':'l2'}.items()}
    if not all(v>0 for v in deltas.values()):
        raise AssertionError('An actual connected module was not updated')
    if sha(cp)!=cp_sha or sha(a.assignment)!=assignment_sha or source_identity()!=source_before:
        raise AssertionError('Original checkpoint/assignment/production sources changed')
    a.output.mkdir(parents=True,exist_ok=False)
    payload=dict(case=a.case,margin_mm=ds.meta['local_cnn']['margin_mm'],donor=donor_scene,
        candidates=candidates,weight_source=dict(label=f'CNN: 로컬 DEBUG saved step {saved_step} · GraphSAGE/readout: 새 초기값 seed42'),
        scope='실제 forward의 노드·엣지 · 연결선은 메시지 관계이며 혈관/CT 탐색 경로가 아님 · 추천 정확도 평가 전',profile=vars(profile))
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),9)).decode()
    template=(ROOT/'tools/sparse-feature-graph.template.html').read_text(encoding='utf-8')
    html=template.replace('__DATA__',packed)
    if len(html.encode('utf-8'))>=1_000_000:
        raise ValueError('Visualization exceeds inline budget; do not alter model graph')
    with a.visual.open('x',encoding='utf-8') as stream:
        stream.write(html)
    report=dict(debug=True,actual_CT=True,actual_CUDA=True,production_ready=False,
        production_default_changed=False,production_training_started=False,checkpoint_written=False,
        full_training=False,full_evaluation=False,case=a.case,original_P=len(p),original_U=len(u),
        total_original_case_observations=len(case_rows),complete_debug_context_observations=len(rows),
        physical_batch=32,effective_batch=32,gradient_accumulation=1,short_updates=1,
        support=dict(reencoded=True,records=len(support_ids),ids=support_ids,scope='complete saved DEBUG cohort; not full production support',
            old_mean_embeddings_used=False,cluster_plan_refitted=True),
        weight_source=dict(checkpoint=str(cp),sha256=cp_sha,source='local DEBUG only',
            saved_step=saved['state'].get('step'),CNN_initialization='copied from integrity-verified snapshot',
            graph_and_readout_initialization='new seed42',old_checkpoint_exact_resume=False),
        profile=vars(profile),graph_audits=all_stats,gradient_norms=grad,parameter_delta_norms=deltas,
        loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
        full_objective_unchanged=True,L1_L2_architecture_and_equations_unchanged=True,
        cloned_L1_L2_weights_updated_in_debug=True,Basic_CP_unchanged=True,original_masks_changed=False,
        P_U_definition_unchanged=True,original_assignment_sha256=assignment_sha,
        assignment_receipt=dict(path=str(a.assignment_receipt),sha256=sha(a.assignment_receipt)),
        timing=dict(support_teacher_all_case_and_CPU_visual_export_seconds=scope_encode_seconds,
            all_case_forward_only_seconds=sum(times),batch_forward_seconds=times,
            update_forward_seconds=forward_seconds,update_backward_seconds=backward_seconds,
            optimizer_seconds=optimizer_seconds,production_speed_comparison=False),
        model=dict(CNN_channels=[12,24,32],CNN_convolutions=[2,3,3],GNN_layers=3,hidden_dim=128,
            total_parameters=sum(v.numel() for v in net.parameters()),
            trainable_parameters=sum(v.numel() for v in net.parameters() if v.requires_grad)),
        implementation_sha256={name:sha(ROOT/name) for name in (
            'l0_sparse_feature/__init__.py','l0_sparse_feature/model.py',
            'tools/verify_sparse_feature_ct_debug.py','tests/test_sparse_feature_graph.py',
            'tools/sparse-feature-graph.template.html')},
        precision=dict(parameter_dtype='FP32',autocast=False,
            cuda_matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_TF32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        resources=dict(gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),
            total_VRAM=total,free_at_start=free,cuda_limit=budget.cuda_bytes,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            RSS=psutil.Process().memory_info().rss,RSS_limit=budget.rss_bytes,workers=a.workers,
            cpu_logical=psutil.cpu_count(),available_RAM=psutil.virtual_memory().available),
        visual=dict(path=str(a.visual),bytes=len(html.encode()),displayed_P=len(p),displayed_U=3,
            displayed_graphs=len(candidates)+1,actual_forward_nodes_edges=True,
            annotation_overlay_only=True,accuracy_qualified=False),
        limitations=['Feature/space diversity selector is a research rule, not learned CP importance.',
            'Hard selection indices are not differentiable; selected CNN features are.',
            'An edge denotes a message relation, not a vessel or voxel sampling path.',
            'A single local DEBUG update is not accuracy or production performance evidence.'])
    with (a.output/'report.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    with (a.output/'visual_data.json').open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(report=str(a.output/'report.json'),visual=str(a.visual),
        original_P=len(p),original_U=len(u),gradient=grad,peak_GiB=report['resources']['peak_allocated']/2**30),
        ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--assignment',type=Path,required=True)
    p.add_argument('--assignment-receipt',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--nodes-per-band',type=int,required=True)
    p.add_argument('--query-radius-mm',type=float,required=True)
    p.add_argument('--near-radius-mm',type=float,required=True)
    p.add_argument('--mid-radius-mm',type=float,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True)
    p.add_argument('--rss-gib',type=float,required=True)
    p.add_argument('--resident-gib',type=float,required=True)
    original_rng=rng_state()
    try:
        run(p.parse_args())
    finally:
        restore_rng(original_rng)


if __name__=='__main__':
    main()
