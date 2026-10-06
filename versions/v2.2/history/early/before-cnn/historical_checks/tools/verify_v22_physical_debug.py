"""Explicit DEBUG: fresh real CT features, full graph rules, CUDA batch 2/4.

Two explicitly chosen training patients x two placements are diagnostics only.
No final cohort subset, model reduction, saved weights or clinical claims.
"""
from pathlib import Path
import argparse,json,time,sys,csv
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    import numpy as np,torch
    from hiercp.common import discover_cases,load_case,organ_depth_mm
    from hiercp.schema import graph_config_from_dict
    from hiercp.preparation_runtime import snapshot
    from hiercp_v22.contracts import load_config,verify_v1,read_json,write_new,sha,safe_new_root,source_identity
    from hiercp_v22.data import sources,local_record,candidate_pool,materialize,collate
    from hiercp_v22.geometry import prepare_local_source
    from hiercp_v22.features import NAMES
    from hiercp_v22.model import PromptGraphModel,context_descriptor,prompt_loss
    from hiercp_v22.storage import GraphWriter,load_record
    from hiercp_v2.data import local_record as old_record,materialize as old_materialize
    from test_prompt_graph_v22_debug import fixtures
    from test_prompt_graph_v22_alignment_debug import memory_fixture
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--medical-root',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--cases',nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg,base=load_config();verify_v1();split=read_json(args.split)
    if len(set(args.cases))!=2 or not set(args.cases)<=set(split['inner_train']):raise ValueError('Two distinct inner-training patients required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4);root=safe_new_root(args.output);start_resources=snapshot();start=time.perf_counter()
    paths={c.case_id:c for c in discover_cases(args.medical_root/'Data')};writer=GraphWriter(root,minimum_free_bytes=0)
    def build(case_id):
        began=time.perf_counter();case=load_case(paths[case_id]);organ=np.isin(case.label,[1,2]);depth=organ_depth_mm(organ,case.spacing)
        collection=sources(case,base['cache']['source_pad'],20)
        if not len(collection):raise ValueError(f'DEBUG donor case {case_id} has no <=20mm tumor; select an explicit eligible donor for this diagnostic, never omit a production recipient')
        source,diameter=collection[0]
        prepared=prepare_local_source(case,source,full_organ_mask=organ,organ_depth=depth,
            config=graph_config_from_dict(base['graph']),rng=np.random.default_rng(cfg['seed']),ct_clip=tuple(base['ct_clip']))
        # Candidate generator retains its full 128 pool. Explicitly inspect one U
        # plus actual T, to test the extractor, not to create a training cache.
        pool,_=candidate_pool(case,source,cfg,base,depth)
        records=[];audit=[]
        for index,center in enumerate((source.anchor_center,pool[0].center)):
            fresh=local_record(case,source,center,base,prepared=prepared,depth=depth,organ=organ)
            old=old_record(case,source,center,base,depth=depth,organ=organ)
            for branch in ('source_local','target_local'):
                for kind in fresh[branch]['nodes']:
                    torch.testing.assert_close(fresh[branch]['nodes'][kind]['pos_mm'],old[branch]['nodes'][kind]['pos_mm'],rtol=0,atol=0)
                for edge in fresh[branch]['edges']:
                    torch.testing.assert_close(fresh[branch]['edges'][edge],old[branch]['edges'][edge],rtol=0,atol=0)
            saved=writer.write(f'{case_id}/{index}.pt.gz',fresh,(case_id,source.component_id))
            restored=load_record(root,saved['path']);g=materialize(restored);og=old_materialize(old)[0]
            for edge in g.edge_types:torch.testing.assert_close(g[edge].edge_index,og[edge].edge_index,rtol=0,atol=0)
            summaries=[]
            for kind in g.node_types:
                values=g[kind].observed.numpy()
                if not np.isfinite(values).all():raise ValueError('Nonfinite real CT features')
                for j,name in enumerate(NAMES):
                    summaries.append(dict(case_id=case_id,placement=index,node_type=kind,feature=name,count=len(values),
                        minimum=float(values[:,j].min()),mean=float(values[:,j].mean()),maximum=float(values[:,j].max())))
            records.append(g);audit.append(dict(case_id=case_id,component=source.component_id,placement=index,center=list(map(int,center)),
                spacing_mm=case.spacing.tolist(),diameter_mm=diameter,view_nodes=g.num_nodes,
                view_edges=sum(g[e].edge_index.shape[1] for e in g.edge_types),canonical_counts={b:fresh[b]['counts'] for b in ('source_local','target_local')},
                saved=saved,summaries=summaries))
        report=dict(case=case_id,seconds=time.perf_counter()-began,
            image_sha256=sha(paths[case_id].image_path),label_sha256=sha(paths[case_id].label_path))
        print(dict(stage='fresh_real_CT_DEBUG',**report),flush=True)
        return records,audit,report
    with ThreadPoolExecutor(max_workers=2) as pool:built=list(pool.map(build,args.cases))
    graphs=[g for item in built for g in item[0]];audits=[a for item in built for a in item[1]]
    profiles=[]
    for count in (2,4):
        torch.manual_seed(42);model=PromptGraphModel(cfg,base,patient_ids=args.cases).cuda().eval()
        batch=collate([(g,i) for i,g in enumerate(graphs[:count])]).to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            model.encode_local(batch);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[]
            for _ in range(3):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();encoded=model.encode_local(batch);end.record();end.synchronize()
                timings.append(begin.elapsed_time(end))
                if not torch.isfinite(encoded).all():raise ValueError('Nonfinite real graph encoding')
        profiles.append(dict(physical_batch=count,precision='BF16',inference_only=True,milliseconds=timings,
            graphs_per_second=count/(float(np.median(timings))/1000),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print(dict(stage='CUDA_real_graph_DEBUG',**profiles[-1]),flush=True)
        del batch,model,encoded;torch.cuda.empty_cache()
    _,batch=fixtures();batch=batch.to('cuda');model=PromptGraphModel(cfg,base,patient_ids=['A','B','C']).cuda().train()
    memory={k:v.cuda() if torch.is_tensor(v) else v for k,v in memory_fixture().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=base['training']['lr']);before={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model.forward_tasks(memory,model.encode_local(batch),torch.tensor([0,1,1,2],device='cuda'))
        loss,_=prompt_loss(result,context_descriptor(batch),torch.tensor([1,-1,0,1],device='cuda'),cfg['loss_weights'],cfg['temperature'])
    loss.backward()
    missing=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    if missing:raise RuntimeError(f'Gradient failure: {missing}')
    optimizer.step()
    updated=[n for n,p in model.named_parameters() if not torch.equal(before[n],p.detach())]
    if not updated or not torch.isfinite(loss):raise RuntimeError('Optimizer or loss failed')
    rows=[s for a in audits for s in a.pop('summaries')]
    with (root/'feature_audit.csv').open('x',newline='',encoding='utf-8-sig') as stream:
        writer_csv=csv.DictWriter(stream,fieldnames=list(rows[0]));writer_csv.writeheader();writer_csv.writerows(rows)
    verify_v1();report=dict(scope='DEBUG only; fresh real CT L0 and synthetic CUDA L0/L1/L2 gradient; no final training or clinical evaluation',
        format=cfg['format'],source_identity=source_identity(),resources_before=start_resources,resources_after=snapshot(),
        gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),model=base['model'],config=cfg,
        input_nodes_width=24,descriptor_width=144,graph_geometry_unchanged=True,canonical_and_view_edge_indices_equal=True,
        real_graphs=audits,real_cases=[item[2] for item in built],profiles=profiles,
        synthetic_cuda_gradient=dict(optimizer_steps=1,loss=float(loss.detach()),all_parameter_gradients_finite=True,updated_parameter_names=updated),
        seconds=time.perf_counter()-start,final_training_executed=False,clinical_evaluation_executed=False)
    write_new(root/'verification.json',report);print(root/'verification.json',flush=True)


if __name__=='__main__':main()
