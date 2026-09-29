"""Short actual-CT A/B against 6be85aa; identical captured GPU partition.

Measures extraction/receipt work and verified disk serialization, not training or
server epoch throughput. No source/graph/model/profile contract is altered.
"""
import argparse,copy,io,json,statistics,subprocess,sys,time,types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,psutil
import l0_regions.preparation as current
from l0_regions.training_data import dataset,reference_from_checkpoint,Budget,sha
from l0_regions.materialization import load_pairs
from l0_regions.data import validate_item,save_new,load,profile_status
from tools.v22_artifacts import tree_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    budget=Budget(6*2**30,12*2**30)
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/torch.cuda.get_device_properties(0).total_memory)
    code=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','show','6be85aa:l0_regions/preparation.py'],cwd=ROOT).decode()
    old=types.ModuleType('l0_regions._baseline_debug');old.__file__=str(ROOT/'l0_regions/preparation.py');old.__package__='l0_regions'
    exec(compile(code,old.__file__,'exec'),old.__dict__)
    ds=dataset(a.cache,'inner_train',True)
    if len(ds.rows)!=8:raise ValueError('This DEBUG probe requires the existing complete 8-pair fixture')
    ids=list(range(len(ds.rows)));profile=current.single_profile(.02)
    frozen=reference_from_checkpoint(a.checkpoint,True).cuda().eval().requires_grad_(False)
    records=[dict(ds.rows[i],donor_component=ds.record(i)['component_id']) for i in ids]
    kw=dict(cache_sha256=sha(a.cache),frozen_cnn_sha256=tree_hash(frozen.dense_encoder.state_dict()),
        profile=profile,view_epoch=0,view_index=0,feature_evidence='checkpoint_partition_quality_unverified')
    source_times={}
    for name,fn in [('old',lambda:[old.binding(record=r,dataset_index=i,**kw) for r,i in zip(records,ids)]),
                    ('new',lambda:current.batch_bindings(records,ids,**kw))]:
        ts=[]
        for _ in range(3):
            start=time.perf_counter();bindings=fn();ts.append(time.perf_counter()-start)
        source_times[name]=ts
        if name=='old':oldbindings=bindings
        else:assert oldbindings==bindings
    fine=load_pairs(ds,ids,workers=8,epoch=0,cache_path=a.cache).to('cuda')
    captured={};original=current._single_items
    def capture(*args):captured['args']=args;return original(*args)
    torch.cuda.synchronize();start=time.perf_counter()
    with patch.object(current,'_single_items',capture):
        new_items,audit=current.prepare(fine,frozen,profile,bindings,budget,allow_unvalidated_profile=True)
    torch.cuda.synchronize();new_prepare=time.perf_counter()-start
    results={};baseline=None
    for repeat in range(4):
        for name,module in ([('old',old),('new',current)] if repeat%2==0 else [('new',current),('old',old)]):
            count={'large_edge_hashes':0};real=module.tree_hash
            def hashed(value):
                if isinstance(value,dict) and set(value)==set(fine.graph.edge_types):count['large_edge_hashes']+=1
                return real(value)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad(),patch.object(module,'tree_hash',hashed):items,_=module._single_items(*captured['args'])
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if repeat:
                results.setdefault(name,dict(seconds=[],edge_hash_calls=[]))
                results[name]['seconds'].append(elapsed);results[name]['edge_hash_calls'].append(count['large_edge_hashes'])
            if name=='old':baseline=items
            else:new_items=items
    for i,(left,right) in enumerate(zip(baseline,new_items)):
        validate_item(left);validate_item(right)
        for key in ('binding','source_patch','target_patch','fine','scales','edges','materialization_receipt','fine_graph_evidence','profile_exceeded'):
            assert tree_hash(left[key])==tree_hash(right[key]),key
        for role,stats in right['audit']['scale1']['roles'].items():
            expected=[g for g in left['audit']['scale1']['roles'][role]['group_diagnostics'] if g['owner']==i]
            assert expected==stats['group_diagnostics']
        assert right['audit']['batch_audit_sha256']==tree_hash(audit)
        assert profile_status(left)==profile_status(right)
    for name,items in [('old',baseline),('new',new_items)]:
        folder=a.output/name;folder.mkdir()
        paths=[folder/f'{i}.pt' for i in ids]
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:total=sum(pool.map(lambda v:save_new(*v),zip(paths,items)))
        results[name]['verified_save_seconds']=time.perf_counter()-start
        results[name]['pair_pt_bytes']=total
        results[name]['embedded_audit_json_bytes']=sum(len(json.dumps(it['audit']).encode()) for it in items)
        for path,item in zip(paths,items):load(path,item['binding'])
        results[name]['mean_packaging_seconds']=statistics.mean(results[name]['seconds'])
    report=dict(debug=True,actual_CT=True,physical_batch=8,workers=8,baseline_commit='6be85aa',
        gpu=torch.cuda.get_device_name(0),peak_cuda_bytes=torch.cuda.max_memory_allocated(),rss_bytes=psutil.Process().memory_info().rss,
        results=results,binding_seconds=source_times,new_prepare_seconds=new_prepare,
        equality='all tensor payloads, assignments, edges, evidence, own-group diagnostics and admission flags equal on the same partition',
        scope='warm alternating packaging with identical captured GPU partition; disk save once per side; full old/new pipeline and server speedup not measured',
        full_training=False,production_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (a.output/'batch_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
