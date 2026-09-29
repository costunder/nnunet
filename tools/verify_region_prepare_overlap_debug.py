"""Short actual-CT serial/overlap comparison, including repeated recovery.

Both arms use the entire existing DEBUG cache and the same explicit batch.
Only the preparation orchestration changes; no long training is started.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def canonical_payload(item):
    """Comparison only: official WCC can permute IDs even in serial repeats.

    Do not alter saved partitions. Align by stable fine-member identity and
    compare membership, every region tensor and all typed quotient edges.
    """
    import torch
    result=copy.deepcopy({k:item[k] for k in ('source_patch','target_patch','fine','scales','edges')})
    maps={}
    for role,region in result['scales'][0].items():
        order=torch.argsort(region['stable_id'])
        inv=torch.empty_like(order);inv[order]=torch.arange(len(order));maps[role]=inv
        result['scales'][0][role]={k:v[order] for k,v in region.items()}
        result['fine'][role]['parent']=inv[result['fine'][role]['parent']]
    for rel,edge in result['edges'][0].items():
        mapped=torch.stack((maps[rel[0]][edge[0]],maps[rel[2]][edge[1]]))
        order=torch.argsort(mapped[0]*len(maps[rel[2]])+mapped[1])
        result['edges'][0][rel]=mapped[:,order]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','prepared-frozen','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True);p.add_argument('--rss-gib',type=float,required=True)
    a=p.parse_args()
    import psutil
    import torch
    import l0_regions.training_data as current
    from l0_regions.data import load
    from tools.v22_artifacts import tree_hash
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.workers)
    budget=current.Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('Explicit headroom required')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    ds=current.dataset(a.cache,'inner_train',True)
    if len(ds.rows)<=a.batch:raise ValueError('DEBUG comparison needs consecutive batches')
    origin=json.loads((a.prepared_frozen/'request.json').read_text())
    if not origin['debug']:raise ValueError('DEBUG prepared snapshot required')
    # No completed batches: both arms must actually load/merge/save all records.
    cnn=a.output/'cnn_only';cnn.mkdir()
    for name in ('request.json','frozen_cnn.pt'):shutil.copyfile(a.prepared_frozen/name,cnn/name)
    code=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
        'show','f07b13f:l0_regions/training_data.py'],cwd=ROOT).decode()
    old=types.ModuleType('l0_regions._serial_debug');old.__package__='l0_regions';old.__file__=current.__file__
    exec(compile(code,old.__file__,'exec'),old.__dict__)
    opts=dict(batch=a.batch,workers=a.workers,reg1=origin['profile']['partition']['reg_scale1'],
        view_epoch=origin['view_epoch'],budget=budget,debug=True,profile_policy='research-report',reuse_prepared=cnn)
    seconds={};indices={}
    # Alternate ordering to expose warm/cold/order effects instead of quoting one speed ratio.
    for name,module in (('serial_cold',old),('overlap_warm',current),('overlap_repeat',current),('serial_warm',old)):
        torch.cuda.synchronize();start=time.perf_counter()
        indices[name]=module.prepare_cache(a.cache,None,a.output/name,**opts)
        torch.cuda.synchronize();seconds[name]=time.perf_counter()-start
    baseline=json.loads(indices['serial_cold'].read_text())
    tensor_keys=('source_patch','target_patch','fine','scales','edges')
    checks=[]
    for name,index in indices.items():
        meta=json.loads(index.read_text());count=0;raw_equal=0
        for part in ('inner_train','inner_val'):
            assert len(meta['partitions'][part])==len(baseline['partitions'][part])
            for left,right in zip(baseline['partitions'][part],meta['partitions'][part]):
                before=load(indices['serial_cold'].parent/left['file'],left['binding'])
                after=load(index.parent/right['file'],right['binding'])
                assert left['row']==right['row'] and before['profile_exceeded']==after['profile_exceeded']
                raw_equal+=tree_hash({k:before[k] for k in tensor_keys})==tree_hash({k:after[k] for k in tensor_keys})
                assert tree_hash(canonical_payload(before))==tree_hash(canonical_payload(after)),name
                assert before['materialization_receipt']==after['materialization_receipt']
                assert before['fine_graph_evidence']==after['fine_graph_evidence']
                count+=1
        checks.append(dict(arm=name,records=count,raw_tensor_exact_records=raw_equal,
            partition_membership_and_typed_edges_exact_after_id_alignment=True))
    # A second migration must reuse BOTH old-source and current-source batches.
    full=indices['overlap_repeat'].parent
    partial=a.output/'partial';partial.mkdir()
    for name in ('request.json','frozen_cnn.pt'):shutil.copyfile(full/name,partial/name)
    for file in full.glob('inner_train_00000[0-3].*'):shutil.copyfile(file,partial/file.name)
    shutil.copyfile(full/'inner_train_000000_audit.json',partial/'inner_train_000000_audit.json')
    if a.batch!=4:raise ValueError('Recovery fixture explicitly expects DEBUG batch 4')
    calls=[];real=current.prepare
    def counted(batch,*args,**kwargs):calls.append(len(batch));return real(batch,*args,**kwargs)
    with patch.object(current,'prepare',side_effect=counted):
        mixed=current.prepare_cache(a.cache,None,a.output/'mixed_recovery',**dict(opts,reuse_prepared=partial))
    assert calls==[4,2],calls
    with patch.object(current,'prepare',side_effect=AssertionError('Completed batch recomputed')):
        recovered=current.prepare_cache(a.cache,None,a.output/'chained_recovery',**dict(opts,reuse_prepared=mixed.parent))
    recovered_meta=json.loads(recovered.read_text())
    assert recovered_meta['reused_preparation']['reused_records']==10
    report=dict(debug=True,actual_CT=True,status='PASS',full_training=False,production_ready=False,
        gpu=torch.cuda.get_device_name(),physical_batch=a.batch,workers=a.workers,
        seconds=seconds,comparisons=checks,recovery_new_batches=calls,chained_reused_records=10,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),rss_bytes=psutil.Process().memory_info().rss,
        scope='Complete existing DEBUG train8/val2; same batch4 all arms; alternating serial/overlap. Not A6000 batch32/full-cohort throughput.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
