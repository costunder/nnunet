"""Actual CT support-stage timing and reuse parity; no long training.

Read an existing complete DEBUG cache in place, including f07b13f artifacts.
Compare cold/warm reader paths and calibration CNN chunk4 vs requested batch.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--batch',type=int,required=True)
    a=p.parse_args()
    import torch
    from l0_regions import training as train,resident,data
    from l0_regions.training_data import RegionDataset,Budget,sha
    from hiercp_v222.v1_training import contiguous
    from tools.v22_artifacts import tree_hash
    torch.set_num_threads(a.workers);a.output.mkdir(parents=True,exist_ok=False)
    budget=Budget(6*2**30,12*2**30)
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/torch.cuda.get_device_properties(0).total_memory)
    before=sha(a.cache)
    ds=RegionDataset(a.cache,'inner_train',True,'research-report');net=train.make_model(ds,budget,True).eval()
    all_runs=[];outputs={};runtime_load=resident.load
    optimized_unique=data.unique_edge_count
    def run(label,reader,chunk):
        loader=train.Loader(ds,a.workers,4*2**30)
        net.local.dense_batch_size=chunk;parts=[];rows=[]
        iterator=iter(loader.batches(contiguous(ds,a.batch)));start_all=time.perf_counter()
        for ids in contiguous(ds,a.batch):
            start=time.perf_counter();cpu=next(iterator);read=time.perf_counter()-start
            torch.cuda.synchronize();start=time.perf_counter();gpu=cpu.to('cuda');torch.cuda.synchronize();transfer=time.perf_counter()-start
            start=time.perf_counter()
            with torch.no_grad():value=net.local(gpu).float()
            torch.cuda.synchronize();forward=time.perf_counter()-start
            parts.append(value);rows.append(dict(records=len(ids),loader_wait_seconds=read,h2d_validation_seconds=transfer,forward_seconds=forward))
        return torch.cat(parts),dict(arm=label,cnn_chunk=chunk,physical_batch=a.batch,
            seconds=time.perf_counter()-start_all,batches=rows)
    for label,reader,chunk in [('old_cold',data.load,4),('new_warm',runtime_load,a.batch),('old_warm',data.load,4),('new_repeat',runtime_load,a.batch)]:
        # Reproduce the actual pre-fix duplicate check as well as its two-read loader.
        count=optimized_unique if label.startswith('new') else lambda edge,ns,nt:len(torch.unique(edge.T,dim=0))
        with patch.object(resident,'load',reader),patch.object(data,'unique_edge_count',count):value,row=run(label,reader,chunk)
        outputs[label]=value;all_runs.append(row)
    errors={}
    for name,value in outputs.items():
        torch.testing.assert_close(value,outputs['old_cold'],rtol=1e-5,atol=1e-6)
        errors[name]=float((value-outputs['old_cold']).abs().max())
    loader=train.Loader(ds,a.workers,4*2**30)
    model_before=train.hash_state(net.state_dict())
    calls=[];initial=train.initial_memory
    def counted(*args,**kwargs):
        memory=initial(*args,**kwargs);calls.append(tree_hash(memory));return memory
    with patch.object(train,'initial_memory',side_effect=counted):
        batch,report,memory=train.calibrate(net,ds,loader,[a.batch],ds.meta['base'],budget,a.output/'support_timing.jsonl')
    assert len(calls)==1 and calls[0]==tree_hash(memory)
    assert train.hash_state(net.state_dict())==model_before
    assert memory['record_ids']==[r['id'] for r in ds.rows]
    assert not memory['embeddings'].requires_grad and sha(a.cache)==before
    result=dict(debug=True,actual_CT=True,status='PASS',full_training=False,cache_rebuilt=False,
        cache_preparation_revision=ds.preparation_revision,runs=all_runs,max_abs_embedding_error=errors,
        calibration_support_passes=len(calls),returned_support_unchanged=True,initial_model_unchanged=True,
        support_records=len(memory['record_ids']),physical_batch=a.batch,workers=a.workers,
        gpu=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='Entire supplied DEBUG inner_train8, alternating local cold/warm; not A6000 batch32 or full-cohort throughput')
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
