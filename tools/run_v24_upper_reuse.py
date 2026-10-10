"""Verify and prepare complete v24 upper graphs using admitted full128 inputs.

This CPU-only helper never controls a training process. Existing upper files
are admitted by the original cache and never overwritten. A complete native
equivalence receipt is required before publishing missing production inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
STAGES=(7,23,39,55,71,87,103,119,128)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('verify','prepare'),required=True)
    parser.add_argument('--native-experiment',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--input-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verification',type=Path)
    parser.add_argument('--shared-job',type=Path,required=True)
    args=parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import psutil
    import torch
    from hiercp_v1x import v24_factory
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    from hiercp_v1x.v24_upper_reuse import bind,runtime_contract
    from hiercp_v1x.v24_inputs import tensor_digest
    from hiercp_v1x.v24_geometry import V24UpperGeometryCache
    from hiercp_v1x.contracts import canonical_hash
    process=psutil.Process()
    if len(process.cpu_affinity())!=4:
        raise ValueError('This helper must share exactly four admitted logical CPU IDs')
    torch.set_num_threads(1)
    job_status=json.loads((args.shared_job/'status.json').read_text())
    owned=psutil.Process(job_status['worker_pid'])
    if (owned.uids().real!=os.getuid() or abs(owned.create_time()-job_status['worker_create_time'])>.01
            or owned.cpu_affinity()!=process.cpu_affinity() or job_status['request']['RAM_GiB']!=64):
        raise ValueError('Helper must share the exact admitted GPU5 worker CPU/RAM allocation')
    resources=dict(peak_helper_RSS_bytes=0,peak_combined_RSS_bytes=0,worker_PID=owned.pid,
                   worker_create_time=owned.create_time(),shared_RSS_limit_bytes=64*2**30)
    def check_shared_resources():
        if abs(owned.create_time()-resources['worker_create_time'])>.01 or owned.uids().real!=os.getuid():
            raise ValueError('Admitted shared worker process identity changed')
        helper=process.memory_info().rss
        total=helper
        for child in [owned,*owned.children(recursive=True)]:
            try:
                if child.uids().real!=os.getuid():
                    raise ValueError('Unexpected process owner in admitted worker tree')
                total+=child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        resources['peak_helper_RSS_bytes']=max(resources['peak_helper_RSS_bytes'],helper)
        resources['peak_combined_RSS_bytes']=max(resources['peak_combined_RSS_bytes'],total)
        if total>resources['shared_RSS_limit_bytes']:
            raise MemoryError('Preparation helper plus existing worker exceeded original shared64GiB budget')
    check_shared_resources()
    config=json.loads(args.config.read_text(encoding='utf8'))
    runtime=config['v24_runtime']
    if (runtime['workers']!=4 or runtime['rss_gib']!=64
            or runtime['raw_resident_gib']!=8 or runtime['curriculum']['total_u']!=128):
        raise ValueError('Original complete population and bounded execution contract required')
    args.output.mkdir(parents=True,exist_ok=False)
    install_runtime(v24_factory)
    started=time.perf_counter()
    inputs=v24_factory.V24NativeInputs(args.native_experiment,args.inventory,args.input_cache,runtime)
    inputs.geometry.memory_guard=check_shared_resources
    adapter=bind(inputs)
    cases=(inputs.population.partition_cases('inner_train',ranking_only=True)
           +inputs.population.partition_cases('inner_val'))
    if len(cases)!=86 or len(set(cases))!=86:
        raise ValueError('Complete 65 ranking train and 21 validation cases required')
    identity=dict(input_cache=str(args.input_cache.resolve()),input_binding_file_sha256=_sha(args.input_cache/'input_bindings.json'),
                  runtime=runtime_contract(),case_ids=cases,stages=list(STAGES),
                  model_and_supervision_unchanged=True,CPU_affinity=process.cpu_affinity())
    document=dict(action=args.action,identity=identity,CPU_only=True,training_processes_signalled=False,
                  production_optimizer_updates=0,old_cache_files_overwritten=False,resources=resources,stages=[])
    _write_new(args.output/'admission.json',document)
    try:
        if args.action=='verify':
            for count in (7,23):
                rows=[]
                for case in cases:
                    plan=inputs.population.case(case,count)
                    path,key,binding=inputs.geometry._path(plan)
                    if not path.exists():
                        if count==7:
                            raise FileNotFoundError('All86 original U7 graphs required for equivalence')
                        continue
                    old_graph,old_prototype,old_audit,proof=inputs.geometry._load(path,plan,key,binding)
                    began=time.perf_counter()
                    graph,prototype,audit=adapter.derive(plan)
                    actual=tensor_digest((graph.to_dict(),prototype.to_dict()))
                    expected=tensor_digest((old_graph.to_dict(),old_prototype.to_dict()))
                    if actual!=expected or canonical_hash(audit)!=canonical_hash(old_audit):
                        raise ValueError('Complete native upper graph/prototype/audit mismatch: '+case+' U'+str(count))
                    if inputs.geometry._file_identity(path)!=proof:
                        raise ValueError('Original verified upper cache changed during comparison')
                    rows.append(dict(case_id=case,tensor_sha256=actual,audit_sha256=canonical_hash(audit),seconds=time.perf_counter()-began))
                if count==7 and len(rows)!=86:
                    raise ValueError('Incomplete actual native equivalence population')
                document['stages'].append(dict(active_U=count,compared_cases=len(rows),rows=rows))
                print(json.dumps(dict(verification_U=count,compared_cases=len(rows),tensor_and_audit_bit_exact=True)),flush=True)
            document['status']='FULL_NATIVE_EXISTING_GRAPH_EQUIVALENCE_PASS'
        else:
            if args.verification is None:
                raise ValueError('Explicit complete native equivalence receipt required before production cache publication')
            proof=json.loads(args.verification.read_text(encoding='utf8'))
            if (proof.get('status')!='FULL_NATIVE_EXISTING_GRAPH_EQUIVALENCE_PASS'
                    or proof.get('identity')!=identity or proof['stages'][0]['compared_cases']!=86):
                raise ValueError('Source/cache/runtime/full-population verification identity differs')
            document['verification_file_sha256']=_sha(args.verification)
            for count in STAGES:
                plans=[inputs.population.case(case,count) for case in cases]
                existing=sum(inputs.geometry._path(plan)[0].exists() for plan in plans)
                began=time.perf_counter()
                V24UpperGeometryCache.prepare(inputs.geometry,count)
                receipt=inputs.geometry.admit(count)
                document['stages'].append(dict(active_U=count,existing_before=existing,complete_cases=receipt['admitted_cases'],seconds=time.perf_counter()-began))
                print(json.dumps(document['stages'][-1]),flush=True)
            document['status']='ALL_NINE_COMPLETE_NATIVE_CURRICULUM_BANKS_PREPARED'
        inputs.guard_source()
        document['adapter_receipt']=adapter.receipt()
        document['wall_seconds']=time.perf_counter()-started
        document['final_RSS_bytes']=process.memory_info().rss
        _write_new(args.output/'result.json',document)
        print(json.dumps(dict(status=document['status'],receipt=str(args.output/'result.json'),wall_seconds=document['wall_seconds'])),flush=True)
        return document
    finally:
        inputs.close()


if __name__=='__main__':
    main()
