"""Evaluate V1/A/B/C own-task BEST on the same complete held-out P+128U.

No training, optimizer, original checkpoint write, CP or nnU-Net run. Each arm
has an isolated process because the preserved archived model namespace is global.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    for name in ('baseline','half-a','half-b','crossed-c','native-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--physical-batch-candidates',type=int,nargs='+',required=True)
    for name in ('cuda-gib','rss-gib','resident-gib','minimum-free-gib'):
        p.add_argument('--'+name,type=float,required=True)
    p.add_argument('--prepared-cache',type=Path)
    p.add_argument('--worker-arm',choices=['V1','A','B','C'])
    p.add_argument('--debug',action='store_true')
    p.add_argument('--debug-source',type=Path)
    p.add_argument('--debug-original-fixture',type=Path)
    p.add_argument('--debug-support-fixture',type=Path)
    p.add_argument('--debug-validation-cases',nargs='+')
    p.add_argument('--debug-region-cache',type=Path)
    a=p.parse_args()
    if (a.gpu<0 or a.workers<2 or not 0<a.resident_gib<a.rss_gib or a.cuda_gib<=0
            or a.minimum_free_gib<=0 or a.physical_batch_candidates!=sorted(set(a.physical_batch_candidates))
            or min(a.physical_batch_candidates)<1):
        raise ValueError('Explicit physical GPU, parallel workers, batches and resource budgets required')
    flags=(a.debug_source,a.debug_original_fixture,a.debug_support_fixture,a.debug_validation_cases)
    if (a.debug and not all(flags)) or (not a.debug and (any(flags) or a.debug_region_cache)):
        raise ValueError('Real-CT DEBUG fixtures/cases must be explicit and separate from production')
    return a


def worker(a):
    from hiercp_v1x.historical_evaluation import sha,verify_inventory_request
    from hiercp_v1x.contracts import canonical_hash
    request=json.loads((a.output/'request.json').read_text(encoding='utf8'))
    def verify_execution_source():
        if any(sha(ROOT/name)!=digest for name,digest in request['source'].items()):
            raise ValueError('Evaluation source changed during execution; prior outputs preserved')
        verify_inventory_request(a.native_run/'inventory/index.json',request)
    verify_execution_source()
    from tools.local_cnn_device import select
    select(a.gpu)
    import torch,psutil,numpy as np,random
    from hiercp_v1x.historical_evaluation import (ResourceBudget,write_new,sha,
        original_geometry_provider,evaluate_historical,calibrate,encode_original_fields)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one explicitly selected CUDA device required; no CPU fallback')
    total=torch.cuda.get_device_properties(0).total_memory; free,_=torch.cuda.mem_get_info()
    if not 0<a.cuda_gib*2**30<min(total,free):
        raise ValueError('Explicit CUDA budget must fit current free VRAM and leave headroom')
    if a.rss_gib*2**30>psutil.virtual_memory().available+psutil.Process().memory_info().rss:
        raise ValueError('Explicit RSS exceeds available RAM')
    torch.cuda.set_per_process_memory_fraction(a.cuda_gib*2**30/total)
    torch.set_num_threads(a.workers); torch.manual_seed(42); np.random.seed(42); random.seed(42)
    budget=ResourceBudget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    import uuid
    arm_output=a.output/a.worker_arm
    output=arm_output/'attempts'/uuid.uuid4().hex; output.mkdir(parents=True,exist_ok=False)
    resources=dict(GPU=torch.cuda.get_device_name(),total_VRAM_bytes=total,free_VRAM_bytes=free,
        CPU_affinity=len(psutil.Process().cpu_affinity()),host_available_RAM_bytes=psutil.virtual_memory().available,
        workers=a.workers,cuda_limit_bytes=budget.cuda_bytes,rss_limit_bytes=budget.rss_bytes,
        resident_bytes=int(a.resident_gib*2**30),debug=a.debug,optimizer_updates=0,training_started=False)
    manager=None
    if a.worker_arm=='C':
        from hiercp_v1x.historical_c_checkpoint import load_c
        bundle=load_c(a.crossed_c,a.baseline,a.native_run,budget,a.workers,
            int(a.resident_gib*2**30),output,debug=a.debug,
            debug_original_fixture=a.debug_original_fixture,debug_support_fixture=a.debug_support_fixture)
        from tools.run_v17_crossed_training import NativeRows,NativeRowsLoader
        ds=NativeRows(a.native_run/'inventory/index.json','inner_val',a.debug,a.debug_validation_cases)
        loader=NativeRowsLoader(ds,a.workers,int(a.resident_gib*2**30),budget.rss_bytes)
        inventory=bundle.inventory; model=bundle.net
        preparation=dict(native_crop_cache=True,no_original_graph_preparation=True)
    else:
        from hiercp_v1x.historical_checkpoint import load_historical,load_debug_historical,rebuild_historical_support
        experiment={'V1':a.baseline,'A':a.half_a,'B':a.half_b}[a.worker_arm]
        bundle=(load_debug_historical(a.worker_arm,a.debug_source,a.debug_original_fixture,
            a.debug_support_fixture,budget=budget,workers=a.workers) if a.debug else
            load_historical(a.worker_arm,experiment,budget=budget))
        inventory=json.loads((a.native_run/'inventory/index.json').read_text(encoding='utf8'))
        from hiercp_v1x.transition_evaluation import validate_cohort
        validate_cohort(inventory,case_ids=a.debug_validation_cases,debug=a.debug)
        from hiercp_v1x.transition_v1_data import NativeObservationDataset
        ds=NativeObservationDataset(a.native_run/'inventory/index.json','inner_val',a.debug,
                                    case_ids=a.debug_validation_cases)
        loader,preparation=original_geometry_provider(ds,root=a.output/'shared_validation_geometry',
            workers=a.workers,resident_bytes=int(a.resident_gib*2**30),rss_bytes=budget.rss_bytes,
            prepared_cache=a.prepared_cache,minimum_free_bytes=int(a.minimum_free_gib*2**30))
        model=bundle.model
        if a.worker_arm=='B' and not a.debug:
            manager=rebuild_historical_support(bundle,output=output/'support',budget=budget)
    model.eval()
    from hiercp.preparation_runtime import snapshot
    resources['effective_allocation']=snapshot()
    if a.workers>resources['effective_allocation']['cpu_capacity']:
        raise ValueError('Explicit workers exceed affinity/cgroup CPU allocation')
    if budget.rss_bytes>resources['effective_allocation']['rss_bytes']+resources['effective_allocation']['available_memory_bytes']:
        raise ValueError('Explicit RSS exceeds effective allocation')
    if a.worker_arm=='C':
        from hiercp_v1x.transition_native_local import native_transition_spec
        from hiercp_v1x.half_b_model import half_b_spec
        reference=half_b_spec()
        shared_upper={k:reference[k] for k in ('legacy_upper','hidden_dim','heads',
            'task_layers','alignment_layers','temperature','L1','L2','clusters','scorer')}
        executed_model=dict(local=native_transition_spec(),upper=shared_upper,
                            support_policy=bundle.receipt['support_policy'],
                            local_graph_layers_executed=0)
    elif a.worker_arm=='A':
        from hiercp_v1x.half_a_model import model_contract
        executed_model=model_contract()
    elif a.worker_arm=='B':
        from hiercp_v1x.half_b_model import model_contract
        executed_model=model_contract()
    else:
        executed_model=bundle.config['model']
    resources.update(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        validation_records=len(ds),whole_128_U_per_case=True,margin_mm=10,
        model=type(model).__name__,archived_model_arguments=bundle.config['model'],
        executed_model_configuration=executed_model,
        original_checkpoint_receipt=bundle.receipt)
    write_new(output/'resources.json',resources)
    def encode(cpu):
        budget()
        with torch.no_grad(),torch.autocast('cuda',enabled=bool(bundle.config['training']['amp'])):
            gpu=cpu.to('cuda')
            result=model.local(gpu) if a.worker_arm=='C' else encode_original_fields(model,gpu)
        budget(); return result
    batch,calibration=calibrate(encode,loader,ds,a.physical_batch_candidates,budget,arm=a.worker_arm)
    write_new(output/'calibration.json',calibration)
    report,_=evaluate_historical(bundle,inventory,ds,loader,batch=batch,budget=budget,
        output=output,debug=a.debug,c=a.worker_arm=='C',region_output=a.output/'shared_regions',
        region_reuse=a.debug_region_cache)
    if a.worker_arm=='C':
        report['checkpoint']=bundle.receipt
        report['old_evidence_preserved']=all(sha(p)==h for p,h in bundle.receipt['source_artifacts_sha256'].items())
        if not report['old_evidence_preserved']:
            raise ValueError('Old C artifacts changed during evaluation')
    report.update(resources=resources,calibration=calibration,preparation=preparation,
        model_training=False,old_experiment_files_written=False)
    verify_execution_source()
    report['evaluation_request_sha256']=canonical_hash(request)
    report_path=arm_output/'report.json'
    write_new(report_path,report)
    write_new(arm_output/'complete.json',dict(arm=a.worker_arm,request_sha256=canonical_hash(request),
              report_sha256=sha(report_path),report=str(report_path.resolve())))
    m=report['metrics']
    print(f"{a.worker_arm} full P+128U | MRR={m['case_first_P_mrr']:.6f} Hit@1={m['case_hit_at_1']:.6f} "
          f"pair-win={m['P_U_pair_win_rate']:.6f} loss={m['P_U_softplus_loss']:.6f} "
          f"cases={report['denominators']['cases']} batch={batch}",flush=True)
    print('REPORT: '+str(report_path.resolve()),flush=True)
    if manager is not None:
        manager.close()


def main():
    a=parse()
    from hiercp_v1x.historical_evaluation import assert_new_destination,write_new,sha
    if a.debug:
        assert_new_destination(a.output,(a.native_run,a.crossed_c))
    else:
        assert_new_destination(a.output,(a.baseline,a.half_a,a.half_b,a.crossed_c,a.native_run))
    if a.worker_arm:
        worker(a); return
    inventory_path=a.native_run/'inventory/index.json'
    sources=set([Path(__file__),*ROOT.joinpath('hiercp_v1x').glob('historical_*.py')])
    # Close over actual imported local dependencies. A running worktree may not
    # silently switch its helper implementation after the request is bound.
    import ast
    pending=list(sources)
    while pending:
        path=pending.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            names=[]
            if isinstance(node,ast.ImportFrom):
                name=node.module or ''
                if node.level and path.parent.name=='hiercp_v1x':
                    name='hiercp_v1x.'+name
                    if not node.module:
                        names.extend('hiercp_v1x.'+alias.name for alias in node.names)
                names.append(name)
            elif isinstance(node,ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                dependency=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if (name.split('.')[0] in ('tools','hiercp_v1x','hiercp_v22','hiercp_v222',
                    'l0_local_cnn','l0_exploration','l0_regions','l0_sage','l0_ezsp')
                        and dependency.is_file() and dependency not in sources):
                    sources.add(dependency);pending.append(dependency)
    request=dict(format='historical_V1_A_B_C_full128_readonly_evaluation_v1',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        inventory_sha256=sha(inventory_path),debug=a.debug,training_started=False,
        checkpoint_selection='existing original own-task BEST; never latest or full128 post-hoc best',
        source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)})
    a.output.mkdir(parents=True,exist_ok=True)
    request_path=a.output/'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text(encoding='utf8'))!=request:
            raise ValueError('Existing evaluation request changed; outputs preserved')
    else:
        write_new(request_path,request)
    from tools.run_local_cnn_experiment import foreground,exclusive
    from hiercp_v1x.contracts import canonical_hash
    reports={}; started=time.perf_counter()
    with exclusive(a.output):
        for arm in ('V1','A','B','C'):
            path=a.output/arm/'report.json'
            if not path.exists():
                command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker-arm',arm]
                foreground(command)
            report=json.loads(path.read_text(encoding='utf8'))
            seal_path=path.with_name('complete.json')
            seal=json.loads(seal_path.read_text(encoding='utf8'))
            if (report.get('arm')!=arm or report.get('training_started') is not False
                    or report.get('debug')!=a.debug or report.get('old_evidence_preserved') is not True
                    or report.get('evaluation_request_sha256')!=canonical_hash(request)
                    or seal.get('arm')!=arm or seal.get('request_sha256')!=canonical_hash(request)
                    or seal.get('report_sha256')!=sha(path) or not report.get('execution_contract_bound')
                    or report.get('full_evaluation') is not (not a.debug)):
                raise ValueError('Stored historical evaluation report contract differs')
            checkpoint=report['checkpoint']
            preserved=checkpoint['source_artifacts_sha256'] if arm=='C' else checkpoint['files_preserved']
            if any(sha(p)!=digest for p,digest in preserved.items()):
                raise ValueError('Historical source checkpoint bytes changed after completed evaluation')
            reports[arm]=report
        cohort={r['cohort']['cohort_sha256'] for r in reports.values()}
        if len(cohort)!=1:
            raise ValueError('Historical arms did not score the identical full P/U cohort')
        summary=dict(format=request['format'],complete=True,debug=a.debug,training_started=False,
            full_evaluation=not a.debug,quality_verified=False,seconds=time.perf_counter()-started,
            same_native_query_cohort=True,same_support_semantics=False,
            V1_A_annotation_aware=True,blind_recommendation_quality_verified=False,
            weights_preserved=True,arms={arm:dict(metrics=r['metrics'],denominators=r['denominators'],
                checkpoint=r['checkpoint'],report=str((a.output/arm/'report.json').resolve()),
                support_policy=r['support_policy'],annotation_derived_recipient_graph=r['annotation_derived_recipient_graph'])
                for arm,r in reports.items()})
        path=a.output/'summary.json'
        if not path.exists():
            write_new(path,summary)
        print('\nExisting own-task BEST → identical full P+128U evaluation',flush=True)
        for arm,r in reports.items():
            m=r['metrics']; print(f"{arm:2} | MRR {m['case_first_P_mrr']:.6f} | Hit@1 {m['case_hit_at_1']:.6f} | "
                f"pair-win {m['P_U_pair_win_rate']:.6f} | loss {m['P_U_softplus_loss']:.6f}",flush=True)
        print('V1/A retain recipient lesion annotations. B/C retain their own anchor-label support.',flush=True)
        print('SUMMARY: '+str(path.resolve()),flush=True)


if __name__=='__main__':
    main()
