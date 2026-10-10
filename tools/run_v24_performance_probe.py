"""DEBUG: one actual full128 calibrated physical batch parity and timing probe.

The complete production40epoch/calibration3repeat configuration is untouched.
This explicit diagnostic uses the same four maximum-cost native patients,
all their P+128U and real forward/loss/backward, with zero optimizer updates.
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
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CASES=('liver_6','liver_129','liver_123','liver_69')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--variant',choices=('baseline','optimized'),required=True)
    parser.add_argument('--source-output',type=Path,required=True)
    parser.add_argument('--source-code',type=Path,required=True)
    parser.add_argument('--source-interruption-proof',type=Path)
    parser.add_argument('--probe-output',type=Path,required=True)
    parser.add_argument('--debug-performance-probe',action='store_true',required=True)
    options,remaining=parser.parse_known_args(argv)
    from tools.run_v24_all_p import parse,validate_config
    args=parse(remaining)
    if args.mode!='train':raise ValueError('Full original training config required for diagnostic')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import psutil
    process=psutil.Process()
    if len(process.cpu_affinity())!=4:raise ValueError('Original four logical CPU IDs required')
    import torch
    torch.set_num_threads(1)
    from tools.run_v24_performance_continuation import admit
    admission=admit(args,options.source_output,options.source_code,options.source_interruption_proof)
    from hiercp_v1x.v24_training_continuation import inspect_continuation
    source=inspect_continuation(args.output,options.source_code)
    if source['latest']['status'] not in ('PAUSED','RUNNING'):
        raise ValueError('Exact admitted paused or interrupted snapshot required for diagnostic')
    if source['latest']['status']=='RUNNING' and options.source_interruption_proof is None:
        raise ValueError('Interrupted source requires explicit verified interruption proof')
    owner=json.loads((args.output/'training/training_identity.json').read_text())
    saved=torch.load(args.output/'training/checkpoint_latest.pt',map_location='cpu',weights_only=False)
    calibration=json.loads((args.output/'calibration.json').read_text())
    physical=calibration['selected_physical_patient_batch'];chunk=calibration['selected_physical_candidate_batch']
    trial=[row for row in calibration['trials'] if row.get('accepted')
           and row['physical_patient_batch']==physical and row['physical_candidate_batch']==chunk]
    if len(trial)!=1 or tuple(trial[0]['case_ids'])!=CASES or physical!=4:
        raise ValueError('Original measured maximum-cost full128 diagnostic batch differs')
    config=validate_config(json.loads(args.config.read_text()),args.gpu,args.stunet_checkpoint)
    from hiercp_v1x import v24_hash_runtime as hashes
    hash_receipt=hashes.install() if options.variant=='optimized' else None
    from hiercp_v1x import v24_factory,v24_memory_runtime as memory,v24_prefetch_runtime as prefetch
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    install_runtime(v24_factory);memory.install_memory_runtime(v24_factory)
    prefetch_receipt=prefetch.install_runtime(memory) if options.variant=='optimized' else None
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Real singleton CUDA required, no fallback')
    properties=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info()
    if 'A6000' not in properties.name or free<=40*2**30:
        raise ValueError('Original40GiB free A6000 budget required')
    torch.cuda.set_per_process_memory_fraction(40*2**30/total)
    from hiercp_v1x.historical_evaluation import ResourceBudget
    from hiercp_v1x.u_bridge_training import capture_rng,restore_rng,digest,gradient_receipt,_groups
    from hiercp_v1x.v24_training import optimizer_groups,_plans,_positive_indices,patient_balanced_objective
    budget=ResourceBudget(40*2**30,64*2**30)
    options.probe_output.mkdir(parents=True,exist_ok=False)
    net,scorer,population,training_config,contract,close=v24_factory.build_runtime(args,config,budget)
    try:
        memory.bind_memory_runtime(scorer)
        if options.variant=='optimized':prefetch.bind(scorer)
        net.load_state_dict(saved['model'],strict=True);net.train()
        scorer.physical_candidate_batch=chunk
        optimizer=torch.optim.AdamW(optimizer_groups(net,training_config['training']))
        optimizer.load_state_dict(saved['optimizer'])
        scaler=torch.amp.GradScaler('cuda',enabled=training_config['training']['amp'])
        scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rank_rng'][0])
        before_model=digest(net.state_dict());before_rng=digest(capture_rng())
        before_providers={key:provider.profile() for key,provider in scorer.providers.items()}
        plans=_plans(population,list(CASES),128)
        if len(plans)!=4 or any(plan.active_u_count!=128 for plan in plans):
            raise ValueError('Complete actual four-patient allP+128U plans required')
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        # Epoch1 matches the original full128 maximum-cost calibration's seeds.
        result=scorer(plans,epoch=1,training=True)
        loss,terms=patient_balanced_objective(result.scores,[_positive_indices(plan) for plan in plans],result.consistency)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite native diagnostic loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        gradient=gradient_receipt(net,_groups(net))
        if not gradient['finite'] or gradient['missing']:
            raise FloatingPointError('Native diagnostic gradient nonfinite or disconnected: '+repr(gradient))
        budget.check()
        proof=dict(debug=True,diagnostic='actual_full128_forward_loss_backward_no_optimizer_update',
            variant=options.variant,source_checkpoint=source['latest'],
            original_training_identity_sha256=owner['identity_sha256'],
            initial_model_sha256=before_model,initial_RNG_sha256=before_rng,
            output_sha256=digest(tuple(score.detach().cpu() for score in result.scores)),
            loss=float(loss.detach()),gradient_sha256=digest({name:parameter.grad.detach().cpu()
                for name,parameter in net.named_parameters() if parameter.requires_grad}),
            after_forward_model_sha256=digest(net.state_dict()),after_RNG_sha256=digest(capture_rng()),
            case_ids=list(CASES),active_U=128,all_P_included=True,
            physical_patient_batch=physical,physical_candidate_chunk=chunk,gradient_accumulation=1,
            total_ranking_train_patients=65,actual_probe_patients=4,production_patient_subset_used=False,
            production_optimizer_updates=0,production_calibration_repeats=3,diagnostic_batches=1,
            seconds=seconds,workload=result.workload,gradient=gradient,
            peak_CUDA_bytes=torch.cuda.max_memory_allocated(),RSS_bytes=process.memory_info().rss,
            CPU_affinity=process.cpu_affinity(),GPU=properties.name,
            provider_profiles_before=before_providers,
            provider_profiles_after={key:provider.profile() for key,provider in scorer.providers.items()},
            hash_runtime=hash_receipt,prefetch_runtime=prefetch_receipt,
            model_contract=contract,full_training_or_evaluation_completion_claimed=False)
        if inspect_continuation(args.output,options.source_code)!=source:
            raise ValueError('Diagnostic changed the exact preserved continuation snapshot')
        with (options.probe_output/'result.json').open('x') as stream:json.dump(proof,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(variant=options.variant,seconds=seconds,loss=proof['loss'],
            result=str(options.probe_output/'result.json'),peak_CUDA_bytes=proof['peak_CUDA_bytes'],
            exposed_CPU_wait=result.workload.get('CPU_timing',{}).get('cpu_input_exposed_wait_seconds'))),flush=True)
    finally:close()


if __name__=='__main__':main()
