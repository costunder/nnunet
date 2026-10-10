"""DEBUG CPU transport probe using explicit actual frozen-bank patients.

Every trial retains the native 128-cube crop, original raw candidate, source,
geometry and engine. This does not perform native neural forward/backward or
optimizer updates and cannot establish whole-epoch training speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def _tensor_digest(result):
    digest=hashlib.sha256()
    for value in result[:3]:
        digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order='C'))
    digest.update(json.dumps(result[3],sort_keys=True,separators=(',',':')).encode())
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.repeats<3:parser.error('At least three explicit DEBUG measurements required')
    if args.output.exists():raise FileExistsError('Fresh DEBUG report path required; no overwrite')
    native=json.loads(args.native.read_text(encoding='utf8'))
    private=Path(native['root'])/'runtime';sys.path.insert(0,str(private))
    import numpy as np
    import torch
    import psutil
    from hiercp_v1x import v24_native_execution_runtime as runtime
    from hiercp_v1x import v24_native_crop_runtime as crop
    from custom_trainers.onlinecp_raw_resampling import _clip_rows
    frozen=runtime.importlib.import_module(crop.FROZEN_MODULE)
    crop.install_crop_protocol()
    original=frozen.FrozenV23Loader.__dict__[crop.METHOD]
    wrapped=runtime.wrap_paste_method(original)
    bank=frozen.FrozenV23Bank(native['bank'])
    bank_hash=hashlib.sha256(Path(native['bank']).read_bytes()).hexdigest()
    if bank_hash!=native['bank_sha256']:raise ValueError('Admitted original frozen bank changed')
    if len(args.cases)!=len(set(args.cases)) or not set(args.cases)<=set(bank.metadata['split']['outer_train']):
        raise ValueError('Distinct actual 105-training-bank patients required; validation is excluded')
    torch.set_num_threads(1)
    rng_before=pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy()))
    process=psutil.Process();before_cpu=sum(process.cpu_times()[:2]);before=time.perf_counter()
    samples=[];largest_RSS=process.memory_info().rss
    for case_id in args.cases:
        entry_names=bank.entries_by_case[case_id]
        if len(entry_names)!=1:raise ValueError('Exactly one original frozen donor payload required per patient')
        entry=bank._load(entry_names[0]);selected=int(entry['selected_candidate'][0])
        case,candidate=bank.load_raw_candidate(entry,selected)
        shape=np.asarray(case['metadata']['preprocessed_shape'],dtype=np.int64)
        focus=np.clip(np.asarray(entry['candidate_centers'][selected]),
                      np.asarray(candidate['output_bbox'])[:,0],np.asarray(candidate['output_bbox'])[:,1]-1)
        lower=np.minimum(np.maximum(focus-64,0),np.maximum(shape-128,0));upper=lower+128
        lo=np.maximum(lower,0);hi=np.minimum(upper,shape)
        box=np.stack((lo,hi),axis=1)
        source_slices=tuple(slice(int(a),int(b)) for a,b in box)
        target_slices=tuple(slice(int(a-origin),int(b-origin)) for a,b,origin in zip(lo,hi,lower))
        prepared=time.perf_counter()
        low,high=_clip_rows(case,box,case['clip_min'],case['clip_max'])
        data=np.zeros((1,128,128,128),dtype=np.float32);seg=np.full((1,128,128,128),-1,dtype=np.int16)
        data[(slice(None),*target_slices)]=np.clip(case['baseline_unclipped'][(slice(None),*source_slices)],low,high).astype(np.float32)
        seg[(slice(None),*target_slices)]=case['baseline_seg'][(slice(None),*source_slices)]
        preparation_seconds=time.perf_counter()-prepared
        for scale,shift in ((1.,0.),(1.3,350.),(.4,-550.)):
            plan=dict(raw_case=case,raw_candidate=candidate,scale=scale,shift_hu=shift,paste_contract='onlinecp_raw_target_paste_v1')
            for repeat in range(args.repeats):
                compared={}
                # Alternate trial order to expose order/warm-cache effects.
                modes=('baseline','optimized') if repeat%2==0 else ('optimized','baseline')
                for mode in modes:
                    loader=SimpleNamespace(online_bank=bank);actual_data=data.copy();actual_seg=seg.copy()
                    runtime.storage._CropArray.__getitem__=runtime._cached_getitem if mode=='optimized' else runtime._BASE_GETITEM
                    started=time.perf_counter()
                    (wrapped if mode=='optimized' else original)(loader,actual_data,actual_seg,lower.tolist(),plan,case_id)
                    wall=time.perf_counter()-started
                    result=(actual_data,actual_seg,loader._last_raw_pasted_support,loader._last_raw_paste_audit)
                    digest=_tensor_digest(result)
                    compared[mode]=digest
                    samples.append(dict(case_id=case_id,scale=scale,shift_hu=shift,repeat=repeat,mode=mode,
                        wall_seconds=wall,output_seg_support_audit_sha256=digest,
                        crop_shape=[1,128,128,128],selected_candidate=selected,
                        source_mask_voxels=int(np.count_nonzero(candidate['source_mask'])),
                        paste_audit=loader._last_raw_paste_audit,
                        event_reuse=getattr(loader,'_v24_native_crop_reuse_last',None)))
                    largest_RSS=max(largest_RSS,process.memory_info().rss)
                if compared['baseline']!=compared['optimized']:raise ValueError('Actual CPU transport output/support/audit parity failed')
        samples[-1]['baseline_crop_preparation_seconds']=preparation_seconds
    runtime.storage._CropArray.__getitem__=runtime._BASE_GETITEM
    rng_after=pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy()))
    if rng_before!=rng_after:raise ValueError('Native CP crop runtime changed RNG state')
    by_mode={mode:[row['wall_seconds'] for row in samples if row['mode']==mode] for mode in ('baseline','optimized')}
    summary={mode:dict(samples=len(times),mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),
                       minimum_seconds=min(times),maximum_seconds=max(times)) for mode,times in by_mode.items()}
    report=dict(format='v24_native_CP_actual_lossless_transport_CPU_DEBUG_v1',debug=True,
        original_bank_sha256=bank_hash,actual_training_cases=args.cases,full_training_population=105,
        complete_dataset_training_run=False,full_model_neural_forward_backward_performed=False,
        optimizer_updates=0,production_epochs_unchanged=250,physical_batch_contract_unchanged=2,
        runtime_contract=runtime.runtime_contract(),source_shape_and_source_masks_unchanged=True,
        all_output_seg_support_audit_equal=True,RNG_equal=True,summary=summary,samples=samples,
        CPU_wall_seconds=time.perf_counter()-before,CPU_seconds=sum(process.cpu_times()[:2])-before_cpu,
        CPU_affinity=process.cpu_affinity(),sampled_peak_RSS_bytes=largest_RSS,
        timing_scope='CP paste only, includes snapshot/copy/stat/source guards; input assembly and output hashing outside timed interval',
        whole_epoch_speedup_claimed=False)
    with args.output.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(report=str(args.output),summary=summary,RNG_equal=True,all_output_seg_support_audit_equal=True,
                         optimizer_updates=0,whole_epoch_speedup_claimed=False)),flush=True)
    return report


if __name__=='__main__':main()
