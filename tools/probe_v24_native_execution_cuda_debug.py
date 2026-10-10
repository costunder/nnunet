"""DEBUG full native batch2/128-cube numerical admission, no optimizer updates.

Run baseline and optimized in separate fresh processes with the same explicit
checkpoint, batch position and full 105/26 loaders. Compare all saved numerical
tensors and input/RNG receipts before admitting an execution runtime. The
diagnostic uses deterministic CUDA kernels explicitly; production precision,
seeds, architecture and optimization schedule are not changed.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def _hash_cpu_tensors(values):
    import numpy as np
    import torch
    digest=hashlib.sha256()
    def visit(path,value):
        digest.update(path.encode())
        if torch.is_tensor(value):
            array=value.detach().cpu().contiguous()
            digest.update(str(array.dtype).encode());digest.update(str(tuple(array.shape)).encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,np.ndarray):
            digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes(order='C'))
        elif isinstance(value,dict):
            for key in sorted(value):visit(path+'/'+str(key),value[key])
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):visit(path+'/'+str(index),item)
        else:
            digest.update(repr(value).encode())
    visit('root',values)
    return digest.hexdigest()


def _rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_CPU=torch.get_rng_state(),torch_CUDA=torch.cuda.get_rng_state_all())


def _finish_owned_debug_workers(augmenters):
    import psutil
    own=psutil.Process();records=[]
    for augmenter in augmenters:
        for worker in getattr(augmenter,'_processes',[]):
            if worker.is_alive():
                process=psutil.Process(worker.pid)
                if process.ppid()!=own.pid or process.uids().real!=own.uids().real:
                    raise ValueError('DEBUG cleanup refused non-owned augmentation process')
                records.append(dict(pid=process.pid,ppid=process.ppid(),create_time=process.create_time(),argv=process.cmdline()))
    print(json.dumps(dict(phase='finish_owned_DEBUG_augmentation_workers',reason='isolated numerical probe complete',processes=records,
                         original_training_process_touched=False)),flush=True)
    for augmenter in augmenters:
        if hasattr(augmenter,'_finish'):augmenter._finish()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--mode',choices=('baseline','optimized'),required=True)
    parser.add_argument('--batch-position',type=int,required=True)
    parser.add_argument('--gpu',type=int,choices=(1,),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.batch_position<0:parser.error('Nonnegative explicit DEBUG batch position required')
    if args.output.exists():raise FileExistsError('Fresh isolated DEBUG output required; no overwrite')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from tools.local_cnn_device import select
    select(args.gpu)
    from hiercp_v1x import v24_nnunet_cp as pipeline
    from hiercp_v1x import v24_native_calibration_runtime as calibration
    native,bank,source_proof=calibration._metadata(pipeline,args.native)
    if pipeline.sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Explicit source checkpoint SHA changed')
    os.environ.update(pipeline.environment(native));os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['nnUNet_results']=str(args.output/'native_results')
    sys.path.insert(0,native['private_runtime'])
    import numpy as np
    import torch
    from hiercp_v1x import v24_native_crop_runtime as crop
    from hiercp_v1x import v24_native_execution_runtime as runtime
    from hiercp_v1x import v24_native_gradient_runtime as gradient
    if args.mode=='optimized':runtime.install_runtime()
    else:crop.install_crop_protocol()
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP import nnUNetTrainer_250epochs_FrozenV23CP
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    pipeline.require_project_budget();torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Assigned singleton native CUDA required; no fallback')
    args.output.mkdir()
    pre=Path(native['root'])/'nnUNet_preprocessed'/bank['baseline']['dataset_name']
    reader,_=calibration._plans_reader(pipeline.read,pre/(pipeline.PLANS+'.json'))
    plans=reader(pre/(pipeline.PLANS+'.json'));dataset=pipeline.read(pre/'dataset.json')
    # The original OnlineCP.initialize already seeds all four generators from
    # ONLINE_CP_SEED=42. Repeat that declared seed explicitly before the DEBUG
    # constructor so pre-initialize library work is paired in both processes.
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    trainer=nnUNetTrainer_250epochs_FrozenV23CP(plans,'3d_fullres',0,dataset,torch.device('cuda'))
    if trainer.online_seed!=42:raise ValueError('Original OnlineCP seed42 required for paired DEBUG probe')
    trainer.initialize();trainer.load_checkpoint(str(args.checkpoint))
    if trainer.batch_size!=2 or trainer.num_epochs!=250 or trainer.configuration_manager.patch_size!=[128]*3:
        raise ValueError('Full native 250-epoch batch2/128-cube contract changed')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    parameters={name:parameter for name,parameter in trainer.network.named_parameters() if parameter.requires_grad}
    if {id(p) for group in trainer.optimizer.param_groups for p in group['params']}!={id(p) for p in parameters.values()}:
        raise ValueError('Native optimizer omits or adds trainable model parameters')
    helper=gradient.clone_step(pipeline._native_clone_step_with_amp_retry).__globals__[gradient.HELPER]
    gradient_proof=helper(trainer,parameters);inactive=set(gradient_proof['official_inactive_parameter_names'])
    initial_model={name:value.detach().cpu().clone() for name,value in trainer.network.state_dict().items()}
    initial_optimizer_hash=_hash_cpu_tensors(trainer.optimizer.state_dict())
    initial_model_hash=_hash_cpu_tensors(initial_model)
    initial_scaler=trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None
    loaders=[]
    try:
        train_loader,val_loader=trainer.get_dataloaders();loaders=[train_loader,val_loader]
        for position in range(args.batch_position+1):batch=next(train_loader)
        if list(batch['data'].shape)!=[2,1,128,128,128]:raise ValueError('Actual full native batch2 input required')
        input_hash=_hash_cpu_tensors(batch)
        flags=np.asarray(batch['online_cp_applied']).tolist()
        if not any(flags):raise ValueError('Explicit DEBUG batch has no actual CP event; choose a verified CP batch position')
        trainer._consume_native_transport_audit(batch,batch['online_cp_applied'])
        rng_before=_rng();rng_before_hash=_hash_cpu_tensors(rng_before)
        trainer.network.train();trainer.optimizer.zero_grad(set_to_none=True)
        data=batch['data'].to(trainer.device,non_blocking=True)
        target=[item.to(trainer.device,non_blocking=True) for item in batch['target']] if isinstance(batch['target'],list) else batch['target'].to(trainer.device,non_blocking=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.autocast(trainer.device.type,enabled=True):
            outputs=trainer.network(data);loss=trainer.loss(outputs,target)
        if trainer.grad_scaler is not None:
            trainer.grad_scaler.scale(loss).backward();trainer.grad_scaler.unscale_(trainer.optimizer)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.network.parameters(),12)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        gradients={name:parameter.grad.detach().cpu().clone() for name,parameter in parameters.items() if parameter.grad is not None}
        missing=set(parameters)-set(gradients)
        if missing!=inactive or any(not torch.isfinite(value).all() for value in gradients.values()):
            raise ValueError('Native active gradients are missing/nonfinite or disabled auxiliary-head proof differs')
        if not torch.isfinite(loss):raise ValueError('Actual native loss is nonfinite')
        rng_after=_rng();rng_after_hash=_hash_cpu_tensors(rng_after)
        final_model={name:value.detach().cpu().clone() for name,value in trainer.network.state_dict().items()}
        if _hash_cpu_tensors(trainer.optimizer.state_dict())!=initial_optimizer_hash:
            raise ValueError('DEBUG numerical admission unexpectedly changed optimizer state')
        numerical=dict(outputs=[item.detach().cpu().clone() for item in outputs] if isinstance(outputs,list) else outputs.detach().cpu().clone(),
            loss=loss.detach().cpu().clone(),gradients=gradients,initial_model=initial_model,final_model=final_model,
            rng_before=rng_before,rng_after=rng_after)
        path=args.output/'numerical_tensors.pt'
        with path.open('xb') as stream:torch.save(numerical,stream)
        report=dict(format='v24_native_batch2_full128_CUDA_execution_admission_DEBUG_v1',debug=True,mode=args.mode,
            original_checkpoint_sha256=args.checkpoint_sha256,source_epoch=int(trainer.current_epoch),
            original_source_proof=source_proof,execution_runtime=runtime.runtime_contract(),
            original_native_train_step_sha256=hashlib.sha256(inspect.getsource(nnUNetTrainer.train_step).encode()).hexdigest(),
            actual_input_shape=list(batch['data'].shape),physical_batch=2,epochs=250,full105_train=True,ordinary26_validation=True,
            original_production_augmentation_workers=4,original_validation_workers=2,batch_position=args.batch_position,
            input_keys=list(batch['keys']),CP_flags=flags,native_transport=trainer._online_native_transport,
            input_tensor_and_CP_schedule_sha256=input_hash,initial_model_sha256=initial_model_hash,
            final_model_sha256=_hash_cpu_tensors(final_model),initial_optimizer_sha256=initial_optimizer_hash,
            initial_scaler=initial_scaler,final_scaler=trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None,
            RNG_before_sha256=rng_before_hash,RNG_after_sha256=rng_after_hash,
            trainable_parameter_count=sum(p.numel() for p in parameters.values()),gradient_tensors=len(gradients),
            missing_parameter_names=sorted(missing),inactive_head_proof=gradient_proof,
            forward_backward_unscale_clip_seconds=elapsed,peak_CUDA_bytes=torch.cuda.max_memory_allocated(),
            resources=pipeline.require_project_budget(),optimizer_updates=0,production_training_performed=False,
            CUDA_deterministic_DEBUG_only=True,production_numerics_changed=False,
            explicit_DEBUG_seed=42,original_ONLINE_CP_SEED=trainer.online_seed,
            original_OnlineCP_initialize_sha256=hashlib.sha256(inspect.getsource(frozen_online_initialize(trainer)).encode()).hexdigest(),
            numerical_tensors_path=str(path),numerical_tensors_sha256=pipeline.sha(path),
            exact_future_original_production_RNG_recovery_claimed=False)
        with (args.output/'report.json').open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(report=str(args.output/'report.json'),mode=args.mode,CP_flags=flags,
            loss=float(loss.detach().cpu()),gradient_tensors=len(gradients),optimizer_updates=0,
            forward_backward_seconds=elapsed,peak_CUDA_bytes=report['peak_CUDA_bytes'])),flush=True)
        return report
    finally:
        _finish_owned_debug_workers(loaders)


def frozen_online_initialize(trainer):
    """Identify the inherited original all-generator seeding implementation."""
    owners=[owner for owner in type(trainer).__mro__ if owner.__name__=='_nnUNetTrainer_250epochs_OnlineCP']
    if len(owners)!=1 or 'initialize' not in owners[0].__dict__:
        raise ValueError('Exact original OnlineCP seeding initializer required')
    return owners[0].__dict__['initialize']


if __name__=='__main__':main()
