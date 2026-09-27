"""DEBUG actual native network/SGD/GradScaler on CUDA; no full cohort claim."""
import argparse
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epoch-boundary-audit',action='store_true',help='DEBUG installed first/middle/final boundary and failure cases')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    native=json.loads(args.native.read_text());plans=json.loads(Path(native['plans']).read_text())
    os.environ.update(nnUNet_preprocessed=str(Path(native['preprocessed']).parent),nnUNet_raw=str(Path(native['raw']).parent),
        nnUNet_results=str(root/'results'),nnUNet_compile='false',nnUNet_wandb_enabled='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from tools.v22_online_checkpoint import OnlineCheckpointMixin,validate_checkpoint,file_hash,FORMAT
    from tools.v22_seg_state import snapshot,content_hash
    from hiercp.preparation_runtime import Measurement,snapshot as resources
    if not torch.cuda.is_available():raise RuntimeError('This validation requires actual CUDA; no CPU fallback')
    torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    # Installed CUDA CrossEntropy has no strict deterministic implementation.
    # Keep the actual CUDA loss; do not replace it with CPU or a different loss.
    torch.use_deterministic_algorithms(False);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    contract=dict(format=FORMAT,epochs=250,debug=True,scope='actual installed native network/optimizer; DEBUG one-batch epochs',
        plans_sha256=file_hash(native['plans']),nnunet_source_sha256=file_hash(inspect.getfile(nnunetv2_trainer:=nnUNetTrainer)),seed=42)
    trainer_type=type('nnUNetTrainer_250epochs_OnlineRankV22',(OnlineCheckpointMixin,nnUNetTrainer),{'_online_contract':lambda self:contract})
    owner=trainer_type(dict(plans,continue_training=False),'3d_fullres',0,json.loads((Path(native['preprocessed'])/'dataset.json').read_text()),torch.device('cuda'))
    owner.num_epochs=250;owner.num_iterations_per_epoch=1;owner.num_val_iterations_per_epoch=1;owner.save_every=1
    owner.ranking_gpu_lock=threading.RLock();owner.initialize()
    shape=owner.configuration_manager.patch_size;batch=owner.configuration_manager.batch_size
    directory=Path(native['preprocessed'])/owner.configuration_manager.data_identifier
    dataset=infer_dataset_class(str(directory))(str(directory),['liver_31'])
    loader=nnUNetDataLoader(dataset,batch,shape,shape,owner.label_manager,oversample_foreground_percent=.33)
    def batch_on_gpu():
        value=loader.generate_train_batch();data=torch.as_tensor(value['data']).cuda()
        label=torch.as_tensor(value['target']).cuda()
        # Native train/validation transforms both remove the padding label -1.
        # Apply that exact remapping across this full GPU batch before loss.
        label=torch.where(label==-1,0,label)
        if not ((label>=0)&(label<owner.label_manager.num_segmentation_heads)).all():raise ValueError('Invalid native target labels')
        targets=[torch.nn.functional.interpolate(label.float(),size=[int(round(n*s)) for n,s in zip(shape,scale)],mode='nearest').long()
                 for scale in owner._get_deep_supervision_scales()]
        return dict(data=data,target=targets)
    first=batch_on_gpu();second=batch_on_gpu();third=batch_on_gpu()
    print(json.dumps(dict(debug=True,stage='actual_native_CUDA',GPU=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in owner.network.parameters()),
        physical_batch=batch,input_shape=list(first['data'].shape),optimizer=str(type(owner.optimizer)),scaler=str(type(owner.grad_scaler)),
        resources=resources(),compile=False,production_architecture_unchanged=True)),flush=True)
    measure=Measurement();start=time.perf_counter()
    with measure:
        owner.logger.log('epoch_start_timestamps',time.time(),0);owner.on_train_epoch_start()
        result=owner.train_step(first)
        # Actual second update establishes accumulated SGD momentum.
        result2=owner.train_step(second)
        owner.on_train_epoch_end([result,result2]);owner.network.eval()
        val=owner.validation_step(first);owner.on_validation_epoch_end([val]);owner.on_epoch_end()
        checkpoint=Path(owner.output_folder)/'checkpoint_latest.pth'
        stored=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_checkpoint(stored,contract)
        if not stored['optimizer_state']['state']:raise AssertionError('CUDA optimizer did not establish momentum')
        def update():
            owner.on_train_epoch_start();out=owner.train_step(third)
            torch.cuda.synchronize()
            rng=(random.random(),float(np.random.random()),torch.rand(8,device='cuda').cpu().tolist())
            state=snapshot(dict(model=owner.network.state_dict(),optimizer=owner.optimizer.state_dict(),scaler=owner.grad_scaler.state_dict(),
                gradients={n:p.grad for n,p in owner.network.named_parameters() if p.grad is not None}))
            return float(out['loss']),rng,content_hash(state),state
        continuous=update();owner.load_checkpoint(checkpoint);resumed=update()
        if continuous[1]!=resumed[1]:raise AssertionError('CUDA continuation RNG differs')
        differences={}
        def compare(a,b,path):
            if torch.is_tensor(a):
                if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError(f'Resume schema differs: {path}')
                error=float((a-b).abs().max()) if a.numel() else 0.
                differences[path.split('.')[0]]=max(differences.get(path.split('.')[0],0.),error)
                if not torch.allclose(a,b,rtol=1e-5,atol=1e-6):raise AssertionError(f'Resume numeric mismatch {path}: {error}')
            elif isinstance(a,dict):
                if a.keys()!=b.keys():raise AssertionError('Resume mapping mismatch')
                for k in a:compare(a[k],b[k],f'{path}.{k}' if path else str(k))
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b):raise AssertionError('Resume length mismatch')
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}.{i}')
            elif a!=b:raise AssertionError(f'Resume value mismatch: {path}')
        compare(continuous[3],resumed[3],'')
        if not np.isclose(continuous[0],resumed[0],atol=1e-6,rtol=1e-5):raise AssertionError('CUDA continuation loss differs')
        exact=continuous[:3]==resumed[:3]
        print(json.dumps(dict(stage='CUDA_next_update_compared',bitwise_exact=exact,loss=continuous[0],max_abs_differences=differences)),flush=True)
        continuous=continuous[:3];resumed=resumed[:3]
        mutations={
            'empty_momentum':lambda v:v['optimizer_state'].update(state={}),
            'model_nan':lambda v:next(t for t in v['network_weights'].values() if t.is_floating_point()).flatten().__setitem__(0,float('nan')),
            'model_inf':lambda v:next(t for t in v['network_weights'].values() if t.is_floating_point()).flatten().__setitem__(0,float('inf')),
            'momentum_nan':lambda v:next(iter(v['optimizer_state']['state'].values()))['momentum_buffer'].flatten().__setitem__(0,float('nan')),
            'momentum_shape':lambda v:next(iter(v['optimizer_state']['state'].values())).update(momentum_buffer=torch.zeros(1)),
            'momentum_dtype':lambda v:next(iter(v['optimizer_state']['state'].values())).update(momentum_buffer=torch.zeros(1,dtype=torch.float64)),
            'group_mapping':lambda v:v['optimizer_state']['param_groups'][0]['params'].reverse(),
            'scaler_missing':lambda v:v.pop('grad_scaler_state'),
            'scaler_none':lambda v:v.update(grad_scaler_state=None),
            'scaler_nan':lambda v:v['grad_scaler_state'].update(scale=float('nan')),
            'scaler_counter':lambda v:v['grad_scaler_state'].update(_growth_tracker=-1),
        }
        rejected=[]
        from unittest.mock import patch
        for name,mutate in mutations.items():
            bad=copy.deepcopy(stored);mutate(bad)
            # Actual load method, full native payload in RAM; fail before GPU mutation.
            with patch('tools.v22_online_checkpoint.torch.load',return_value=bad):
                try:owner.load_checkpoint(checkpoint)
                except ValueError:rejected.append(name)
                else:raise AssertionError(f'Corrupt native state admitted: {name}')
            del bad
        # Controlled lifecycle metrics are NOT measured segmentation accuracy.
        folder=root/'best_lifecycle_DEBUG';folder.mkdir();owner.output_folder=str(folder)
        owner.logger=MetaLogger(str(folder),False);owner.current_epoch=0;owner._best_ema=None
        def epoch(metric):
            i=owner.current_epoch
            for key,value in [('epoch_start_timestamps',time.time()),('train_losses',1.),('val_losses',1.),('dice_per_class_or_region',[metric,metric]),('ema_fg_dice',metric)]:
                owner.logger.log(key,value,i)
            owner.logger.local_logger.log('mean_fg_dice',metric,i)
            owner.logger.log('lrs',owner.optimizer.param_groups[0]['lr'],i)
            owner.on_epoch_end()
        epoch(.8);latest=folder/'checkpoint_latest.pth';best=folder/'checkpoint_best.pth';best_hash=file_hash(best)
        owner.load_checkpoint(latest);epoch(.7)
        if owner._best_ema!=.8 or file_hash(best)!=best_hash:raise AssertionError('Worse resumed epoch replaced historical best')
        lower=torch.load(latest,map_location='cpu',weights_only=False)
        if lower['current_epoch']!=2 or lower['online_best']['epoch']!=1:raise AssertionError('Completed logger/epoch/best cursor mismatch')
        original=owner._write_online_payload
        def fail_best(path,value):
            if Path(path).name=='checkpoint_best.pth':raise OSError('DEBUG interruption between latest and best')
            return original(path,value)
        with patch.object(owner,'_write_online_payload',side_effect=fail_best):
            try:epoch(.9)
            except OSError:pass
            else:raise AssertionError('Interruption was not injected')
        owner.load_checkpoint(latest)
        recovered=torch.load(best,map_location='cpu',weights_only=False)
        if owner.current_epoch!=3 or owner._best_ema!=.9 or recovered['online_best']['epoch']!=3:raise AssertionError('Pending best recovery failed')
        boundary=None
        if args.epoch_boundary_audit:
            from tools.verify_v22_epoch_boundary_debug import verify
            boundary=verify(owner,root/'epoch_boundary_DEBUG')
        owner.optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize()
    evidence=dict(debug=True,actual_CT='liver_31',actual_installed_native_network=True,actual_CUDA=True,
        production_constructor=False,full_GNN_support_coexistence=False,full_training=False,full_evaluation=False,
        scope='Native checkpoint CUDA smoke: actual CT two initial updates and identical next update; controlled best lifecycle metrics are synthetic',
        GPU=torch.cuda.get_device_name(),physical_batch=batch,input_shape=list(first['data'].shape),
        model_parameters=sum(p.numel() for p in owner.network.parameters()),architecture=plans['configurations']['3d_fullres']['architecture'],
        optimizer='torch.optim.SGD',optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in owner.optimizer.state_dict()['param_groups']],
        grad_scaler=stored['grad_scaler_state'],next_loss=continuous[0],resumed_loss=resumed[0],loss_abs_difference=abs(continuous[0]-resumed[0]),
        next_batch_is_distinct_third_crop=True,next_update_exact=exact,next_update_within_tolerance=True,
        max_abs_differences=differences,gradient_model_optimizer_scaler_hash=continuous[2],resumed_state_hash=resumed[2],rng_exact=True,
        CUDA_cross_entropy_strict_determinism_unsupported=True,global_determinism=False,cudnn_deterministic=True,benchmark=False,tf32=False,
        corruption_rejected=rejected,best_worse_epoch_preserved=True,latest_best_interruption_recovered=True,
        installed_trainer_path=inspect.getfile(nnunetv2_trainer),installed_trainer_sha256=contract['nnunet_source_sha256'],
        upstream_latest_before_best=True,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.perf_counter()-start,resources=measure.report,epoch_boundary_audit=boundary)
    (root/'result.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(result=str(root/'result.json'),seconds=evidence['seconds'],peak_CUDA=evidence['peak_CUDA_allocated_bytes'])),flush=True)


if __name__=='__main__':main()
