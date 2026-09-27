"""Explicit epoch-boundary segmentation continuation with online dependencies."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch

FORMAT='online_rank_epoch_resume_v1'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def run_contract(bank_path,meta,native,workers,plans):
    from tools.v22_online_rank_adapter import RankedBank
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    clean=dict(plans);clean.pop('continue_training',None)
    return dict(format=FORMAT,bank_sha256=file_hash(bank_path),gnn_sha256=meta['checkpoint_sha256'],
        native_sha256=meta['native_sha256'],plans=clean,split=meta['split'],
        selection=meta['selection'],online_identity=meta['online_identity'],
        paste_engine={k:v['sha256'] for k,v in RankedBank.paste_engine_identity().items()},
        nnunet_trainer_sha256=file_hash(inspect.getfile(nnUNetTrainer)),torch_version=torch.__version__,
        seed=42,workers=int(workers),epochs=250,configuration='3d_fullres',fold=0,
        continuation='epoch_boundary; interrupted epoch repeats; not mid-epoch exact resume')


def validate_checkpoint(value,expected):
    if value.get('online_run_contract')!=expected:raise ValueError('Segmentation checkpoint online/plans/split/seed/worker contract mismatch')
    if value.get('trainer_name')!='nnUNetTrainer_250epochs_OnlineRankV22':raise ValueError('Wrong segmentation trainer')
    epoch=value.get('current_epoch')
    if type(epoch)!=int or not 1<=epoch<=expected['epochs']:raise ValueError('Invalid completed epoch cursor')
    required={'network_weights','optimizer_state','logging','grad_scaler_state','online_rng',
              '_best_ema','init_args','inference_allowed_mirroring_axes'}
    if not required<=value.keys():raise ValueError('Incomplete segmentation continuation state')
    rng=value['online_rng']
    if set(rng)!={'python','numpy','torch','cuda'}:raise ValueError('Incomplete segmentation RNG')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    if not isinstance(rng['cuda'],list) or any(not torch.is_tensor(t) or t.dtype!=torch.uint8 or t.ndim!=1 for t in rng['cuda']):
        raise ValueError('Invalid CUDA RNG state')
    return value


def choose_checkpoint(folder,requested=None):
    folder=Path(folder).resolve()
    if requested:
        selected=Path(requested).resolve()
        if selected.parent!=folder:raise ValueError('Resume checkpoint must belong to this fold')
        if not selected.is_file():raise FileNotFoundError(selected)
        return selected
    for name in ('final','latest','best'):
        selected=folder/f'checkpoint_{name}.pth'
        if selected.is_file():return selected
    raise FileNotFoundError(f'No segmentation checkpoint in {folder}; refusing a fresh-training fallback')


class OnlineCheckpointMixin:
    def _online_contract(self):
        from hiercp_v222.contracts import read_json
        meta=self.ranking_metadata;native=read_json(meta['native_preparation'])
        return run_contract(self.online_bank_path,meta,native,int(os.environ['nnUNet_n_proc_DA']),self.plans_manager.plans)

    def save_checkpoint(self,filename):
        if self.local_rank!=0:return
        if self.disable_checkpointing:raise RuntimeError('Online segmentation requires resumable checkpoints')
        if self.is_ddp:raise RuntimeError('This admitted online owner is single-device')
        model=getattr(self.network,'_orig_mod',self.network)
        with self.ranking_gpu_lock:
            value=dict(network_weights=model.state_dict(),optimizer_state=self.optimizer.state_dict(),
                grad_scaler_state=self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                logging=self.logger.get_checkpoint(),_best_ema=self._best_ema,current_epoch=self.current_epoch+1,
                init_args=self.my_init_kwargs,trainer_name=self.__class__.__name__,
                inference_allowed_mirroring_axes=self.inference_allowed_mirroring_axes,
                online_run_contract=self._online_contract(),online_rng=dict(python=random.getstate(),
                    numpy=np.random.get_state(),torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
            validate_checkpoint(value,self._online_contract())
            from tools.v22_online_storage import check_tree_write,_write_lock
            with _write_lock:
                storage=check_tree_write(Path(filename).parent,value,80*1024**3,kind='segmentation_checkpoint')
                value['storage_admission']=storage
                path=Path(filename);path.parent.mkdir(parents=True,exist_ok=True)
                fd,temp=tempfile.mkstemp(prefix='.online-checkpoint-',dir=path.parent)
                try:
                    with os.fdopen(fd,'wb') as stream:
                        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
                    os.replace(temp,path)
                finally:
                    if Path(temp).exists():Path(temp).unlink()

    def load_checkpoint(self,filename_or_checkpoint):
        if not isinstance(filename_or_checkpoint,(str,Path)):
            raise TypeError('An actual segmentation checkpoint path is required')
        value=torch.load(filename_or_checkpoint,map_location='cpu',weights_only=False)
        validate_checkpoint(value,self._online_contract())
        if len(value['online_rng']['cuda'])!=torch.cuda.device_count():raise ValueError('Resume CUDA RNG/device count differs')
        if not self.was_initialized:self.initialize()
        if self.is_ddp:raise RuntimeError('Single-device online owner required')
        if (self.grad_scaler is None)!=(value['grad_scaler_state'] is None):raise ValueError('Resume precision/scaler differs')
        model=getattr(self.network,'_orig_mod',self.network)
        model.load_state_dict(value['network_weights'],strict=True)
        self.optimizer.load_state_dict(value['optimizer_state'])
        if self.grad_scaler is not None and value['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(value['grad_scaler_state'])
        self.my_init_kwargs=value['init_args'];self.current_epoch=value['current_epoch']
        self.logger.load_checkpoint(value['logging']);self._best_ema=value['_best_ema']
        self.inference_allowed_mirroring_axes=value.get('inference_allowed_mirroring_axes')
        rng=value['online_rng']
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
