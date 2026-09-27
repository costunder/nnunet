"""Explicit one-case DEBUG admission; real production transforms/train/resume.

Used by verify_v22_online_actual_debug. No production validators are relaxed.
The GNN DEBUG support is not full support and metrics are not research results.
"""
import copy
import inspect
import json
import os
from pathlib import Path
import random
from types import MethodType,SimpleNamespace
import time
import numpy as np
import torch


class IntegratedSmoke:
    def __init__(self,root,native,index,bank,names,donor_index,lock,checkpoint):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from custom_trainers.nnUNetTrainer_OnlineRankV22 import nnUNetTrainer_250epochs_OnlineRankV22
        from custom_trainers.nnUNetTrainer_OnlinePairedCP import _nnUNetTrainer_250epochs_OnlineCP
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from comparison_randomness import PairedNativeLoader
        from tools.v22_online_checkpoint import FORMAT,file_hash
        from tools.v22_online_rank_bank import online_identity
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
        plans=json.loads(Path(native['plans']).read_text())
        os.environ.update(nnUNet_compile='false',nnUNet_wandb_enabled='false',ONLINE_CP_BANK=str(index))
        contract=dict(format=FORMAT,epochs=250,debug=True,scope='One-process GNN/CP/standard augmentation/native update/epoch resume smoke',
                      bank_sha256=file_hash(index),gnn_sha256=file_hash(checkpoint),native=native,
                      online_identity=online_identity(),installed_trainer_sha256=file_hash(inspect.getfile(nnUNetTrainer)))
        # Full-cohort catalog admission intentionally remains untested: this is
        # an existing DEBUG GNN artifact. All subsequent methods are inherited
        # from the actual production trainer, including CP accounting and locks.
        trainer=nnUNetTrainer_250epochs_OnlineRankV22.__new__(nnUNetTrainer_250epochs_OnlineRankV22)
        trainer.ranking_metadata=bank.metadata;trainer.ranking_gpu_lock=lock;trainer.ranking_service=None
        _nnUNetTrainer_250epochs_OnlineCP.__init__(trainer,dict(plans,continue_training=False),'3d_fullres',0,
            json.loads((Path(native['preprocessed'])/'dataset.json').read_text()),torch.device('cuda'))
        trainer._online_contract=MethodType(lambda self:contract,trainer)
        trainer.output_folder=str(root/'segmentation_DEBUG');Path(trainer.output_folder).mkdir()
        trainer.num_epochs=250;trainer.num_iterations_per_epoch=1;trainer.num_val_iterations_per_epoch=1;trainer.save_every=1
        trainer.initialize()
        torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.trainer=trainer;self.bank=bank;self.names=names;self.di=donor_index;self.index=index;self.native=native
        self.dataset_class=infer_dataset_class(str(Path(native['preprocessed'])/trainer.configuration_manager.data_identifier))
        self.directory=str(Path(native['preprocessed'])/trainer.configuration_manager.data_identifier)
        self.training=self.dataset_class(self.directory,['liver_31'])
        self.validation=self.dataset_class(self.directory,[native['split']['outer_val'][0]])
        self.trace=[];self.augmentation_checks=[];self.contract=contract
        self.rotation,self.dummy,self.initial_patch,self.mirrors=trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if self.dummy:raise AssertionError('True 3D transforms expected')
        self.shared=dict(is_cascaded=trainer.is_cascaded,foreground_labels=trainer.label_manager.foreground_labels,
                         regions=None,ignore_label=None)
        self.val_transform=trainer.get_validation_transforms(trainer._get_deep_supervision_scales(),**self.shared)
        # Establish actual SGD momentum before the first GNN request; measure
        # coexistence with optimizer buffers, not just an initialized CNN.
        warm=PairedNativeLoader(self.training,trainer.batch_size,trainer.configuration_manager.patch_size,
             trainer.configuration_manager.patch_size,trainer.label_manager,transforms=self.val_transform)
        warm.set_thread_id(0);trainer.network.train()
        output=nnUNetTrainer.train_step(trainer,warm.generate_train_batch())
        if not trainer.optimizer.state:raise AssertionError('Warm update did not create SGD state')
        self.warm_loss=float(output['loss'])
        print(json.dumps(dict(stage='integrated_native_optimizer_ready_before_GNN',parameters=sum(p.numel() for p in trainer.network.parameters()),
            physical_batch=int(trainer.batch_size),initial_patch=[int(n) for n in self.initial_patch],CUDA_allocated=torch.cuda.memory_allocated())),flush=True)

    def setup_loaders(self):
        from tools.v22_online_rank_adapter import RankedLoader
        from comparison_randomness import PairedLoaderMixin,PairedNativeLoader
        trainer=self.trainer;patch=trainer.configuration_manager.patch_size
        transform=trainer.get_training_transforms(patch,self.rotation,trainer._get_deep_supervision_scales(),self.mirrors,self.dummy,
                use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,**self.shared)
        self.transform_description=repr(transform)
        pending={}
        def observed_transform(**sample):
            # The production transform receives the real pasted sample. A
            # no-paste counterfactual uses identical RNG only for verification;
            # its output never enters training and all RNG state is restored.
            before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            actual=transform(**sample)
            if pending:
                after=(random.getstate(),np.random.get_state(),torch.get_rng_state())
                try:
                    random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])
                    original=transform(image=torch.from_numpy(pending['data']).float(),
                                       segmentation=torch.from_numpy(pending['seg']).to(torch.int16))
                finally:
                    random.setstate(after[0]);np.random.set_state(after[1]);torch.set_rng_state(after[2])
                    pending.clear()
                changed=int(torch.count_nonzero(actual['image']!=original['image']))
                added=int(torch.count_nonzero((actual['segmentation'][0]==2)&(original['segmentation'][0]!=2)))
                self.augmentation_checks.append(dict(epoch=trainer.current_epoch,batch=loader.comparison_batch,
                     augmented_CT_changed_voxels=changed,augmented_target_added_tumor_voxels=added))
            return actual
        loader=RankedLoader.__new__(RankedLoader)
        PairedLoaderMixin.__init__(loader,self.training,trainer.batch_size,self.initial_patch,patch,trainer.label_manager,
            bank_path=str(self.index),policy='hier_argmax',online_seed=42,transforms=observed_transform,
            oversample_foreground_percent=trainer.oversample_foreground_percent,
            probabilistic_oversampling=trainer.probabilistic_oversampling)
        loader.online_bank=self.bank
        loader.get_indices=lambda:['liver_31']*trainer.batch_size
        loader._source_entry_names=lambda case:self.names
        loader._load_selected_source=lambda case,i:self.bank._load(self.names[i])
        apply=loader._apply_paste_to_crop
        def observe_paste(data,seg,*args):
            pending.update(data=data.copy(),seg=seg.copy())
            return apply(data,seg,*args)
        loader._apply_paste_to_crop=observe_paste
        def debug_draws():
            i=loader._phase_counts.get('DEBUG_event',0);loader._phase_counts['DEBUG_event']=i+1
            # Deliberate positive and no-CP branch coverage. Metadata p=.8 stays
            # unchanged; observed 1/2 here is NOT an estimated CP frequency.
            draws=iter([.1 if i==0 else .9,(self.di+.5)/len(self.names),.3,.4,.5])
            return SimpleNamespace(random=lambda:next(draws))
        loader._rng=debug_draws
        val=PairedNativeLoader(self.validation,trainer.batch_size,patch,patch,trainer.label_manager,
                              transforms=self.val_transform,comparison_stage='validation')
        trainer._comparison_loaders=(loader,val);trainer._comparison_worker_counts=(0,0)
        trainer._comparison_augmenters=None;trainer._comparison_active_epoch=-1
        # DEBUG serial augmenter avoids process-launch effects in this short
        # boundary test. Production worker settings and graph workers unchanged.

    def batch_record(self,batch):
        from tools.v22_seg_state import content_hash
        record=dict(epoch=self.trainer.current_epoch,keys=list(batch['keys']),shape=list(batch['data'].shape),
                    cp_flags=np.asarray(batch['online_cp_applied']).tolist(),
                    crop_support=np.asarray(batch['online_cp_crop_support_voxels']).tolist(),
                    schedule_tokens=np.asarray(batch['online_cp_schedule_token']).tolist(),
                    input_sha256=content_hash(dict(data=batch['data'],target=batch['target'])),
                    target_shapes=[list(t.shape) for t in batch['target']])
        if record['cp_flags']!=[1,0] or record['crop_support'][0]<=0:raise AssertionError('Positive/no-CP paths not both exercised')
        if record['shape']!=[2,1,128,128,128]:raise AssertionError('Native physical batch or patch changed')
        if not torch.isfinite(batch['data']).all():raise AssertionError('Nonfinite augmented CT')
        self.trace.append(record);return record

    def run(self):
        from tools.v22_seg_state import snapshot,content_hash
        from tools.v22_online_checkpoint import file_hash
        trainer=self.trainer;self.setup_loaders()
        trainer.on_epoch_start();trainer.on_train_epoch_start()
        batch=next(trainer.dataloader_train);first=self.batch_record(batch)
        if not all(r['augmented_CT_changed_voxels']>0 and r['augmented_target_added_tumor_voxels']>0
                   for r in self.augmentation_checks if r['batch']==1):
            raise AssertionError('CP effect did not survive standard augmentation in the training batch')
        before=content_hash(snapshot(trainer.network.state_dict()))
        out=trainer.train_step(batch)
        after=content_hash(snapshot(trainer.network.state_dict()))
        if before==after:raise AssertionError('Integrated CP update did not change model')
        trainer.on_train_epoch_end([out]);trainer.network.eval()
        val=trainer.validation_step(next(trainer.dataloader_val));trainer.on_validation_epoch_end([val]);trainer.on_epoch_end()
        checkpoint=Path(trainer.output_folder)/'checkpoint_latest.pth'
        print(json.dumps(dict(stage='integrated_CP_augmented_CUDA_update_saved',loss=float(out['loss']),input=first,
                             checkpoint=str(checkpoint))),flush=True)
        def next_update():
            trainer.on_train_epoch_start();batch=next(trainer.dataloader_train);record=self.batch_record(batch)
            value=trainer.train_step(batch);torch.cuda.synchronize()
            state=snapshot(dict(model=trainer.network.state_dict(),optimizer=trainer.optimizer.state_dict(),
                    scaler=trainer.grad_scaler.state_dict(),gradients={n:p.grad for n,p in trainer.network.named_parameters() if p.grad is not None}))
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
            return dict(loss=float(value['loss']),input=record,state=state,state_hash=content_hash(state),rng_hash=content_hash(rng))
        continuous=next_update();trainer.load_checkpoint(checkpoint)
        self.setup_loaders()  # Recreate worker/loader state as in a fresh resume.
        resumed=next_update()
        if not all(r['augmented_CT_changed_voxels']>0 and r['augmented_target_added_tumor_voxels']>0
                   for r in self.augmentation_checks if r['batch']==1):
            raise AssertionError('CP effect absent from a next training batch')
        if continuous['input']!=resumed['input']:raise AssertionError('Resumed CP/standard augmentation batch differs')
        if continuous['rng_hash']!=resumed['rng_hash']:raise AssertionError('Resumed runtime RNG differs')
        differences={}
        def compare(a,b,path=''):
            if torch.is_tensor(a):
                if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('State schema differs')
                error=float((a-b).abs().max()) if a.numel() else 0.
                key=path.split('.')[0];differences[key]=max(differences.get(key,0.),error)
                if not torch.allclose(a,b,atol=1e-6,rtol=1e-5):raise AssertionError(f'Update differs: {path} {error}')
            elif isinstance(a,dict):
                if a.keys()!=b.keys():raise AssertionError('State keys differ')
                for k in a:compare(a[k],b[k],f'{path}.{k}' if path else str(k))
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b):raise AssertionError('State length differs')
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}.{i}')
            elif a!=b:raise AssertionError(f'State differs: {path}')
        compare(continuous['state'],resumed['state'])
        if not np.isclose(continuous['loss'],resumed['loss'],atol=1e-6,rtol=1e-5):raise AssertionError('Resumed loss differs')
        if continuous['input']['input_sha256']==first['input_sha256']:raise AssertionError('Next epoch did not produce a distinct augmented batch')
        return dict(debug=True,one_process=True,actual_production_train_step=True,standard_training_transforms=self.transform_description,
            actual_epoch_loader_reset=True,actual_validation_case=self.native['split']['outer_val'][0],batch_trace=self.trace,
            standard_augmentation_counterfactual=self.augmentation_checks,CP_survives_training_augmentation=True,
            warm_loss=self.warm_loss,integrated_loss=float(out['loss']),continuous_loss=continuous['loss'],resumed_loss=resumed['loss'],
            next_input_exact=True,next_RNG_exact=True,next_state_exact=continuous['state_hash']==resumed['state_hash'],
            next_loss_exact=continuous['loss']==resumed['loss'],max_abs_differences=differences,
            continuous_state_hash=continuous['state_hash'],resumed_state_hash=resumed['state_hash'],
            checkpoint_sha256=file_hash(checkpoint),model_parameters=sum(p.numel() for p in trainer.network.parameters()),
            trainable_parameters=sum(p.numel() for p in trainer.network.parameters() if p.requires_grad),
            physical_batch=trainer.batch_size,gradient_accumulation=1,effective_batch=trainer.batch_size,
            production_constructor=False,DEBUG_constructor_admission='Base OnlineCP constructor plus existing DEBUG catalog; production validators unchanged',
            augmentation_workers=0,compile=False,full_support=False,full_training=False,full_evaluation=False)
