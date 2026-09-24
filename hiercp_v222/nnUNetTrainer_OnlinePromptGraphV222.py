"""Native segmentation with frozen L0/L1/L2-aligned scores; isolated v2 trainer."""
import os
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import _nnUNetTrainer_250epochs_OnlineCP,OnlineCPBank
from hiercp_v222 import PIPELINE_VERSION
from hiercp_v222.contracts import read_json,sha,validate_native
from comparison_randomness import PairedTrainerMixin

class nnUNetTrainer_250epochs_OnlinePromptGraphV222(PairedTrainerMixin,_nnUNetTrainer_250epochs_OnlineCP):
    required_paste_contract='onlinecp_raw_target_paste_v1'
    online_policy='hier_argmax'

    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        path=Path(os.environ['ONLINE_CP_BANK']); meta=read_json(path)
        if meta.get('pipeline_version')!=PIPELINE_VERSION or meta.get('complete') is not True or meta.get('debug'):
            raise ValueError('This trainer requires a completed v2 alignment-scored bank')
        if meta['selection']!='aligned_observation_context_argmax' or meta['candidate_count']!=128 or meta['cp_probability']!=.8:
            raise ValueError('Wrong v2 CP selection contract')
        if sha(meta['checkpoint'])!=meta['checkpoint_sha256'] or sha(meta['native_preparation'])!=meta['native_sha256']:
            raise ValueError('v2 GNN/native provenance mismatch')
        native=validate_native(read_json(meta['native_preparation']))
        if meta['split']!=native['split'] or set(meta['eligible_sources_by_case'])!=set(meta['split']['outer_train']):
            raise ValueError('v2 bank does not cover the exact outer-training cohort')
        for relative,digest in meta['entry_sha256'].items():
            if sha(path.parent/relative)!=digest: raise ValueError(f'Changed v2 bank entry: {relative}')
        from hiercp_v222.bank import validate_catalog
        validate_catalog(meta)
        self.v2_metadata=meta
        super().__init__(plans,configuration,fold,dataset_json,device)
        import threading
        self.v2_gpu_lock=threading.RLock(); self.v2_service=None

    def initialize(self):
        result=super().initialize()
        from hiercp.preparation_runtime import snapshot
        self.print_to_log_file({'pipeline':PIPELINE_VERSION,'parameters':sum(p.numel() for p in self.network.parameters()),
            'trainable_parameters':sum(p.numel() for p in self.network.parameters() if p.requires_grad),
            'physical_batch':self.batch_size,'effective_batch':self.batch_size,'gradient_accumulation':1,
            'patch_size':self.configuration_manager.patch_size,'epochs':self.num_epochs,
            'batch_basis':'unchanged native ResEncM planner contract; target-GPU throughput benchmark still required',
            'resource_snapshot':snapshot(),'full_cohort':True,'debug':False})
        return result

    def do_split(self):
        train,val=super().do_split()
        if set(train)!=set(self.v2_metadata['split']['outer_train']) or set(val)!=set(self.v2_metadata['split']['outer_val']):
            raise ValueError('Native training/validation IDs differ from v2 bank')
        return train,val

    def _selection_name(self): return 'v2-cross-patient-positive-transport-argmax'

    def get_dataloaders(self):
        if self.is_cascaded or self.label_manager.has_regions or self.label_manager.ignore_label is not None:
            raise ValueError('v2 raw CP requires standard non-cascade 0/1/2 labels')
        _,dummy,_,_=self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if dummy or len(self.configuration_manager.patch_size)!=3:
            raise ValueError('v2 raw CP requires true 3D augmentation')
        from hiercp_v222.native_adapter import MaterializationService,native_dataloaders
        self.v2_service=MaterializationService(self.online_bank_path,self.v2_gpu_lock)
        return native_dataloaders(self)

    def train_step(self,batch):
        with self.v2_gpu_lock:return super().train_step(batch)

    def validation_step(self,batch):
        with self.v2_gpu_lock:return super().validation_step(batch)

    def on_train_end(self):
        try:return super().on_train_end()
        finally:
            if self.v2_service is not None:self.v2_service.close()
