"""Separate native trainer for reviewed observed-rank weights and eligible CP."""
import os
import threading
from pathlib import Path
import torch
from comparison_randomness import PairedTrainerMixin
from custom_trainers.nnUNetTrainer_OnlinePairedCP import _nnUNetTrainer_250epochs_OnlineCP
from tools.v22_online_rank_bank import validate_catalog
from tools.v22_online_rank_adapter import RankedMaterializationService,native_dataloaders
from hiercp_v222.contracts import read_json
from tools.v22_online_checkpoint import OnlineCheckpointMixin


class nnUNetTrainer_250epochs_OnlineRankV22(OnlineCheckpointMixin,PairedTrainerMixin,_nnUNetTrainer_250epochs_OnlineCP):
    required_paste_contract='onlinecp_raw_target_paste_v1'
    online_policy='hier_argmax'

    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        self.ranking_metadata=validate_catalog(read_json(Path(os.environ['ONLINE_CP_BANK'])))
        if self.ranking_metadata.get('debug'):raise ValueError('DEBUG bank is not production')
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.ranking_gpu_lock=threading.RLock();self.ranking_service=None
        self.save_every=1  # Persist each completed epoch; does not alter training steps.

    def do_split(self):
        train,val=super().do_split()
        split=self.ranking_metadata['split']
        if set(train)!=set(split['outer_train']) or set(val)!=set(split['outer_val']):
            raise ValueError('Native/ranking partitions differ')
        return train,val

    def _selection_name(self):return 'observed-rank-then-eligible-selection'

    def get_dataloaders(self):
        if self.is_cascaded or self.label_manager.has_regions or self.label_manager.ignore_label is not None:
            raise ValueError('Ranking CP requires standard noncascade 0/1/2 labels')
        _,dummy,_,_=self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if dummy or len(self.configuration_manager.patch_size)!=3:raise ValueError('True 3D augmentation required')
        self.ranking_service=RankedMaterializationService(self.online_bank_path,self.ranking_gpu_lock)
        return native_dataloaders(self)

    def train_step(self,batch):
        with self.ranking_gpu_lock:return super().train_step(batch)

    def validation_step(self,batch):
        with self.ranking_gpu_lock:return super().validation_step(batch)

    def on_train_end(self):
        try:return super().on_train_end()
        finally:
            if self.ranking_service is not None:self.ranking_service.close()
