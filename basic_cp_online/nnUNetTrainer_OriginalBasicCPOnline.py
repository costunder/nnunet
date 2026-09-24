"""Original Basic CP policy, made online without GNN or candidate pools."""
import os
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from basic_cp_online.runtime import Dataset,validate_manifest
from comparison_randomness import PairedTrainerMixin,paired_dataloaders

class nnUNetTrainer_250epochs_OriginalBasicCPOnline(PairedTrainerMixin,nnUNetTrainer):
    cp_probability=1.0
    comparison_variant='original policy, online only'
    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        self.basic_manifest=Path(os.environ['ORIGINAL_BASIC_CP_MANIFEST']).resolve();self.basic_meta=validate_manifest(self.basic_manifest)
        if self.basic_meta.get('debug'):raise ValueError('DEBUG preparation cannot train production Basic CP')
        super().__init__(plans,configuration,fold,dataset_json,device);self.num_epochs=250
    def do_split(self):
        train,val=super().do_split()
        if set(train)!=set(self.basic_meta['split']['outer_train']) or set(val)!=set(self.basic_meta['split']['outer_val']):
            raise ValueError('Native split differs from original Basic CP preparation')
        return train,val
    def get_tr_and_val_datasets(self):
        train,val=super().get_tr_and_val_datasets()
        if self.is_cascaded or self.label_manager.has_regions or self.label_manager.ignore_label is not None:
            raise ValueError('Original Basic CP supports non-cascade CT / standard 0,1,2 labels')
        if len(self.configuration_manager.patch_size)!=3:raise ValueError('Original Basic CP requires 3-D training')
        return Dataset(train,self.basic_manifest,cp_probability=self.cp_probability),val
    def initialize(self):
        result=super().initialize()
        from hiercp.preparation_runtime import snapshot
        self.print_to_log_file(dict(basic_cp=self.comparison_variant,source_size_filter=None,candidate_pool=None,
            donor='same CT only',cp_probability=self.cp_probability,attempts_per_trigger=1,maximum_proposals=4000,reference_distance_bug_preserved=True,
            ordinary_nnunet_crop=True,validation_cp=False,epochs=self.num_epochs,physical_batch=self.batch_size,
            effective_batch=self.batch_size,gradient_accumulation=1,resources=snapshot(),
            parameters=sum(p.numel() for p in self.network.parameters()),patch=self.configuration_manager.patch_size,
            workers=os.environ.get('nnUNet_n_proc_DA','nnU-Net default'),production_throughput_benchmark_required=True))
        return result
    def get_dataloaders(self):
        return paired_dataloaders(self)
