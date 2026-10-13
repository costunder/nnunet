"""Explicit constant 80% donor / 20% recipient experiment; no boundary feather."""
import json
import os
from pathlib import Path
import numpy as np
from . import nnUNetTrainer_BasicCPAblations as base


def constant_blend80(recipient, source):
    return .2 * recipient + .8 * source


class Blend80DataLoader(base.AblationDataLoader):
    def _apply_paste_to_crop(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        base.require(self.arm == 'blend', 'Constant80 loader requires its explicit blend arm')
        entry = plan['entry']
        mask = entry['source_mask'].astype(bool, copy=False)
        slices = base.historical._anchored_slices(np.asarray(plan['center']) - np.asarray(bbox_lbs),
            mask.shape, entry['anchor_offset'], data_cropped.shape[1:])
        base.require(slices is not None and entry['source_data'].shape[0] == data_cropped.shape[0],
                     'Whole source and original channels must fit')
        donor = entry['source_data'].astype(np.float32, copy=False) * float(plan['scale']) + float(plan['normalized_offset'])
        roi = data_cropped[(slice(None), *slices)]
        roi[:, mask] = constant_blend80(roi[:, mask], donor[:, mask])
        seg_cropped[(0, *slices)][mask] = self.online_bank.tumor_label


_loaders = base.private_function(base.historical._nnUNetTrainer_250epochs_OnlineCP.get_dataloaders,
    nnUNetDataLoaderOnlineCP=Blend80DataLoader, OnlineCPBank=base.AblationBank)


class nnUNetTrainer_250epochs_BasicCPBlend80(base._BasicCPAblation):
    ablation_arm = 'blend'

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: base.historical.torch.device = base.historical.torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        witness = json.loads(Path(os.environ['BASIC_CP_ABLATION_PREFLIGHT']).read_text())
        base.require(witness.get('blend_mode') == 'constant' and witness.get('blend_alpha') == .8
            and witness.get('blend_border_preprocessed_voxels') == 0,
            'Actual fixed80 image-only preflight required; old feather evidence forbidden')

    def get_dataloaders(self):
        return _loaders(self)

    def on_train_epoch_end(self, train_outputs):
        result = base.historical.nnUNetTrainer_250epochs_OnlineBasicCP.on_train_epoch_end(self, train_outputs)
        self.print_to_log_file(f'[BasicCPAblation] epoch={self.current_epoch} arm=blend '
            f'pasted_lesions={self._ablation_pasted_lesions} events={self._online_cp_events} '
            'blend_border_voxels=0 blend_mode=constant blend_alpha=0.8', also_print_to_console=True)
        return result


def preflight_bank(*args, **kwargs):
    result = base.preflight_bank(*args, **kwargs)
    result.update(blend_mode='constant', blend_alpha=.8, blend_recipient_weight=.2,
        blend_border_preprocessed_voxels=0, label_blending=False)
    return result
