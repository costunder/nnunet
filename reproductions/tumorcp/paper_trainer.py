"""KiTS19 paper settings on the preserved official TumorCP Generic_UNet.

The 42 held-out patients never feed the training monitor or checkpoint choice.
The inherited network, image augmentation, loss and optimizer are unchanged.
"""
from collections import OrderedDict
import copy
import json
from pathlib import Path

import numpy as np

from nnunet.training.network_training.nnUNetTrainerV2_ImgDAObjCPAllInter import (
    nnUNetTrainerV2_ImgDAObjCPAllInter,
)


class nnUNetTrainerV2_TumorCPPaper(nnUNetTrainerV2_ImgDAObjCPAllInter):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True,
                 fp16=False, max_num_epochs=500, num_batches_per_epoch=250,
                 num_val_batches_per_epoch=50):
        if (max_num_epochs, num_batches_per_epoch, num_val_batches_per_epoch) != (500, 250, 50):
            raise ValueError('Paper production requires 500 epochs, 250 updates and 50 train-monitor batches')
        if fold != 0:
            raise ValueError('This reproduction fixes official-code fold 0, not a new split')
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16,
                         max_num_epochs, num_batches_per_epoch, num_val_batches_per_epoch)
        self.cp_configs = copy.deepcopy(self.cp_configs)
        self.cp_configs.update({
            'paper_profile': True, 'cp_times': 1, 'p_cp': 0.8,
            'paper_scale_range': (0.75, 1.25), 'paper_angle_range': (-np.pi, np.pi),
            'paper_elastic_alpha': (0., 900.), 'paper_elastic_sigma': (9., 13.),
            'blur_sigma': (0.5, 1.),
        })
        self._paper_split_ready = False
        self._paper_holdout_evaluation = False
        self.paper_holdout_keys = ()
        self.paper_training_keys = ()

    def process_plans(self, plans):
        super().process_plans(plans)
        if not self.threeD or self.batch_size != 2:
            raise ValueError('Paper plans must preserve 3D full-resolution and physical batch 2')

    def do_split(self):
        super().do_split()
        train = tuple(self.dataset_tr)
        holdout = tuple(self.dataset_val)
        expected = {f'case_{index:05d}' for index in range(210)}
        if len(train) != 168 or len(holdout) != 42 or set(train) & set(holdout):
            raise ValueError('KiTS19 paper split must be disjoint 168 training / 42 held-out patients')
        if set(train) | set(holdout) != expected:
            raise ValueError('KiTS19 reproduction requires all annotated case_00000 through case_00209')
        # Independently bind an existing split file to the documented code split.
        from sklearn.model_selection import KFold
        keys = np.asarray(sorted(expected))
        indices_tr, indices_val = next(KFold(5, shuffle=True, random_state=12345).split(keys))
        if set(keys[indices_tr]) != set(train) or set(keys[indices_val]) != set(holdout):
            raise ValueError('Existing split differs from official-code fold 0')
        self.paper_training_keys, self.paper_holdout_keys = train, holdout
        self._paper_split_ready = True
        chosen = holdout if self._paper_holdout_evaluation else train
        self.dataset_val = OrderedDict((key, self.dataset[key]) for key in chosen)
        receipt = {
            'format': 'tumorcp_kits19_paper_split_v1', 'train_ids': list(train),
            'heldout_ids': list(holdout), 'training_monitor_ids': list(train),
            'checkpoint_selection': 'fixed epoch 500 FINAL; heldout not used',
            'split_provenance': 'official-code KFold5 shuffle seed12345 fold0',
            'paper_exact_split_ids_published': False,
        }
        destination = Path(self.output_folder) / 'paper_split_contract.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if json.loads(destination.read_text(encoding='utf-8')) != receipt:
                raise ValueError('Saved split contract changed; existing experiment preserved')
        else:
            with destination.open('x', encoding='utf-8') as handle:
                json.dump(receipt, handle, indent=2)
        self.print_to_log_file('TumorCP paper split: train=168, heldout=42; monitor uses TRAIN ONLY; FINAL epoch500 primary')

    def enter_evaluation(self):
        """Select all 42 held-out CTs only for final full-volume prediction."""
        if len(self.all_tr_losses) != 500:
            raise RuntimeError('Full heldout evaluation requires all 500 completed training epochs')
        self._paper_holdout_evaluation = True
        if self.dataset is None:
            self.load_dataset()
        if not self._paper_split_ready:
            self.do_split()
        self.dataset_val = OrderedDict((key, self.dataset[key]) for key in self.paper_holdout_keys)
        self.print_to_log_file('TumorCP final heldout evaluation: all 42 original-resolution CTs; no CP; no heldout-selected postprocessing')

    def validate(self, *args, **kwargs):
        # Re-load the fixed FINAL even if a caller previously requested BEST.
        self.load_final_checkpoint(train=False)
        self.enter_evaluation()
        kwargs['run_postprocessing_on_folds'] = False
        kwargs['validation_folder_name'] = 'validation_final_42_raw'
        return super().validate(*args, **kwargs)
