"""Paper settings on the preserved official TumorCP Generic_UNet.

Held-out patients never feed the training monitor or checkpoint choice.
The inherited network, image augmentation, loss and optimizer are unchanged.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from nnunet.training.network_training.nnUNetTrainerV2_ImgDAObjCPAllInter import (
    nnUNetTrainerV2_ImgDAObjCPAllInter,
)


def _dataset_profile():
    filename = os.environ.get('TUMORCP_PAPER_DATASET_JSON')
    if not filename:
        return None, None, None
    path = Path(filename).resolve()
    content = path.read_bytes()
    profile = json.loads(content)
    if (profile.get('format') != 'tumorcp_paper_dataset_profile_v1'
            or profile.get('dataset') != 'MSD Task03_Liver'
            or profile.get('task') != 'Task029_Liver'
            or profile.get('allow_empty_intra') is not True):
        raise ValueError('Unsupported or incomplete explicit liver paper dataset profile')
    all_ids, train, heldout = (profile[key] for key in ('all_ids', 'train_ids', 'heldout_ids'))
    if (len(all_ids) != 131 or len(train) != 105 or len(heldout) != 26
            or len(set(all_ids)) != 131 or len(set(train)) != 105 or len(set(heldout)) != 26
            or set(train) & set(heldout) or set(train) | set(heldout) != set(all_ids)
            or not profile.get('split_provenance')):
        raise ValueError('Explicit liver profile must preserve the complete original 131-patient 105/26 split')
    return profile, str(path), hashlib.sha256(content).hexdigest()


class _ConsumedCPAudit:
    """Observe actual consumed batches without copying the inherited optimizer path."""
    def __init__(self, generator, callback):
        self.generator, self.callback = generator, callback

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.generator)
        self.callback(batch['cp_audit'])
        return batch


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
        self.paper_profile, self.paper_profile_file, self.paper_profile_sha256 = _dataset_profile()
        self.cp_configs['allow_empty_intra'] = bool(self.paper_profile and self.paper_profile['allow_empty_intra'])
        self._paper_cp_counts = {}
        self._paper_split_ready = False
        self._paper_holdout_evaluation = False
        self.paper_holdout_keys = ()
        self.paper_training_keys = ()

    def process_plans(self, plans):
        super().process_plans(plans)
        if not self.threeD or self.batch_size != 2:
            raise ValueError('Paper plans must preserve 3D full-resolution and physical batch 2')

    def do_split(self):
        self._verify_profile()
        super().do_split()
        train = tuple(self.dataset_tr)
        holdout = tuple(self.dataset_val)
        expected = set(self.paper_profile['all_ids']) if self.paper_profile else {f'case_{index:05d}' for index in range(210)}
        expected_train, expected_holdout = (105, 26) if self.paper_profile else (168, 42)
        if len(train) != expected_train or len(holdout) != expected_holdout or set(train) & set(holdout):
            raise ValueError(f'Paper dataset split must be disjoint {expected_train}/{expected_holdout} patients')
        if set(train) | set(holdout) != expected:
            raise ValueError('Paper dataset split does not preserve every original patient')
        # Independently bind an existing split file to the documented code split.
        if self.paper_profile:
            if set(train) != set(self.paper_profile['train_ids']) or set(holdout) != set(self.paper_profile['heldout_ids']):
                raise ValueError('Liver split differs from the exact preserved 105/26 profile IDs')
        else:
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
            'split_provenance': self.paper_profile['split_provenance'] if self.paper_profile else 'official-code KFold5 shuffle seed12345 fold0',
            'paper_exact_split_ids_published': False,
        }
        if self.paper_profile:
            receipt.update(format='tumorcp_liver_paper_split_v1', dataset=self.paper_profile['dataset'],
                           dataset_profile_sha256=self.paper_profile_sha256,
                           allow_empty_intra=True)
        destination = Path(self.output_folder) / 'paper_split_contract.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if json.loads(destination.read_text(encoding='utf-8')) != receipt:
                raise ValueError('Saved split contract changed; existing experiment preserved')
        else:
            with destination.open('x', encoding='utf-8') as handle:
                json.dump(receipt, handle, indent=2)
        self.print_to_log_file(f'TumorCP paper split: train={expected_train}, heldout={expected_holdout}; monitor uses TRAIN ONLY; FINAL epoch500 primary; dataset_profile_sha256={self.paper_profile_sha256}')

    def _verify_profile(self):
        if self.paper_profile_file and hashlib.sha256(Path(self.paper_profile_file).read_bytes()).hexdigest() != self.paper_profile_sha256:
            raise ValueError('Dataset profile changed during the experiment; exact original split required')

    def _record_consumed_cp(self, events):
        if len(events) != self.batch_size:
            raise ValueError('Actual consumed training batch lacks complete per-recipient CP audit')
        counts = self._paper_cp_counts
        counts['consumed_batches'] = counts.get('consumed_batches', 0) + 1
        for event in events:
            status = event['status']
            if status not in ('gate_off', 'applied', 'no_source', 'empty_transform'):
                raise ValueError('Unknown actual CP event status: ' + str(status))
            if status == 'no_source' and not self.cp_configs['allow_empty_intra']:
                raise ValueError('Empty intra source is only allowed by the explicit liver profile')
            counts['recipients'] = counts.get('recipients', 0) + 1
            for key, active in (('attempted', event['attempted']),
                                ('intra', event['mode'] == 'intra'), ('inter', event['mode'] == 'inter'),
                                ('applied', status == 'applied'), ('no_source_skip', status == 'no_source'),
                                ('empty_transform_skip', status == 'empty_transform')):
                counts[key] = counts.get(key, 0) + int(active)
        return events

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        if do_backprop:
            data_generator = _ConsumedCPAudit(data_generator, self._record_consumed_cp)
        return super().run_iteration(data_generator, do_backprop, run_online_evaluation)

    def on_epoch_end(self):
        result = super().on_epoch_end()
        self.print_to_log_file('TumorCP consumed-input audit epoch ' + str(self.epoch + 1) + ': ' + json.dumps(self._paper_cp_counts, sort_keys=True))
        self._paper_cp_counts = {}
        return result

    def enter_evaluation(self):
        """Select every held-out CT only for final full-volume prediction."""
        if len(self.all_tr_losses) != 500:
            raise RuntimeError('Full heldout evaluation requires all 500 completed training epochs')
        self._paper_holdout_evaluation = True
        self._verify_profile()
        if self.dataset is None:
            self.load_dataset()
        if not self._paper_split_ready:
            self.do_split()
        self.dataset_val = OrderedDict((key, self.dataset[key]) for key in self.paper_holdout_keys)
        self.print_to_log_file(f'TumorCP final heldout evaluation: all {len(self.paper_holdout_keys)} original-resolution CTs; no CP; no heldout-selected postprocessing')

    def validate(self, *args, **kwargs):
        # Re-load the fixed FINAL even if a caller previously requested BEST.
        self.load_final_checkpoint(train=False)
        self.enter_evaluation()
        kwargs['run_postprocessing_on_folds'] = False
        kwargs['validation_folder_name'] = f'validation_final_{len(self.paper_holdout_keys)}_raw'
        return super().validate(*args, **kwargs)
