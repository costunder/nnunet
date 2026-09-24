"""Shared segmentation RNG schedule for original Basic CP and v2.22.

This controls random draws, not equality of the two CP policies or GPU kernels.
The native patch/transform algorithms and physical batch size are unchanged.
"""
from contextlib import contextmanager
import hashlib
import os
import random

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

FORMAT = 'paired_segmentation_rng_v1'
DEFAULT_SEED = 42


def master_seed():
    value = int(os.environ.get('COMPARISON_SEED', str(DEFAULT_SEED)))
    if not 0 <= value < 2**32:
        raise ValueError('COMPARISON_SEED must be a uint32')
    if 'ONLINE_CP_SEED' in os.environ and int(os.environ['ONLINE_CP_SEED']) != value:
        raise ValueError('ONLINE_CP_SEED differs from the shared comparison seed')
    return value


def schedule_seed(seed, *parts):
    value = '|'.join(map(str, (FORMAT, seed, *parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'little')


@contextmanager
def cpu_rng(seed):
    """Isolate CPU worker draws without initializing/reseeding CUDA in workers."""
    states = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(states[2])


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SeededCall:
    """Picklable phase wrapper; one call per native sample, in each worker."""
    def __init__(self, loader, function, phase):
        self.loader, self.function, self.phase = loader, function, phase

    def __call__(self, *args, **kwargs):
        with cpu_rng(self.loader.next_phase_seed(self.phase)):
            return self.function(*args, **kwargs)


class PairedLoaderMixin:
    def __init__(self, *args, comparison_stage='train', **kwargs):
        self.comparison_seed = master_seed()
        self.comparison_stage = comparison_stage
        self.comparison_epoch = self.comparison_worker = self.comparison_batch = 0
        self._phase_counts = {}
        super().__init__(*args, **kwargs)
        self.get_do_oversample = SeededCall(self, self.get_do_oversample, 'oversample')
        if self.transforms is not None:
            self.transforms = SeededCall(self, self.transforms, 'standard_augmentation')

    def set_epoch(self, epoch):
        self.comparison_epoch, self.comparison_batch = int(epoch), 0

    def set_thread_id(self, thread_id):
        super().set_thread_id(thread_id)
        self.comparison_worker, self.comparison_batch = int(thread_id), 0

    def next_phase_seed(self, phase):
        ordinal = self._phase_counts.get(phase, 0)
        self._phase_counts[phase] = ordinal + 1
        return schedule_seed(self.comparison_seed, self.comparison_stage,
                             self.comparison_epoch, self.comparison_worker,
                             self.comparison_batch, phase, ordinal)

    def generate_train_batch(self):
        self._phase_counts = {}
        cp_seed = self.next_phase_seed('cp')
        self._comparison_cp_rng = np.random.default_rng(cp_seed)
        if hasattr(self._data, 'set_cp_schedule'):
            self._data.set_cp_schedule(cp_seed)
        if hasattr(self._data, 'set_comparison_batch'):
            self._data.set_comparison_batch(self.next_phase_seed('original_cp'))
        with cpu_rng(self.next_phase_seed('loader')):
            batch = super().generate_train_batch()
        self.comparison_batch += 1
        return batch

    def get_indices(self):
        with cpu_rng(self.next_phase_seed('case_selection')):
            return super().get_indices()

    def get_bbox(self, *args, **kwargs):
        with cpu_rng(self.next_phase_seed('crop')):
            return super().get_bbox(*args, **kwargs)

    def _raw_candidate_crop_bbox(self, *args, **kwargs):
        with cpu_rng(self.next_phase_seed('crop')):
            return super()._raw_candidate_crop_bbox(*args, **kwargs)

    def _rng(self):
        return self._comparison_cp_rng


class PairedNativeLoader(PairedLoaderMixin, nnUNetDataLoader):
    pass


def ordered_augmenter(loader, workers, device):
    if workers == 0:  # Existing explicit DEBUG-only single-process path.
        loader.set_thread_id(0)
        return SingleThreadedAugmenter(loader, None)
    seeds = [schedule_seed(loader.comparison_seed, loader.comparison_stage,
                           loader.comparison_epoch, 'worker', i) for i in range(workers)]
    return MultiThreadedAugmenter(loader, None, num_processes=workers,
                                 num_cached_per_queue=2, seeds=seeds,
                                 pin_memory=device.type == 'cuda', wait_time=0.002)


def paired_dataloaders(trainer, train_loader=PairedNativeLoader, train_kwargs=None):
    if trainer.dataset_class is None:
        trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)
    patch = trainer.configuration_manager.patch_size
    scales = trainer._get_deep_supervision_scales()
    rotation, dummy, initial_patch, mirrors = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    shared_transform = dict(is_cascaded=trainer.is_cascaded,
                           foreground_labels=trainer.label_manager.foreground_labels,
                           regions=trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else None,
                           ignore_label=trainer.label_manager.ignore_label)
    tr_transform = trainer.get_training_transforms(
        patch, rotation, scales, mirrors, dummy,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm, **shared_transform)
    val_transform = trainer.get_validation_transforms(scales, **shared_transform)
    training, validation = trainer.get_tr_and_val_datasets()
    shared_loader = dict(oversample_foreground_percent=trainer.oversample_foreground_percent,
                         sampling_probabilities=None, pad_sides=None,
                         probabilistic_oversampling=trainer.probabilistic_oversampling)
    tr = train_loader(training, trainer.batch_size, initial_patch, patch, trainer.label_manager,
                      transforms=tr_transform, comparison_stage='train',
                      **shared_loader, **(train_kwargs or {}))
    val = PairedNativeLoader(validation, trainer.batch_size, patch, patch, trainer.label_manager,
                             transforms=val_transform, comparison_stage='validation', **shared_loader)
    workers = get_allowed_n_proc_DA()
    trainer._comparison_loaders = (tr, val)
    trainer._comparison_worker_counts = (workers, max(1, workers // 2) if workers else 0)
    trainer._comparison_augmenters = None
    trainer._reset_comparison_epoch()
    trainer.print_to_log_file(dict(randomness_contract=FORMAT, seed=master_seed(),
                                   train_workers=workers, validation_workers=trainer._comparison_worker_counts[1],
                                   ordered_workers=True, independent_cp_rng=True,
                                   phase_seeds='case/crop/oversampling/standard augmentation',
                                   epoch_warmup_batches_discarded=1,
                                   identical_cp_policy=False, bitwise_gpu_reproducibility_verified=False))
    return trainer._comparison_augmenters


class PairedTrainerMixin:
    def initialize(self):
        seed_model(master_seed())
        return super().initialize()

    def _reset_comparison_epoch(self):
        old = self._comparison_augmenters
        if old is not None:
            for augmenter in old:
                if isinstance(augmenter, MultiThreadedAugmenter):
                    augmenter._finish()  # Only this trainer's own loader workers.
        epoch = int(self.current_epoch)
        for loader in self._comparison_loaders:
            loader.set_epoch(epoch)
        self._comparison_augmenters = tuple(
            ordered_augmenter(loader, workers, self.device)
            for loader, workers in zip(self._comparison_loaders, self._comparison_worker_counts))
        self.dataloader_train, self.dataloader_val = self._comparison_augmenters
        for augmenter in self._comparison_augmenters:
            next(augmenter)  # Same native initial warm-up convention at every epoch/resume.
        self._comparison_active_epoch = epoch

    def on_train_epoch_start(self):
        if self._comparison_active_epoch != int(self.current_epoch):
            self._reset_comparison_epoch()
        return super().on_train_epoch_start()
