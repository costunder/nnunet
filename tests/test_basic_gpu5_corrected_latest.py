"""CPU DEBUG full-state fixtures for the narrowly scoped e6 latest-header cache."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import cache_basic_gpu5_corrected_latest as cache

ROOT=Path(__file__).resolve().parents[1]
TRAINER='nnUNetTrainer_250epochs_BasicCPCandidates2048'


def saved_fixture(count=3,delta=1):
    logging={key:[.7]*count for key in cache.LOGGER_KEYS}
    logging['epoch_start_timestamps']=[100.+2*i for i in range(count)]
    logging['epoch_end_timestamps']=[101.+2*i for i in range(count)]
    logging['dice_per_class_or_region']=[np.array([.95,.75],dtype=np.float32) for _ in range(count)]
    logging['lrs']=[np.float32(.01)]*count
    weights={'conv.weight':torch.arange(12,dtype=torch.float32).reshape(3,4),
             'norm.num_batches_tracked':torch.tensor(31,dtype=torch.int64)}
    return dict(current_epoch=count+delta,trainer_name=TRAINER,network_weights=weights,
        optimizer_state={'state':{0:{'momentum_buffer':torch.arange(12,dtype=torch.float32).reshape(3,4)}},
                         'param_groups':[{'params':[0],'lr':.003,'momentum':.99}]},
        grad_scaler_state={'scale':65536.,'growth_factor':2.,'backoff_factor':.5,
                          'growth_interval':2000,'_growth_tracker':14},
        logging=logging,init_args={'fold':0,'device':torch.device('cpu'),'dataset_json':{'name':'DEBUG'}},
        _best_ema=.76,inference_allowed_mirroring_axes=(0,1,2))


class EpochCorrectionTests(unittest.TestCase):
    def test_only_header_changes_all_full_state_references_and_bytes_preserved(self):
        saved=saved_fixture()
        original=cache.state_fingerprint(saved)
        corrected,proof=cache.corrected_state(saved,trainer=TRAINER)
        self.assertEqual(saved['current_epoch'],4)
        self.assertEqual(corrected['current_epoch'],3)
        for key in saved:
            if key!='current_epoch':self.assertIs(corrected[key],saved[key])
        self.assertEqual(cache.state_fingerprint(corrected),original)
        self.assertEqual(proof['logger_lengths'],{key:3 for key in cache.LOGGER_KEYS})
        self.assertEqual(proof['header_delta'],1)

    def test_already_correct_whole_latest_is_admitted_without_header_fabrication(self):
        saved=saved_fixture(delta=0)
        corrected,proof=cache.corrected_state(saved,trainer=TRAINER)
        self.assertEqual(corrected['current_epoch'],saved['current_epoch'])
        self.assertEqual(proof['header_delta'],0)
        self.assertEqual(cache.state_fingerprint(corrected),cache.state_fingerprint(saved))

    def test_partial_loggers_and_bad_timestamps_are_never_promoted(self):
        for key in cache.LOGGER_KEYS:
            saved=saved_fixture();saved['logging'][key].pop()
            with self.assertRaisesRegex(ValueError,'eight logger lengths equal'):
                cache.corrected_state(saved,trainer=TRAINER)
        saved=saved_fixture();saved['logging']['epoch_end_timestamps'][-1]=0.
        with self.assertRaisesRegex(ValueError,'Partial epoch timestamps'):
            cache.corrected_state(saved,trainer=TRAINER)

    def test_unknown_delta_missing_full_state_or_wrong_trainer_fail_closed(self):
        for delta in (-1,2):
            with self.assertRaisesRegex(ValueError,'one-step latest-header'):
                cache.corrected_state(saved_fixture(delta=delta),trainer=TRAINER)
        for key in ('network_weights','optimizer_state','grad_scaler_state','init_args','_best_ema'):
            saved=saved_fixture();saved.pop(key)
            with self.assertRaises(ValueError):cache.corrected_state(saved,trainer=TRAINER)
        with self.assertRaisesRegex(ValueError,'trainer name'):
            cache.corrected_state(saved_fixture(),trainer='another_trainer')

    def test_fingerprint_detects_model_optimizer_scaler_metric_or_ema_mutation(self):
        original=saved_fixture();expected=cache.state_fingerprint(original)
        changes=[lambda value:value['network_weights']['conv.weight'].add_(1),
                 lambda value:value['optimizer_state']['state'][0]['momentum_buffer'].add_(1),
                 lambda value:value['grad_scaler_state'].update(scale=32768.),
                 lambda value:value['logging']['dice_per_class_or_region'][0].__setitem__(1,np.float32(.3)),
                 lambda value:value.update(_best_ema=.2)]
        for mutate in changes:
            changed=copy.deepcopy(original);mutate(changed)
            self.assertNotEqual(cache.state_fingerprint(changed),expected)


class SerializedRollingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='gpu5_header_DEBUG_',dir=ROOT/'tmp')
        self.root=Path(self.temporary.name).resolve()
        self.original=self.root/'original'
        self.output=self.root/'derived'
        self.original.mkdir();self.output.mkdir();(self.output/'epochs').mkdir()
        self.source=self.original/'checkpoint_latest.pth'
        self.real_fstat=os.fstat

    def tearDown(self):self.temporary.cleanup()

    def own_fstat(self,fd):
        value=self.real_fstat(fd)
        # DEBUG Windows fixture supplies Linux ownership only for the reader's
        # UID assertion; all actual held device/inode/size/mtime bytes remain.
        return types.SimpleNamespace(st_dev=value.st_dev,st_ino=value.st_ino,st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,st_uid=cache.UID,st_mode=value.st_mode)

    def derive(self,prior=0):
        with patch.object(cache,'exact',side_effect=lambda path,**kwargs:Path(path)),\
             patch.object(cache.os,'fstat',side_effect=self.own_fstat):
            return cache.derive_checkpoint(self.source,self.output,trainer=TRAINER,prior_completed=prior)

    def test_serialized_full_state_verification_and_rolling_evidence_leave_original_bytes_untouched(self):
        torch.save(saved_fixture(),self.source)
        before=cache.sha(self.source)
        first=self.derive()
        self.assertEqual(cache.sha(self.source),before)
        self.assertEqual(torch.load(self.source,map_location='cpu',weights_only=False)['current_epoch'],4)
        derived=torch.load(self.output/'CURRENT.pth',map_location='cpu',weights_only=False)
        self.assertEqual(derived['current_epoch'],3)
        self.assertEqual(cache.state_fingerprint(derived),first['state_except_epoch_sha256'])
        self.assertTrue(first['original_torch_storage_locations_preserved'])
        torch.save(saved_fixture(count=4),self.source)
        second=self.derive(prior=3)
        self.assertEqual(second['completed_epochs'],4)
        self.assertEqual(len(list((self.output/'epochs').glob('*.json'))),2)
        self.assertEqual([path.name for path in self.output.glob('*.pth')],['CURRENT.pth'])

    def test_regression_or_partial_source_preserves_previous_valid_rolling_cache(self):
        torch.save(saved_fixture(),self.source);first=self.derive()
        prior=cache.sha(self.output/'CURRENT.pth')
        torch.save(saved_fixture(count=2),self.source)
        with self.assertRaisesRegex(ValueError,'regressed'):self.derive(prior=3)
        self.assertEqual(cache.sha(self.output/'CURRENT.pth'),prior)
        partial=saved_fixture(count=4);partial['logging']['epoch_end_timestamps'].pop()
        torch.save(partial,self.source)
        with self.assertRaisesRegex(ValueError,'eight logger lengths equal'):self.derive(prior=3)
        self.assertEqual(cache.sha(self.output/'CURRENT.pth'),prior)


if __name__=='__main__':unittest.main()
