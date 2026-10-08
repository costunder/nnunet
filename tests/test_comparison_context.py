"""CPU UNIT checks for compatibility keys; no actual CUDA measurement."""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x.comparison_context import gpu_cache_context


class ContextTests(unittest.TestCase):
    def context(self, net, **kwargs):
        with patch('torch.cuda.current_device', return_value=0), patch(
                'torch.cuda.get_device_properties', return_value=SimpleNamespace(
                    uuid='GPU-UNIT-device', name='UNIT', total_memory=40 * 2**30, major=8, minor=6)):
            return gpu_cache_context(net, binding_hash='a' * 64, arm='native',
                                     training={'amp': True, 'consistency_weight': .1}, **kwargs)

    def test_weights_do_not_change_key_but_structure_does(self):
        model = torch.nn.Linear(3, 2)
        before = self.context(model)
        with torch.no_grad(): model.weight.add_(1)
        self.assertEqual(before, self.context(model))
        self.assertNotEqual(before['model_structure_sha256'],
                            self.context(torch.nn.Linear(3, 4))['model_structure_sha256'])

    def test_objective_policy_enters_context(self):
        model = torch.nn.Linear(3, 2)
        left = self.context(model, comparison_policy={'objective_kind': 'UNIT_pairwise'})
        right = self.context(model, comparison_policy={'objective_kind': 'UNIT_listwise'})
        self.assertNotEqual(left['loss_identity_sha256'], right['loss_identity_sha256'])

    def test_missing_uuid_does_not_use_unverified_device_name(self):
        with patch('torch.cuda.current_device', return_value=0), patch(
                'torch.cuda.get_device_properties', return_value=SimpleNamespace(name='UNIT')), patch.dict(
                os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}):
            with self.assertRaisesRegex(RuntimeError, 'verified single-device'):
                gpu_cache_context(torch.nn.Linear(3, 2), binding_hash='a' * 64,
                                  arm='native', training={'amp': True})


if __name__ == '__main__':
    unittest.main()
