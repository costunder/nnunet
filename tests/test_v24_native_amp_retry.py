"""CPU DEBUG simulated native AMP skips; never CUDA/native-performance evidence."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hiercp_v1x import v24_nnunet_cp as proposal


class NativeDecoderDebug(torch.nn.Module):
    def __init__(self):
        super().__init__();self.seg_layers=torch.nn.ModuleList([torch.nn.Linear(2,1) for _ in range(2)])
    def forward(self, x):
        seg_outputs=[head(x) for head in self.seg_layers]
        return seg_outputs[::-1]


class WrongDecoderDebug(NativeDecoderDebug):
    def forward(self,x):
        seg_outputs=[head(x) for head in self.seg_layers]
        return seg_outputs


class NativeModelDebug(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=NativeDecoderDebug()
    def forward(self,x):return self.decoder(self.encoder(x))


class ScalerDebug:
    def __init__(self,enabled=True):self.value=64.;self.enabled=enabled
    def get_scale(self):return self.value
    def is_enabled(self):return self.enabled


class NativeTrainerDebug:
    def __init__(self, mode='one_overflow'):
        self.network=NativeModelDebug().train();self.loss=SimpleNamespace(weight_factors=(1.,0.))
        self.enable_deep_supervision=True;self.grad_scaler=ScalerDebug()
        self.optimizer=torch.optim.SGD(self.network.parameters(),lr=.01,momentum=.9)
        self._online_cp_events=0;self._online_cp_samples=0;self._online_schedule_hash=123
        self._online_native_transport=dict(raw_events=0,native_support_voxels=0)
        self.calls=0;self.observed=[];self.mode=mode
    def train_step(self,batch):
        self.calls+=1
        flags=batch.pop('online_cp_applied');tokens=batch.pop('online_cp_schedule_token')
        self._online_cp_events+=int(flags.sum());self._online_cp_samples+=len(flags)
        self._online_schedule_hash+=int(tokens.sum())
        self._online_native_transport['raw_events']+=int(flags.sum())
        self._online_native_transport['native_support_voxels']+=int(batch.pop('online_cp_native_support_voxels').sum())
        draw=torch.rand(4)
        self.observed.append(dict(data_id=id(batch['data']),target_id=id(batch['target']),draw=draw.clone(),
            flags=flags.copy(),before=[p.detach().clone() for p in self.network.parameters()]))
        self.optimizer.zero_grad(set_to_none=True)
        outputs=self.network(batch['data'])
        loss=sum(weight*(output-batch['target']).square().mean()
                 for weight,output in zip(self.loss.weight_factors,outputs) if weight>0)
        loss.backward()
        overflow=self.calls==1 and self.mode not in ('healthy','finite_no_update','missing_active')
        if self.mode=='missing_active':self.network.decoder.seg_layers[1].weight.grad=None
        if overflow:
            next(self.network.parameters()).grad.fill_(float('inf'))
            if self.mode!='unchanged_scale':self.grad_scaler.value/=2
            if self.mode=='changed_on_overflow':
                with torch.no_grad():next(self.network.parameters()).add_(1)
        elif self.mode!='finite_no_update':self.optimizer.step()
        result=loss.detach().numpy()
        if self.mode=='nonfinite_forward':result=np.asarray(float('nan'))
        return dict(loss=result)


class NativeAMPAdmissionDebug(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713)
        self.batch=dict(data=torch.tensor([[1.,2.],[2.,1.],[-1.,3.],[.2,.4]]),target=torch.ones(4,1),
            online_cp_applied=np.ones(4,dtype=np.uint8),online_cp_schedule_token=np.arange(4,dtype=np.uint64),
            online_cp_native_support_voxels=np.full(4,11,dtype=np.int64))
    def run_update(self,trainer):
        parameters=dict(trainer.network.named_parameters())
        before={name:p.detach().clone() for name,p in parameters.items()}
        with patch('torch.cuda.is_available',return_value=False):
            return proposal._native_clone_step_with_amp_retry(trainer,self.batch,parameters,before)
    def test_same_real_batch_RNG_and_CP_counters_once_after_native_skip(self):
        trainer=NativeTrainerDebug();original_keys=set(self.batch)
        result,proof=self.run_update(trainer)
        self.assertEqual(trainer.calls,2);self.assertTrue(np.isfinite(result['loss']))
        self.assertEqual(set(self.batch),original_keys)
        self.assertEqual([row['data_id'] for row in trainer.observed],[id(self.batch['data'])]*2)
        self.assertEqual([row['target_id'] for row in trainer.observed],[id(self.batch['target'])]*2)
        torch.testing.assert_close(trainer.observed[0]['draw'],trainer.observed[1]['draw'],rtol=0,atol=0)
        self.assertEqual(trainer._online_cp_events,4);self.assertEqual(trainer._online_cp_samples,4)
        self.assertEqual(trainer._online_schedule_hash,129)
        self.assertEqual(trainer._online_native_transport,dict(raw_events=4,native_support_voxels=44))
        self.assertEqual(len(proof['AMP_overflow_attempts']),1)
        self.assertEqual(proof['AMP_overflow_attempts'][0]['scale_before'],64.)
        self.assertEqual(proof['AMP_overflow_attempts'][0]['scale_after'],32.)
        self.assertEqual(proof['official_inactive_parameter_names'],['decoder.seg_layers.0.weight','decoder.seg_layers.0.bias'])
        self.assertEqual(proof['missing_native_gradient_parameters'],proof['official_inactive_parameter_names'])
        self.assertEqual(proof['present_gradient_tensors'],4);self.assertEqual(proof['trainable_parameter_tensors'],6)
        for a,b in zip(trainer.observed[0]['before'],trainer.observed[1]['before']):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_healthy_step_has_no_retry_and_retains_actual_inactive_head(self):
        trainer=NativeTrainerDebug('healthy');_,proof=self.run_update(trainer)
        self.assertEqual(trainer.calls,1);self.assertEqual(proof['AMP_overflow_attempts'],[])
        self.assertEqual(proof['zero_weight_output_indices'],[1])
        self.assertEqual(proof['inactive_decoder_layer_indices'],[0])
    def test_nonfinite_forward_is_immediate_error(self):
        trainer=NativeTrainerDebug('nonfinite_forward')
        with self.assertRaisesRegex(ValueError,'forward loss is nonfinite'):self.run_update(trainer)
        self.assertEqual(trainer.calls,1)
    def test_nonfinite_gradient_without_strict_scale_decrease_is_error(self):
        trainer=NativeTrainerDebug('unchanged_scale')
        with self.assertRaisesRegex(ValueError,'strictly decreased'):self.run_update(trainer)
        self.assertEqual(trainer.calls,1)
    def test_overflow_that_changes_weights_is_never_retried(self):
        trainer=NativeTrainerDebug('changed_on_overflow')
        with self.assertRaisesRegex(ValueError,'changed model weights'):self.run_update(trainer)
        self.assertEqual(trainer.calls,1)
    def test_disabled_AMP_cannot_hide_nonfinite_gradients(self):
        trainer=NativeTrainerDebug();trainer.grad_scaler.enabled=False
        with self.assertRaisesRegex(ValueError,'strictly decreased'):self.run_update(trainer)
    def test_unexpected_missing_active_head_is_immediate_error(self):
        trainer=NativeTrainerDebug('missing_active')
        with self.assertRaisesRegex(ValueError,'active/core gradients'):self.run_update(trainer)
        self.assertEqual(trainer.calls,1)
    def test_finite_gradients_without_update_cannot_be_called_success(self):
        trainer=NativeTrainerDebug('finite_no_update')
        with self.assertRaisesRegex(ValueError,'actual optimizer update'):self.run_update(trainer)
    def test_unproved_decoder_order_or_invalid_loss_weights_rejected_before_forward(self):
        trainer=NativeTrainerDebug();trainer.network.decoder=WrongDecoderDebug()
        with self.assertRaisesRegex(ValueError,'head ordering is unproved'):self.run_update(trainer)
        self.assertEqual(trainer.calls,0)
        trainer=NativeTrainerDebug();trainer.loss.weight_factors=(1.,-.1)
        with self.assertRaisesRegex(ValueError,'finite/nonnegative'):self.run_update(trainer)
        self.assertEqual(trainer.calls,0)


if __name__=='__main__':unittest.main(verbosity=2)
