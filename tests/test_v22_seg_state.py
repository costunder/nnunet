"""CPU schema/failure fixtures; full native CUDA verification has its own driver."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.v22_seg_state import snapshot,seal,validate,content_hash,optimizer_contract


def make_value(contract,epoch=10):
    model=torch.nn.Linear(4,2)
    model.register_parameter('unused',torch.nn.Parameter(torch.ones(3)))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.99,nesterov=True)
    model(torch.ones(2,4)).square().sum().backward();optimizer.step()
    value=snapshot(dict(network_weights=model.state_dict(),optimizer_state=optimizer.state_dict(),
        grad_scaler_state=None,logging={'epoch':epoch-1},_best_ema=.8,current_epoch=epoch,init_args={},
        trainer_name='nnUNetTrainer_250epochs_OnlineRankV22',inference_allowed_mirroring_axes=None,
        online_run_contract=contract,online_optimizer_contract=optimizer_contract(model,optimizer),
        online_rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[])))
    value['online_best']=dict(epoch=epoch,metric=.8,weights_sha256=content_hash(value['network_weights']))
    return seal(value),model,optimizer


class SegStateTests(unittest.TestCase):
    def test_snapshot_preserves_aliases_without_following_live_weights(self):
        original=torch.arange(16,dtype=torch.float32).reshape(4,4)
        saved=snapshot({'encoder':original.detach(),'decoder_encoder':original.detach()})
        self.assertIs(saved['encoder'],saved['decoder_encoder'])
        self.assertNotEqual(saved['encoder'].data_ptr(),original.data_ptr())
        original.add_(10)
        self.assertEqual(float(saved['encoder'][0,0]),0.)

    def test_empty_scaler_cannot_be_saved_as_enabled_native_state(self):
        value,_,_=make_value({'epochs':250});value['grad_scaler_state']={};seal(value)
        with self.assertRaisesRegex(ValueError,'Incomplete GradScaler'):validate(value)

    def test_lazy_unused_state_is_valid_but_used_momentum_cannot_disappear(self):
        value,model,opt=make_value({'epochs':250})
        validate(value,model,opt)
        rows=value['online_optimizer_contract']['groups'][0]
        self.assertTrue(any(not r['initialized'] for r in rows))
        bad=copy.deepcopy(value);bad['optimizer_state']['state']={}
        with self.assertRaises(ValueError):validate(bad)
        seal(bad)
        with self.assertRaisesRegex(ValueError,'coverage'):validate(bad)

    def test_content_finite_shape_dtype_group_and_scaler_mutations(self):
        value,model,opt=make_value({'epochs':250});index=next(iter(value['optimizer_state']['state']))
        def weight(v):v['network_weights']['weight'][0,0]=float('nan')
        def weight_inf(v):v['network_weights']['weight'][0,0]=float('inf')
        def momentum_nan(v):v['optimizer_state']['state'][index]['momentum_buffer'].flatten()[0]=float('nan')
        def shape(v):v['optimizer_state']['state'][index]['momentum_buffer']=torch.zeros(1)
        def dtype(v):v['optimizer_state']['state'][index]['momentum_buffer']=v['optimizer_state']['state'][index]['momentum_buffer'].double()
        def group(v):v['optimizer_state']['param_groups'][0]['params'].reverse()
        def missing(v):v.pop('grad_scaler_state')
        def scaler(v):v['grad_scaler_state']={'scale':float('nan')}
        def finite_weight(v):v['network_weights']['weight'][0,0]+=1
        for mutate in (weight,weight_inf,momentum_nan,shape,dtype,group,missing,scaler,finite_weight):
            with self.subTest(mutation=mutate.__name__):
                bad=copy.deepcopy(value);mutate(bad)
                with self.assertRaises(ValueError):validate(bad,model,opt)
        bad=copy.deepcopy(value);shape(bad);seal(bad)
        with self.assertRaisesRegex(ValueError,'shape/dtype'):validate(bad)

    def test_snapshot_is_immutable_and_live_optimizer_mapping_is_checked(self):
        value,model,opt=make_value({'epochs':250});digest=value['online_state_integrity']['sha256']
        with torch.no_grad():model.weight.add_(2)
        self.assertEqual(value['online_state_integrity']['sha256'],digest);validate(value,model,opt)
        opt.param_groups[0]['params'].reverse()
        with self.assertRaisesRegex(ValueError,'mapping'):validate(value,model,opt)

    def test_atomic_publication_failure_preserves_existing_checkpoint(self):
        from tools.v22_online_checkpoint import OnlineCheckpointMixin
        value,_,_=make_value({'epochs':250})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint_latest.pth'
            OnlineCheckpointMixin._write_online_payload(path,value);before=path.read_bytes()
            with patch('tools.v22_online_checkpoint.os.replace',side_effect=OSError('DEBUG interrupted replace')):
                with self.assertRaises(OSError):OnlineCheckpointMixin._write_online_payload(path,value)
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(list(Path(directory).glob('.online-checkpoint-*')))


if __name__=='__main__':unittest.main()
