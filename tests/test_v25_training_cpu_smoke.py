"""Explicit CPU DEBUG control-flow smoke, never native training/quality evidence.

One complete synthetic65/21 epoch executes real all-P loss/backward/AdamW;
CUDA counters and the previously unit-tested calibration receipt are mocked.
The unchanged40-epoch production configuration pauses at an owned boundary.
"""
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from hiercp_v1x import v25_training as training
from hiercp_v1x.u_bridge_training import digest
from tests.test_v25_targets import policy

ROOT=Path(__file__).resolve().parents[1]


class V25ActualLoopCPUDebugSmoke(unittest.TestCase):
    def test_full_epoch_uses_fixed29_complete_probes_B4_and_owned_best_then_pauses(self):
        torch.manual_seed(42)
        net=torch.nn.Linear(2,1)
        train=[f'CPU_DEBUG_train_{i:03d}' for i in range(65)]
        val=[f'CPU_DEBUG_val_{i:03d}' for i in range(21)]
        positives={case:8+int(i>=58) for i,case in enumerate(train)}
        positives.update({case:8 if i<15 else 15 if i==15 else 0 for i,case in enumerate(val)})
        self.assertEqual(sum(positives[c] for c in train),527)
        self.assertEqual(sum(positives[c] for c in val),135)
        class Population:
            def partition_cases(self,partition,**kwargs):
                return train if partition=='inner_train' else val
            def case(self,case,active_u_count=128,**kwargs):
                count=positives[case]+active_u_count
                rows=[dict(id=case+':'+str(i),case_id=case,donor_case_id='CPU_DEBUG_independent_donor',
                    donor_component=1,center=[i,0,0],target=int(i<positives[case])) for i in range(count)]
                return SimpleNamespace(case_id=case,rows=rows,record_ids=tuple(r['id'] for r in rows),
                    positive_indices=tuple(range(positives[case])),
                    unobserved_indices=tuple(range(positives[case],count)),
                    partition='inner_train' if case in train else 'inner_val')
            def manifest(self):
                return dict(CPU_DEBUG_SYNTHETIC=True,train65=train,val21=val,observed_P=662)
        calls=[];prepared=[]
        class Scorer:
            def __init__(self):
                self.net=net;self.amp=False;self.physical_candidate_batch=32
                self.geometry=SimpleNamespace(prepare=lambda active:prepared.append(active))
            def __call__(self,plans,*,epoch,training):
                calls.append(dict(epoch=epoch,training=training,cases=[p.case_id for p in plans],
                    active_u=len(plans[0].unobserved_indices),grad_enabled=torch.is_grad_enabled(),
                    model_training=self.net.training))
                scores=[]
                for plan in plans:
                    features=torch.arange(len(plan.record_ids)*2,dtype=torch.float32).reshape(-1,2)/1000
                    scores.append(self.net(features).flatten())
                total=sum(len(value) for value in scores)
                return SimpleNamespace(scores=tuple(scores),
                    consistency=self.net.bias.square().expand(total),
                    workload=dict(patients=len(plans),upper_invocations=1,CPU_DEBUG_SYNTHETIC=True))
        scorer=Scorer();population=Population()
        gradient=dict(finite=True,missing=[],trainable_parameter_tensors=2,gradient_present=2)
        trial=dict(physical_patient_batch=4,physical_candidate_batch=32,accepted=True,all_P_and_U128=True,
            measurements=[dict(seconds=.1,gradient=copy.deepcopy(gradient)) for _ in range(3)],
            optimizer_updates_on_clone=3,peak_cuda_bytes=100,workload=dict(patients=4,upper_invocations=1))
        contract=dict(parameters=3,trainable_parameters=3,recipient_GT_used_in_forward=False,
            trained_v23_weights_loaded=False,CPU_DEBUG_SYNTHETIC_MODEL=True)
        calibration=dict(format=training.FORMAT,initial_state_sha256=digest(net.state_dict()),world_size=1,
            selected_physical_patient_batch=4,selected_physical_candidate_batch=32,model_contract=contract,
            recipient_GT_used_in_forward=False,debug=True,original_model_and_RNG_preserved=True,
            measured_full_P_U128_backward=True,trials=[trial])
        config=dict(v24_runtime=dict(curriculum=policy(),batch_calibration=calibration,calibration_repeats=3),
            training=dict(lr=1e-4,weight_decay=.01,fused_optimizer=False,amp=False,
                grad_clip=1.,consistency_weight=.1,gradient_accumulation_steps=1))
        root=ROOT/'tmp'/('UNIT_v25_training_smoke_'+uuid.uuid4().hex)
        root.mkdir(parents=True,exist_ok=False)
        self.addCleanup(self.cleanup_owned,root)
        saved={};initial=[p.detach().clone() for p in net.parameters()]
        def save(path,payload):
            saved[Path(path).name]=payload
            if payload['state']['phase']=='training' and len(payload['state']['history'])==1:
                (root/'STOP_AFTER_BATCH').write_text('Explicit CPU DEBUG pause at complete epoch boundary')
        with ExitStack() as stack:
            stack.enter_context(patch.object(training,'_groups',side_effect=lambda model:dict(CPU_DEBUG=model)))
            stack.enter_context(patch.object(training,'atomic_save',side_effect=save))
            for name in ('reset_peak_memory_stats','synchronize'):
                stack.enter_context(patch('torch.cuda.'+name))
            stack.enter_context(patch('torch.cuda.max_memory_allocated',return_value=0))
            stack.enter_context(redirect_stdout(io.StringIO()))
            receipt=training.run_training(net,scorer,population,config,output=root,
                identity=dict(CPU_DEBUG_SYNTHETIC=True),budget=SimpleNamespace(cuda_bytes=1000,check=lambda:None),
                model_contract=contract,physical_patient_batch=4,workers=4,debug=True)
        self.assertEqual(receipt['status'],'PAUSED')
        self.assertEqual(receipt['completed_epochs'],1)
        self.assertEqual(receipt['optimizer_updates'],17)
        self.assertFalse(receipt['full_training'])
        self.assertFalse(receipt['actual_CUDA'])
        self.assertTrue(receipt['debug'])
        self.assertEqual(receipt['connected_parameter_tensors'],2)
        self.assertTrue(all(not torch.equal(old,new) for old,new in zip(initial,net.parameters())))
        updates=[row for row in calls if row['training']]
        self.assertEqual(len(updates),17)
        self.assertEqual([case for row in updates for case in row['cases']].__len__(),65)
        self.assertEqual(set(case for row in updates for case in row['cases']),set(train))
        self.assertTrue(all(2<=len(row['cases'])<=4 and row['epoch']==1 and row['active_u']==8 for row in updates))
        evaluations=[row for row in calls if not row['training']]
        self.assertTrue(all(row['epoch']==29 and not row['grad_enabled'] and not row['model_training'] for row in evaluations))
        history=saved['checkpoint_latest.pt']['state']['history'][0]
        self.assertEqual(history['train_probe']['denominators']['patients'],65)
        self.assertEqual(history['stage_validation']['denominators']['patients'],21)
        self.assertEqual(history['stage_validation']['denominators']['zero_P_patients'],5)
        self.assertEqual(history['full_validation']['active_u'],128)
        self.assertEqual(history['train_probe']['active_u'],8)
        best=saved['checkpoint_best.pt']['state']
        self.assertEqual(best['best']['selected_by'],'fixed_full128_validation_only')
        self.assertEqual(best['pending_epoch_completion']['full_validation']['active_u'],128)
        self.assertEqual(best['best']['epoch'],1)
        self.assertEqual(config['v24_runtime']['curriculum']['total_epochs'],40)

    @staticmethod
    def cleanup_owned(root):
        if root.parent!=ROOT/'tmp' or not root.name.startswith('UNIT_v25_training_smoke_'):
            raise ValueError('CPU DEBUG cleanup escaped the fresh owned folder')
        for path in root.iterdir():
            if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise ValueError('Unexpected CPU DEBUG output cleanup path')
            path.unlink()
        root.rmdir()


if __name__=='__main__':
    unittest.main()
