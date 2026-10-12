"""CPU DEBUG real clone backward; mocked resource measurements are not CUDA evidence."""
import copy
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from hiercp_v1x.v25_training import calibrate_training_batches, validate_calibration

class V25CalibrationCPUUnit(unittest.TestCase):
    def test_faster_measured_B8_cannot_change_B4_or_seventeen_epoch_updates(self):
        from hiercp_v1x.u_bridge_training import digest
        torch.manual_seed(42)
        net=torch.nn.Linear(2,1)
        cases=['CPU_DEBUG_'+str(i) for i in range(8)]
        plans={case:SimpleNamespace(case_id=case,record_ids=tuple(case+'_'+str(i) for i in range(129)),
            positive_indices=(0,)) for case in cases}
        population=SimpleNamespace(partition_cases=lambda *a,**kw:cases,
            case=lambda case,*a,**kw:plans[case])
        class Scorer:
            def __init__(self):
                self.net=net;self.physical_candidate_batch=32
                self.providers={'inner_train':SimpleNamespace(records={record:dict(sampled_two_view_edges=1)
                    for plan in plans.values() for record in plan.record_ids})}
            def __call__(self,supplied,**kwargs):
                features=torch.arange(len(supplied)*129*2,dtype=torch.float32).reshape(len(supplied),129,2)/1000
                values=self.net(features).squeeze(-1)
                return SimpleNamespace(scores=tuple(values.unbind()),consistency=values.new_zeros(values.numel()),
                    case_ids=tuple(plan.case_id for plan in supplied),workload=dict(patients=len(supplied),upper_invocations=1))
        scorer=Scorer();before=digest(net.state_dict());rng=torch.get_rng_state().clone()
        contract=dict(recipient_GT_used_in_forward=False,trained_v23_weights_loaded=False,
            parameters=3,trainable_parameters=3)
        config=dict(v24_runtime=dict(physical_patient_batch_candidates=[4,8],
            physical_candidate_batch_candidates=[32],calibration_repeats=3),
            training=dict(lr=1e-4,weight_decay=.01,fused_optimizer=False,amp=False,grad_clip=1.))
        # Six real CPU clone updates, with explicit synthetic timing/VRAM.
        # B8 is deliberately twice as fast, preventing a vacuous selection test.
        times=[0.,.2,1.,1.2,2.,2.2,3.,3.1,4.,4.1,5.,5.1]
        with ExitStack() as stack:
            stack.enter_context(patch('hiercp_v1x.v25_training._groups',side_effect=lambda model:dict(CPU_DEBUG=model)))
            stack.enter_context(patch('hiercp_v1x.v25_training.time.perf_counter',side_effect=times))
            for name in ('reset_peak_memory_stats','synchronize','empty_cache'):
                stack.enter_context(patch('torch.cuda.'+name))
            stack.enter_context(patch('torch.cuda.max_memory_allocated',return_value=100))
            result=calibrate_training_batches(net,scorer,population,config,
                budget=SimpleNamespace(cuda_bytes=1000,check=lambda:None),model_contract=contract,debug=True)
        self.assertEqual([row['physical_patient_batch'] for row in result['trials']],[4,8])
        self.assertTrue(all(row['accepted'] for row in result['trials']))
        self.assertGreater(result['trials'][1]['patients_per_second'],result['trials'][0]['patients_per_second'])
        self.assertEqual(result['selected_physical_patient_batch'],4)
        self.assertEqual(digest(net.state_dict()),before)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertIs(scorer.net,net)
        validate_calibration(result,net,scorer,contract,4,repeats=3,
            budget=SimpleNamespace(cuda_bytes=1000),debug=True)
        from hiercp_v1x.v25_training import patient_batches
        self.assertEqual(len(patient_batches(list(range(65)),4)),17)

    def test_calibration_rejects_transient_peak_and_actual_RSS_budget_without_losing_safe_trial(self):
            """CPU DEBUG clone updates; mocked resource counters are not native evidence."""
            from hiercp_v1x.historical_evaluation import ResourceBudget
            from hiercp_v1x.u_bridge_training import digest
            net=torch.nn.Linear(2,1)
            cases=['UNIT_'+str(i) for i in range(4)]
            plans={case:SimpleNamespace(case_id=case,record_ids=tuple(case+'_'+str(i) for i in range(129)),
                positive_indices=(0,)) for case in cases}
            population=SimpleNamespace(partition_cases=lambda *a,**kw:cases,case=lambda case,*a,**kw:plans[case])
            class Scorer:
                def __init__(self):
                    self.net=net;self.physical_candidate_batch=32
                    self.providers={'inner_train':SimpleNamespace(records={record:dict(sampled_two_view_edges=1)
                        for plan in plans.values() for record in plan.record_ids})}
                def __call__(self, supplied, **kwargs):
                    if self.physical_candidate_batch==128:
                        # Exercise the exact production ResourceBudget exception,
                        # rather than an invented message or a generic exception.
                        with patch('psutil.Process',return_value=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=2))):
                            ResourceBudget(1000,1).check()
                    features=torch.arange(len(supplied)*129*2,dtype=torch.float32).reshape(len(supplied),129,2)/1000
                    values=self.net(features).squeeze(-1)
                    return SimpleNamespace(scores=tuple(values.unbind()),consistency=values.new_zeros(values.numel()),
                        case_ids=tuple(plan.case_id for plan in supplied),workload=dict(patients=len(supplied),upper_invocations=1))
            scorer=Scorer();before=digest(net.state_dict())
            contract=dict(recipient_GT_used_in_forward=False,trained_v23_weights_loaded=False,
                parameters=3,trainable_parameters=3)
            config=dict(v24_runtime=dict(physical_patient_batch_candidates=[4],
                physical_candidate_batch_candidates=[32,64,128],calibration_repeats=3),
                training=dict(lr=1e-4,weight_decay=.01,fused_optimizer=False,amp=False,grad_clip=1.))
            with ExitStack() as stack:
                stack.enter_context(patch('hiercp_v1x.v25_training._groups',side_effect=lambda model:dict(CPU_DEBUG=model)))
                stack.enter_context(patch('torch.cuda.is_available',return_value=False))
                for name in ('reset_peak_memory_stats','synchronize','empty_cache'):
                    stack.enter_context(patch('torch.cuda.'+name))
                stack.enter_context(patch('torch.cuda.memory_allocated',return_value=0))
                stack.enter_context(patch('torch.cuda.max_memory_allocated',side_effect=[100,1001]))
                result=calibrate_training_batches(net,scorer,population,config,
                    budget=SimpleNamespace(cuda_bytes=1000,check=lambda:None),model_contract=contract,debug=True)
            self.assertEqual(result['selected_physical_candidate_batch'],32)
            self.assertEqual([row['accepted'] for row in result['trials']],[True,False,False])
            self.assertEqual([row.get('failure') for row in result['trials']],[None,'CUDA_BUDGET','RAM_BUDGET'])
            self.assertEqual(result['trials'][1]['peak_cuda_bytes'],1001)
            self.assertEqual(result['trials'][2]['error'],'Historical evaluation RSS budget exceeded; no cases skipped')
            self.assertEqual(digest(net.state_dict()),before)
            validate_calibration(result,net,scorer,contract,4,repeats=3,
                budget=SimpleNamespace(cuda_bytes=1000),debug=True)
