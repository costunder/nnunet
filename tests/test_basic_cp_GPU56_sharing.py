"""CPU DEBUG tests for explicit three-experiment GPU5/6 admission."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from hiercp_v1x import basic_cp_GPU56_sharing as sharing
from hiercp_v1x import basic_cp_ablation_resources as common
from tests.test_basic_cp_seven_resources import Actor


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.uuids = {'5': 'GPU-e41d3421-8819-0a49-4a97-6c4518a54f5f',
                      '6': 'GPU-e2ad7465-3573-504b-3913-4277bd99b2ed'}
        self.evidence = dict(experiments={'5': dict(peak_cuda_bytes=12359319040,
            model=dict(parameters=102350575, physical_batch=2, epochs=250, patch_size=[128]*3))})
        self.receipt = dict(format=sharing.FORMAT, operation=sharing.OPERATION,
            logical_GPU_arms={str(k): v for k, v in sharing.ARMS.items()},
            execution_physical_GPUs={'0':5, '1':None, '2':6, '3':5, '4':6, '5':5, '6':6},
            execution_GPU_UUIDs=self.uuids, max_concurrent_CUDA_experiments_per_GPU=3,
            VRAM_reservation_MiB_by_stage=dict(train=14336, preflight=0, historical_metrics=0),
            VRAM_driver_floor_MiB=1024, VRAM_measurement_evidence=self.write('profile.json', self.evidence))
        self.ref = self.write('sharing.json', self.receipt)
        self.requests = {slot:self.native(slot) for slot in range(7)}

    def write(self, name, value):
        path = self.root/name; raw=json.dumps(value,sort_keys=True).encode(); path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def native(self, slot):
        gpu=self.receipt['execution_physical_GPUs'][str(slot)]
        return dict(gpu=slot, arm=sharing.ARMS[slot], root=str(self.root/('arm'+str(slot))),
            gpu_uuid='GPU-legacy-logical'+str(slot), GPU_memory_total_MiB=46068,
            cpu_affinity=list(range(slot*8,slot*8+8)), code_sha256={sharing.FILE:'a'*64},
            GPU56_sharing_receipt=self.ref, execution_physical_GPU=gpu,
            execution_GPU_UUID=None if gpu is None else self.uuids[str(gpu)])

    def claim(self, slot, stage='train'):
        native=self.requests[slot]
        return dict(gpu=slot, kind='native', stage=stage, command=['python',str(slot)],
            controller_identity=dict(pid=100+slot,create_time=700,UID=41833,command=['python','controller']),
            child_identity=None,captured=[], **sharing.claim_fields(native,stage))

    def state(self, *slots):
        return dict(source_pending=False, claims={str(slot):self.claim(slot) for slot in slots})

    def test_legacy_request_has_no_execution_override_or_sharing_effect(self):
        native=dict(gpu=0,gpu_uuid='GPU-old')
        self.assertIsNone(sharing.validate_sharing_request(native))
        self.assertEqual(sharing.execution_gpu(native),0)
        self.assertEqual(sharing.execution_uuid(native),'GPU-old')
        self.assertEqual(sharing.claim_fields(native,'train'),{})
        self.assertEqual(sharing.verify_claim_capacity({},native),{})

    def test_receipt_targets_only5and6_and_preserves_all_logical_arms(self):
        for slot,native in self.requests.items():
            self.assertEqual(sharing.validate_sharing_request(native),self.receipt)
            self.assertEqual(sharing.execution_gpu(native),native['execution_physical_GPU'])
            self.assertEqual(native['gpu'],slot)

    def test_execution_override_without_receipt_is_rejected(self):
        with self.assertRaises(ValueError):sharing.execution_gpu(dict(gpu=0,execution_physical_GPU=5))

    def test_substituted_receipt_or_profile_bytes_are_rejected(self):
        for name in ('sharing.json','profile.json'):
            path=self.root/name; before=path.read_bytes();path.write_bytes(before+b' ')
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'SHA'):
                sharing.validate_sharing_request(self.requests[0])
            path.write_bytes(before)

    def test_logical_identity_target_or_receipt_pin_mismatch_is_rejected(self):
        for key,value in [('gpu',4),('arm','multi10'),('execution_physical_GPU',0),
                          ('execution_GPU_UUID',self.uuids['6']),('code_sha256',{})]:
            native=dict(self.requests[0],**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):sharing.validate_sharing_request(native)

    def test_old_queue_cannot_dispatch_beside_new_sharing(self):
        with self.assertRaisesRegex(ValueError,'queue'):
            sharing.validate_sharing_request(dict(self.requests[0],GPU_queue_deferred_transfer={}))

    def test_completed_blend_source_generation_and_unmeasured_debug_never_dispatch(self):
        for slot,stage in [(1,'train'),(5,'candidate_bank'),(0,'debug_B4'),(0,'debug_B2')]:
            with self.subTest(slot=slot,stage=stage),self.assertRaises(ValueError):
                sharing.claim_fields(self.requests[slot],stage)

    def test_six_native32_experiments_can_reserve_three_on_each_physical_gpu(self):
        state=self.state(0,2,3,4,5,6)
        result=sharing.verify_claim_capacity(state,self.requests[0])
        self.assertEqual(result['GPU56_VRAM_reservations']['5']['reserved_MiB'],3*14336)
        self.assertEqual(result['GPU56_VRAM_reservations']['6']['CUDA_experiments'],3)
        self.assertEqual(len({c['gpu'] for c in state['claims'].values()}),6)

    def test_fourth_cuda_claim_on_same_physical_gpu_is_not_admitted(self):
        state=self.state(0,3,5);state['claims']['extra']=self.claim(0)
        with self.assertRaises(sharing.ResourceClaimUnavailable):sharing.verify_claim_capacity(state,self.requests[0])

    def test_cpu_metric_claim_reserves_no_cuda_slot(self):
        state=self.state(0,3,5);state['claims']['3']['stage']='historical_metrics'
        state['claims']['3'].update(sharing.claim_fields(self.requests[3],'historical_metrics'))
        result=sharing.verify_claim_capacity(state,self.requests[0])
        self.assertEqual(result['GPU56_VRAM_reservations']['5']['CUDA_experiments'],2)

    def test_claim_uuid_reservation_or_receipt_change_is_fatal(self):
        for key,value in [('execution_physical_GPU',6),('execution_GPU_UUID',self.uuids['6']),
                          ('GPU56_VRAM_reservation_MiB',1),('GPU56_sharing_receipt_sha256','b'*64)]:
            state=self.state(0);state['claims']['0'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):sharing.verify_claim_capacity(state,self.requests[0])

    def test_actual_source_completion_remains_required(self):
        state=self.state(0);state['source_pending']=True
        with self.assertRaises(ValueError):sharing.verify_claim_capacity(state,self.requests[0])

    def gpu_fixture(self, slots=(0,3,5), *, free=23000, actors=True):
        state=self.state(*slots);processes={};trees={slot:{} for slot in range(7)}
        for slot in slots:
            process=Actor(1000+slot,['python',str(slot)]);process.cpus=self.requests[slot]['cpu_affinity']
            processes[process.pid]=process;trees[slot][process.pid]=process
            state['claims'][str(slot)]['child_identity']=dict(pid=process.pid,create_time=700,UID=41833,command=process.command)
        lease=SimpleNamespace(request=self.requests[slots[0]],_admit=Mock(),_claim_state=lambda:copy.deepcopy(state),
            registered_requests=self.requests,_group_tree=lambda slot:(trees[slot],{}),
            _terminal_cuda_actor=Mock(return_value=None),_live_outside_cuda_actor=Mock(return_value=dict(
                pid=9999,create_time=700,UID=41833,command=['foreign'],status='running')))
        def query(command,**kwargs):
            if '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current' in command:
                return f'5, {self.uuids["5"]}, NVIDIA RTX A6000, 46068, {free}, Disabled'
            return '\n'.join(f'{pid}, {self.uuids["5"]}, 7700' for pid in processes) if actors else ''
        def process(pid):
            if pid not in processes:raise psutil.NoSuchProcess(pid)
            return processes[pid]
        return lease,trees[slots[0]],processes,query,process

    def test_exact_same_cohort_peer_claims_are_accepted_on_one_gpu(self):
        lease,tree,processes,query,process=self.gpu_fixture()
        with patch.object(common,'_query',side_effect=query),patch.object(psutil,'Process',side_effect=process):
            result=sharing.verify_gpu(lease,tree,idle=False)
        self.assertEqual(set(result['admitted_same_cohort_GPU_compute_identities']),set(processes))
        self.assertEqual(result['physical_GPU'],5)
        self.assertEqual(result['logical_GPU'],0)

    def test_registered_cuda_actor_with_changed_argv_is_fatal(self):
        lease,tree,processes,query,process=self.gpu_fixture();processes[1003].command=['wrong']
        with patch.object(common,'_query',side_effect=query),patch.object(psutil,'Process',side_effect=process):
            with self.assertRaisesRegex(RuntimeError,'argv'):sharing.verify_gpu(lease,tree,idle=False)

    def test_unknown_own_or_foreign_cuda_actor_is_rejected(self):
        for uid in (41833,99999):
            lease,tree,processes,query,process=self.gpu_fixture();foreign=Actor(9999,['foreign'],uid=uid)
            processes[9999]=foreign
            with self.subTest(uid=uid),patch.object(common,'_query',side_effect=query),patch.object(psutil,'Process',side_effect=process):
                with self.assertRaises(sharing.GPUAdmissionUnavailable):sharing.verify_gpu(lease,tree,idle=False)

    def test_reserved_future_peak_must_fit_actual_free_memory(self):
        lease,tree,processes,query,process=self.gpu_fixture(free=20000)
        with patch.object(common,'_query',side_effect=query),patch.object(psutil,'Process',side_effect=process):
            with self.assertRaises(sharing.ResourceClaimUnavailable):sharing.verify_gpu(lease,tree,idle=True)

    def test_separate_driver_query_race_does_not_abort_bound_healthy_model(self):
        lease,tree,processes,query,process=self.gpu_fixture(free=20000)
        with patch.object(common,'_query',side_effect=query),patch.object(psutil,'Process',side_effect=process):
            value=sharing.verify_gpu(lease,tree,idle=False)
        self.assertFalse(value['free_headroom_checked_for_new_admission'])

    def test_individual_experiment_cannot_overrun_its_admitted_vram(self):
        lease,tree,processes,query,process=self.gpu_fixture()
        def overuse(command,**kwargs):
            result=query(command,**kwargs)
            return result.replace('7700','15000') if '--query-compute-apps=pid,gpu_uuid,used_gpu_memory' in command else result
        with patch.object(common,'_query',side_effect=overuse),patch.object(psutil,'Process',side_effect=process):
            with self.assertRaisesRegex(MemoryError,'reservation'):sharing.verify_gpu(lease,tree,idle=False)

    def test_completed_blend_has_no_cuda_execution_permission(self):
        lease=SimpleNamespace(request=self.requests[1],_admit=Mock())
        with patch.object(common,'_query',return_value=''):
            result=sharing.verify_gpu(lease,{},idle=False)
        self.assertTrue(result['completed_experiment_GPU_dispatch_disabled'])

    def test_runner_environment_and_status_project_actual_gpu_without_mutating_native(self):
        from tools import run_basic_cp_RAM512 as runner
        native=self.requests[0];before=copy.deepcopy(native)
        with patch.object(runner.historical,'environment',return_value={'CUDA_VISIBLE_DEVICES':'0'}):
            self.assertEqual(runner.environment(native)['CUDA_VISIBLE_DEVICES'],self.uuids['5'])
        with patch.object(runner.historical,'status',return_value={}) as write:
            runner.status(native,'train','RUNNING')
        self.assertEqual(write.call_args.args[0]['gpu'],5)
        self.assertEqual(write.call_args.kwargs['logical_GPU'],0)
        self.assertEqual(native,before)

    def test_metrics_label_uses_physical_gpu_only_in_new_sharing_mode(self):
        from tools import run_basic_cp_RAM512 as runner
        command=['python','metrics','--gpu-label','GPU0']
        projected=runner._provider_command(self.requests[0],command,'historical_metrics')
        self.assertEqual(projected,['python','metrics','--gpu-label','GPU5'])
        self.assertEqual(command[-1],'GPU0')
        self.assertEqual(runner._provider_command({},command,'historical_metrics'),command)

    def test_foreign_wait_receipt_retains_all_identity_checks_with_actual_execution_gpu(self):
        from tools import run_basic_cp_RAM512 as runner
        native=dict(self.requests[0],bank_sha256='b'*64)
        actor=dict(pid=99,birth=700,UID=41833,command=['python','train'])
        observation=dict(GPU_UUID=self.uuids['5'],physical_GPU=5,pid=999,admitted=False,
            processes_signaled=False,fresh_tree_PIDs=[99],create_time=701,UID=999,
            command=['foreign'],status='running')
        receipt=dict(format=runner._SCOPE['GPU_WAIT_FORMAT'],label='train',stage='train',
            actor_identity=dict(pid=99,create_time=700,UID=41833,command=['python','train']),
            bank_sha256='b'*64,exit_code=75,native_stage_finally_completed=True,
            processes_signaled=False,observation=observation)
        with patch.dict(runner._SCOPE,read=Mock(return_value=receipt)):
            result=runner._validate_native_GPU_wait(native,'train',actor)
            self.assertEqual(result.observation['physical_GPU'],5)
            receipt['native_stage_finally_completed']=False
            with self.assertRaisesRegex(RuntimeError,'finally'):
                runner._validate_native_GPU_wait(native,'train',actor)


if __name__=='__main__':unittest.main()
