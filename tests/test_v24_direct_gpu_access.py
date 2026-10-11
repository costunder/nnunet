"""CPU UNIT guard fixtures only: no real GPU, PBS ownership or launch claim."""
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import v24_direct_gpu_access as guard
PATH=Path(guard.__file__).resolve()


class DirectGuardCPUUnit(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name)
        self.controller=dict(format='CPU_UNIT_controller',GPU_arm=4,GPU_UUID='GPU-UNIT4',CPU_affinity=[0,1,2,3],
            chain_root=str(self.root),source_output=str(self.root/'CPU_UNIT_source'),source_code=str(self.root/'CPU_UNIT_science'),
            source_job=dict(format='v24_interrupted_direct_GNN_source_job_v1'),
            direct_mode=guard.DIRECT_MODE,PBS_owned_reservation=False,context_depth1=True,scheduler=None,
            storage_profile='reuse_historical_Basic642_scores_only_and_full131_Blosc2_readonly',
            scoring_RAM_GiB=64,native_RAM_GiB=48,minimum_runtime_free_disk_GiB=10,
            stages=[dict(action=action)for action in guard.STAGE_ACTIONS])
        self.request=self.root/'direct_GPU_stage_request.json'
        self.request.write_text(json.dumps(guard.stage_request_document(self.controller)),encoding='utf8')
        self.proof_path=self.root/'direct_GPU_access_proof.json'
        self.root_process=SimpleNamespace(pid=100,status=lambda:'running',create_time=lambda:1000.,
            cmdline=lambda:['CPU_UNIT_managed_worker'],uids=lambda:SimpleNamespace(real=41833),
            cpu_affinity=lambda:[0,1,2,3],children=lambda recursive:[])
        self.host_process=SimpleNamespace(pid=200,status=lambda:'running',create_time=lambda:999.,
            cmdline=lambda:['CPU_UNIT_separate_GPU6_container'],uids=lambda:SimpleNamespace(real=41833),
            environ=lambda:dict(PBS_JOBID='129528.ECE-util1'))
        self.proof=dict(format=guard.FORMAT,GPU_arm=4,GPU_UUID='GPU-UNIT4',CPU_affinity=[0,1,2,3],UID=41833,
            PBS_owned_reservation=False,other_PBS_reservations_known_not_owned=True,
            explicit_existing_direct_execution_requested=True,guard_sha256='pending',
            request=dict(path=str(self.request),sha256='pending'),poll_seconds=30,
            minimum_initial_remaining_seconds=28800,native_minimum_initial_remaining_seconds=25200,expiry_safety_reserve_seconds=180,
            host_container_lifetime=dict(role=guard.LIFETIME_ROLE,expires_at_unix=100000.,PBS_job_id='129528.ECE-util1',
                allocation_process=dict(pid=200,create_time=999.,command=['CPU_UNIT_separate_GPU6_container'],UID=41833)))
        self.stack=[]
        for item in (patch.object(guard,'_owned',side_effect=lambda path:Path(path)),patch.object(guard,'_uid',return_value=41833),
                     patch.object(guard.psutil,'Process',side_effect=lambda pid=None:self.host_process if pid==200 else self.root_process),
                     patch.object(guard.time,'time',return_value=1000.),
                     patch.object(guard,'_gpu',return_value=dict(index=4,UUID='GPU-UNIT4',total_MiB=49140,free_MiB=49140)),
                     patch.object(guard,'_apps',return_value=[]),patch.dict(guard.os.environ,{'CUDA_VISIBLE_DEVICES':''})):
            item.start();self.stack.append(item)
        self.proof['guard_sha256']=guard._sha(PATH);self.proof['request']['sha256']=guard._sha(self.request)
        self.write()

    def tearDown(self):
        for item in reversed(self.stack):item.stop()
        self.temp.cleanup()

    def write(self):self.proof_path.write_text(json.dumps(self.proof),encoding='utf8')
    def make(self):return guard.make_guard(self.proof_path,4,[0,1,2,3])

    def test_honest_direct_mode_idle_admission_and_selected_own_process(self):
        check=self.make();guard.os.environ['CUDA_VISIBLE_DEVICES']='4';check.last_gpu_check=-100
        with patch.object(guard,'_apps',return_value=[100]):receipt=check()
        self.assertFalse(receipt['PBS_owned_reservation']);self.assertFalse(receipt['signals_sent'])
        self.assertTrue(receipt['other_PBS_reservations_known_not_owned'])

    def test_cannot_fabricate_pbs_ownership_or_expand_other_gpu(self):
        self.proof['PBS_owned_reservation']=True;self.write()
        with self.assertRaises(ValueError):self.make()
        with self.assertRaises(ValueError):guard.make_guard(self.proof_path,6,[0,1,2,3])

    def test_initial_gpu_must_be_idle_and_have40gib_free(self):
        with patch.object(guard,'_apps',return_value=[100]),self.assertRaises(guard.DirectGPUAccessUnavailable):self.make()
        with patch.object(guard,'_gpu',return_value=dict(index=4,UUID='GPU-UNIT4',total_MiB=49140,free_MiB=40959)),\
             self.assertRaises(guard.DirectGPUAccessUnavailable):self.make()

    def test_any_foreign_or_unmanaged_compute_pid_refuses_cooperatively(self):
        check=self.make();check.last_gpu_check=-100
        with patch.object(guard,'_apps',return_value=[777]),self.assertRaisesRegex(guard.DirectGPUAccessUnavailable,'Foreign or unmanaged'):
            check()

    def test_original_affinity_uid_cvd_and_process_birth_are_rechecked(self):
        check=self.make();guard.os.environ['CUDA_VISIBLE_DEVICES']='4';check();guard.os.environ['CUDA_VISIBLE_DEVICES']=''
        with self.assertRaises(guard.DirectGPUAccessUnavailable):check()
        guard.os.environ['CUDA_VISIBLE_DEVICES']='4'
        with patch.object(guard,'_uid',return_value=41834),self.assertRaises(guard.DirectGPUAccessUnavailable):check()
        self.root_process.create_time=lambda:1001.
        with self.assertRaises(guard.DirectGPUAccessUnavailable):check()

    def test_proof_request_and_guard_sha_are_not_reinterpreted(self):
        check=self.make();self.request.write_text('{}',encoding='utf8')
        with self.assertRaisesRegex(guard.DirectGPUAccessUnavailable,'Immutable'):check()

    def test_gpu6_lifetime_witness_cannot_be_relabelled_as_gpu4_allocation(self):
        self.proof['host_container_lifetime']['role']='GPU4_owned_PBS_allocation';self.write()
        with self.assertRaises(ValueError):self.make()

    def test_expiry_propagates_exception_at_call_boundary_without_signal(self):
        check=self.make()
        with patch.object(guard.time,'time',return_value=99821.),self.assertRaises(guard.DirectGPUAccessUnavailable):check()

    def seal_controller(self):
        self.controller['direct_GPU_access']=dict(path=str(self.proof_path),sha256=guard._sha(self.proof_path),
                                                 stat=guard._file_stat(self.proof_path))
        return self.controller

    def test_new_stage_request_has_no_hash_cycle_and_binds_all_controller_fields(self):
        result=guard.validate_controller_request(self.seal_controller())
        self.assertFalse(result['PBS_owned_reservation'])
        self.assertFalse(result['actual_GPU_idle_check_performed'])
        altered=copy.deepcopy(self.controller);altered['source_output']=str(self.root/'other_source')
        with self.assertRaisesRegex(ValueError,'Every sealed controller'):
            guard.validate_controller_request(altered)

    def test_new_stage_cannot_reuse_old_gnn_proof_or_an_unsealed_controller(self):
        self.proof['format']='v24_existing_direct_GPU_availability_guard_v1';self.write()
        with self.assertRaises(ValueError):guard.validate_controller_request(self.seal_controller())
        with self.assertRaises(ValueError):self.make()
        with self.assertRaises(ValueError):guard.validate_controller_request({})

    def test_nonfinite_deadlines_and_driver_memory_are_rejected(self):
        for key in ('minimum_initial_remaining_seconds','native_minimum_initial_remaining_seconds','expiry_safety_reserve_seconds'):
            baseline=self.proof[key];self.proof[key]=float('nan');self.write()
            with self.assertRaises(ValueError):guard.validate_controller_request(self.seal_controller())
            self.proof[key]=baseline
        self.write()
        with patch.object(guard,'_gpu',return_value=dict(index=4,UUID='GPU-UNIT4',total_MiB=float('nan'),free_MiB=49140)),self.assertRaises(guard.DirectGPUAccessUnavailable):
            self.make()

    def test_native_seven_hour_floor_and_scoring_eight_hour_floor_are_distinct(self):
        self.proof['host_container_lifetime']['expires_at_unix']=28000.;self.write()
        check=guard.make_guard(self.proof_path,4,[0,1,2,3],stage_action='train')
        self.assertEqual(check()['stage_action'],'train')
        with self.assertRaises(guard.DirectGPUAccessUnavailable):self.make()
        with self.assertRaises(ValueError):guard.make_guard(self.proof_path,4,[0,1,2,3],stage_action='debug_unknown')

    def test_actual_descendant_is_pinned_and_reused_pid_or_foreign_uid_is_refused(self):
        check=self.make()
        child=SimpleNamespace(pid=333,status=lambda:'running',create_time=lambda:2000.,
            cmdline=lambda:['CPU_UNIT_native_child'],uids=lambda:SimpleNamespace(real=41833))
        self.root_process.children=lambda recursive:[child]
        def process(pid=None):return self.host_process if pid==200 else child if pid==333 else self.root_process
        with patch.object(guard.psutil,'Process',side_effect=process),patch.object(guard,'_apps',return_value=[333]):
            check.last_gpu_check=-100;check()
            child.create_time=lambda:2001.
            check.last_gpu_check=-100
            with self.assertRaisesRegex(guard.DirectGPUAccessUnavailable,'identity changed'):check()

    def test_early_stage_guard_selects_exact_uuid_and_reuses_current_guard(self):
        from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
        from hiercp_v1x import v24_readonly_native_storage as storage
        self.seal_controller();controller_path=self.root/'request.json'
        controller_path.write_text(json.dumps(self.controller),encoding='utf8')
        env=dict(V24_CONTEXT_DEPTH1_CHAIN_REQUEST=str(controller_path),
                 V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256=guard._sha(controller_path),CUDA_VISIBLE_DEVICES='')
        with patch.dict(pipeline.os.environ,env),patch.object(storage,'regular',side_effect=lambda path:Path(path)):
            first=pipeline.early_stage_guard(stage_action='prepare-current-bank',gpu=4,chain_root=self.root)
            self.assertEqual(pipeline.os.environ['CUDA_VISIBLE_DEVICES'],'GPU-UNIT4')
            with patch.object(guard,'_apps',return_value=[100]):
                second=pipeline.scoring_deadline_guard(gpu=4,chain_root=self.root)
            self.assertIs(first,second);self.assertFalse(second()['PBS_owned_reservation'])
        pipeline._DIRECT_STAGE_GUARDS.clear()

    def test_matched_cli_runs_early_guard_before_original_budget_or_science(self):
        from tools import run_v24_matched_basic_cp as cli
        from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
        args=SimpleNamespace(action='pin-current-gnn',gpu=4)
        class AdmissionFailed(RuntimeError):pass
        with patch.object(pipeline,'early_stage_guard',side_effect=AdmissionFailed('CPU_UNIT_before_science'))as early:
            with self.assertRaisesRegex(AdmissionFailed,'before_science'):cli.main(args)
        early.assert_called_once_with(stage_action='pin-current-gnn',gpu=4)

    def test_original_owned_direct_gpu6_branch_remains_exact_and_does_not_use_v2_guard(self):
        from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
        from hiercp_v1x import v24_readonly_native_storage as storage
        request=dict(context_depth1=True,GPU_arm=6,GPU_UUID='GPU-UNIT6',CPU_affinity=[32,33,34,35],
                     chain_root=str(self.root),scheduler=None,context_depth1_direct_allocation_proof=dict(
                         format='v24_context_depth1_direct_allocation_v1',GPU_arm=6,GPU_UUID='GPU-UNIT6',
                         CPU_affinity=[32,33,34,35],owner_uid=41833,CUDA_VISIBLE_DEVICES='6',expires_at_unix=100000.))
        path=self.root/'request.json';path.write_text(json.dumps(request),encoding='utf8')
        env=dict(V24_CONTEXT_DEPTH1_CHAIN_REQUEST=str(path),V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256=guard._sha(path),CUDA_VISIBLE_DEVICES='6')
        old_affinity=self.root_process.cpu_affinity;self.root_process.cpu_affinity=lambda:[32,33,34,35]
        try:
            with patch.dict(pipeline.os.environ,env),patch.object(storage,'regular',side_effect=lambda path:Path(path)),\
                 patch.object(pipeline.os,'getuid',return_value=41833,create=True),\
                 patch.object(guard,'make_guard',side_effect=AssertionError('Legacy owned direct must not instantiatev2')):
                self.assertIsNone(pipeline.early_stage_guard(stage_action='prepare-current-bank',gpu=6,chain_root=self.root))
                check=pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root);check()
                with patch.object(pipeline.os,'getuid',return_value=41834),self.assertRaises(ValueError):
                    pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root)
        finally:self.root_process.cpu_affinity=old_affinity

    def test_original_native_methods_delegate_exact_returns_and_stop_before_update(self):
        calls=[];access=dict(format=guard.FORMAT,PBS_owned_reservation=False,signals_sent=False)
        class OriginalCPUUnitTrainer:
            def train_step(self,batch):calls.append(('train',batch));return batch
            def validation_step(self,batch):calls.append(('validation',batch));return batch
            def perform_actual_validation(self,**kwargs):calls.append(('actual',kwargs));return kwargs
        original=OriginalCPUUnitTrainer.train_step
        with patch.object(guard,'_NATIVE_STEP_BINDINGS',{}):
            state=dict(allowed=True)
            def check():
                if not state['allowed']:raise guard.DirectGPUAccessUnavailable('CPU_UNIT_foreign_GPU')
                return access
            proof=guard.install_native_step_guards(OriginalCPUUnitTrainer,check)
            trainer=OriginalCPUUnitTrainer();batch=object()
            self.assertIs(trainer.train_step(batch),batch)
            self.assertIs(trainer.validation_step(batch),batch)
            self.assertEqual(trainer.perform_actual_validation(save_probabilities=True),dict(save_probabilities=True))
            self.assertTrue(proof['math_weights_optimizer_unchanged'])
            self.assertEqual(len(calls),3)
            state['allowed']=False
            with self.assertRaises(guard.DirectGPUAccessUnavailable):trainer.train_step(batch)
            self.assertEqual(len(calls),3)
            state['allowed']=True
            OriginalCPUUnitTrainer.train_step=original
            with self.assertRaisesRegex(ValueError,'methods changed'):trainer.validation_step(batch)

    def test_direct_validation_wrapper_preserves_actual_best_loaded_guard(self):
        from hiercp_v1x import v24_native_best_eval_runtime as best
        from tests.test_v24_native_best_eval_runtime import _debug_trainer_class
        base=_debug_trainer_class()
        class CPUUnitMatchedTrainer(base):
            def train_step(self,batch):return batch
            def validation_step(self,batch):return batch
        cases=['DEBUG_case_%02d'%index for index in range(26)]
        best_path=self.root/best.BEST;best_path.write_text('{"current_epoch":17}')
        final_path=self.root/'checkpoint_final.pth';final_path.write_text('{"current_epoch":250}')
        best._install(base,cases,bank_sha256='a'*64)
        checks=[]
        def access():checks.append(True);return dict(format=guard.FORMAT,PBS_owned_reservation=False,signals_sent=False)
        with patch.object(guard,'_NATIVE_STEP_BINDINGS',{}):
            guard.install_native_step_guards(CPUUnitMatchedTrainer,access)
            trainer=CPUUnitMatchedTrainer();trainer.output_folder=str(self.root);trainer.cases=cases
            trainer.load_calls=trainer.validation_calls=0
            trainer.mutate_load=trainer.mutate_validation=trainer.fail_validation=False;trainer.produce_count=26
            trainer.load_checkpoint(final_path)
            with self.assertRaisesRegex(ValueError,'BEST must be loaded last'):trainer.perform_actual_validation()
            self.assertEqual(trainer.validation_calls,0)
            trainer.load_checkpoint(best_path)
            self.assertEqual(trainer.perform_actual_validation(export_probabilities=True),'original validation return')
            receipt=best.validate_best_validation_receipt(self.root/best.RECEIPT,fold=self.root,
                expected_cases=cases,bank_sha256='a'*64)
            self.assertEqual(receipt['checkpoint_name'],best.BEST)
            self.assertFalse(receipt['final_checkpoint_fallback'])
            self.assertEqual(trainer.validation_calls,1);self.assertGreaterEqual(len(checks),3)


if __name__=='__main__':unittest.main()
