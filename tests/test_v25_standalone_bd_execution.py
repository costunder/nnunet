"""CPU DEBUG checks; no GPU/server/training is executed."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from hiercp_v1x import v25_standalone_bd_resources as resources
from tools import run_v25_bd_execution as execution,run_v25_bd_continuation as controller


def identity(pid):return dict(pid=pid,create_time=1700000000.+pid,command=['python','owned-'+str(pid)],UID=41833)


class BDExecutionTests(unittest.TestCase):
    def seal(self,group):
        h='a'*64
        slots={}
        for arm in 'BD':
            gpu,cpus=resources.MAPPING[arm]
            cp=dict(path=str(group/arm/'training/checkpoint_latest.pt'),sha256=h,identity_sha256=h,content_sha256=h,
                epoch=2,updates=34 if arm=='B'else 27,phase='full_validation'if arm=='B'else'training',
                train_position=65 if arm=='B'else 40,evaluation_position=18 if arm=='B'else 0,
                active_u=16 if arm=='B'else 128,status='RUNNING'if arm=='B'else'PAUSED')
            slots[arm]=dict(arm=arm,gpu=gpu,CPU_affinity=cpus,direct_proof=dict(path=arm+'proof',sha256=h),
                checkpoint=cp,previous_actor=identity(90+ord(arm)))
        return dict(format=resources.FORMAT,group_root=str(group),private_code=str(group/'newcode'),previous_code=str(group/'oldcode'),
            controller_identity=identity(1),slots=slots,sibling_manifest=dict(path='oldmanifest',sha256=h),
            sibling_controller_identity=identity(2),sibling_slots={a:dict(path=a+'proof',sha256=h)for a in 'AC'},
            preserved_files={'source':h},runtime_files={'runtime':h},created_at=1.,
            scientific_configuration_unchanged=True,no_process_signaled=True)

    def test_B_fixed_validation_cursor_and_D_dynamic_PAUSED_are_required(self):
        with tempfile.TemporaryDirectory()as temporary:
            seal=self.seal(Path(temporary));self.assertIs(resources.validate_seal(seal),seal)
            seal['slots']['B']['checkpoint']['evaluation_position']=17
            with self.assertRaisesRegex(ValueError,'eval18'):resources.validate_seal(seal)
            seal['slots']['B']['checkpoint']['evaluation_position']=18
            seal['slots']['D']['checkpoint'].update(updates=39,train_position=20,epoch=3)
            resources.validate_seal(seal)
            seal['slots']['D']['checkpoint']['status']='RUNNING'
            with self.assertRaisesRegex(ValueError,'safely paused'):resources.validate_seal(seal)

    def test_slots_remain_gpu5_42to47_and_gpu4_0to5(self):
        with tempfile.TemporaryDirectory()as temporary:
            seal=self.seal(Path(temporary));seal['slots']['B']['gpu']=4
            with self.assertRaises(ValueError):resources.validate_seal(seal)

    def test_D_pause_train_probe_retains_65_case_cursor_and_validation_21(self):
        with tempfile.TemporaryDirectory()as temporary:
            seal=self.seal(Path(temporary));cp=seal['slots']['D']['checkpoint']
            cp.update(phase='train_probe',train_position=65,evaluation_position=60)
            resources.validate_seal(seal)
            cp.update(phase='full_validation',evaluation_position=22)
            with self.assertRaisesRegex(ValueError,'safely paused'):resources.validate_seal(seal)
            cp['evaluation_position']=21;resources.validate_seal(seal)
            seal=self.seal(Path(temporary));seal['slots']['D']['CPU_affinity']=list(range(42,48))
            with self.assertRaises(ValueError):resources.validate_seal(seal)
            seal=self.seal(Path(temporary));seal['hidden_subset']=True
            with self.assertRaises(ValueError):resources.validate_seal(seal)

    def test_native_clone_routes_exact_original_downstream_and_retains_B2_B4(self):
        argv=[sys.executable,'-B','-u',str(execution.ROOT/'tools/run_v25_downstream.py'),
            'clone-debug-worker','--physical-batch','4','--native','actualnative','--output','clone']
        adapted=execution.adapt_native_child(argv)
        self.assertEqual(adapted[3],str(execution.ROOT/'tools/run_v25_bd_execution.py'))
        self.assertEqual(adapted[4],'--native');self.assertEqual(adapted[5:],argv[4:])

    def test_native_training_preserves_optimizer_resume_and_best_flags(self):
        argv=[sys.executable,'-B','-u','-c',execution.NATIVE_ENTRY,'730','3d_fullres','0','--c','--val_best']
        adapted=execution.adapt_native_child(argv)
        self.assertEqual(adapted[5:],argv[5:]);self.assertEqual(adapted[4],execution.NATIVE_BOOTSTRAP+argv[4])
        self.assertIn('v25_standalone_bd_resources',adapted[4])
        with self.assertRaises(ValueError):execution.adapt_native_child([sys.executable,'-B','-u','-c','unknown()'])

    def test_native_private_clone_callback_is_also_guarded(self):
        original=Mock();pipeline=SimpleNamespace(_child=original);native=SimpleNamespace(_child=original,_MODULES={'NATIVE_PIPELINE':pipeline})
        execution.install_native_children(native);self.assertIs(native._child,pipeline._child)
        argv=[sys.executable,'-B','-u','-c',execution.NATIVE_ENTRY,'--val_best']
        env={'exact':'env'};pipeline._child(argv,env,'log')
        self.assertIs(original.call_args.args[1],env)
        self.assertEqual(original.call_args.args[0][-1],'--val_best')

    def test_stage_commands_use_each_exact_latest_without_prepare_or_calibration(self):
        args=SimpleNamespace(native_experiment=Path('native'),inventory=Path('inventory'),stunet_checkpoint=Path('official'),
            input_cache=Path('fullcache'),group=Path('group'),storage_admission=Path('storage'),source_bank=Path('642bank'),
            baseline_preprocessed=Path('baseline'),execution_manifest=Path('seal'),project=Path('project'),
            basic_validation=Path('basic26'),evaluator_root=Path('evaluator'))
        for arm,slot in self.seal(args.group)['slots'].items():
            argv=controller.stage_command(args,slot,'train')
            self.assertEqual(argv[argv.index('--arm')+1],arm)
            self.assertEqual(argv[argv.index('--gpu')+1],str(slot['gpu']))
            self.assertEqual(argv[argv.index('--resume')+1],slot['checkpoint']['path'])
            self.assertNotIn('--parallel-prepare',argv);self.assertNotIn('--independent-prepare',argv)
            self.assertEqual(controller.native_command(args,slot)[4],'--native')

    def test_other_BD_slot_never_admitted_as_target_GPU_owner(self):
        controller_actor=Mock();controller_actor.cpu_affinity.return_value=sorted([*range(6),*range(42,48)])
        B=Mock();B.cpu_affinity.return_value=list(range(42,48));D=Mock();D.cpu_affinity.return_value=list(range(6))
        tree={1:controller_actor,11:B,12:D}
        lease=object.__new__(resources.StandaloneBDAllocationLease)
        lease.request=dict(arm='B',gpu=5,GPU_UUID='GPU-B',controller_identity=identity(1),CPU_affinity=list(range(42,48)))
        with (patch.object(resources,'_stage_tree',return_value={11:B}),patch.object(resources._BASE,'_gpu')as original):
            lease._gpu(tree,initial=True)
        original.assert_called_once_with(lease,{11:B},initial=True)
        with (patch.object(resources,'_stage_tree',return_value={11:B}),
                patch.object(resources.direct,'_query',side_effect=['5, GPU-B, 49152, 45000, Disabled','12, GPU-B'])):
            with self.assertRaisesRegex(RuntimeError,'Foreign or unrelated'):lease._gpu(tree,initial=True)

    def test_exact_stage_origin_rejects_other_arm_with_forged_matching_CPU(self):
        with tempfile.TemporaryDirectory()as temporary:
            group=Path(temporary);seal=self.seal(group);monitor=group/'B/monitor/status.json'
            monitor.parent.mkdir(parents=True)
            B=Mock();B.pid=11;B.cpu_affinity.return_value=list(range(42,48));B.children.return_value=[]
            D=Mock();D.pid=12;D.cpu_affinity.return_value=list(range(42,48))
            argv=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
                str(Path(seal['private_code'])/'tools/run_v25_bd_execution.py'),'--arm','B','--gpu','5',
                '--output',str(group/'B'),'--allocation-request','Bproof','--continuation-seal','seal',
                '--mode','train']
            value=dict(controller_pid=1,child_pid=11,child_create_time=1700000011.,child_command=argv,UID=41833,stage='train')
            monitor.write_text(json.dumps(value))
            lease=object.__new__(resources.StandaloneBDAllocationLease);lease.group=group
            lease.request=dict(arm='B',gpu=5,GPU_UUID='GPU-B',controller_identity=identity(1),CPU_affinity=list(range(42,48)))
            tree={1:Mock(),11:B,12:D}
            with (patch.object(resources.direct,'_owned_regular',side_effect=lambda p:p),
                    patch.object(resources,'load_seal',return_value=seal),patch.object(resources.direct,'_exact',return_value=B),
                    patch.dict(os.environ,{resources.ENV_PATH:'seal'})):
                self.assertEqual(resources._stage_tree(lease,tree),{11:B})
                with patch.object(resources.direct,'_query',side_effect=['5, GPU-B, 49152, 45000, Disabled','12, GPU-B']):
                    with self.assertRaisesRegex(RuntimeError,'Foreign or unrelated'):lease._gpu(tree,initial=True)
                value['child_command']=argv.copy();value['child_command'][value['child_command'].index('--arm')+1]='D'
                monitor.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError,'Own per-arm'):resources._stage_tree(lease,tree)

    def test_combined_RAM_uses_original_416_budget_and_leaves_CUDA_tree_unchanged(self):
        lease=object.__new__(resources.StandaloneBDAllocationLease);lease.actual_resources={}
        controller_actor=Mock();own=Mock();tree={1:controller_actor,11:own};union=dict(tree);union[2]=Mock()
        with (patch.object(resources,'combined_resource_tree',return_value=union),patch.object(resources._BASE,'_resources')as original):
            lease._resources(controller_actor,own,tree)
        original.assert_called_once_with(lease,controller_actor,own,union)
        self.assertEqual(set(tree),{1,11});self.assertTrue(lease.actual_resources['aggregate416GiB_unchanged'])

    def test_AC_missing_completion_verifies_only_AC_in_isolated_CPU_process(self):
        lease=SimpleNamespace(group=Path('group'))
        with (patch.object(resources.direct,'_owned_regular',side_effect=lambda p:p),
                patch.object(resources,'sha',return_value='a'*64),
                patch.object(resources.subprocess,'check_output',return_value='{"actual_full_AC_completion_verified": true}')as run):
            resources._completed_AC(lease);resources._completed_AC(lease)
        self.assertEqual(run.call_count,1)
        self.assertIn('for arm in "AC"',run.call_args.args[0][3])
        self.assertNotIn('"ABC"',run.call_args.args[0][3])
        self.assertEqual(len(lease._completed_AC_files),2)

    def test_missing_certificate_does_not_admit_dead_AC(self):
        lease=SimpleNamespace(group=Path('group'))
        with patch.object(resources.direct,'_owned_regular',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):resources._completed_AC(lease)
        self.assertFalse(hasattr(lease,'_completed_AC_files'))

    def test_wrapper_installs_same_allocator_and_source_guard_before_frozen(self):
        from tools import run_v25 as frozen,run_v25_execution as original
        from hiercp_v1x import v25_upper_memory_runtime as upper,v25_provider_memory_runtime as provider
        for arm,gpu in (('B',5),('D',4)):
            with tempfile.TemporaryDirectory()as temporary:
                output=Path(temporary)/arm;output.mkdir();calls=[];seal=self.seal(output.parent)
                args=SimpleNamespace(arm=arm,gpu=gpu,mode='train',output=output,
                    allocation_request=Path(arm+'proof'),resume=Path(seal['slots'][arm]['checkpoint']['path']))
                with ExitStack()as stack:
                    stack.enter_context(patch.object(resources,'install_from_environment',side_effect=lambda:calls.append('resource')or{}))
                    stack.enter_context(patch.object(resources,'load_seal',return_value=seal))
                    stack.enter_context(patch.object(frozen,'parse',return_value=args))
                    admit=stack.enter_context(patch.object(upper,'install_upper_memory_runtime',side_effect=lambda **k:calls.append('source')or{}))
                    stack.enter_context(patch.object(provider,'install_runtime',side_effect=lambda:calls.append('provider')or{}))
                    stack.enter_context(patch.object(frozen,'publish'))
                    stack.enter_context(patch.object(original,'main',side_effect=lambda a:calls.append('frozen')or 0))
                    execution.main(['--mode','train'])
                self.assertEqual(calls,['resource','source','provider','frozen'])
                self.assertIn('host_memory.py',{p.name for p in admit.call_args.kwargs['runtime_files']})

    def test_two_chains_start_concurrently_and_one_failure_does_not_stop_sibling(self):
        started={arm:threading.Event()for arm in 'BD'};finished=[];reports=[]
        def chain(arm):
            started[arm].set()
            if not started['D'if arm=='B'else'B'].wait(3):raise AssertionError('BD chain was serialized')
            if arm=='B':raise RuntimeError('debug owned B failure')
            finished.append(arm)
        failures=controller.run_chains(chain,reports.append)
        self.assertEqual(finished,['D']);self.assertEqual(failures[0]['arm'],'B');self.assertEqual(len(reports),1)

    def test_each_final_completion_uses_isolated_CPU_interpreter_without_parent_ENV_changes(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-parent','V25_NATIVE_REQUEST':'parent-proof'}):
            before=os.environ.copy()
            with patch.object(controller.subprocess,'check_output',return_value='{"actual_full_completion_verified":true}')as run:
                controller.verify_arm_completion(Path('group'),'B');controller.verify_arm_completion(Path('group'),'D')
            self.assertEqual(os.environ.copy(),before)
            self.assertEqual([call.args[0][-1]for call in run.call_args_list],['B','D'])
            for call in run.call_args_list:
                self.assertEqual(call.args[0][:3],[sys.executable,'-B','-c'])
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertIsNot(call.kwargs['env'],os.environ)


if __name__=='__main__':unittest.main()
