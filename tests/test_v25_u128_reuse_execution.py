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
from hiercp_v1x import v25_u128_reuse_resources as resources
from tools import run_v25_u128_reuse_execution as execution,run_v25_u128_reuse_continuation as controller


def identity(pid):return dict(pid=pid,create_time=1700000000.+pid,command=['python','owned-'+str(pid)],UID=41833)


class RecoveryExecutionTests(unittest.TestCase):
    def seal(self,group):
        h='a'*64;slots={};preserved={'source':h,str(execution.ROOT/'hiercp_v1x/v25_training.py'):h}
        for arm,(gpu,cpus)in resources.MAPPING.items():
            cp=dict(path=str(group/arm/'training/checkpoint_latest.pt'),sha256=h,identity_sha256=h,content_sha256=h,
                epoch=2,updates=34,phase='stage_validation'if arm=='D'else'full_validation',
                train_position=65,evaluation_position=12 if arm=='D'else 18,
                active_u=resources.INCREMENTS[arm],status='PAUSED')
            slots[arm]=dict(arm=arm,gpu=gpu,CPU_affinity=cpus,direct_proof=dict(path=arm+'proof',sha256=h),
                checkpoint=cp,previous_actor=identity(90+ord(arm)))
            recovery=dict(mode='batch_paused',registration=dict(path=str(group/(arm+'_registration.json')),sha256=h),
                monitor=dict(path=str(group/(arm+'_archived_status.json')),sha256=h),failure_log=None,
                previous_controller_identity=identity(2))
            slots[arm]['recovery']=recovery
            for key in ('registration','monitor'):preserved[recovery[key]['path']]=h
        return dict(format=resources.FORMAT,group_root=str(group),private_code=str(group/'newcode'),previous_code=str(group/'oldcode'),
            controller_identity=identity(1),slots=slots,preserved_files=preserved,runtime_files={'runtime':h},created_at=1.,
            scientific_configuration_unchanged=True,no_process_signaled=True)

    def test_closed_single_controller_four_slots_and_dynamic_paused_cursors_are_required(self):
        with tempfile.TemporaryDirectory()as temporary:
            group=Path(temporary);seal=self.seal(group);resources.validate_seal(seal)
            for arm,key in [('B','evaluation_position'),('D','evaluation_position')]:
                changed=self.seal(group);changed['slots'][arm]['checkpoint'][key]-=1
                resources.validate_seal(changed)
                changed['slots'][arm]['checkpoint']['status']='RUNNING'
                with self.assertRaises(ValueError):resources.validate_seal(changed)
            seal['sibling_manifest']={'path':'obsolete','sha256':'a'*64}
            with self.assertRaisesRegex(ValueError,'all-four'):resources.validate_seal(seal)
            seal=self.seal(group);del seal['slots']['C']
            with self.assertRaises(ValueError):resources.validate_seal(seal)

    def test_original_mapping_and_curriculum_scales_are_exact(self):
        self.assertEqual(resources.MAPPING,{'A':(6,list(range(32,38))),'B':(5,list(range(42,48))),
            'C':(1,list(range(16,22))),'D':(4,list(range(6)))})
        with tempfile.TemporaryDirectory()as temporary:
            group=Path(temporary)
            for arm in 'ABCD':
                seal=self.seal(group);seal['slots'][arm]['gpu']=0
                with self.assertRaises(ValueError):resources.validate_seal(seal)
            seal=self.seal(group);seal['slots']['C']['checkpoint']['active_u']=16
            with self.assertRaises(ValueError):resources.validate_seal(seal)

    def test_dynamic_A_pause_preserves_65_train_probe_cursor_and_21_validation(self):
        with tempfile.TemporaryDirectory()as temporary:
            seal=self.seal(Path(temporary));cp=seal['slots']['A']['checkpoint']
            cp.update(phase='train_probe',train_position=65,evaluation_position=60,updates=51,epoch=3,active_u=24)
            resources.validate_seal(seal)
            cp.update(phase='full_validation',evaluation_position=22)
            with self.assertRaises(ValueError):resources.validate_seal(seal)
            cp['evaluation_position']=21;resources.validate_seal(seal)
            cp['status']='RUNNING'
            with self.assertRaises(ValueError):resources.validate_seal(seal)
            cp['status']='PAUSED'
            seal['slots']['D']['checkpoint']['status']='RUNNING'
            with self.assertRaises(ValueError):resources.validate_seal(seal)

    def test_all_four_resource_documents_share_actual_controller_lifetime_and_cgroup(self):
        with tempfile.TemporaryDirectory()as temporary:
            group=Path(temporary);seal=self.seal(group);slot=seal['slots']['A']
            shared=dict(controller_identity=seal['controller_identity'],group_root=str(group),host_lifetime={'exact':'life'},
                aggregate_CPU_affinity=sorted(c for _,cpus in resources.MAPPING.values()for c in cpus),
                resource_guard_sha256='a'*64,cgroup_admission={'exact':'cgroup'})
            proofs={arm:dict(shared,arm=arm,gpu=value['gpu'],CPU_affinity=value['CPU_affinity'])for arm,value in seal['slots'].items()}
            lease=SimpleNamespace(path=Path('Aproof'),group=group,request=proofs['A'])
            with (patch.object(resources,'load_seal',return_value=seal),patch.object(resources,'sha',return_value='a'*64),
                    patch.object(resources,'_guard_proof',side_effect=lambda p:proofs[p['path'][0]]),
                    patch.object(resources,'_validate_terminal_resume'),
                    patch.object(resources.direct,'validate_direct_document',side_effect=lambda p:p)):
                resources.validate_own_contracts(lease)
                proofs['C']['controller_identity']=identity(2)
                with self.assertRaisesRegex(ValueError,'four-arm'):resources.validate_own_contracts(lease)

    def test_RSS_guard_receives_single_all_four_tree_with_original_416_budget(self):
        lease=object.__new__(resources.U128ReuseAllocationLease);lease.actual_resources={}
        actor=Mock();tree={1:actor,11:Mock(),12:Mock(),13:Mock(),14:Mock()}
        with (patch.object(resources,'validate_own_contracts'),patch.object(resources._BASE,'_resources')as original):
            lease._resources(actor,tree[11],tree)
        original.assert_called_once_with(lease,actor,tree[11],tree)
        self.assertTrue(lease.actual_resources['single_controller_tree'])
        self.assertTrue(lease.actual_resources['aggregate416GiB_unchanged'])
        self.assertEqual(lease.actual_resources['GNN_aggregate_admission_bytes'],288*2**30)

    def test_native_clone_routes_exact_original_downstream_and_retains_B2_B4(self):
        argv=[sys.executable,'-B','-u',str(execution.ROOT/'tools/run_v25_downstream.py'),
            'clone-debug-worker','--physical-batch','4','--native','actualnative','--output','clone']
        adapted=execution.adapt_native_child(argv)
        self.assertEqual(adapted[3],str(execution.ROOT/'tools/run_v25_u128_reuse_execution.py'))
        self.assertEqual(adapted[4],'--native');self.assertEqual(adapted[5:],argv[4:])


    def test_native_training_preserves_optimizer_resume_and_best_flags(self):
        argv=[sys.executable,'-B','-u','-c',execution.NATIVE_ENTRY,'730','3d_fullres','0','--c','--val_best']
        adapted=execution.adapt_native_child(argv)
        self.assertEqual(adapted[5:],argv[5:]);self.assertEqual(adapted[4],execution.NATIVE_BOOTSTRAP+argv[4])
        self.assertIn('v25_u128_reuse_resources',adapted[4])
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
        lease=object.__new__(resources.U128ReuseAllocationLease)
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
                str(Path(seal['private_code'])/'tools/run_v25_u128_reuse_execution.py'),'--arm','B','--gpu','5',
                '--output',str(group/'B'),'--allocation-request','Bproof','--continuation-seal','seal',
                '--mode','train']
            value=dict(controller_pid=1,child_pid=11,child_create_time=1700000011.,child_command=argv,UID=41833,stage='train')
            monitor.write_text(json.dumps(value))
            lease=object.__new__(resources.U128ReuseAllocationLease);lease.group=group;lease.managed={}
            lease.request=dict(arm='B',gpu=5,GPU_UUID='GPU-B',controller_identity=identity(1),CPU_affinity=list(range(42,48)))
            parent=Mock();parent.pid=1;B.parent.return_value=parent;B.ppid.return_value=1
            tree={1:parent,11:B,12:D}
            with (patch.object(resources.direct,'_owned_regular',side_effect=lambda p:p),
                    patch.object(resources,'load_seal',return_value=seal),
                    patch.object(resources.direct,'_exact',side_effect=lambda value:B if value['pid']==11 else parent),
                    patch.object(resources.direct,'_actor',side_effect=lambda process:identity(1) if process is parent else dict(pid=11,create_time=1700000011.,command=argv,UID=41833)),
                    patch.dict(os.environ,{resources.ENV_PATH:'seal'})):
                self.assertEqual(resources._stage_tree(lease,tree),{11:B})
                with patch.object(resources.direct,'_query',side_effect=['5, GPU-B, 49152, 45000, Disabled','12, GPU-B']):
                    with self.assertRaisesRegex(RuntimeError,'Foreign or unrelated'):lease._gpu(tree,initial=True)
                value['child_command']=argv.copy();value['child_command'][value['child_command'].index('--arm')+1]='D'
                monitor.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError,'Own per-arm'):resources._stage_tree(lease,tree)


    def test_new_owned_worker_after_controller_snapshot_is_proven_by_live_ancestry(self):
        lease=SimpleNamespace(request={'controller_identity':identity(1)},managed={})
        origin=identity(11);root=Mock();root.pid=11
        parent=Mock();parent.pid=21;parent.parent.return_value=root;parent.ppid.return_value=11
        worker=Mock();worker.pid=22;worker.parent.return_value=parent;worker.ppid.return_value=21
        for process in (parent,worker):process.cpu_affinity.return_value=list(range(42,48))
        tree={1:Mock(),11:root};observed={21:identity(21),22:identity(22)}
        with (patch.object(resources.terminal,'_descendant',side_effect=lambda lease,p:observed[p.pid]),
                patch.object(resources.direct,'_actor',return_value=origin)):
            selected=resources._current_descendant(lease,worker,origin,tree,list(range(42,48)))
        self.assertEqual(set(selected),{21,22});self.assertEqual(set(tree),{1,11,21,22})
        self.assertEqual(lease.managed,observed)


    def test_snapshot_race_cannot_admit_foreign_ancestry_or_PID_reuse(self):
        lease=SimpleNamespace(request={'controller_identity':identity(1)},managed={})
        origin=identity(11);foreign=Mock();foreign.pid=1
        worker=Mock();worker.pid=22;worker.parent.return_value=foreign;worker.ppid.return_value=1
        worker.cpu_affinity.return_value=list(range(42,48));tree={1:foreign,11:Mock()}
        with patch.object(resources.terminal,'_descendant',side_effect=lambda lease,p:identity(p.pid)):
            with self.assertRaisesRegex(RuntimeError,'outside exact stage ancestry'):
                resources._current_descendant(lease,worker,origin,tree,list(range(42,48)))
        self.assertEqual(set(tree),{1,11});self.assertEqual(lease.managed,{})
        root=Mock();root.pid=11;worker.parent.return_value=root;worker.ppid.return_value=11
        lease.managed[22]=dict(identity(22),create_time=1700000021.)
        with (patch.object(resources.terminal,'_descendant',return_value=identity(22)),
                patch.object(resources.direct,'_actor',return_value=origin)):
            with self.assertRaisesRegex(RuntimeError,'PID/birth/argv identity changed'):
                resources._current_descendant(lease,worker,origin,tree,list(range(42,48)))


    def test_new_stage_workers_trigger_unchanged_aggregate_RAM_readmission_before_CUDA(self):
        lease=object.__new__(resources.U128ReuseAllocationLease)
        lease.request={'controller_identity':identity(1)};lease.self_identity=identity(11)
        controller_actor=Mock();stage_actor=Mock();worker=Mock();tree={1:controller_actor,11:stage_actor};events=[]
        def collect(instance,actual_tree):
            actual_tree[22]=worker;events.append('lineage');return {11:stage_actor,22:worker}
        with (patch.object(resources,'_stage_tree',side_effect=collect),
                patch.object(resources.direct,'_exact',side_effect=lambda value:controller_actor if value['pid']==1 else stage_actor),
                patch.object(lease,'_resources',side_effect=lambda *args:events.append('RAM'))as readmit,
                patch.object(resources._BASE,'_gpu',side_effect=lambda *args,**kwargs:events.append('CUDA'))as gpu):
            lease._gpu(tree,initial=False)
        self.assertEqual(events,['lineage','RAM','CUDA'])
        readmit.assert_called_once_with(controller_actor,stage_actor,tree)
        gpu.assert_called_once_with(lease,{11:stage_actor,22:worker},initial=False)


    def test_gone_or_terminal_new_workers_do_not_gain_CUDA_ownership(self):
        lease=SimpleNamespace(request={'controller_identity':identity(1)},managed={})
        worker=Mock();worker.pid=22;tree={1:Mock(),11:Mock()}
        with patch.object(resources.terminal,'_descendant',return_value=None):
            self.assertIsNone(resources._current_descendant(lease,worker,identity(11),tree,list(range(42,48))))
        self.assertEqual(set(tree),{1,11});self.assertEqual(lease.managed,{})
        with patch.object(resources.terminal,'_descendant',side_effect=resources.direct.psutil.NoSuchProcess(22)):
            with self.assertRaises(resources.direct.psutil.NoSuchProcess):
                resources._current_descendant(lease,worker,identity(11),tree,list(range(42,48)))


    def test_preserved_RUNNING_failure_checkpoints_require_exact_sealed_failure_mode(self):
        with tempfile.TemporaryDirectory()as temporary:
            group=Path(temporary);seal=self.seal(group)
            for arm,slot in seal['slots'].items():
                evidence=slot['recovery'];evidence['mode']='natural_failed'
                evidence['failure_log']=dict(path=str(group/arm/'monitor/actual_failure.log'),sha256='a'*64)
                seal['preserved_files'][evidence['failure_log']['path']]='a'*64
                slot['checkpoint']['status']='RUNNING'
            before=json.dumps(seal['slots'],sort_keys=True);resources.validate_seal(seal)
            self.assertEqual(json.dumps(seal['slots'],sort_keys=True),before)
            del seal['preserved_files'][seal['slots']['D']['recovery']['failure_log']['path']]
            with self.assertRaisesRegex(ValueError,'must be sealed'):resources.validate_seal(seal)


    def test_real_terminal_failure_evidence_binds_exact_actor_controller_monitor_and_log(self):
        with tempfile.TemporaryDirectory()as temporary:
            group=Path(temporary);seal=self.seal(group);slot=seal['slots']['D'];arm='D';old=Path(seal['previous_code'])
            controller_identity=dict(pid=2488819,create_time=1791808638.21,UID=41833,
                command=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
                    str(old/'tools/run_v25_recovery_continuation.py'),'--group',str(group)])
            command=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
                str(old/'tools/run_v25_recovery_execution.py'),'--arm',arm,'--gpu','4','--output',str(group/arm),'--mode','train']
            actor=dict(pid=resources.PREVIOUS_TRAINING[arm][0],create_time=resources.PREVIOUS_TRAINING[arm][1],command=command,UID=41833)
            log=group/arm/'monitor/own_failure.log';log.parent.mkdir(parents=True)
            log.write_text('CPU_DEBUG synthetic traceback fixture\nRuntimeError: Recorded stage descendant is outside exact controller tree\n')
            evidence=slot['recovery'];evidence.update(mode='natural_failed',previous_controller_identity=controller_identity,
                failure_log=dict(path=str(log),sha256=resources.sha(log)))
            slot['checkpoint']['status']='RUNNING';slot['previous_actor']=actor
            registration=dict(format='v25_owned_validation_reuse_terminal_v1',arm=arm,marker=None,child=actor,
                controller=controller_identity,created_at=1.,reason='CPU_DEBUG synthetic actual control flow',
                full40_GNN_preserved=True,no_signal_sent=True,token=None,status_path=str(group/arm/'monitor/status.json'),
                status_sha256=evidence['monitor']['sha256'],failure_log=str(log),failure_log_sha256=evidence['failure_log']['sha256'])
            monitor=dict(child_pid=actor['pid'],child_create_time=actor['create_time'],child_command=command,UID=41833,
                controller_pid=controller_identity['pid'],status='FAILED',stage='train',stage_log=str(log))
            def read(proof):return registration if proof is evidence['registration']else monitor
            with (patch.object(resources,'_guard_proof',side_effect=read),
                    patch.object(resources.direct,'_owned_regular',side_effect=lambda p:Path(p)),
                    patch.object(resources.direct.psutil,'Process',side_effect=resources.direct.psutil.NoSuchProcess(1))):
                resources._validate_terminal_resume(slot,seal)
                changed=dict(actor,create_time=actor['create_time']+1);registration['child']=changed
                with self.assertRaisesRegex(ValueError,'original training actor'):resources._validate_terminal_resume(slot,seal)
                registration['child']=actor;monitor['child_command']=command+['foreign']
                with self.assertRaisesRegex(ValueError,'FAILED terminal monitor'):resources._validate_terminal_resume(slot,seal)
                monitor['child_command']=command
                with patch.object(resources,'_released',side_effect=RuntimeError('not reaped')):
                    with self.assertRaisesRegex(RuntimeError,'not reaped'):resources._validate_terminal_resume(slot,seal)


    def test_wrapper_installs_same_allocator_and_source_guard_before_frozen(self):
        from tools import run_v25 as frozen,run_v25_execution as original
        from hiercp_v1x import v25_upper_memory_runtime as upper,v25_provider_memory_runtime as provider,v25_u128_validation_reuse as reuse
        for arm,(gpu,_)in resources.MAPPING.items():
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
                    stack.enter_context(patch.object(reuse,'install_runtime',side_effect=lambda **k:calls.append('reuse')or{}))
                    stack.enter_context(patch.object(frozen,'publish'))
                    stack.enter_context(patch.object(original,'main',side_effect=lambda a:calls.append('frozen')or 0))
                    execution.main(['--mode','train'])
                self.assertEqual(calls,['resource','source','provider','reuse','frozen'])
                self.assertIn('host_memory.py',{p.name for p in admit.call_args.kwargs['runtime_files']})


    def test_each_final_completion_uses_isolated_CPU_interpreter_without_parent_ENV_changes(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-parent','V25_NATIVE_REQUEST':'parent-proof'}):
            before=os.environ.copy()
            with patch.object(controller.subprocess,'check_output',return_value='{"actual_full_completion_verified":true}')as run:
                
                for arm in 'ABCD':controller.verify_arm_completion(Path('group'),arm)
            self.assertEqual(os.environ.copy(),before)
            self.assertEqual([call.args[0][-1]for call in run.call_args_list],list('ABCD'))
            for call in run.call_args_list:
                self.assertEqual(call.args[0][:3],[sys.executable,'-B','-c'])
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertIsNot(call.kwargs['env'],os.environ)


    def test_four_chains_start_concurrently_and_one_failure_does_not_stop_other_three(self):
        started={arm:threading.Event()for arm in 'ABCD'};finished=[];reports=[]
        def chain(arm):
            started[arm].set()
            for event in started.values():
                if not event.wait(3):raise AssertionError('four chains were serialized')
            if arm=='B':raise RuntimeError('debug own B failure')
            finished.append(arm)
        failures=controller.run_chains(chain,reports.append)
        self.assertEqual(set(finished),set('ACD'));self.assertEqual(failures[0]['arm'],'B');self.assertEqual(len(reports),1)


if __name__=='__main__':unittest.main()
