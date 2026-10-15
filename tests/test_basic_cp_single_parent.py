"""CPU DEBUG admission tests; no CUDA, PBS or production process is started."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import basic_cp_single_parent_resources as resource
from tools import run_basic_cp_single_parent as runner
from tools import prepare_basic_cp_single_parent as prepare


class SingleParentDEBUGTests(unittest.TestCase):
    def fixture(self,mig=False):
        parent='GPU-11111111-1111-1111-1111-111111111111'
        tokens=['MIG-'+str(i)*8+'-'+str(i)*4+'-'+str(i)*4+'-'+str(i)*4+'-'+str(i)*12 for i in range(1,7)]if mig else[parent]*6
        assigned=dict(zip(resource.ARMS,tokens));holder_id=dict(pid=100,create_time=100,UID=41833,command=['/bin/bash'])
        receipt=dict(format=resource.FORMAT,UID=41833,owner='aicompetition06',node='test-node',
            mode='six_MIG_same_parent'if mig else'full_GPU_shared6',physical_parent_GPU_UUID=parent,
            execution_UUID_by_arm=assigned,job_id='123.ECE-util1',
            PBS_lifetime=dict(node_boot_id='boot-DEBUG',query_started_monotonic=100,walltime_seconds=86400,
                elapsed_seconds=0,remaining_seconds=86400,raw_PBS_response_sha256='a'*64),
            CPU_affinity=list(range(48)),holder_identity=holder_id,
            VRAM_reservation_bytes_by_arm={a:9*resource.GIB for a in resource.ARMS},VRAM_floor_bytes=256*2**20)
        job=dict(job_id='123.ECE-util1',job_state='R',Job_Owner='aicompetition06@host',exec_host='test-node/0',
            stime=100,resources_used={'walltime':'00:15:00'},
            Resource_List=dict(ngpus=6 if mig else 1,ncpus=48,walltime='24:00:00'))
        holder=dict(identity=holder_id,CPU_affinity=list(range(48)),memory_limit_bytes=512*resource.GIB,
            environment={'PBS_JOBID':'123.ECE-util1','CUDA_VISIBLE_DEVICES':','.join(dict.fromkeys(tokens))})
        devices={token:dict(parent_UUID=parent,index=0,MIG=mig,GI=i if mig else None,CI=0 if mig else None,
            MIG_enabled=mig,total_bytes=10*resource.GIB if mig else 80*resource.GIB,used_bytes=0,compute_actors=[])for i,token in enumerate(tokens)}
        receipt['execution_device_observations']=copy.deepcopy(devices)
        return receipt,job,holder,devices

    def test_full80_allocation_admits_six_original_B2_experiments(self):
        receipt,job,holder,devices=self.fixture()
        self.assertEqual(self.validate(receipt,job,holder,devices),set(devices))
        result=resource.validate_capacity(receipt,devices,native_RSS={a:32*resource.GIB for a in resource.ARMS},
            auxiliary_RSS=16*resource.GIB,cgroup_usage_bytes=512*resource.GIB)
        self.assertEqual(result['native_RSS_bytes'],192*resource.GIB)

    def test_six_distinct_owned_MIG_instances_must_share_one_parent(self):
        receipt,job,holder,devices=self.fixture(True)
        self.assertEqual(len(self.validate(receipt,job,holder,devices)),6)
        devices[next(iter(devices))]['parent_UUID']='GPU-OTHER'
        with self.assertRaisesRegex(ValueError,'one physical parent'):
            self.validate(receipt,job,holder,devices)

    def direct_fixture(self):
        receipt,job,holder,devices=self.fixture(True)
        host=copy.deepcopy(holder['identity']);launcher=dict(pid=200,create_time=200,UID=41833,command=['python','direct_launcher'])
        receipt.update(mode=resource.DIRECT_MODE,PBS_GPU_owned_reservation=False,PBS_CPU_owned_reservation=False,
            PBS_job_role='host_lifetime_only',explicit_existing_direct_execution_requested=True,
            holder_identity=launcher,host_allocation_process=host,
            direct_launcher_cgroup={'path':'/system.slice/ssh.service','limit':2*1024*resource.GIB})
        holder.update(identity=launcher,environment={},memory_cgroup=copy.deepcopy(receipt['direct_launcher_cgroup']),
            host_allocation_holder={'identity':host,'environment':{'PBS_JOBID':receipt['job_id']}})
        job['Resource_List'].update(ngpus=1,ncpus=1)
        return receipt,job,holder,devices

    def test_prior_authorized_direct_mode_uses_PBS_only_for_host_lifetime(self):
        receipt,job,holder,devices=self.direct_fixture()
        self.assertEqual(len(self.validate(receipt,job,holder,devices)),6)
        self.assertNotIn('PBS_JOBID',holder['environment'])
        self.assertNotIn('CUDA_VISIBLE_DEVICES',holder['environment'])
        receipt['mode']='six_MIG_same_parent'
        with self.assertRaisesRegex(ValueError,'requested GPU'):
            self.validate(receipt,job,holder,devices)

    def test_direct_mode_never_claims_GPU_CPU_PBS_ownership(self):
        for field,value in(('PBS_GPU_owned_reservation',True),('PBS_CPU_owned_reservation',True),
                ('PBS_job_role','execution_allocation'),('explicit_existing_direct_execution_requested',False)):
            receipt,job,holder,devices=self.direct_fixture();receipt[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'cannot claim'):
                self.validate(receipt,job,holder,devices)

    def test_direct_mode_keeps_actual_shell_lifetime_and_current_cgroup_independent(self):
        for failure in('host_shell','host_job','launcher_cgroup','cross_parent'):
            receipt,job,holder,devices=self.direct_fixture()
            if failure=='host_shell':holder['host_allocation_holder']['identity']={'pid':999,'UID':41833}
            if failure=='host_job':holder['host_allocation_holder']['environment']['PBS_JOBID']='another'
            if failure=='launcher_cgroup':holder['memory_cgroup']['path']='/another_cgroup'
            if failure=='cross_parent':devices[next(iter(devices))]['parent_UUID']='GPU-another'
            with self.subTest(failure=failure),self.assertRaises(ValueError):
                self.validate(receipt,job,holder,devices)

    def test_multiple_compute_UUIDs_cannot_count_shared_GI_memory_twice(self):
        receipt,job,holder,devices=self.direct_fixture()
        first,second=list(devices)[:2]
        devices[second]['GI']=devices[first]['GI'];devices[second]['CI']=1
        receipt['execution_device_observations']=copy.deepcopy(devices)
        with self.assertRaisesRegex(ValueError,'independent MIG memory partitions'):
            self.validate(receipt,job,holder,devices)

    def test_one_MIG_cannot_be_declared_six_owned_devices(self):
        receipt,job,holder,devices=self.fixture(True)
        token=next(iter(devices));receipt['execution_UUID_by_arm']={a:token for a in resource.ARMS}
        with self.assertRaisesRegex(ValueError,'six distinct'):
            self.validate(receipt,job,holder,devices)

    def test_foreign_owner_queued_job_host_GPUcount_and_CPU_shortfall_are_rejected(self):
        for field,value in [('Job_Owner','other@host'),('job_state','Q'),('exec_host','other-node/0'),
                             ('Resource_List',dict(ngpus=1,ncpus=47,walltime='24:00:00'))]:
            receipt,job,holder,devices=self.fixture();job[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.validate(receipt,job,holder,devices)

    def test_holder_identity_visibility_and_actual_cgroup_capacity_are_mandatory(self):
        for change in('identity','visibility','memory'):
            receipt,job,holder,devices=self.fixture()
            if change=='identity':holder['identity']=dict(holder['identity'],UID=999)
            if change=='visibility':holder['environment']['CUDA_VISIBLE_DEVICES']='GPU-OTHER'
            if change=='memory':holder['memory_limit_bytes']=36*resource.GIB
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.validate(receipt,job,holder,devices)

    def test_full40GB_or_MIG_enabled_parent_cannot_be_called_full80(self):
        for memory,mig in((45*resource.GIB,False),(80*resource.GIB,True)):
            receipt,job,holder,devices=self.fixture();devices[next(iter(devices))].update(total_bytes=memory,MIG=mig)
            with self.subTest(memory=memory,mig=mig),self.assertRaises(ValueError):
                self.validate(receipt,job,holder,devices)

    def test_monotonic_expiry_cannot_be_extended_and_ignores_server_wall_clock(self):
        receipt,job,holder,devices=self.fixture();receipt['PBS_lifetime']['remaining_seconds']+=3600
        with self.assertRaises(ValueError):self.validate(receipt,job,holder,devices)
        receipt['PBS_lifetime']['remaining_seconds']-=3600
        job['stime']=-8900000
        self.validate(receipt,job,holder,devices)
        with self.assertRaises(ValueError):self.validate(receipt,job,holder,devices,now=86400)
        with self.assertRaises(ValueError):
            resource.validate_allocation(receipt,job,holder,devices,now_monotonic=1000,node_boot_id='another-boot')

    def validate(self,receipt,job,holder,devices,now=1000):
        return resource.validate_allocation(receipt,job,holder,devices,now_monotonic=now,node_boot_id='boot-DEBUG')

    def test_new_actual_PBS_elapsed_can_only_shorten_monotonic_deadline(self):
        receipt,job,holder,devices=self.fixture()
        job['resources_used']['walltime']='23:59:00'
        with self.assertRaisesRegex(ValueError,'180second reserve'):
            resource.validate_allocation(receipt,job,holder,devices,now_monotonic=1000,node_boot_id='boot-DEBUG',
                query_started_monotonic=990)
        job['resources_used']['walltime']='00:00:00'
        with self.assertRaisesRegex(ValueError,'180second reserve'):
            resource.validate_allocation(receipt,job,holder,devices,now_monotonic=86400,node_boot_id='boot-DEBUG',
                query_started_monotonic=86399)

    def test_only_admitted_inactive_auxiliary_gradients_can_be_missing(self):
        from hiercp_v1x import v24_native_gradient_runtime as runtime
        parameters={'decoder.seg_layers.0.weight':SimpleNamespace(requires_grad=True,grad=None),
            'encoder.weight':SimpleNamespace(requires_grad=True,grad=object())}
        trainer=SimpleNamespace(network=SimpleNamespace(named_parameters=lambda:parameters.items()))
        helper=lambda trainer,parameters:dict(official_inactive_parameter_names=['decoder.seg_layers.0.weight'],
            gradient_runtime={'complete_ascending_head_order':True})
        adapted=SimpleNamespace(__globals__={'_native_inactive_auxiliary_gradient_proof':helper})
        original=SimpleNamespace(_native_clone_step_with_amp_retry=object())
        with patch.dict(sys.modules,{'hiercp_v1x.v24_nnunet_cp':original}),patch.object(runtime,'clone_step',return_value=adapted):
            proof=runner.inactive_gradient_proof(trainer)
            self.assertEqual(proof['actual_missing_gradient_names'],['decoder.seg_layers.0.weight'])
            self.assertNotIn('all_trainable_gradients_present',proof)
            parameters['encoder.weight'].grad=None
            with self.assertRaisesRegex(ValueError,'outside proven zero-weight'):
                runner.inactive_gradient_proof(trainer)

    def test_log_prefix_preserves_raw_bytes_and_rejects_missing_completed_epochs(self):
        completed=('Epoch 0\n[OnlineCP] epoch=0 applied=200/500\nEpoch time: 80 s\n'
            'Epoch 1\n[OnlineCP] epoch=1 applied=210/500\nEpoch time: 81 s\n')
        raw=(completed+'Epoch 2\n[OnlineCP] epoch=2 applied=13/500\npartial-error\n').encode()
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='basic_CP_CPU_DEBUG_',dir=str(workspace))as folder, \
                patch.object(resource,'exact',side_effect=lambda path,**kwargs:Path(path)):
            root=Path(folder).resolve();source=root/'raw.log';source.write_bytes(raw)
            self.assertTrue(root.is_relative_to(workspace))
            reference=dict(path=str(source),sha256=resource.digest(source))
            admitted,proof=prepare.completed_log_prefix(reference,2,root/'prefix.log')
            self.assertEqual(Path(admitted['path']).read_bytes(),completed.encode())
            self.assertEqual(source.read_bytes(),raw);self.assertEqual(proof['original_raw_log'],reference)
            self.assertEqual(proof['CP_attempted'],1000)
            with self.assertRaisesRegex(ValueError,'completion evidence'):
                prepare.completed_log_prefix(reference,3,root/'incomplete.log')
            source2=root/'missing_epoch.log';source2.write_text(completed.replace('Epoch 1','Epoch 2'))
            with self.assertRaisesRegex(ValueError,'unique complete training log'):
                prepare.completed_log_prefix(dict(path=str(source2),sha256=resource.digest(source2)),2,root/'missing.log')

    def test_failure_before_native_initialization_keeps_actual_console_and_error_stage(self):
        cohort={'experiments':{'multi5':{'output_root':'/own/multi5','logical_slot':0}},'physical_parent_index':3}
        with patch.object(runner,'read',return_value={'stage':'prepare','native_substage':'admission',
                'completed_epochs':130}),patch.object(runner,'publish')as publish:
            try:raise ValueError('Actual full source SHA differs')
            except ValueError as error:runner.record_child_failure(cohort,'multi5',error)
        path,value=publish.call_args.args
        self.assertEqual(path,Path('/own/multi5/status.json'))
        self.assertEqual(value['stage'],'train');self.assertEqual(value['status'],'FAILED')
        self.assertEqual(value['failed_stage'],'admission');self.assertEqual(value['completed_epochs'],130)
        self.assertIn('Actual full source SHA differs',value['error'])
        self.assertIn('ValueError',value['traceback'])

    def test_current_MIG_and_CPU8_metadata_cannot_display_original_A6000_hardware(self):
        receipt,job,holder,devices=self.fixture(True)
        receipt.update(physical_parent_index=3,physical_parent_observation={'model':'CPU_DEBUG_A10080','total_bytes':80*resource.GIB})
        receipt['CPU_affinity']=list(range(100,148))
        cohort=dict(experiments={'candidates512':{'output_root':'/own/candidates512'}},allocation={'path':'/own/allocation'},
            physical_parent_index=3,canonical_code={'commit':'CPU_DEBUG_commit','files_sha256':{}})
        source=dict(gpu=3,gpu_uuid='OLD_A6000',GPU_memory_total_MiB=49140,cpu_affinity=list(range(8)),
            baseline={'dataset_name':'Dataset730_LiverOnlineCP_OF0','data_identifier':'original_data'})
        with patch.object(runner,'pin',return_value=Path('/own/allocation')),patch.object(runner,'read',return_value=receipt), \
                patch.object(runner,'digest',return_value='CPU_DEBUG_bankSHA'):
            native=runner.native_for(cohort,'candidates512',source,Path('/own/bank'),Path('/shared/dataset'),{'pid':100})
        self.assertEqual(native['original_hardware']['GPU_memory_total_MiB'],49140)
        self.assertEqual(native['GPU_memory_total_MiB'],10*1024)
        self.assertEqual(native['cpu_affinity'],list(range(124,132)))
        self.assertEqual(native['gpu_uuid'],receipt['physical_parent_GPU_UUID'])
        self.assertEqual(native['execution_GPU_UUID'],receipt['execution_UUID_by_arm']['candidates512'])
        self.assertEqual(source['gpu_uuid'],'OLD_A6000')

    def test_historical_readiness_is_not_final_evaluation_and_changed_inputs_reject(self):
        proof=dict(format='basic_cp_historical_reference_actual26_readiness_v1',complete=True,
            ground_truth_files=26,basic_prediction_files=26,validation_cases=['CPU_DEBUG_case'+str(i)for i in range(26)],
            NIfTI_header_geometry_verified=True,full_evaluation_executed=False,inputs_modified=False,
            files={'/shared/CPU_DEBUG_GT.nii.gz':{'stat':[1,2,3]}})
        cohort={'historical_reference_inputs':{'path':'/own/CPU_DEBUG_receipt','sha256':'a'*64}}
        with patch.object(runner,'pin',return_value=Path('/own/CPU_DEBUG_receipt')),patch.object(runner,'read',return_value=proof), \
                patch.object(runner,'signature',return_value=(1,2,3)):
            runner.verify_historical_reference(cohort,proof['validation_cases'])
            proof['full_evaluation_executed']=True
            with self.assertRaisesRegex(ValueError,'no final-evaluation claim'):
                runner.verify_historical_reference(cohort,proof['validation_cases'])
            proof['full_evaluation_executed']=False;proof['files']['/shared/CPU_DEBUG_GT.nii.gz']['stat']=[1,2,4]
            with self.assertRaisesRegex(ValueError,'reference changed'):
                runner.verify_historical_reference(cohort,proof['validation_cases'])

    def test_native_auxiliary_totalRAM_and_VRAM_limits_are_independent(self):
        for failure in('native','auxiliary','total','VRAM'):
            receipt,job,holder,devices=self.fixture();rss={a:2*resource.GIB for a in resource.ARMS};aux=0;total=20*resource.GIB
            if failure=='native':rss[resource.ARMS[0]]=32*resource.GIB+1
            if failure=='auxiliary':aux=16*resource.GIB+1
            if failure=='total':total=512*resource.GIB+1
            if failure=='VRAM':receipt['VRAM_reservation_bytes_by_arm']={a:14*resource.GIB for a in resource.ARMS}
            with self.subTest(failure=failure),self.assertRaises(ValueError):
                resource.validate_capacity(receipt,devices,native_RSS=rss,auxiliary_RSS=aux,cgroup_usage_bytes=total)

    def test_foreign_or_unregistered_own_CUDA_actor_is_never_admitted(self):
        for uid in(41833,999):
            receipt,job,holder,devices=self.fixture();devices[next(iter(devices))]['compute_actors']=[dict(pid=700)]
            owners={}if uid==41833 else{700:dict(arm=resource.ARMS[0],UID=uid)}
            with self.subTest(uid=uid),self.assertRaises(ValueError):resource.validate_execution_actors(receipt,devices,owners)

    def test_native_descendant_failure_identifies_exact_violated_resource(self):
        observed=dict(identity={'pid':700,'create_time':1,'UID':41833,'command':['CPU_DEBUG_worker']},
            PPID=600,CPU_affinity=[1,2],memory_cgroup={'path':'/own','limit':512*resource.GIB})
        expected=dict(arm='multi5',expected_CPUs=[1,2],expected_cgroup=observed['memory_cgroup'])
        resource.validate_native_child(observed,**expected)
        for field,value in(('UID',999),('CPU_affinity',[1,2,3]),('cgroup',{'path':'/another','limit':512*resource.GIB})):
            altered=copy.deepcopy(observed)
            if field=='UID':altered['identity']['UID']=value
            elif field=='CPU_affinity':altered[field]=value
            else:altered['memory_cgroup']=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'"failed": \["'+field+'"\]'):
                resource.validate_native_child(altered,**expected)

    def test_registered_actor_cannot_switch_MIG_device(self):
        receipt,job,holder,devices=self.fixture(True);wrong=receipt['execution_UUID_by_arm'][resource.ARMS[1]]
        devices[wrong]['compute_actors']=[dict(pid=700)]
        with self.assertRaisesRegex(ValueError,'another execution'):
            resource.validate_execution_actors(receipt,devices,{700:dict(arm=resource.ARMS[0],UID=41833)})

    def test_runner_has_no_runtime_or_source_array_copy_and_uses_own_BEST(self):
        source=(Path(runner.__file__)).read_text()
        self.assertNotIn('shutil.copytree',source)
        self.assertNotIn('run_training_entry()',source)
        self.assertIn('trainer.run_training()',source)
        self.assertIn("trainer.load_checkpoint(str(goal/'checkpoint_best.pth'))",source)
        self.assertIn('trainer.perform_actual_validation(save_probabilities=False)',source)
        self.assertIn("row['effective_bank']",source)
        self.assertEqual(set(resource.ARMS),{'multi5','multi7','multi10','candidates512','candidates1024','candidates2048'})


class ProcessLifetimeRaceDEBUGTests(unittest.TestCase):
    """Reproduce real finishing-worker observations without relaxing live guards."""
    def worker(self,states=None,**changes):
        worker=SimpleNamespace(pid=700,create_time=Mock(return_value=10),
            status=Mock(side_effect=states)if states else Mock(return_value=resource.psutil.STATUS_RUNNING),
            uids=Mock(return_value=SimpleNamespace(real=resource.UID)),
            cmdline=Mock(return_value=['CPU_DEBUG_worker']),ppid=Mock(return_value=600),
            cpu_affinity=Mock(return_value=[1,2]),memory_info=Mock(return_value=SimpleNamespace(rss=4096)))
        for field,value in changes.items():setattr(worker,field,value)
        return worker

    def observe(self,worker,*,cgroup=None,released=None,expected=None,validator=None):
        memory=cgroup or dict(path='/own',limit=512*resource.GIB,usage=4096)
        if validator is None:
            validator=lambda value:resource.validate_native_child(value,arm='multi5',expected_CPUs=[1,2],
                expected_cgroup=dict(path='/own',limit=512*resource.GIB))
        with patch.object(resource.psutil,'Process',return_value=worker), \
                patch.object(resource,'_memory_cgroup',return_value=memory):
            return resource.process_snapshot(worker,expected=expected,validator=validator,
                released=released,context='CPU_DEBUG finishing worker')

    def test_zombie_or_dead_worker_is_excluded_before_empty_argv_root_cgroup_reads(self):
        for state in(resource.psutil.STATUS_ZOMBIE,resource.psutil.STATUS_DEAD):
            worker=self.worker([state]);released=[]
            self.assertIsNone(self.observe(worker,cgroup={'path':'/','limit':0},released=released))
            worker.cmdline.assert_not_called();self.assertEqual(released[0]['state'],state)

    def test_worker_becoming_zombie_during_cgroup_read_is_not_a_foreign_live_actor(self):
        worker=self.worker([resource.psutil.STATUS_RUNNING,resource.psutil.STATUS_ZOMBIE]);released=[]
        worker.cmdline.return_value=[]
        self.assertIsNone(self.observe(worker,cgroup={'path':'/','limit':0},released=released))
        self.assertEqual(released[0]['state'],resource.psutil.STATUS_ZOMBIE)

    def test_terminal_recheck_also_handles_exit_between_postread_and_validation(self):
        worker=self.worker([resource.psutil.STATUS_RUNNING,resource.psutil.STATUS_RUNNING,resource.psutil.STATUS_ZOMBIE])
        released=[]
        self.assertIsNone(self.observe(worker,cgroup={'path':'/','limit':0},released=released))
        self.assertEqual(released[0]['state'],resource.psutil.STATUS_ZOMBIE)

    def test_ZombieProcess_during_identity_read_requires_fresh_terminal_confirmation(self):
        worker=self.worker([resource.psutil.STATUS_RUNNING,resource.psutil.STATUS_ZOMBIE])
        worker.cmdline.side_effect=resource.psutil.ZombieProcess(worker.pid);released=[]
        self.assertIsNone(self.observe(worker,released=released))
        self.assertEqual(released[0]['evidence'],'fresh kernel process status')
        live=self.worker();live.cmdline.side_effect=resource.psutil.ZombieProcess(live.pid)
        with self.assertRaises(resource.psutil.ZombieProcess):self.observe(live)

    def test_reaped_worker_requires_two_fresh_missing_observations(self):
        worker=self.worker();released=[]
        worker.memory_info.side_effect=resource.psutil.NoSuchProcess(worker.pid)
        with patch.object(resource.psutil,'Process',side_effect=[worker,
                resource.psutil.NoSuchProcess(worker.pid),resource.psutil.NoSuchProcess(worker.pid)]), \
                patch.object(resource,'_memory_cgroup',return_value={'path':'/own','limit':512*resource.GIB}):
            self.assertIsNone(resource.process_snapshot(worker,released=released))
        self.assertEqual(released[0]['state'],'reaped');self.assertEqual(len(released[0]['evidence']),2)

    def test_worker_reaped_before_initial_starttime_read_is_not_root_enumeration_failure(self):
        worker=self.worker();worker.create_time.side_effect=resource.psutil.NoSuchProcess(worker.pid);released=[]
        with patch.object(resource.psutil,'Process',side_effect=resource.psutil.NoSuchProcess(worker.pid))as fresh:
            self.assertIsNone(resource.process_snapshot(worker,released=released))
        self.assertEqual(fresh.call_count,2);self.assertEqual(released[0]['state'],'reaped')

    def test_live_UID_CPU_and_cgroup_changes_never_receive_terminal_exemption(self):
        for failure in('UID','CPU_affinity','cgroup'):
            worker=self.worker();memory=dict(path='/own',limit=512*resource.GIB);released=[]
            if failure=='UID':worker.uids.return_value=SimpleNamespace(real=999)
            if failure=='CPU_affinity':worker.cpu_affinity.return_value=[1,2,99]
            if failure=='cgroup':memory['path']='/'
            with self.subTest(failure=failure),self.assertRaisesRegex(ValueError,'"failed": \\["'+failure+'"\\]'):
                self.observe(worker,cgroup=memory,released=released)
            self.assertEqual(released,[])

    def test_missing_cgroup_file_for_still_live_worker_is_not_hidden(self):
        worker=self.worker();released=[]
        with patch.object(resource.psutil,'Process',return_value=worker), \
                patch.object(resource,'_memory_cgroup',side_effect=FileNotFoundError('actual cgroup unavailable')):
            with self.assertRaisesRegex(FileNotFoundError,'actual cgroup unavailable'):
                resource.process_snapshot(worker,released=released)
        self.assertEqual(released,[])

    def test_PID_replacement_does_not_inherit_registered_worker_identity(self):
        worker=self.worker();replacement=self.worker();replacement.create_time.return_value=20;released=[]
        with patch.object(resource.psutil,'Process',return_value=replacement):
            self.assertIsNone(resource.process_snapshot(worker,released=released))
        self.assertEqual(released[0]['state'],'PID_reused');replacement.cmdline.assert_not_called()

    def test_snapshot_metadata_can_be_reused_after_worker_exit_without_new_identity_reads(self):
        worker=self.worker();value=self.observe(worker)
        worker.cmdline.side_effect=resource.psutil.ZombieProcess(worker.pid)
        previous={worker.pid:value['identity']};owner=dict(UID=value['identity']['UID'],identity=value['identity'],PPID=value['PPID'])
        self.assertEqual(previous[700]['create_time'],10);self.assertEqual(owner['UID'],resource.UID)
        worker.cmdline.assert_called_once();self.assertEqual(value['RSS_bytes'],4096)

    def test_stale_NVML_is_requeried_only_after_known_actor_terminal_proof(self):
        receipt={'execution_UUID_by_arm':{'multi5':'MIG-own'}}
        devices={'MIG-own':{'compute_actors':[{'pid':700}]}};fresh={'MIG-own':{'compute_actors':[]}}
        known={700:{'pid':700,'create_time':10}};worker=self.worker([resource.psutil.STATUS_ZOMBIE]);released=[]
        with patch.object(resource.psutil,'Process',return_value=worker),patch.object(resource,'nvml_devices',return_value=fresh)as query:
            result=resource.refresh_terminal_CUDA_observations(receipt,devices,{},known,released=released)
        self.assertIs(result,fresh);query.assert_called_once();resource.validate_execution_actors(receipt,result,{})
        self.assertEqual(released[0]['state'],resource.psutil.STATUS_ZOMBIE)

    def test_live_unknown_or_still_reported_CUDA_actor_remains_rejected(self):
        receipt={'execution_UUID_by_arm':{'multi5':'MIG-own'}}
        devices={'MIG-own':{'compute_actors':[{'pid':700}]}};known={700:{'pid':700,'create_time':10}}
        for registered,state in((known,resource.psutil.STATUS_RUNNING),({},resource.psutil.STATUS_ZOMBIE),
                (known,resource.psutil.STATUS_ZOMBIE)):
            worker=self.worker([state])
            with self.subTest(registered=bool(registered),state=state), \
                    patch.object(resource.psutil,'Process',return_value=worker), \
                    patch.object(resource,'nvml_devices',return_value=devices)as query:
                result=resource.refresh_terminal_CUDA_observations(receipt,devices,{},registered,released=[])
                with self.assertRaisesRegex(ValueError,'Unknown own or foreign CUDA actor'):
                    resource.validate_execution_actors(receipt,result,{})
                self.assertEqual(query.call_count,1 if registered and state==resource.psutil.STATUS_ZOMBIE else 0)

    def test_lease_previous_worker_loop_preserves_full_identity_map_for_CUDA_refresh(self):
        """Exercise _verify dataflow, including a previous worker after all roots."""
        receipt,job,holder,devices=SingleParentDEBUGTests().fixture()
        actors={arm:dict(pid=1000+i,create_time=10+i,UID=resource.UID,command=['CPU_DEBUG_native',arm])
            for i,arm in enumerate(resource.ARMS)}
        orphan=dict(pid=777,create_time=77,UID=resource.UID,command=['CPU_DEBUG_previous_worker'])
        observations={value['pid']:dict(identity=value,PPID=500,CPU_affinity=list(range(i*8,i*8+8)),
            memory_cgroup={'path':'/123.ECE-util1','limit':512*resource.GIB},RSS_bytes=4096)
            for i,value in enumerate(actors.values())}
        observations[777]=dict(identity=orphan,PPID=1,CPU_affinity=list(range(40,48)),
            memory_cgroup={'path':'/123.ECE-util1','limit':512*resource.GIB},RSS_bytes=4096)
        processes={pid:SimpleNamespace(pid=pid,children=lambda recursive:[])for pid in observations}
        lease=resource.Lease.__new__(resource.Lease)
        lease.receipt=receipt;lease.actors=actors;lease.own=actors['multi5'];lease.arm='multi5';lease.cohort={}
        lease.files={};lease.previous_trees={'candidates2048':{777:orphan}};lease.last=0
        memory=dict(path='/123.ECE-util1',limit=512*resource.GIB,usage=8192)
        def snapshot(process,*,validator=None,**kwargs):
            value=observations[process.pid]
            if validator is not None:validator(value)
            return value
        class InspectedIdentityMap(Exception):pass
        def refresh(receipt,devices,owners,known,**kwargs):
            self.assertEqual(set(known),{777,*[value['pid']for value in actors.values()]})
            self.assertEqual(known[777],orphan)
            self.assertEqual(set(owners),set(observations))
            raise InspectedIdentityMap
        pbs=dict(raw=json.dumps({'Jobs':{receipt['job_id']:job}}),query_started_monotonic=1000)
        with patch.object(resource.os,'getuid',return_value=resource.UID,create=True), \
                patch.object(resource.socket,'gethostname',return_value='test-node'), \
                patch.object(resource.time,'monotonic',return_value=1000), \
                patch.object(resource,'observe_holder',return_value=(SimpleNamespace(pid=500),memory,holder)), \
                patch.object(resource,'_memory_cgroup',return_value=memory), \
                patch.object(resource,'actual_PBS',return_value=pbs), \
                patch.object(resource,'nvml_devices',return_value=devices), \
                patch.object(resource.Path,'read_text',return_value='boot-DEBUG'), \
                patch.dict(resource.os.environ,{'CUDA_VISIBLE_DEVICES':receipt['execution_UUID_by_arm']['multi5']}), \
                patch.object(resource,'cuda_execution_UUID'), \
                patch.object(resource.psutil,'Process',side_effect=lambda pid:processes[pid]), \
                patch.object(resource,'process_snapshot',side_effect=snapshot), \
                patch.object(resource,'refresh_terminal_CUDA_observations',side_effect=refresh):
            with self.assertRaises(InspectedIdentityMap):lease._verify(force=True)

    @unittest.skipUnless(os.name=='posix'and hasattr(os,'getuid')and os.getuid()==resource.UID,
        'Actual own UID41833 Linux worker-lifetime DEBUG')
    def test_actual_100_own_workers_exit_naturally_during_resource_snapshots(self):
        """No signal, CUDA, dataset or production model; each child ends itself."""
        expected_cgroup=resource.cgroup_identity(resource._memory_cgroup(os.getpid()))
        expected_CPUs=resource.psutil.Process().cpu_affinity();released=[];terminal=0
        command=[sys.executable,'-B','-c','import time;time.sleep(0.01)']
        def validate(value):
            resource.validate_native_child(value,arm='CPU_DEBUG_natural_worker',
                expected_CPUs=expected_CPUs,expected_cgroup=expected_cgroup)
        for batch in range(10):
            children=[subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)for index in range(10)]
            try:
                pending={child.pid:resource.psutil.Process(child.pid)for child in children};deadline=time.monotonic()+15
                while pending:
                    self.assertLess(time.monotonic(),deadline,'Bounded CPU DEBUG workers did not finish naturally')
                    for pid,worker in list(pending.items()):
                        value=resource.process_snapshot(worker,validator=validate,released=released,
                            context='actual Linux natural worker DEBUG')
                        if value is None:pending.pop(pid);terminal+=1
                    if pending:time.sleep(0.001)
            finally:
                # wait() only reaps these exact short-lived children; it never
                # sends a signal or changes a user/SSH/session process.
                for child in children:self.assertEqual(child.wait(timeout=15),0)
        self.assertEqual(terminal,100);self.assertGreaterEqual(len(released),100)


if __name__=='__main__':unittest.main()
