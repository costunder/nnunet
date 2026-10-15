"""CPU DEBUG admission tests; no CUDA, PBS or production process is started."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

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


if __name__=='__main__':unittest.main()
