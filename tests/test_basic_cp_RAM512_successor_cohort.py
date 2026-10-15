"""CPU DEBUG only: actual future queue metadata, no remote or CUDA action."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parents[1]/'tools'

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

builder=load('deferred_cohort_DEBUG','recover_basic_cp_RAM512_successor_cohort.py')
adapter=load('completed_log_DEBUG','basic_cp_RAM512_completed_log_adapter.py')


def arms(first=False,second=False):
    result=dict(builder.INITIAL_ARMS)
    if first:result[0],result[5]=result[5],result[0]
    if second:result[4],result[6]=result[6],result[4]
    return result


class DeferredPreparationDEBUGTests(unittest.TestCase):
    def test_independent_first_handoff(self):
        self.assertEqual(builder.placement_mapping(arms(),arms(True)),{0:5,1:1,2:2,3:3,4:4,5:0,6:6})
        self.assertEqual(builder.placement_mapping(arms(),arms(False,True)),{0:0,1:1,2:2,3:3,4:6,5:5,6:4})

    def test_both_forward_and_second_handoff(self):
        self.assertEqual(builder.placement_mapping(arms(),arms(True,True)),{0:5,1:1,2:2,3:3,4:6,5:0,6:4})
        self.assertEqual(builder.placement_mapping(arms(True),arms(True,True)),{0:0,1:1,2:2,3:3,4:6,5:5,6:4})
        self.assertEqual(builder.placement_mapping(arms(False,True),arms(True,True)),{0:5,1:1,2:2,3:3,4:4,5:0,6:6})

    def test_no_reverse_or_unsupported_mapping(self):
        for before,after in[(arms(True),arms()),(arms(False,True),arms()),(arms(),arms())]:
            with self.assertRaises(ValueError):builder.placement_mapping(before,after)
        bad=arms();bad[2],bad[3]=bad[3],bad[2]
        with self.assertRaises(ValueError):builder.placement_mapping(arms(),bad)
        bad=arms();bad[0]='candidates2048'
        with self.assertRaises(ValueError):builder.placement_mapping(arms(),bad)

    def test_each_new_handoff_requires_full_upstream_completion(self):
        states={str(g):dict(state='TRAINING')for g in range(7)}
        with self.assertRaises(ValueError):builder.validate_handoff_completion(arms(),arms(True),states)
        states['0']['state']='FULL_COMPLETE'
        builder.validate_handoff_completion(arms(),arms(True),states)
        with self.assertRaises(ValueError):builder.validate_handoff_completion(arms(),arms(True,True),states)
        states['4']['state']='FULL_COMPLETE'
        builder.validate_handoff_completion(arms(),arms(True,True),states)
        # Already handed-off pair is not newly gated on its old completion metadata.
        states['0']['state']='TRAINING'
        builder.validate_handoff_completion(arms(True),arms(True,True),states)

    def test_queue_plan_exact_and_separate(self):
        expected=dict(format=builder.QUEUE_FORMAT,queues=copy.deepcopy(builder.QUEUES))
        self.assertEqual(builder.validate_queue_plan(expected),expected)
        wrong=copy.deepcopy(expected);wrong['queues'][0]['target_physical_GPU']=6
        with self.assertRaises(ValueError):builder.validate_queue_plan(wrong)
        with self.assertRaises(ValueError):builder.validate_queue_plan(dict(expected,extra=True))

    def test_only_pending_physical_slots_receive_dependency(self):
        reference=dict(path='/dev/shm/new/deferred.json',sha256='2'*64)
        self.assertEqual(builder.pending_reference_for('multi5',0,reference),reference)
        self.assertEqual(builder.pending_reference_for('candidates1024',4,reference),reference)
        for arm,gpu in[('multi5',5),('candidates1024',6),('candidates2048',0),('multi7',4),('blend50',1)]:
            self.assertIsNone(builder.pending_reference_for(arm,gpu,reference))

    def test_retarget_preserves_all7_actual_scientific_native_state(self):
        reference=dict(path='/dev/shm/new/deferred.json',sha256='2'*64)
        for target_arms in[arms(True),arms(False,True),arms(True,True)]:
            builder.ARMS=target_arms
            for gpu,oldgpu in builder.placement_mapping(arms(),target_arms).items():
                old=dict(root='/dev/shm/old/GPU'+str(oldgpu),gpu=oldgpu,arm=arms()[oldgpu],gpu_uuid='GPU-OLD',cpu_affinity=[999],
                    native_root='/dev/shm/old/native'+str(oldgpu),trainer='FULL_NATIVE',bank='/dev/shm/bank',bank_sha256='1'*64,
                    split=dict(outer_train=list(range(105)),outer_val=list(range(26))),copies_per_CP_event=7 if oldgpu==6 else 1,
                    candidates_per_source=2048 if oldgpu==5 else 128,epochs=250,physical_batch=2,
                    GPU_queue_deferred_transfer=dict(path='/dev/shm/old/queue.json',sha256='4'*64))
                before=copy.deepcopy(old);physical=dict(gpu_uuid='GPU-PHYSICAL-'+str(gpu),cpu_affinity=list(range(gpu*8,(gpu+1)*8)),GPU_memory_total_MiB=49140)
                result=builder.retarget(old,gpu,physical,Path('/dev/shm/new/GPU'+str(gpu)),Path('/dev/shm/new/registry'),
                    dict(commit='a'*40,code_sha256={'module':'3'*64}),Path('/dev/shm/fullbanks'),Path('/dev/shm/new/producer'),{'source':'complete'},reference,reference)
                self.assertEqual(old,before)
                for key in('arm','native_root','trainer','bank','bank_sha256','split','copies_per_CP_event','candidates_per_source','epochs','physical_batch'):
                    self.assertEqual(result[key],old[key])
                self.assertEqual(result['gpu'],gpu);self.assertEqual(result['gpu_uuid'],physical['gpu_uuid'])
                expected=builder.pending_reference_for(old['arm'],gpu,reference)
                self.assertEqual(result.get('GPU_queue_deferred_transfer'),expected)

    def test_completed_native_roles(self):
        self.assertEqual(builder.classification(latest_epoch=250,final_epoch=250,best26=True,metrics=True,complete_archive=True),'FULL_COMPLETE')
        self.assertEqual(builder.classification(latest_epoch=144,final_epoch=None,best26=False,metrics=False,complete_archive=False),'TRAINING')

    def test_exact_completed_log_adapter_still_binds_one_saved_logger(self):
        path=HERE/'basic_cp_RAM512_completed_log_adapter.py'
        routed=adapter.private_builder_main(builder,adapter_path=path,adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(callable(routed));self.assertEqual(routed.__name__,'main')

    def test_actual_spawn_identity_sealed_before_any_request_publication(self):
        source=(HERE/'recover_basic_cp_RAM512_successor_cohort.py').read_text(encoding='utf8')
        actor=source.index("base.seal(root/'controller_spawned_actor.json',actor)")
        self.assertLess(source.index("actor=base.identity(process)"),actor)
        self.assertLess(actor,source.index("request.update(controller_identity=actor"))
        self.assertLess(actor,source.index("base.seal(root/'request.json',request)"))

    def test_dynamic_current_source_and_no_calibration(self):
        source=(HERE/'recover_basic_cp_RAM512_successor_cohort.py').read_text(encoding='utf8')
        ast.parse(source)
        self.assertIn("OLDGROUP=Path(manifest['group'])",source)
        self.assertIn("source_request=read(source_reference['path'])",source)
        self.assertNotIn('source_request=requests[5]',source)
        self.assertNotIn("sample=read(GROUP/'GPU1_blend50/request.json')",source)
        self.assertIn("validate_handoff_completion(OLD_ARMS,ARMS,states)",source)
        self.assertIn("CPU_source_actor_created=False",source)
        self.assertNotIn('torch.cuda',source)

if __name__=='__main__':unittest.main()
