"""CPU DEBUG: fresh operational placement preserves actual scientific state."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('GPU56_builder_DEBUG',HERE/'tools/recover_basic_cp_RAM512_GPU56_cohort.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
from tools import basic_cp_RAM512_completed_log_adapter as adapter
from tools import watch_basic_cp_RAM512 as view
from hiercp_v1x import basic_cp_512_resources as resources
from hiercp_v1x import basic_cp_ablation_resources as common


class GPU56MigrationDEBUGTests(unittest.TestCase):
    def test_logical_slots_never_permute(self):
        expected={g:g for g in range(7)}
        self.assertEqual(builder.placement_mapping(builder.INITIAL_ARMS,builder.INITIAL_ARMS),expected)
        swapped=dict(builder.INITIAL_ARMS);swapped[0],swapped[5]=swapped[5],swapped[0]
        with self.assertRaises(ValueError):builder.placement_mapping(builder.INITIAL_ARMS,swapped)

    def test_full_retained_blend_is_mandatory(self):
        states={str(g):dict(state='TRAINING',arm=arm)for g,arm in builder.INITIAL_ARMS.items()}
        with self.assertRaises(ValueError):builder.validate_complete_retained(states)
        states['1']['state']='FULL_COMPLETE';builder.validate_complete_retained(states)
        states['6']['state']='BEST26_READY'
        with self.assertRaises(ValueError):builder.validate_complete_retained(states)

    def test_retarget_preserves_native_data_and_science(self):
        arms=dict(builder.INITIAL_ARMS)
        sharing_data={'execution_GPU_UUIDs':{'5':'GPU-FIVE','6':'GPU-SIX'}}
        reference={'path':'/dev/shm/new/sharing.json','sha256':'a'*64}
        for gpu,arm in builder.INITIAL_ARMS.items():
            old=dict(root='/dev/shm/old/GPU'+str(gpu),gpu=gpu,arm=arm,gpu_uuid='GPU-LEGACY-'+str(gpu),
                cpu_affinity=list(range(gpu*8,gpu*8+8)),GPU_memory_total_MiB=46068,
                native_root='/dev/shm/original/native'+str(gpu),private_runtime='/dev/shm/runtime',
                private_pre='/dev/shm/preprocessed',trainer='FULL_NATIVE',split={'train':105,'validation':26},
                bank='/dev/shm/bank',bank_sha256='b'*64,copies_per_CP_event=7 if gpu==6 else 1,
                candidates_per_source=2048 if gpu==5 else 128,epochs=250,physical_batch=2,
                host_lifetime={'path':'/dev/shm/old/lifetime.json','sha256':'8'*64,'expires_at_unix':100},
                GPU_queue_deferred_transfer={'path':'/dev/shm/old/deferred.json','sha256':'c'*64})
            before=copy.deepcopy(old)
            new_lifetime={'path':'/dev/shm/new/lifetime.json','sha256':'9'*64,'expires_at_unix':100000}
            new=builder.retarget(old,gpu,old,Path('/dev/shm/new/GPU'+str(gpu)),Path('/dev/shm/new/registry.json'),
                {'commit':'d'*40,'code_sha256':{'guard':'e'*64}},Path('/dev/shm/original/banks'),
                Path('/dev/shm/new/GPU5_candidates2048'),{'source':'complete'},reference,reference,
                sharing_data=sharing_data,arms=arms,host_lifetime=new_lifetime)
            self.assertEqual(old,before)
            for key in('gpu','gpu_uuid','arm','cpu_affinity','native_root','private_runtime','private_pre','trainer',
                        'split','bank','bank_sha256','copies_per_CP_event','candidates_per_source','epochs','physical_batch'):
                self.assertEqual(new[key],old[key],key)
            self.assertNotIn('GPU_queue_deferred_transfer',new)
            physical=builder.EXECUTION_GPUS[gpu]
            self.assertEqual(new['execution_physical_GPU'],physical)
            self.assertEqual(new['execution_GPU_UUID'],None if physical is None else sharing_data['execution_GPU_UUIDs'][str(physical)])
            self.assertEqual(new['GPU56_sharing_receipt'],reference)
            self.assertEqual(new['host_lifetime'],new_lifetime)
            self.assertIsNot(new['host_lifetime'],new_lifetime)

    def test_completed_log_adapter_accepts_builder_exact_routes(self):
        path=HERE/'tools/basic_cp_RAM512_completed_log_adapter.py'
        routed=adapter.private_builder_main(builder,adapter_path=path,adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(callable(routed))

    def test_adapted_main_passes_validated_receipt_and_arms_without_module_globals(self):
        import ast
        import inspect
        path=HERE/'tools/basic_cp_RAM512_completed_log_adapter.py'
        routed=adapter.private_builder_main(builder,adapter_path=path,adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIsNot(routed.__globals__,builder.main.__globals__)
        self.assertIs(routed.__globals__['retarget'],builder.retarget)
        tree=ast.parse(inspect.getsource(builder.main))
        self.assertFalse(any(isinstance(node,ast.Global)for node in ast.walk(tree)))
        calls=[node for node in ast.walk(tree)if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name)and node.func.id=='retarget']
        self.assertEqual(len(calls),1)
        self.assertEqual({item.arg:item.value.id for item in calls[0].keywords},
            {'sharing_data':'SHARING_DATA','arms':'ARMS','host_lifetime':'lifetime'})
        for name in('ARMS','SHARING_DATA','GROUP','OLDGROUP','OLD_ARMS','OLD_GPU_FOR_NEW'):
            self.assertNotIn(name,builder.retarget.__code__.co_names)

    def lifetime_fixture(self,root):
        def witness(name,job,expiry):
            path=root/name
            document={'format':'v25_owned_GPU_allocation_v1','UID':41833,'owner':'aicompetition06',
                'node':'ece-a6gpu6','job_id':job,'expires_at_unix':expiry}
            path.write_text(json.dumps(document))
            return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'expires_at_unix':expiry}
        previous=witness('old_lifetime.json','1.ECE-util1',100)
        current=witness('new_lifetime.json','2.ECE-util1',86500)
        requests={gpu:{'host_lifetime':copy.deepcopy(previous)}for gpu in range(7)}
        job={'job_state':'R','Job_Owner':'aicompetition06@host','exec_host':'ece-a6gpu6/0',
            'Resource_List':{'ngpus':1,'walltime':'24:00:00'},'stime':100}
        query=Mock(return_value=json.dumps({'Jobs':{'2.ECE-util1':job}}))
        guarded=SimpleNamespace(validate_host_lifetime_GPU=Mock(return_value={'allocated_GPU':5}),BasicLease=common.BasicLease)
        return previous,current,requests,job,query,guarded

    def test_expired_old_allocation_rolls_to_one_verified_new_allocation(self):
        import tempfile
        (HERE/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'tmp')as temp:
            previous,current,requests,job,query,guarded=self.lifetime_fixture(Path(temp))
            before=copy.deepcopy(requests)
            selected,transition,evidence=builder.select_host_lifetime(requests,current,
                SimpleNamespace(owned=Path),guarded,query=query,now=1000)
            self.assertEqual(requests,before)
            self.assertEqual(selected,current)
            query.assert_called_once_with('2.ECE-util1')
            self.assertEqual(transition['previous_host_lifetime'],previous)
            self.assertEqual(transition['current_job_id'],'2.ECE-util1')
            self.assertFalse(transition['previous_allocation_required_to_be_running'])
            self.assertFalse(transition['old_lifetime_witness_written'])
            self.assertEqual(evidence,{previous['path']:previous['sha256'],current['path']:current['sha256']})
            guarded.validate_host_lifetime_GPU.assert_called_once()

    def test_new_allocation_requires_real_owner_host_R_GPUcount_and_deadline(self):
        import tempfile
        (HERE/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'tmp')as temp:
            previous,current,requests,job,query,guarded=self.lifetime_fixture(Path(temp))
            for name,value in [('job_state','Q'),('Job_Owner','foreign@host'),('exec_host','other-node/0'),
                               ('Resource_List',{'ngpus':2,'walltime':'24:00:00'}),('stime',10100)]:
                wrong=dict(job,**{name:value})
                query.return_value=json.dumps({'Jobs':{'2.ECE-util1':wrong}})
                with self.subTest(field=name),self.assertRaises(RuntimeError):
                    builder.select_host_lifetime(requests,current,SimpleNamespace(owned=Path),guarded,query=query,now=1000)
            query.return_value=json.dumps({'Jobs':{'2.ECE-util1':job}})
            with self.assertRaises(ValueError):
                builder.select_host_lifetime(requests,current,SimpleNamespace(owned=Path),guarded,query=query,now=86400)

    def test_mixed_previous_refs_or_changed_lifetime_bytes_are_rejected(self):
        import tempfile
        (HERE/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'tmp')as temp:
            previous,current,requests,job,query,guarded=self.lifetime_fixture(Path(temp))
            requests[6]['host_lifetime']=current
            with self.assertRaises(ValueError):
                builder.select_host_lifetime(requests,current,SimpleNamespace(owned=Path),guarded,query=query,now=1000)
            requests[6]['host_lifetime']=previous
            def changed(job_id):
                Path(previous['path']).write_text('changed after fresh PBS query')
                return json.dumps({'Jobs':{'2.ECE-util1':job}})
            with self.assertRaises(ValueError):
                builder.select_host_lifetime(requests,current,SimpleNamespace(owned=Path),guarded,query=changed,now=1000)

    def test_source_reuse_and_immediate_actor_capture_remain(self):
        source=(HERE/'tools/recover_basic_cp_RAM512_GPU56_cohort.py').read_text()
        self.assertIn('source_request=read(source_reference[',source)
        self.assertIn('source_banks_reused_without_producer=True',source)
        self.assertIn("CUDA_VISIBLE_DEVICES=request['execution_GPU_UUID']or''",source)
        self.assertNotIn('torch.cuda',source)
        self.assertLess(source.index("base.seal(root/'controller_spawned_actor.json',actor)"),
            source.index("base.seal(root/'request.json',request)"))

    def test_arm_viewer_selects_exact_experiment_and_old_gpu_keeps_slot(self):
        registration={str(g):{'arm':arm,'root':'/dev/shm/experiment/'+arm}for g,arm in builder.INITIAL_ARMS.items()}
        import tempfile
        (HERE/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'tmp')as temp:
            code=Path(temp)/'code';code.mkdir();manifest=Path(temp)/'launch.json'
            import json
            manifest.write_text(json.dumps({'code':str(code),'registration':registration}))
            observed=[]
            def build(expanded,viewer):
                return lambda argv:observed.append(argv)
            with patch.object(view.admission,'load_pinned_viewers',return_value=(SimpleNamespace(),SimpleNamespace())),patch.object(view,'build_entry',build),patch.object(Path,'resolve',lambda value,**kw:value):
                view.main(['--manifest',str(manifest),'--code-root',str(code),'--arm','multi5','--once'])
                view.main(['--manifest',str(manifest),'--code-root',str(code),'--gpu','6','--json'])
            self.assertEqual(observed,[['--root',str(Path(registration['0']['root'])),'--once'],
                ['--root',str(Path(registration['6']['root'])),'--json']])

    def test_shared_observer_routes_around_legacy_physical_query(self):
        lease=SimpleNamespace(request={'GPU56_sharing_receipt':{}})
        with patch.object(resources,'_observe_GPU56_waiting',return_value={'physical_GPU':5})as current,\
                patch.object(resources.group.BasicLease,'observe_waiting',side_effect=AssertionError('Legacy logical GPU query')):
            self.assertEqual(resources.BasicLease.observe_waiting(lease),{'physical_GPU':5})
        current.assert_called_once_with(lease)

    def test_held_reservation_observer_uses_current_state_without_legacy_lock_route(self):
        import time
        actor=SimpleNamespace(cpu_affinity=lambda:list(range(8)))
        state={'claims':{},'source_pending':False}
        lease=SimpleNamespace(request={'gpu':0,'GPU56_sharing_receipt':{},'controller_identity':{'pid':100}},
            self_identity={'pid':100},_inputs=Mock(),_host_lifetime=Mock(),
            lifetime={'expires_at_unix':time.time()+10000},monotonic_expiry=time.monotonic()+10000,
            _claim_override=None)
        def current(value):
            self.assertIs(value,lease)
            self.assertEqual(value._claim_override,state)
            self.assertIsNot(value._claim_override,state)
            return {'physical_GPU':5}
        with patch.object(common,'_exact',return_value=actor),\
                patch.object(resources,'_observe_GPU56_waiting',side_effect=current),\
                patch.object(resources.group.BasicLease,'observe_waiting',side_effect=AssertionError('Legacy query under held lock')):
            result=resources._RAM512.reservation_observer(lease,state,None)
        self.assertEqual(result,{'physical_GPU':5})
        self.assertIsNone(lease._claim_override)


if __name__=='__main__':unittest.main()
