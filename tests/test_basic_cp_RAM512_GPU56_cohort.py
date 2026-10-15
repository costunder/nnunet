"""CPU DEBUG: fresh operational placement preserves actual scientific state."""
import copy
import hashlib
import importlib.util
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
                GPU_queue_deferred_transfer={'path':'/dev/shm/old/deferred.json','sha256':'c'*64})
            before=copy.deepcopy(old)
            new=builder.retarget(old,gpu,old,Path('/dev/shm/new/GPU'+str(gpu)),Path('/dev/shm/new/registry.json'),
                {'commit':'d'*40,'code_sha256':{'guard':'e'*64}},Path('/dev/shm/original/banks'),
                Path('/dev/shm/new/GPU5_candidates2048'),{'source':'complete'},reference,reference,
                sharing_data=sharing_data,arms=arms)
            self.assertEqual(old,before)
            for key in('gpu','gpu_uuid','arm','cpu_affinity','native_root','private_runtime','private_pre','trainer',
                        'split','bank','bank_sha256','copies_per_CP_event','candidates_per_source','epochs','physical_batch'):
                self.assertEqual(new[key],old[key],key)
            self.assertNotIn('GPU_queue_deferred_transfer',new)
            physical=builder.EXECUTION_GPUS[gpu]
            self.assertEqual(new['execution_physical_GPU'],physical)
            self.assertEqual(new['execution_GPU_UUID'],None if physical is None else sharing_data['execution_GPU_UUIDs'][str(physical)])
            self.assertEqual(new['GPU56_sharing_receipt'],reference)

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
            {'sharing_data':'SHARING_DATA','arms':'ARMS'})
        for name in('ARMS','SHARING_DATA','GROUP','OLDGROUP','OLD_ARMS','OLD_GPU_FOR_NEW'):
            self.assertNotIn(name,builder.retarget.__code__.co_names)

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
