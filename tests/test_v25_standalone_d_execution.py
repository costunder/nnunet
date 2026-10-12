"""CPU DEBUG regressions; no server actor, CUDA or training is launched."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from hiercp_v1x import v25_standalone_d_resources as resources
from tools import run_v25_d_execution as execution,run_v25_d_continuation as controller


def identity(pid):return dict(pid=pid,create_time=1700000000.+pid,command=['python','owned-'+str(pid)],UID=41833)


class StandaloneDTests(unittest.TestCase):
    def seal(self,group):
        digest='a'*64
        return dict(format=resources.FORMAT,group_root=str(group),private_code=str(group/'newcode'),
            previous_code=str(group/'oldcode'),controller_identity=identity(1),
            direct_proof=dict(path='Dproof',sha256=digest),sibling_manifest=dict(path='manifest',sha256=digest),
            sibling_controller_identity=identity(2),sibling_slots={arm:dict(path=arm+'proof',sha256=digest) for arm in 'ABC'},
            preserved_files={'source':digest},runtime_files={'adapter':digest},
            checkpoint=dict(path=str(group/'D/training/checkpoint_latest.pt'),sha256=digest,
                identity_sha256=digest,content_sha256=digest,epoch=2,updates=20,train_position=12),
            previous_D_actor=identity(9),created_at=1.,scientific_configuration_unchanged=True,no_process_signaled=True)

    def test_seal_requires_exact_original_cursor_and_distinct_controller(self):
        with tempfile.TemporaryDirectory() as temporary:
            seal=self.seal(Path(temporary));self.assertIs(resources.validate_seal(seal),seal)
            altered=dict(seal,checkpoint=dict(seal['checkpoint'],updates=19))
            with self.assertRaisesRegex(ValueError,'epoch2/update20'):resources.validate_seal(altered)
            with self.assertRaises(ValueError):resources.validate_seal(dict(seal,controller_identity=seal['sibling_controller_identity']))
            with self.assertRaises(ValueError):resources.validate_seal(dict(seal,hidden_fast_mode=True))

    def test_actual_combination_validates_ABC_proofs_and_never_extends_D_GPU_tree(self):
        digest='a'*64;group=Path('group');seal=self.seal(group)
        aggregate=sorted([*range(6),*range(16,22),*range(32,38),*range(42,48)])
        cpus={'A':list(range(32,38)),'B':list(range(42,48)),'C':list(range(16,22))}
        mapping=dict(A=6,B=5,C=1);cgroup=dict(controller_cgroup='0::/',memory_limits=[])
        lifetime=dict(allocation_request='owned',allocation_request_sha256=digest,expires_at_unix=1700000000.)
        documents={arm:dict(format=resources.direct.DIRECT_FORMAT,arm=arm,gpu=mapping[arm],GPU_UUID='GPU-'+arm,
            CPU_affinity=cpus[arm],UID=41833,owner='aicompetition06',node='ece-a6gpu6',PBS_owned_reservation=False,
            explicit_direct_execution_authorized=True,controller_identity=seal['sibling_controller_identity'],
            group_root=str(group),host_lifetime=lifetime,resource_guard_sha256=digest,aggregate_CPU_affinity=aggregate,
            aggregate_native_and_cache_RSS_bytes=416*2**30,aggregate_GNN_and_cache_RSS_bytes=288*2**30,
            canonical_cache_reserve_bytes=32*2**30,cgroup_admission=cgroup) for arm in 'ABC'}
        manifest=dict(format='v25_four_GPU_independent_execution_v1',group_root=str(group),
            controller_identity=seal['sibling_controller_identity'],slots={arm:dict(arm=arm,gpu=mapping[arm],
                CPU_affinity=cpus[arm],resource_request_path=arm+'proof',resource_request_sha256=digest) for arm in 'ABC'})
        lease=SimpleNamespace(path=Path('Dproof'),group=group,managed={},request=dict(arm='D',gpu=4,
            CPU_affinity=list(range(6)),controller_identity=seal['controller_identity'],
            host_lifetime=lifetime,aggregate_CPU_affinity=aggregate))
        peer=Mock(pid=2);peer.cpu_affinity.return_value=aggregate;child=Mock(pid=44);peer.children.return_value=[child]
        Dtree={1:Mock(),3:Mock()};child_identity=identity(44)
        def proof(value):return manifest if value['path']=='manifest' else documents[value['path'][0]]
        with ExitStack() as stack:
            stack.enter_context(patch.object(resources,'load_seal',return_value=seal))
            stack.enter_context(patch.object(resources,'sha',return_value=digest))
            stack.enter_context(patch.object(resources,'_guard_proof',side_effect=proof))
            stack.enter_context(patch.object(resources.direct,'_exact',return_value=peer))
            stack.enter_context(patch.object(resources.direct,'capture_cgroup_admission',return_value=cgroup))
            stack.enter_context(patch.object(resources.terminal,'_descendant',return_value=child_identity))
            union=resources.combined_resource_tree(lease,Dtree[1],Dtree[3],Dtree)
            self.assertEqual(set(union),{1,2,3,44});self.assertEqual(set(Dtree),{1,3})
            self.assertEqual(lease._observed_ABC_descendants[44],child_identity)
            documents['B']['GPU_UUID']='GPU-altered'
            documents['B']['gpu']=4
            with self.assertRaisesRegex(ValueError,'ABC resource allocation'):
                resources.combined_resource_tree(lease,Dtree[1],Dtree[3],Dtree)

    def test_native_child_routing_preserves_scientific_arguments(self):
        flags=['730','3d_fullres','0','-tr','full_trainer','--val_best','--c']
        original=[sys.executable,'-B','-u','-c',execution.NATIVE_ENTRY,*flags]
        adapted=execution.adapt_native_child(original)
        self.assertEqual(adapted[:4],original[:4]);self.assertEqual(adapted[5:],flags)
        self.assertEqual(adapted[4],execution.NATIVE_BOOTSTRAP+execution.NATIVE_ENTRY)
        self.assertEqual(original[4],execution.NATIVE_ENTRY)
        clone=[sys.executable,'-B','-u',str(execution.ROOT/'tools/run_v25_downstream.py'),
            'clone-debug-worker','--physical-batch','4','--native','full_native.json','--output','full']
        adapted=execution.adapt_native_child(clone)
        self.assertEqual(adapted[3],str(execution.ROOT/'tools/run_v25_d_execution.py'))
        self.assertEqual(adapted[4],'--native');self.assertEqual(adapted[5:],clone[4:])

    def test_unknown_native_child_entry_is_refused(self):
        for argv in ([sys.executable,'-B','-u','-c','unrecognized()'],
                [sys.executable,'-B','-u','unreviewed.py'],['python','-c',execution.NATIVE_ENTRY]):
            with self.assertRaises(ValueError):execution.adapt_native_child(argv)

    def test_native_callback_updates_existing_private_clone(self):
        original=Mock(return_value='actual child result');pipeline=SimpleNamespace(_child=original)
        native=SimpleNamespace(_child=original,_MODULES={'NATIVE_PIPELINE':pipeline})
        self.assertIs(execution.install_native_children(native),original)
        self.assertIs(pipeline._child,native._child)
        argv=[sys.executable,'-B','-u','-c',execution.NATIVE_ENTRY,'--val_best']
        self.assertEqual(pipeline._child(argv,{'exact':'env'},'log'),'actual child result')
        args=original.call_args.args
        self.assertEqual(args[0][-1],'--val_best');self.assertEqual(args[1:],({'exact':'env'},'log'))

    def test_stage_commands_only_D_with_explicit_same_latest(self):
        args=SimpleNamespace(native_experiment=Path('native'),inventory=Path('inventory'),
            stunet_checkpoint=Path('official'),input_cache=Path('fullcache'),group=Path('group'),
            storage_admission=Path('storage'),source_bank=Path('642bank'),baseline_preprocessed=Path('baseline'),
            execution_manifest=Path('seal'),project=Path('project'),basic_validation=Path('basic26'),evaluator_root=Path('eval'))
        slot=dict(arm='D',gpu=4,CPU_affinity=list(range(6)),resource_request_path='newDproof')
        argv=controller.stage_command(args,slot,'train')
        self.assertEqual(argv[3],str(controller.ROOT/'tools/run_v25_d_execution.py'))
        self.assertEqual(argv[argv.index('--arm')+1],'D');self.assertEqual(argv[argv.index('--gpu')+1],'4')
        self.assertEqual(argv[argv.index('--resume')+1],str(args.group/'D/training/checkpoint_latest.pt'))
        self.assertNotIn('--parallel-prepare',argv);self.assertNotIn('--independent-prepare',argv)
        native=controller.native_command(args,slot)
        self.assertEqual(native[4],'--native');self.assertIn('--evaluator-root',native)

    def test_D_GPU_tree_unchanged_but_original_RSS_guard_receives_union(self):
        lease=object.__new__(resources.StandaloneDAllocationLease)
        lease.actual_resources={};own=Mock();controller_actor=Mock();Dtree={1:controller_actor,2:own};ABC=Mock()
        union=dict(Dtree,**{}) ;union[3]=ABC
        with (patch.object(resources,'combined_resource_tree',return_value=union) as join,
                patch.object(resources._BASE,'_resources') as original):
            lease._resources(controller_actor,own,Dtree)
        join.assert_called_once_with(lease,controller_actor,own,Dtree)
        original.assert_called_once_with(lease,controller_actor,own,union)
        self.assertIs(resources.StandaloneDAllocationLease._gpu,resources._BASE._gpu)
        self.assertEqual(set(Dtree),{1,2});self.assertTrue(lease.actual_resources['GPU4_admitted_actors_only_standalone_D_tree'])

    def test_original_GPU4_guard_rejects_ABC_actor_even_if_RAM_accounted(self):
        lease=object.__new__(resources.StandaloneDAllocationLease)
        lease.request=dict(gpu=4,GPU_UUID='GPU-exactD')
        gpu='4, GPU-exactD, 49152, 45000, Disabled'
        with patch.object(resources.direct,'_query',side_effect=[gpu,'333, GPU-exactD']):
            with self.assertRaisesRegex(RuntimeError,'Foreign or unrelated'):
                lease._gpu({111:Mock(),222:Mock()},initial=True)

    def test_missing_ABC_controller_requires_actual_full_completion(self):
        lease=SimpleNamespace(group=Path('group'),_observed_ABC_descendants={})
        # Failed full-completion verification must not create a cached witness.
        with patch.object(resources.direct,'_owned_regular',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):resources._abc_complete(lease)
        self.assertFalse(hasattr(lease,'_completed_ABC_files'))

    def test_surviving_reparented_actor_remains_in_RAM_accounting(self):
        actor=identity(44);process=Mock(pid=44);lease=SimpleNamespace(_observed_ABC_descendants={44:actor})
        with (patch.object(resources.direct.psutil,'Process',return_value=process),
                patch.object(resources.terminal,'_descendant',return_value=actor)):
            union=resources._surviving_ABC_descendants(lease,{1:Mock()})
        self.assertIs(union[44],process)
        with (patch.object(resources.direct.psutil,'Process',return_value=process),
                patch.object(resources.terminal,'_descendant',return_value=dict(actor,create_time=3))):
            with self.assertRaises(RuntimeError):resources._surviving_ABC_descendants(lease,{})

    def test_wrapper_installs_provider_and_source_admission_before_frozen_train(self):
        from tools import run_v25 as frozen,run_v25_execution as original
        from hiercp_v1x import v25_upper_memory_runtime as upper,v25_provider_memory_runtime as provider
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'D';folder.mkdir();calls=[]
            seal=dict(group_root=str(folder.parent),direct_proof=dict(path='Dproof'),checkpoint=dict(path='latest'))
            args=SimpleNamespace(arm='D',gpu=4,mode='train',output=folder,allocation_request=Path('Dproof'),resume=Path('latest'))
            with ExitStack() as stack:
                stack.enter_context(patch.object(resources,'install_from_environment',side_effect=lambda:calls.append('resources') or {}))
                stack.enter_context(patch.object(resources,'load_seal',return_value=seal))
                stack.enter_context(patch.object(frozen,'parse',return_value=args))
                admission=stack.enter_context(patch.object(upper,'install_upper_memory_runtime',side_effect=lambda **k:calls.append('source') or {}))
                stack.enter_context(patch.object(provider,'install_runtime',side_effect=lambda:calls.append('provider') or {}))
                stack.enter_context(patch.object(frozen,'publish'))
                stack.enter_context(patch.object(original,'main',side_effect=lambda a:calls.append('frozen') or 0))
                self.assertEqual(execution.main(['--mode','train']),0)
            self.assertEqual(calls,['resources','source','provider','frozen'])
            names={p.name for p in admission.call_args.kwargs['runtime_files']}
            self.assertIn('host_memory.py',names);self.assertIn('v25_provider_memory_runtime.py',names)

    def test_helper_build_is_pure_and_remote_template_parses(self):
        path=execution.ROOT/'outputs/v24_gpu6_recovery_20261011/deploy_v25_D_provider_recovery_20261012.py'
        spec=importlib.util.spec_from_file_location('D_deploy_debug',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        payload=b'CPU DEBUG archive';manifest=dict(commit='a'*40,sha256=hashlib.sha256(payload).hexdigest(),paths=['new.py'],
            proof={'new.py':dict(previous_sha256=None,new_sha256='b'*64)})
        for phase in ('stage','start'):ast.parse(module.build_remote_script(manifest,payload,phase))
        with self.assertRaises(ValueError):module.build_remote_script(manifest,payload,'automatic')
        tree=ast.parse(module.REMOTE_SCRIPT)
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&{'kill','terminate','send_signal','killpg'})
        self.assertNotIn("root/'parallel_execution_manifest_shared_reuse_1.json';publish",module.REMOTE_SCRIPT)


if __name__=='__main__':unittest.main()
