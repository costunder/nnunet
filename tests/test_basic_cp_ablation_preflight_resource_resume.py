"""CPU external-control boundary fixtures; no clinical geometry/model/CUDA work."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from tools import run_basic_cp_ablation as runner
from hiercp_v1x import basic_cp_ablation_preflight as helper


class WaitingFixture(Exception):pass


class ResourcePreflightResumeBoundaryTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'tmp';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='preflight_control_debug_',dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.old=self.root/'old';self.new=self.root/'new'
        self.old.mkdir();self.new.mkdir()
        self.controller={'pid':101,'create_time':3.1,'UID':41833,'command':['CPU_DEBUG_parent']}
        self.actor={'pid':102,'create_time':3.2,'UID':41833,'command':['CPU_DEBUG_preflight']}
        self.record={'pid':102,'birth':3.2,'UID':41833,'command':['CPU_DEBUG_preflight'],'stage':'preflight'}
        self.write(self.old/'request.json',{'controller_identity':self.controller})
        self.write(self.old/'preflight_actor.json',self.record)
        self.write(self.old/'status.json',{'status':'FAILED','stage':'preflight',
            'error':'RuntimeError: PBS timeout admission has no verified running job within60seconds'})
        self.result={'complete':True,'source_entries':642,'eligible_cases':81,'validated_positions':82176,
                     'three_paste_positions':82176,'physical_batch_size':2,'epochs':250,
                     'summary':{'sources':642,'positions_validated':82176}}
        self.write(self.old/'preflight.json',self.result)
        self.native={'root':str(self.new),'arm':'multi3','bank':'/CPU_DEBUG/original128/index.json',
            'bank_sha256':'b'*64,'private_data_folder':'/CPU_DEBUG/private/data',
            'readonly_admission':'/CPU_DEBUG/all131.json','readonly_admission_sha256':'c'*64,
            'split':{'outer_train':[f'CPU_DEBUG_case{i}' for i in range(105)]},
            'reused_preflight':{'root':str(self.old),'request_sha256':runner.sha(self.old/'request.json'),
                'controller_identity':self.controller,'preflight_actor_identity':self.actor}}
        # Windows CPU host has no /dev/shm: relocate only that namespace literal.
        # Actual Path.resolve/no-symlink/absolute/relative safety checks stay intact.
        tree=ast.parse(Path(runner.__file__).read_text(encoding='utf8'))
        function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight'))
        changes=0
        for node in ast.walk(function):
            if isinstance(node,ast.Constant) and node.value=='/dev/shm/aicompetition06_basic_cp_ablation_20261013':
                node.value=str(self.root);changes+=1
        self.assertEqual(changes,1)
        self.namespace=dict(runner.__dict__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                     '<actual preflight function with CPU namespace only>','exec'),self.namespace)
        self.module=SimpleNamespace(__file__=str(self.new/'CPU_DEBUG_scientific_trainer.py'),
                                   preflight_bank=Mock(side_effect=AssertionError('Geometry recomputed at reuse boundary')))
        self.guard=Mock();self.alive=Mock(return_value=False);self.sleep=Mock();self.publish=Mock();self.status=Mock()
        self.namespace.update(resource_guard=self.guard,exact_actor_alive=self.alive,publish=self.publish,status=self.status,
            initialize_native=Mock(return_value=(self.module,{'CPU_DEBUG_adapter_only':True})),time=SimpleNamespace(sleep=self.sleep))
        leaf=ModuleType('nnunetv2.training.dataloading.nnunet_dataset')
        self.dataset_constructor=Mock(side_effect=lambda folder,ids:SimpleNamespace(keys=lambda:list(ids)))
        leaf.nnUNetDatasetBlosc2=self.dataset_constructor
        self.module_patch=patch.dict(sys.modules,{'nnunetv2.training.dataloading.nnunet_dataset':leaf})
        self.module_patch.start();self.addCleanup(self.module_patch.stop)
        self.verify=Mock(return_value=copy.deepcopy(self.result))
        self.verify_patch=patch.object(helper,'verify_preflight_reuse',self.verify)
        self.verify_patch.start();self.addCleanup(self.verify_patch.stop)

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value),encoding='utf8')

    def run_boundary(self):return self.namespace['preflight'](self.native)

    def assert_no_copy_or_recompute(self):
        self.publish.assert_not_called();self.module.preflight_bank.assert_not_called()

    def test_complete_terminal_actor_invokes_full_helper_and_publishes_only_verified_result(self):
        self.run_boundary()
        self.alive.assert_called_once_with(self.actor)
        self.verify.assert_called_once_with(self.old,self.native['reused_preflight']['request_sha256'],
            Path(self.module.__file__),self.native['bank'],self.native['bank_sha256'],'multi3',
            self.native['split']['outer_train'],readonly_admission=self.native['readonly_admission'],
            readonly_admission_sha256=self.native['readonly_admission_sha256'])
        self.dataset_constructor.assert_not_called()
        self.publish.assert_called_once();self.assertEqual(self.publish.call_args.args[0],self.new/'preflight.json')
        self.assertTrue(self.publish.call_args.args[1]['complete'])
        self.assertEqual(self.publish.call_args.args[1]['validated_positions'],82176)
        self.module.preflight_bank.assert_not_called()

    def test_live_exact_preflight_waits_without_helper_copy_or_recompute(self):
        self.alive.return_value=True;self.sleep.side_effect=WaitingFixture('CPU_DEBUG still live')
        with self.assertRaises(WaitingFixture):self.run_boundary()
        self.alive.assert_called_once_with(self.actor)
        self.status.assert_called_once();self.assertEqual(self.status.call_args.args[1],'waiting_preflight')
        self.verify.assert_not_called();self.assert_no_copy_or_recompute()

    def test_complete_proof_arrives_only_after_exact_actor_natural_exit(self):
        (self.old/'preflight.json').unlink();self.alive.side_effect=[True,False]
        def completed(_):
            self.verify.assert_not_called();self.publish.assert_not_called()
            self.write(self.old/'preflight.json',self.result)
        self.sleep.side_effect=completed
        self.run_boundary()
        self.assertEqual(self.alive.call_args_list,[unittest.mock.call(self.actor),unittest.mock.call(self.actor)])
        self.verify.assert_called_once();self.publish.assert_called_once();self.assertEqual(self.guard.call_count,2)

    def test_missing_whole_preflight_proof_does_not_call_helper_or_publish(self):
        (self.old/'preflight.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'lacks a complete proof'):self.run_boundary()
        self.verify.assert_not_called();self.assert_no_copy_or_recompute()

    def test_geometry_failure_is_not_relabelled_as_resource_resume(self):
        self.write(self.old/'status.json',{'status':'FAILED','stage':'preflight',
            'error':'OnlineCPError: full three-paste geometry unavailable'})
        with self.assertRaisesRegex(RuntimeError,'full three-paste geometry'):self.run_boundary()
        self.verify.assert_not_called();self.assert_no_copy_or_recompute()

    def test_changed_registration_before_wait_is_rejected(self):
        row=dict(self.record,command=['CPU_DEBUG_other_actor']);self.write(self.old/'preflight_actor.json',row)
        with self.assertRaisesRegex(RuntimeError,'registration changed'):self.run_boundary()
        self.alive.assert_not_called();self.verify.assert_not_called();self.assert_no_copy_or_recompute()

    def test_registration_and_request_changed_during_wait_are_rejected(self):
        for path in (self.old/'preflight_actor.json',self.old/'request.json'):
            self.alive.reset_mock();self.alive.side_effect=[True,False]
            original=path.read_bytes()
            def changed(_):path.write_bytes(original+b'\n')
            self.sleep.side_effect=changed
            try:
                with self.assertRaisesRegex(RuntimeError,'changed'):self.run_boundary()
            finally:path.write_bytes(original)
        self.verify.assert_not_called();self.assert_no_copy_or_recompute()

    def test_full_source_helper_rejection_never_publishes_or_recomputes(self):
        self.verify.side_effect=ValueError('CPU_DEBUG actual source SHA/full642 proof rejected')
        with self.assertRaisesRegex(ValueError,'full642 proof rejected'):self.run_boundary()
        self.verify.assert_called_once();self.assert_no_copy_or_recompute()

    def test_known_constructor_only_boundary_is_retained(self):
        del self.native['reused_preflight']['preflight_actor_identity']
        self.write(self.old/'status.json',{'status':'FAILED','stage':'debug','error':"KeyError: 'args'"})
        (self.old/'debug_B2.log').write_text("CPU_DEBUG known constructor KeyError: 'args'",encoding='utf8')
        self.run_boundary();self.alive.assert_called_once_with(self.controller)
        self.verify.assert_called_once();self.publish.assert_called_once();self.module.preflight_bank.assert_not_called()


if __name__=='__main__':unittest.main()
