"""DEBUG CPU launcher identity regression, without real server admission."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_v25_execution as execution


class ExecutionRuntimeTests(unittest.TestCase):
    def test_shared_child_sets_six_core_affinity_before_live_lease_constructor(self):
        from tools import run_v25 as frozen
        from hiercp_v1x import v25_direct_resources as resources
        from hiercp_v1x import v25_server_resources as owned
        from hiercp_v1x import v25_upper_memory_runtime as memory
        from hiercp_v1x import v25_independent_preparation as independent
        from hiercp_v1x import v25_terminal_descendant_resources as terminal
        import psutil
        with tempfile.TemporaryDirectory(prefix='DEBUG_v25_execution_',dir=execution.ROOT/'tmp') as temporary:
            root=Path(temporary);resource=root/'allocation.json'
            resource.write_text(json.dumps({'format':resources.DIRECT_FORMAT}),encoding='utf8')
            args=SimpleNamespace(mode='prepare',arm='A',output=root/'A',
                allocation_request=resource,cpu_affinity=list(range(6)))
            affinity=list(range(24));events=[]
            def cpu_affinity(value=None):
                nonlocal affinity
                if value is not None:
                    affinity=list(value);events.append('six_cpu_affinity')
                return list(affinity)
            def constructor(path):
                self.assertEqual(affinity,args.cpu_affinity)
                events.append('live_lease_constructor');return SimpleNamespace()
            original=owned.AllocationLease
            try:
                with patch.object(frozen,'parse',return_value=args), \
                     patch.object(frozen,'publish'), \
                     patch.object(resources,'DirectAllocationLease',side_effect=constructor), \
                     patch.object(terminal,'install_runtime',return_value={'DEBUG':True}), \
                     patch.object(memory,'install_upper_memory_runtime',return_value={'DEBUG':True}), \
                     patch.object(independent,'prepare_shared',return_value='DEBUG complete') as prepare, \
                     patch.object(psutil,'Process',return_value=SimpleNamespace(cpu_affinity=cpu_affinity)):
                    self.assertEqual(execution.main(['--shared-preparation']),'DEBUG complete')
            finally:
                owned.AllocationLease=original
            self.assertEqual(events,['six_cpu_affinity','live_lease_constructor'])
            prepare.assert_called_once()


class PrefixExecutionRuntimeTests(unittest.TestCase):
    """DEBUG CPU lifecycle fixtures; no server/CUDA/cache performance claim."""
    def _fixture(self,mode):
        from hiercp_v1x import v25_factory as factory,v25_preparation as preparation
        from hiercp_v1x import v25_prefix_geometry_reuse as reuse
        from tools import run_v25 as frozen
        events=[];state={'CUDA_initialized':False};constructed=[]
        class Inputs:
            def __init__(inputs,*arguments,**keywords):
                events.append(('construct',state['CUDA_initialized']))
                inputs._runtime_file_sha256={};inputs._raw_cache={};inputs._raw_bytes=0
                inputs._lock=object();inputs._donors={};inputs._regions={}
                constructed.append(inputs)
            def _admit_runtime_files(inputs,paths):
                inputs._runtime_file_sha256={str(path):'DEBUG_SHA' for path in paths}
            def guard_source(inputs):events.append(('guard',state['CUDA_initialized']))
            def close(inputs):events.append(('close',inputs))
        def install(module):
            module.V25NativeInputs=Inputs
            return {'format':'DEBUG_original_CPU_runtime'}
        def bind(inputs):
            if state['CUDA_initialized']:raise RuntimeError('DEBUG binding after CUDA')
            events.append(('bind',False));return SimpleNamespace(contract={'DEBUG_CPU_ONLY':True})
        def build(*arguments,inputs=None,**keywords):
            events.append(('build',inputs,state['CUDA_initialized']));return inputs
        args=SimpleNamespace(mode=mode,arm='B',output=execution.ROOT/'tmp'/'DEBUG_unused_arm',
            config=Path('DEBUG_config'),native_experiment=Path('DEBUG_native'),
            inventory=Path('DEBUG_inventory'),input_cache=Path('DEBUG_cache'))
        patches=(patch.object(factory,'V25NativeInputs',Inputs),patch.object(factory,'build_runtime',side_effect=build),
            patch.object(preparation,'AcceleratedNativeInputs',Inputs),patch.object(preparation,'install_runtime',side_effect=install),
            patch.object(reuse,'install',side_effect=bind),patch.object(frozen,'publish'),
            patch.object(frozen,'configuration',return_value={'runtime':{'DEBUG_CPU_ONLY':True}}))
        return factory,preparation,frozen,args,events,state,constructed,patches

    def test_all_CUDA_modes_bind_same_input_before_CUDA_and_supply_it_to_original_build(self):
        from contextlib import ExitStack
        for mode in ('calibrate','train','score-cp'):
            with self.subTest(mode=mode):
                factory,preparation,frozen,args,events,state,instances,patches=self._fixture(mode)
                with ExitStack() as stack:
                    for item in patches:stack.enter_context(item)
                    original_install=preparation.install_runtime;original_build=factory.build_runtime
                    original_constructor=factory.V25NativeInputs
                    with execution.prefix_geometry_runtime(args,frozen):
                        self.assertEqual(preparation.install_runtime(factory),{'format':'DEBUG_original_CPU_runtime'})
                        self.assertEqual(len(instances),1)
                        self.assertEqual([item for item in events if item[0]=='bind'],[('bind',False)])
                        state['CUDA_initialized']=True
                        self.assertIs(factory.build_runtime(args,{'DEBUG':True},object()),instances[0])
                    self.assertIs(preparation.install_runtime,original_install)
                    self.assertIs(factory.build_runtime,original_build)
                    self.assertIs(factory.V25NativeInputs,original_constructor)
                    self.assertIn(('build',instances[0],True),events)
                    self.assertEqual([item for item in events if item[0]=='close'],[('close',instances[0])])

    def test_prepare_constructor_binds_each_instance_and_common_stage_is_unchanged(self):
        from contextlib import ExitStack
        factory,preparation,frozen,args,events,state,instances,patches=self._fixture('prepare')
        with ExitStack() as stack:
            for item in patches:stack.enter_context(item)
            with execution.prefix_geometry_runtime(args,frozen):
                preparation.install_runtime(factory)
                self.assertEqual(instances,[])
                factory.V25NativeInputs('DEBUG source')
                self.assertEqual(len(instances),1)
            events.clear();instances.clear()
            original=preparation.install_runtime
            with execution.prefix_geometry_runtime(args,frozen,common_only=True):
                self.assertIs(preparation.install_runtime,original)
                preparation.install_runtime(factory)
                factory.V25NativeInputs('DEBUG source')
            self.assertFalse(any(item[0]=='bind' for item in events))

    def test_early_lease_failure_closes_preconstructed_inputs_and_restores_modules(self):
        from contextlib import ExitStack
        factory,preparation,frozen,args,events,state,instances,patches=self._fixture('train')
        with ExitStack() as stack:
            for item in patches:stack.enter_context(item)
            original=preparation.install_runtime
            with self.assertRaisesRegex(RuntimeError,'DEBUG lease failed'):
                with execution.prefix_geometry_runtime(args,frozen):
                    preparation.install_runtime(factory)
                    raise RuntimeError('DEBUG lease failed')
            self.assertIs(preparation.install_runtime,original)
            self.assertEqual([item for item in events if item[0]=='close'],[('close',instances[0])])

    def test_geometry_binding_failure_closes_original_input_owner(self):
        from contextlib import ExitStack
        from hiercp_v1x import v25_prefix_geometry_reuse as reuse
        factory,preparation,frozen,args,events,state,instances,patches=self._fixture('calibrate')
        with ExitStack() as stack:
            for item in patches:stack.enter_context(item)
            with patch.object(reuse,'install',side_effect=ValueError('DEBUG invalid full128 binding')):
                with self.assertRaisesRegex(ValueError,'DEBUG invalid full128 binding'):
                    with execution.prefix_geometry_runtime(args,frozen):
                        preparation.install_runtime(factory)
            self.assertEqual([item for item in events if item[0]=='close'],[('close',instances[0])])


class NativeTerminalExecutionTests(unittest.TestCase):
    """DEBUG CLI guard wiring, without any actual native training."""
    def test_downstream_installs_terminal_guard_before_first_native_lease_request(self):
        from tools import run_v25_downstream as downstream_cli
        from hiercp_v1x import v25_downstream as downstream,v25_terminal_descendant_resources as terminal
        events=[]
        def install():events.append('guard');return {'DEBUG':True}
        def run(**keywords):events.append('native');return {'DEBUG':True}
        with tempfile.TemporaryDirectory(prefix='DEBUG_v25_native_guard_',dir=execution.ROOT/'tmp') as temporary:
            root=Path(temporary)
            with patch.object(downstream_cli,'execution_cpu_affinity',return_value=list(range(6))), \
                    patch.object(terminal,'install_runtime',side_effect=install), \
                    patch.object(downstream,'run_downstream',side_effect=run),patch('builtins.print'):
                result=downstream_cli.main(['--output',str(root),'--arm','A','--gpu','6',
                    '--score-overlay','DEBUG_overlay','--storage-admission','DEBUG_storage',
                    '--project','DEBUG_project','--basic-validation','DEBUG_basic','--cpu-affinity','0,1,2,3,4,5',
                    '--allocation-request','DEBUG_allocation','--evaluator-root','DEBUG_evaluator'])
            self.assertEqual(result,0);self.assertEqual(events,['guard','native'])
            records=list((root/'execution').glob('terminal_descendant_*.json'))
            self.assertEqual(len(records),1)
            self.assertEqual(json.loads(records[0].read_text())['terminal_descendant_runtime'],{'DEBUG':True})

    def test_fresh_native_stage_installs_guard_before_constructing_direct_lease(self):
        import os
        from hiercp_v1x import v25_native_cp as native,v25_cp as cp
        from hiercp_v1x import v25_direct_resources as direct,v25_terminal_descendant_resources as terminal
        events=[]
        with tempfile.TemporaryDirectory(prefix='DEBUG_v25_fresh_native_guard_',dir=execution.ROOT/'tmp') as temporary:
            group=Path(temporary);proof_path=group/'DEBUG_allocation.json';request_path=group/'DEBUG_request.json'
            proof=dict(format=direct.DIRECT_FORMAT,DEBUG_CPU_SYNTHETIC_METADATA=True,
                arm='C',gpu=1,CPU_affinity=list(range(6)),group_root=str(group))
            proof_path.write_text(json.dumps(proof),encoding='utf8')
            request_path.write_text(json.dumps({'DEBUG':True}),encoding='utf8')
            declared=dict(proof,output=str(group/'C'),allocation_request=str(proof_path),
                allocation_request_sha256=cp._sha(proof_path))
            lease=SimpleNamespace(request=proof,verify=lambda **kwargs:events.append('verify'),
                select_cuda=lambda:events.append('select'))
            def installer():events.append('install');return {'DEBUG':True}
            def constructor(path):events.append('lease');return lease
            with patch.dict(os.environ,{native.ENV_SHA:cp._sha(request_path)},clear=True), \
                    patch.object(native,'request',return_value=(declared,request_path)), \
                    patch.object(native,'_GUARDS',{}), \
                    patch.object(native,'require_project_budget',return_value={'DEBUG':True}), \
                    patch.object(terminal,'install_runtime',side_effect=installer), \
                    patch.object(direct,'DirectAllocationLease',side_effect=constructor):
                native.early_stage_guard(stage_action='train',gpu=1)
            self.assertEqual(events,['install','lease','verify','select'])
            self.assertIn('hiercp_v1x/v25_terminal_descendant_resources.py',native.FILES)


if __name__=='__main__':unittest.main()
