"""CPU UNIT only: DEBUG probe policy and complete real tensor observation.

Small synthetic CPU graphs test the observer, not clinical model performance.
No CUDA work, production training, server calls or relaxed numerical gate.
"""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData, Batch

from tools import run_v24_performance_probe as probe
from hiercp_v1x.transition_v1_local import TransitionLocalBatch
from hiercp_v1x.v24_hash_runtime import tensor_digest
from hiercp_v1x.v24_inputs import tensor_digest as original_digest


def fake_torch():
    state={'enabled':False,'warn':True}
    value=SimpleNamespace(
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False,benchmark=True,allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
        utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True)),
        are_deterministic_algorithms_enabled=lambda:state['enabled'],
        is_deterministic_algorithms_warn_only_enabled=lambda:state['warn'],
        get_float32_matmul_precision=lambda:'high')
    def set_policy(enabled,*,warn_only):state.update(enabled=enabled,warn=warn_only)
    value.use_deterministic_algorithms=set_policy
    return value


def graph(index,view):
    result=HeteroData()
    result['candidate'].x=torch.tensor([[float(index),-0.],[float(view),1.]])
    result['candidate'].pos=torch.arange(6,dtype=torch.float64).reshape(2,3)+index
    result['candidate','spatial','candidate'].edge_index=torch.tensor([[0,1],[1,0]])
    result['candidate','spatial','candidate'].edge_attr=torch.tensor([[1.],[2.]])
    result.actual_index=index;result.view=view
    return result


class Provider:
    """Genuine PyG tensors in a CPU-only observer fixture."""
    def __init__(self):
        self.ds=SimpleNamespace(rows=[{'id':f'record{i}'} for i in range(5)])
        self.batches=[];self.offset=0.;self.one_view=False

    def get(self,indices,*,epoch=0):
        views=1 if self.one_view else 2
        batch=TransitionLocalBatch(
            Batch.from_data_list([graph(index,view) for index in indices for view in range(views)]),
            torch.arange(40,dtype=torch.float32).reshape(1,5,2,2,2),
            torch.arange(len(indices)*40,dtype=torch.float32).reshape(len(indices),5,2,2,2)+self.offset,
            torch.zeros(len(indices),dtype=torch.long),
            torch.arange(len(indices)).repeat_interleave(views),torch.tensor(indices))
        self.batches.append(batch)
        return batch


class Geometry:
    recipient_GT_used_in_forward=False
    def __init__(self):self.values=[];self.offset=0.
    def __call__(self,plan,provider=None):
        upper=graph(len(plan.record_ids),0)
        upper['candidate'].x+=self.offset
        prototype=graph(8,1)
        value=(upper,prototype,{'recipient_GT_used_in_forward':False})
        self.values.append(value)
        return value


def fixture():
    provider=Provider();geometry=Geometry()
    scorer=SimpleNamespace(providers={'inner_train':provider},geometry=geometry)
    plans=[SimpleNamespace(partition='inner_train',case_id='patientA',record_ids=('record0','record1','record2')),
        SimpleNamespace(partition='inner_train',case_id='patientB',record_ids=('record3','record4'))]
    return scorer,provider,geometry,plans


def collect(*,offset=0.,upper_offset=0.,digest=tensor_digest):
    scorer,provider,geometry,plans=fixture();provider.offset=offset;geometry.offset=upper_offset
    observer=probe.InputTensorObserver(scorer,digest)
    for plan in plans:
        value=scorer.geometry(plan,provider)
        if value is not geometry.values[-1]:raise AssertionError('Observer changed upper return identity')
    for indices in ([0,1],[2,3],[4]):
        value=provider.get(indices,epoch=1)
        if value is not provider.batches[-1]:raise AssertionError('Observer changed local return identity')
    result=observer.proof(plans,epoch=1,physical_candidate_chunk=2)
    observer.close()
    return result,scorer,provider,geometry


class NumericalPolicy(unittest.TestCase):
    def test_explicit_flag_default_and_remaining_original_args(self):
        args=['--variant','baseline','--source-output','source','--source-code','code',
            '--probe-output','new','--debug-performance-probe','--gpu','6']
        options,remaining=probe.parse_probe_options(args)
        self.assertFalse(options.debug_deterministic_numerics)
        self.assertEqual(remaining,['--gpu','6'])
        options,remaining=probe.parse_probe_options(args+['--debug-deterministic-numerics'])
        self.assertTrue(options.debug_deterministic_numerics)
        self.assertEqual(remaining,['--gpu','6'])

    def test_environment_before_cuda_and_late_init_rejected(self):
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':'prior'}):
            probe.configure_debug_environment(False)
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],'prior')
            with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))}):
                probe.configure_debug_environment(True)
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True))}):
                with self.assertRaisesRegex(RuntimeError,'pre-CUDA'):probe.configure_debug_environment(True)

    def test_production_original_policy_unchanged_without_debug_flag(self):
        value=fake_torch()
        before=probe.numerical_flags(value)
        receipt=probe.configure_debug_numerics(value,False)
        self.assertEqual(probe.numerical_flags(value),before)
        self.assertFalse(receipt['production_config_modified'])
        self.assertFalse(receipt['debug_deterministic_numerics'])

    def test_strict_debug_policy_preserves_precision(self):
        value=fake_torch()
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            receipt=probe.configure_debug_numerics(value,True)
        actual=receipt['actual_forward_flags']
        self.assertTrue(actual['deterministic_algorithms'])
        self.assertFalse(actual['deterministic_warn_only'])
        self.assertTrue(actual['cudnn_deterministic']);self.assertFalse(actual['cudnn_benchmark'])
        for name in ('matmul_allow_tf32','cudnn_allow_tf32','float32_matmul_precision'):
            self.assertEqual(actual[name],receipt['original_after_build_flags'][name])

    def test_missing_workspace_and_invalid_flags_fail_closed(self):
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':'wrong'}):
            with self.assertRaises(ValueError):probe.configure_debug_numerics(fake_torch(),True)
        for flag in (1,None,'true'):
            with self.assertRaises(TypeError):probe.configure_debug_environment(flag)
            with self.assertRaises(TypeError):probe.configure_debug_numerics(fake_torch(),flag)

    def test_unsupported_kernel_policy_error_has_no_fallback(self):
        value=fake_torch()
        def unsupported(*args,**kwargs):raise RuntimeError('unsupported deterministic operator')
        value.use_deterministic_algorithms=unsupported
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            with self.assertRaisesRegex(RuntimeError,'unsupported deterministic'):
                probe.configure_debug_numerics(value,True)


class FullTensorProof(unittest.TestCase):
    def test_original_vs_optimized_hash_complete_real_tensors_and_no_rng_change(self):
        state=torch.get_rng_state().clone()
        original,*_=collect(digest=original_digest)
        optimized,scorer,provider,geometry=collect()
        self.assertEqual(original['content_sha256'],optimized['content_sha256'])
        self.assertEqual(original['content'],optimized['content'])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(optimized['complete']);self.assertEqual(optimized['ordered_records'],5)
        self.assertEqual(optimized['observed_native_chunks'],3)
        self.assertEqual(optimized['observed_upper_graphs'],2)
        self.assertIs(scorer.geometry,geometry);self.assertNotIn('get',vars(provider))
        self.assertTrue(optimized['every_actual_value_hashed'])
        self.assertFalse(optimized['hash_memoization'])

    def test_changed_local_or_upper_tensor_changes_full_proof(self):
        original,*_=collect()
        local,*_=collect(offset=1.)
        upper,*_=collect(upper_offset=1.)
        self.assertNotEqual(original['content_sha256'],local['content_sha256'])
        self.assertNotEqual(original['content_sha256'],upper['content_sha256'])

    def test_actual_cpu_input_values_are_returned_identically(self):
        scorer,provider,geometry,plans=fixture()
        observer=probe.InputTensorObserver(scorer,tensor_digest)
        upper=scorer.geometry(plans[0],provider)
        self.assertIs(upper,geometry.values[0])
        local=provider.get([0,1],epoch=1)
        before=tensor_digest(local.graph.to_dict())
        self.assertIs(local,provider.batches[0])
        self.assertEqual(before,tensor_digest(local.graph.to_dict()))
        observer.close()

    def test_missing_order_epoch_or_chunk_boundary_is_rejected(self):
        for schedule,epoch,chunk in (([[0,1],[2,3]],1,2),([[1,0],[2,3],[4]],1,2),
                ([[0,1],[2,3],[4]],2,2),([[0,1,2],[3,4]],1,2)):
            scorer,provider,geometry,plans=fixture()
            observer=probe.InputTensorObserver(scorer,tensor_digest)
            try:
                for plan in plans:scorer.geometry(plan,provider)
                for indices in schedule:provider.get(indices,epoch=epoch)
                with self.assertRaisesRegex(ValueError,'complete ordered'):
                    observer.proof(plans,epoch=1,physical_candidate_chunk=chunk)
            finally:observer.close()

    def test_missing_upper_and_one_view_are_rejected(self):
        scorer,provider,geometry,plans=fixture();observer=probe.InputTensorObserver(scorer,tensor_digest)
        try:
            scorer.geometry(plans[0],provider)
            for indices in ([0,1],[2,3],[4]):provider.get(indices,epoch=1)
            with self.assertRaisesRegex(ValueError,'complete patient'):
                observer.proof(plans,epoch=1,physical_candidate_chunk=2)
            provider.one_view=True
            with self.assertRaisesRegex(ValueError,'two views'):provider.get([0],epoch=1)
        finally:observer.close()

    def test_instance_get_restored_and_foreign_geometry_refused(self):
        scorer,provider,geometry,plans=fixture()
        original=provider.get;provider.get=original
        observer=probe.InputTensorObserver(scorer,tensor_digest);observer.close()
        self.assertIs(provider.get,original)
        observer=probe.InputTensorObserver(scorer,tensor_digest);scorer.geometry=Geometry()
        with self.assertRaisesRegex(ValueError,'changed'):observer.close()
        scorer.geometry=observer.geometry;observer.close()


class SourceOrdering(unittest.TestCase):
    def test_complete_script_compiles_and_debug_policy_is_after_original_build(self):
        source=Path(probe.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
        compile(tree,str(probe.__file__),'exec')
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='main')
        calls={}
        for node in ast.walk(main):
            if isinstance(node,ast.Call):calls.setdefault(ast.unparse(node.func),[]).append(node.lineno)
        self.assertLess(calls['configure_debug_environment'][0],calls['torch.cuda.is_available'][0])
        self.assertLess(calls['v24_factory.build_runtime'][0],calls['configure_debug_numerics'][0])
        self.assertLess(calls['input_runtime.install_runtime'][0],calls['torch.cuda.is_available'][0])
        self.assertLess(calls['memory.bind_memory_runtime'][0],calls['input_runtime.bind'][0])
        self.assertLess(calls['prefetch.bind'][0],calls['input_runtime.bind'][0])
        self.assertLess(calls['hashes.install'][0],calls['input_runtime.install_runtime'][0])
        self.assertNotIn('optimizer.step',calls);self.assertNotIn('scaler.step',calls)
        self.assertNotIn('native_scientific_equivalence_passed',source)
        self.assertIn('ordered_CPU_input_sha256',source)


if __name__=='__main__':unittest.main()
