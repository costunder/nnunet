"""CPU DEBUG exact archived equations, not native CT/CUDA performance results."""
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

from tests.test_v24_upper_reuse import UpperReuseDebug
from hiercp_v1x import v25_prefix_geometry_reuse as reuse
from hiercp_v1x.v24_inputs import tensor_digest


class V25PrefixGeometryReuseDebug(UpperReuseDebug):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='DEBUG_v25_prefix_reuse_')
        self.addCleanup(self.directory.cleanup)
        self.state = dict(valid=True)
        self.inputs = self.make_inputs(Path(self.directory.name))
        self.full = self.population.case(self.f.recipient.case_id,128)
        self.inputs.geometry.get(self.full)
        self.adapter = reuse.install(self.inputs)

    def test_V25_U8_U16_U32_full128_match_every_original_tensor_and_audit(self):
        for count in (8,16,32,128):
            with self.subTest(active_U=count):
                plan=self.population.case(self.f.recipient.case_id,count)
                expected=self.original(plan)
                actual=self.adapter.derive(plan)
                self.assertEqual(self.signature(actual),self.signature(expected))
                self.assertEqual(tensor_digest(actual[2]),tensor_digest(expected[2]))
                self.assertEqual(actual[0]['candidate'].num_nodes,len(plan.record_ids))
                self.assertTrue({'P:0','P:1'}.issubset(plan.record_ids))

    def test_full128_and_other_cache_builders_immutable_and_installer_idempotent(self):
        root=Path(__file__).resolve().parents[1]
        files=[root/'hiercp_v1x'/name for name in reuse._FROZEN]+[root/'tools/run_v25.py']
        before={str(path):path.read_bytes() for path in files}
        full_path=self.inputs.geometry._path(self.full)[0];source=full_path.read_bytes()
        other=self.make_inputs(Path(self.directory.name)/'other')
        other_builder=other.geometry.builder
        self.assertIs(reuse.install(self.inputs),self.adapter)
        plan=self.population.case(self.f.recipient.case_id,16)
        self.inputs.geometry.get(plan)
        first=copy.deepcopy(self.adapter.receipt())
        with patch.object(self.inputs.geometry,'builder',side_effect=AssertionError('cached prefix unexpectedly rebuilt')):
            self.inputs.geometry.get(plan)
        self.assertEqual(first,self.adapter.receipt())
        self.assertEqual(full_path.read_bytes(),source)
        self.assertIs(other.geometry.builder,other_builder)
        self.assertEqual(before,{str(path):path.read_bytes() for path in files})
        self.assertEqual(self.adapter.contract['format'],reuse.FORMAT)

    def test_idempotent_CPU_binding_and_foreign_replacement_rejected(self):
        self.assertIs(reuse.install(self.inputs),self.adapter)
        self.inputs.geometry.builder=lambda plan,provider:None
        with self.assertRaisesRegex(ValueError,'Foreign'):
            reuse.install(self.inputs)
        with patch('torch.cuda.is_initialized',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'CPU-only'):
                reuse.install(self.make_inputs(Path(self.directory.name)/'wrong_GPU'))

    def test_CPU_bound_geometry_stays_exact_after_later_CUDA_initialization(self):
        plan=self.population.case(self.f.recipient.case_id,32)
        expected=self.original(plan)
        with patch('torch.cuda.is_initialized',return_value=True):
            actual=self.adapter.derive(plan)
        self.assertEqual(self.signature(actual),self.signature(expected))
        self.assertEqual(tensor_digest(actual[2]),tensor_digest(expected[2]))
        with patch('torch.cuda.is_initialized',side_effect=AssertionError('pure receipt queried CUDA')):
            receipt=reuse.runtime_contract()
        self.assertFalse(receipt['CUDA_context_initialized_before_binding_allowed'])
        self.assertTrue(receipt['CPU_geometry_derivation_after_CUDA_context_initialization_allowed'])


if __name__=='__main__':
    import unittest
    unittest.main()
