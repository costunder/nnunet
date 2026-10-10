"""DEBUG CPU fixtures for memory-only source and actual tensor admission.

Synthetic scalar gradients/scores exercise gate failure handling. They are
not clinical inputs, a production model, CUDA evidence or a speed benchmark.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools import v24_memory_reclamation_gate as gate
from tools import run_v24_memory_reclamation_continuation as continuation
from tests import test_v24_performance_continuation as shared_fixtures


class MemoryOnlyActualTensorGateDEBUG(unittest.TestCase):
    def setUp(self):
        self.fixture=shared_fixtures.IncrementalCurrentProbeDebug();self.fixture.setUp()
        self.root=self.fixture.root;self.source=self.root/'preserved_DEBUG_source';self.source.mkdir()
        (self.source/'hiercp_v1x').mkdir()
        for name in (*gate.UNCHANGED_NAMES,'v24_memory_runtime.py'):
            shutil.copyfile(gate.ROOT/'hiercp_v1x'/name,self.source/'hiercp_v1x'/name)
        with (self.source/'hiercp_v1x/v24_memory_runtime.py').open('a',encoding='utf8') as stream:
            stream.write('\n# Explicit DEBUG source-identity fixture; no production performance evidence.\n')
        self.provenance=gate.memory_runtime_provenance(self.source)
        from hiercp_v1x import v24_memory_runtime as memory
        candidate=memory.MemorySafeCoordinator(64*2**30).memory_runtime_receipt()
        old_path=self.source/'hiercp_v1x/v24_memory_runtime.py'
        spec=importlib.util.spec_from_file_location('hiercp_v1x._DEBUG_memory_source_fixture',old_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old=module.MemorySafeCoordinator(64*2**30).memory_runtime_receipt()
        self.baseline=self.fixture.baseline;self.optimized=self.fixture.optimized
        for row,variant,receipt,prefix in ((self.baseline,'current-control',old,'preserved'),
                (self.optimized,'optimized',candidate,'candidate')):
            row.pop('comparison_current_runtime_provenance',None)
            row.update(format=gate.PROBE_FORMAT,variant=variant,memory_only_execution_change=True,
                memory_runtime_source_sha256=self.provenance[prefix+'_memory_sha256'],
                memory_runtime_module_path=self.provenance[prefix+'_memory_path'],memory_runtime=receipt,
                comparison_memory_runtime_provenance=copy.deepcopy(self.provenance),
                probe_source_sha256=gate._sha(gate.ROOT/'tools/run_v24_memory_reclamation_probe.py'),
                input_runtime=dict(contract=dict(runtime_source_sha256=self.provenance['input_adapter_sha256'])),
                hash_runtime=dict(explicit_DEBUG_contract=True),
                prefetch_runtime=dict(explicit_DEBUG_contract=True,
                    memory_runtime_source_sha256=self.provenance[prefix+'_memory_sha256'],
                    original_locked_pressure_function_sha256=self.provenance[prefix+'_pressure_function_sha256']),
                actual_memory_module_selection=dict(actual_module_sha256=self.provenance[prefix+'_memory_sha256'],
                    actual_module_path=self.provenance[prefix+'_memory_path'],actual_contract=receipt['contract'],
                    source_provenance=copy.deepcopy(self.provenance),imported_before_prefetch_input_factory=True),
                trainable_parameter_schema={f'CPU_GATE_fixture_parameter_{i}':dict(shape=[1],dtype='torch.float32',numel=1)
                                           for i in range(537)})
            folder=self.root/(variant+'_DEBUG_actual_tensor_fixture');folder.mkdir()
            destination=folder/'numerical_tensors.pt'
            shutil.copyfile(row['numerical_tensor_file']['path'],destination)
            row['numerical_tensor_file']['path']=str(destination)
            row['numerical_tensor_file']['raw_sha256']=gate._sha(destination)
            self.write_report(row)

    def tearDown(self):self.fixture.tearDown()

    def write_report(self,row):
        path=Path(row['numerical_tensor_file']['path']).parent/'result.json'
        path.write_text(json.dumps(row),encoding='utf8')
        return path

    def compare(self):
        return gate.compare_probe_files(self.write_report(self.baseline),self.write_report(self.optimized),'actual-snapshot')

    def test_owned_actual537_tensor_bits_and_real_sources_pass_without_unobserved_pressure_claim(self):
        result=self.compare()
        self.assertTrue(result['numerical_tensor_comparison']['all_tensors_bitwise_equal'])
        self.assertEqual(result['numerical_tensor_comparison']['all_named_trainable_gradients'],537)
        self.assertFalse(result['observed_pressure_in_both']);self.assertFalse(result['memory_reclamation_speedup_claimed'])
        self.assertNotEqual(self.provenance['preserved_memory_sha256'],self.provenance['candidate_memory_sha256'])
        self.assertEqual(len(result['owned_report_and_artifact_proofs']),4)

    def test_scientific_hash_prefetch_input_changes_are_refused(self):
        for name in ('v24_factory.py','v24_inputs.py','v24_hash_runtime.py','v24_prefetch_runtime.py','v24_input_runtime.py'):
            path=self.source/'hiercp_v1x'/name;original=path.read_bytes();path.write_bytes(original+b'\n# DEBUG mutated source\n')
            with self.subTest(source=name),self.assertRaisesRegex(ValueError,'Only memory reclamation'):
                gate.memory_runtime_provenance(self.source)
            path.write_bytes(original)

    def test_changed_memory_artifact_or_source_binding_fails(self):
        for name in ('memory_runtime_source_sha256','memory_runtime_module_path','probe_source_sha256'):
            original=self.optimized[name];self.optimized[name]='0'*64
            with self.subTest(field=name),self.assertRaises(ValueError):self.compare()
            self.optimized[name]=original
        path=Path(self.optimized['numerical_tensor_file']['path'])
        with path.open('ab') as stream:stream.write(b'DEBUG tamper')
        with self.assertRaisesRegex(ValueError,'source/SHA/stat'):self.compare()

    def test_pressure_limit_unexpected_release_or_wrong_real_prefetch_binding_fails(self):
        for mutate in (lambda row:row['memory_runtime'].update(RSS_limit_bytes=65*2**30),
                lambda row:row['memory_runtime']['profile'].update(strict_failures=1),
                lambda row:row['prefetch_runtime'].update(memory_runtime_source_sha256='0'*64),
                lambda row:row['prefetch_runtime'].update(original_locked_pressure_function_sha256='0'*64),
                lambda row:row['actual_memory_module_selection'].update(imported_before_prefetch_input_factory=False)):
            original=copy.deepcopy(self.optimized);mutate(self.optimized)
            with self.assertRaises(ValueError):self.compare()
            self.optimized=original

    def test_actual_gradient_signedzero_tamper_cannot_pass_metadata_digest_equality(self):
        from hiercp_v1x.u_bridge_training import digest
        path=Path(self.optimized['numerical_tensor_file']['path']);document=torch.load(path,map_location='cpu',weights_only=True)
        document['gradients']['CPU_GATE_fixture_parameter_0'][0]=-0.
        torch.save(document,path)
        row=self.optimized;row['gradient_sha256']=digest(document['gradients']);self.baseline['gradient_sha256']=row['gradient_sha256']
        row['numerical_tensor_file'].update(raw_sha256=gate._sha(path),size=path.stat().st_size,content_sha256=digest(document))
        with self.assertRaisesRegex(ValueError,'values differ|bits differ'):self.compare()

    def test_complete_parameter_identity_schema_checkpoint_and_optimizer_are_mandatory(self):
        for mutate in (lambda row:row['trainable_parameter_schema'].pop('CPU_GATE_fixture_parameter_0'),
                lambda row:row['trainable_parameter_schema']['CPU_GATE_fixture_parameter_0'].update(dtype='torch.float64'),
                lambda row:row['source_checkpoint']['numerical_state_sha256'].pop('scheduler'),
                lambda row:row.update(after_optimizer_sha256='0'*64),
                lambda row:row.update(optimizer_contains_all_trainable_parameters_exactly_once=False),
                lambda row:row['input_tensor_proof'].update(ordered_records=523)):
            original=copy.deepcopy(self.optimized);mutate(self.optimized)
            with self.assertRaises(ValueError):self.compare()
            self.optimized=original

    def test_same_report_or_foreign_owned_evidence_is_rejected(self):
        path=self.write_report(self.baseline)
        with self.assertRaisesRegex(ValueError,'Distinct'):gate.compare_probe_files(path,path,'actual-snapshot')
        with patch.object(gate.os,'getuid',return_value=path.lstat().st_uid+1,create=True):
            with self.assertRaisesRegex(ValueError,'Owned regular'):gate._guard(path)

    def test_cold_subprocess_selects_real_old_memory_for_all_three_adapter_class_captures(self):
        script='''from pathlib import Path
from tools.v24_memory_reclamation_gate import route_memory_runtime
proof=route_memory_runtime(Path(SOURCE),"current-control")
from hiercp_v1x import v24_memory_runtime as memory,v24_factory as factory,v24_prefetch_runtime as prefetch,v24_input_runtime as inputs
assert Path(memory.__file__).resolve()==Path(SOURCE)/"hiercp_v1x/v24_memory_runtime.py"
assert prefetch.memory is memory and inputs.memory is memory
assert prefetch._COORDINATOR is memory.MemorySafeCoordinator
memory.install_memory_runtime(factory);prefetch.install_runtime(memory);inputs.install_runtime(pin_final_outputs=True)
assert factory.V24InputProvider is memory.MemorySafeInputProvider
assert memory._GET.__globals__["materialize_pair"] is inputs.materialize_pair
print("DEBUG cold actual old module/classes/helper capture PASS; no CUDA or training")
'''.replace('SOURCE',repr(str(self.source)))
        result=subprocess.run([sys.executable,'-X','utf8','-B','-c',script],cwd=gate.ROOT,text=True,capture_output=True,check=False)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('capture PASS',result.stdout)

    def test_candidate_continuation_proof_binds_actual_current_file(self):
        value=continuation.candidate_continuation_runtime_proof()
        self.assertEqual(value['memory_module_path'],str(gate.ROOT/'hiercp_v1x/v24_memory_runtime.py'))
        self.assertEqual(value['memory_module_sha256'],self.provenance['candidate_memory_sha256'])
        self.assertEqual(value['memory_runtime_contract']['runtime_source_sha256'],self.provenance['candidate_memory_sha256'])

    def test_original_production_function_clone_preserves_code_and_global_function(self):
        def original(left):return selected(left)
        marker=lambda value:('DEBUG_only',value)
        clone=continuation._clone(original,selected=marker)
        self.assertIs(clone.__code__,original.__code__)
        self.assertEqual(clone(42),('DEBUG_only',42));self.assertNotIn('selected',original.__globals__)


if __name__=='__main__':unittest.main()
