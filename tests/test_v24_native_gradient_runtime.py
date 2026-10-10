"""CPU DEBUG actual decoder source proof and unchanged native AMP simulations."""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from hiercp_v1x import v24_native_gradient_runtime as runtime
from hiercp_v1x import v24_nnunet_cp as pipeline
from tests import test_v24_native_amp_retry as AMP_fixtures


class NativeGradientRuntimeDebug(unittest.TestCase):
    def state(self,heads=5):
        return SimpleNamespace(deep_supervision=True,seg_layers=[object() for _ in range(heads)],
            stages=[object() for _ in range(heads)])

    def source(self):return inspect.getsource(UNetDecoder.forward)

    def test_actual_UNetDecoder_assignment_alias_return_and_head_count_proved(self):
        source=self.source();proof=runtime.prove_decoder_head_order(self.state(),source)
        self.assertIn('one reverse assignment',proof['output_dataflow'])
        self.assertTrue(proof['deep_supervision_active']);self.assertEqual(proof['head_count'],5)
        self.assertEqual(proof['reversal_count'],1)
        import textwrap
        self.assertEqual(proof['decoder_forward_source_sha256'],hashlib.sha256(textwrap.dedent(source).encode()).hexdigest())

    def test_semantic_reverse_head_index_return_mutation_and_stage_skipping_are_rejected(self):
        source=self.source()
        variants=[source.replace('seg_outputs = seg_outputs[::-1]','seg_outputs = seg_outputs'),
            source.replace('seg_outputs = seg_outputs[::-1]','seg_outputs = seg_outputs[::-1]\n        seg_outputs = seg_outputs[::-1]'),
            source.replace('self.seg_layers[s](x)','self.seg_layers[-1](x)'),
            source.replace('range(len(self.stages))','range(len(self.stages)-1)'),
            source.replace('r = seg_outputs\n','r = seg_outputs[0]\n'),
            source.replace('return r','return seg_outputs[0]'),
            source.replace('return r','seg_outputs.clear()\n        return r'),
            source.replace('lres_input = x','lres_input = x\n            break'),
            source.replace('lres_input = x','lres_input = x\n            s = 0'),
            source.replace('return r','r = seg_outputs[0]\n        return r'),
            source.replace('seg_outputs = []','seg_outputs = [skips[0]]'),
            source.replace('seg_outputs = []','seg_outputs = []\n        aliased = seg_outputs'),
            source.replace('seg_outputs = []','seg_outputs = []\n        self.seg_layers.reverse()')]
        for changed in variants:
            self.assertNotEqual(changed,source)
            with self.subTest(source=changed),self.assertRaisesRegex(ValueError,'head ordering is unproved'):
                runtime.prove_decoder_head_order(self.state(),changed)

    def test_actual_supervision_flag_and_all_stage_head_counts_required(self):
        false=self.state();false.deep_supervision=False
        mismatch=self.state();mismatch.stages=mismatch.stages[:-1]
        missing=self.state();del missing.deep_supervision
        for state in (false,mismatch,missing):
            with self.subTest(state=state),self.assertRaisesRegex(ValueError,'active complete'):
                runtime.prove_decoder_head_order(state,self.source())

    def test_only_helper_expression_changes_and_step_code_and_global_helpers_stay_original(self):
        before=runtime._source_function(runtime.HELPER);after,expression=runtime._replace_expression(before)
        assignments=[node for node in ast.walk(after) if isinstance(node,ast.Assign)
            and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='reversed_outputs']
        assignments[0].value=expression
        self.assertEqual(ast.dump(before),ast.dump(after))
        original=pipeline._native_clone_step_with_amp_retry
        old_helper=original.__globals__[runtime.HELPER];cloned=runtime.clone_step(original)
        self.assertIs(cloned.__code__,original.__code__)
        self.assertIs(original.__globals__[runtime.HELPER],old_helper)
        self.assertIsNot(cloned.__globals__[runtime.HELPER],old_helper)
        contract=runtime.gradient_runtime_contract();self.assertEqual(contract,runtime.gradient_runtime_contract())
        self.assertTrue(contract['original_step_code_preserved'])
        self.assertTrue(contract['loss_weights_parameter_identity_and_AMP_equations_unchanged'])

    def test_actual_assignment_decoder_zero_weight_mapping_and_source_file_receipt(self):
        class ActualForwardDecoderMetadataDebug(torch.nn.Module):
            # The actual library method is inspected, not replaced or executed
            # on a smaller pretend native model. Only head identity is tested.
            forward=UNetDecoder.forward
            def __init__(self):
                super().__init__();self.deep_supervision=True
                self.seg_layers=torch.nn.ModuleList([torch.nn.Linear(2,1) for _ in range(5)])
                self.stages=torch.nn.ModuleList([torch.nn.Identity() for _ in range(5)])
        network=torch.nn.Module();network.decoder=ActualForwardDecoderMetadataDebug()
        trainer=SimpleNamespace(network=network,loss=SimpleNamespace(weight_factors=[.5,.25,.125,.125,0.]),enable_deep_supervision=True)
        parameters=dict(network.named_parameters())
        with self.assertRaisesRegex(ValueError,'head ordering is unproved'):
            pipeline._native_inactive_auxiliary_gradient_proof(trainer,parameters)
        cloned=runtime.clone_step(pipeline._native_clone_step_with_amp_retry)
        helper=cloned.__globals__[runtime.HELPER];proof=helper(trainer,parameters)
        self.assertEqual(proof['zero_weight_output_indices'],[4]);self.assertEqual(proof['inactive_decoder_layer_indices'],[0])
        self.assertEqual(proof['official_inactive_parameter_names'],['decoder.seg_layers.0.weight','decoder.seg_layers.0.bias'])
        source_path=Path(inspect.getsourcefile(UNetDecoder.forward)).resolve()
        self.assertEqual(proof['gradient_runtime']['decoder_source_file'],str(source_path))
        self.assertEqual(proof['gradient_runtime']['decoder_source_file_sha256'],hashlib.sha256(source_path.read_bytes()).hexdigest())
        trainer.enable_deep_supervision=False
        with self.assertRaisesRegex(ValueError,'both be active'):helper(trainer,parameters)
        trainer.enable_deep_supervision=True
        with self.assertRaisesRegex(ValueError,'parameters'):
            helper(trainer,{k:v for k,v in parameters.items() if k!='decoder.seg_layers.0.weight'})


class OriginalAMPTestsThroughRuntimeDebug(AMP_fixtures.NativeAMPAdmissionDebug):
    """All nine established AMP invariants run through the private step clone."""
    def run_update(self,trainer):
        parameters=dict(trainer.network.named_parameters())
        before={name:p.detach().clone() for name,p in parameters.items()}
        with patch('torch.cuda.is_available',return_value=False):
            return runtime.clone_step(pipeline._native_clone_step_with_amp_retry)(trainer,self.batch,parameters,before)


if __name__=='__main__':unittest.main()
