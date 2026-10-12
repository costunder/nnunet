"""CPU UNIT admission/route tests; no native training or actual evaluation."""
from __future__ import annotations
import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x import v25_native_cp as native,v25_downstream as downstream,v25_cp as cp
from hiercp_v1x import v24_matched_basic_cp_pipeline as original_pipeline
from hiercp_v1x import v24_matched_basic_cp_runtime as original_runtime


class UnchangedNativeRouteCPUUnit(unittest.TestCase):
    def test_private_module_clone_routes_only_admission_resources_child_entries(self):
        old_pipeline=original_pipeline.prepare_native.__code__;old_runtime=original_runtime._install_overlay.__code__
        modules=native.private_modules();pipeline=modules['NATIVE_PIPELINE'];runtime=modules['NATIVE_RUNTIME']
        self.assertIs(original_pipeline.prepare_native.__code__,old_pipeline)
        self.assertIs(original_runtime._install_overlay.__code__,old_runtime)
        # This complete Basic loader/paste adapter has no changed expression.
        self.assertEqual(runtime._install_overlay.__code__.co_code,old_runtime.co_code)
        self.assertEqual(pipeline.TRAINER,'nnUNetTrainer_250epochs_FrozenV23CP')
        self.assertEqual(pipeline.PLANS,'nnUNetResEncUNetMPlans')
        self.assertIs(pipeline._child,native._child)
        self.assertFalse(pipeline.v25_adaptation_proof['model_loader_paste_loss_optimizer_expressions_changed'])
        self.assertEqual(pipeline.v25_adaptation_proof['admission_expression_counts']
            ["native.get('physical_GPU') not in (4, 5, 6)"],1)
        self.assertIs(runtime.validate_overlay,cp.validate_overlay)

    def test_native_profile_rejects_cross_arm_and_inference_only_policy(self):
        context={'CPU_UNIT_SYNTHETIC_METADATA':True,'depth':1,'inner':0,'outer':10}
        value=dict(arm='C',context_contract=context,GNN_train_validation_CP_context_identical=True)
        overlay=dict(best_provenance={'arm':'C'},context_contract=context)
        with patch.object(native,'request',return_value=({'arm':'C'},None)):
            native._native_profile(value,overlay)
            for bad in (dict(value,arm='A'),dict(value,context_contract={'inner':2}),
                    dict(value,GNN_train_validation_CP_context_identical=False)):
                with self.assertRaises(ValueError):native._native_profile(bad,overlay)

    def test_request_environment_is_byte_sealed_and_v25_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'request.json';cp._publish(path,dict(format=downstream.FORMAT,arm='D'))
            with patch.dict(os.environ,{native.ENV_REQUEST:str(path),native.ENV_SHA:cp._sha(path)}):
                self.assertEqual(native.request()[0]['arm'],'D')
                path.write_text('{"format":"v24_other","arm":"D"}')
                with self.assertRaises(ValueError):native.request()

    def test_resume_configuration_preserves_first_actual_file_and_full_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'actual_nnunet_configuration.json'
            first=dict(CPU_UNIT_SYNTHETIC_METADATA=True,epochs=250,physical_batch=2,train_cases=105,
                validation_cases=26,patch_size=[128]*3,source_entries=642,validation_checkpoint='checkpoint_best.pth')
            native.publish_json(path,first);before=path.read_bytes()
            native.publish_json(path,dict(first,resources={'CPU_UNIT':1}))
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(len(list(path.parent.glob('*resume*'))),1)
            with self.assertRaises(ValueError):native.publish_json(path,dict(first,epochs=249))


class HonestCompletionCPUUnit(unittest.TestCase):
    def test_original_native_NaN_is_null_only_in_the_report_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'summary.json'
            path.write_text(json.dumps(dict(CPU_UNIT_SYNTHETIC_METADATA=True,mean={'1':{'Dice':.91}},
                metric_per_case=[{'2':{'Dice':float('nan'),'IoU':float('nan')}}])))
            original=path.read_bytes();checksum=cp._sha(path)
            copied=downstream._summary_evidence(path)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(copied['summary_sha256'],checksum)
            self.assertEqual(copied['summary']['mean']['1']['Dice'],.91)
            self.assertIsNone(copied['summary']['metric_per_case'][0]['2']['Dice'])
            self.assertEqual(len(copied['summary_nonfinite_copy_normalization']),2)
            self.assertTrue(math.isnan(cp._read(path)['metric_per_case'][0]['2']['Dice']))
            json.dumps(copied,allow_nan=False)

    def test_new_PBS_lease_rebinds_only_resources_and_preserves_scientific_request(self):
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{},clear=True):
            root=Path(temporary);old=root/'old_allocation.json';new=root/'new_allocation.json'
            cp._publish(old,dict(CPU_UNIT_SYNTHETIC_METADATA=True,gpu=6,GPU_UUID='GPU-CPU_UNIT',job='old'))
            cp._publish(new,dict(CPU_UNIT_SYNTHETIC_METADATA=True,gpu=6,GPU_UUID='GPU-CPU_UNIT',job='new'))
            science=dict(format=downstream.FORMAT,arm='A',gpu=6,
                CPU_UNIT_SYNTHETIC_METADATA=True,epochs=250,physical_batch=2,
                allocation_request=str(old),allocation_request_sha256=cp._sha(old),CPU_affinity=[32,33,34,35,36,37])
            path=root/'request.json';cp._publish(path,science);before=path.read_bytes()
            candidate=dict(science,allocation_request=str(new),allocation_request_sha256=cp._sha(new),
                CPU_affinity=[1,2,3,4,5,6])
            execution=downstream._bind_execution(science,candidate,path)
            self.assertTrue(execution.is_file());self.assertEqual(path.read_bytes(),before)
            os.environ.update({native.ENV_REQUEST:str(path),native.ENV_SHA:cp._sha(path)})
            effective,base=native.request()
            self.assertEqual(effective['allocation_request'],str(new));self.assertEqual(effective['epochs'],250)
            self.assertEqual(base,path)
            with self.assertRaises(ValueError):downstream._bind_execution(science,dict(candidate,epochs=249),path)
            foreign=root/'different_gpu.json';cp._publish(foreign,dict(gpu=6,GPU_UUID='GPU-DIFFERENT'))
            with self.assertRaisesRegex(ValueError,'identical calibrated'):
                downstream._bind_execution(science,dict(candidate,allocation_request=str(foreign),
                    allocation_request_sha256=cp._sha(foreign)),path)
            execution.write_text('{}')
            with self.assertRaises(ValueError):native.request()

    def test_failed_train_status_never_claims_complete_scopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cp._publish(root/'request.json',{'CPU_UNIT_SYNTHETIC_METADATA':True})
            downstream._status(root,{'arm':'A'},'train','FAILED',error='CPU_UNIT_EXPECTED_ERROR')
            value=cp._read(root/'status.json')
            self.assertEqual(value['state'],'FAILED');self.assertEqual(value['stage'],'train')
            self.assertFalse(value['full_native_training_completed']);self.assertFalse(value['result_schema1_published'])
            self.assertFalse((root/'completion.json').exists())

    def test_completed_native_metrics_still_needs_report_and_true_receipt_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cp._publish(root/'request.json',{'CPU_UNIT_SYNTHETIC_METADATA':True})
            cp._publish(root/'completion.json',dict(format=downstream.COMPLETE_FORMAT,
                request_sha256=cp._sha(root/'request.json'),complete=True,
                actual_data_evaluation=True,full_native_training_completed=True,
                full_native_evaluation_completed=True,full_lesion_metric_evaluation_completed=True,
                result_schema1_published=False))
            with patch.object(downstream,'_guard_request'):
                with self.assertRaisesRegex(ValueError,'actual own|Actual own'):downstream.verify_completion(root/'completion.json')

    def test_clone_debug_cli_and_full_result_layout_are_separate(self):
        from tools.run_v25_downstream import main
        with patch.object(native,'calibration_worker') as worker:
            self.assertEqual(main(['clone-debug-worker','--native','CPU_UNIT.json',
                '--physical-batch','4','--output','CPU_UNIT_DEBUG']),0)
            worker.assert_called_once_with(Path('CPU_UNIT.json'),4,Path('CPU_UNIT_DEBUG'))
        schema=cp._read(downstream.ROOT/'config/nnunet_result_report_schema_v1.json')
        self.assertEqual(schema['result_schema_version'],1)
        self.assertEqual(len(schema['detection_criteria_order']),6)
        self.assertEqual(schema['checkpoint_required_for_evaluation'],'checkpoint_best.pth')

    def test_unverified_Basic_BEST_keeps44_primary_differences_NA_and_separate_history(self):
        from tools.render_nnunet_result_schema_v1 import build_report
        schema=cp._read(downstream.ROOT/'config/nnunet_result_report_schema_v1.json')
        stats=dict(difference=.1,ci_low=.01,ci_high=.2,permutation_p=.04)
        # Synthetic published values test display policy only, never metric math.
        summary=dict(CPU_UNIT_SYNTHETIC_METADATA=True,validation_cases=26,
            evaluation_contract={'contract_sha256':'a'*64},
            case_tumor_dice=dict(basic_mean=.4,hier_mean=.5,statistics=stats),
            lesion_dice_quality=dict(basic_mean=.4,hier_mean=.5,**stats),criteria={})
        for criterion in schema['detection_criteria_order']:
            summary['criteria'][criterion]=dict(basic_cp=dict(recall=.4,precision=.4,f1=.4,fp_per_case=1.),
                hiercp=dict(recall=.5,precision=.5,f1=.5,fp_per_case=.9),
                statistics={name:stats for name in ('recall','precision','f1','fp_per_case')},
                by_size={name:dict(gt=10,basic_detected=4,hier_detected=5,basic_recall=.4,
                    hier_recall=.5,statistics=stats) for name in schema['size_bins_order']})
        evidence=dict(kind='CPU_UNIT_SYNTHETIC_DISPLAY_ONLY',experiments=[{'experiment':'v2.5 A CPU_UNIT'}],
            comparison=dict(status='HISTORICAL_REFERENCE',reason='CPU_UNIT original Basic BEST unverified'),
            online_evaluation=dict(experiment='v2.5 A CPU_UNIT',summary=summary,
                completion=dict(format='online_eval_completion_v1',complete=True,actual_data_evaluation=True,
                    summary_sha256='b'*64,outputs={'summary.json':'b'*64},evaluation_contract_sha256='a'*64),
                verification=dict(outputs_sha256_verified=True,source='CPU_UNIT SYNTHETIC DISPLAY METADATA')))
        report=build_report(evidence)
        self.assertEqual(len(report['tables']['basic_cp_comparison']),44)
        self.assertTrue(all(row['experiment_minus_basic']is None for row in report['tables']['basic_cp_comparison']))
        self.assertEqual(len(report['historical_reference_comparison']),44)
        self.assertEqual(report['historical_reference_comparison'][0]['experiment_minus_basic'],.1)
        self.assertFalse(report['metric_math_executed'])
        self.assertEqual(downstream.HISTORICAL_EVALUATOR['online_eval_v2.py'],
            '93c703edb20e074645ef443f6ca2bd7db03268bbb90fda8591f337993592270b')


if __name__=='__main__':unittest.main()
