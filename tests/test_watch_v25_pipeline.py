"""CPU UNIT metadata fixtures, never actual CUDA training/completion results."""
from __future__ import annotations

import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import watch_v25_pipeline as watch
from tools import training_monitor_ui as ui

ROOT=Path(__file__).absolute().parents[1]


class PipelineMetadataUnitTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='UNIT_v25_viewer_',dir=ROOT/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.group=Path(self.temp.name)
        self.output=self.group/'A';self.monitor=self.output/'monitor'
        self.monitor.mkdir(parents=True)
        self.uid=os.getuid() if hasattr(os,'getuid') else None
        self.commands={stage:['python','-B','run_v25.py','--arm','A','--output',str(self.output),'--mode',stage]
            for stage in ('prepare_inputs','calibrate','train','scoring','native')}
        self.identity=dict(pid=12345,create_time=111.25,command=self.commands['train'],UID=self.uid)
        self.actors={12345:dict(self.identity,parent_pid=100,process_status='sleeping',RSS_bytes=3*2**30)}
        self.observer=mock.patch.object(watch,'actor_snapshot',side_effect=lambda pid:copy.deepcopy(self.actors.get(pid)))
        self.observer.start();self.addCleanup(self.observer.stop)
        self.log=self.monitor/'train.log';self.log.write_text('',encoding='utf8')
        self.request=dict(GPU=6,arm='A',production_output=str(self.output),RAM_GiB=64,epochs=40,
            stages=[dict(name=name,command=command) for name,command in self.commands.items()])
        self.state=dict(status='RUNNING',stage='train',arm='A',request=self.request,
            child_pid=12345,child_create_time=111.25,child_command=self.identity['command'],UID=self.uid,
            stage_log=str(self.log))
        self.write(self.monitor/'status.json',self.state)
        self.write(self.group/'status.json',dict(arm='A',output=str(self.output),status='RUNNING'))
        self.viewer=watch.Viewer(self.group)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf8')
        return watch.sha(path)

    def row(self,name,value):
        path=self.output/'training'/(name+'.jsonl');path.parent.mkdir(exist_ok=True)
        with path.open('a',encoding='utf8') as stream:stream.write(json.dumps(value)+'\n')

    def stage(self,name,status='RUNNING'):
        self.identity['command']=self.commands[name]
        self.actors[12345]['command']=self.commands[name]
        self.state.update(stage=name,status=status,child_command=self.commands[name])
        self.request['RAM_GiB']=96 if name=='native' else 64
        self.write(self.monitor/'status.json',self.state)

    def training_identity(self):
        self.write(self.output/'training/training_identity.json',dict(binding=dict(
            train_cases=['train'+str(n) for n in range(65)],val_cases=['val'+str(n) for n in range(21)],epochs=40)))

    def cp(self,*,count=4,status='RUNNING'):
        actor=dict(self.identity,command=self.commands['scoring'])
        self.actors[12345]['command']=actor['command']
        self.state.update(stage='scoring',child_command=actor['command'])
        self.write(self.monitor/'status.json',self.state)
        pin=dict(arm='A',completed_epochs=40,optimizer_updates=680,
            selected_by='fixed_full128_validation_only',full_training=True,debug=False)
        request=dict(format=watch.CP_REQUEST,best_provenance=pin)
        digest=self.write(self.output/'cp_bank/request.json',request)
        progress=dict(format=watch.CP_PROGRESS,arm='A',status=status,completed_sources=count,total_sources=642,
            completed_positions=count*128,total_positions=82176,request_sha256=digest,
            last_batch_seconds=2.7,updated_at=123.,actor=actor)
        self.write(self.output/'cp_bank/progress.json',progress)
        return progress,request

    def cp_overlay(self,request):
        overlay=dict(format=watch.CP_OVERLAY,complete=True,complete_sources=642,complete_positions=82176,
            debug=False,actual_CUDA=True,full_CP_scoring_executed=True,best_provenance=request['best_provenance'],
            durable_score_request_sha256=watch.sha(self.output/'cp_bank/request.json'))
        overlay['content_sha256']=watch.canonical(overlay)
        self.write(self.output/'cp_bank/score_overlay.json',overlay)
        return overlay

    def native(self,*,state='RUNNING',stage='train',with_child=True,console=''):
        self.stage('native')
        request=dict(format='v25_full_own_arm_downstream_request_v1',arm='A',output=str(self.output),
            debug=False,train_cases=105,validation_cases=26,source_entries=642,candidates_per_source=128,
            native_epochs=250,native_physical_batch=2,native_patch_size=[128]*3,cp_probability=.5,
            evaluation_checkpoint='checkpoint_best.pth',result_schema_version=1,
            process_and_children_RSS_bytes=96*2**30,shared_cache_memory_reserve_bytes=32*2**30)
        digest=self.write(self.output/'downstream/request.json',request)
        native_state=dict(format=watch.NATIVE_STATUS,arm='A',state=state,stage=stage,actor=copy.deepcopy(self.identity),
            request_sha256=digest,stage_log=str(self.output/'downstream/controller.log'),
            full_native_training_completed=True,full_native_evaluation_completed=True)
        self.write(self.output/'downstream/controller.log',{})
        self.write(self.output/'downstream/status.json',native_state)
        if with_child:
            log=self.output/'native/train.log';log.parent.mkdir(exist_ok=True);log.write_text(console,encoding='utf8')
            child=dict(format='v25_exact_owned_downstream_child_v1',arm='A',pid=23456,create_time=222.5,
                command=['python','-B','native_train.py','--root',str(self.output/'native')],UID=self.uid,
                parent_actor=copy.deepcopy(self.identity),state='RUNNING',request_sha256=digest,log_path=str(log))
            self.write(self.output/'downstream/native_child.json',child)
            self.actors[23456]=dict(child,parent_pid=12345,process_status='running',RSS_bytes=5*2**30)
        return request,native_state

    def native_certificate(self):
        """Explicit synthetic UNIT bytes: only metadata admission is exercised."""
        request,state=self.native(state='COMPLETE',stage='complete',with_child=False)
        cases=['UNITcase'+str(n).zfill(2) for n in range(26)]
        native=dict(format='v25_four_arm_Basic642_full250_native_v1',arm='A',root=str(self.output/'native'),
            split=dict(outer_val=cases),baseline=dict(dataset_name='Dataset730_UNIT'),trainer='UNITTrainer',bank_sha256='a'*64)
        self.write(self.output/'native/native.json',native)
        fold=self.output/'native/nnUNet_results/Dataset730_UNIT/UNITTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0'
        fold.mkdir(parents=True)
        (fold/'checkpoint_best.pth').write_bytes(b'explicit CPU UNIT BEST witness, no model')
        (fold/'checkpoint_final.pth').write_bytes(b'explicit CPU UNIT FINAL witness, no model')
        self.write(fold/'validation/summary.json',dict(UNIT_fixture=True))
        receipt=dict(format='v24_native_best_loaded_full26_validation_v1',checkpoint_name='checkpoint_best.pth',
            checkpoint_path=str(fold/'checkpoint_best.pth'),checkpoint_sha256=watch.sha(fold/'checkpoint_best.pth'),
            native_fold=str(fold),validation_cases=cases,bank_sha256='a'*64,complete=True,
            native_validation_returned_successfully=True,original_checkpoint_load_called=True,
            original_full_validation_called=True,final_checkpoint_fallback=False,GT_passed_to_predictor=False,
            CP_at_inference=False,predictions_sha256={case:'b'*64 for case in cases},
            summary_sha256=watch.sha(fold/'validation/summary.json'))
        digest=self.write(fold/'best_validation_complete.json',receipt)
        root=self.output/'downstream';metrics=root/'metrics'
        metrics_digest=self.write(metrics/'completion.json',dict(complete=True,UNIT_fixture=True))
        reports={}
        for name in ('evidence','text','json'):
            path=root/('result.'+name)
            if name=='json':self.write(path,dict(format='nnunet_completed_result_layout_v1',result_schema_version=1))
            else:path.write_text('explicit CPU UNIT report',encoding='utf8')
            reports[name]=dict(path=str(path),sha256=watch.sha(path))
        proof=dict(format='v25_full_own_arm_downstream_completion_v1',arm='A',
            request_sha256=watch.sha(root/'request.json'),debug=False,complete=True,actual_data_evaluation=True,
            full_native_training_completed=True,full_native_evaluation_completed=True,
            full_lesion_metric_evaluation_completed=True,result_schema1_published=True,
            native_checkpoint='checkpoint_best.pth',completed_epochs=250,evaluated_cases=26,
            native_best_receipt_sha256=digest,native_final_checkpoint=dict(path=str(fold/'checkpoint_final.pth'),
            sha256=watch.sha(fold/'checkpoint_final.pth')),metrics_directory=str(metrics),
            metrics_completion_sha256=metrics_digest,reports=reports)
        self.write(root/'completion.json',proof)
        return proof,receipt,fold

    def test_unknown_epoch_is_not_zero_or_complete_and_fixed_ui_rows(self):
        self.training_identity()
        value=self.viewer.snapshot()
        self.assertEqual(value['status'],'RUNNING')
        self.assertIsNone(value['gnn']['epoch']);self.assertIsNone(value['gnn']['completed_epochs'])
        self.assertIsNone(value['gnn']['optimizer_updates']);self.assertIsNone(value['gnn']['progress'])
        self.assertEqual(value['gnn']['epochs'],40);self.assertFalse(value['gnn']['full_training_completed'])
        text=watch.display(value)
        self.assertEqual([line.split(':',1)[0] for line in text.splitlines()],list(ui.GNN_ROW_LABELS))
        self.assertIn('버전 v2.5 | UI 2',text);self.assertIn('arm A',text)

    def test_actual_prepare_and_calibration_records(self):
        self.stage('prepare_inputs')
        self.log.write_text('v25 CPU canonical source: 256/14102 records\n',encoding='utf8')
        self.assertEqual(self.viewer.snapshot()['gnn']['progress'],dict(done=256,total=14102,unit='canonical 기록'))
        self.log.write_text('UPPER PREPARE completed=18/86 active_U=128\n',encoding='utf8')
        self.assertEqual(self.viewer.snapshot()['gnn']['progress']['done'],18)
        self.stage('calibrate')
        self.log.write_text('v25 CALIBRATION patients4/chunk16/allP+128U\nv25 CALIBRATION repeat2 12.3s\n',encoding='utf8')
        self.assertEqual(self.viewer.snapshot()['gnn']['calibration_progress'],dict(patients=4,chunk=16,repeats_completed=2,last_seconds=12.3))

    def test_actual_update_and_fixed_full_validation_reports(self):
        self.training_identity()
        self.row('update_timing',dict(status='OPTIMIZER_UPDATED',epoch=1,update=1,active_u=8,
            case_ids=['train0','train1','train2','train3'],loss=.4,RSS_bytes=7*2**30))
        self.write(self.output/'training/initial_full_validation_epoch_0.json',dict(epoch=0,active_u=128,updates=0,
            metrics=dict(per_P_patient_mrr=.25,per_P_patient_top1=.1)))
        value=self.viewer.snapshot()['gnn']
        self.assertEqual((value['epoch'],value['active_U'],value['optimizer_updates']),(1,8,1))
        self.assertEqual(value['progress'],dict(done=4,total=65,unit='환자'))
        self.assertEqual(value['RSS_GiB'],3);self.assertEqual(value['reports']['initial_full_validation']['active_u'],128)

    def test_full_gnn_completion_requires_real_full_receipt_not_flags(self):
        self.stage('train','COMPLETE');self.training_identity()
        self.row('invocations',dict(full_training=True,status='COMPLETE'))
        value=self.viewer.snapshot()
        self.assertFalse(value['gnn']['full_training_completed'])
        self.row('invocations',dict(format=watch.GNN_FORMAT,status='COMPLETE',full_training=True,debug=False,
            actual_CUDA=True,completed_epochs=40,optimizer_updates=680,connected_parameter_tensors=537,
            expected_parameter_tensors=537,best=dict(selected_by='fixed_full128_validation_only')))
        value=self.viewer.snapshot()
        self.assertEqual(value['status'],'TRAINING_COMPLETE')
        self.assertEqual((value['gnn']['epoch'],value['gnn']['completed_epochs'],value['gnn']['optimizer_updates']),(40,40,680))
        self.assertFalse(value['full_native_training'])

    def test_partial_metadata_is_visible_stale_then_error_and_recovers(self):
        self.training_identity();self.row('update_timing',dict(status='OPTIMIZER_UPDATED',epoch=3,update=35,active_u=24,case_ids=['t'],loss=.2))
        actual=self.viewer.snapshot();path=self.monitor/'status.json';path.write_text('{',encoding='utf8')
        with mock.patch.object(watch.time,'monotonic',return_value=100):pending=self.viewer.snapshot()
        with mock.patch.object(watch.time,'monotonic',return_value=111):error=self.viewer.snapshot()
        self.assertEqual(pending['status'],'READ_PENDING');self.assertEqual(error['status'],'READ_ERROR')
        self.assertFalse(error['snapshot_fresh']);self.assertTrue(error['last_valid_snapshot_preserved'])
        self.assertEqual(error['gnn']['optimizer_updates'],actual['gnn']['optimizer_updates'])
        self.assertIn('새 기록이 아님',watch.display(error));self.write(path,self.state)
        self.assertTrue(self.viewer.snapshot()['snapshot_fresh'])

    def test_exact_actor_binding_rejects_pid_reuse_wrong_argv_and_foreign_uid(self):
        for key,value in [('create_time',999.),('command',['python','another.py']),('UID',987654)]:
            with self.subTest(key=key):
                previous=self.actors[12345][key];self.actors[12345][key]=value
                result=self.viewer.snapshot();self.assertEqual(result['status'],'READ_ERROR');self.assertFalse(result['full_native_training'])
                self.actors[12345][key]=previous
        self.actors.pop(12345)
        self.assertEqual(self.viewer.snapshot()['status'],'EXITED_WITHOUT_COMPLETION')

    def test_request_own_arm_output_ram_gpu_constraints_rejected(self):
        for key,value in [('arm','B'),('production_output',str(self.group/'B')),('RAM_GiB',32),('GPU','6')]:
            with self.subTest(key=key):
                prior=self.request[key];self.request[key]=value;self.write(self.monitor/'status.json',self.state)
                self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR');self.request[key]=prior

    def test_durable_cp_committed_counts_sleeping_and_fixed_ui_rows(self):
        self.cp();value=self.viewer.snapshot();cp=value['scoring']
        self.assertEqual((cp['source_done'],cp['CP_done'],cp['source_total'],cp['CP_total']),(4,512,642,82176))
        self.assertEqual(cp['status'],'RUNNING');self.assertEqual(cp['process_status'],'sleeping')
        self.assertEqual([line.split(':',1)[0] for line in watch.display(value).splitlines()],list(ui.CP_ROW_LABELS))
        self.assertIn('arm A | GPU6 GNN BEST',watch.display(value))
        value=self.viewer.scoring(self.output,'A',list(self.commands.values()))
        self.assertIsNone(value['seconds_since_console_write'])

    def test_cp_completion_needs_complete_canonical_overlay_and_own_request(self):
        progress,request=self.cp(count=642,status='COMPLETE')
        self.assertEqual(self.viewer.snapshot()['scoring']['status'],'SCORING_COMPLETION_UNVERIFIED')
        overlay=self.cp_overlay(request)
        self.assertEqual(self.viewer.snapshot()['scoring']['status'],'SCORING_COMPLETE')
        overlay['complete_sources']=641;self.write(self.output/'cp_bank/score_overlay.json',overlay)
        self.assertFalse(self.viewer.snapshot()['scoring']['complete'])
        request['best_provenance']['arm']='B';self.write(self.output/'cp_bank/request.json',request)
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')

    def test_cp_partial_progress_preserves_last_count_and_missing_is_unknown(self):
        self.stage('scoring');value=self.viewer.snapshot();self.assertIsNone(value['scoring'])
        self.cp(count=8);self.viewer.snapshot()
        (self.output/'cp_bank/progress.json').write_text('',encoding='utf8')
        result=self.viewer.snapshot();self.assertEqual(result['status'],'READ_PENDING')
        self.assertEqual(result['scoring']['source_done'],8);self.assertFalse(result['snapshot_fresh'])

    def test_native_unknown_epoch_then_original_console_mapping_is_preserved(self):
        self.native(console='using pin_memory on device 0\n')
        self.assertIsNone(self.viewer.snapshot()['native']['epoch'])
        self.assertEqual(self.viewer.snapshot()['native']['RAM_limit_GiB'],96)
        console=('2026-10-12 01:28:56.248687: Epoch 0\n'
            '2026-10-12 01:28:56.248969: Current learning rate: 0.01\n'
            '2026-10-12 01:30:11.250355: [OnlineCP] epoch=0 applied=207/500 rate=0.4140 schedule=abc\n'
            '2026-10-12 01:30:15.627689: train_loss -0.7644\n'
            '2026-10-12 01:30:15.628042: val_loss -0.7228\n'
            '2026-10-12 01:30:15.628116: Pseudo dice [np.float32(0.9603), np.float32(0.7794)]\n'
            '2026-10-12 01:30:15.628209: Epoch time: 79.97 s\n')
        (self.output/'native/train.log').write_text(console,encoding='utf8')
        result=self.viewer.snapshot();native=result['native']
        self.assertEqual((native['epoch'],native['completed_epochs']),(1,1));self.assertEqual(native['native_metrics']['pseudo_dice'],[.9603,.7794])
        self.assertEqual(watch.display(result),console.rstrip('\n'));self.assertFalse(native['full_native_training'])

    def test_native_child_exact_parent_pid_and_request_binding_rejected(self):
        self.native(console='Epoch 0\n')
        self.actors[23456]['parent_pid']=4321
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')
        self.actors[23456]['parent_pid']=12345
        path=self.output/'downstream/native_child.json';child=json.loads(path.read_text())
        child['request_sha256']='f'*64;self.write(path,child)
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')

    def test_owned_metrics_child_log_is_admitted_without_native_epoch_claim(self):
        _,state=self.native(stage='metrics',console='2026-10-12 01:00:00: Epoch 249\n')
        child_path=self.output/'downstream/native_child.json';child=json.loads(child_path.read_text())
        log=self.output/'downstream/metric_123456789.log'
        log.write_text('SIX_METRIC_COMPARISON [{"UNIT_fixture": true}]\n',encoding='utf8')
        child.update(command=['python','-B','-u','evaluate_v24_historical_metrics.py',
            '--output',str(self.output/'downstream/metrics_123456789')],log_path=str(log))
        self.write(child_path,child);self.actors[23456]['command']=child['command']
        value=self.viewer.snapshot();self.assertEqual(value['status'],'RUNNING')
        self.assertEqual(value['native']['console_lines'],['SIX_METRIC_COMPARISON [{"UNIT_fixture": true}]'])
        self.assertIsNone(value['native']['epoch']);self.assertIsNone(value['native']['native_metrics'])
        self.assertFalse(value['native']['native_training_started'])
        text=watch.display(value);self.assertIn('nnU-Net 후속 단계: metrics',text)
        self.assertNotIn('SIX_METRIC_COMPARISON',text);self.assertNotIn('Epoch 249',text)
        state['stage']='report';child['state']='EXITED';self.actors.pop(23456)
        self.write(child_path,child);self.write(self.output/'downstream/status.json',state)
        value=self.viewer.snapshot();self.assertEqual(value['status'],'RUNNING')
        self.assertIsNone(value['native']['epoch']);self.assertIn('report',watch.display(value))

    def test_metric_log_admission_does_not_allow_unrelated_downstream_or_train_log(self):
        _,state=self.native(stage='metrics')
        path=self.output/'downstream/native_child.json';child=json.loads(path.read_text())
        child['log_path']=str(self.output/'downstream/arbitrary.log');self.write(path,child)
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')
        child['log_path']=str(self.output/'downstream/metric_123.log');self.write(path,child)
        state['stage']='train';self.write(self.output/'downstream/status.json',state)
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')

    def test_native_complete_flags_without_certificate_do_not_mean_complete(self):
        self.native(state='COMPLETE',stage='complete',with_child=False)
        value=self.viewer.snapshot();self.assertEqual(value['status'],'COMPLETION_UNVERIFIED')
        self.assertFalse(value['full_native_training']);self.assertFalse(value['full_native_evaluation'])

    def test_native_failed_state_remains_visible_beside_original_epoch_console(self):
        _,state=self.native(state='FAILED',console='2026-10-12 01:28:56: Epoch 7\n')
        state['error']='ImportError: actual fixture failure';self.write(self.output/'downstream/status.json',state)
        value=self.viewer.snapshot();self.assertEqual(value['status'],'FAILED')
        text=watch.display(value);self.assertIn('nnU-Net controller: FAILED',text)
        self.assertIn('ImportError: actual fixture failure',text);self.assertIn('2026-10-12 01:28:56: Epoch 7',text)

    def test_complete_certificate_checks_all_report_and_native_byte_witnesses(self):
        proof,receipt,fold=self.native_certificate()
        value=self.viewer.snapshot();self.assertEqual(value['status'],'COMPLETE')
        self.assertTrue(value['full_native_training']);self.assertTrue(value['full_native_evaluation'])
        self.assertEqual((value['native']['epoch'],value['native']['completed_epochs']),(250,250))
        self.assertIn('no model deserialization',value['native']['completion_verification_scope'])
        (fold/'checkpoint_best.pth').write_bytes(b'changed witness')
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')

    def test_complete_certificate_final_cannot_replace_best_or_change26_cohort(self):
        proof,receipt,fold=self.native_certificate()
        path=fold/'best_validation_complete.json';receipt['checkpoint_name']='checkpoint_final.pth'
        proof['native_best_receipt_sha256']=self.write(path,receipt)
        self.write(self.output/'downstream/completion.json',proof)
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')
        receipt['checkpoint_name']='checkpoint_best.pth';receipt['validation_cases']=receipt['validation_cases'][:-1]
        proof['native_best_receipt_sha256']=self.write(path,receipt);self.write(self.output/'downstream/completion.json',proof)
        self.assertEqual(self.viewer.snapshot()['status'],'READ_ERROR')

    def test_missing_new_arm_record_does_not_reuse_old_arm_progress(self):
        self.cp(count=8);self.viewer.snapshot()
        self.write(self.group/'status.json',dict(arm='B',output=str(self.group/'B'),status='RUNNING'))
        value=self.viewer.snapshot();self.assertEqual(value['status'],'READ_PENDING')
        self.assertFalse(value['last_valid_snapshot_preserved']);self.assertIsNone(value['scoring'])

    def test_snapshots_do_not_write_artifacts_or_import_scientific_modules(self):
        self.native_certificate()
        before={str(path):path.read_bytes() for path in self.group.rglob('*') if path.is_file()}
        with mock.patch('subprocess.Popen',side_effect=AssertionError('observer must never start children')):
            self.viewer.snapshot();self.viewer.snapshot()
        self.assertEqual(before,{str(path):path.read_bytes() for path in self.group.rglob('*') if path.is_file()})
        result=subprocess.run([sys.executable,'-B','-c',
            'import sys; from tools import watch_v25_pipeline; assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        tree=ast.parse(Path(watch.__file__).read_text(encoding='utf8'))
        imported=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any(name and name.startswith('hiercp_v1x') for name in imported))

    def test_cli_once_json_declares_ui2_without_native_completion_claim(self):
        output=io.StringIO()
        with mock.patch.object(sys,'stdout',output):
            watch.main(['--group',str(self.group),'--arm','A','--once','--json'])
        value=json.loads(output.getvalue());self.assertEqual(value['ui_schema_version'],2)
        self.assertEqual(value['experiment_version'],'v2.5');self.assertIsNone(value['gnn']['epoch'])
        self.assertFalse(value['full_native_evaluation'])


if __name__=='__main__':unittest.main()
