"""CPU DEBUG metadata/OS fixtures for the read-only native startup retry UI."""
import ast
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import watch_v24_native_startup_retry as viewer
from tools import watch_v24_training as renderer
from tests.test_watch_v24_matched_cp_scoring import event, scoring_fixture, write_json


class NativeStartupRetryViewerCpuDebugTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='startup_retry_viewer_CPU_DEBUG_',dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.path=self.root/'retry_DEBUG/request.json'
        self.old_path=self.root/'original_DEBUG/request.json'
        self.source=self.root/'original_DEBUG/gpu6/gnn'
        self.code=self.root/'code_DEBUG'
        self.digest='a'*64;self.old_digest='b'*64;self.stat=[1,2,3,4,5]
        self.request=dict(format=viewer.REQUEST,GPU_arm=6,recovery_root=str(self.path.parent),
            original_request=dict(path=str(self.old_path),stat=self.stat,sha256=self.old_digest),
            python='/DEBUG/nnunet/bin/python3.10',code=str(self.code))
        self.original=dict(GPU_arm=6,chain_root=str(self.old_path.parent),source_output=str(self.source),
            source_code='/DEBUG/frozen_GNN_code',scoring_root=str(self.root/'scoring_DEBUG'),
            epochs=250,physical_batch=2,train_cases=105,validation_cases=26,
            evaluation_checkpoint='checkpoint_best.pth')
        self.command=[self.request['python'],'-B','-u',str(self.code/'tools/run_v24_native_startup_retry.py'),
            'watch','--request',str(self.path)]
        self.entry=[self.request['python'],'-B','-u',str(self.code/'tools/run_v24_native_startup_retry.py'),
            'entry','--request',str(self.path),'--','730','3d_fullres','0','-tr',
            'nnUNetTrainer_250epochs_FrozenV23CP','-p','nnUNetResEncUNetMPlans','--val_best']
        self.registration=dict(request_sha256=self.digest,command=self.command,UID=41833,pid=999,create_time=1.)
        self.lease=copy.deepcopy(self.registration)
        self.state=dict(format=viewer.STATUS,request_sha256=self.digest,GPU_arm=6,
            source_output=str(self.source),supervisor_pid=999,supervisor_create_time=1.,
            status='RUNNING',stage='train',command=self.entry,child_pid=123,child_create_time=2.,
            stage_log=str(self.path.parent/'train.log'),child_process_tree_RSS_bytes=5*2**30)
        self.base=dict(GPU=6,snapshot_fresh=True,status='FAILED',active_stage='native',
            gnn=dict(job_root=str(self.source),full_training_completed=True,epoch=40,epochs=40,completed_epochs=40),
            scoring=dict(source_done=642,source_total=642,CP_done=82176,CP_total=82176),
            native=dict(status='FAILED',native_training_started=False),native_pending_reason='DEBUG old ImportError',
            full_native_training=False,full_native_evaluation=False)
        self.records={str(self.path):self.request,str(self.old_path):self.original,
            str(self.path.parent/'registration.json'):self.registration,
            str(self.path.parent/'lease.json'):self.lease,str(self.path.parent/'status.json'):self.state}
        self.os_processes={999:dict(birth=1.,command=self.command,UID=41833),
            123:dict(birth=2.,command=self.entry,UID=41833)}
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(viewer.pipeline,'read',side_effect=lambda path,**_:self.records[str(path)]))
        self.stack.enter_context(patch.object(viewer.pipeline,'input_path',side_effect=Path))
        self.stat_reader=self.stack.enter_context(patch.object(viewer.pipeline,'_file_stat',return_value=self.stat))
        self.digest_reader=self.stack.enter_context(patch.object(viewer.direct,'sha',
            side_effect=lambda path:self.old_digest if Path(path)==self.old_path else self.digest))
        self.process_reader=self.stack.enter_context(patch.object(viewer.scoring.psutil,'Process',side_effect=self.process))
        self.stack.enter_context(patch.object(viewer.scoring.os,'getuid',return_value=41833,create=True))
        self.log=Mock(return_value=[])
        self.renderer=SimpleNamespace(log_tail=self.log,parse_nnunet_lines=renderer.parse_nnunet_lines,
            JsonRecordError=renderer.JsonRecordError)

    def process(self,pid):
        row=self.os_processes.get(pid)
        if row is None:raise viewer.scoring.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda:row['birth'],cmdline=lambda:row['command'],
            uids=lambda:SimpleNamespace(real=row['UID']),status=lambda:'sleeping',
            memory_info=lambda:SimpleNamespace(rss=2*2**30))

    def snapshot(self):
        return viewer.retry_snapshot(copy.deepcopy(self.base),self.renderer,self.path)

    def completion_fixture(self):
        self.os_processes.clear()
        self.state.update(status='FAILED',error='DEBUG original post-training proof import failure')
        self.request['original_declaration']=copy.deepcopy(self.original)
        self.completion_path=self.old_path.parent/'completion_DEBUG/request.json'
        completion_proof=lambda path:dict(path=str(path),stat=self.stat,sha256=self.digest)
        names=dict(failed_retry_request=self.path,status=self.path.parent/'status.json',
            registration=self.path.parent/'registration.json',lease=self.path.parent/'lease.json',
            global_retry_lease=self.old_path.parent/'native_startup_retry.lease.json',
            entry_lease=self.path.parent/'entry.lease.json',training_started=self.path.parent/'training_started.json',
            train_log=self.path.parent/'train.log',actual_configuration=self.old_path.parent/'native/actual_nnunet_configuration.json',
            best_receipt=self.old_path.parent/'native/fold_0/best_validation_complete.json',
            best_checkpoint=self.old_path.parent/'native/fold_0/checkpoint_best.pth',
            final_checkpoint=self.old_path.parent/'native/fold_0/checkpoint_final.pth')
        self.completion_request=dict(format=viewer.COMPLETION_REQUEST,GPU_arm=6,
            recovery_root=str(self.completion_path.parent),python=self.request['python'],
            code=str(self.root/'completion_code_DEBUG'),commit='c'*40,code_files={'DEBUG.py':'d'*64},
            failed_retry_declaration=copy.deepcopy(self.request),failed_retry_request=completion_proof(self.path),
            proofs={key:completion_proof(path) for key,path in names.items()},training_rerun=False,
            evaluation_rerun=False,CUDA_compute=False,epochs=250,physical_batch=2,evaluation_checkpoint='checkpoint_best.pth')
        self.completion_reg=dict(pid=998,create_time=3.,command=[self.request['python'],'-B','-u',
            str(self.root/'completion_code_DEBUG/complete_DEBUG.py'),'--request',str(self.completion_path)],
            UID=41833,request_sha256=self.digest)
        self.completion_lease=dict(self.completion_reg,format=viewer.COMPLETION_REQUEST,
            recovery_root=str(self.completion_path.parent),training_rerun=False,evaluation_rerun=False,CUDA_compute=False)
        self.completion_state=dict(format=viewer.COMPLETION_STATUS,GPU_arm=6,request_sha256=self.digest,
            source_output=str(self.source),source_code=self.original['source_code'],supervisor_pid=998,
            supervisor_create_time=3.,UID=41833,status='COMPLETE',stage=None,child_pid=None,child_create_time=None,
            completed_epochs=250,evaluated_cases=26,full_native_training_completed=True,
            full_native_evaluation_completed=True,training_rerun=False,evaluation_rerun=False,CUDA_compute=False)
        self.records.update({str(self.completion_path):self.completion_request,
            str(self.completion_path.parent/'registration.json'):self.completion_reg,
            str(self.completion_path.parent/'lease.json'):self.completion_lease,
            str(self.completion_path.parent/'status.json'):self.completion_state})
        self.os_processes[998]=dict(birth=3.,command=self.completion_reg['command'],UID=41833)

    def completion_snapshot(self):
        return viewer.completion_snapshot(self.snapshot(),self.path,self.completion_path)

    def test_genuine_running_requires_exact_current_controller_child_and_preserves_whole_arm(self):
        result=self.snapshot()
        self.assertEqual(result['status'],'RUNNING')
        self.assertEqual(result['active_stage'],'native')
        self.assertTrue(result['native']['controller_alive'])
        self.assertTrue(result['native']['child_alive'])
        self.assertTrue(result['native']['native_training_started'])
        self.assertEqual(result['native']['RSS_GiB'],5)
        self.assertEqual(result['native']['RAM_limit_GiB'],48)
        self.assertEqual(result['native']['epochs'],250)
        self.assertEqual(result['gnn'],self.base['gnn'])
        self.assertEqual(result['scoring'],self.base['scoring'])
        self.assertFalse(result['full_native_training']);self.assertFalse(result['full_native_evaluation'])
        self.assertIsNone(result['native'].get('epoch'))
        self.assertIsNone(result['native'].get('completed_epochs'))
        self.assertIsNone(result['native'].get('native_metrics'))
        self.assertEqual([call.args[0] for call in self.process_reader.call_args_list],[999,123])
        self.log.assert_called_once_with(self.path.parent/'train.log',65536)

    def test_actual_native_log_parser_maps_partial_and_completed_epochs_without_final_dice_claim(self):
        self.log.return_value=['2026-10-12: Epoch 0','2026-10-12: Current learning rate: 0.01']
        first=self.snapshot()['native']
        self.assertEqual((first['epoch'],first['completed_epochs']),(1,0))
        self.log.return_value=['2026-10-12: Epoch 7','2026-10-12: train_loss -0.7644',
            '2026-10-12: val_loss -0.7228',
            '2026-10-12: Pseudo dice [np.float32(0.9603), np.float32(0.7794)]',
            '2026-10-12: Epoch time: 79.97 s']
        result=self.snapshot()
        native=result['native']
        self.assertEqual((native['epoch'],native['completed_epochs']),(8,8))
        self.assertEqual(native['native_metrics']['train_loss'],-.7644)
        self.assertEqual(native['native_metrics']['val_loss'],-.7228)
        self.assertEqual(native['native_metrics']['pseudo_dice'],[.9603,.7794])
        self.assertFalse(result['full_native_training']);self.assertFalse(result['full_native_evaluation'])

    def test_old_failure_is_superseded_only_by_bound_fresh_retry_and_stale_input_is_untouched(self):
        self.base['read_problem'] = dict(path='/DEBUG/old_native/train.log', detail='DEBUG old ImportError')
        result=self.snapshot()
        self.assertEqual(result['status'],'RUNNING');self.assertIsNone(result['native_pending_reason'])
        self.assertNotIn('read_problem',result)
        self.assertEqual(self.base['status'],'FAILED')
        stale=copy.deepcopy(self.base);stale['snapshot_fresh']=False
        before_reads=self.digest_reader.call_count
        returned=viewer.retry_snapshot(stale,self.renderer,self.path)
        self.assertIs(returned,stale);self.assertEqual(returned['status'],'FAILED')
        self.assertEqual(self.digest_reader.call_count,before_reads)
        self.request['original_request']['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'immutable witness'):self.snapshot()

    def test_foreign_GPU_scale_BEST_or_source_metadata_cannot_replace_original_arm(self):
        mutations=[(self.request,'GPU_arm',5),(self.request,'format','foreign_retry'),
            (self.request,'recovery_root','/DEBUG/foreign_root'),(self.original,'GPU_arm',4),
            (self.original,'epochs',249),(self.original,'physical_batch',1),
            (self.original,'train_cases',104),(self.original,'validation_cases',25),
            (self.original,'evaluation_checkpoint','checkpoint_final.pth'),
            (self.base,'GPU',5),(self.base['gnn'],'full_training_completed',False),
            (self.base['gnn'],'job_root','/DEBUG/another_GNN'),
            (self.base['scoring'],'source_done',641),(self.base['scoring'],'CP_done',82048)]
        for record,key,value in mutations:
            with self.subTest(key=key,value=value):
                previous=record[key];record[key]=value
                try:
                    with self.assertRaisesRegex(ValueError,'retry binding'):self.snapshot()
                finally:record[key]=previous

    def test_forged_controller_registration_lease_and_status_bindings_are_rejected(self):
        mutations=[(self.registration,'request_sha256','c'*64),(self.registration,'UID',0),
            (self.registration,'command',['DEBUG foreign command']),(self.lease,'command',['DEBUG foreign command']),
            (self.lease,'request_sha256','c'*64),(self.lease,'UID',0),(self.lease,'pid',998),
            (self.lease,'create_time',9.),(self.state,'supervisor_pid',998),
            (self.state,'supervisor_create_time',9.),(self.state,'request_sha256','c'*64),
            (self.state,'format','foreign_status'),(self.state,'GPU_arm',5)]
        for record,key,value in mutations:
            with self.subTest(key=key,value=value):
                previous=record[key];record[key]=value
                try:
                    with self.assertRaisesRegex(ValueError,'controller/lease'):self.snapshot()
                finally:record[key]=previous

    def test_child_argv_birth_UID_and_log_substitution_are_rejected(self):
        for key,value in [('command',['DEBUG foreign child']),('birth',9.),('UID',0)]:
            with self.subTest(key=key):
                previous=self.os_processes[123][key];self.os_processes[123][key]=value
                try:
                    with self.assertRaisesRegex(ValueError,'PID/birth/full argv/UID'):self.snapshot()
                finally:self.os_processes[123][key]=previous
        self.state['command']=['DEBUG foreign child']
        with self.assertRaisesRegex(ValueError,'child command'):self.snapshot()
        self.state['command']=self.entry
        self.state['stage_log']='/DEBUG/another_train.log'
        with self.assertRaisesRegex(ValueError,'log binding'):self.snapshot()

    def test_dead_controller_or_child_is_never_training_and_PID_reuse_is_rejected(self):
        controller=self.os_processes.pop(999)
        result=self.snapshot()
        self.assertEqual(result['status'],'CONTROLLER_EXITED_CHILD_ALIVE')
        self.assertFalse(result['native']['native_training_started']);self.log.assert_not_called()
        self.os_processes[999]=controller
        self.os_processes.pop(123)
        result=self.snapshot();self.assertFalse(result['native']['native_training_started'])
        self.os_processes[123]=dict(birth=9.,command=['DEBUG foreign PID reuse'],UID=41833)
        with self.assertRaisesRegex(ValueError,'PID/birth/full argv/UID'):self.snapshot()

    def test_completion_flags_alone_require_actual_native_completion_evidence(self):
        self.state.update(status='COMPLETE',full_native_training_completed=True,full_native_evaluation_completed=True)
        with patch.object(viewer.pipeline,'native_completion',side_effect=ValueError('DEBUG actual BEST26 evidence missing')) as complete:
            with self.assertRaisesRegex(ValueError,'BEST26 evidence missing'):self.snapshot()
        complete.assert_called_once_with(self.old_path.parent,self.original,self.state,
            self.source,6,self.original['source_code'])

    def test_missing_completion_flags_cannot_call_or_bypass_native_completion(self):
        self.state.update(status='COMPLETE',full_native_training_completed=True,full_native_evaluation_completed=False)
        with patch.object(viewer.pipeline,'native_completion') as complete:
            with self.assertRaisesRegex(ValueError,'Full250/BEST26'):self.snapshot()
        complete.assert_not_called()

    def test_verified_actual_native_completion_sets_flags_only_after_evidence_admission(self):
        self.state.update(status='COMPLETE',full_native_training_completed=True,full_native_evaluation_completed=True)
        with patch.object(viewer.pipeline,'native_completion',return_value=dict(full_native_training=True,
                full_native_evaluation=True,completed_epochs=250,BEST_evaluated_cases=26)) as complete:
            result=self.snapshot()
        complete.assert_called_once()
        self.assertTrue(result['full_native_training']);self.assertTrue(result['full_native_evaluation'])
        self.assertEqual(result['native']['completed_epochs'],250)
        self.assertEqual(result['native']['BEST_evaluated_cases'],26)

    def test_actual_retry_failure_is_visible_and_cannot_claim_training_from_live_PID(self):
        self.state.update(status='FAILED',error='DEBUG actual retry failure: CP metadata digest')
        self.base['read_problem'] = dict(path='/DEBUG/old_native/train.log',detail='DEBUG old ImportError')
        result=self.snapshot()
        self.assertEqual(result['status'],'FAILED');self.assertEqual(result['active_stage'],'native')
        self.assertIn('CP metadata digest',result['native_pending_reason'])
        self.assertIn('CP metadata digest',result['native']['native_pending_reason'])
        self.assertFalse(result['native']['native_training_started'])
        self.log.assert_called_once_with(self.path.parent/'train.log',65536)
        self.assertEqual(result['read_problem']['path'],str(self.path.parent/'status.json'))
        shown=viewer.pipeline.display(result)
        self.assertIn('CP metadata digest',shown);self.assertIn('실패 단계 train',shown)
        self.assertNotIn('old ImportError',shown)
        self.assertFalse(result['full_native_training']);self.assertFalse(result['full_native_evaluation'])

    def test_failed_post_training_projection_keeps_observed_250_separate_from_verified_completion(self):
        self.state.update(status='FAILED',error='ValueError: Scientific module was already imported from a different checkout')
        self.os_processes.clear()
        self.log.return_value=['2026-10-12 11:24:20: Epoch 249',
            '2026-10-12 11:25:45: train_loss -0.7771',
            '2026-10-12 11:25:45: val_loss -0.6806',
            '2026-10-12 11:25:45: Pseudo dice [0.9654, 0.8528]',
            '2026-10-12 11:25:45: Epoch time: 86.15 s',
            '2026-10-12 11:25:46: Training done.',
            '2026-10-12 12:23:17: Validation complete',
            '2026-10-12 12:23:17: Mean Validation Dice: 0.8283640724339547']
        with patch.object(viewer.pipeline,'native_completion') as completion:
            result=self.snapshot()
        completion.assert_not_called()
        native=result['native']
        self.assertEqual((native['epoch'],native['completed_epochs']),(250,250))
        self.assertEqual(native['native_log_observation_source'],str(self.path.parent/'train.log'))
        self.assertFalse(native['native_log_observation_is_completion_proof'])
        self.assertEqual(native['native_metrics']['pseudo_dice'],[.9654,.8528])
        self.assertNotIn('RSS_GiB',native)  # A failed actor's last RSS is not current.
        self.assertFalse(native['native_training_started'])
        self.assertFalse(native['controller_alive']);self.assertFalse(native['child_alive'])
        for record in (native,result):
            self.assertFalse(record['full_native_training']);self.assertFalse(record['full_native_evaluation'])
        shown=viewer.pipeline.display(result)
        self.assertIn('FAILED',shown);self.assertIn('250/250',shown)
        self.assertIn('완료 증거 미확인',shown);self.assertIn('different checkout',shown)

    def test_failed_retry_still_rejects_foreign_log_or_lease_before_displaying_console_evidence(self):
        self.state.update(status='FAILED',error='DEBUG post-training validation failure')
        self.state['stage_log']='/DEBUG/old_native/train.log'
        with self.assertRaisesRegex(ValueError,'log binding'):self.snapshot()
        self.log.assert_not_called()
        self.state['stage_log']=str(self.path.parent/'train.log')
        self.lease['request_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'controller/lease'):self.snapshot()
        self.log.assert_not_called()

    def test_explicit_CPU_completion_requires_canonical_full250_BEST26_before_true_flags(self):
        self.completion_fixture()
        before=copy.deepcopy(self.state)
        with patch.object(viewer.pipeline,'native_completion',return_value=dict(full_native_training=True,
                full_native_evaluation=True,completion_metadata_verified=True)) as canonical:
            result=self.completion_snapshot()
        canonical.assert_called_once_with(self.old_path.parent,self.original,self.completion_state,
            self.source,6,self.original['source_code'])
        self.assertEqual(result['status'],'COMPLETE')
        self.assertTrue(result['full_native_training']);self.assertTrue(result['full_native_evaluation'])
        self.assertFalse(result['native']['native_training_started'])
        self.assertEqual(result['native']['BEST_evaluated_cases'],26)
        self.assertNotIn('read_problem',result)
        self.assertIn('재학습·재평가 없음',viewer.pipeline.display(result))
        self.assertEqual(self.state,before);self.assertEqual(self.state['status'],'FAILED')
        paths=[Path(call.args[0]) for call in self.digest_reader.call_args_list]
        self.assertNotIn(Path(self.completion_request['proofs']['best_checkpoint']['path']),paths)
        self.assertNotIn(Path(self.completion_request['proofs']['final_checkpoint']['path']),paths)

    def test_missing_canonical_BEST26_or_false_completion_flags_cannot_complete(self):
        self.completion_fixture()
        with patch.object(viewer.pipeline,'native_completion',side_effect=ValueError('DEBUG actual BEST26 evidence missing')):
            with self.assertRaisesRegex(ValueError,'BEST26 evidence missing'):self.completion_snapshot()
        for key,value in [('completed_epochs',249),('evaluated_cases',25),
                ('full_native_training_completed',False),('full_native_evaluation_completed',False)]:
            with self.subTest(key=key):
                previous=self.completion_state[key];self.completion_state[key]=value
                try:
                    with patch.object(viewer.pipeline,'native_completion') as canonical:
                        with self.assertRaisesRegex(ValueError,'full250/BEST26'):self.completion_snapshot()
                        canonical.assert_not_called()
                finally:self.completion_state[key]=previous

    def test_completion_foreign_failed_request_GPU_rerun_and_actor_lease_are_rejected(self):
        self.completion_fixture()
        mutations=[(self.completion_request,'GPU_arm',5),(self.completion_request,'training_rerun',True),
            (self.completion_request,'CUDA_compute',True),(self.completion_request,'failed_retry_declaration',{}),
            (self.completion_reg,'request_sha256','c'*64),(self.completion_lease,'command',['DEBUG foreign actor']),
            (self.completion_lease,'recovery_root','/DEBUG/foreign_root'),(self.completion_state,'source_code','/DEBUG/foreign_source'),
            (self.completion_state,'child_pid',123),(self.completion_state,'UID',0)]
        for record,key,value in mutations:
            with self.subTest(key=key):
                previous=record[key];record[key]=value
                try:
                    with patch.object(viewer.pipeline,'native_completion') as canonical:
                        with self.assertRaises(ValueError):self.completion_snapshot()
                        canonical.assert_not_called()
                finally:record[key]=previous
        self.completion_request['failed_retry_request']['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'immutable witness'):self.completion_snapshot()

    def test_CPU_completion_failure_and_dead_finalizer_show_actual_new_error_without_running_claim(self):
        self.completion_fixture()
        self.completion_state.update(status='FAILED',stage='train',full_native_training_completed=False,
            full_native_evaluation_completed=False,error='DEBUG new completion proof mismatch')
        result=self.completion_snapshot()
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['read_problem']['path'],str(self.completion_path.parent/'status.json'))
        self.assertIn('new completion proof mismatch',viewer.pipeline.display(result))
        self.assertNotIn('original post-training proof import failure',viewer.pipeline.display(result))
        self.assertFalse(result['full_native_training']);self.assertFalse(result['native']['native_training_started'])
        self.completion_state.update(status='FINALIZING_EXISTING_ARTIFACTS',error=None)
        live=self.completion_snapshot()
        self.assertEqual(live['status'],'FINALIZING_EXISTING_ARTIFACTS')
        self.assertFalse(live['full_native_training']);self.assertNotIn('read_problem',live)
        self.os_processes.pop(998)
        dead=self.completion_snapshot()
        self.assertEqual(dead['status'],'EXITED_WITHOUT_COMPLETION')
        self.assertFalse(dead['full_native_training']);self.assertFalse(dead['full_native_evaluation'])

    def test_explicit_completion_actor_PID_reuse_and_live_old_retry_are_rejected(self):
        self.completion_fixture()
        self.os_processes[998]['birth']=9.
        with self.assertRaisesRegex(ValueError,'PID/birth/full argv/UID'):self.completion_snapshot()
        self.os_processes[998]['birth']=3.
        self.os_processes[999]=dict(birth=1.,command=self.command,UID=41833)
        with self.assertRaisesRegex(ValueError,'CPU completion binding'):self.completion_snapshot()

    def test_RetryViewer_uses_only_explicit_completion_path_and_stale_source_never_calls_overlay(self):
        self.completion_fixture()
        instance=viewer.RetryViewer.__new__(viewer.RetryViewer)
        instance.renderer=self.renderer;instance.retry_path=self.path;instance.completion_path=self.completion_path
        with patch.object(viewer.pipeline.PipelineViewer,'_snapshot',return_value=copy.deepcopy(self.base)), \
                patch.object(viewer.pipeline,'native_completion',return_value=dict(full_native_training=True,
                    full_native_evaluation=True,completion_metadata_verified=True)):
            result=instance._snapshot()
        self.assertEqual(result['native']['completion_root'],str(self.completion_path.parent))
        stale=copy.deepcopy(self.base);stale['snapshot_fresh']=False
        count=self.digest_reader.call_count
        self.assertIs(viewer.completion_snapshot(stale,self.path,self.completion_path),stale)
        self.assertEqual(self.digest_reader.call_count,count)

    def test_malformed_explicit_completion_declaration_becomes_clear_read_error(self):
        self.completion_fixture()
        instance=viewer.RetryViewer.__new__(viewer.RetryViewer)
        instance.renderer=self.renderer;instance.retry_path=self.path;instance.completion_path=self.completion_path
        instance.last_valid=None;instance.problem=None;instance.pending_reason='DEBUG completion metadata wait'
        self.records[str(self.completion_path)]={}
        with patch.object(viewer.pipeline.PipelineViewer,'_snapshot',return_value=copy.deepcopy(self.base)):
            result=instance.snapshot()
        self.assertEqual(result['status'],'READ_ERROR')
        self.assertIn(str(self.completion_path),result['read_problem']['detail'])
        self.assertFalse(result['full_native_training']);self.assertFalse(result['snapshot_fresh'])

    def test_real_completed_GPU6_source_and_fresh_retry_failure_never_read_old_native_status(self):
        # Actual JSON/SHA readers, CompletedGNNSource and PipelineViewer routes;
        # only ownership portability and the OS process API use DEBUG fixtures.
        self.stack.close()
        self.path.parent.mkdir();self.old_path.parent.mkdir()
        (self.source/'training').mkdir(parents=True)
        invocation=self.source/'training/invocations.jsonl'
        invocation.write_text(json.dumps(dict(status='COMPLETE',full_training=True,debug=False,
            completed_epochs=40,optimizer_updates=680,actual_CUDA=True,
            connected_parameter_tensors=537,expected_parameter_tensors=537))+'\n',encoding='utf8')
        score=Path(self.original['scoring_root'])
        score_request,score_registration=scoring_fixture(score,6)
        score_request.pop('source_output');score_request.pop('source_code')
        write_json(score/'request.json',score_request)
        score_registration['request_sha256']=viewer.direct.sha(score/'request.json')
        write_json(score/'registration.json',score_registration)
        write_json(score/'own_best_pin.json',dict(format='v24_completed_single_arm_GT_blind_best_v1',
            physical_GPU=6,source_output=str(self.source),source_code=self.original['source_code'],
            full_training=True,debug=False,completed_epochs=40,optimizer_updates=680,
            source_artifacts_sha256={'training/invocations.jsonl':viewer.direct.sha(invocation)}))
        (score/'console.log').write_text(json.dumps(event(6,642))+'\n',encoding='utf8')
        write_json(score/'complete.json',dict(status='FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING',
            completed_sources=642,completed_CP_positions=82176,full_native_training=False,full_native_evaluation=False))
        # A stale/malformed old status must not invalidate this independent retry.
        old_status=self.old_path.parent/'status.json'
        old_status.write_text('{DEBUG stale original native failure',encoding='utf8')
        write_json(self.old_path,self.original)
        self.request['original_request']=dict(path=str(self.old_path),
            sha256=viewer.direct.sha(self.old_path),stat=[getattr(self.old_path.stat(),key)
                for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')])
        write_json(self.path,self.request)
        digest=viewer.direct.sha(self.path)
        self.registration['request_sha256']=digest;self.lease['request_sha256']=digest
        self.state.update(request_sha256=digest,status='FAILED',
            error='DEBUG new scientific import admission failure')
        for name,record in [('registration.json',self.registration),('lease.json',self.lease),('status.json',self.state)]:
            write_json(self.path.parent/name,record)
        (self.path.parent/'train.log').write_text('DEBUG: Epoch 249\nDEBUG: Epoch time: 86.15 s\n',encoding='utf8')
        protected={path:path.read_bytes() for path in (old_status,self.old_path,self.path,
            self.path.parent/'status.json',self.path.parent/'train.log')}
        actual_read=viewer.pipeline.read
        actual_renderer=viewer.direct.load_renderer(Path(renderer.__file__),viewer.direct.sha(Path(renderer.__file__)))
        with patch.object(viewer.pipeline,'input_path',side_effect=Path), \
                patch.object(viewer.scoring.psutil,'Process',side_effect=viewer.scoring.psutil.NoSuchProcess(999)), \
                patch.object(viewer.pipeline,'read',wraps=actual_read) as reader:
            instance=viewer.RetryViewer(actual_renderer,self.path)
            self.assertIsInstance(instance.source,viewer.pipeline.CompletedGNNSource)
            self.assertIsNone(instance.native_root)
            result=instance.snapshot()
        self.assertTrue(result['snapshot_fresh'])
        self.assertEqual(result['gnn']['status'],'TRAINING_COMPLETE')
        self.assertEqual(result['gnn']['optimizer_updates'],680)
        self.assertEqual((result['scoring']['source_done'],result['scoring']['CP_done']),(642,82176))
        self.assertEqual(result['status'],'FAILED');self.assertEqual(result['native']['completed_epochs'],250)
        self.assertEqual(result['read_problem']['path'],str(self.path.parent/'status.json'))
        self.assertIn('new scientific import',viewer.pipeline.display(result))
        self.assertFalse(result['full_native_training']);self.assertFalse(result['full_native_evaluation'])
        self.assertNotIn(old_status,[Path(call.args[0]) for call in reader.call_args_list])
        self.assertEqual(protected,{path:path.read_bytes() for path in protected})

    def test_binding_failure_is_rendered_as_read_error_with_explicitly_stale_previous_snapshot(self):
        instance=viewer.RetryViewer.__new__(viewer.RetryViewer)
        instance.retry_path=self.path;instance.renderer=self.renderer
        instance.problem=None;instance.last_valid=None;instance.pending_reason='DEBUG native retry wait'
        with patch.object(viewer.pipeline.PipelineViewer,'_snapshot',side_effect=lambda:copy.deepcopy(self.base)):
            first=instance.snapshot()
            self.assertEqual(first['status'],'RUNNING');self.assertTrue(first['snapshot_fresh'])
            self.lease['request_sha256']='c'*64
            failed=instance.snapshot()
        self.assertEqual(failed['status'],'READ_ERROR');self.assertFalse(failed['snapshot_fresh'])
        self.assertTrue(failed['last_valid_snapshot_preserved'])
        self.assertEqual(failed['last_confirmed_status'],'RUNNING')
        self.assertIn('controller/lease',failed['read_problem']['detail'])
        self.assertEqual(failed['native'],first['native'])
        self.assertFalse(failed['full_native_training']);self.assertFalse(failed['full_native_evaluation'])

    def test_viewer_has_no_scientific_imports_filesystem_writes_or_process_signals(self):
        tree=ast.parse(Path(viewer.__file__).read_text(encoding='utf8'))
        modules=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.startswith(('torch','hiercp_v1x','nnunet')) for name in modules))
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(name.endswith(('.write_text','.write_bytes','.unlink','.rename','.replace',
            '.kill','.terminate','.send_signal','.load_state_dict')) for name in calls))

    def test_actual_controller_schema_and_native_cli_match_read_only_viewer(self):
        source=Path(viewer.__file__).parents[1]/'hiercp_v1x/v24_native_startup_retry.py'
        constants={}
        for node in ast.parse(source.read_text(encoding='utf8')).body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ('FORMAT','STATUS_FORMAT','NATIVE_ARGV','COMPLETION_FORMAT','COMPLETION_STATUS'):
                        constants[target.id]=ast.literal_eval(node.value)
        self.assertEqual(constants['FORMAT'],viewer.REQUEST)
        self.assertEqual(constants['STATUS_FORMAT'],viewer.STATUS)
        self.assertEqual(constants['COMPLETION_FORMAT'],viewer.COMPLETION_REQUEST)
        self.assertEqual(constants['COMPLETION_STATUS'],viewer.COMPLETION_STATUS)
        self.assertEqual(constants['NATIVE_ARGV'],self.entry[self.entry.index('--')+1:])


if __name__=='__main__':unittest.main()
