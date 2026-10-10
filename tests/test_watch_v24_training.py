"""Viewer metadata tests; no training, model, process control or GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('v24_watch', Path(__file__).resolve().parents[1]/'tools/watch_v24_training.py')
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class ViewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='DEBUG_v24_viewer_', dir=Path(__file__).absolute().parents[1]/'outputs')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'job'
        self.output = Path(self.tmp.name)/'experiment'
        self.train = self.output/'training'
        self.root.mkdir(); self.train.mkdir(parents=True)
        self.request = dict(GPU=5, production_output=str(self.output), RAM_GiB=64)
        self.write(self.root/'status.json', dict(status='RUNNING', stage='train', request=self.request))
        self.write(self.train/'training_identity.json', dict(binding=dict(epochs=40,
            train_cases=['t'+str(i) for i in range(65)], val_cases=['v'+str(i) for i in range(21)])))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf8')

    @staticmethod
    def lines(path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')

    def test_actual_identity_binding_and_first_validation(self):
        s = watch.Viewer().snapshot(self.root)
        self.assertEqual((s['epoch'], s['epochs'], s['progress']), (1,40,dict(done=0,total=21,unit='환자')))
        self.assertEqual(s['stage'], 'initial_full_validation')

    def test_prepare_bar_does_not_claim_training(self):
        self.write(self.root/'status.json', dict(status='RUNNING', stage='prepare_inputs', request=self.request))
        (self.root/'console.log').write_text('V24 UPPER PREPARE completed=2/86 active_U=128 elapsed_seconds=5\n', encoding='utf8')
        s = watch.Viewer().snapshot(self.root)
        self.assertEqual((s['stage'],s['progress']['done'],s['epoch']), ('prepare_inputs',2,None))

    def test_partial_upper_preparation_inside_training_has_own_bar(self):
        self.lines(self.train/'curve.jsonl',[dict(epoch=4)])
        (self.root/'console.log').write_text('V24 UPPER PREPARE completed=2/86 active_U=23 elapsed_seconds=5\n',encoding='utf8')
        s=watch.Viewer().snapshot(self.root)
        self.assertEqual((s['stage'],s['epoch'],s['active_U']),('upper_prepare',5,23))
        self.assertEqual(s['progress'],dict(done=2,total=86,unit='upper 환자'))

    def test_completed_upper_preparation_restores_derived_training_phase(self):
        self.lines(self.train/'curve.jsonl',[dict(epoch=4)])
        (self.root/'console.log').write_text('V24 UPPER PREPARE completed=2/86 active_U=23 elapsed_seconds=5\nV24 UPPER PREPARE completed=86/86 active_U=23 elapsed_seconds=100\n',encoding='utf8')
        s=watch.Viewer().snapshot(self.root)
        self.assertEqual((s['stage'],s['epoch'],s['progress']['done'],s['progress']['total']),('training',5,0,65))

    def test_new_epoch_uses_recorded_curriculum_U_before_first_new_update(self):
        self.lines(self.train/'update_timing.jsonl',[dict(status='OPTIMIZER_UPDATED',epoch=4,update=68,
            active_u=7,case_ids=['t0','t1','t2','t3'],loss=.7)])
        self.lines(self.train/'curve.jsonl',[dict(epoch=4,curriculum_transition=dict(previous_active_u=7,next_active_u=23))])
        for name in ('update_timing.jsonl','curve.jsonl'):
            os.utime(self.train/name, ns=(1_700_000_000_000_000_000,)*2)
        s=watch.Viewer().snapshot(self.root)
        self.assertEqual((s['stage'],s['epoch'],s['active_U'],s['progress']['done']),('training',5,23,0))
        self.assertNotEqual(s['stage'],'upper_prepare')

    def test_partial_line_is_not_counted_twice(self):
        path = self.train/'update_timing.jsonl'
        row = dict(status='OPTIMIZER_UPDATED', epoch=1, update=1, active_u=7, case_ids=['t0','t1','t2','t3'], loss=.7)
        path.write_text(json.dumps(row), encoding='utf8')
        viewer = watch.Viewer()
        self.assertIsNone(viewer.snapshot(self.root)['optimizer_updates'])
        with path.open('a',encoding='utf8') as f: f.write('\n')
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],4)
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],4)

    def test_complete_pipeline_with_paused_invocation_is_paused(self):
        self.write(self.root/'status.json',dict(status='COMPLETE',stage='train',request=self.request))
        self.lines(self.train/'invocations.jsonl',[dict(status='PAUSED',full_training=False)])
        s = watch.Viewer().snapshot(self.root)
        self.assertEqual(s['status'],'PAUSED'); self.assertFalse(s['full_training_completed'])

    def test_running_continuation_ignores_stale_failure_pause_and_preserves_cursor(self):
        old = dict(epoch=4,phase='full_validation',updates=68,active_u=7,train_position=65,evaluation_position=8,history_epochs=[1,2,3])
        self.lines(self.train/'invocations.jsonl',[dict(status='PAUSED',full_training=False)])
        self.lines(self.train/'failures.jsonl',[dict(error='old MemoryError')])
        val = self.train/'validation_timing.jsonl'
        self.lines(val,[dict(epoch=4,phase='full_validation',active_u=128,case_ids=['v0','v1','v2','v3'])])
        copied = {'training/'+name:dict(bytes=(self.train/name).stat().st_size) for name in ('invocations.jsonl','failures.jsonl','validation_timing.jsonl')}
        self.write(self.output/'training_continuation.json',dict(source=dict(latest=old),copied_files=copied,created_at=1e12))
        viewer=watch.Viewer();s=viewer.snapshot(self.root)
        self.assertEqual((s['status'],s['stage'],s['progress']['done'],s['error']),('RUNNING','full_validation',8,None))
        with val.open('a',encoding='utf8') as f:f.write(json.dumps(dict(epoch=4,phase='full_validation',active_u=128,case_ids=['v8','v9','v10','v11']))+'\n')
        # File mtimes of this local fixture are earlier than its intentionally future receipt.
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],12)
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],12)

    def test_failed_status_wins_over_old_paused(self):
        self.write(self.root/'status.json',dict(status='FAILED',stage='train',request=self.request,error='current failure'))
        self.lines(self.train/'invocations.jsonl',[dict(status='PAUSED',full_training=False)])
        self.assertEqual(watch.Viewer().snapshot(self.root)['status'],'FAILED')

    def test_pointer_retarget_resets_reader(self):
        pointer=Path(self.tmp.name)/'pointer.json';self.write(pointer,dict(job_root=str(self.root)))
        self.assertEqual(watch.resolve_job_root(pointer=pointer),self.root.resolve())
        newer=Path(self.tmp.name)/'next';newer.mkdir();self.write(pointer,dict(job_root=str(newer)))
        self.assertEqual(watch.resolve_job_root(pointer=pointer),newer.resolve())
        self.assertEqual(watch.Viewer().snapshot(newer)['status'],'WAITING')

    def test_log_resolves_current_symlink_target_each_call(self):
        stable=Path(self.tmp.name)/'gpu5.log';target=self.root/'console.log';target.touch()
        with patch.object(Path,'resolve',return_value=target) as mocked:
            self.assertEqual(watch.resolve_job_root(log=stable),self.root)
            self.assertEqual(watch.resolve_job_root(log=stable),self.root)
            self.assertEqual(mocked.call_count,2)

    def test_numeric_report_selection_and_actual_metrics(self):
        metric=dict(per_P_patient_mrr=.4,per_P_patient_top1=.3)
        for epoch in (9,10):self.write(self.train/f'stage_validation_epoch_{epoch}.json',dict(epoch=epoch,active_u=23,metrics=metric))
        s=watch.Viewer().snapshot(self.root)
        self.assertEqual(s['reports']['stage_validation']['epoch'],10)
        self.assertIn('MRR 0.4000 top1 0.3000',watch.text_snapshot(s))

    def test_empty_new_report_preserves_explicitly_stale_snapshot_then_recovers(self):
        metric=dict(per_P_patient_mrr=.4,per_P_patient_top1=.3)
        previous=self.train/'full_validation_epoch_009.json'
        self.write(previous,dict(epoch=9,active_u=128,metrics=metric))
        viewer=watch.Viewer(); observed=viewer.snapshot(self.root)
        newest=self.train/'full_validation_epoch_010.json';newest.write_text('',encoding='utf8')
        pending=viewer.snapshot(self.root)
        self.assertEqual(pending['status'],'READ_PENDING')
        self.assertFalse(pending['snapshot_fresh'])
        self.assertEqual(pending['reports'],observed['reports'])
        self.assertEqual(pending['last_confirmed_status'],'RUNNING')
        self.assertEqual(pending['read_problem']['path'],str(newest))
        self.assertIn('새 기록이 아님',watch.text_snapshot(pending))
        self.write(newest,dict(epoch=10,active_u=128,metrics=dict(per_P_patient_mrr=.5,per_P_patient_top1=.4)))
        fresh=viewer.snapshot(self.root)
        self.assertEqual((fresh['status'],fresh['reports']['full_validation']['epoch']),('RUNNING',10))
        self.assertTrue(fresh['snapshot_fresh']);self.assertNotIn('read_problem',fresh)

    def test_partial_report_on_first_poll_has_no_invented_epoch_or_metrics(self):
        path=self.train/'stage_validation_epoch_001.json';path.write_text('{"epoch":',encoding='utf8')
        pending=watch.Viewer().snapshot(self.root)
        self.assertEqual(pending['status'],'READ_PENDING')
        self.assertIsNone(pending['progress']);self.assertNotIn('epoch',pending)
        self.assertNotIn('reports',pending);self.assertFalse(pending['last_valid_snapshot_preserved'])
        self.assertIn(str(path),watch.text_snapshot(pending))

    def test_invalid_report_shape_is_visible_and_viewer_keeps_retrying(self):
        path=self.train/'full_validation_epoch_001.json'
        viewer=watch.Viewer();viewer.snapshot(self.root)
        for value in ([],dict(epoch='bad',active_u=128,metrics={}),dict(epoch=1,active_u=128,metrics=[]),
                      dict(epoch=1,active_u=128,metrics=dict(per_P_patient_mrr='bad'))):
            with self.subTest(value=value):
                self.write(path,value);failed=viewer.snapshot(self.root)
                self.assertEqual(failed['status'],'READ_ERROR')
                self.assertTrue(failed['read_problem']['retries_continue'])
                self.assertFalse(failed['full_training_completed'])
        self.write(path,dict(epoch=1,active_u=128,metrics=dict(per_P_patient_mrr=.2,per_P_patient_top1=.1)))
        self.assertEqual(viewer.snapshot(self.root)['status'],'RUNNING')

    def test_unchanged_malformed_json_becomes_clear_read_error_without_exit(self):
        path=self.train/'train_probe_epoch_001.json';path.write_text('{malformed}',encoding='utf8')
        viewer=watch.Viewer()
        with patch.object(watch.time,'monotonic',side_effect=[100.,111.]):
            first=viewer.snapshot(self.root);second=viewer.snapshot(self.root)
        self.assertEqual(first['status'],'READ_PENDING');self.assertEqual(second['status'],'READ_ERROR')
        self.assertIn('JSONDecodeError',second['read_problem']['detail'])
        self.assertTrue(second['read_problem']['retries_continue'])

    def test_partial_status_preserves_last_valid_progress_without_consuming_updates(self):
        row=dict(status='OPTIMIZER_UPDATED',epoch=1,update=1,active_u=7,case_ids=['t0','t1','t2','t3'],loss=.7)
        self.lines(self.train/'update_timing.jsonl',[row])
        viewer=watch.Viewer();observed=viewer.snapshot(self.root)
        (self.root/'status.json').write_text('{"status":',encoding='utf8')
        pending=viewer.snapshot(self.root)
        self.assertEqual(pending['optimizer_updates'],observed['optimizer_updates'])
        self.assertEqual(pending['progress']['done'],4);self.assertFalse(pending['snapshot_fresh'])
        self.write(self.root/'status.json',dict(status='RUNNING',stage='train',request=self.request))
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],4)

    def test_broken_later_jsonl_line_cannot_consume_valid_earlier_update(self):
        path=self.train/'update_timing.jsonl'
        row=dict(status='OPTIMIZER_UPDATED',epoch=1,update=1,active_u=7,case_ids=['t0','t1','t2','t3'],loss=.7)
        path.write_text(json.dumps(row)+'\n{malformed}\n',encoding='utf8')
        viewer=watch.Viewer();self.assertEqual(viewer.snapshot(self.root)['status'],'READ_PENDING')
        self.lines(path,[row])
        observed=viewer.snapshot(self.root)
        self.assertEqual(observed['optimizer_updates'],1);self.assertEqual(observed['progress']['done'],4)
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],4)

    def test_broken_replacement_jsonl_commits_reset_only_after_valid_repair(self):
        path=self.train/'update_timing.jsonl'
        row=dict(status='OPTIMIZER_UPDATED',epoch=1,update=1,active_u=7,case_ids=['t0','t1','t2','t3'],loss=.7)
        self.lines(path,[row]);viewer=watch.Viewer()
        self.assertEqual(viewer.snapshot(self.root)['optimizer_updates'],1)
        replacement=self.train/'DEBUG_replacement.jsonl';replacement.write_text('{malformed}\n',encoding='utf8')
        os.replace(replacement,path)
        self.assertEqual(viewer.snapshot(self.root)['status'],'READ_PENDING')
        newer=dict(row,update=2,case_ids=['t4','t5','t6','t7'])
        self.lines(path,[newer])
        repaired=viewer.snapshot(self.root)
        self.assertEqual((repaired['optimizer_updates'],repaired['progress']['done']),(2,4))
        self.assertEqual(viewer.snapshot(self.root)['progress']['done'],4)

    def test_partial_utf8_write_is_pending_and_completed_record_recovers(self):
        path=self.train/'full_validation_epoch_001.json'
        path.write_bytes(b'{"message":"\xec')
        viewer=watch.Viewer();pending=viewer.snapshot(self.root)
        self.assertEqual(pending['status'],'READ_PENDING')
        self.assertIn('UTF8',pending['read_problem']['detail'])
        self.write(path,dict(epoch=1,active_u=128,metrics=dict(per_P_patient_mrr=.2,per_P_patient_top1=.1)))
        self.assertEqual(viewer.snapshot(self.root)['status'],'RUNNING')

    def test_root_switch_does_not_display_another_jobs_stale_metrics(self):
        viewer=watch.Viewer();viewer.snapshot(self.root)
        new=Path(self.tmp.name)/'new_job';new.mkdir();(new/'status.json').write_text('',encoding='utf8')
        pending=viewer.snapshot(new)
        self.assertEqual(pending['job_root'],str(new));self.assertFalse(pending['last_valid_snapshot_preserved'])
        self.assertNotIn('GPU',pending)

    def test_once_json_display_returns_pending_record_instead_of_traceback(self):
        import io
        path=self.train/'full_validation_epoch_001.json';path.write_text('',encoding='utf8')
        stream=io.StringIO()
        with patch.object(watch.sys,'stdout',stream):
            watch.main(['--job-root',str(self.root),'--json'])
        shown=json.loads(stream.getvalue())
        self.assertEqual(shown['status'],'READ_PENDING');self.assertEqual(shown['read_problem']['path'],str(path))

    def test_empty_pointer_once_reports_pending_without_losing_control(self):
        import io
        pointer=Path(self.tmp.name)/'pointer.json';pointer.write_text('',encoding='utf8')
        stream=io.StringIO()
        with patch.object(watch.sys,'stdout',stream):
            watch.main(['--pointer',str(pointer),'--json'])
        shown=json.loads(stream.getvalue());self.assertEqual(shown['status'],'READ_PENDING')

    def test_native_text_uses_observed_losses_and_pseudo_dice(self):
        n=watch.parse_nnunet_lines(['2026: Epoch 3','train_loss -0.18','val_loss -0.16','Pseudo dice [0.93, 0.42]','Epoch time: 40 s'])
        self.assertEqual(n,dict(epoch_index=3,epoch_finished=True,train_loss=-.18,val_loss=-.16,pseudo_dice=[.93,.42]))
        self.assertIsNone(watch.parse_nnunet_lines(['not an epoch']))

    def test_actual_numpy_scalar_native_log(self):
        n=watch.parse_nnunet_lines(['2026-10-10: Epoch 2',
            '2026-10-10 13:50:16.767042: Pseudo dice [np.float32(0.9322), np.float64(0.6782)]'])
        self.assertEqual(n['pseudo_dice'],[.9322,.6782])

    def test_compact_four_row_tty_renderer(self):
        import io
        stream=io.StringIO(); terminal=watch.TerminalView(stream)
        self.addCleanup(terminal.close)
        metric=dict(per_P_patient_mrr=.4,per_P_patient_top1=.3)
        report=dict(epoch=4,active_u=7,metrics=metric)
        snapshot=dict(GPU=5,status='RUNNING',epoch=4,epochs=40,completed_epochs=3,
            active_U=128,stage='full_validation',stage_label='전체 validation',
            progress=dict(done=8,total=21),loss=.7,optimizer_updates=68,RSS_GiB=48,RAM_limit_GiB=64,
            reports=dict(stage_validation=report,full_validation=dict(report,active_u=128)))
        terminal.render(snapshot)
        self.assertEqual(len(terminal.bars),4)
        self.assertEqual((terminal.bars[1].n,terminal.bars[1].total),(8,21))
        self.assertTrue(all(len(bar.desc)<=100 for bar in terminal.bars))
        self.assertIn('stage e4 U7',terminal.bars[3].desc)
        self.assertIn('full128 e4 U128',terminal.bars[3].desc)

    def test_tty_renderer_explicitly_overrides_disabled_environment(self):
        import io, os
        with patch.dict(os.environ, {'TQDM_DISABLE':'1'}):
            stream=io.StringIO();terminal=watch.TerminalView(stream)
            try:
                terminal.render(dict(GPU=6,status='RUNNING',epoch=4,epochs=40,
                    completed_epochs=3,stage='full_validation',stage_label='전체 validation',
                    active_U=128,progress=dict(done=0,total=21),reports={}))
                self.assertTrue(all(bar.disable is False for bar in terminal.bars))
                self.assertIn('\r',stream.getvalue())
                self.assertIn('GPU6 RUNNING',stream.getvalue())
            finally:terminal.close()

    def test_rss_sums_verified_arm_tree_and_excludes_other_uid(self):
        import sys
        from types import SimpleNamespace as NS
        def process(rss,uid):return NS(memory_info=lambda:NS(rss=rss),uids=lambda:NS(real=uid))
        owned=process(100,42);owned.create_time=lambda:123.;owned.children=lambda recursive:[process(200,42),process(300,7)]
        fake=NS(Process=lambda pid:owned,NoSuchProcess=type('NoSuchProcess',(Exception,),{}))
        with patch.dict(sys.modules,{'psutil':fake}), patch.object(watch.os,'getuid',return_value=42,create=True):
            rss,label=watch.Viewer._rss(dict(status='RUNNING',child_pid=1,child_create_time=123.),None)
        self.assertEqual(rss,300)
        self.assertEqual(label,'live owned arm process tree')

    def test_no_training_or_torch_imports(self):
        import ast
        tree=ast.parse(Path(watch.__file__).read_text(encoding='utf8'))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(x and (x.startswith('torch')or x.startswith('hiercp')or x.startswith('nnunet')) for x in imports))


if __name__=='__main__':
    unittest.main()
