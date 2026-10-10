"""Viewer metadata tests; no training, model, process control or GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('v24_watch', Path(__file__).resolve().parents[1]/'tools/watch_v24_training.py')
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class ViewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
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

    def test_no_training_or_torch_imports(self):
        import ast
        tree=ast.parse(Path(watch.__file__).read_text(encoding='utf8'))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(x and (x.startswith('torch')or x.startswith('hiercp')or x.startswith('nnunet')) for x in imports))


if __name__=='__main__':
    unittest.main()
