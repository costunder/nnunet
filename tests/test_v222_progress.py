"""Console/protocol tests only, never presented as a medical training test."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from tools.run_v222_server import launch_worker, execute, ROOT
from tools.watch_v222_server import LogReader, Progress, terminal_status, watch, request_pause, training_display


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_support_display_includes_saved_epoch_and_step_without_inventing_epoch41(self):
        state=Progress();state.event(dict(stage='support_memory',completed=6144,total=11279,physical_batch=32))
        saved=dict(phase='refresh_memory',epoch=10,step=1234)
        label,detail=training_display(state,saved,dict(epochs=40))
        self.assertIn('epoch 10/40',label);self.assertIn('support refresh',label)
        self.assertIn('saved step=1234',detail)
        saved.update(phase='initial_memory',epoch=1,step=0)
        self.assertIn('before epoch 1/40',training_display(state,saved,dict(epochs=40))[0])
        saved.update(phase='final_memory',epoch=41)
        self.assertNotIn('41',training_display(state,saved,dict(epochs=40))[0])
        self.assertEqual(state.done,6144)

    def test_pause_request_requires_initialized_selected_training_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'not initialized'):request_pause(root)
            training=root/'training';training.mkdir();(training/'initialization.json').write_text('{}')
            checkpoint=training/'checkpoint_latest.pt';checkpoint.write_bytes(b'protocol fixture, not trained weights')
            first=request_pause(root);before=first.read_bytes()
            self.assertEqual(request_pause(root),first)
            self.assertEqual(first.read_bytes(),before)
            self.assertEqual(checkpoint.read_bytes(),b'protocol fixture, not trained weights')

    def test_ctrl_c_requests_pause_and_waits_for_confirmation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            root=Path(tmp);training=root/'training';training.mkdir()
            (training/'initialization.json').write_text('{}')
            (training/'execution_contract.json').write_text(json.dumps(dict(epochs=40)))
            (training/'checkpoint_status.json').write_text(json.dumps(dict(phase='refresh_memory',epoch=10,step=100,saved_at=1)))
            (root/'console.log').write_text(json.dumps(dict(stage='support_memory',completed=6144,total=11279))+'\n')
            calls=[]
            def interrupt_then_confirm(_):
                calls.append(1)
                if len(calls)==1:raise KeyboardInterrupt
                self.assertTrue((training/'STOP_AFTER_BATCH').is_file())
                (root/'pipeline_paused.json').write_text('{}')
            screen=Terminal()
            with patch('tools.watch_v222_server.time.sleep',side_effect=interrupt_then_confirm):
                self.assertEqual(watch(root,stream=screen),0)
            self.assertEqual(len(calls),2)
            self.assertIn('epoch 10/40',screen.getvalue())
            self.assertIn('6144/11279',screen.getvalue())
            self.assertIn('PAUSE REQUESTED',screen.getvalue())
            self.assertIn('PAUSED |',screen.getvalue())

    def test_real_cooperative_child_finishes_only_after_viewer_requests_pause(self):
        import time
        real_sleep=time.sleep
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            root=Path(tmp)/'run'
            script="\n".join([
                'import time', 'from pathlib import Path', f'root=Path({str(root)!r})',
                "(root/'training').mkdir()", "(root/'training/initialization.json').write_text('{}')",
                'end=time.monotonic()+10',
                "while not (root/'training/STOP_AFTER_BATCH').exists() and time.monotonic()<end: time.sleep(.02)",
                "assert (root/'training/STOP_AFTER_BATCH').exists(), 'pause was not requested'",
                "(root/'training/paused.json').write_text('{}')", "(root/'pipeline_paused.json').write_text('{}')"])
            child=launch_worker([sys.executable,'-u','-c',script],root)
            for _ in range(200):
                if (root/'training/initialization.json').is_file():break
                real_sleep(.02)
            interrupted=[]
            def interrupt_once(_):
                if not interrupted:
                    interrupted.append(True);raise KeyboardInterrupt
                real_sleep(.02)
            with patch('tools.watch_v222_server.time.sleep',side_effect=interrupt_once):
                self.assertEqual(watch(root,stream=Terminal()),0)
            self.assertEqual(child.wait(timeout=15),0)

    def test_stage_completion_is_not_an_item_progress_event(self):
        for name in ('raw_inventory', 'raw_inventory_heartbeat', 'paired_cache', 'support_memory'):
            with self.subTest(stage=name):
                state=Progress()
                state.event(dict(event='stage_started',stage=name))
                state.event(dict(stage=name,completed=14102,total=14102,active_cases=[]))
                # Exact producer schema: no completed/total on stage_complete.
                state.event(dict(event='stage_complete',stage=name,seconds=7604.0))
                self.assertEqual((state.done,state.total),(14102,14102))
                self.assertIn('complete',state.detail)
                state.event(dict(event='stage_started',stage='profile_DEBUG'))
                self.assertEqual(state.stage,'profile_DEBUG')
                self.assertIsNone(state.total)

    def test_actual_runner_event_stream_can_be_replayed_after_cache_completion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            root=Path(tmp);log=root/'console.log'
            def child(command,**kwargs):
                # Protocol-only child stand-in; execute emits real lifecycle events.
                print(json.dumps(dict(stage='paired_cache',completed=14102,total=14102)))
                print(json.dumps(dict(stage='paired_cache_complete',observations=14102)))
                return subprocess.CompletedProcess(command,0)
            with log.open('w') as stream,redirect_stdout(stream):
                execute([('paired_cache',['protocol-test'])],root,{},child)
            state=Progress()
            for row in LogReader(log).read():state.event(row)
            self.assertEqual(state.done,14102)
            screen=Terminal()
            with patch('tools.watch_v222_server.time.sleep',side_effect=KeyboardInterrupt):
                self.assertEqual(watch(root,stream=screen,detach_on_interrupt=True),0)
            self.assertIn('14102/14102',screen.getvalue())
            self.assertNotIn('it/s',screen.getvalue())

    def test_split_log_record_and_old_python_dict(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            path=Path(tmp)/'console.log'
            path.write_bytes(b'{"stage":"raw_inventory","completed":2')
            reader=LogReader(path)
            self.assertEqual(list(reader.read()),[])
            with path.open('ab') as f:
                f.write(b',"total":131}\nordinary text\n')
                f.write(b"{'stage': 'preparation_admission', 'workers': 1, 'tasks': 128}\n")
            rows=list(reader.read())
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['completed'],2)
            state=Progress()
            for row in rows:state.event(row)
            self.assertEqual(state.phase,'preparation_admission')
            self.assertIsNone(state.total)  # Never fabricate benchmark percent.
            self.assertIn('workers=1',state.detail)
            self.assertEqual(list(reader.read()),[])

    def test_training_epoch_progress_and_unknown_validation(self):
        state=Progress()
        state.event(dict(stage='optimization',epoch=2,visited_queries=512,total_queries=11279,
                         step=240,total_steps=9000,loss=.45))
        self.assertEqual((state.done,state.total),(512,11279))
        self.assertIn('epoch 2',state.phase)
        state.phase_to('epoch 2 validation')
        self.assertIsNone(state.total)
        self.assertEqual(state.done,0)

    def test_no_false_completion_and_visible_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            root=Path(tmp)
            (root/'06_gnn_training.complete.json').write_text('{}')
            self.assertIsNone(terminal_status(root))
            (root/'02_observations.failed.json').write_text('{"returncode":7}')
            (root/'console.log').write_text('Traceback: protocol-test failure\n')
            screen=Terminal()
            self.assertEqual(watch(root,stream=screen),1)
            self.assertIn('FAILED',screen.getvalue())
            self.assertIn('Traceback: protocol-test failure',screen.getvalue())

    def test_detached_worker_survives_viewer_interrupt_and_logs_to_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            output=Path(tmp)/'run'
            script=("import time,json; from pathlib import Path; "
                    "print(json.dumps(dict(stage='raw_inventory',completed=1,total=2)),flush=True); "
                    "time.sleep(1); "
                    f"Path({str(output/'pipeline_complete.json')!r}).write_text('{{}}'); "
                    "print('PROTOCOL TEST FINISHED',flush=True)")
            child=launch_worker([sys.executable,'-u','-c',script],output)
            screen=Terminal()
            with patch('tools.watch_v222_server.time.sleep',side_effect=KeyboardInterrupt):
                self.assertEqual(watch(output,stream=screen,detach_on_interrupt=True),0)
            self.assertIsNone(child.poll())
            self.assertEqual(child.wait(timeout=20),0)
            self.assertIn('PROTOCOL TEST FINISHED',(output/'console.log').read_text())
            self.assertNotIn('PROTOCOL TEST FINISHED',screen.getvalue())
            self.assertIn('training was not signalled',screen.getvalue())
            self.assertTrue(terminal_status(output)[0].startswith('COMPLETE'))


if __name__=='__main__':unittest.main()
