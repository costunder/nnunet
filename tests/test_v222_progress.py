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
from tools.watch_v222_server import LogReader, Progress, terminal_status, watch


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
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
                self.assertEqual(watch(root,stream=screen),0)
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
                self.assertEqual(watch(output,stream=screen),0)
            self.assertIsNone(child.poll())
            self.assertEqual(child.wait(timeout=20),0)
            self.assertIn('PROTOCOL TEST FINISHED',(output/'console.log').read_text())
            self.assertNotIn('PROTOCOL TEST FINISHED',screen.getvalue())
            self.assertIn('training was not signalled',screen.getvalue())
            self.assertTrue(terminal_status(output)[0].startswith('COMPLETE'))


if __name__=='__main__':unittest.main()
