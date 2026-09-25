"""Process/lifecycle tests only: no GPU training or process termination."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import psutil
from tools.v222_resume_guard import assert_source_runs_idle,output_argument
from tools.run_v222_server import execute

ROOT=Path(__file__).resolve().parents[1]/'work'


class GuardTests(unittest.TestCase):
    def test_live_recorded_source_worker_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp)
            (root/'worker.json').write_text(json.dumps(dict(pid=123,created_at=456)))
            p=Mock(pid=123);p.create_time.return_value=456;p.status.return_value='running'
            with patch('tools.v222_resume_guard.psutil.Process',return_value=p),patch('tools.v222_resume_guard.psutil.process_iter',return_value=[]):
                with self.assertRaisesRegex(RuntimeError,'123'):
                    assert_source_runs_idle(root/'paired_cache/index.json',root/'training/checkpoint_latest.pt')

    def test_pid_reuse_and_dead_worker_are_not_false_blockers(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp)
            (root/'worker.json').write_text(json.dumps(dict(pid=123,created_at=456)))
            reused=Mock(pid=123);reused.create_time.return_value=999
            own=Mock();own.username.return_value='test-user'
            for factory in (lambda pid:reused if pid==123 else own,
                            lambda pid:(_ for _ in ()).throw(psutil.NoSuchProcess(pid)) if pid==123 else own):
                with patch('tools.v222_resume_guard.psutil.Process',side_effect=factory),patch('tools.v222_resume_guard.psutil.process_iter',return_value=[]):
                    self.assertTrue(assert_source_runs_idle(resume=root/'training/checkpoint_latest.pt')['source_runs_idle'])

    def test_orphan_trainer_with_exact_output_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp).resolve()
            p=Mock(pid=123,info=dict(name='python3.10'))
            p.username.return_value='test-user';p.cwd.return_value=str(root)
            p.cmdline.return_value=['python','run_v222_v1_l0.py','train','--output','training']
            p.status.return_value='running'
            own=Mock();own.username.return_value='test-user'
            with patch('tools.v222_resume_guard.psutil.Process',return_value=own),patch('tools.v222_resume_guard.psutil.process_iter',return_value=[p]):
                with self.assertRaisesRegex(RuntimeError,'source trainer'):
                    assert_source_runs_idle(resume=root/'training/checkpoint_latest.pt')

    def test_unrelated_python_and_other_user_are_not_touched(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp).resolve();own=Mock();own.username.return_value='test-user'
            unrelated=Mock(pid=123,info=dict(name='python'));unrelated.username.return_value='test-user'
            unrelated.cmdline.return_value=['python','unrelated.py','--output',str(root/'training')]
            other=Mock(pid=124,info=dict(name='python'));other.username.return_value='other-user'
            with patch('tools.v222_resume_guard.psutil.Process',return_value=own),patch('tools.v222_resume_guard.psutil.process_iter',return_value=[unrelated,other]):
                self.assertTrue(assert_source_runs_idle(resume=root/'training/checkpoint_latest.pt')['source_runs_idle'])
            other.cmdline.assert_not_called()
            for p in (unrelated,other):
                p.terminate.assert_not_called();p.kill.assert_not_called();p.send_signal.assert_not_called()

    def test_busy_source_never_launches_next_training_and_records_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            runner=Mock()
            with patch('tools.run_v222_server.assert_source_runs_idle',side_effect=RuntimeError('source still active')):
                with self.assertRaisesRegex(RuntimeError,'source still active'):
                    execute([('gnn_training',['python','run_v222_optimized.py','--resume','old','--cache','index'])],Path(tmp),{},runner)
            runner.assert_not_called()
            self.assertTrue((Path(tmp)/'00_gnn_training.failed.json').is_file())

    def test_output_flag_forms(self):
        self.assertEqual(output_argument(['python','run_v222_optimized.py','--output=/run/training']),'/run/training')
        self.assertIsNone(output_argument(['python','watch_v222_server.py','--output','run']))


if __name__=='__main__':unittest.main()
