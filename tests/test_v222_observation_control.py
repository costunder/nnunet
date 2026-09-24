"""Read-only guard tests; never signal a process in this suite."""
from pathlib import Path
import tempfile
import unittest
from tools.stop_v222_observation_job import is_runner,check_phase


class ObservationControlGuards(unittest.TestCase):
    def test_exact_runner_and_output_required(self):
        project=Path.cwd().resolve();output=project/'work/guard-test'
        command=['python','-u','tools/run_v222_server.py','--output','work/guard-test']
        self.assertTrue(is_runner(command,project,project,output))
        self.assertFalse(is_runner(command,project,project,project/'work/another'))
        self.assertFalse(is_runner(['bash','--output','work/guard-test'],project,project,output))
        self.assertFalse(is_runner(command+['--output','work/other'],project,project,output))

    def test_refuse_missing_and_later_stages(self):
        root=Path.cwd()/'work';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            output=Path(tmp)
            with self.assertRaises(RuntimeError):check_phase(output)
            (output/'02_observations.started.json').write_text('{}')
            check_phase(output)
            (output/'03_graph_DEBUG.started.json').write_text('{}')
            with self.assertRaises(RuntimeError):check_phase(output)


if __name__=='__main__':unittest.main()
