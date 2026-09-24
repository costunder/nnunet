"""Orchestration tests only; mocks are not medical training evidence."""
import subprocess
import tempfile
from pathlib import Path
import unittest
from tools.run_v222_server import stages,execute,ROOT


class ServerRunnerTests(unittest.TestCase):
    def test_pipeline_preserves_production_and_sets_only_debug_batch(self):
        plan=stages(Path('/medical'),Path('/output'),32,9.0)
        names=[name for name,_ in plan]
        self.assertEqual(names,['resources','model_check','observations','graph_DEBUG','paired_cache','profile_DEBUG','gnn_training'])
        debug=dict(plan)['profile_DEBUG'];train=dict(plan)['gnn_training']
        self.assertEqual(debug[debug.index('--batch-size')+1],'32')
        self.assertEqual(debug[debug.index('--allocator-gb')+1],'9.0')
        self.assertNotIn('--calibration',train)
        self.assertNotIn('--batch-size',train)
        self.assertIn('--release-unused',train)

    def test_failure_records_stage_and_does_not_start_training(self):
        (ROOT/'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            output=Path(tmp);seen=[];env={'CUDA_VISIBLE_DEVICES':'existing-allocation'}
            def runner(command,**kwargs):
                seen.append(command)
                self.assertIs(kwargs['env'],env)
                return subprocess.CompletedProcess(command,7)
            with self.assertRaisesRegex(RuntimeError,'prepare failed'):
                execute([('prepare',['prepare-command']),('train',['train-command'])],output,env,runner)
            self.assertEqual(seen,[['prepare-command']])
            self.assertTrue((output/'00_prepare.failed.json').is_file())
            self.assertFalse((output/'01_train.started.json').exists())

    def test_sequential_success_has_stage_receipts(self):
        (ROOT/'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            output=Path(tmp);seen=[]
            def runner(command,**kwargs):
                seen.append(command);return subprocess.CompletedProcess(command,0)
            execute([('prepare',['prepare-command']),('train',['train-command'])],output,{},runner)
            self.assertEqual(seen,[['prepare-command'],['train-command']])
            self.assertTrue((output/'01_train.complete.json').is_file())


if __name__=='__main__':unittest.main()
