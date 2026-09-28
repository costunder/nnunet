"""Orchestration tests only; mocks are not medical training evidence."""
import subprocess
import tempfile
from pathlib import Path
import unittest
from tools.run_v222_server import stages,execute,ROOT


class ServerRunnerTests(unittest.TestCase):
    def test_prepared_cache_process_run_reaches_full_ranking_training(self):
        cache=Path('/medical/prepared/paired_cache/index.json')
        output=Path('/medical/gnn40')
        plan=stages(Path('/medical'),output,32,9.0,256,
                    optimized=True,process_loader=True,cache=cache,
                    feature_coordinates='stride4',training_objective='observed_rank_v1')
        self.assertEqual([name for name,_ in plan],
                         ['resources','model_check','graph_DEBUG','profile_DEBUG','gnn_training'])
        train=dict(plan)['gnn_training']
        self.assertEqual(Path(train[2]),ROOT/'tools/run_v222_process_runtime.py')
        for flag,value in (('--cache',str(cache)),('--output',str(output/'training')),
                           ('--feature-coordinates','stride4'),
                           ('--training-objective','observed_rank_v1'),('--workspace-mib','256')):
            self.assertEqual(train[train.index(flag)+1],value)
        for forbidden in ('--resume','--debug-profile','--debug-stop-after-step','--batch-size','--calibration'):
            self.assertNotIn(forbidden,train)

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
        with tempfile.TemporaryDirectory() as tmp:
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
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);seen=[]
            def runner(command,**kwargs):
                seen.append(command);return subprocess.CompletedProcess(command,0)
            execute([('prepare',['prepare-command']),('train',['train-command'])],output,{},runner)
            self.assertEqual(seen,[['prepare-command'],['train-command']])
            self.assertTrue((output/'01_train.complete.json').is_file())


if __name__=='__main__':unittest.main()
