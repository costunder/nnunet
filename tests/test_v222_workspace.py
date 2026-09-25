"""Execution-policy guards, not medical model performance evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.v222_gpu_workspace import output_path,policy_path,check_resume,ROOT
from tools.run_v222_server import stages


class WorkspaceTests(unittest.TestCase):
    def test_workspace_only_wraps_gpu_stages_and_preserves_cache_command(self):
        default=dict(stages(Path('/medical'),Path('/output'),32,9))
        tuned=dict(stages(Path('/medical'),Path('/output'),32,9,256))
        self.assertEqual(default['paired_cache'],tuned['paired_cache'])
        self.assertEqual(default['observations'],tuned['observations'])
        for stage in ('graph_DEBUG','profile_DEBUG','gnn_training'):
            self.assertIn('--workspace-mib',tuned[stage])
            self.assertIn('256',tuned[stage])
        self.assertNotIn('--batch-size',tuned['gnn_training'])

    def test_entry_points_are_explicit_and_use_new_output(self):
        self.assertEqual(output_path('run_v222_v1_l0.py',['train','--cache','c','--output','o']),Path('o').resolve())
        self.assertEqual(output_path('tools/profile_v1_execution.py',['c','o','release_unused']),Path('o').resolve())
        with self.assertRaises(ValueError):output_path('run_v222_v1_l0.py',['prepare','--output','o'])
        with self.assertRaises(ValueError):output_path('unrelated.py',['--output','o'])

    def test_resume_preserves_recorded_numerical_policy(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            previous=Path(tmp)/'train';checkpoint=previous/'checkpoint_latest.pt'
            args=['train','--resume',str(checkpoint),'--output','new']
            check_resume(args,64)
            with self.assertRaisesRegex(ValueError,'mismatch'):check_resume(args,256)
            policy_path(previous).write_text(json.dumps(dict(workspace_mib=256)))
            check_resume(args,256)
            with self.assertRaisesRegex(ValueError,'mismatch'):check_resume(args,64)


if __name__=='__main__':unittest.main()
