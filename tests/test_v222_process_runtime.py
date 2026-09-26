"""Resume-policy tests; synthetic metadata is not a training checkpoint."""
import copy
from pathlib import Path
import tempfile
import unittest
from tools import run_v222_process_runtime as runtime
from tools.run_v222_server import stages


class ProcessRuntimeTest(unittest.TestCase):
    def saved(self, identity):
        return dict(execution_policy=dict(runtime_sha256=identity,workspace_mib=256),
                    state=dict(release_unused=True,step=104,epoch=2,memory_next=3232))

    def test_known_prior_runtime_accepts_without_mutating_state(self):
        saved=self.saved(runtime.prior_identity());before=copy.deepcopy(saved)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(runtime.validate_resume(saved,Path(root)/'checkpoint_latest.pt'),(256,True))
        self.assertEqual(saved,before)

    def test_current_process_runtime_can_resume(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(runtime.validate_resume(self.saved(runtime.runtime_identity()),Path(root)/'checkpoint_latest.pt'),(256,True))

    def test_snapshot_release_retains_legacy_coordinates(self):
        from tools.v222_review_contracts import resolve_feature_contract
        saved=self.saved(runtime.PUBLISHED_SNAPSHOT_RUNTIME);before=copy.deepcopy(saved)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(runtime.validate_resume(saved,Path(root)/'checkpoint_latest.pt'),(256,True))
        self.assertEqual(resolve_feature_contract(saved),'legacy')
        self.assertEqual(saved,before)

    def test_fresh_process_preflight_uses_same_coordinates(self):
        for contract in ('legacy','stride4',None):
            result=dict(stages(Path('/medical'),Path('/new'),32,9.,optimized=True,
                cache=Path('/old/index.json'),process_loader=True,feature_coordinates=contract,
                training_objective='observation_ce' if contract=='legacy' else None))
            for name in ('graph_DEBUG','profile_DEBUG'):
                command=result[name]
                self.assertEqual(command[command.index('--feature-coordinates')+1],contract or 'stride4')
            if contract:
                command=result['gnn_training']
                self.assertEqual(command[command.index('--feature-coordinates')+1],contract)

    def test_published_process_runtime_migration_preserves_state(self):
        saved=self.saved(runtime.PUBLISHED_PROCESS_RUNTIME);before=copy.deepcopy(saved)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(runtime.validate_resume(saved,Path(root)/'checkpoint_latest.pt'),(256,True))
            changed=copy.deepcopy(saved)
            changed['execution_policy']['runtime_sha256']['tools/v222_process_loader.py']='unverified'
            with self.assertRaisesRegex(ValueError,'runtime changed'):
                runtime.validate_resume(changed,Path(root)/'checkpoint_latest.pt')
        self.assertEqual(saved,before)

    def test_unknown_runtime_and_allocator_change_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint=Path(root)/'checkpoint_latest.pt'
            with self.assertRaisesRegex(ValueError,'runtime changed'):
                runtime.validate_resume(self.saved({'unknown':'hash'}),checkpoint)
            with self.assertRaisesRegex(ValueError,'allocator'):
                runtime.validate_resume(self.saved(runtime.prior_identity()),checkpoint,release_override=False)

    def test_server_uses_existing_cache_and_resume_without_preparation(self):
        result=stages(Path('/medical'),Path('/new'),32,9.,optimized=True,
            cache=Path('/old/index.json'),resume=Path('/old/checkpoint_latest.pt'),process_loader=True)
        self.assertEqual([name for name,_ in result],['resources','model_check','gnn_training'])
        command=result[-1][1]
        self.assertTrue(any(str(value).endswith('run_v222_process_runtime.py') for value in command))
        self.assertIn('--resume',command)


if __name__=='__main__':unittest.main()
