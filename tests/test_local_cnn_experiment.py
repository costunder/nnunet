"""Synthetic orchestration tests only; no model accuracy or training evidence."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.run_local_cnn_experiment import run_experiment,exclusive,select_checkpoint

class ExperimentTests(unittest.TestCase):
    def setUp(self):
        parent=(Path(__file__).resolve().parents[1]/'work');parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='cnn_experiment_unit_',dir=parent)
        assert Path(self.tmp.name).resolve().is_relative_to(parent.resolve())
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.a=SimpleNamespace(experiment=self.root/'m20',cache=self.root/'original.json',config=self.root/'config.json',
            margin_mm=20,debug=True,workers=4,cuda_gib=12,rss_gib=32,resident_gib=8,batch_candidates=[32],
            support_patients=16,device_cache_gib=0,debug_pause_step=None)
        self.request=dict(original_inventory_sha256='synthetic-unit-fixture',local_cnn={'margin_mm':20})
        self.calls=[]
    def fake(self,command):
        self.calls.append(command);out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
        if 'prepare' in command:
            (out/'index.json').write_text(json.dumps(dict(format='native_local_cnn_inventory_v1',
                original_inventory_sha256=self.request['original_inventory_sha256'],local_cnn=self.request['local_cnn'],debug=True)))
        else:(out/'checkpoint_latest.pt').write_bytes(b'SYNTHETIC TEST CHECKPOINT; NOT MODEL WEIGHTS')
    def run_one(self):return run_experiment(self.a,self.request,self.fake)
    def test_new_then_resume_uses_own_checkpoint_and_preserves_old(self):
        first=self.run_one();data=first.read_bytes();second=self.run_one()
        self.assertNotEqual(first,second);self.assertEqual(first.read_bytes(),data)
        self.assertEqual(self.calls[-1][self.calls[-1].index('--resume')+1],str(first))
        self.assertEqual(sum('prepare' in c for c in self.calls),1)
    def test_separate_experiments_never_share_latest(self):
        first=self.run_one();self.a.experiment=self.root/'m30';second=self.run_one()
        self.assertNotEqual(first,second);self.assertNotIn('--resume',self.calls[-1])
    def test_changed_contract_rejected_before_child(self):
        self.run_one();old=len(self.calls);changed=copy.deepcopy(self.request);changed['local_cnn']['margin_mm']=30
        with self.assertRaisesRegex(ValueError,'settings'):run_experiment(self.a,changed,self.fake)
        self.assertEqual(len(self.calls),old)
    def test_lock_rejects_second_writer_and_releases_on_error(self):
        with exclusive(self.a.experiment):
            with self.assertRaisesRegex(RuntimeError,'locked'):self.run_one()
        self.assertFalse((self.a.experiment/'.experiment.lock').exists())
        with self.assertRaisesRegex(RuntimeError,'failure'):
            with exclusive(self.a.experiment):raise RuntimeError('failure')
        self.assertFalse((self.a.experiment/'.experiment.lock').exists())
    def test_exact_legacy_root_adoption(self):
        self.fake(['prepare','--output',str(self.a.experiment/'inventory')])
        old=self.a.experiment/'training';old.mkdir();cp=old/'checkpoint_latest.pt';cp.write_bytes(b'SYNTHETIC')
        (old/'paused.json').write_text('{}')
        self.run_one();self.assertEqual(self.calls[-1][self.calls[-1].index('--resume')+1],str(cp))
    def test_wrong_legacy_fov_does_not_write_manifest(self):
        self.fake(['prepare','--output',str(self.a.experiment/'inventory')]);(self.a.experiment/'training').mkdir()
        (self.a.experiment/'training/paused.json').write_text('{}')
        changed=copy.deepcopy(self.request);changed['local_cnn']['margin_mm']=30
        with self.assertRaisesRegex(ValueError,'FOV'):run_experiment(self.a,changed,self.fake)
        self.assertFalse((self.a.experiment/'experiment.json').exists())
    def test_missing_saved_checkpoint_not_silent_restart(self):
        cp=self.run_one();cp.unlink();(cp.parent/'checkpoint_timing.jsonl').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'missing'):self.run_one()
    def test_failure_before_first_save_reuses_bound_previous_checkpoint(self):
        first=self.run_one()
        def fail(command):
            Path(command[command.index('--output')+1]).mkdir(parents=True)
            raise RuntimeError('child failed before saving')
        with self.assertRaisesRegex(RuntimeError,'child'):run_experiment(self.a,self.request,fail)
        self.run_one();self.assertEqual(self.calls[-1][self.calls[-1].index('--resume')+1],str(first))
    def test_calibration_failure_retries_without_dropping_output(self):
        def fail(command):
            if 'prepare' in command:self.fake(command)
            else:
                out=Path(command[command.index('--output')+1]);out.mkdir(parents=True);(out/'diagnostic.txt').write_text('fixture')
                raise RuntimeError('calibration failed')
        with self.assertRaisesRegex(RuntimeError,'calibration'):run_experiment(self.a,self.request,fail)
        self.run_one();self.assertNotIn('--resume',self.calls[-1])
        self.assertTrue((self.a.experiment/'attempts/0001/diagnostic.txt').exists())
    def test_path_outside_experiment_rejected(self):
        state={'attempts':[{'output':'../other','resume_from':None}]}
        with self.assertRaisesRegex(ValueError,'escapes'):select_checkpoint(self.a.experiment,state)
    def test_unmanaged_running_experiment_not_adopted(self):
        self.fake(['prepare','--output',str(self.a.experiment/'inventory')]);(self.a.experiment/'training').mkdir()
        with self.assertRaisesRegex(ValueError,'still be running'):self.run_one()
        self.assertFalse((self.a.experiment/'experiment.json').exists())
    def test_completed_run_does_not_create_another_attempt(self):
        def complete(command):
            self.fake(command)
            if 'train' in command:
                out=Path(command[command.index('--output')+1]);(out/'training_complete.json').write_text('{}')
                (out/'checkpoint.pt').write_bytes(b'SYNTHETIC FINAL')
        final=run_experiment(self.a,self.request,complete)
        self.assertEqual(final.name,'checkpoint.pt');count=len(self.calls)
        with patch('torch.load',return_value={'synthetic':True}),patch('l0_local_cnn.artifact.validate') as validate:
            self.assertEqual(self.run_one(),final);validate.assert_called_once()
        self.assertEqual(len(self.calls),count)

if __name__=='__main__':unittest.main()
