"""CPU UNIT checkpoint/lineage guards; no CT, production model or CUDA training."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import torch

from hiercp_v1x import v24_training_continuation as original
from hiercp_v1x.u_bridge_training import capture_rng,digest
from hiercp_v1x.v24_targets import new_curriculum
from tools import run_v24_all_p as cli
from tools import run_v24_gpu4_exact_continuation as overlay

ROOT=Path(__file__).resolve().parents[1]


class CheckpointBoundaryLogsUnit(unittest.TestCase):
    def state(self,**values):
        return dict(epoch=1,updates=1,phase='training',train_position=4,evaluation_position=0,**values)

    def lines(self,rows):return b''.join(json.dumps(row).encode()+b'\n' for row in rows)

    def test_unsaved_second_update_is_archived_but_never_active_progress(self):
        first=dict(status='OPTIMIZER_UPDATED',epoch=1,update=1,patients=4)
        second=dict(status='OPTIMIZER_UPDATED',epoch=1,update=2,patients=4)
        raw=self.lines([first,second])+b'{"epoch":1,"partial":'
        result,proof=overlay.derive_checkpoint_log(raw,'update_timing.jsonl',self.state())
        self.assertEqual(result,self.lines([first]))
        self.assertEqual(proof['rows_kept'],1);self.assertEqual(proof['rows_excluded'],1)
        self.assertEqual(proof['partial_rows_archived'],1)
        self.assertEqual(proof['boundary_updates'],1)

    def test_overflow_at_unsaved_patient_cursor_is_not_a_durable_attempt(self):
        rows=[dict(status='AMP_OVERFLOW_RETRY_SAME_INPUT',epoch=1,cursor=0),
              dict(status='OPTIMIZER_UPDATED',epoch=1,update=1,patients=4),
              dict(status='AMP_OVERFLOW_RETRY_SAME_INPUT',epoch=1,cursor=4)]
        result,proof=overlay.derive_checkpoint_log(self.lines(rows),'update_timing.jsonl',self.state())
        self.assertEqual(result,self.lines(rows[:2]));self.assertEqual(proof['rows_excluded'],1)

    def test_initial_full_validation_is_retained_future_stage_metrics_are_excluded(self):
        rows=[dict(epoch=1,phase='initial_full_validation',patients=4),
              dict(epoch=1,phase='initial_full_validation',patients=3),
              dict(epoch=1,phase='train_probe',patients=4)]
        result,_=overlay.derive_checkpoint_log(self.lines(rows),'validation_timing.jsonl',self.state())
        self.assertEqual(result,self.lines(rows[:2]))

    def test_checkpoint_validation_position_limits_partially_finished_batch_metrics(self):
        state=self.state();state.update(phase='train_probe',evaluation_position=4)
        rows=[dict(epoch=1,phase='initial_full_validation',patients=4),
              dict(epoch=1,phase='train_probe',patients=4),
              dict(epoch=1,phase='train_probe',patients=4)]
        result,proof=overlay.derive_checkpoint_log(self.lines(rows),'validation_timing.jsonl',state)
        self.assertEqual(result,self.lines(rows[:2]));self.assertEqual(proof['rows_excluded'],1)

    def test_old_failure_and_incomplete_epoch_are_not_new_run_events(self):
        raw=self.lines([dict(epoch=1,error='CPU_UNIT_quota_failure')])
        for name in ('failures.jsonl','invocations.jsonl','curve.jsonl'):
            with self.subTest(name=name):
                result,proof=overlay.derive_checkpoint_log(raw,name,self.state())
                self.assertEqual(result,b'');self.assertEqual(proof['rows_excluded'],1)

    def test_unknown_or_corrupt_complete_record_stops_explicitly(self):
        with self.assertRaises(ValueError):
            overlay.derive_checkpoint_log(b'{bad}\n','update_timing.jsonl',self.state())
        with self.assertRaisesRegex(ValueError,'Unknown original update event'):
            overlay.derive_checkpoint_log(self.lines([dict(epoch=1,status='CPU_UNIT_UNKNOWN')]),'update_timing.jsonl',self.state())
        with self.assertRaisesRegex(ValueError,'Unknown original training log'):
            overlay.derive_checkpoint_log(b'','CPU_UNIT_unknown.jsonl',self.state())


class ExactCheckpointContinuationUnit(unittest.TestCase):
    """Synthetic metadata and tiny owned checkpoint exercise actual serialization.

    The production source/model admission is replaced by a labeled UNIT fixture;
    original checkpoint hashing, copying, six-state guards and cursor validation
    run unchanged. This is not a production-model or actual-data validation.
    """
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='GPU4_exact_CPU_UNIT_',dir=ROOT/'outputs')
        self.directory=Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to((ROOT/'outputs').resolve()))
        self.source=self.directory/'CPU_UNIT_original';self.source.mkdir();(self.source/'training').mkdir()
        self.output=self.directory/'CPU_UNIT_fresh';self.job=self.directory/'CPU_UNIT_failed_job';self.job.mkdir()
        self.inventory=self.directory/'CPU_UNIT_inventory.json';self.inventory.write_text('{"CPU_UNIT_metadata_only":true}')
        self.asset=self.directory/'CPU_UNIT_no_model_asset';self.asset.write_bytes(b'CPU_UNIT_NOT_STU_WEIGHTS')
        self.args=SimpleNamespace(mode='train',gpu=4,resume=None,cpu_affinity=[0,1,2,3],
            config=ROOT/'config/v24_gpu4_STUNetS_all_U_GT_blind.json',output=self.output,
            native_experiment=self.directory/'CPU_UNIT_native',input_cache=self.directory/'CPU_UNIT_cache',
            inventory=self.inventory,stunet_checkpoint=self.asset)
        self.config=json.loads(self.args.config.read_text());self.request=cli.request(self.args,self.config)
        self.json(self.source/'request.json',self.request)
        self.calibration=dict(CPU_UNIT_metadata_only=True,request_sha256=self.request['request_sha256'],
            initial_state_sha256='CPU_UNIT_initial',selected_physical_patient_batch=4,selected_physical_candidate_batch=64)
        self.json(self.source/'calibration.json',self.calibration)
        bound_config=copy.deepcopy(self.config);bound_config['v24_runtime']['batch_calibration']=self.calibration
        self.train=[f'CPU_UNIT_train_{index:02}' for index in range(65)]
        self.val=[f'CPU_UNIT_val_{index:02}' for index in range(21)]
        binding=dict(identity=self.request,config=bound_config,train_cases=self.train,val_cases=self.val,
            physical_patient_batch=4,physical_candidate_batch=64,epochs=40,initial_state_sha256='CPU_UNIT_initial',
            model_contract=dict(CPU_UNIT_metadata_only=True,not_a_production_model=True))
        self.owner=dict(binding=binding,identity_sha256=digest(binding))
        self.json(self.source/'training/training_identity.json',self.owner)
        self.state=dict(epoch=1,phase='training',status='RUNNING',updates=1,attempts=1,overflows=0,
            history=[],best=None,curriculum=new_curriculum(self.config['v24_runtime']['curriculum']),
            train_order=self.train,train_position=4,train_rows=[dict(case_id=case) for case in self.train[:4]],
            evaluation_position=0,evaluation_rows=[],epoch_start_updates=0,connected=['CPU_UNIT_weight'])
        model={'CPU_UNIT_weight':torch.arange(7,dtype=torch.float32)}
        self.saved=dict(format='v24_GT_blind_all_P_patient_balanced_training_v1',
            identity_sha256=self.owner['identity_sha256'],model=model,
            optimizer=dict(state={0:dict(step=torch.tensor(1.),exp_avg=torch.ones(7),exp_avg_sq=torch.ones(7))},
                param_groups=[dict(params=[0],lr=1e-4)]),scheduler=dict(last_epoch=0),
            scaler=dict(scale=65536.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=1),
            rank_rng=[capture_rng()],shuffle_generator=torch.Generator().manual_seed(2045).get_state(),state=self.state)
        self.saved['content_sha256']=digest(self.saved)
        torch.save(self.saved,self.source/'training/checkpoint_latest.pt')
        _,latest=original._checkpoint(self.source/'training/checkpoint_latest.pt',self.owner)
        self.proof=dict(source_output=str(self.source),source_code=str(ROOT),source_commit=overlay.SOURCE_COMMIT,
            latest=latest,BEST=None,source_job=dict(CPU_UNIT_metadata_only=True))
        self.raw_log=b''.join(json.dumps(dict(status='OPTIMIZER_UPDATED',epoch=1,update=update,patients=4)).encode()+b'\n'
                              for update in (1,2))
        (self.source/'training/update_timing.jsonl').write_bytes(self.raw_log)
        (self.source/'training/failures.jsonl').write_bytes(b'{"CPU_UNIT":true,"epoch":1,"updates":2}\n')
        self.partial=self.source/'training/checkpoint_latest.pt.CPU_UNIT_partial.tmp'
        self.partial.write_bytes(b'CPU_UNIT_PARTIAL_NOT_CHECKPOINT')
        self.execution=dict(execution_code=str(ROOT),execution_commit='CPU_UNIT_uncommitted_test_fixture',
            execution_files_sha256={'CPU_UNIT_fixture': 'CPU_UNIT_not_deployed'})

    def json(self,path,value):path.write_text(json.dumps(value),encoding='utf8')

    def context(self):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(overlay,'owned',side_effect=lambda path,**kwargs:Path(path)))
        stack.enter_context(patch.object(overlay,'execution_identity',return_value=self.execution))
        stack.enter_context(patch.object(overlay,'source_snapshot',side_effect=lambda *args:(copy.deepcopy(self.proof),copy.deepcopy(self.saved))))
        stack.enter_context(patch.object(overlay.shutil,'disk_usage',return_value=SimpleNamespace(free=20*2**30)))
        return stack

    def prepare(self):
        with self.context():return overlay.prepare(self.args,self.source,ROOT,self.job,cli,original)

    def tearDown(self):self.temporary.cleanup()

    def test_actual_checkpoint_bytes_six_states_and_cursor_survive_new_namespace_copy(self):
        before=(self.source/'training/checkpoint_latest.pt').read_bytes()
        original_partial=self.partial.read_bytes();result=self.prepare()
        self.assertEqual((self.output/'training/checkpoint_latest.pt').read_bytes(),before)
        self.assertEqual((self.source/'training/checkpoint_latest.pt').read_bytes(),before)
        _,latest=original._checkpoint(self.output/'training/checkpoint_latest.pt',self.owner)
        self.assertEqual(latest['numerical_state_sha256'],self.proof['latest']['numerical_state_sha256'])
        self.assertEqual(latest['state_sha256'],self.proof['latest']['state_sha256'])
        self.assertEqual(latest['train_position'],4);self.assertEqual(latest['updates'],1)
        self.assertEqual(self.partial.read_bytes(),original_partial)
        self.assertFalse((self.output/'training'/self.partial.name).exists())
        self.assertEqual(result['production_optimizer_updates_performed'],0)

    def test_unsaved_update2_full_log_retained_as_lineage_active_boundary1_and_truthful_viewer(self):
        self.prepare()
        lineage=self.output/'source_lineage/training/update_timing.jsonl'
        self.assertEqual(lineage.read_bytes(),self.raw_log)
        self.assertEqual((self.source/'training/update_timing.jsonl').read_bytes(),self.raw_log)
        active=self.output/'training/update_timing.jsonl'
        self.assertEqual(len(active.read_bytes().splitlines()),1)
        self.assertEqual(json.loads(active.read_bytes())['update'],1)
        cursor=json.loads((self.output/'training_continuation.json').read_text())
        self.assertEqual(cursor['source']['latest']['updates'],1)
        self.assertEqual(cursor['copied_files']['training/update_timing.jsonl']['bytes'],active.stat().st_size)
        self.assertEqual((self.output/'training/failures.jsonl').read_bytes(),b'')
        self.assertTrue((self.output/'source_lineage/training/failures.jsonl').stat().st_size>0)

    def test_prepared_state_admitted_but_tampered_canonical_file_is_rejected(self):
        self.prepare()
        with self.context():
            result=overlay.admit(self.args,self.source,ROOT,self.job,cli,original)
            self.assertEqual(result['status'],'PREPARED_EXACT_SAVED_STATE')
            path=self.output/'calibration.json';path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'canonical file changed'):
                overlay.admit(self.args,self.source,ROOT,self.job,cli,original)

    def test_no_existing_output_or_external_resume_or_changed_scientific_cache(self):
        self.output.mkdir()
        with self.context():
            with self.assertRaisesRegex(FileExistsError,'Fresh disjoint'):
                overlay.prepare(self.args,self.source,ROOT,self.job,cli,original)
        with patch.object(overlay,'owned',side_effect=lambda path,**kwargs:Path(path)):
            changed=copy.copy(self.args);changed.resume=Path('CPU_UNIT_external.pt')
            with self.assertRaisesRegex(ValueError,'externally supplied resume'):
                overlay.validate_args(changed,self.source,cli)
            changed=copy.copy(self.args);changed.input_cache=self.directory/'CPU_UNIT_different_cache'
            with self.assertRaisesRegex(ValueError,'Scientific request'):
                overlay.validate_args(changed,self.source,cli)

    def test_atomic_publication_preserves_existing_destination_and_cleans_private_temp(self):
        path=self.directory/'CPU_UNIT_existing.json';path.write_bytes(b'CPU_UNIT_OLD_BYTES')
        with self.assertRaises(FileExistsError):overlay.atomic_bytes(path,b'CPU_UNIT_NEW_BYTES')
        self.assertEqual(path.read_bytes(),b'CPU_UNIT_OLD_BYTES')
        self.assertFalse(list(self.directory.glob('*.GPU4_exact_*.tmp')))
        with self.assertRaises(ValueError):overlay.atomic_json(self.directory/'CPU_UNIT_NaN.json',dict(value=float('nan')))
        self.assertFalse((self.directory/'CPU_UNIT_NaN.json').exists())

    def test_original_failed_job_must_have_absent_worker_and_actual_quota_log(self):
        request=dict(GPU=4,code=str(ROOT),production_output=str(self.source))
        status=dict(status='FAILED',stage='train',request=request,
            worker_pid=123456,worker_create_time=1.,child_pid=123457,child_create_time=2.)
        self.json(self.job/'status.json',status);self.json(self.job/'request.json',request)
        (self.job/'train.log').write_bytes(b'CPU_UNIT_LOG [Errno 122] Disk quota exceeded\n')
        with patch.object(overlay,'owned',side_effect=lambda path,**kwargs:Path(path)):
            with patch('psutil.pid_exists',return_value=False):
                proof=overlay.failed_job(self.job,self.source,ROOT)
                self.assertTrue(proof['quota_failure_verified'])
                self.assertTrue(proof['processes']['worker']['absent'])
            with patch('psutil.pid_exists',return_value=True):
                with self.assertRaisesRegex(ValueError,'never signal or relaunch alongside'):
                    overlay.failed_job(self.job,self.source,ROOT)

    def test_other_scientific_commit_is_rejected_before_any_module_import_or_path_change(self):
        before=list(__import__('sys').path)
        with patch.object(overlay,'owned',side_effect=lambda path,**kwargs:Path(path)):
            with patch.object(overlay.subprocess,'check_output',return_value='f'*40):
                with self.assertRaisesRegex(ValueError,'Exact original GPU4 science commit'):
                    overlay.load_science(ROOT)
        self.assertEqual(__import__('sys').path,before)


if __name__=='__main__':unittest.main()
