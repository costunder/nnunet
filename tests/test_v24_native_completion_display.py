"""CPU DEBUG display regressions; no training, model imports or remote writes."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import watch_v24_gpu_pipeline as pipeline
from tools import watch_v24_native_startup_retry as retry
from tests import test_watch_v24_gpu_pipeline as pipeline_fixtures


def complete_fixture(gpu=6):
    return dict(GPU=gpu, snapshot_fresh=True, active_stage='native', status='COMPLETE',
        gnn=dict(full_training_completed=True),
        scoring=dict(status='SCORING_COMPLETE_NATIVE_STORAGE_PENDING', source_done=642,
            source_total=642, CP_done=82176, CP_total=82176),
        native=dict(status='COMPLETE', native_training_started=False, epochs=250, completed_epochs=250,
            full_native_training=True, full_native_evaluation=True, completion_metadata_verified=True),
        full_native_training=True, full_native_evaluation=True,
        native_pending_reason='DEBUG stale storage pending from preserved old controller')


class NativeCompletionDisplayCPU_DEBUG(unittest.TestCase):
    def test_verified_final_native_screen_has_actual_full250_BEST26_and_CP_completion_for_each_GPU(self):
        for gpu in (4, 5, 6):
            with self.subTest(gpu=gpu):
                value=complete_fixture(gpu)
                before=copy.deepcopy(value)
                shown=pipeline.display(value)
                self.assertEqual(shown.splitlines()[0],f'GPU{gpu} | nnU-Net | COMPLETE')
                self.assertIn('250/250 epoch',shown);self.assertIn('physical batch 2',shown)
                self.assertIn('26/26 | checkpoint_best.pth',shown)
                self.assertIn('642/642 source | 82176/82176 CP',shown)
                self.assertNotIn('SCORING_COMPLETE_NATIVE_STORAGE_PENDING',shown)
                self.assertNotIn('stale storage pending',shown)
                self.assertNotIn('작업 실행 중',shown)
                self.assertEqual(value,before)

    def test_console250_and_full_flags_without_actual_metadata_never_select_completed_native_screen(self):
        mutations=[(None,'snapshot_fresh',False),(None,'active_stage','scoring'),(None,'status','FAILED'),
            (None,'full_native_training',False),(None,'full_native_evaluation',False),
            ('native','status','FAILED'),('native','full_native_training',False),('native','full_native_evaluation',False),
            ('native','completion_metadata_verified',False),('native','completion_metadata_verified',None)]
        for scope,key,value in mutations:
            with self.subTest(scope=scope,key=key,value=value):
                record=complete_fixture()
                target=record if scope is None else record[scope]
                target[key]=value
                self.assertFalse(pipeline.verified_native_completion(record))
                self.assertNotIn('nnU-Net 학습 완료: 250/250',pipeline.display(record))
        for field,value in [('source_done',641),('CP_done',82048)]:
            record=complete_fixture();record['scoring'][field]=value
            self.assertFalse(pipeline.verified_native_completion(record))
            self.assertNotIn('nnU-Net 학습 완료: 250/250',pipeline.display(record))

    def test_CPU_existing_artifacts_completion_labels_no_training_or_evaluation_rerun(self):
        value=complete_fixture()
        value['native']['completion_root']='/DEBUG/fresh_CPU_completion'
        shown=pipeline.display(value)
        self.assertIn('재학습·재평가 없음',shown)
        self.assertFalse(value['native']['native_training_started'])

    def test_real_small_metadata_verifier_selects_completed_screen_only_after_BEST26_admission(self):
        # Reuse DEBUG on-disk metadata fixtures, with the actual source/scoring
        # observer and native_completion validator rather than mocked flags.
        fixture=pipeline_fixtures.CurrentPipelineCPU_DEBUG(methodName='test_new_native_completion_requires_real_full250_and_BEST26_metadata')
        fixture.setUp()
        try:
            fixture.complete_scoring()
            fixture.native(status='COMPLETE',new=True)
            _,best_path=fixture.native_completion()
            with patch.object(pipeline.psutil,'Process',side_effect=fixture.process):
                observed=fixture.viewer.snapshot()
            self.assertTrue(observed['native']['completion_metadata_verified'])
            shown=pipeline.display(observed)
            self.assertIn('GPU4 | nnU-Net | COMPLETE',shown)
            self.assertIn('250/250 epoch',shown);self.assertIn('26/26 | checkpoint_best.pth',shown)
            best=pipeline.read(best_path);best['checkpoint_name']='checkpoint_final.pth'
            best_path.write_text(json.dumps(best),encoding='utf8')
            with patch.object(pipeline.psutil,'Process',side_effect=fixture.process):
                invalid=fixture.viewer.snapshot()
            self.assertEqual(invalid['status'],'READ_ERROR')
            self.assertFalse(pipeline.verified_native_completion(invalid))
            self.assertNotIn('nnU-Net 학습 완료: 250/250',pipeline.display(invalid))
        finally:
            fixture.doCleanups()

    def test_retry_once_prints_completed_native_screen_with_explicit_completion_request(self):
        value=complete_fixture();value['native']['completion_root']='/DEBUG/fresh_CPU_completion'
        stream=io.StringIO()
        with patch.object(retry.direct,'load_renderer',return_value=SimpleNamespace()), \
                patch.object(retry,'RetryViewer',return_value=SimpleNamespace(snapshot=lambda:value)) as observer, \
                patch.object(retry.sys,'stdout',stream):
            retry.main(['--request','/DEBUG/failed/request.json','--completion-request','/DEBUG/completion/request.json',
                '--renderer','/DEBUG/renderer.py','--renderer-sha','a'*64,'--once'])
        self.assertIn('GPU6 | nnU-Net | COMPLETE',stream.getvalue())
        self.assertIn('26/26 | checkpoint_best.pth',stream.getvalue())
        self.assertEqual(observer.call_args.args[2],Path('/DEBUG/completion/request.json'))

    def test_retry_TTY_closes_CP_region_and_bypasses_task_renderer_for_verified_completion(self):
        value=complete_fixture()
        stream=io.StringIO();stream.isatty=lambda:True
        terminal=Mock()
        with patch.object(retry.direct,'load_renderer',return_value=SimpleNamespace()), \
                patch.object(retry,'RetryViewer',return_value=SimpleNamespace(snapshot=lambda:value)), \
                patch.object(retry.ui,'TerminalView',return_value=terminal), \
                patch.object(retry.sys,'stdout',stream), \
                patch.object(retry.time,'sleep',side_effect=KeyboardInterrupt):
            retry.main(['--request','/DEBUG/failed/request.json','--completion-request','/DEBUG/completion/request.json',
                '--renderer','/DEBUG/renderer.py','--renderer-sha','a'*64])
        terminal.close.assert_called_once();terminal.render.assert_not_called()
        self.assertIn('GPU6 | nnU-Net | COMPLETE',stream.getvalue())


if __name__=='__main__':unittest.main()
