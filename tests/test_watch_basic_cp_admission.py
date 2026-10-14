"""Read-only DEBUG viewer acceptance; no training, GPU launch or file mutation."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import watch_basic_cp_admission as adapter
from tools import watch_basic_cp_ablation as viewer
from tools import watch_basic_cp_expanded as expanded
from tools import training_monitor_ui as ui


class AdmissionViewerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.log=self.root/'train.log'
        self.native_bytes=(b'2026-10-14 09:00:00: Epoch 51\n'
            b'2026-10-14 09:01:30: train_loss -0.6\n'
            b'2026-10-14 09:01:35: Epoch time: 95.0 s\n'
            b'2026-10-14 09:01:37: Epoch 52\n')
        self.log.write_bytes(self.native_bytes)
        self.value=dict(GPU='0 / Basic CP multi5',physical_GPU=0,arm='multi5',status='WAITING',
            stage='nnunet_training',stage_label='waiting_GPU',stage_log=str(self.root/'waiting_GPU.log'),
            epochs=250,epoch=None,completed_epochs=0,native_metrics={},reports={},active_U=None,progress=None,
            sources_done=None,sources_total=642,candidates_done=None,candidates_total=642*128,
            cp_input='Basic uniform / 자기 환자 병변 / GNN 미사용',RSS_GiB=.03,
            RSS_source='현재 프로세스 트리',RAM_limit_GiB=96,process_state='sleeping',error=None,
            controller_status='WAITING',controller_stage='waiting_GPU',full_training_completed=False,
            snapshot_fresh=True,full_native_evaluation=False,native_started=False)
        self.entry=adapter.build_entry(expanded,viewer)

    def show(self,phase='waiting_GPU',reason='GPU0 PID123 UID66006 occupied',json_output=False):
        self.value.update(controller_stage=phase,stage_label=phase,stage_log=str(self.root/(phase+'.log')))
        stream=io.StringIO()
        with (patch.object(expanded,'expanded_snapshot',return_value=copy.deepcopy(self.value)),
              patch.object(viewer,'read',return_value=dict(stage=phase,reason=reason)),
              patch.object(viewer.sys,'stdout',stream)):
            self.entry(['--root',str(self.root),'--json'if json_output else'--once'])
        return stream.getvalue()

    def test_waiting_gpu_with_native_metadata_is_existing_ui2_and_actual_reason(self):
        shown=self.show()
        self.assertEqual(tuple(line.partition(': ')[0]for line in shown.splitlines()),ui.CP_ROW_LABELS)
        self.assertIn('UI 2',shown);self.assertIn('GPU 점유 해제 대기',shown)
        self.assertIn('GPU0 PID123 UID66006 occupied',shown)
        self.assertIn('단계 waiting_GPU',shown)
        self.assertNotIn('Epoch 51',shown)

    def test_waiting_ram_and_cuda_debug_are_not_blank_native_logs(self):
        for phase in('waiting_RAM','CUDA_DEBUG'):
            with self.subTest(phase=phase):
                shown=self.show(phase,'exact actual admission reason')
                self.assertIn(adapter.ADMISSION_LABELS[phase],shown)
                self.assertIn('exact actual admission reason',shown)
                self.assertEqual(len(shown.splitlines()),8)

    def test_json_uses_admission_stage_and_controller_phase_without_fabricated_progress(self):
        value=json.loads(self.show(json_output=True))
        self.assertEqual(value['stage'],'scoring')
        self.assertEqual(value['controller_stage'],'waiting_GPU')
        self.assertEqual(value['waiting_reason'],'GPU0 PID123 UID66006 occupied')
        self.assertIsNone(value['sources_done']);self.assertIsNone(value['candidates_done'])

    def test_running_native_console_keeps_exact_original_bytes_and_no_new_header(self):
        value=copy.deepcopy(self.value)
        value.update(stage='nnunet_training',controller_stage='train',stage_log=str(self.log),status='RUNNING')
        stream=io.StringIO();before=self.log.read_bytes()
        with (patch.object(expanded,'expanded_snapshot',return_value=value),patch.object(viewer.sys,'stdout',stream)):
            self.entry(['--root',str(self.root),'--once'])
        self.assertEqual(stream.getvalue().encode(),self.native_bytes)
        self.assertEqual(self.log.read_bytes(),before)
        self.assertNotIn('UI 2',stream.getvalue())

    def test_running_snapshot_json_and_pinned_viewer_globals_remain_unchanged(self):
        value=copy.deepcopy(self.value);value.update(controller_stage='train',stage_log=str(self.log))
        before_snapshot=viewer.snapshot;before_model=viewer.preparation_model;stream=io.StringIO()
        with (patch.object(expanded,'expanded_snapshot',return_value=value),patch.object(viewer.sys,'stdout',stream)):
            self.entry(['--root',str(self.root),'--json'])
        self.assertEqual(json.loads(stream.getvalue()),value)
        self.assertIs(viewer.snapshot,before_snapshot);self.assertIs(viewer.preparation_model,before_model)

    def test_explicit_manifest_export_mismatch_is_rejected_before_import(self):
        manifest=self.root/'launch.json';manifest.write_text(json.dumps(dict(code=str(self.root),registration={'0':dict(root=str(self.root))})))
        with patch.object(adapter,'load_pinned_viewers')as load,self.assertRaisesRegex(ValueError,'manifest pinned export'):
            adapter.main(['--manifest',str(manifest),'--code-root',str(Path(__file__).resolve().parents[1]),'--gpu','0','--once'])
        load.assert_not_called()

    def test_cli_routes_gpu_once_and_json_to_unchanged_pinned_base_entry(self):
        code=Path(__file__).resolve().parents[1];manifest=self.root/'launch.json'
        manifest.write_text(json.dumps(dict(code=str(code),registration={'0':dict(root=str(self.root))})))
        stream=io.StringIO()
        with (patch.object(adapter,'load_pinned_viewers',return_value=(expanded,viewer)),
              patch.object(expanded,'expanded_snapshot',return_value=copy.deepcopy(self.value)),
              patch.object(viewer,'read',return_value=dict(reason='actual GPU wait')),
              patch.object(viewer.sys,'stdout',stream)):
            adapter.main(['--manifest',str(manifest),'--code-root',str(code),'--gpu','0','--json'])
        self.assertEqual(json.loads(stream.getvalue())['waiting_reason'],'actual GPU wait')


if __name__=='__main__':unittest.main()
