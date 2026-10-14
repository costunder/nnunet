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

    def show(self,phase='waiting_GPU',reason='GPU0 PID123 UID66006 occupied',json_output=False,details=None):
        self.value.update(controller_stage=phase,stage_label=phase,stage_log=str(self.root/(phase+'.log')))
        stream=io.StringIO()
        state=dict(stage=phase,reason=reason);state.update(details or {})
        with (patch.object(expanded,'expanded_snapshot',return_value=copy.deepcopy(self.value)),
              patch.object(viewer,'read',return_value=state),
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

    def test_gpu_literal_observation_is_concise_and_full_detail_remains_in_json(self):
        reason=str(dict(occupied_GPU_compute_actors=[dict(pid=123,GPU_UUID='GPU-exact'),dict(pid=456)],
                        group_controller_observations={'exact_detail':'x'*8000}))
        shown=self.show(reason=reason)
        self.assertIn('GPU0 점유 중: PID 123, 456 | 점유 해제 대기',shown)
        self.assertNotIn('group_controller_observations',shown)
        self.assertLess(len(shown),1500)
        value=json.loads(self.show(reason=reason,json_output=True))
        self.assertEqual(value['waiting_reason_detail'],reason)
        self.assertEqual(value['waiting_reason'],'GPU0 점유 중: PID 123, 456 | 점유 해제 대기')

    def test_ram_structured_measurement_shows_actual_capacity_and_preserves_error(self):
        observation=dict(host_available_RAM_bytes=790*2**30,group_process_tree_RSS_bytes=5*2**30,
                         required_group_and_cache_capacity_bytes=800*2**30)
        reason='Actual transient RAM admission failure: '+json.dumps(observation)
        shown=self.show('waiting_RAM',reason,details={'RAM_admission':observation})
        self.assertIn('여유 790.00GiB + 실험 RSS 5.00GiB = 795.00GiB | 필요 800.00GiB',shown)
        value=json.loads(self.show('waiting_RAM',reason,json_output=True,details={'RAM_admission':observation}))
        self.assertEqual(value['waiting_reason_detail'],reason)
        # The typed error itself is sufficient if an older status lacks RAM_admission.
        self.assertEqual(adapter.concise_reason('waiting_RAM',dict(reason=reason),0),value['waiting_reason'])

    def test_unrecognized_or_unsafe_reason_is_preserved_without_execution(self):
        for reason in ('psutil.NoSuchProcess(pid=3622007)',
                       "__import__('os').system('must never execute')",
                       str(dict(occupied_GPU_compute_actors=[dict(pid='123')]))):
            with self.subTest(reason=reason):
                self.assertEqual(adapter.concise_reason('waiting_GPU',dict(reason=reason),0),reason)
        reason='Actual unrecognized RAM error'
        self.assertEqual(adapter.concise_reason('waiting_RAM',dict(reason=reason,RAM_admission={
            'host_available_RAM_bytes':-1,'group_process_tree_RSS_bytes':0,
            'required_group_and_cache_capacity_bytes':800*2**30}),0),reason)

    def test_empty_gpu_actor_list_reports_actual_vram_instead_of_invented_actor(self):
        reason=str(dict(occupied_GPU_compute_actors=[],GPU_free_MiB=31.5*1024))
        self.assertEqual(adapter.concise_reason('waiting_GPU',dict(reason=reason),4),
                         'GPU4 VRAM 여유 31.50GiB | 필요 40.00GiB')

    def test_active1024_progress_is_visible_while_final2048_counters_stay_unchanged(self):
        banks=self.root/'banks';banks.mkdir()
        progress=dict(active_candidate_count=1024,source_total=642,
                      prefix_completed={'512':642,'1024':219,'2048':0})
        (banks/'generation_progress.json').write_text(json.dumps(progress))
        value=copy.deepcopy(self.value)
        value.update(physical_GPU=5,GPU='5 / Basic CP candidates2048',arm='candidates2048',stage='scoring',
                     controller_stage='candidate_bank',controller_status='RUNNING',status='RUNNING',
                     sources_done=0,candidates_done=0,candidates_total=642*2048,
                     stage_label='CP 위치 후보 2048개 준비')
        request=dict(shared_banks=str(banks),candidates_per_source=2048)
        for flag in('--once','--json'):
            stream=io.StringIO()
            with(patch.object(expanded,'expanded_snapshot',return_value=copy.deepcopy(value)),
                 patch.object(viewer,'read',return_value=request),patch.object(viewer.sys,'stdout',stream)):
                self.entry(['--root',str(self.root),flag])
            shown=stream.getvalue()
            self.assertIn('1024 후보 준비 219/642 source; 2048 후보 준비 0/642 source; nnU-Net 시작 전',shown)
            if flag=='--once':self.assertEqual(tuple(line.partition(': ')[0]for line in shown.splitlines()),ui.CP_ROW_LABELS)
            else:
                actual=json.loads(shown)
                self.assertEqual(actual['sources_done'],0);self.assertEqual(actual['candidates_done'],0)
                self.assertEqual(actual['candidates_total'],642*2048)
                self.assertEqual(actual['waiting_reason_detail'],progress)

    def test_missing_malformed_or_unknown_candidate_progress_is_explicit_without_fake_zero(self):
        path=self.root/'generation_progress.json'
        reason,detail=adapter.candidate_progress_reason(path,2048)
        self.assertIn('없음',reason);self.assertIsNone(detail);self.assertNotIn('0/642',reason)
        path.write_text('{"active_candidate_count":')
        reason,detail=adapter.candidate_progress_reason(path,2048)
        self.assertIn('해석 오류',reason);self.assertIsNone(detail);self.assertNotIn('0/642',reason)
        for progress in (dict(active_candidate_count=1024,source_total=642,prefix_completed={'2048':0}),
                         dict(active_candidate_count=1024,source_total=642,prefix_completed={'1024':True,'2048':0}),
                         dict(active_candidate_count=1024,source_total=642,prefix_completed={'1024':643,'2048':0})):
            path.write_text(json.dumps(progress));reason,detail=adapter.candidate_progress_reason(path,2048)
            self.assertIn('확인 필요',reason);self.assertEqual(detail,progress);self.assertNotIn('0/642',reason)

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
