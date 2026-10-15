"""CPU DEBUG: intentional checkpoint queues use the existing unchanged UI2."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import watch_basic_cp_RAM512 as view
from tools import watch_basic_cp_ablation as native


class QueueViewerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.root = self.folder/'GPU0_multi5'; self.root.mkdir()
        self.anchor = self.folder/'queue'; self.anchor.mkdir()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(view,'QUEUE_ANCHOR',self.anchor).start()
        view._QUEUE_SHA_CACHE.clear()
        self.old = dict(pid=100,create_time=1.,UID=41833,command=['CPU_DEBUG_original_controller'])
        self.child = dict(pid=200,birth=2.,UID=41833,command=['CPU_DEBUG_original_native'])
        self.watcher = dict(pid=300,create_time=3.,UID=41833,command=['CPU_DEBUG_queue_watcher'])
        self.released = mock.patch.object(view,'_queue_actor_released',side_effect=lambda a:a['pid']!=300).start()
        self.request = dict(root=str(self.root),gpu=0,arm='multi5',controller_identity=self.old,
            memory_budget_revision=512,aggregate_memory_cap_GiB=512,
            effective_native_RSS_limit_GiB=32,effective_shared_bank_RSS_limit_GiB=320)
        self.write(self.root/'request.json',self.request)
        self.manifest = dict(registration={'0':dict(root=str(self.root),arm='multi5',
            controller_identity=self.old,request_sha256=self.sha(self.root/'request.json'))})
        self.write(self.folder/'launch.json',self.manifest)
        self.checkpoints = {}
        for role,epoch in (('LATEST',116),('BEST',99)):
            path=self.folder/(role+'.pth');path.write_bytes(('CPU_DEBUG_'+role).encode())
            self.checkpoints[role]=dict(path=str(path),sha256=self.sha(path),completed_epoch=epoch,
                whole_completed_epoch=True,scaler_present=True)
        self.pause = dict(format='basic_cp_two_selected_queue_paused_v1',
            only_selected_own_actors_signaled=True,other_running_experiments_signaled=False,
            original_checkpoint_bytes_changed=False,released_actors={'0':[self.child,self.old]},
            checkpoint_proofs={'0':self.checkpoints})
        self.write(self.anchor/'selected_pause_proof.json',self.pause)
        self.config = dict(initial_manifest=self.reference(self.folder/'launch.json'),
            selected_pause_proof=self.reference(self.anchor/'selected_pause_proof.json'))
        self.write(self.anchor/'config.json',self.config)
        self.state = dict(format='basic_cp_RAM512_successor_queue_status_v1',
            config_sha256=self.sha(self.anchor/'config.json'),phase='WAITING',actor=self.watcher,
            current_manifest=self.config['initial_manifest'],routes=copy.deepcopy(view.QUEUE_ROUTES),
            GPU0_GPU4_autostart_disabled=True)
        self.write_state()
        self.original = dict(physical_GPU=0,arm='multi5',controller_stage='train',controller_status='FAILED',
            RAM_limit_GiB=96,status='FAILED',stage='nnunet_training',stage_label='nnU-Net 학습',
            error='intentional interrupt reflected by old runner',completed_epochs=115,
            epochs=250,stage_log=str(self.root/'train.log'))

    @staticmethod
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value),encoding='utf8')

    def reference(self,path):return dict(path=str(path),sha256=self.sha(path))

    def write_state(self):self.write(self.anchor/'status.json',self.state)

    def entry(self):
        viewer=SimpleNamespace(main=native.main,read=lambda p:self.request if p.name=='request.json'else{})
        expanded=SimpleNamespace(expanded_snapshot=lambda root:dict(self.original))
        return view.build_entry(expanded,viewer)

    def snapshot(self):return self.entry().__globals__['snapshot'](self.root)

    def test_actual_selected_pause_displays_saved116_and_target_gpu5_in_existing_layout(self):
        entry=self.entry();result=entry.__globals__['snapshot'](self.root)
        self.assertIs(entry.__code__,native.main.__code__)
        self.assertEqual(result['status'],'WAITING')
        self.assertEqual(result['completed_epochs'],116)
        self.assertEqual(result['queued_physical_GPU'],5)
        self.assertEqual(result['controller_stage'],'waiting_GPU_queue_transfer')
        self.assertEqual(result['stage'],'scoring')
        self.assertIn('GPU5',result['waiting_reason'])
        self.assertIsNone(result['error'])
        self.assertEqual(result['RAM_limit_GiB'],32)

    def test_original_request_and_status_files_are_not_modified(self):
        before={p:p.read_bytes()for p in self.folder.rglob('*')if p.is_file()}
        self.snapshot()
        self.assertEqual(before,{p:p.read_bytes()for p in self.folder.rglob('*')if p.is_file()})

    def test_live_original_controller_is_not_presented_as_a_paused_queue(self):
        self.released.side_effect=lambda a:False
        result=self.snapshot()
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['stage'],'nnunet_training')

    def test_live_original_native_child_is_not_hidden(self):
        self.released.side_effect=lambda a:a['pid']==100
        self.assertEqual(self.snapshot()['status'],'FAILED')

    def test_failed_queue_watcher_surfaces_actual_error(self):
        self.state.update(phase='FAILED_REQUIRES_INSPECTION',error='CPU_DEBUG handoff failed')
        self.write_state();result=self.snapshot()
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['error'],'CPU_DEBUG handoff failed')
        self.assertEqual(result['waiting_reason'],'CPU_DEBUG handoff failed')

    def test_dead_waiting_watcher_cannot_be_misreported_as_automatic_resume(self):
        self.released.side_effect=lambda a:True
        result=self.snapshot()
        self.assertEqual(result['status'],'FAILED')
        self.assertIn('대기 실행기가 종료',result['error'])

    def test_expired_allocation_displays_saved_queue_and_need_for_new_allocation(self):
        self.state['phase']='QUEUE_SAVED_ALLOCATION_ENDED';self.write_state()
        result=self.snapshot()
        self.assertEqual(result['status'],'WAITING')
        self.assertIn('새 할당 필요',result['waiting_reason'])

    def test_changed_config_or_original_checkpoint_is_a_visible_error(self):
        self.snapshot()
        Path(self.checkpoints['LATEST']['path']).write_bytes(b'CPU_DEBUG_CHANGED_CHECKPOINT')
        with self.assertRaisesRegex(ValueError,'checkpoint evidence changed'):
            self.snapshot()

    def test_new_current_cohort_does_not_overlay_the_old_root(self):
        self.state['current_manifest']=dict(path=str(self.folder/'new_launch.json'),sha256='f'*64)
        self.write_state()
        self.assertEqual(self.snapshot()['status'],'FAILED')

    def test_different_initial_manifest_row_cannot_borrow_pause_proof(self):
        self.manifest['registration']['0']['root']=str(self.folder/'unrelated')
        self.write(self.folder/'launch.json',self.manifest)
        self.config['initial_manifest']=self.reference(self.folder/'launch.json')
        self.write(self.anchor/'config.json',self.config)
        self.state.update(config_sha256=self.sha(self.anchor/'config.json'),current_manifest=self.config['initial_manifest'])
        self.write_state()
        self.assertEqual(self.snapshot()['status'],'FAILED')

    def test_without_actual_saved_queue_no_old_failure_is_hidden(self):
        (self.anchor/'status.json').unlink()
        self.assertEqual(self.snapshot()['status'],'FAILED')

    def test_fresh_deferred_request_uses_same_generic_layout_instead_of_stale_native_console(self):
        (self.anchor/'status.json').unlink()
        self.original.update(status='WAITING',controller_status='WAITING',
            controller_stage='waiting_GPU_queue_transfer')
        state=dict(reason='GPU6 upstream completion pending',completed_epochs=66,queued_physical_GPU=6)
        viewer=SimpleNamespace(main=native.main,read=lambda p:self.request if p.name=='request.json'else state)
        expanded=SimpleNamespace(expanded_snapshot=lambda root:dict(self.original))
        entry=view.build_entry(expanded,viewer);result=entry.__globals__['snapshot'](self.root)
        self.assertEqual(result['stage'],'scoring')
        self.assertEqual(result['completed_epochs'],66)
        self.assertIn('GPU6',result['waiting_reason'])
        self.assertIs(entry.__code__,native.main.__code__)


if __name__=='__main__':unittest.main()
