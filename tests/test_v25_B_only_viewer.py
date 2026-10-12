"""Debug-only read observer contracts; no training, CUDA, signals or writes to prior results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import watch_v25_B_only as viewer


class BOnlyViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent); self.addCleanup(self.tmp.cleanup)
        self.group = Path(self.tmp.name).absolute()
        self.transition = self.group/'B_only_user_transition_debug'
        self.transition.mkdir()
        self.authorization = self.transition/'request.json'
        self.route = self.transition/'B_viewer_route.json'
        self.old_seal = self.group/'ALL_recovery_debug.json'
        self.old = dict(format=viewer.previous.FORMAT, selected_arms=list(viewer.previous.GPUS),
            group_root=str(self.group), scientific_configuration_unchanged=True, no_process_signaled=True, slots={})
        self.actors = {}
        for arm, gpu in viewer.previous.GPUS.items():
            monitor = self.group/arm/'monitor/ALL_recovery_debug/status.json'
            monitor.parent.mkdir(parents=True)
            self.put(monitor.parent.parent/'status.json', {'status':'FAILED'})
            self.old['slots'][arm] = dict(arm=arm, gpu=gpu, monitor_path=str(monitor))
            self.actors[arm] = dict(pid=1000+gpu, create_time=100., command=['python',arm], UID=41833)
            (self.group/arm/'training').mkdir()
            (self.group/arm/'training/checkpoint_latest.pt').write_bytes(b'debug checkpoint only')
        self.old_sha = self.put(self.old_seal, self.old)
        self.auth = dict(group_root=str(self.group), original_seal_path=str(self.old_seal), actors=self.actors,
            user_instruction='GPU5 only; pause ACD', GNN_only_early_stop_authorized=True,
            CP_sources=642, CP_candidates_per_source=128, native_epochs=250, evaluation_cases=26,
            evaluation_checkpoint='checkpoint_best.pth', B_finish_epoch=13)
        self.auth_sha = self.put(self.authorization, self.auth)
        observer = patch.object(viewer.psutil, 'Process', side_effect=viewer.psutil.NoSuchProcess(999999))
        observer.start(); self.addCleanup(observer.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(value).encode(); path.write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    def proof(self, path, value):
        return dict(path=str(path), sha256=self.put(path, value))

    def context(self):
        return viewer.Transition(self.group, self.old_seal, self.old_sha,
            self.authorization, self.auth_sha, self.route)

    def invocation(self, arm='A', **changes):
        row = dict(format=viewer.original.GNN_FORMAT, status='PAUSED', full_training=False,
            debug=False, actual_CUDA=True, connected_parameter_tensors=537, expected_parameter_tensors=537,
            completed_epochs=13 if arm == 'B' else 10, optimizer_updates=221 if arm == 'B' else 174,
            active_u=64, checkpoint=str(self.group/arm/'training/checkpoint_latest.pt'))
        row.update(changes)
        (self.group/arm/'training/invocations.jsonl').write_text(json.dumps(row)+'\n', encoding='utf8')
        return row

    def early(self):
        sealpath = self.transition/'execution/seal.json'
        best = self.group/'B/training/checkpoint_best.pt'; best.write_bytes(b'debug BEST only')
        bestsha = hashlib.sha256(best.read_bytes()).hexdigest()
        release = dict(format=viewer.RELEASE_FORMAT, status='PAUSED_ACD_B_READY_FOR_OWN_BEST',
            selected_downstream_arms=['B'], paused_arms=list(viewer.PAUSED_ARMS),
            user_instruction=self.auth['user_instruction'], old_controller_released=True,
            registered_stage_actors_released=True, all_GNN_children_stopped_cooperatively=True,
            states={arm:dict(status='PAUSED', latest_path=str(self.group/arm/'training/checkpoint_latest.pt'),
                completed_epochs=13 if arm=='B' else 10, updates=221 if arm=='B' else 174, epoch=14 if arm=='B' else 11)
                for arm in viewer.previous.GPUS})
        sourceauth = dict(format='v25_user_authorized_B_epoch_boundary_stop_v1', arm='B', physical_GPU=5,
            explicit_user_authorization=True, target_completed_epochs=13, native_epochs=250,
            native_evaluation_cases=26, native_evaluation_checkpoint='checkpoint_best.pth')
        sourceproof = self.proof(self.transition/'science_authorization.json', sourceauth)
        science = dict(source_gnn_output=str(self.group/'B'), cp_output=str(self.transition/'execution/cp_bank'),
            authorization_request=sourceproof)
        seal = dict(format=viewer.EARLY_FORMAT, selected_arms=['B'], group_root=str(self.group),
            authorization_request=dict(path=str(self.authorization),sha256=self.auth_sha),
            previous_all_seal=dict(path=str(self.old_seal),sha256=self.old_sha),
            GNN_early_stop_authorized=True, scientific_configuration_unchanged=True,
            processes_signaled_by_this_adapter=False,
            slot=dict(arm='B',gpu=5,CPU_affinity=list(range(42,48)),
                monitor_path=str(self.transition/'execution/monitor/status.json')),
            scientific_request=self.proof(self.transition/'science.json',science),
            release_receipt=self.proof(self.transition/'release.json',release),
            preserved_files={str(best):bestsha})
        self.put(self.route, dict(format=viewer.ROUTE_FORMAT,
            authorization_request=seal['authorization_request'], early_seal=self.proof(sealpath,seal)))
        return self.context(), seal, sourceproof

    def cp(self):
        context, seal, sourceproof = self.early(); early=context.early(); root=early['cp_root']
        pin = dict(format=viewer.PIN_FORMAT,arm='B',physical_GPU=5,full_training=False,debug=False,
            completed_epochs=13,optimizer_updates=221,original_planned_epochs=40,
            original_planned_optimizer_updates=680,source_completion_kind='USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP',
            selected_by='fixed_full128_validation_only',checkpoint=str(self.group/'B/training/checkpoint_best.pt'),
            checkpoint_sha256=next(iter(seal['preserved_files'].values())),
            original_patient_and_full128_history_verified=True,source_checkpoint_written=False,
            source_model_optimizer_RNG_rewritten=False,authorization=sourceproof)
        request=dict(format=viewer.original.CP_REQUEST,best_provenance=pin)
        requestsha=self.put(root/'request.json',request)
        progress=dict(format=viewer.original.CP_PROGRESS,arm='B',request_sha256=requestsha,
            total_sources=642,total_positions=82176,completed_sources=1,completed_positions=128,status='RUNNING')
        self.put(root/'progress.json',progress)
        return context,early,pin,progress

    def test_authorization_checksum_tamper_rejected(self):
        self.authorization.write_bytes(self.authorization.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.context()

    def test_old_public_and_canonical_files_remain_byte_identical(self):
        before={str(path):path.read_bytes()for path in self.group.rglob('*')if path.is_file()}
        self.context().early()
        self.assertEqual(before,{str(path):path.read_bytes()for path in self.group.rglob('*')if path.is_file()})

    def test_pending_request_is_not_a_false_pause(self):
        with patch.object(viewer.original,'observe_actor',return_value=({'process_status':'running'},None)):
            result=self.context().pause('A')
        self.assertEqual(result['status'],'PAUSE_REQUESTED_PENDING')
        self.assertFalse(result['process_released'])

    def test_saved_pause_with_live_actor_reports_closing(self):
        self.invocation()
        with patch.object(viewer.original,'observe_actor',return_value=({'process_status':'running'},None)):
            result=self.context().pause('A')
        self.assertEqual(result['status'],'PAUSED_USER_REQUEST_CLOSING')
        self.assertFalse(result['process_released'])

    def test_saved_pause_and_released_actor_reports_paused(self):
        self.invocation()
        with patch.object(viewer.original,'observe_actor',return_value=(None,'EXITED_WITHOUT_COMPLETION')):
            result=self.context().pause('A')
        self.assertEqual(result['status'],'PAUSED_USER_REQUEST')
        self.assertTrue(result['process_released'])

    def test_B_saved_pause_waits_for_published_handoff(self):
        self.invocation('B')
        with patch.object(viewer.original,'observe_actor',return_value=(None,'EXITED_WITHOUT_COMPLETION')):
            result=self.context().pause('B')
        self.assertEqual(result['status'],'WAITING_FOR_BEST_HANDOFF')

    def test_partial_pause_cannot_claim_full_training(self):
        self.invocation(full_training=True)
        with patch.object(viewer.original,'observe_actor',return_value=(None,None)):
            with self.assertRaises(ValueError):self.context().pause('A')

    def test_actor_identity_reuse_is_not_hidden_by_paused_invocation(self):
        self.invocation()
        actual=dict(self.actors['A'],create_time=200.)
        with patch.object(viewer.original,'actor_snapshot',return_value=actual):
            with self.assertRaisesRegex(ValueError,'PID was reused'):self.context().pause('A')

    def test_reused_zombie_PID_is_rejected(self):
        process=Mock();process.create_time.return_value=200.;process.uids.return_value.real=41833
        process.status.return_value=viewer.psutil.STATUS_ZOMBIE
        with patch.object(viewer.original,'observe_actor',return_value=(None,None)),patch.object(
                viewer.psutil,'Process',return_value=process,side_effect=None):
            with self.assertRaisesRegex(ValueError,'PID was reused'):
                viewer.observe_predecessor(self.actors['A'],self.authorization)

    def test_zombie_with_live_owned_workers_is_closing(self):
        process=Mock();process.create_time.return_value=100.;process.uids.return_value.real=41833
        process.status.return_value=viewer.psutil.STATUS_ZOMBIE
        child=Mock();child.status.return_value='sleeping';child.uids.return_value.real=41833
        child.memory_info.return_value.rss=4096;process.children.return_value=[child]
        with patch.object(viewer.original,'observe_actor',return_value=(None,None)),patch.object(
                viewer.psutil,'Process',return_value=process,side_effect=None):
            actual,_=viewer.observe_predecessor(self.actors['A'],self.authorization)
        self.assertIsNotNone(actual);self.assertEqual(actual['RSS_bytes'],4096)

    def test_exact_zombie_without_workers_is_released(self):
        process=Mock();process.create_time.return_value=100.;process.uids.return_value.real=41833
        process.status.return_value=viewer.psutil.STATUS_ZOMBIE;process.children.return_value=[]
        with patch.object(viewer.original,'observe_actor',return_value=(None,None)),patch.object(
                viewer.psutil,'Process',return_value=process,side_effect=None):
            actual,_=viewer.observe_predecessor(self.actors['A'],self.authorization)
        self.assertIsNone(actual)

    def test_canonical_monitor_cannot_be_new_B_route(self):
        context,seal,_=self.early()
        seal['slot']['monitor_path']=str(self.group/'B/monitor/status.json')
        route=json.loads(self.route.read_bytes());route['early_seal']=self.proof(Path(route['early_seal']['path']),seal)
        self.put(self.route,route)
        with self.assertRaises(ValueError):context.early()

    def test_incomplete_epoch_release_rejected(self):
        context,seal,_=self.early()
        release=json.loads(Path(seal['release_receipt']['path']).read_bytes())
        release['states']['B']['completed_epochs']=12
        with self.assertRaises(ValueError):context.validate_release(release)

    def test_truthful_partial_BEST_full642_progress_is_accepted(self):
        context,early,_,_=self.cp()
        with patch.object(viewer.original,'observe_actor',return_value=({'RSS_bytes':123,'process_status':'running'},None)):
            result=viewer.Viewer(self.group,'B',context).early_scoring(early,[['python']],{})
        self.assertEqual((result['source_done'],result['CP_done'],result['CP_total']),(1,128,82176))
        self.assertFalse(result['GNN_full_training']);self.assertEqual(result['GNN_completed_epochs'],13)

    def test_fabricated_full40_pin_is_rejected(self):
        context,early,pin,progress=self.cp();pin.update(full_training=True,completed_epochs=40,optimizer_updates=680)
        progress['request_sha256']=self.put(early['cp_root']/'request.json',dict(format=viewer.original.CP_REQUEST,best_provenance=pin))
        self.put(early['cp_root']/'progress.json',progress)
        with self.assertRaises(ValueError):viewer.Viewer(self.group,'B',context).early_scoring(early,[['python']],{})

    def test_partial_BEST_bytes_tamper_is_rejected(self):
        context,early,_,_=self.cp();(self.group/'B/training/checkpoint_best.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'witness changed'):
            viewer.Viewer(self.group,'B',context).early_scoring(early,[['python']],{})

    def test_wrapper_reuses_original_UI_main_and_restores_reader(self):
        normalread,normalviewer=viewer.original.read,viewer.original.Viewer;calls=[]
        def read(path,*,required=False):calls.append(Path(path));return {}
        def main(args):
            viewer.original.read(self.group/'A/monitor/status.json')
            self.assertEqual(args,['--group',str(self.group),'--arm','B','--once'])
        # Construct the context before replacing the real immutable input reader.
        context=self.context()
        with patch.object(viewer,'Transition',return_value=context),patch.object(viewer.original,'read',side_effect=read),patch.object(viewer.original,'main',side_effect=main):
            viewer.main(['--group',str(self.group),'--recovery-seal',str(self.old_seal),
                '--recovery-seal-sha256',self.old_sha,'--authorization-request',str(self.authorization),
                '--authorization-request-sha256',self.auth_sha,'--B-route',str(self.route),'--arm','B','--once'])
        self.assertIs(viewer.original.read,normalread);self.assertIs(viewer.original.Viewer,normalviewer)
        self.assertEqual(calls,[Path(self.old['slots']['A']['monitor_path'])])

    def test_ACD_frozen_running_monitor_is_overruled_by_actual_pause(self):
        self.invocation('A'); context=self.context()
        value=dict(arm='A',GPU=6,status='EXITED_WITHOUT_COMPLETION',active_stage='gnn',
            gnn=dict(full_training_completed=False,error='old frozen monitor'),native=None,scoring=None)
        with patch.object(viewer.original.Viewer,'_snapshot',return_value=value),patch.object(
                viewer.original,'observe_actor',return_value=(None,'EXITED_WITHOUT_COMPLETION')):
            result=viewer.Viewer(self.group,'A',context)._snapshot()
        self.assertEqual(result['status'],'PAUSED_USER_REQUEST')
        self.assertFalse(result['gnn']['full_training_completed'])
        self.assertEqual(result['gnn']['optimizer_updates'],174)
        self.assertNotIn('error',result['gnn'])
        self.assertIn('중단',viewer.original.display(result))

    def test_B_before_pointer_keeps_original_live_GNN_view(self):
        context=self.context()
        value=dict(arm='B',GPU=5,status='RUNNING',active_stage='gnn',gnn=dict(epoch=13),native=None,scoring=None)
        with patch.object(viewer.original.Viewer,'_snapshot',return_value=value),patch.object(
                viewer.original,'observe_actor',return_value=({'process_status':'running'},None)):
            result=viewer.Viewer(self.group,'B',context)._snapshot()
        self.assertEqual(result['status'],'RUNNING'); self.assertEqual(result['gnn']['epoch'],13)
        self.assertNotIn('authorized_B_downstream',result)

    def test_B_after_pointer_routes_actual_CP_but_keeps_partial_GNN_counters(self):
        context,early,_,_=self.cp();self.invocation('B')
        command=['python','score'];actor=dict(pid=2010,create_time=200.,command=command,UID=41833)
        self.put(early['monitor'],dict(arm='B',GPU=5,status='RUNNING',stage='scoring',actor=actor,
            request=dict(arm='B',GPU=5,production_output=str(self.group/'B'),RAM_GiB=64,
                GNN_full_training=False,GNN_completed_epochs=13,GNN_actual_updates=221,
                stages=[dict(name='scoring',command=command),dict(name='native',command=['python','native'])])))
        progress=json.loads((early['cp_root']/'progress.json').read_bytes());progress['actor']=actor
        self.put(early['cp_root']/'progress.json',progress)
        value=dict(arm='B',GPU=5,status='EXITED_WITHOUT_COMPLETION',active_stage='gnn',
            gnn=dict(full_training_completed=False),native=None,scoring=None)
        def observe(declaration,**kwargs):
            if declaration['pid']==2010:return dict(RSS_bytes=1024,process_status='sleeping'),None
            return None,'EXITED_WITHOUT_COMPLETION'
        with patch.object(viewer.original.Viewer,'_snapshot',return_value=value),patch.object(
                viewer.original,'observe_actor',side_effect=observe),patch.object(viewer.original.Viewer,'native',return_value=None):
            result=viewer.Viewer(self.group,'B',context)._snapshot()
        self.assertEqual((result['active_stage'],result['status']),('scoring','RUNNING'))
        self.assertEqual((result['gnn']['completed_epochs'],result['gnn']['optimizer_updates']),(13,221))
        self.assertFalse(result['gnn']['full_training_completed']);self.assertEqual(result['scoring']['CP_done'],128)
        self.assertIn('UI 2',viewer.original.display(result))


if __name__=='__main__':unittest.main()
