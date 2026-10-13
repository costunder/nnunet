"""Read-only recovery observer fixtures; no training, CUDA or real actors."""
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests import test_v25_B_only_viewer as baseline
from tools import watch_v25_B_CP_recovery as watch


class RecoveryDisplayTests(unittest.TestCase):
    def setUp(self):
        self.base = baseline.BOnlyViewerTests(); self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.old_context, self.old, _, _ = self.base.cp()
        self.old_seal = self.old['seal']
        self.old_controller = dict(pid=111, create_time=1., UID=41833, command=['python', 'old_controller'])
        self.old_child = dict(pid=112, create_time=2., UID=41833, command=['python', 'old_child'])
        self.old_seal['controller_identity'] = self.old_controller
        route = json.loads(self.base.route.read_bytes())
        route['early_seal'] = self.base.proof(Path(route['early_seal']['path']), self.old_seal)
        self.base.put(self.base.route, route)
        failed = dict(status='FAILED', arm='B', controller_pid=111, actor=self.old_child,
            error='MemoryError: Actual complete workspace RSS limit')
        failed_ref = self.base.proof(self.old['monitor'], failed)
        self.controller = dict(pid=211, create_time=3., UID=41833, command=['python', 'new_controller'])
        self.child = dict(pid=212, create_time=4., UID=41833, command=['python', 'new_child'])
        self.monitor = self.base.group/'B/monitor/B_CP529_RAM_recovery_DEBUG/status.json'
        self.base.put(self.monitor, dict(status='RUNNING', arm='B', controller_pid=211, actor=self.child))
        fresh = copy.deepcopy(self.old_seal)
        fresh['controller_identity'] = self.controller
        fresh['slot']['monitor_path'] = str(self.monitor)
        namespace = self.base.group/'B_CP529_RAM_recovery_DEBUG/execution'
        self.seal_path = namespace/'controller_seal.json'
        self.recovery_path = namespace/'recovery.json'
        self.fresh = fresh
        self.recovery = dict(format=watch.RECOVERY_FORMAT, gpu=5,
            gpu_uuid='GPU-e41d3421-8819-0a49-4a97-6c4518a54f5f', CPU_affinity=list(range(42, 48)),
            previous_results_preserved=True, processes_signaled=False,
            B_controller_seal=self.base.proof(self.seal_path, fresh),
            previous_failed_execution=dict(controller=self.old_controller, child=self.old_child,
                monitor=failed_ref, controller_seal=route['early_seal']),
            durable=dict(root=str(self.old['cp_root']), completed_batches=133, completed_sources=529,
                completed_positions=67712, request=dict(path=str(self.old['cp_root']/'request.json'),
                    sha256=watch.original.sha(self.old['cp_root']/'request.json'))))
        self.recovery_ref = self.base.proof(self.recovery_path, self.recovery)
        self.observer = patch.object(watch.original, 'observe_actor', side_effect=self.actor)
        self.observer.start(); self.addCleanup(self.observer.stop)

    def actor(self, declared, **kwargs):
        if declared['pid'] in (111, 112):return None, 'EXITED_WITHOUT_COMPLETION'
        return dict(declared, parent_pid=211 if declared['pid']==212 else 99), None

    def context(self):
        return watch.RecoveryTransition(self.base.group, self.base.old_seal, self.base.old_sha,
            self.base.authorization, self.base.auth_sha, self.base.route,
            recovery_proof=self.recovery_ref)

    def republish(self):
        self.recovery['B_controller_seal'] = self.base.proof(self.seal_path, self.fresh)
        self.recovery_ref = self.base.proof(self.recovery_path, self.recovery)

    def test_original_529_science_and_new_exact_monitor_are_bound_without_writes(self):
        before={str(path):path.read_bytes() for path in self.base.group.rglob('*') if path.is_file()}
        result=self.context().early()
        self.assertEqual(result['monitor'], self.monitor)
        self.assertEqual(result['cp_root'], self.old['cp_root'])
        self.assertEqual(result['science'], self.old['science'])
        self.assertEqual(result['release'], self.old['release'])
        self.assertEqual(result['seal']['controller_identity'], self.controller)
        self.assertEqual(before, {str(path):path.read_bytes() for path in self.base.group.rglob('*') if path.is_file()})

    def test_other_CP_root_cannot_replace_original_durable_prefix(self):
        self.recovery['durable']['root']=str(self.base.group/'new_CP_bank'); self.republish()
        with self.assertRaisesRegex(ValueError, 'immutable 529-source'):
            self.context().early()

    def test_fabricated_642_prefix_is_rejected(self):
        self.recovery['durable']['completed_sources']=642; self.republish()
        with self.assertRaisesRegex(ValueError, 'immutable 529-source'):
            self.context().early()

    def test_changed_own_BEST_witness_is_rejected(self):
        self.fresh['preserved_files'][str(self.base.group/'B/training/checkpoint_best.pt')]='0'*64
        self.republish()
        with self.assertRaisesRegex(ValueError, 'BEST witness'):
            self.context().early()

    def test_canonical_original_monitor_is_never_recovery_monitor(self):
        self.fresh['slot']['monitor_path']=str(self.base.group/'B/monitor/status.json'); self.republish()
        with self.assertRaisesRegex(ValueError, 'separate B recovery monitor'):
            self.context().early()

    def test_changed_scientific_request_proof_is_rejected(self):
        self.fresh['scientific_request']['sha256']='0'*64; self.republish()
        with self.assertRaisesRegex(ValueError, 'original science'):
            self.context().early()

    def test_previous_live_controller_cannot_be_claimed_released(self):
        with patch.object(watch.original,'observe_actor',side_effect=lambda identity,**kwargs:(dict(identity,parent_pid=99),None)):
            with self.assertRaisesRegex(ValueError,'naturally released'):
                self.context().early()

    def test_live_stage_child_must_belong_to_exact_new_controller(self):
        def actor(identity,**kwargs):
            result,problem=self.actor(identity,**kwargs)
            if identity['pid']==212:result['parent_pid']=999
            return result,problem
        with patch.object(watch.original,'observe_actor',side_effect=actor):
            with self.assertRaisesRegex(ValueError,'another controller'):
                self.context().early()

    def test_departed_new_controller_is_not_live_RUNNING(self):
        with patch.object(watch.original,'observe_actor',return_value=(None,'EXITED_WITHOUT_COMPLETION')):
            with self.assertRaisesRegex(ValueError,'exited before completion'):
                self.context().early()

    def test_recovery_proof_byte_tamper_is_not_hidden(self):
        with self.recovery_path.open('ab') as output:output.write(b' ')
        with self.assertRaisesRegex(ValueError,'witness changed'):
            self.context().early()

    def test_original_paused_ACD_invocation_remains_partial(self):
        self.base.invocation('A')
        with patch.object(watch.original,'observe_actor',return_value=(None,'EXITED_WITHOUT_COMPLETION')):
            paused=self.context().pause('A')
        self.assertEqual(paused['status'],'PAUSED_USER_REQUEST')
        self.assertFalse(paused['invocation']['full_training'])
        self.assertEqual(paused['invocation']['optimizer_updates'],174)

    def value(self,stage='scoring'):
        return dict(GPU=5,arm='B',status='RUNNING',snapshot_fresh=True,active_stage=stage,
            gnn=dict(), native=None, scoring=dict(status='RUNNING',source_done=529,source_total=642,
                CP_done=67712,CP_total=82176,own_BEST_pin_verified=True,RSS_GiB=20,RAM_limit_GiB=64))

    def test_CP_uses_the_same_eight_labels_and_own_BEST(self):
        output=io.StringIO()
        with patch.object(watch.sys,'stdout',output):watch.observe(Mock(snapshot=lambda:self.value()),once=True)
        shown=output.getvalue()
        self.assertEqual(tuple(line.partition(': ')[0] for line in shown.splitlines()),watch.original.ui.CP_ROW_LABELS)
        self.assertIn('529/642',shown);self.assertIn('GPU5 GNN BEST',shown)

    def test_native_uses_actual_original_epoch_console_without_new_layout(self):
        path=self.base.group/'actual_native.log'
        raw=b'2026-10-13 15:00:00: Epoch 8\n2026-10-13 15:00:00: Current learning rate: 0.009\n'
        path.write_bytes(raw);value=self.value('native');value['native']=dict(stage='train',stage_log=str(path))
        output=io.StringIO()
        with patch.object(watch.sys,'stdout',output):watch.observe(Mock(snapshot=lambda:value),once=True)
        self.assertEqual(output.getvalue().encode(),raw)

    def test_json_preserves_existing_v25_snapshot_fields(self):
        value=self.value();output=io.StringIO()
        with patch.object(watch.sys,'stdout',output):watch.observe(Mock(snapshot=lambda:copy.deepcopy(value)),as_json=True)
        expected=dict(value,ui_schema_version=2,experiment_version='v2.5')
        self.assertEqual(json.loads(output.getvalue()),expected)


if __name__=='__main__':unittest.main()
