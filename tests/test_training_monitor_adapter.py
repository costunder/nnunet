"""DEBUG CPU tests: UI adapters never invent measurements or completion."""
import unittest
from tools import training_monitor_adapter as adapter


class SharedMonitorAdapterCPU_DEBUG(unittest.TestCase):
    def test_read_failure_overrides_nested_running_and_preserves_provenance(self):
        model = adapter.from_pipeline(dict(GPU=5,status='READ_ERROR',active_stage='gnn',snapshot_fresh=False,
            last_valid_snapshot_preserved=True,last_confirmed_status='RUNNING',
            read_problem=dict(detail='/real/report.json: incomplete'),
            gnn=dict(status='RUNNING',epoch=27,epochs=40,active_U=119,RSS_GiB=47,
                RSS_source='last successful train update',reports=dict(full_validation=dict(epoch=26,
                    active_u=128,metrics=dict(per_P_patient_mrr=.2,per_P_patient_top1=.15))))))
        self.assertEqual(model['state']['status'],'READ_ERROR')
        self.assertIn('새 기록이 아님',model['warning'])
        self.assertIn('/real/report.json',model['error'])
        self.assertEqual(model['gnn']['full_validation']['active_u'],128)
        self.assertEqual(model['gnn']['active_u'],119)
        self.assertEqual(model['resources']['source'],'과거 GNN train update 기록')

    def test_missing_fields_are_unknown_even_after_gnn_complete(self):
        model = adapter.from_pipeline(dict(GPU=4,status='WAITING_FOR_FULL642_SCORING',active_stage='scoring',
            snapshot_fresh=True,gnn=dict(full_training_completed=True,completed_epochs=40,epochs=40),
            scoring=dict(status='STARTING'),native=dict(status='WAITING_FOR_FULL642_SCORING')))
        self.assertTrue(model['gnn']['full_completed'])
        self.assertIsNone(model['cp']['source_done'])
        self.assertIsNone(model['native']['epoch'])
        self.assertIsNone(model['native']['full_training'])
        self.assertIsNone(model['native']['full_evaluation'])

    def test_top_controller_reason_overrides_old_scoring_storage_text(self):
        model = adapter.from_pipeline(dict(GPU=6,status='SCORING_PROCESS_ALIVE',active_stage='scoring',
            snapshot_fresh=True,scoring=dict(native_pending_reason='old wrong storage quota',
                source_done=448,source_total=642,CP_done=57344,CP_total=82176),
            native=dict(status='WAITING_FOR_FULL642_SCORING'),native_pending_reason='actual scorer pending'))
        self.assertEqual(model['waiting_reason'],'actual scorer pending')
        self.assertEqual(model['cp']['positions_done'],57344)

    def test_zero_progress_is_real_and_sleeping_is_not_failure(self):
        model=adapter.from_scoring(dict(GPU=6,status='SCORING_PROCESS_ALIVE',snapshot_fresh=True,
            source_done=0,source_total=642,CP_done=0,CP_total=82176,process_status='sleeping',RSS_GiB=33,
            native_pending_reason='actual full scoring pending'))
        self.assertEqual(model['cp']['source_done'],0)
        self.assertEqual(model['resources']['process_status'],'sleeping')
        self.assertFalse(model['error'])
        self.assertEqual(model['waiting_reason'],'actual full scoring pending')

    def test_native_best_evaluation_and_pseudo_dice_remain_distinct(self):
        model=adapter.from_pipeline(dict(GPU=6,status='RUNNING',active_stage='native',snapshot_fresh=True,
            native=dict(status='RUNNING',stage='train',native_training_started=True,epoch=250,epochs=250,
                completed_epochs=250,full_native_training=False,full_native_evaluation=False,
                native_metrics=dict(train_loss=-.3,val_loss=-.2,pseudo_dice=[.8,.7]))))
        self.assertTrue(model['native']['started'])
        self.assertFalse(model['native']['full_evaluation'])
        self.assertEqual(model['native']['pseudo_dice'],[.8,.7])
        self.assertNotIn('dice',model['native'])

    def test_standalone_native_does_not_leak_native_epochs_into_gnn(self):
        model=adapter.from_training(dict(GPU=1,status='RUNNING',stage='nnunet_training',snapshot_fresh=True,
            epoch=4,epochs=250,completed_epochs=3,native_metrics=dict(pseudo_dice=[.4])))
        self.assertIsNone(model['gnn']['epoch'])
        self.assertEqual(model['native']['epoch'],4)
        self.assertIsNone(model['native']['full_evaluation'])


if __name__=='__main__':
    unittest.main()
