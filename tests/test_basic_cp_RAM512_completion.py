"""CPU DEBUG migration dispatch; no completed research result is fabricated."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tools import run_basic_cp_RAM512 as runner


class CompletionDispatchTests(unittest.TestCase):
    def test_already_best26_requires_original_admitted_validator(self):
        with mock.patch.object(runner,'_migration',return_value={'state':'BEST26_READY'}),\
                mock.patch.object(runner,'_prior_validator',return_value=None):
            with self.assertRaisesRegex(ValueError,'original pinned'):runner._needs_training_stage({})

    def test_final250_pending_best26_keeps_only_original_validation_route(self):
        with mock.patch.object(runner,'_migration',return_value={'state':'FINAL250_PENDING_BEST26'}),\
                mock.patch.object(runner,'_prior_validator')as prior:
            self.assertTrue(runner._needs_training_stage({}));prior.assert_not_called()
        # The actual unchanged native train code selects --val when FINAL250
        # exists; no new shortened training or alternate evaluation is added.
        self.assertIs(runner._train_native.__code__,runner.historical._train_native.__code__)

    def test_full_complete_returns_before_any_gpu_or_actor_dispatch(self):
        with mock.patch.dict(runner._SCOPE,_carry_complete_pipeline=lambda _:True,
                _wait_for_GPU=mock.Mock(side_effect=AssertionError('No GPU allocation')),
                prepare_bank=mock.Mock(side_effect=AssertionError('No bank regeneration')),
                child=mock.Mock(side_effect=AssertionError('No native duplicate')),
                actor=mock.Mock(side_effect=AssertionError('No evaluation duplicate'))):
            self.assertIsNone(runner.run({'DEBUG':True}))

    def run_existing_best26(self,metrics_complete):
        with tempfile.TemporaryDirectory(prefix='DEBUG_RAM512_COMPLETION_',dir=Path(__file__).parent)as folder:
            root=Path(folder);native=dict(root=str(root),gpu=6,arm='blend50',trainer='DEBUG_only',
                evaluator_root='DEBUG_evaluator',project='DEBUG_project')
            (root/'native.json').write_text(json.dumps(native))
            for name in('preflight.json','calibration.json'):(root/name).write_text('{}')
            (root/'training_and_BEST26_complete.json').write_text(json.dumps({'fold':str(root/'DEBUG_preserved_fold')}))
            child=mock.Mock(side_effect=AssertionError('Already BEST26 must not retrain/revalidate'))
            actor=mock.Mock();report=mock.Mock(return_value={'DEBUG_report':True});archive=mock.Mock(return_value={'DEBUG_archive':True})
            with mock.patch.dict(runner._SCOPE,_carry_complete_pipeline=lambda _:False,
                    _wait_for_GPU=mock.Mock(),prepare_bank=lambda q:q,child=child,actor=actor,
                    _needs_training_stage=lambda n:False,_needs_historical_metrics=lambda n:not metrics_complete,
                    _completed_report_for=lambda n:report,_archive_for=lambda n:archive,status=mock.Mock(),
                    environment=lambda n:{}):
                runner.run(dict(native,candidates_per_source=128))
            child.assert_not_called();report.assert_called_once();archive.assert_called_once()
            return actor

    def test_best26_ready_goes_directly_to_historical_metrics(self):
        actor=self.run_existing_best26(False)
        actor.assert_called_once();self.assertEqual(actor.call_args.args[2],'historical_metrics')

    def test_existing_verified_metrics_are_not_recomputed(self):
        self.run_existing_best26(True).assert_not_called()

    def test_original_validator_scope_rebind_does_not_patch_historical_guard(self):
        from tools import basic_cp_expanded_result as result
        old=copy.copy(result.completed_report.__globals__);prior=object()
        scope=runner._report_scope(result.completed_report,prior)
        self.assertIs(scope['_RAM512_completed_validator'],prior)
        self.assertIs(scope['validate_completed'].__globals__,scope)
        self.assertNotIn('_RAM512_completed_validator',result.completed_report.__globals__)
        self.assertEqual(set(result.completed_report.__globals__),set(old))
        self.assertIsNot(scope['completed_report'],result.completed_report)

    def test_actual_completed_validator_local_import_uses_only_original_guard(self):
        from tools import basic_cp_expanded_result as result
        from hiercp_v1x import v24_native_best_eval_runtime as current_guard
        cases=['DEBUG_val_'+str(i)for i in range(26)];train=['DEBUG_train_'+str(i)for i in range(105)]
        native=dict(arm='blend50',root='/DEBUG/newroot',native_root='/DEBUG/original_native',
            split={'outer_train':train,'outer_val':cases},bank_sha256='DEBUG_bank')
        proof=dict(epochs=250,checkpoint='checkpoint_best.pth',evaluation_checkpoint_is_final=False,
            fold='/DEBUG/original_native/fold',best_sha256='DEBUG_best_sha',validation_cases=cases,
            final_training_witness_sha256='DEBUG_final_sha')
        model=dict(physical_batch=2,effective_batch=2,gradient_accumulation=1,epochs=250,patch_size=[128]*3,
            debug=False,subset=False,pretrained_segmentation_used=False,actual_train_cases=train,actual_validation_cases=cases)
        prior=mock.Mock(BEST='checkpoint_best.pth',RECEIPT='best_validation_complete.json')
        prior.validate_best_validation_receipt.return_value={'checkpoint_sha256':'DEBUG_best_sha','validation_cases':cases}
        scope=runner._report_scope(result.completed_report,prior)
        scope.update(exact=lambda path,*a,**kw:Path(path),read=mock.Mock(side_effect=[proof,model]),sha=lambda _: 'DEBUG_final_sha')
        with mock.patch.object(current_guard,'validate_best_validation_receipt',side_effect=AssertionError('New-path guard forbidden')):
            scope['validate_completed'](native,proof)
        prior.validate_best_validation_receipt.assert_called_once_with(
            Path('/DEBUG/original_native/fold/best_validation_complete.json'),
            fold=Path('/DEBUG/original_native/fold'),expected_cases=cases,bank_sha256='DEBUG_bank')

    def test_archive_pending_forecast_queues_before_copy_and_outside_lock(self):
        events=[]
        class Reservation:
            def __init__(self,blocked):self.blocked=blocked
            def __enter__(self):
                events.append('enter')
                if self.blocked:raise runner.ResourceClaimUnavailable('DEBUG forecast waits for STORE272',{})
                events.append('unlocked');return self
            def __exit__(self,*args):events.append('exit')
        lease=mock.Mock();lease.archive_reservation.side_effect=[Reservation(True),Reservation(False)]
        with mock.patch.object(runner,'_lease_for',return_value=lease),mock.patch.object(runner.historical,'status'),\
                mock.patch.object(runner.time,'sleep',side_effect=lambda _:events.append('queue')):
            with runner._archive_context({},100,Path('/DEBUG_only')):
                events.append('independent copy')
        self.assertEqual(events,['enter','queue','enter','unlocked','independent copy','exit'])

    def test_archive_io_failure_is_not_replayed_or_hidden(self):
        reservation=mock.MagicMock();lease=mock.Mock();lease.archive_reservation.return_value=reservation
        with mock.patch.object(runner,'_lease_for',return_value=lease):
            with self.assertRaisesRegex(OSError,'DEBUG IO'):
                with runner._archive_context({},100,Path('/DEBUG_only')):raise OSError('DEBUG IO failure')
        self.assertEqual(lease.archive_reservation.call_count,1)
        self.assertIs(reservation.__exit__.call_args.args[0],OSError)


if __name__=='__main__':unittest.main()
