"""CPU state/contract tests; no neural forward, training or fabricated metrics."""
import copy
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import transition_c_training as c


def config():
    return dict(seed=42,cache=dict(total_candidates=8,candidate_pool_size=128),
        training=dict(epochs=40,gradient_accumulation_steps=1,
                      target_effective_batch_size=None,fixed_validation_epoch=29),
        transition_runtime=dict(physical_batch_size=1,support_batch_size=2,
            validation_batch_size=2,workers=4,device='cuda:0',debug_curriculum_epoch=29))


class TrainingMetadataContracts(unittest.TestCase):
    def test_measured_original_sample_batch_one_remains_eight_queries(self):
        result=c.runtime_contract(config(),debug=False,debug_updates=None,train_samples=151,val_samples=36)
        self.assertEqual(result['physical_candidate_batch'],8)
        self.assertEqual(result['epochs'],40)
        self.assertEqual(result['effective_sample_batch'],1)
        self.assertFalse(result['original_support_two_views'])
        self.assertTrue(result['original_ranking_objective_preserved'])

    def test_no_implicit_cpu_worker_or_short_production_fallback(self):
        edits=[('device','cpu'),('workers',1),('physical_batch_size','auto'),('support_batch_size',0)]
        for key,value in edits:
            cfg=config();cfg['transition_runtime'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                c.runtime_contract(cfg,debug=False,debug_updates=None,train_samples=151,val_samples=36)
        with self.assertRaises(ValueError):
            c.runtime_contract(config(),debug=False,debug_updates=2,train_samples=151,val_samples=36)

    def test_debug_bounds_are_explicit_and_do_not_change_production_epochs(self):
        result=c.runtime_contract(config(),debug=True,debug_updates=2,train_samples=3,val_samples=1)
        self.assertEqual(result['successful_DEBUG_updates'],2)
        self.assertEqual(result['epochs'],40)
        for changes in (('epochs',2),('fixed_validation_epoch',1),('gradient_accumulation_steps',2)):
            cfg=config();cfg['training'][changes[0]]=changes[1]
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                c.runtime_contract(cfg,debug=True,debug_updates=2,train_samples=3,val_samples=1)
        cfg=config();cfg['transition_runtime'].pop('debug_curriculum_epoch')
        with self.assertRaises(ValueError):
            c.runtime_contract(cfg,debug=True,debug_updates=2,train_samples=3,val_samples=1)
        for key,value in (('total_candidates',7),('candidate_pool_size',64)):
            cfg=config();cfg['cache'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                c.runtime_contract(cfg,debug=True,debug_updates=2,train_samples=3,val_samples=1)

    def test_best_selection_uses_margin_then_actual_ranking_and_rounding(self):
        # These are metadata sentinels only, never predictions/evaluation outputs.
        base=dict(MRR=.5,top1=.25,mean_margin=.1,loss=2.)
        self.assertGreater(c.own_selection_key({**base,'mean_margin':.2}),c.own_selection_key(base))
        self.assertGreater(c.own_selection_key({**base,'loss':1.}),c.own_selection_key(base))
        self.assertEqual(c.own_selection_key({**base,'MRR':.50000000001}),c.own_selection_key(base))
        with self.assertRaises(ValueError):c.own_selection_key({**base,'loss':float('nan')})
        self.assertEqual(len(c.own_selection_key(base)),5)

    def test_materialized_cohort_preserves_configured_unmaterialized_patients(self):
        class Loader:
            samples=[dict(case_id='case_a',sample_index=0,split='val')]
            sample_sources=[dict(kind='signed_original_cache',path='/actual/case_a__000.pt',sha256='a'*64)]
            def __len__(self):return len(self.samples)
        runtime=dict(actual_signed_val_cases=['case_a'],baseline_proof_sha256='b'*64,cache_index_sha256='c'*64,
            actual_signed_val_samples=[dict(case_id='case_a',sample_index=0,path='case_a__000.pt',sha256='a'*64)])
        result=c.validate_signed_cohort(Loader(),runtime,'val',['case_a','unmaterialized'],debug=False)
        self.assertEqual(result['configured_but_not_materialized_cases'],['unmaterialized'])
        self.assertEqual(result['samples'],1)
        runtime['actual_signed_val_samples'][0]['sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'bytes differ'):
            c.validate_signed_cohort(Loader(),runtime,'val',['case_a','unmaterialized'],debug=False)

    def test_scoped_sigint_sets_flag_and_restores_previous_handler(self):
        net=SimpleNamespace();previous=signal.getsignal(signal.SIGINT)
        def run(*args,**kwargs):
            self.assertFalse(net._transition_pause_requested())
            signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
            self.assertTrue(net._transition_pause_requested())
            self.assertTrue(kwargs['_pause_flag']['requested'])
            return {'status':'METADATA_ONLY'}
        directory=Path(__file__).resolve().parents[1]/'work/transition_C_metadata_UNIT_virtual'
        with patch.object(c,'_train_c_impl',side_effect=run),patch.object(Path,'exists',return_value=False):
            result=c.train_c(net,None,None,config(),output=directory,budget=None,debug=True,debug_updates=2)
        self.assertEqual(result['status'],'METADATA_ONLY')
        self.assertIs(signal.getsignal(signal.SIGINT),previous)
        self.assertFalse(hasattr(net,'_transition_pause_requested'))

    def test_pause_file_and_callback_restored_even_on_failure(self):
        callback=lambda:False;net=SimpleNamespace(_transition_pause_requested=callback)
        previous=signal.getsignal(signal.SIGINT)
        directory=Path(__file__).resolve().parents[1]/'work/transition_C_metadata_UNIT_virtual'
        def run(*args,**kwargs):
            self.assertTrue(net._transition_pause_requested())
            raise RuntimeError('Explicit metadata test failure')
        with patch.object(c,'_train_c_impl',side_effect=run),patch.object(Path,'exists',return_value=True),\
                patch.object(Path,'read_text',return_value=json.dumps({'action':'pause'})),\
                self.assertRaisesRegex(RuntimeError,'Explicit metadata'):
            c.train_c(net,None,None,config(),output=directory,budget=None,debug=True,debug_updates=2)
        self.assertIs(net._transition_pause_requested,callback)
        self.assertIs(signal.getsignal(signal.SIGINT),previous)


class SupportRestartState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tests import tensors only. Hide GPU in the invoking subprocess too.
        import torch
        cls.torch=torch

    def fixture(self):
        torch=self.torch
        bank=dict(generation='metadata-generation',owners=torch.tensor([0,1]),classes=torch.tensor([1,0]))
        episode=dict(key='case_a',owners=torch.tensor([0,1]),classes=torch.tensor([1,0]),indices=torch.tensor([0,1]))
        plan=dict(owners=episode['owners'].clone(),classes=episode['classes'].clone(),centers=torch.arange(4.).reshape(2,2))
        class Upper:
            def __init__(self):
                self._plans={'case_a':copy.deepcopy(plan)}
                self._episodes={'case_a':copy.deepcopy(episode)}
                self._seen_generations={'previous','metadata-generation'}
                self.admitted=0
            def bind_support(self,memory):
                self.admitted+=1;self._plans={};self._episodes={};self._seen_generations={memory['generation']}
            def _episode(self,key):
                self._episodes[key]=copy.deepcopy(episode)
                return self._episodes[key]
        return SimpleNamespace(upper=Upper()),bank

    def test_snapshot_is_detached_and_own_digest_rejects_mutation(self):
        net,bank=self.fixture();snapshot=c.support_runtime_snapshot(net,bank)
        bank['owners'][0]=9
        self.assertEqual(snapshot['bank']['owners'][0],0)
        destination,_=self.fixture()
        restored=c.restore_support_runtime(destination,snapshot,'cpu')
        self.assertEqual(destination.upper.admitted,1)
        self.assertEqual(restored['generation'],destination._last_C_generation)
        self.assertEqual(destination.upper._seen_generations,{'previous','metadata-generation'})
        self.assertTrue(self.torch.equal(destination.upper._plans['case_a']['centers'],snapshot['teacher_plans']['case_a']['centers']))
        snapshot['teacher_plans']['case_a']['centers'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'digest changed'):
            c.restore_support_runtime(destination,snapshot,'cpu')

    def test_rehashed_wrong_teacher_ownership_cannot_be_resumed(self):
        net,bank=self.fixture();snapshot=c.support_runtime_snapshot(net,bank)
        snapshot['teacher_plans']['case_a']['owners'][0]=9
        snapshot['content_sha256']=c._digest({k:v for k,v in snapshot.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'another query exclusion'):
            c.restore_support_runtime(net,snapshot,'cpu')

    def test_untrained_restart_does_not_invent_a_support_bank(self):
        net,_=self.fixture();net.upper._plans={};net.upper._episodes={};net.upper._seen_generations=set()
        snapshot=c.support_runtime_snapshot(net,None)
        self.assertIsNone(c.restore_support_runtime(net,snapshot,'cpu'))
        self.assertEqual(net.upper.admitted,0)


class MonotonicClock:
    def __init__(self,value=0.):self.value=value
    def __call__(self):return self.value
    def advance(self,seconds):self.value+=seconds


class ExactEpochWallContracts(unittest.TestCase):
    def test_continuous_wall_counts_operations_and_writer_wait_but_not_downtime(self):
        now=MonotonicClock();wall=c.ActiveEpochWall(3,clock=now)
        for phase,duration in (('train_loader',2.),('optimizer_update',4.),('support_refresh',3.),
                ('own_evaluation',1.),('common_evaluation',5.),('checkpoint_save',2.)):
            with wall.phase(phase):now.advance(duration)
        snapshot=wall.snapshot()
        self.assertEqual(snapshot['elapsed_seconds'],17.)
        self.assertEqual(snapshot['phase_wall_seconds']['checkpoint_save'],2.)
        self.assertFalse(snapshot['stopped_process_downtime_included'])
        restarted=MonotonicClock(9000.)
        resumed=c.ActiveEpochWall(3,snapshot,clock=restarted)
        with resumed.phase('common_evaluation'):restarted.advance(3.)
        self.assertEqual(resumed.elapsed(),20.)
        self.assertEqual(resumed.history_fields()['phase_wall_seconds']['common_evaluation'],8.)

    def test_overlapped_phases_never_double_count_continuous_epoch_wall(self):
        now=MonotonicClock();wall=c.ActiveEpochWall(1,clock=now)
        writer=wall.begin('writer');update=wall.begin('update');now.advance(5.)
        wall.finish(writer);wall.finish(update)
        snapshot=wall.snapshot()
        self.assertEqual(snapshot['elapsed_seconds'],5.)
        self.assertEqual(sum(snapshot['phase_wall_seconds'].values()),10.)
        self.assertTrue(snapshot['phase_durations_may_overlap'])

    def receipt(self,*,closing=False):
        now=MonotonicClock();wall=c.ActiveEpochWall(3,clock=now)
        token=wall.begin('checkpoint_save');now.advance(1.)
        before=wall.snapshot('before_atomic_write');now.advance(2.)
        after=wall.snapshot('through_atomic_write_before_receipt_write')
        # The final receipt writer/close is known to the live clock, but cannot
        # be recursively included in that same receipt's own serialized row.
        now.advance(.25);wall.finish(token)
        history=[dict(epoch=3,epoch_wall_seconds=before['elapsed_seconds'])] if closing else []
        cursor=dict(epoch=4 if closing else 3,updates=7,history=history,
            active_epoch_wall=None if closing else before,position=2)
        saved=dict(run_identity_sha256='a'*64,cursor=cursor,
            checkpoint_wall_receipt=dict(checkpoint_id='UNIT_checkpoint',
                journal_file='c_checkpoint_timings_'+'b'*32+'.jsonl'))
        row=dict(format='crossed_C_checkpoint_wall_receipt_v1',checkpoint_id='UNIT_checkpoint',
            run_identity_sha256='a'*64,cursor_epoch=cursor['epoch'],updates=7,
            closing_epoch=closing,active_epoch_wall=after,receipt_self_write_tail_in_this_row=False)
        row['content_sha256']=c._wall_receipt_digest(row)
        self.assertEqual(wall.elapsed(),3.25)
        return saved,row

    def restore(self,saved,row,*,complete=True):
        text=json.dumps(row)+('\n' if complete else '')
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'is_symlink',return_value=False),\
                patch.object(Path,'read_text',return_value=text):
            return c.restore_epoch_wall_timing(saved,'UNIT_owned_output')

    def test_bound_receipt_recovers_atomic_write_tail_without_mutating_scientific_cursor(self):
        saved,row=self.receipt();restored=self.restore(saved,row)
        self.assertEqual(restored['active_epoch_wall']['elapsed_seconds'],3.)
        self.assertEqual(saved['cursor']['active_epoch_wall']['elapsed_seconds'],1.)
        self.assertEqual(restored['position'],2);self.assertEqual(restored['updates'],7)
        self.assertEqual(restored['wall_timing_resume_scope'],
            'through_atomic_checkpoint_write_before_its_timing_receipt_write')
        cursor=self.restore(saved,row,complete=False)
        self.assertEqual(cursor['active_epoch_wall']['elapsed_seconds'],1.)
        self.assertEqual(cursor['wall_timing_resume_scope'],'cursor_before_atomic_checkpoint_write')
        changed=copy.deepcopy(row);changed['active_epoch_wall']['elapsed_seconds']+=1
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.restore(saved,changed)
        changed=copy.deepcopy(row);changed['updates']+=1
        changed['content_sha256']=c._wall_receipt_digest({k:v for k,v in changed.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'another scientific cursor'):self.restore(saved,changed)

    def test_closing_receipt_recovers_history_wall_and_leaves_next_epoch_inactive(self):
        saved,row=self.receipt(closing=True);restored=self.restore(saved,row)
        self.assertEqual(restored['epoch'],4);self.assertIsNone(restored['active_epoch_wall'])
        self.assertEqual(restored['history'][0]['epoch_wall_seconds'],3.)
        self.assertEqual(restored['history'][0]['phase_wall_seconds']['checkpoint_save'],3.)
        self.assertEqual(saved['cursor']['history'][0]['epoch_wall_seconds'],1.)

    def test_invalid_or_backwards_elapsed_measurements_are_refused(self):
        now=MonotonicClock();wall=c.ActiveEpochWall(1,clock=now);saved=wall.snapshot()
        for changed in ({**saved,'elapsed_seconds':float('nan')},{**saved,'elapsed_seconds':-1},
                {**saved,'phase_wall_seconds':{'writer':True}},{**saved,'epoch':2}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):c.ActiveEpochWall(1,changed,clock=now)
        now.value=-1.
        with self.assertRaisesRegex(ValueError,'moved backwards'):wall.elapsed()


if __name__=='__main__':unittest.main()
