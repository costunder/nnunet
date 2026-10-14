"""CPU DEBUG dispatch tests; no GPU, tiny model, or research result is used."""
import ast
import contextlib
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import run_basic_cp_RAM512 as runner


class RunnerTests(unittest.TestCase):
    def test_full_train_and_cleanup_bytecode_are_preserved(self):
        for name in ('_train_native','train','preflight','debug_trial','child','prepare_bank'):
            with self.subTest(name=name):
                actual=getattr(runner,name);old=getattr(runner.historical,name)
                self.assertIs(actual.__code__,old.__code__)
                self.assertIsNot(actual.__globals__,old.__globals__)

    def test_torch_import_cannot_precede_bound_resource_guard(self):
        imports=[]
        import builtins
        original_import=builtins.__import__
        def observed_import(name,*args,**kwargs):
            if name=='torch':imports.append(name)
            return original_import(name,*args,**kwargs)
        def missing_claim(value):raise RuntimeError('DEBUG missing exact allocation')
        with mock.patch.dict(runner._SCOPE,resource_guard=missing_claim),mock.patch.object(builtins,'__import__',side_effect=observed_import):
            with self.assertRaisesRegex(RuntimeError,'missing exact allocation'):runner._train_native({},[])
        self.assertEqual(imports,[])

    def test_private_child_uses_new_named_runner_without_historical_mutation(self):
        self.assertEqual(Path(runner.child.__globals__['__file__']).name,'run_basic_cp_RAM512.py')
        self.assertEqual(Path(runner.historical.child.__globals__['__file__']).name,'run_basic_cp_expanded.py')
        self.assertIs(runner.historical.subprocess,runner._subprocess)
        self.assertIsNot(runner._SCOPE['subprocess'],runner._subprocess)

    def test_provider_dispatch_changes_only_operational_module_and_cap(self):
        native=dict(gpu=5,shared_candidate_geometry_cache=True,shared_bank_workers=24,shared_bank_RSS_limit_GiB=384)
        command=['NNpython','-B','-u','-m','hiercp_v1x.basic_cp_large_bank_recovery',
            '--binding','actual_frozen_binding','--output','fresh_bank','--workers','24',
            '--rss-limit-gib','384','--recovery-snapshot','sealed_source','--recovery-snapshot-sha256','a'*64]
        old=list(command)
        with mock.patch.object(runner,'_admit_execution'):
            result=runner._provider_command(native,command,'candidate_bank')
        expected=list(old);expected[4]=runner.PROVIDER_MODULE;expected[12]='320'
        self.assertEqual(result,expected);self.assertEqual(command,old)
        self.assertEqual(native['shared_bank_RSS_limit_GiB'],384)

    def test_provider_rejects_worker_or_cap_shrink_before_dispatch(self):
        native=dict(gpu=5,shared_candidate_geometry_cache=True,shared_bank_workers=24,shared_bank_RSS_limit_GiB=384)
        base=['NNpython','-m','hiercp_v1x.basic_cp_large_bank_recovery','--workers','24','--rss-limit-gib','384']
        with mock.patch.object(runner,'_admit_execution'):
            for index,value in ((4,'1'),(6,'96')):
                bad=list(base);bad[index]=value
                with self.subTest(index=index),self.assertRaises(ValueError):runner._provider_command(native,bad,'candidate_bank')

    def test_native_or_historical_metrics_command_remains_byte_exact(self):
        command=['NNpython','-B','-u','actual_full26_metrics.py','--checkpoint','BEST']
        self.assertEqual(runner._provider_command({},command,'historical_metrics'),command)

    def test_popen_without_prior_claim_is_rejected_before_process_creation(self):
        facade=runner._ReservedSubprocess()
        with mock.patch.object(runner,'_ACTIVE_RESERVATION',None),mock.patch.object(runner._subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'previously published'):facade.Popen(['DEBUG'])
            popen.assert_not_called()

    def test_popen_orders_published_claim_idle_gpu_launch_exact_bind(self):
        events=[];command=['DEBUG_native','--stage','train']
        reservation=mock.Mock();reservation.claim={'command':command};reservation.native={'DEBUG':True}
        reservation.bind_child.side_effect=lambda value:events.append(('bound',value))
        actual=dict(pid=301,create_time=3001.0,UID=41833,command=command)
        child=mock.Mock(pid=301)
        def popen(*args,**kwargs):events.append(('Popen',args[0]));return child
        with mock.patch.object(runner,'_ACTIVE_RESERVATION',reservation),\
                mock.patch.object(runner.resources,'check_initial_GPU_idle',side_effect=lambda _:events.append(('idle',True))),\
                mock.patch.object(runner._subprocess,'Popen',side_effect=popen),\
                mock.patch('psutil.Process'),\
                mock.patch('hiercp_v1x.basic_cp_ablation_resources._actor',return_value=actual):
            self.assertIs(runner._ReservedSubprocess().Popen(command),child)
        self.assertEqual([e[0] for e in events],['idle','Popen','bound'])
        self.assertEqual(events[-1][1],actual)

    def test_foreign_gpu_at_last_admission_prevents_popen(self):
        command=['DEBUG_native'];reservation=mock.Mock();reservation.claim={'command':command};reservation.native={}
        with mock.patch.object(runner,'_ACTIVE_RESERVATION',reservation),\
                mock.patch.object(runner.resources,'check_initial_GPU_idle',side_effect=RuntimeError('DEBUG foreign GPU')),\
                mock.patch.object(runner._subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'foreign GPU'):runner._ReservedSubprocess().Popen(command)
            popen.assert_not_called()

    def test_actor_encloses_all_stages_in_reservation_before_execution(self):
        events=[]
        class Reservation:
            def __enter__(self):events.append('claim');return self
            def __exit__(self,*args):events.append('release')
        def execute(*args):
            self.assertIsNotNone(runner._ACTIVE_RESERVATION)
            events.append('actor');return 'whole actual actor result'
        for stage in ('preflight','debug_B2','debug_B4','train','historical_metrics','candidate_bank'):
            events.clear()
            with mock.patch.object(runner,'_admit_execution'),mock.patch.object(runner,'_provider_command',side_effect=lambda _,c,l:c),\
                    mock.patch.object(runner.resources,'allocation_reservation',return_value=Reservation()),\
                    mock.patch.object(runner,'_ACTOR',side_effect=execute):
                self.assertEqual(runner.actor({},['DEBUG'],stage,{}),'whole actual actor result')
            self.assertEqual(events,['claim','actor','release']);self.assertIsNone(runner._ACTIVE_RESERVATION)

    def queue_fixture(self):
        events=[]
        class Reservation:
            def __enter__(self):events.append('claim');return self
            def __exit__(self,*args):events.append('release')
        return events,Reservation

    def test_unstarted_cpu_gpu_race_queues_after_releasing_claim(self):
        events,Reservation=self.queue_fixture()
        failure=runner.GPUAdmissionUnavailable({'pid':999,'DEBUG_outside_actor':True})
        with mock.patch.object(runner,'_admit_execution'),mock.patch.object(runner,'_provider_command',side_effect=lambda _,c,l:c),\
                mock.patch.object(runner.resources,'allocation_reservation',side_effect=lambda *a:Reservation()),\
                mock.patch.object(runner,'_ACTOR',side_effect=[failure,'full source actor result']),\
                mock.patch.object(runner.historical,'status'),mock.patch.object(runner,'_lease_for'),\
                mock.patch.object(runner.time,'sleep',side_effect=lambda _:events.append('wait')):
            self.assertEqual(runner.actor({},['DEBUG_source'],'candidate_bank',{}),'full source actor result')
        self.assertEqual(events,['claim','release','wait','claim','release'])

    def test_started_cpu_actor_failure_is_never_blindly_retried(self):
        events,Reservation=self.queue_fixture()
        def started(*args):
            runner._ACTIVE_RESERVATION.popen_started=True
            raise runner.GPUAdmissionUnavailable({'DEBUG_outside_actor':True})
        with mock.patch.object(runner,'_admit_execution'),mock.patch.object(runner,'_provider_command',side_effect=lambda _,c,l:c),\
                mock.patch.object(runner.resources,'allocation_reservation',side_effect=lambda *a:Reservation()),\
                mock.patch.object(runner,'_ACTOR',side_effect=started) as executed,\
                mock.patch.object(runner.time,'sleep') as wait:
            with self.assertRaises(runner.GPUAdmissionUnavailable):runner.actor({},['DEBUG_metrics'],'historical_metrics',{})
            self.assertEqual(executed.call_count,1);wait.assert_not_called()
        self.assertEqual(events,['claim','release'])

    def test_native_gpu_retry_remains_with_original_debug_output_workflow(self):
        events,Reservation=self.queue_fixture()
        with mock.patch.object(runner,'_admit_execution'),\
                mock.patch.object(runner.resources,'allocation_reservation',side_effect=lambda *a:Reservation()),\
                mock.patch.object(runner,'_ACTOR',side_effect=runner.GPUAdmissionUnavailable({'DEBUG_outside_actor':True})),\
                mock.patch.object(runner.time,'sleep') as wait:
            with self.assertRaises(runner.GPUAdmissionUnavailable):runner.actor({},['DEBUG_native'],'debug_B2',{})
            wait.assert_not_called()

    def test_full512_typed_ram_shortage_queues_without_holding_mutex(self):
        events,Reservation=self.queue_fixture()
        failure=runner.resources.RAMAdmissionUnavailable(host_total_bytes=1008*2**30,
            available_bytes=300*2**30,group_resident_bytes=200*2**30,required_bytes=512*2**30)
        with mock.patch.object(runner,'_admit_execution'),\
                mock.patch.object(runner.resources,'allocation_reservation',side_effect=lambda *a:Reservation()),\
                mock.patch.object(runner,'_ACTOR',side_effect=[failure,'whole full model actor']),\
                mock.patch.object(runner.historical,'status'),mock.patch.object(runner,'_lease_for'),\
                mock.patch.object(runner.time,'sleep',side_effect=lambda _:events.append('wait')):
            self.assertEqual(runner.actor({},['DEBUG_native'],'train',{}),'whole full model actor')
        self.assertEqual(events,['claim','release','wait','claim','release'])

    def test_private_actor_refreshes_all_workers_before_each_guard(self):
        source=inspect.getsource(runner._private_actor)
        self.assertIn("'_refresh_current_reservation'",source)
        self.assertIn('loops[0].body.insert(1',source)
        self.assertIs(runner._ACTOR.__globals__['_refresh_current_reservation'],runner._refresh_current_reservation)
        self.assertIs(runner._SCOPE['resource_guard'],runner.resource_guard)

    def test_archive_copy_is_guarded_after_actual_full_measurement(self):
        source=inspect.getsource(runner._guarded_archive_function)
        self.assertIn("target.id=='measured'",source)
        self.assertIn('ast.With',source)
        self.assertIs(runner._ARCHIVE.__globals__['_archive_context'],runner._archive_context)
        from tools import basic_cp_expanded_result
        self.assertIsNot(runner._ARCHIVE,basic_cp_expanded_result.archive_completed)
        self.assertNotIn('_archive_context',basic_cp_expanded_result.archive_completed.__globals__)


if __name__=='__main__':unittest.main()
