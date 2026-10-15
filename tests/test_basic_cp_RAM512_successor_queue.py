"""CPU DEBUG queue orchestration checks; never contacts or signals real actors."""
import ast
import copy
import importlib.util
import contextlib
import io
import sys
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

FILE=Path(__file__).resolve().parents[1]/'tools/watch_basic_cp_RAM512_successor_queue.py'
spec=importlib.util.spec_from_file_location('successor_queue_DEBUG',FILE)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def done():
    return dict(status='COMPLETE',stage='complete',completed_epochs=250,evaluated_cases=26,
        full_native_training=True,full_native_evaluation=True,historical_metrics_complete=True,
        reports={'schema1':dict(path='/report',sha256='a'*64)},archive=dict(path='/archive',sha256='b'*64))


def manifest(first=False,second=False):
    arms={0:'multi5',1:'blend50',2:'multi10',3:'candidates512',4:'candidates1024',5:'candidates2048',6:'multi7'}
    if first:arms[0],arms[5]=arms[5],arms[0]
    if second:arms[4],arms[6]=arms[6],arms[4]
    return dict(group='/current',code_sha256={'tools/persist_basic_cp_seven.py':'c'*64},
        registration={str(g):dict(arm=a,root='/current/GPU'+str(g)+'_'+a,
            controller_identity=dict(pid=100+g,create_time=12.,UID=41833,command=['own',str(g)]))for g,a in arms.items()})


class QueueWorkerDEBUGTests(unittest.TestCase):
    def test_complete_requires_every_real_completion_field(self):
        self.assertTrue(worker.complete(done()))
        for key in done():
            value=done();value.pop(key)
            self.assertFalse(worker.complete(value),key)
        for key,value in [('completed_epochs',249),('evaluated_cases',25),('completed_epochs',250.),('full_native_evaluation',False),('reports',{})]:
            state=done();state[key]=value;self.assertFalse(worker.complete(state),key)

    def test_pending_initial_and_each_partial_order(self):
        self.assertEqual([r['arm']for r in worker.pending_routes(manifest())],['multi5','candidates1024'])
        self.assertEqual([r['arm']for r in worker.pending_routes(manifest(True))],['candidates1024'])
        self.assertEqual([r['arm']for r in worker.pending_routes(manifest(False,True))],['multi5'])
        self.assertEqual(worker.pending_routes(manifest(True,True)),[])

    def test_placements_serialize_both_independent_orders(self):
        initial=manifest();before=copy.deepcopy(initial)
        self.assertEqual(worker.placements(initial,[worker.ROUTES[0]]),{int(g):r['arm']for g,r in manifest(True)['registration'].items()})
        self.assertEqual(worker.placements(initial,[worker.ROUTES[1]]),{int(g):r['arm']for g,r in manifest(False,True)['registration'].items()})
        self.assertEqual(worker.placements(manifest(True),[worker.ROUTES[1]]),worker.placements(manifest(False,True),[worker.ROUTES[0]]))
        self.assertEqual(initial,before)

    def test_ready_first_upstream_does_not_wait_for_second(self):
        def status(path):return done()if 'GPU5_'in str(path)else dict(status='RUNNING',stage='train')
        with patch.object(worker,'read',side_effect=status),patch.object(worker,'gone',return_value=True),patch.object(Path,'glob',return_value=[]):
            self.assertEqual(worker.ready_routes(manifest()),[worker.ROUTES[0]])

    def test_ready_second_upstream_does_not_wait_for_first(self):
        def status(path):return done()if 'GPU6_'in str(path)else dict(status='RUNNING',stage='train')
        with patch.object(worker,'read',side_effect=status),patch.object(worker,'gone',return_value=True),patch.object(Path,'glob',return_value=[]):
            self.assertEqual(worker.ready_routes(manifest()),[worker.ROUTES[1]])

    def test_ready_requires_controller_and_all_stage_actors_released(self):
        with patch.object(worker,'read',return_value=done()),patch.object(worker,'gone',return_value=False),patch.object(Path,'glob',return_value=[]):
            self.assertEqual(worker.ready_routes(manifest()),[])
        actor=dict(pid=999,create_time=2.,UID=41833,command=['own'])
        def load(path):return actor if str(path).endswith('_actor.json')else done()
        def released(value):return value['pid']!=999
        with patch.object(worker,'read',side_effect=load),patch.object(worker,'gone',side_effect=released),patch.object(Path,'glob',return_value=[Path('/own_actor.json')]):
            self.assertEqual(worker.ready_routes(manifest()),[])

    def test_unknown_upstream_mapping_fails_before_dispatch(self):
        wrong=manifest();wrong['registration']['5']['arm']='multi10'
        with self.assertRaises(ValueError):worker.ready_routes(wrong)

    def can_pause(self,states=None,live_controllers=None,live_trainers=None,finals=()):
        states=states or {};live_controllers=set(range(7))if live_controllers is None else set(live_controllers)
        live_trainers=set()if live_trainers is None else set(live_trainers)
        def load(path):
            path=Path(path);gpu=int(path.parent.name.split('_')[0].removeprefix('GPU'))
            if path.name=='native.json':return dict(gpu=gpu)
            if path.name=='status.json':return states.get(gpu,dict(status='WAITING',stage='waiting_GPU'))
            if path.name=='train_actor.json':return dict(pid=200+gpu)
            raise AssertionError(path)
        def released(actor):
            return actor['pid']-200 not in live_trainers if actor['pid']>=200 else actor['pid']-100 not in live_controllers
        def exists(path):
            path=Path(path)
            if path.name=='train_actor.json':return True
            if path.name=='checkpoint_final.pth':return int(path.parent.name.removeprefix('fold'))in finals
            raise AssertionError(path)
        with patch.object(worker,'read',side_effect=load),patch.object(worker,'gone',side_effect=released),patch.object(worker,'fold',side_effect=lambda n:Path('/fold'+str(n['gpu']))),patch.object(Path,'exists',exists):
            return worker.can_pause_all(manifest())

    def test_can_pause_waits_for_complete_controller_release(self):
        self.assertFalse(self.can_pause(states={1:done()}))
        self.assertTrue(self.can_pause(states={1:done()},live_controllers={0,2,3,4,5,6},finals={1}))

    def test_can_pause_waits_for_non_native_active_phase(self):
        for stage in('calibration','preflight','evaluation','archive','BEST_validation'):
            self.assertFalse(self.can_pause(states={2:dict(status='RUNNING',stage=stage)}),stage)

    def test_can_pause_requires_actual_live_native_actor(self):
        state={2:dict(status='RUNNING',stage='train')}
        self.assertFalse(self.can_pause(states=state))
        self.assertTrue(self.can_pause(states=state,live_trainers={2}))

    def test_can_pause_defers_existing_final_without_full_completion(self):
        self.assertFalse(self.can_pause(finals={2}))
        self.assertFalse(self.can_pause(states={2:dict(status='RUNNING',stage='train')},live_trainers={2},finals={2}))

    def test_failure_hold_retains_actual_actor_before_manifest_publication(self):
        group=Path('/partial');root=group/'GPU5_multi5';receipt=root/'controller_spawned_actor.json'
        actor=dict(pid=999,create_time=2.,UID=41833,command=['own'])
        def glob(path,pattern):
            if Path(path)==group and pattern=='GPU[0-6]_*':return [root]
            if Path(path)==root and pattern=='*_actor.json':return [receipt]
            return []
        with patch.object(Path,'exists',return_value=False),patch.object(Path,'glob',glob),patch.object(worker,'read',return_value=actor),patch.object(worker,'gone',side_effect=[False,True]),patch.object(worker.time,'sleep')as sleep:
            worker.hold_existing([group])
        sleep.assert_called_once_with(10)

    def pause_mock(self,*,recheck_valid=True,never_valid=False):
        root=Path('/current/GPU2_multi10');group=Path('/fresh');code=Path('/code')
        command=[worker.PY,'-B','-u',str(code/'tools/run_basic_cp_RAM512.py'),'--request',str(root/'request.json')]
        control=dict(pid=102,create_time=12.,UID=41833,command=command)
        train=dict(pid=202,create_time=13.,UID=41833,command=[worker.PY,'-B','-u',command[3],'--native',str(root/'native.json'),'--stage','train'])
        cache=dict(pid=999,create_time=1.,UID=41833,command=['old_cache'])
        row=dict(root=str(root),arm='multi10',controller_identity=control,request_sha256='1'*64)
        m=dict(group='/current',registration={'2':row});request=dict(controller_identity=control)
        native=dict(gpu=2);model=dict(epochs=250,physical_batch=2,effective_batch=2,actual_train_cases=list(range(105)),actual_validation_cases=list(range(26)),parameters=102350575)
        window=dict(epoch=10,latest_inode=[1,2,3,4],BEST_inode=[5,6,7,8])
        latest=dict(completed_epoch=10,source_inode=window['latest_inode']);best=dict(completed_epoch=8,source_inode=window['BEST_inode'])
        helper=SimpleNamespace(training_window=Mock(side_effect=[None,None]if never_valid else [None,window,window,window if recheck_valid else None]),checkpoint_info=Mock(side_effect=[latest,best]))
        spec=SimpleNamespace(loader=Mock(exec_module=Mock()))
        ended={999};seals=[];events=[]
        parent=SimpleNamespace(pid=102,children=lambda recursive=True:[])
        process=SimpleNamespace(pid=202,ppid=lambda:102,children=lambda recursive=True:[])
        def signal(value):events.append(value);ended.update({102,202})
        process.send_signal=Mock(side_effect=signal)
        def exact(actor):return process if actor['pid']==202 else parent
        def load(path):
            name=Path(path).name
            return {'request.json':request,'native.json':native,'train_actor.json':train,'actual_model.json':model,
                'checkpoint_preserver_actor.json':cache,'RAM512_allocation_ledger.json':{'claims':{}},'status.json':dict(status='RUNNING',stage='train')}[name]
        monotonic=[0,1,2,1801]if never_valid else [0,1,2,3]
        failure=None
        with contextlib.ExitStack()as stack:
            stack.enter_context(patch.object(worker.importlib.util,'spec_from_file_location',return_value=spec))
            stack.enter_context(patch.object(worker.importlib.util,'module_from_spec',return_value=helper))
            stack.enter_context(patch.dict(sys.modules,{'torch':SimpleNamespace(set_num_threads=Mock())}))
            stack.enter_context(patch.object(worker,'read',side_effect=load))
            stack.enter_context(patch.object(worker,'sha',return_value='1'*64))
            stack.enter_context(patch.object(worker,'gone',side_effect=lambda a:a['pid']in ended))
            stack.enter_context(patch.object(worker,'exact_actor',side_effect=exact))
            stack.enter_context(patch.object(worker,'capture'))
            stack.enter_context(patch.object(worker,'fold',return_value=Path('/fold')))
            stack.enter_context(patch.object(worker,'seal',side_effect=lambda p,v:seals.append((Path(p).name,copy.deepcopy(v)))))
            stack.enter_context(patch.object(worker,'wait_gone'))
            stack.enter_context(patch.object(Path,'exists',lambda p:Path(p).name=='train_actor.json'))
            stack.enter_context(patch.object(Path,'glob',return_value=[]))
            stack.enter_context(patch.object(worker.time,'sleep'))
            stack.enter_context(patch.object(worker.time,'monotonic',side_effect=monotonic))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:worker.pause_cohort(m,group,code)
            except ValueError as error:failure=str(error)
        return process.send_signal,helper,seals,failure

    def test_pause_waits_for_verified_window_and_signals_exact_native_once(self):
        send,helper,seals,failure=self.pause_mock()
        self.assertIsNone(failure)
        send.assert_called_once_with(worker.signal.SIGINT)
        self.assertEqual(helper.training_window.call_count,4)
        self.assertEqual(helper.checkpoint_info.call_count,2)
        intent=next(v for name,v in seals if name=='single_signal_2.intent.json')
        self.assertEqual(intent['actor']['pid'],202)
        self.assertEqual(intent['checkpoint_proof']['early_next_epoch_window']['epoch'],10)
        self.assertEqual(intent['checkpoint_proof']['LATEST']['source_inode'],[1,2,3,4])

    def test_pause_stale_last_window_never_signals(self):
        send,helper,seals,failure=self.pause_mock(recheck_valid=False)
        send.assert_not_called()
        self.assertIn('signal window changed',failure)
        self.assertFalse(any(name.startswith('single_signal_')for name,_ in seals))

    def test_pause_no_verified_window_stops_at_bound_without_signal(self):
        send,helper,seals,failure=self.pause_mock(never_valid=True)
        send.assert_not_called();helper.checkpoint_info.assert_not_called()
        self.assertIn('No verified saved-epoch window',failure)
        self.assertFalse(any(name.startswith('single_signal_')for name,_ in seals))

    def test_handoff_subprocess_builder_adopter_and_post_apis(self):
        current=manifest();new=dict(current,group='/new',commit='d'*40)
        refs=dict(path='/full_source_proof.json',sha256='e'*64)
        config=dict(full_source_proof=refs,queue_plan=dict(path='/queue_plan.json',sha256='f'*64))
        export=dict(code_sha256={'tools/basic_cp_RAM512_completed_log_adapter.py':'a'*64,'tools/persist_basic_cp_seven.py':'b'*64})
        calls=[]
        def read(path):
            if Path(path).name=='queue_plan.json':return dict(format='basic_cp_approved_waiting_GPU_dependencies_v1',queues=copy.deepcopy(worker.ROUTES))
            if Path(path).name=='launch.json':return new
            raise AssertionError(path)
        def run(command,**kwargs):calls.append((command,kwargs))
        group=Path('/fresh')
        with patch.object(Path,'mkdir'):
            with patch.object(worker,'confirm_upstream_evidence'),patch.object(worker,'can_pause_all',return_value=True),patch.object(worker,'publish_code',return_value=export),patch.object(worker,'seal'),patch.object(worker,'pause_cohort',return_value=group/'released.json'),patch.object(worker,'read',side_effect=read),patch.object(worker,'pin',side_effect=lambda r:Path(r['path'])),patch.object(worker,'sha',return_value='1'*64),patch.object(worker,'run_checked',side_effect=run):
                self.assertEqual(worker.handoff(config,current,[worker.ROUTES[0]],group),new)
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0][0],[worker.PY,'-B','-'])
        wrapper=calls[0][1]['stdin'];tree=ast.parse(wrapper)
        argv=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Attribute)and n.targets[0].attr=='argv')
        args=ast.literal_eval(argv.value.elts[1].value)
        for flag in('--new-group','--current-manifest','--current-manifest-sha256','--release-proof','--full-source-proof','--code-export','--placement-receipt','--queue-plan','--queue-plan-sha256'):
            self.assertIn(flag,args)
        self.assertIn('private_builder_main',wrapper)
        adopter=calls[1][0]
        self.assertEqual(Path(adopter[3]).name,'adopt_basic_cp_RAM512_cache_transfer.py')
        for flag in('--current-manifest','--new-manifest','--old-manifest-sha256','--old-helper-actor-sha256','--new-code-export','--new-code-export-sha256','--new-commit'):
            self.assertIn(flag,adopter)
        self.assertIn('m.postlaunch(',calls[2][1]['stdin'])
        for _,kwargs in calls:self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')

if __name__=='__main__':unittest.main()
