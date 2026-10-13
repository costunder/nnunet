"""CPU-only resource/proof fixtures; no trained-model or server claim."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import v25_B_CP_recovery_resources as recovery
from tests import test_basic_cp_ablation_resources as basic_fixtures
Actor = basic_fixtures.Actor
identity = basic_fixtures.identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf8')
    return dict(path=str(path),sha256=sha(path))


def own_cpu_fixture(path,*,directory=False):
    """Real local path safety; Linux UID observations mocked only in tests."""
    path=Path(path)
    if not path.is_absolute() or '..'in path.parts or any(x.is_symlink()for x in (path,*path.parents)):
        raise ValueError('Fixture path safety failed')
    path=path.resolve(strict=True)
    if directory and not path.is_dir()or not directory and not path.is_file():
        raise ValueError('Fixture regular-file/directory safety failed')
    return path


class RecoveryProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.group=self.root/'group';self.bank=self.group/'old_transition/cp_bank'
        self.bank.mkdir(parents=True)
        self.controller=dict(pid=111,create_time=700.,UID=41833,command=['python','oldcontroller'])
        self.child=dict(pid=222,create_time=701.,UID=41833,command=['python','oldchild'])
        self.new_controller=dict(pid=333,create_time=702.,UID=41833,command=['python','newcontroller'])
        old_seal=write(self.group/'old_seal.json',dict(controller_identity=self.controller,
            slot=dict(monitor_path=str(self.group/'old_monitor.json'))))
        monitor=write(self.group/'old_monitor.json',dict(status='FAILED',stage='scoring',arm='B',
            controller_pid=111,child_pid=222,child_create_time=701.,UID=41833,
            child_command=self.child['command'],B_early_downstream_seal=old_seal['path']))
        self.new_seal=write(self.group/'new_seal.json',{'CPU_fixture':True})
        self.lifetime=write(self.root/'lifetime.json',dict(format=recovery.direct.OWNED_FORMAT,
            UID=41833,owner='aicompetition06',node='ece-a6gpu6',gpu=6,job_id='129776.ECE-util1',expires_at_unix=5000.))
        self.profile={'contract_sha256':'a'*64}
        entries=[f'entries/case_{n:03d}.npz'for n in range(642)]
        sizes=[4]*131+[2,3]+[4]*27+[2,3]
        batches=[];cursor=0
        for size in sizes:batches.append(entries[cursor:cursor+size]);cursor+=size
        self.assertEqual(cursor,642)
        source={'root':str(self.root/'original_bank'),'index_sha256':recovery.SOURCE_SHA,
            'complete_cases':81,'complete_sources':642,'complete_positions':82176,
            'entries':{entry:dict(case_id=f'case_{n%81}',source_component=n,
                sha256=hashlib.sha256(entry.encode()).hexdigest(),non_score_arrays_sha256={'source_mask':'b'*64})for n,entry in enumerate(entries)}}
        declaration=dict(format='v25_owned_BEST_Basic642_durable_score_store_v1',source_batches=batches,
            source_bank=source,context_contract=self.profile,model_sha256='c'*64,
            best_provenance=dict(arm='B',checkpoint=str(self.group/'B/training/checkpoint_best.pt'),
                checkpoint_sha256=recovery.BEST_SHA,full_training=False,best_record=dict(epoch=3,updates=51)),
            cp_contract=dict(total_sources=642,candidates_per_source=128,total_scored_candidates=82176,
                donor_policy='same_patient',cp_probability=.5,train_cases=105,validation_cases=26,
                physical_batch=2,patch=[128,128,128],epochs=250,selection_policy='argmax',
                paste_policy='historical_preprocessed_patch_hard_paste'))
        request=write(self.bank/'request.json',declaration);files={}
        for batch_entries in batches[:133]:
            scores={}
            for entry in batch_entries:
                original=source['entries'][entry]
                row=dict(entry=entry,case_id=original['case_id'],source_component=original['source_component'],
                    original_sha256=original['sha256'],scores=list(range(128)),selected_index=127,
                    model_sha256='c'*64,all_original128_CP_scored=True,all_original_P_in_context=True,
                    recipient_GT_used_in_forward=False,original_non_score_payloads_preserved=True,
                    input_binding=dict(context_contract=self.profile,source_bank_entry_sha256=original['sha256'],
                        original_non_score_arrays_sha256=original['non_score_arrays_sha256'],
                        source_context_geometry=dict(profile_sha256='a'*64,minimum_liver_depth_exclusive_mm=1.,
                            context_inner_radius_mm=0.,context_outer_radius_mm=10.,canonical_context_nodes=10)))
                p=self.bank/'scores'/(hashlib.sha256(entry.encode()).hexdigest()+'.json')
                proof=write(p,dict(format=declaration['format'],request_sha256=request['sha256'],
                    row=row,row_sha256=recovery._canonical(row)))
                files[proof['path']]=proof['sha256'];scores[entry]=proof['sha256']
            p=self.bank/'batches'/(recovery._canonical(batch_entries)+'.json')
            proof=write(p,dict(format=declaration['format'],request_sha256=request['sha256'],
                entries=batch_entries,score_rows_sha256=scores));files[proof['path']]=proof['sha256']
        self.value=dict(format=recovery.FORMAT,B_controller_seal=self.new_seal,
            previous_failed_execution=dict(controller=self.controller,child=self.child,monitor=monitor,controller_seal=old_seal),
            durable=dict(root=str(self.bank),request=request,completed_batches=133,completed_sources=529,
                completed_positions=67712,committed_files_sha256=files),fresh_host_lifetime=self.lifetime,
            PBS_shared_cache_root=str(self.root/'pbs'),gpu=5,gpu_uuid=recovery.GPU_UUID,CPU_affinity=recovery.CPUS,
            runtime_files={str(Path(recovery.__file__).resolve()):sha(recovery.__file__),
                str(Path(recovery.basic.__file__).resolve()):sha(recovery.basic.__file__)},
            created_at=1000.,previous_results_preserved=True,processes_signaled=False)
        self.previous_cache=copy.deepcopy(recovery.previous._IMMUTABLE)
        self.previous_expected=copy.deepcopy(recovery.previous._EXPECTED)
        self.addCleanup(self.restore_cache)
        recovery.previous._IMMUTABLE.clear();recovery.previous._EXPECTED.clear()
        self.owned=patch.object(recovery.direct,'_owned_regular',side_effect=own_cpu_fixture)
        self.owned.start();self.addCleanup(self.owned.stop)

    def restore_cache(self):
        recovery.previous._IMMUTABLE.clear();recovery.previous._IMMUTABLE.update(self.previous_cache)
        recovery.previous._EXPECTED.clear();recovery.previous._EXPECTED.update(self.previous_expected)

    def rewrite_request(self, change):
        request=Path(self.value['durable']['request']['path']);value=json.loads(request.read_bytes());change(value)
        self.value['durable']['request']=write(request,value)

    def test_exact_frozen_full529_prefix_admits_without_writing(self):
        self.assertIs(recovery.validate_recovery_proof(self.value),self.value)
        before={p:p.stat().st_mtime_ns for p in self.bank.rglob('*.json')}
        declaration=recovery._verify_durable(self.value)
        self.assertEqual(len(declaration['source_bank']['entries']),642)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in self.bank.rglob('*.json')})

    def test_528_source_or_other_GPU_CPU_cannot_be_admitted(self):
        for key,replacement in [('gpu',4),('gpu_uuid','GPU-foreign'),('CPU_affinity',[42]),('processes_signaled',True)]:
            value=copy.deepcopy(self.value);value[key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError):recovery.validate_recovery_proof(value)
        value=copy.deepcopy(self.value);value['durable']['completed_sources']=528
        with self.assertRaises(ValueError):recovery.validate_recovery_proof(value)

    def test_unknown_field_or_unsealed_PBS_code_rejected(self):
        value=copy.deepcopy(self.value);value['relax_expiry']=True
        with self.assertRaises(ValueError):recovery.validate_recovery_proof(value)
        value=copy.deepcopy(self.value);value['runtime_files'].pop(str(Path(recovery.basic.__file__).resolve()))
        with self.assertRaisesRegex(ValueError,'shared-PBS'):recovery.validate_recovery_proof(value)

    def test_incomplete_missing_original_batch_is_not_hidden(self):
        path=next(Path(p)for p in self.value['durable']['committed_files_sha256']if Path(p).parent.name=='batches')
        path.unlink()
        with self.assertRaises(FileNotFoundError):recovery._verify_durable(self.value)

    def test_ownBEST_final_other_epoch_or_shrunken_contract_rejected(self):
        cases=[lambda x:x['best_provenance'].update(checkpoint_sha256='d'*64),
            lambda x:x['best_provenance']['best_record'].update(epoch=4),
            lambda x:x['cp_contract'].update(candidates_per_source=64),
            lambda x:x['source_bank'].update(index_sha256='e'*64)]
        for mutation in cases:
            original=Path(self.value['durable']['request']['path']).read_bytes()
            self.rewrite_request(mutation)
            with self.assertRaises(ValueError):recovery._verify_durable(self.value)
            Path(self.value['durable']['request']['path']).write_bytes(original)
            self.value['durable']['request']['sha256']=hashlib.sha256(original).hexdigest()
            recovery.previous._IMMUTABLE.clear();recovery.previous._EXPECTED.clear()

    def test_row_metric_or_context_changes_are_not_accepted(self):
        path=next(Path(p)for p in self.value['durable']['committed_files_sha256']if Path(p).parent.name=='scores')
        original=path.read_bytes()
        for mutate in [lambda x:x['row'].update(selected_index=0),
                lambda x:x['row'].update(recipient_GT_used_in_forward=True),
                lambda x:x['row']['input_binding']['source_context_geometry'].update(minimum_liver_depth_exclusive_mm=4.)]:
            payload=json.loads(original);mutate(payload);payload['row_sha256']=recovery._canonical(payload['row']);write(path,payload)
            with self.assertRaisesRegex(ValueError,'score row changed'):recovery._verify_durable(self.value)
        path.write_bytes(original)

    def test_original529_digest_table_cannot_gain_or_drop_files(self):
        self.value['durable']['committed_files_sha256'].pop(next(iter(self.value['durable']['committed_files_sha256'])))
        with self.assertRaises(ValueError):recovery.validate_recovery_proof(self.value)
        with self.assertRaises(ValueError):recovery._verify_durable(self.value)

    def test_released_old_actor_exact_monitor_identity_required(self):
        with patch.object(recovery.previous,'_released')as released:
            recovery._verify_failed_execution(self.value)
            self.assertEqual([call.args[0]for call in released.call_args_list],[self.controller,self.child])
        value=copy.deepcopy(self.value);value['previous_failed_execution']['child']['create_time']+=1
        with patch.object(recovery.previous,'_released')as released,self.assertRaises(ValueError):
            recovery._verify_failed_execution(value)
        released.assert_not_called()

    def test_still_live_old_controller_blocks_fresh_binding(self):
        with (patch.object(recovery.previous,'_released',side_effect=RuntimeError('still live')),
                self.assertRaisesRegex(RuntimeError,'still live')):
            recovery._verify_failed_execution(self.value)

    def test_immutable_digest_cache_detects_replacement_and_content_metadata(self):
        path=Path(self.value['durable']['request']['path']);digest=sha(path)
        recovery.previous.immutable_digest(path,digest)
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):recovery.previous.immutable_digest(path,digest)

    def test_fresh_lifetime_cannot_extend_or_borrow_foreign_job_identity(self):
        with patch.object(recovery.direct.time,'time',return_value=1000):
            self.assertEqual(recovery._fresh_lifetime(self.value)['job_id'],'129776.ECE-util1')
        with patch.object(recovery.direct.time,'time',return_value=4821),self.assertRaisesRegex(RuntimeError,'180seconds'):
            recovery._fresh_lifetime(self.value)
        document=json.loads(Path(self.lifetime['path']).read_bytes());document['UID']=0
        self.value['fresh_host_lifetime']=write(Path(self.lifetime['path']),document)
        recovery.previous._IMMUTABLE.clear();recovery.previous._EXPECTED.clear()
        with self.assertRaisesRegex(ValueError,'owned GPU6'):recovery._fresh_lifetime(self.value)

    def test_binding_changes_only_inmemory_lifetime_and_new_controller(self):
        proof_path=self.group/'direct.json';proof=write(proof_path,{'CPU_fixture_old_direct':True})
        seal=dict(group_root=str(self.group),controller_identity=self.new_controller,
            controller_cgroup_admission={'fixture_cgroup':'same'},slot=dict(direct_proof=proof))
        lease=recovery.FreshBLifetimeLease.__new__(recovery.FreshBLifetimeLease)
        lease.path=proof_path;lease.group=self.group;lease.self_identity=self.new_controller
        lease.request=dict(controller_identity=self.controller,arm='B',gpu=5,CPU_affinity=recovery.CPUS,
            GPU_UUID=recovery.GPU_UUID,host_lifetime={'old_original_expired':True})
        original=copy.deepcopy(lease.request);before=proof_path.read_bytes()
        with (patch.object(recovery,'load_recovery_proof',return_value=(self.value,seal)),
                patch.object(recovery.previous,'previous_contract',return_value={'controller_identity':self.controller}),
                patch.object(recovery.previous,'verify_initial_release')as release,
                patch.dict(recovery.os.environ,{recovery.previous.ENV_PATH:self.new_seal['path']}),
                patch.object(recovery.direct.time,'time',return_value=1000),
                patch.object(recovery.direct.time,'monotonic',return_value=500)):
            lease._bind_fresh_recovery()
        release.assert_called_once_with(seal)
        self.assertEqual(original['host_lifetime'],{'old_original_expired':True})
        self.assertEqual(proof_path.read_bytes(),before)
        self.assertEqual(lease.request['controller_identity'],self.new_controller)
        self.assertEqual(lease.request['host_lifetime']['expires_at_unix'],5000)
        self.assertEqual(lease.monotonic_expiry,4500)
        self.assertEqual(lease.request['aggregate_CPU_affinity'],recovery.CPUS)
        self.assertTrue(lease._B_rebound)


class SharedPBSAndActorBoundaryTests(unittest.TestCase):
    def setUp(self):
        # Reuse existing Linux-observation fixture, not its inherited tests.
        self.fixture=basic_fixtures.BasicLeaseTests('test_spawned_stage_and_worker_are_accepted_actual_descendants')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        for actor in f.actors.values():actor.cpus=recovery.CPUS
        lease=SimpleNamespace(request=dict(controller_identity=identity(f.controller)),
            self_identity=identity(f.stage),lifetime=f.lease.lifetime,monotonic_expiry=f.lease.monotonic_expiry)
        value=dict(PBS_shared_cache_root=str(f.lease._pbs_shared_root),fresh_host_lifetime=f.value['host_lifetime'])
        self.observer=recovery._observer(lease,value)

    def test_fresh_actual_R_cached_no_duplicate_query_and_original_timestamps(self):
        self.observer._host_lifetime();f=self.fixture
        before=json.loads((self.observer._pbs_shared_root/'observation.json').read_bytes())
        with patch.object(recovery.basic,'_pbs_query',side_effect=AssertionError('duplicate query')):
            self.observer._host_lifetime()
        after=json.loads((self.observer._pbs_shared_root/'observation.json').read_bytes())
        self.assertEqual(before,after)
        self.assertEqual(self.observer._pbs_observation['job_state'],'R')

    def test_timeout_never_refreshes_raw_timestamp_and_300sec_expires(self):
        self.observer._host_lifetime();f=self.fixture;f.monotonic=570
        with patch.object(recovery.basic,'_pbs_query',side_effect=recovery.basic.subprocess.TimeoutExpired(['qstat'],20)):
            self.observer._host_lifetime()
        document=json.loads((self.observer._pbs_shared_root/'observation.json').read_bytes())
        self.assertEqual(document['observed_at_unix'],1000)
        self.assertEqual(document['observed_at_monotonic'],500)
        f.monotonic=801
        with (patch.object(recovery.basic,'_pbs_query',side_effect=recovery.basic.subprocess.TimeoutExpired(['qstat'],20)),
                self.assertRaisesRegex(RuntimeError,'300seconds')):
            self.observer._host_lifetime()

    def test_initial_fresh_actor_cannot_borrow_older_than60sec_R(self):
        self.observer._host_lifetime();f=self.fixture;f.monotonic=570
        other=recovery._observer(self.observer.recovery_lease,dict(PBS_shared_cache_root=str(self.observer._pbs_shared_root),fresh_host_lifetime=f.value['host_lifetime']))
        with (patch.object(recovery.basic,'_pbs_query',side_effect=recovery.basic.subprocess.TimeoutExpired(['qstat'],20)),
                self.assertRaisesRegex(RuntimeError,'60seconds')):
            other._host_lifetime()

    def test_foreign_UID_or_live_finalized_argv_change_rejected(self):
        worker=self.fixture.worker
        self.observer._descendant(worker);worker.command=['python','different_actual_actor']
        with self.assertRaisesRegex(RuntimeError,'live argv'):self.observer._descendant(worker)
        worker.uid=0
        with self.assertRaisesRegex(RuntimeError,'foreign'):self.observer._descendant(worker)

    def test_known_preexec_first_exec_transition_and_natural_exit_preserve_identity(self):
        f=self.fixture;worker=f.worker;worker.fork_no_exec=True;worker.parent_actor=f.stage
        worker.command=f.stage.command.copy();self.observer._descendant(worker)
        worker.fork_no_exec=False;worker.command=['python','actual_final_worker'];self.observer._descendant(worker)
        worker.state=recovery.basic.psutil.STATUS_ZOMBIE
        self.assertIsNone(self.observer._descendant(worker))
        self.assertEqual(self.observer._terminal_observations[worker.pid]['create_time'],worker.birth)

    def test_reused_PID_birth_cannot_be_accepted_as_terminal(self):
        worker=self.fixture.worker;self.observer._descendant(worker)
        worker.birth+=1;worker.state=recovery.basic.psutil.STATUS_ZOMBIE
        with self.assertRaisesRegex(RuntimeError,'PID/birth'):self.observer._descendant(worker)

    def test_GPU_actor_must_be_current_exact_selected_stage_or_cleared_terminal(self):
        f=self.fixture;controller=f.controller;stage=f.stage;worker=f.worker
        self.observer.recovery_lease._tree=lambda:(controller,stage,{p.pid:p for p in [controller,stage,worker]})
        self.observer.recovery_lease._resources=lambda *args:None
        self.observer.recovery_lease._selected_stage_tree=lambda tree:{stage.pid:stage,worker.pid:worker}
        apps=f'{stage.pid}, {recovery.GPU_UUID}'
        def query(argv,**kwargs):
            if '--query-compute-apps=pid,gpu_uuid'in argv:return apps
            return f'5, {recovery.GPU_UUID}, NVIDIA RTX A6000, 46068, 45622, [N/A]'
        with patch.object(recovery.basic,'_query',side_effect=query):
            result=self.observer._gpu({stage.pid:stage,worker.pid:worker},idle=False)
            self.assertEqual(result['own_GPU_compute_PIDs'],[stage.pid])
            foreign=Actor(777,['python','foreign'],cpus=recovery.CPUS);f.actors[777]=foreign
            apps=f'777, {recovery.GPU_UUID}'
            with self.assertRaisesRegex(RuntimeError,'non-admitted'):self.observer._gpu({stage.pid:stage},idle=False)


if __name__=='__main__':unittest.main()
