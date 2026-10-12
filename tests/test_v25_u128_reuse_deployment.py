"""CPU UNIT deployment safety only; no remote training, signals or fixtures as results."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from tools import deploy_v25_u128_reuse as deploy


class U128ReuseDeploymentCPUUnit(unittest.TestCase):
    def temporary_directory(self):
        root=Path(__file__).resolve().parents[1]/'tmp'
        root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root,prefix='CPU_UNIT_U128_')

    def fixture(self):
        payload=b'CPU_UNIT archive fixture, not deployed'
        paths=sorted(deploy.PATHS)
        manifest=dict(commit='a'*40,sha256=hashlib.sha256(payload).hexdigest(),paths=paths,
            proof={name:dict(previous_sha256=None,new_sha256='b'*64)for name in paths})
        return manifest,payload

    def script(self,phase='start'):
        manifest,payload=self.fixture()
        return deploy.build_remote_script(manifest,payload,phase)

    def function(self,name,**scope):
        node=next(n for n in ast.parse(self.script()).body if isinstance(n,ast.FunctionDef)and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'CPU_UNIT_'+name,'exec'),scope)
        return scope[name]

    def test_stage_and_start_are_parseable_without_running(self):
        for phase in ('stage','start'):
            code=self.script(phase);ast.parse(code)
            self.assertIn("phase='"+phase+"'",code)
            self.assertNotIn('MANIFEST_DATA',code)

    def test_closed_new_overlay_rejects_frozen_science_current_files_missing_duplicates_or_bad_SHA(self):
        manifest,payload=self.fixture()
        for change in ('science','current','duplicate','missing','hash'):
            mutated=copy.deepcopy(manifest)
            if change in ('science','current'):
                name='hiercp_v1x/v25_training.py'if change=='science'else'tools/run_v25_recovery_execution.py'
                mutated['paths'].append(name);mutated['proof'][name]=dict(previous_sha256='c'*64,new_sha256='b'*64)
            elif change=='duplicate':mutated['paths'].append(mutated['paths'][0])
            elif change=='missing':
                name='hiercp_v1x/v25_u128_validation_reuse.py';mutated['paths'].remove(name);del mutated['proof'][name]
            else:mutated['proof'][mutated['paths'][0]]['previous_sha256']='bad'
            with self.subTest(change=change),self.assertRaises(ValueError):
                deploy.build_remote_script(mutated,payload,'stage')

    def test_remote_AST_contains_no_process_signal_broad_delete_or_shutdown(self):
        forbidden={'kill','killpg','terminate','send_signal','unlink','rmtree','remove','system','exit'}
        for node in ast.walk(ast.parse(self.script())):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute)else node.func.id if isinstance(node.func,ast.Name)else None
                self.assertNotIn(name,forbidden)

    def test_current_ab23_snapshot_and_singleton_intent_are_independent(self):
        code=self.script()
        self.assertIn("oldcode=group/'code_RECOVERY_provider_ab23c1622c56'",code)
        self.assertIn("code=group/('code_U128_reuse_'",code)
        self.assertIn('ab23c1622c566e9865a29ca4a5b0ad3e4232c54f',code)
        self.assertIn("identity['pid']!=2488819",code)
        self.assertIn("identity['create_time']!=1791808638.21",code)
        self.assertIn("globalintentpath=group/'U128_validation_reuse_controller_intent.json'",code)
        self.assertNotIn("globalintentpath=group/'FINAL_standalone_controller_intent.json'",code)

    def test_stage_does_not_read_live_checkpoints_or_require_failed_monitors(self):
        tree=ast.parse(self.script('stage'))
        branch=next(n for n in tree.body if isinstance(n,ast.If)and ast.unparse(n.test)=="phase == 'stage'")
        stage=ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
        self.assertNotIn('checkpoint(arm)',stage)
        self.assertNotIn('terminal_arm(',stage)
        self.assertIn('old_launch(active=None)',stage)
        self.assertIn('all_four_live_latest_not_sealed=True',stage)

    def test_mutable_latest_excluded_static_cache_own_calibration_and_identity_preserved(self):
        tree=ast.parse(self.script('stage'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inputs')
        text=ast.unparse(node)
        self.assertNotIn('checkpoint_latest',text)
        self.assertIn('input_bindings.json',text)
        self.assertIn('calibration.json',text)
        self.assertIn('training_identity.json',text)
        self.assertIn('oldlaunchpath',text)
        self.assertIn('oldsealpath',text)

    def test_allfour_dynamic_cursors_use_frozen_complete_progress_without_status_rewrite(self):
        tree=ast.parse(self.script());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='checkpoint')
        text=ast.unparse(node)
        self.assertIn('checkpoint_status(state, mode)',text)
        self.assertIn('validate_curriculum_progress',text)
        self.assertIn("policy=binding['config']['v24_runtime']['curriculum']",text)
        self.assertIn("train_cases=binding['train_cases']",text)
        self.assertIn("val_cases=binding['val_cases']",text)
        self.assertIn("physical_patient_batch=binding['physical_patient_batch']",text)
        self.assertNotIn("arm == 'B'",text)
        self.assertNotIn("arm == 'D'",text)
        self.assertIn("digest(own['binding']) != own['identity_sha256']",text)
        self.assertIn("len(saved['rank_rng']) != 1",text)
        self.assertIn("('model', 'optimizer', 'scheduler', 'scaler')",text)
        self.assertNotIn("state['status'] =",text)
        check=self.function('checkpoint_status')
        for mode,status in (('natural_failed','RUNNING'),('batch_paused','PAUSED')):
            state=dict(status=status)
            self.assertIsNone(check(state,mode));self.assertEqual(state['status'],status)
            with self.assertRaises(ValueError):check(dict(status='PAUSED'if status=='RUNNING'else'RUNNING'),mode)
        with self.assertRaises(ValueError):check(dict(status='RUNNING'),'unverified_failure')

    def test_four_owned_pause_tokens_births_and_archives_checked_before_launch(self):
        code=self.script()
        for row in ('2489145,1791808660.16','2489075,1791808659.79','2489077,1791808659.81','2489076,1791808659.79'):
            self.assertIn(row,code)
        self.assertIn("token='U128_REUSE_'+arm+'_20261012_'+str(pid)",code)
        self.assertIn('v25_owned_validation_reuse_pause_v1',code)
        self.assertIn("value['arm']!=arm",code)
        self.assertIn("value['controller']!=expected",code)
        self.assertLess(code.index("checkpoints={arm:checkpoint(arm,registrations[arm]['mode'])"),code.index('os.rename(current,target)'))
        self.assertLess(code.index('publish(globalintentpath,intent)'),code.index('os.rename(current,target)'))
        self.assertLess(code.index('os.rename(current,target)'),code.index('child=subprocess.Popen('))
        self.assertIn("same(target,archived_markers[arm]['sha256'])",code)

    def test_terminal_scoring_after_pause_keeps_exact_original_stage_binding(self):
        old=Path('/CPU_UNIT/ab23');group=Path('/CPU_UNIT/group');seal=group/'sealed.json'
        proof='/CPU_UNIT/original_A_direct.json'
        previous=dict(slots={'A':dict(direct_proof=dict(path=proof,sha256='a'*64))})
        file=SimpleNamespace(read_bytes=lambda:json.dumps(previous).encode())
        binding=self.function('stage_binding',oldcode=old,group=group,oldsealpath=seal,json=json,regular=lambda path:file)
        def command(mode):
            return ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(old/'tools/run_v25_recovery_execution.py'),
                '--arm','A','--mode',mode,'--output',str(group/'A'),'--continuation-seal',str(seal),'--allocation-request',proof]
        binding(command('train'),'A','train');binding(command('score-cp'),'A','score-cp')
        for change in ('wrongmode','duplicate','foreignproof','foreignoutput','wrongentry'):
            mutated=command('score-cp')
            if change=='wrongmode':mutated[mutated.index('--mode')+1]='train'
            elif change=='duplicate':mutated+=['--arm','A']
            elif change=='foreignproof':mutated[-1]='/CPU_UNIT/foreign.json'
            elif change=='foreignoutput':mutated[mutated.index('--output')+1]='/CPU_UNIT/foreign'
            else:mutated[3]='/CPU_UNIT/unknown.py'
            with self.subTest(change=change),self.assertRaises(ValueError):binding(mutated,'A','score-cp')
        code=self.script()
        self.assertIn("stage not in ('train','scoring')",code)
        self.assertIn("stage_binding(identity['command'],arm,'train'if stage=='train'else'score-cp')",code)
        self.assertIn("status['stage']=='train' and terminal_actor!=value['child']",code)

    def test_original_four_GPUs_CPU24_native416_and_GPU0_untouched(self):
        code=self.script()
        self.assertIn("('A',6,list(range(32,38))),('B',5,list(range(42,48))),('C',1,list(range(16,22))),('D',4,list(range(6)))",code)
        self.assertIn("{oldproofs[a]['GPU_UUID'] for a in 'ABCD'}",code)
        self.assertIn("row['limit_bytes']<416*2**30",code)
        self.assertIn("psutil.virtual_memory().available<416*2**30",code)
        self.assertIn('GPUs=[6,5,1,4]',code)

    def test_measured_four_backup_bytes_and_GPU0_archive_checked_before_any_backup_copy(self):
        code=self.script();admit=self.function('extra_archive_admission')
        backup=4*161236194;original=13356112786;exact=original+backup+2**30
        result=admit(dict(actual_free_bytes=exact,required_free_bytes=original),backup)
        self.assertEqual(result['required_free_bytes'],exact)
        self.assertEqual(result['checkpoint_backup_bytes'],backup)
        self.assertEqual(result['GPU0_persistent_BEST26_reservation_bytes'],2**30)
        self.assertFalse(result['original_storage_admission_rewritten'])
        with self.assertRaises(OSError):admit(dict(actual_free_bytes=exact-1,required_free_bytes=original),backup)
        self.assertLess(code.index('extra_storage=extra_archive_admission('),code.index('archive.mkdir('))

    def test_only_actual_absence_releases_previous_actor_no_zombie_or_PID_reuse(self):
        class Absent(Exception):pass
        identity=dict(pid=100,create_time=1.5,command=['CPU_UNIT'],UID=41833)
        def missing(pid):raise Absent()
        released=self.function('released',UID=41833,psutil=SimpleNamespace(Process=missing,NoSuchProcess=Absent,STATUS_ZOMBIE='zombie'))
        self.assertIsNone(released(identity))
        for status,birth,uid in (('zombie',1.5,41833),('live',1.5,41833),('live',2.0,41833),('live',1.5,999)):
            process=SimpleNamespace(create_time=lambda:birth,uids=lambda:SimpleNamespace(real=uid),status=lambda:status)
            released=self.function('released',UID=41833,psutil=SimpleNamespace(Process=lambda pid:process,NoSuchProcess=Absent,STATUS_ZOMBIE='zombie'))
            with self.subTest(status=status,birth=birth,uid=uid),self.assertRaises(RuntimeError):released(identity)

    def registration_fixture(self,root,arm='A',mode='natural_failed'):
        group=Path(root)/'group';old=group/'ab23';seal=group/'oldseal.json'
        (group/arm/'training').mkdir(parents=True);(group/arm/'monitor').mkdir()
        previous={'A':(2489145,1791808660.16),'B':(2489075,1791808659.79),
                  'C':(2489077,1791808659.81),'D':(2489076,1791808659.79)}
        parent=dict(pid=2488819,create_time=1791808638.21,command=['CPU_UNIT_controller'],UID=41833)
        proof=str(group/'direct.json');seal.write_text(json.dumps(dict(slots={arm:dict(direct_proof=dict(path=proof,sha256='a'*64))})))
        def command(stage):
            return ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',str(old/'tools/run_v25_recovery_execution.py'),
                '--arm',arm,'--mode',stage,'--output',str(group/arm),'--continuation-seal',str(seal),'--allocation-request',proof]
        pid,birth=previous[arm];training=dict(pid=pid,create_time=birth,command=command('train'),UID=41833)
        marker=group/arm/'training/STOP_AFTER_BATCH';record=group/(arm+'_pause_for_u128_reuse_controller_20261012.json')
        value=dict(format='v25_owned_validation_reuse_pause_v1'if mode=='batch_paused'else'v25_owned_validation_reuse_terminal_v1',
            arm=arm,marker=str(marker)if mode=='batch_paused'else None,child=training,controller=parent,created_at=1,
            reason='CPU_UNIT exact owned evidence',full40_GNN_preserved=True,no_signal_sent=True,
            token='U128_REUSE_'+arm+'_20261012_'+str(pid)if mode=='batch_paused'else None)
        log=group/arm/'monitor/old_train.log';log.write_text('RuntimeError: Recorded stage descendant is outside exact controller tree')
        status=dict(status='FAILED',stage='train',controller_pid=parent['pid'],stage_log=str(log),
            child_pid=pid,child_create_time=birth,child_command=command('train'),UID=41833)
        status_path=group/arm/'monitor/status.json';status_path.write_text(json.dumps(status))
        digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if mode=='natural_failed':value.update(status_path=str(status_path),status_sha256=digest(status_path),
            failure_log=str(log),failure_log_sha256=digest(log))
        record.write_text(json.dumps(value));intent=group/'all4_pause_for_u128_reuse_20261012.json'
        intent.write_text(json.dumps(dict(signals_sent=False,records={arm:value})))
        if mode=='batch_paused':marker.write_text(json.dumps(value))
        released=[]
        def same(path,sha):
            if digest(path)!=sha:raise ValueError('CPU_UNIT immutable SHA mismatch')
        scope=dict(group=group,oldcode=old,oldsealpath=seal,PREVIOUS_TRAINING=previous,
            json=json,sha=digest,regular=lambda path:Path(path),same=same,
            old_launch=lambda active:(None,parent),released=lambda actor:released.append(actor))
        nodes=[node for node in ast.parse(self.script()).body if isinstance(node,ast.FunctionDef)
               and node.name in ('stage_binding','registration_base','pause_release','registration_release')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'CPU_UNIT_mixed_registration','exec'),scope)
        return SimpleNamespace(scope=scope,group=group,record=record,marker=marker,intent=intent,value=value,
            status=status,actor=training,command=command,digest=digest,released=released)

    def test_natural_failed_registration_preserves_exact_RUNNNING_origin_and_rejects_forgery(self):
        with self.temporary_directory()as directory:
            f=self.registration_fixture(directory);read=f.scope['registration_release']
            result=read('A',f.status,f.actor)
            self.assertEqual(result['mode'],'natural_failed');self.assertIsNone(result['pause'])
            self.assertEqual(result['failure_log']['sha256'],f.value['failure_log_sha256'])
            self.assertIn(f.actor,f.released)
            for change in ('actor','birth','uid','parent','log','status','token','marker','intent'):
                original=copy.deepcopy(f.value);mutated=copy.deepcopy(original)
                if change=='actor':mutated['child']['command'][mutated['child']['command'].index('--arm')+1]='B'
                elif change=='birth':mutated['child']['create_time']+=1
                elif change=='uid':mutated['child']['UID']=999
                elif change=='parent':mutated['controller']['pid']+=1
                elif change=='log':mutated['failure_log_sha256']='f'*64
                elif change=='status':mutated['status_sha256']='f'*64
                elif change=='token':mutated['token']='unowned'
                elif change=='marker':f.marker.write_text('{}')
                else:f.intent.write_text('{}')
                f.record.write_text(json.dumps(mutated))
                with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):
                    read('A',f.status,f.actor)
                f.record.write_text(json.dumps(original));f.intent.write_text(json.dumps(dict(signals_sent=False,records={'A':original})))
                if f.marker.exists():f.marker.unlink()  # Local CPU fixture only.

    def test_D_natural_failure_archives_only_exact_owned_pending_pause_without_claiming_PAUSED(self):
        with self.temporary_directory()as directory:
            f=self.registration_fixture(directory,arm='D',mode='batch_paused')
            terminal=copy.deepcopy(f.value);terminal.update(format='v25_owned_validation_reuse_terminal_v1',marker=None,token=None)
            path=f.group/'D/monitor/status.json';log=Path(f.status['stage_log'])
            terminal.update(status_path=str(path),status_sha256=f.digest(path),failure_log=str(log),failure_log_sha256=f.digest(log))
            separate=f.group/'D_terminal_for_u128_reuse_controller_20261012.json';separate.write_text(json.dumps(terminal))
            result=f.scope['registration_release']('D',f.status,f.actor)
            self.assertEqual(result['mode'],'natural_failed');self.assertEqual(result['record'],separate)
            self.assertEqual(result['pause'][2],f.marker);self.assertEqual(result['pause'][0],f.value)
            self.assertTrue(f.marker.exists());self.assertEqual(json.loads(f.marker.read_bytes()),f.value)
            corrupted=copy.deepcopy(f.value);corrupted['token']='unknown';f.marker.write_text(json.dumps(corrupted))
            with self.assertRaises(ValueError):f.scope['registration_release']('D',f.status,f.actor)

    def test_paused_terminal_scoring_is_bound_separately_and_mutable_monitors_are_not_sealed(self):
        with self.temporary_directory()as directory:
            f=self.registration_fixture(directory,arm='D',mode='batch_paused')
            status=copy.deepcopy(f.status);status['stage']='scoring'
            scoring=dict(pid=999999,create_time=2.0,command=f.command('score-cp'),UID=41833)
            result=f.scope['registration_release']('D',status,scoring)
            self.assertEqual(result['mode'],'batch_paused')
            with self.assertRaises(ValueError):f.scope['registration_release']('D',f.status,scoring)
        code=self.script()
        self.assertIn("monitor=dict(path=str(archive/(arm+'_status.json')),sha256=row['status_sha256'])",code)
        self.assertNotIn("preserved[str(group/arm/'monitor/status.json')]",code)
        self.assertIn("previous_actor=actors[arm],recovery=recoveries[arm]",code)
        self.assertIn("for arm in archived_markers:",code)


if __name__=='__main__':unittest.main()
