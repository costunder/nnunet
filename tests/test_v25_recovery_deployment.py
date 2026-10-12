"""CPU UNIT safety checks; no remote calls, training, signals or live files."""
import ast
import copy
import hashlib
from types import SimpleNamespace
import unittest
from tools import deploy_v25_recovery_provider as deploy


class RecoveryDeploymentCPUUnit(unittest.TestCase):
    def fixture(self):
        payload=b'CPU_UNIT archive fixture, not a deployment'
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

    def test_generated_stage_start_parse_without_execution(self):
        for phase in ('stage','start'):
            code=self.script(phase);ast.parse(code)
            self.assertIn("phase='"+phase+"'",code)
            self.assertNotIn('MANIFEST_DATA',code)

    def test_exact_reviewed_overlay_rejects_science_missing_duplicates_and_bad_SHA(self):
        manifest,payload=self.fixture()
        for change in ('science','duplicate','missing','hash'):
            mutated=copy.deepcopy(manifest)
            if change=='science':
                mutated['paths'].append('hiercp_v1x/v25_training.py')
                mutated['proof']['hiercp_v1x/v25_training.py']=dict(previous_sha256=None,new_sha256='b'*64)
            elif change=='duplicate':mutated['paths'].append(mutated['paths'][0])
            elif change=='missing':
                name='tools/run_v25_recovery_execution.py';mutated['paths'].remove(name);del mutated['proof'][name]
            else:mutated['proof'][mutated['paths'][0]]['previous_sha256']='bad'
            with self.subTest(change=change),self.assertRaises(ValueError):
                deploy.build_remote_script(mutated,payload,'stage')

    def test_no_remote_signal_broad_deletion_or_session_shutdown_call(self):
        forbidden={'kill','killpg','terminate','send_signal','unlink','rmtree','remove','system','exit'}
        for node in ast.walk(ast.parse(self.script())):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute)else node.func.id if isinstance(node.func,ast.Name)else None
                self.assertNotIn(name,forbidden)

    def test_mutable_latest_excluded_allfour_static_inputs_preserved(self):
        tree=ast.parse(self.script('stage'))
        inputs=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inputs')
        text=ast.unparse(inputs)
        self.assertNotIn('checkpoint_latest',text)
        self.assertIn("for arm in 'ABCD'",text)
        self.assertIn('training_identity.json',text)
        self.assertIn('calibration.json',text)

    def test_B_D_actual_raw_SHA_and_validation_cursors_protected(self):
        code=self.script()
        self.assertIn('42bb39ded6043a42a6b7b48553216466360db00e9de0cea5a5ac03d8c4e844b3',code)
        self.assertIn('ee5354757756c1f7115ee8b60b17a9a173fd8f6e19c5aa9e1ee2e29f61a8b456',code)
        self.assertIn("!=(2,34,'full_validation',65,18,16)",code)
        self.assertIn("!=(2,34,'stage_validation',65,12,128)",code)
        self.assertIn('validate_curriculum_progress(state,policy=',code)
        self.assertIn("train_cases=binding['train_cases'],val_cases=binding['val_cases'],physical_patient_batch=binding['physical_patient_batch']",code)
        self.assertIn("digest(own['binding'])!=own['identity_sha256']",code)

    def test_exact_A_D_marker_tokens_birth_and_phase_safe_checkpoint_before_archive(self):
        code=self.script()
        self.assertIn('GPU6_A_ALL4_CONTROLLER_20261012_2443276',code)
        self.assertIn('GPU4_D_BD_CONTROLLER_20261012_2467341',code)
        self.assertIn('(2443276,1791802623.53,2443117,1791802599.57)',code)
        self.assertIn('(2467341,1791805734.47,2467215,1791805714.84)',code)
        self.assertLess(code.index("checkpoints={arm:checkpoint(arm)"),code.index('os.rename(current,target)'))
        self.assertIn("for identity in actors.values():released(identity)",code)
        self.assertIn("released(previous_D['controller_identity'])",code)

    def test_A_natural_failure_omits_marker_only_for_exact_owned_original_train(self):
        tree=ast.parse(self.script());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='pause_release')
        text=ast.unparse(node)
        self.assertIn("arm == 'A' and (not record.exists()) and (not current.exists())",text)
        self.assertIn("status.get('stage') != 'train'",text)
        self.assertIn("failed['pid'] != 2443276",text)
        self.assertIn("failed['create_time'] != 1791802623.53",text)
        self.assertIn("old_launch(active=False)",text)
        self.assertIn('Unmatched own pause registration',text)

    def test_global_FINAL_and_BD_intents_prevent_duplicate_launch_before_marker_rename(self):
        code=self.script()
        self.assertIn('globalintentpath,BDintentpath,intentpath,launchpath,sealpath,*proofpaths.values()',code)
        self.assertLess(code.index('publish(globalintentpath,intent)'),code.index('os.rename(current,target)'))
        self.assertLess(code.index('os.rename(current,target)'),code.index('child=subprocess.Popen('))
        self.assertIn("same(target,archived_markers[arm]['sha256'])",code)

    def test_allfour_owned_GPUs_original_CPU24_416GiB_and_GPU0_exclusion(self):
        code=self.script()
        self.assertIn("('A',6,list(range(32,38))),('B',5,list(range(42,48))),('C',1,list(range(16,22))),('D',4,list(range(6)))",code)
        self.assertIn("{oldproofs[a]['GPU_UUID'] for a in 'ABCD'}",code)
        self.assertIn("row['limit_bytes']<416*2**30",code)
        self.assertIn("psutil.virtual_memory().available<416*2**30",code)
        self.assertIn("GPUs=[6,5,1,4]",code)
        self.assertNotIn('sibling_controller_identity',code)
        self.assertNotIn('sibling_manifest',code)

    def test_four_checkpoint_measured_backup_plus_GPU0_archive_keeps_original_V25_budget(self):
        admit=self.function('extra_archive_admission');backup=4*161236194
        original=13356112786;exact=original+backup+2**30
        result=admit(dict(actual_free_bytes=exact,required_free_bytes=original),backup)
        self.assertEqual(result['required_free_bytes'],exact)
        self.assertEqual(result['checkpoint_backup_bytes'],backup)
        self.assertEqual(result['GPU0_persistent_BEST26_reservation_bytes'],2**30)
        self.assertFalse(result['original_storage_admission_rewritten'])
        with self.assertRaises(OSError):admit(dict(actual_free_bytes=exact-1,required_free_bytes=original),backup)
        with self.assertRaises(ValueError):admit(dict(actual_free_bytes=exact,required_free_bytes=original),0)

    def test_release_requires_actual_absence_rejects_zombie_live_PID_reuse_and_other_UID(self):
        class Absent(Exception):pass
        identity=dict(pid=100,create_time=1.5,command=['CPU_UNIT'],UID=41833)
        def missing(pid):raise Absent()
        released=self.function('released',UID=41833,psutil=SimpleNamespace(Process=missing,NoSuchProcess=Absent,STATUS_ZOMBIE='zombie'))
        self.assertIsNone(released(identity))
        for status,birth,uid in (('zombie',1.5,41833),('live',1.5,41833),('live',2.0,41833),('live',1.5,999)):
            process=SimpleNamespace(create_time=lambda:birth,uids=lambda:SimpleNamespace(real=uid),status=lambda:status)
            released=self.function('released',UID=41833,psutil=SimpleNamespace(Process=lambda pid:process,NoSuchProcess=Absent,STATUS_ZOMBIE='zombie'))
            with self.subTest(status=status,birth=birth,uid=uid),self.assertRaises(RuntimeError):released(identity)


if __name__=='__main__':unittest.main()
