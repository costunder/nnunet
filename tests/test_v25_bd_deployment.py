"""CPU UNIT deployment-script safety; never executes remote code or signals."""
import ast
import copy
import hashlib
import unittest
from tools import deploy_v25_bd_provider_recovery as deploy


class BDDeploymentCPUUnit(unittest.TestCase):
    def fixture(self):
        payload=b'CPU_UNIT archive bytes; not an actual deployment'
        paths=sorted(deploy.PATHS)
        manifest=dict(commit='a'*40,sha256=hashlib.sha256(payload).hexdigest(),paths=paths,
            proof={name:dict(previous_sha256=None,new_sha256='b'*64) for name in paths})
        return manifest,payload

    def test_generated_stage_and_start_parse_without_execution(self):
        manifest,payload=self.fixture()
        for phase in ('stage','start'):
            code=deploy.build_remote_script(manifest,payload,phase)
            ast.parse(code)
            self.assertIn("phase='"+phase+"'",code)
            self.assertNotIn('MANIFEST_DATA',code)

    def test_unreviewed_science_paths_duplicates_and_missing_sources_rejected(self):
        manifest,payload=self.fixture()
        for change in ('science','duplicate','missing','hash'):
            mutated=copy.deepcopy(manifest)
            if change=='science':
                mutated['paths'].append('hiercp_v1x/v25_training.py')
                mutated['proof']['hiercp_v1x/v25_training.py']=dict(previous_sha256=None,new_sha256='b'*64)
            elif change=='duplicate':mutated['paths'].append(mutated['paths'][0])
            elif change=='missing':
                name='tools/run_v25_bd_execution.py';mutated['paths'].remove(name);del mutated['proof'][name]
            else:mutated['proof'][mutated['paths'][0]]['previous_sha256']='bad'
            with self.subTest(change=change),self.assertRaises(ValueError):
                deploy.build_remote_script(mutated,payload,'stage')

    def test_no_signal_or_broad_file_deletion_in_remote_ast(self):
        manifest,payload=self.fixture();tree=ast.parse(deploy.build_remote_script(manifest,payload,'start'))
        forbidden={'kill','killpg','terminate','send_signal','unlink','rmtree','remove','system'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else None
                self.assertNotIn(name,forbidden)

    def test_mutable_latest_is_excluded_from_runtime_immutable_input_table(self):
        manifest,payload=self.fixture();tree=ast.parse(deploy.build_remote_script(manifest,payload,'stage'))
        inputs=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='inputs')
        text=ast.unparse(inputs)
        self.assertNotIn('checkpoint_latest',text)
        self.assertIn('training_identity.json',text)
        self.assertIn('calibration.json',text)

    def test_only_exact_own_marker_is_renamed_after_pause_and_checkpoint_validation(self):
        manifest,payload=self.fixture();code=deploy.build_remote_script(manifest,payload,'start')
        self.assertIn("TOKEN='GPU4_D_BD_CONTROLLER_20261012_2467341'",code)
        self.assertIn("value['child']['create_time']!=1791805734.47",code)
        self.assertIn("value['controller']['create_time']!=1791805714.84",code)
        self.assertIn("state['status']!='PAUSED'",code)
        self.assertLess(code.index("checkpoints={arm:checkpoint(arm)"),code.index('os.rename(marker,archived_marker)'))
        self.assertLess(code.index('pause2,bytes2=pause_release()'),code.index('os.rename(marker,archived_marker)'))
        self.assertEqual(code.count('os.rename('),1)

    def test_original_B_exact_cursor_and_SHA_and_duplicate_launch_intent_are_protected(self):
        manifest,payload=self.fixture();code=deploy.build_remote_script(manifest,payload,'start')
        self.assertIn('42bb39ded6043a42a6b7b48553216466360db00e9de0cea5a5ac03d8c4e844b3',code)
        self.assertIn("!=(2,34,'full_validation',65,18,16)",code)
        self.assertIn('globalintentpath,intentpath,launchpath,sealpath,*proofpaths.values()',code)
        self.assertLess(code.index('publish(intentpath,'),code.index('child=subprocess.Popen('))
        self.assertIn('same(target,checkpoints[arm]',code)

    def test_measured_backup_bytes_and_GPU0_archive_add_to_unchanged_V25_reservation(self):
        manifest,payload=self.fixture();tree=ast.parse(deploy.build_remote_script(manifest,payload,'start'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='extra_archive_admission')
        scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'CPU_UNIT_extra_storage','exec'),scope)
        admit=scope['extra_archive_admission'];backup=161073826+161236194
        exact=13356112786+backup+2**30
        result=admit(dict(actual_free_bytes=exact,required_free_bytes=13356112786),backup)
        self.assertEqual(result['checkpoint_backup_bytes'],backup)
        self.assertEqual(result['GPU0_persistent_BEST26_reservation_bytes'],2**30)
        self.assertEqual(result['required_free_bytes'],exact)
        self.assertFalse(result['original_storage_admission_rewritten'])
        with self.assertRaises(OSError):admit(dict(actual_free_bytes=exact-1,required_free_bytes=13356112786),backup)
        with self.assertRaises(ValueError):admit(dict(actual_free_bytes=exact,required_free_bytes=13356112786),0)

    def test_global_intent_precedes_marker_rename_and_process_launch(self):
        manifest,payload=self.fixture();code=deploy.build_remote_script(manifest,payload,'start')
        self.assertLess(code.index('publish(globalintentpath,intent)'),code.index('os.rename(marker,archived_marker)'))
        self.assertLess(code.index('os.rename(marker,archived_marker)'),code.index('child=subprocess.Popen('))
        self.assertIn('same(archived_marker,hashlib.sha256(pause_bytes).hexdigest())',code)
        self.assertIn('awaits reaping; inspect later, no signal',code)

    def test_original_AC_and_GPU0_remain_outside_launch_selection(self):
        manifest,payload=self.fixture();code=deploy.build_remote_script(manifest,payload,'start')
        self.assertIn("for arm,gpu,cpus in (('B',5,list(range(42,48))),('D',4,list(range(6))))",code)
        self.assertIn("for arm in 'AC'",code)
        self.assertIn("identity['pid']!=2443117",code)
        self.assertIn("row['limit_bytes']<416*2**30",code)
        self.assertIn("GPU4/5 live CUDA actor exists; preserve it",code)


if __name__=='__main__':unittest.main()
