"""Offline admission tests; no server, CUDA or experiment state changes."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

from tools import deploy_v25_all_arm_recovery as deployment


def bundle():
    bodies = {name: b'# reviewed execution overlay\n' for name in deployment.REQUIRED}
    bodies['tests/test_v25_all_arm_recovery.py'] = b'# reviewed regression\n'
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, body in bodies.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    payload = output.getvalue()
    manifest = dict(commit='a'*40, sha256=hashlib.sha256(payload).hexdigest(),
        paths=list(bodies), proof={name: dict(previous_sha256=None,
            new_sha256=hashlib.sha256(body).hexdigest()) for name, body in bodies.items()})
    return manifest, payload


def helpers(names, **values):
    syntax = ast.parse(deployment.REMOTE_SCRIPT)
    selected = [node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(pathlib=__import__('pathlib'), json=json, os=__import__('os'), UID=41833)
    namespace.update(values)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<offline deployment admission>', 'exec'), namespace)
    return namespace


def gpu0_contract():
    return dict(persistent_archive_bytes=2**30,canonical_tmpfs_reserve_bytes=32*2**30,
        GPU0_native_tmpfs_required_bytes=13862461217)


class AllArmDeploymentAdmissionTests(unittest.TestCase):
    def test_both_complete_phases_are_valid_python(self):
        manifest, payload = bundle()
        for phase in ('stage', 'start'):
            with self.subTest(phase=phase):
                result = deployment.build_remote_script(manifest, payload, phase)
                ast.parse(result)
                self.assertNotIn('START_IMPLEMENTATION', result)
                self.assertNotIn('MANIFEST_DATA', result)

    def test_changed_archive_is_rejected_before_any_remote_action(self):
        manifest, payload = bundle()
        with self.assertRaisesRegex(ValueError, 'archive bytes'):
            deployment.build_remote_script(manifest, payload+b'changed', 'stage')

    def test_unreviewed_and_scientific_overlay_are_not_admitted(self):
        for extra in ('../outside.py', 'hiercp_v1x/v25_training.py'):
            with self.subTest(extra=extra):
                manifest, payload = bundle()
                manifest['paths'].append(extra)
                manifest['proof'][extra] = dict(previous_sha256=None, new_sha256='b'*64)
                with self.assertRaisesRegex(ValueError, 'overlay'):
                    deployment.build_remote_script(manifest, payload, 'stage')

    def test_live_exact_old_actor_cannot_be_restarted(self):
        old = dict(pid=99, create_time=123.45, command=['python', 'train'], UID=41833)
        process = SimpleNamespace(create_time=lambda: old['create_time'], uids=lambda: SimpleNamespace(real=41833))
        functions = helpers({'released'}, psutil=SimpleNamespace(Process=lambda pid: process, NoSuchProcess=ProcessLookupError))
        with self.assertRaisesRegex(RuntimeError, 'still exists'):
            functions['released'](old)

    def test_reused_pid_is_not_treated_as_released(self):
        old = dict(pid=99, create_time=123.45, command=['python', 'train'], UID=41833)
        process = SimpleNamespace(create_time=lambda: 124, uids=lambda: SimpleNamespace(real=41833))
        functions = helpers({'released'}, psutil=SimpleNamespace(Process=lambda pid: process, NoSuchProcess=ProcessLookupError))
        with self.assertRaisesRegex(RuntimeError, 'PID reused'):
            functions['released'](old)

    def test_only_absent_exact_old_actor_is_released(self):
        def missing(pid):
            raise ProcessLookupError(pid)
        functions = helpers({'released'}, psutil=SimpleNamespace(Process=missing, NoSuchProcess=ProcessLookupError))
        functions['released'](dict(pid=99, create_time=123.45, command=['python', 'train'], UID=41833))

    def test_complete_native_and_existing_GPU0_archive_reserves_required(self):
        functions = helpers({'extra_storage_admission'})
        original = dict(actual_free_bytes=10*2**30, required_free_bytes=8*2**30)
        value = functions['extra_storage_admission'](original, 2**29, 500, gpu0_contract())
        self.assertEqual(value['required_free_bytes'], 9*2**30+500)
        self.assertEqual(value['additional_reserve_bytes'], 2**30)
        self.assertEqual(value['existing_GPU0_persistent_archive_bytes'], 2**30)
        self.assertEqual(value['checkpoint_backup_storage'], 'tmpfs')
        self.assertEqual(value['checkpoint_backup_bytes'], 2**29)
        self.assertTrue(value['original_2GiB_reserve_1GiB_growth_preserved'])
        self.assertFalse(value['original_storage_admission_rewritten'])
        with self.assertRaisesRegex(OSError, 'do not fit'):
            functions['extra_storage_admission'](dict(actual_free_bytes=9*2**30+499,
                required_free_bytes=8*2**30), 2**29, 500, gpu0_contract())

    def test_original_minimum_reserve_and_growth_cannot_be_dropped(self):
        functions = helpers({'extra_storage_admission'})
        with self.assertRaisesRegex(ValueError, 'Original 2GiB reserve'):
            functions['extra_storage_admission'](dict(actual_free_bytes=10*2**30,
                required_free_bytes=3*2**30-1), 1, 500, gpu0_contract())

    def test_exact_original_budget_and_complete_backup_boundary_is_admitted(self):
        functions = helpers({'extra_storage_admission'})
        original = dict(actual_free_bytes=9*2**30+500, required_free_bytes=8*2**30)
        value = functions['extra_storage_admission'](original, 485774438, 500, gpu0_contract())
        self.assertEqual(value['required_free_bytes'], original['actual_free_bytes'])
        with self.assertRaisesRegex(OSError, 'do not fit'):
            functions['extra_storage_admission'](original, 485774438, 501, gpu0_contract())

    def test_all_tmpfs_backups_fit_with_unchanged_canonical_and_GPU0_budgets(self):
        functions = helpers({'tmpfs_backup_admission'})
        backup_bytes=485774438
        required=32*2**30+13862461217+backup_bytes
        memory=416*2**30+backup_bytes
        value=functions['tmpfs_backup_admission'](backup_bytes,required,memory,gpu0_contract())
        self.assertEqual(value['required_tmpfs_free_bytes'],required)
        self.assertEqual(value['required_available_RAM_bytes'],memory)
        self.assertTrue(value['tmpfs_recovery_backup_not_persistent'])
        self.assertTrue(value['original_primary_checkpoints_remain_on_disk'])
        self.assertTrue(value['server_reboot_can_remove_backup'])

    def test_tmpfs_or_RAM_capacity_shortfall_blocks_before_any_backup(self):
        functions = helpers({'tmpfs_backup_admission'})
        backup_bytes=485774438
        required=32*2**30+13862461217+backup_bytes
        memory=416*2**30+backup_bytes
        with self.assertRaisesRegex(OSError,'do not fit'):
            functions['tmpfs_backup_admission'](backup_bytes,required-1,memory,gpu0_contract())
        with self.assertRaisesRegex(MemoryError,'do not fit'):
            functions['tmpfs_backup_admission'](backup_bytes,required,memory-1,gpu0_contract())

    def test_receipt_publish_does_not_overwrite_prior_intent(self):
        functions = helpers({'publish'})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'intent.json'
            functions['publish'](path, {'PID': 1})
            with self.assertRaises(FileExistsError):
                functions['publish'](path, {'PID': 2})
            self.assertEqual(json.loads(path.read_text()), {'PID': 1})

    def test_duplicate_command_flag_is_rejected(self):
        functions = helpers({'argument'})
        with self.assertRaisesRegex(ValueError, 'Exact argument'):
            functions['argument'](['python', '--arm', 'A', '--arm', 'D'], '--arm')

    def test_negative_measurement_is_rejected(self):
        functions = helpers({'extra_storage_admission'})
        with self.assertRaisesRegex(ValueError, 'Measured disk'):
            functions['extra_storage_admission'](dict(actual_free_bytes=-1,
                required_free_bytes=0), 1, 500, gpu0_contract())


    def test_independent_copy_preserves_original_inode_metadata_and_bytes(self):
        import os, shutil
        functions=helpers({'sha','metadata','same','copy_immutable'},regular=lambda path:Path(path),
            hashlib=hashlib,shutil=shutil)
        with tempfile.TemporaryDirectory()as folder:
            source=Path(folder)/'source.zip';target=Path(folder)/'new.zip';source.write_bytes(b'complete runtime archive')
            original=functions['metadata'](source);digest=functions['sha'](source)
            proof=functions['copy_immutable'](source,target,digest)
            self.assertEqual(proof,original)
            self.assertEqual(functions['metadata'](source),original)
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertNotEqual(source.stat().st_ino,target.stat().st_ino)
            self.assertEqual(source.stat().st_nlink,original['st_nlink'])

    def test_same_bytes_but_changed_original_timestamp_blocks_stage(self):
        import os, shutil
        copier=shutil.copyfileobj
        with tempfile.TemporaryDirectory()as folder:
            source=Path(folder)/'source.py';target=Path(folder)/'new.py';source.write_bytes(b'unchanged bytes')
            def changed_copy(stream,output,size):
                copier(stream,output,size)
                info=source.stat();os.utime(source,ns=(info.st_atime_ns,info.st_mtime_ns+10**9))
            functions=helpers({'sha','metadata','same','copy_immutable'},regular=lambda path:Path(path),
                hashlib=hashlib,shutil=SimpleNamespace(copyfileobj=changed_copy,copystat=shutil.copystat))
            with self.assertRaisesRegex(RuntimeError,'metadata changed'):
                functions['copy_immutable'](source,target,functions['sha'](source))
            self.assertEqual(source.read_bytes(),b'unchanged bytes')

    def test_copy_does_not_overwrite_a_prior_owned_target(self):
        import shutil
        functions=helpers({'sha','metadata','same','copy_immutable'},regular=lambda path:Path(path),
            hashlib=hashlib,shutil=shutil)
        with tempfile.TemporaryDirectory()as folder:
            source=Path(folder)/'source.py';target=Path(folder)/'prior.py'
            source.write_bytes(b'full source');target.write_bytes(b'prior result')
            before=functions['metadata'](source)
            with self.assertRaises(FileExistsError):
                functions['copy_immutable'](source,target,functions['sha'](source))
            self.assertEqual(target.read_bytes(),b'prior result')
            self.assertEqual(functions['metadata'](source),before)

    def test_every_runtime_package_and_full_v1_archive_is_copied(self):
        profile=helpers({'execution_profile'})['execution_profile']
        for path in ('hiercp_v1x/v25_factory.py','tools/run_v25_execution.py',
                'custom_trainers/nnUNetTrainer.py','config/v25.json',
                'versions/v1/pipeline_v1_source.zip','versions/v1/README.md'):
            with self.subTest(path=path):self.assertTrue(profile(path,set()))

    def test_only_inactive_history_documents_and_bytecode_are_omitted(self):
        profile=helpers({'execution_profile'})['execution_profile']
        for path in ('versions/v2.2/history/old/source.zip','versions/v2.2/history/old/code.txt',
                'versions/v2.2/documentation.zip','docs/old_results.md','tools/__pycache__/old.pyc'):
            with self.subTest(path=path):self.assertFalse(profile(path,set()))

    def test_explicit_scientific_or_runtime_dependency_overrides_history_filter(self):
        profile=helpers({'execution_profile'})['execution_profile']
        path='versions/v2.2/history/admitted/source.zip'
        self.assertTrue(profile(path,{path}))
        self.assertFalse(profile(path,set()))

    def test_complete_code_capacity_keeps_native_GPU0_and_growth_reserves(self):
        functions=helpers({'stage_capacity'})
        code_bytes=16*2**20;overlay=2**18;original=8*2**30
        needed=original+2**30+code_bytes+overlay+2**20
        value=functions['stage_capacity'](dict(required_free_bytes=original,actual_free_bytes=needed),
            code_bytes,overlay,gpu0_contract())
        self.assertEqual(value['required_free_bytes'],needed)
        self.assertTrue(value['original_native_and_GPU0_budgets_preserved'])
        with self.assertRaisesRegex(OSError,'does not fit'):
            functions['stage_capacity'](dict(required_free_bytes=original,actual_free_bytes=needed-1),
                code_bytes,overlay,gpu0_contract())

    def test_all_four_curricula_preserve_complete_checkpoint_contracts(self):
        from copy import deepcopy
        functions=helpers({'validate_checkpoint_binding'})
        for arm,initial in dict(A=8,B=16,C=32,D=128).items():
            binding=dict(epochs=40,workers=4,debug=False,physical_patient_batch=4,
                train_cases=list(range(65)),val_cases=list(range(65,86)),
                config=dict(v24_runtime=dict(curriculum=dict(total_epochs=40,total_u=128,
                    initial_u=initial,increment_u=None if arm=='D'else initial))))
            functions['validate_checkpoint_binding'](binding,arm)
            for key,value in(('epochs',39),('workers',1),('debug',True),('physical_patient_batch',1),
                    ('train_cases',list(range(64))),('val_cases',list(range(20)))):
                changed=deepcopy(binding);changed[key]=value
                with self.subTest(arm=arm,key=key),self.assertRaisesRegex(ValueError,'Original arm'):
                    functions['validate_checkpoint_binding'](changed,arm)
            changed=deepcopy(binding);changed['config']['v24_runtime']['curriculum']['total_u']=64
            with self.assertRaisesRegex(ValueError,'Original arm'):
                functions['validate_checkpoint_binding'](changed,arm)


if __name__ == '__main__':
    unittest.main()
