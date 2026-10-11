"""CPU DEBUG metadata fixtures; no production model, GPU, process or SSH."""
import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v24_native_startup_retry as retry


HEADER = ('\n#######################################################################\n'
    'Please cite the following paper when using nnU-Net:\n'
    'Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). '
    'nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. '
    'Nature methods, 18(2), 203-211.\n'
    '#######################################################################\n \n')


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='DEBUG_zero_update_retry_',
            dir=str(Path(__file__).absolute().parents[1]))
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).absolute()
        if os.name == 'nt':
            # Restricted Windows hardlinks are unavailable; emulate exclusive
            # new-file publication only in these task-owned CPU fixtures.
            def fixture_link(source, target):
                with Path(target).open('xb') as stream:
                    stream.write(Path(source).read_bytes())
            linker = patch.object(retry.os, 'link', side_effect=fixture_link)
            linker.start(); self.addCleanup(linker.stop)
        self.arm = self.base/'gpu6'; self.arm.mkdir()
        self.native = self.arm/'native'; self.native.mkdir()
        self.fold = self.native/'nnUNet_results/dataset/trainer/fold_0'; self.fold.mkdir(parents=True)
        self.old_code = self.base/'old_code'; self.old_code.mkdir()
        self.new_code = self.base/'new_code'; self.new_code.mkdir()
        self.group = self.base/'storage'; self.group.mkdir()
        self.old = dict(format='v24_scored_bank_to_native_chain_request_v1', GPU_arm=6,
            chain_root=str(self.arm), code=str(self.old_code), python='/actual/python3.10',
            native_claim_path=str(self.base/'claim.json'), source_output='/actual/gnn', source_code='/actual/source',
            storage_binding=str(self.group/'storage_binding.json'), storage_admission=str(self.group/'storage_admission.json'),
            stages=[dict(action='prepare-native'), dict(action='calibrate-native'),
                dict(command=['/actual/python3.10', 'old_entry', 'train'])])
        self.write(self.arm/'request.json',self.old)
        actor = [self.old['python'], '-B', '-u', str(self.old_code/'tools/watch_v24_scored_bank_to_nnunet.py'),
                 '--request', str(self.arm/'request.json')]
        self.state = dict(status='FAILED', stage='train', error=retry.FAILED_ERROR, GPU_arm=6,
            UID=41833, resource_violation=None, stages_completed=['prepare_native','calibrate_native'],
            full_native_training_completed=False, full_native_evaluation_completed=False,
            request_sha256=retry.digest(self.arm/'request.json'), command=self.old['stages'][2]['command'],
            stage_log=str(self.arm/'train.log'), supervisor_pid=100, supervisor_create_time=1.25,
            child_pid=101, child_create_time=2.25, actual_CPU_affinity=retry.AFFINITY)
        self.write(self.arm/'status.json', self.state)
        self.write(self.arm/'registration.json',dict(format='registered',root=str(self.arm), GPU_arm=6,
            command=actor, request_sha256=self.state['request_sha256'],pid=100,create_time=1.25,
            CPU_affinity=retry.AFFINITY))
        self.write(self.base/'claim.json',dict(root=str(self.arm)))
        self.write(self.arm/'lease.json',dict(supervisor_pid=100,supervisor_create_time=1.25,
            command=actor,request_sha256=self.state['request_sha256'],claim=self.old['native_claim_path'],
            claim_sha256=retry.digest(self.base/'claim.json')))
        self.write(self.arm/'admitted_scoring.json',dict(complete_sources=642, complete_positions=82176))
        self.write(self.arm/'train.log','original parent failure', text=True)
        self.metadata = dict(root=str(self.native), bank_sha256='a'*64, split=dict(outer_val=['case'+str(i) for i in range(26)]),
            checkpoint_coordination=dict(folds=dict(gpu6=str(self.fold))))
        self.write(self.native/'native.json',self.metadata)
        self.write(self.native/'calibration.json',dict(debug=True, production_updates=0))
        log = ('Traceback (most recent call last):\n'
            '    from .v24_nnunet_cp import require_project_budget, snapshot\n'
            +retry.IMPORT_ERROR+' ('+str(self.old_code/'hiercp_v1x/v24_nnunet_cp.py')+')\n')
        self.write(self.native/'train_123.log',log,text=True)
        self.write(self.native/'training_started_123.json',dict(command=[self.old['python'],'-B','-u','-c',
            'from hiercp_v1x.v24_matched_basic_cp_runtime import run_training_entry; run_training_entry()', *retry.NATIVE_ARGV],
            epochs=250,physical_batch=2,debug=False,bank_sha256='a'*64))
        self.write(self.fold/'training_log_2026_10_12_03_58_35.txt',HEADER,text=True)
        self.assertEqual(len(HEADER.encode()),404)
        for name in ('matched_storage_prepare_native.json','matched_storage_calibrate_native.json'):
            self.write(self.arm/name,dict(status='COMPLETE'))
        for name in ('storage_binding.json','storage_admission.json'):
            self.write(self.group/name,dict(admission=dict(sha256='b'*64)))
        self.manifest={name:'c'*64 for name in retry.FILES}
        self.request=self.build()
        self.chain=types.SimpleNamespace(REGISTRATION_FORMAT='registered',process_snapshot=Mock(return_value=None))

    def write(self,path,value,*,text=False):
        path.write_text(value if text else json.dumps(value),encoding='utf8',newline='')

    def build(self,**kwargs):
        return retry.build_request(self.arm/'request.json',self.arm/'startup_retry',self.new_code,'d'*40,self.manifest,**kwargs)

    def refresh(self,key):
        self.request['proofs'][key]=retry.proof(self.request['proofs'][key]['path'])

    def test_exact_zero_update_failure_admitted_and_actors_checked(self):
        self.assertEqual(retry.verify_failure(self.request,self.chain),self.state)
        self.assertEqual(self.chain.process_snapshot.call_args_list[1].args,(101,2.25,self.state['command']))
        self.assertEqual(self.request['original_request'],self.request['proofs']['request'])
        self.assertEqual(self.request['original_declaration'],self.old)

    def test_live_old_actor_refused(self):
        self.chain.process_snapshot.return_value=object()
        with self.assertRaisesRegex(RuntimeError,'still live'):
            retry.verify_failure(self.request,self.chain)

    def test_non_snapshot_error_and_changed_failed_state_refused(self):
        self.write(self.native/'train_123.log','Traceback\nImportError: other failure',text=True); self.refresh('native_log')
        with self.assertRaisesRegex(ValueError,'missing-snapshot'):
            retry.verify_failure(self.request,self.chain)
        self.write(self.arm/'status.json',dict(self.state,status='RUNNING'));self.refresh('status')
        with self.assertRaisesRegex(ValueError,'registered failed'):
            retry.verify_failure(self.request,self.chain)

    def test_checkpoint_configuration_and_unaccounted_fold_file_refused(self):
        for target in (self.fold/'checkpoint_latest.pth',self.native/'actual_nnunet_configuration.json',self.fold/'unknown.json'):
            with self.subTest(target=target.name):
                self.write(target,dict(epoch=1))
                with self.assertRaises(ValueError):
                    retry.verify_failure(self.request,self.chain)
                target.unlink()  # This test-owned one-file DEBUG fixture only.

    def test_prior_failure_records_are_hashed_even_while_new_fold_grows(self):
        self.write(self.fold/'new_actual_training_log.txt','Epoch 0',text=True)
        retry.verify_failure(self.request,self.chain,zero=False)
        self.write(self.native/'calibration.json',dict(production_updates=1))
        with self.assertRaisesRegex(ValueError,'Sealed recovery input changed'):
            retry.verify_failure(self.request,self.chain,zero=False)

    def test_exact_inspection_and_native_cli_preserved(self):
        inspection=dict(native=self.metadata,files=[dict(path=(self.fold/'training_log_2026_10_12_03_58_35.txt').relative_to(self.native).as_posix(),
            size=404,sha256=self.request['proofs']['fold_log']['sha256'])])
        request=self.build(inspection=inspection)
        self.assertEqual(request['command'][-len(retry.NATIVE_ARGV):],retry.NATIVE_ARGV)
        inspection['files'][0]['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'inspection differs'):
            self.build(inspection=inspection)

    def test_cross_gpu_new_output_resume_and_wrong_command_refused(self):
        changes=[dict(GPU_arm=5),dict(native_path=str(self.base/'other/native.json')),dict(resume=True),
                 dict(command=self.request['command']+['--c']),dict(physical_batch=1)]
        for changed in changes:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                retry.validate_declaration(dict(self.request,**changed))

    def test_seal_fresh_root_and_single_lease_no_replay(self):
        path=retry.seal_request(self.request)
        self.assertEqual(retry.read(path),self.request)
        with self.assertRaises(FileExistsError):retry.seal_request(self.request)
        lease=self.arm/'native_startup_retry.lease.json'
        retry.publish(lease,dict(request='first'))
        with self.assertRaises(FileExistsError):retry.publish(lease,dict(request='second'))
        self.assertEqual(retry.read(lease),dict(request='first'))

    def test_binding_is_actual_resource_function_identity_and_rebinding_refused(self):
        original=types.SimpleNamespace(require_project_budget=Mock(return_value=dict(actual_process_and_children_RSS_bytes=45)))
        self.assertIs(retry.bind_snapshot(original),original.require_project_budget)
        self.assertEqual(original.snapshot()['actual_process_and_children_RSS_bytes'],45)
        with self.assertRaisesRegex(ValueError,'already bound'):retry.bind_snapshot(original)

    def test_entry_rejects_reduced_cli_before_runtime(self):
        with patch.object(retry,'admit') as admitted:
            with self.assertRaises(ValueError):retry.entry('/unused/request.json',['730','3d_fullres','0'])
            admitted.assert_not_called()

    def test_entry_exclusive_claim_prevents_duplicate_native_call(self):
        path=retry.seal_request(self.request)
        retry.publish(path.parent/'lease.json',dict(request_sha256=retry.digest(path),supervisor_pid=100,
            supervisor_create_time=1.25,command=['exact supervisor']))
        process=types.SimpleNamespace(pid=222,create_time=lambda:3.25,cmdline=lambda:self.request['command'])
        original=types.SimpleNamespace(require_project_budget=Mock(return_value={'actual':1}))
        runtime=types.SimpleNamespace(run_training_entry=Mock(return_value='CPU DEBUG reached original entry'))
        self.chain.process_snapshot.return_value=object()
        modules=(self.chain,object(),runtime,original,object())
        import psutil
        with patch.object(retry,'admit',return_value=(modules,{},{})),patch.object(psutil,'Process',return_value=process):
            self.assertEqual(retry.entry(path,retry.NATIVE_ARGV),'CPU DEBUG reached original entry')
            with self.assertRaises(FileExistsError):retry.entry(path,retry.NATIVE_ARGV)
        self.assertEqual(runtime.run_training_entry.call_count,1)

    def test_private_original_package_import_isolation_in_fresh_cpu_process(self):
        package=self.old_code/'hiercp_v1x';package.mkdir()
        names=('v24_scored_native_chain','v24_matched_basic_cp_pipeline','v24_matched_basic_cp_runtime',
               'v24_nnunet_cp','v24_native_best_eval_runtime')
        for name in names:
            self.write(package/(name+'.py'),'from pathlib import Path\nROOT=Path(__file__).absolute().parents[1]\n',text=True)
        script=('from hiercp_v1x import v24_native_startup_retry as r; '
            'modules=r._load_original({"original_declaration":{"code":'+repr(str(self.old_code))+'}}); '
            'assert all(str(m.ROOT)=='+repr(str(self.old_code))+' for m in modules); '
            'print("CPU original package isolation verified")')
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).absolute().parents[1],
            capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('isolation verified',result.stdout)

    def test_mixed_scientific_module_refused_without_eviction(self):
        foreign=types.ModuleType('hiercp_v1x.other_science');foreign.__file__=str(self.new_code/'foreign.py')
        with patch.dict(sys.modules,{'hiercp_v1x.other_science':foreign}):
            with self.assertRaisesRegex(ValueError,'different checkout'):
                retry._load_original(self.request)
            self.assertIs(sys.modules['hiercp_v1x.other_science'],foreign)

    def test_completion_publishes_original_canonical_full250_best26_proof(self):
        self.write(self.native/'actual_nnunet_configuration.json',dict(epochs=250,physical_batch=2,
            train_cases=105,validation_cases=26,patch_size=[128]*3,CPU_workers=4,source_entries=642,
            candidate_count=128,validation_checkpoint='checkpoint_best.pth',debug=False))
        final=self.fold/'checkpoint_final.pth'; final.write_bytes(b'DEBUG final witness')
        best_path=self.fold/'checkpoint_best.pth';best_path.write_bytes(b'DEBUG best witness')
        receipt=self.fold/'best_validation_complete.json';self.write(receipt,dict(complete=True))
        pipeline=types.SimpleNamespace(NATIVE_FORMAT='native-format',STAGE_FORMAT='stage-format',PROFILE='profile',
            FILES=(),ROOT=self.old_code,stage_proof=Mock(return_value=dict(action='train',physical_GPU=6)))
        best=types.SimpleNamespace(BEST='checkpoint_best.pth',RECEIPT=receipt.name,
            validate_best_validation_receipt=Mock(return_value=dict(checkpoint_path=str(best_path),checkpoint_sha256=retry.digest(best_path))))
        self.assertEqual(retry.publish_completion(self.request,pipeline,best),dict(action='train',physical_GPU=6))
        complete=retry.read(next(self.native.glob('training_complete_*.json')))
        self.assertEqual(complete['epochs'],250);self.assertEqual(complete['checkpoint_role'],'full250_training_completion_witness')
        self.assertEqual(complete['evaluation_checkpoint_path'],str(best_path))
        self.assertEqual(best.validate_best_validation_receipt.call_args.kwargs['expected_cases'],self.metadata['split']['outer_val'])
        self.assertEqual(retry.read(self.arm/'matched_storage_train.json')['source_entries'],642)
        pipeline.stage_proof.assert_called_once_with(self.old,'train')

    def test_failed_or_incomplete_best26_never_publishes_completion(self):
        self.write(self.native/'actual_nnunet_configuration.json',dict(epochs=250,physical_batch=2,
            train_cases=105,validation_cases=26,patch_size=[128]*3,CPU_workers=4,source_entries=642,
            candidate_count=128,validation_checkpoint='checkpoint_best.pth',debug=False))
        pipeline=types.SimpleNamespace()
        best=types.SimpleNamespace(RECEIPT='best_validation_complete.json',validate_best_validation_receipt=Mock(side_effect=ValueError('missing full26')))
        with self.assertRaisesRegex(ValueError,'missing full26'):
            retry.publish_completion(self.request,pipeline,best)
        self.assertEqual(list(self.native.glob('training_complete_*.json')),[])
        self.assertFalse((self.arm/'matched_storage_train.json').exists())


if __name__=='__main__':
    unittest.main()
