"""DEBUG CPU-only execution wiring tests; no scientific training is represented."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import types
import unittest
from unittest import mock

from hiercp_v1x import v24_gpu0_native_execution as execution


class GPU0NativeExecutionDebug(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd()/'tmp/DEBUG_gpu0_native_execution'
        self.request = dict(format=execution.FORMAT,source_arm=4,execution_physical_gpu=0,
            execution_gpu_uuid='GPU-8023fae3-1440-85a0-896d-98d5ad191683',
            source_code=str(self.root/'original'),adapter_code=str(self.root/'adapter'),
            source_code_files={'hiercp_v1x/v24_matched_basic_cp_pipeline.py':'1'*64},
            adapter_code_files={name:'2'*64 for name in execution.FILES},
            python=str(self.root/'python'),metadata_root=str(self.root/'metadata'),
            execution_root=str(self.root/'tmpfs/gpu4'),native_path=str(self.root/'tmpfs/gpu4/native/native.json'),
            resource_proof_path=str(self.root/'metadata/resource_proof.json'),
            overlay=dict(path=str(self.root/'old/score_overlay.json'),sha256='3'*64),
            storage_admission=dict(path=str(self.root/'fresh/storage_admission.json'),sha256='4'*64),
            science=copy.deepcopy(execution.SCIENCE))
        self.request_path=self.root/'metadata/request.json'; self.sha='5'*64

    def test_full_contract_source_and_execution_separate(self):
        value=execution.validate_request(self.request)
        self.assertEqual(value['source_arm'],4)
        self.assertEqual(value['execution_physical_gpu'],0)
        self.assertEqual(value['science']['physical_batch'],2)
        self.assertEqual(value['science']['validation_cases'],26)

    def test_any_scale_or_final_checkpoint_change_rejected(self):
        for key,value in (('epochs',1),('physical_batch',1),('training_cases',2),
                          ('candidate_count',8),('evaluation_checkpoint','checkpoint_final.pth')):
            with self.subTest(key=key):
                changed=copy.deepcopy(self.request); changed['science'][key]=value
                with self.assertRaises(ValueError):execution.validate_request(changed)

    def test_source_arm_relabel_and_other_hardware_rejected(self):
        for key,value in (('source_arm',0),('execution_physical_gpu',4),('execution_physical_gpu',False)):
            changed=copy.deepcopy(self.request);changed[key]=value
            with self.assertRaises(ValueError):execution.validate_request(changed)

    def test_source_overlay_cannot_include_unsealed_alternate_witness(self):
        changed=copy.deepcopy(self.request);changed['overlay']['stat']=[]
        with self.assertRaises(ValueError):execution.validate_request(changed)

    def test_all_runtime_adapters_must_be_sha_admitted(self):
        for name in execution.FILES:
            changed=copy.deepcopy(self.request);changed['adapter_code_files'].pop(name)
            with self.assertRaises(ValueError):execution.validate_request(changed)

    def test_native_path_rebinding_rejected(self):
        changed=copy.deepcopy(self.request);changed['native_path']=str(self.root/'other/native.json')
        with self.assertRaises(ValueError):execution.validate_request(changed)

    def test_clone_reenters_adapter_preserving_original_batch_and_output(self):
        prefix=[self.request['python'],'-B','-u',str(Path(self.request['source_code'])/'tools/run_v24_matched_basic_cp.py'),'clone-debug-worker']
        for batch in (2,4):
            original=prefix+['--native',self.request['native_path'],'--physical-batch',str(batch),'--output',str(self.root/'DEBUG')]
            result=execution.rewrite_child(original,self.request,self.request_path,self.sha)
            self.assertIn('clone-debug-worker',result)
            self.assertEqual(result[-6:],original[-6:])
            self.assertEqual(original[:5],prefix)

    def test_train_reenters_adapter_with_exact_original_cli(self):
        original=[self.request['python'],'-B','-u','-c',execution.TRAIN_ENTRY,*execution.NATIVE_ARGV]
        result=execution.rewrite_child(original,self.request,self.request_path,self.sha)
        self.assertEqual(result[-len(execution.NATIVE_ARGV)-1:],['--',*execution.NATIVE_ARGV])
        self.assertIn('train-entry',result)
        with self.assertRaises(ValueError):
            execution.rewrite_child(original[:-1],self.request,self.request_path,self.sha)

    def test_unknown_child_has_no_fallback(self):
        for child in ([self.request['python'],'-B','-u','-c','print(1)'],
                      [self.request['python'],'-B','-u','unknown.py']):
            with self.assertRaises(ValueError):execution.rewrite_child(child,self.request,self.request_path,self.sha)

    def guard(self):
        value=execution.ExecutionGuard.__new__(execution.ExecutionGuard)
        value.request=self.request
        return value

    def test_gpu_check_admits_only_exact_actual_uuid_and_tree(self):
        guard=self.guard();row='0, '+self.request['execution_gpu_uuid']+', 40960, 40500, Disabled'
        with mock.patch.object(execution,'query',side_effect=[row,self.request['execution_gpu_uuid']+', 321']),mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':self.request['execution_gpu_uuid']}):
            guard.gpu_check({321:object()},initial=True)
        with mock.patch.object(execution,'query',side_effect=[row,self.request['execution_gpu_uuid']+', 322']),mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':self.request['execution_gpu_uuid']}):
            with self.assertRaisesRegex(RuntimeError,'outside'):guard.gpu_check({321:object()})

    def test_wrong_gpu_uuid_mig_free_vram_or_cvd_rejected(self):
        guard=self.guard()
        for row in ('0, GPU-other, 40960, 40500, Disabled',
                    '0, '+self.request['execution_gpu_uuid']+', 40960, 40500, Enabled',
                    '0, '+self.request['execution_gpu_uuid']+', 40960, 1000, Disabled'):
            with mock.patch.object(execution,'query',return_value=row):
                with self.assertRaises(RuntimeError):guard.gpu_check({},initial=True)
        row='0, '+self.request['execution_gpu_uuid']+', 40960, 40500, Disabled'
        with mock.patch.object(execution,'query',side_effect=[row,'']),mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-wrong'}):
            with self.assertRaisesRegex(RuntimeError,'visibility'):guard.gpu_check({})

    def test_physical_selection_happens_before_cuda_and_requires_source4(self):
        guard=self.guard();fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:False))
        with mock.patch.object(execution.ExecutionGuard,'__call__',return_value={}),mock.patch.dict(execution.sys.modules,{'torch':fake}),mock.patch.dict(os.environ,{},clear=True):
            self.assertEqual(guard.select(4),self.request['execution_gpu_uuid'])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],self.request['execution_gpu_uuid'])
            with self.assertRaises(ValueError):guard.select(0)
            fake.cuda.is_initialized=lambda:True
            with self.assertRaisesRegex(RuntimeError,'precede'):guard.select(4)

    def test_original_scientific_functions_delegate_without_code_rewrite(self):
        def original(*args,**kwargs):return 'untouched'
        pipeline=types.SimpleNamespace(**{name:original for name in ('prepare_native','calibrate_native','calibration_worker','train','environment','_child')})
        runtime=types.SimpleNamespace();old=types.SimpleNamespace(require_project_budget=original);device=types.SimpleNamespace(select=original)
        guard=mock.Mock();guard.select=mock.Mock()
        with mock.patch.object(execution,'load_original',return_value=(pipeline,runtime,old,device)):
            returned,_,binding=execution.install(self.request,self.request_path,self.sha,guard)
        self.assertIs(returned.train,original)
        self.assertIs(returned.train.__code__,original.__code__)
        self.assertIs(old.snapshot,old.require_project_budget)
        self.assertTrue(binding['original_native_functions_unchanged'])
        self.assertIs(device.select,guard.select)

    def test_original_child_redirect_preserves_environment_and_global_subprocess(self):
        def original(*args,**kwargs):return 'untouched'
        pipeline=types.SimpleNamespace(**{name:original for name in ('prepare_native','calibrate_native','calibration_worker','train','environment','_child')})
        runtime=types.SimpleNamespace();old=types.SimpleNamespace(require_project_budget=original);device=types.SimpleNamespace(select=original)
        child=[self.request['python'],'-B','-u','-c',execution.TRAIN_ENTRY,*execution.NATIVE_ARGV]
        with mock.patch.object(execution,'load_original',return_value=(pipeline,runtime,old,device)),mock.patch.object(execution.subprocess,'Popen') as actual:
            execution.install(self.request,self.request_path,self.sha,mock.Mock())
            pipeline.subprocess.Popen(child,env={'nnUNet_results':'original-results','PYTHONPATH':'private-nnunet', 'V24_CONTEXT_DEPTH1_CHAIN_REQUEST':'old-hardware-request'})
            rewritten=actual.call_args.args[0];env=actual.call_args.kwargs['env']
            self.assertIn('train-entry',rewritten)
            self.assertEqual(env['nnUNet_results'],'original-results')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],self.request['execution_gpu_uuid'])
            self.assertNotIn('V24_CONTEXT_DEPTH1_CHAIN_REQUEST',env)
            self.assertIs(execution.subprocess.Popen,actual)

    def test_exact_identity_rejects_foreign_uid_and_birth_reuse(self):
        identity=dict(pid=123,create_time=1.5,command=['DEBUG_actor'],UID=execution.UID)
        with mock.patch.object(execution.psutil,'Process') as process,mock.patch.object(execution,'actor',return_value=identity):
            self.assertIs(execution.exact(identity),process.return_value)
            changed=copy.deepcopy(identity);changed['UID']=0
            with self.assertRaises(ValueError):execution.exact(changed)
        reused=copy.deepcopy(identity);reused['create_time']=1.6
        with mock.patch.object(execution.psutil,'Process'),mock.patch.object(execution,'actor',return_value=reused):
            with self.assertRaisesRegex(RuntimeError,'birth'):execution.exact(identity)

    def test_cgroup_and_rss_admission_fail_closed(self):
        guard=self.guard();guard.path=self.root/'request';guard.sha='a'*64
        guard.proof_path=self.root/'resource';guard.proof_sha='a'*64
        guard.lifetime_path=self.root/'allocation';guard.last_checked=0
        observed=dict(controller_cgroup='DEBUG cgroup',memory_limits=[dict(path='/sys/fs/cgroup/memory.max',limit_bytes=464*execution.GIB)])
        guard.proof=dict(host_lifetime={'allocation_request_sha256':'a'*64},cgroup_admission=observed,
            CPU_affinity=[6,7,8,9],controller_identity={'DEBUG':'mock'})
        controller=types.SimpleNamespace(pid=11);own=types.SimpleNamespace(pid=12)
        process=mock.Mock();process.memory_info.return_value.rss=1*execution.GIB
        with mock.patch.object(execution.os,'getuid',return_value=execution.UID,create=True),mock.patch.object(execution.socket,'gethostname',return_value='ece-a6gpu6'),mock.patch.object(execution,'digest',return_value='a'*64),mock.patch.object(execution.ExecutionGuard,'tree',return_value=(controller,own,{11:process})),mock.patch.object(execution.ExecutionGuard,'lifetime_check'),mock.patch.object(execution.ExecutionGuard,'gpu_check'),mock.patch.object(execution,'capture_cgroup_admission',return_value=observed),mock.patch.object(execution.psutil,'virtual_memory',return_value=types.SimpleNamespace(available=48*execution.GIB)):
            receipt=guard(force=True)
            self.assertEqual(receipt['RSS_limit_bytes'],48*execution.GIB)
            process.memory_info.return_value.rss=49*execution.GIB
            with self.assertRaises(MemoryError):guard(force=True)
            process.memory_info.return_value.rss=1*execution.GIB
            observed['memory_limits'][0]['limit_bytes']=463*execution.GIB
            with self.assertRaisesRegex(MemoryError,'concurrent'):guard(force=True)


if __name__ == '__main__':
    unittest.main()
