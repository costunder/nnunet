"""CPU UNIT admission/state fixtures. No model/checkpoint/CUDA training."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from hiercp_v1x import v24_scored_native_chain as chain
from tools import watch_v24_gnn_to_matched_scoring as waiter
from tools import run_v24_completed_matched_scoring as scorer


class ScoredNativeCPUUnit(unittest.TestCase):
    def setUp(self):
        self.repo=Path(__file__).resolve().parents[1]
        self.temp=tempfile.TemporaryDirectory(prefix='scored_native_CPU_UNIT_',dir=self.repo/'outputs')
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.group=self.base/'private';self.group.mkdir()
        for gpu in (4,5,6):(self.group/('gpu'+str(gpu))).mkdir()
        self.code=self.base/'code';self.code.mkdir()
        self.source=self.base/'source';(self.source/'training').mkdir(parents=True)
        self.scoring=self.base/'scoring';self.scoring.mkdir()
        own=patch.object(chain,'owned',side_effect=lambda path,**kw:Path(path));own.start();self.addCleanup(own.stop)
        if os.name=='nt':
            # CPU UNIT exclusive bytes only; actual Linux hardlink atomicity
            # remains production behavior and is exercised by server tests.
            def link(source,target):
                with Path(target).open('xb') as stream:stream.write(Path(source).read_bytes())
            mocked=patch.object(chain.os,'link',side_effect=link);mocked.start();self.addCleanup(mocked.stop)

    def request(self,gpu=4):
        request=chain.build_request(GPU_arm=gpu,GPU_UUID='GPU-CPU-UNIT',chain_root=str(self.group/('gpu'+str(gpu))),
            arm_roots={'gpu'+str(g):str(self.group/('gpu'+str(g))) for g in (4,5,6)},scoring_root=str(self.scoring),
            source_output=str(self.source),source_code=str(self.base/'science'),source_commit='a'*40,
            source_files={'original_science.py':'b'*64},source_execution_code=str(self.base/'execution'),
            source_execution_commit='c'*40,source_execution_files={'original_executor.py':'d'*64},code=str(self.code),
            commit='e'*40,code_files={name:'f'*64 for name in set(chain.FILES)|scorer.REQUIRED_CODE_FILES},
            python=str(self.base/'python'),inventory=str(self.base/'inventory.json'),input_cache=str(self.base/'cache'),
            baseline_preprocessed=str(self.base/'baseline'),historical_basic_bank=str(self.base/'Basic'),
            storage_admission=str(self.group/'storage_admission.json'),storage_binding=str(self.group/'storage_binding.json'),
            scoring_worker=str(self.code/'tools/run_v24_completed_matched_scoring.py'),scoring_worker_sha256='a'*64,
            stunet_checkpoint=str(self.base/'official_STU.pt') if gpu!=5 else None,
            stunet_checkpoint_sha256='b'*64 if gpu!=5 else None,scheduler=None if gpu!=6 else dict(job_id='own.test',state='R',
                expected_owner='aicompetition06',expected_node='ece-a6gpu6',minimum_remaining_seconds=28800,
                native_minimum_remaining_seconds=25200,expires_at_unix=9999999999),PBS_owned_reservation=gpu==6,
            direct_mode='previous_explicit_direct_physical_GPU_number' if gpu!=6 else None)
        return request

    def write(self,path,value):
        Path(path).write_text(json.dumps(value),encoding='utf8');return Path(path)

    def scientific(self,request):
        value=dict(physical_GPU=request['GPU_arm'],config=dict(seed=42,epochs=40,v24_runtime=dict(debug=False,
            hidden_subset=False,all_P_from_epoch_one=True,other_P_as_negative=False,curriculum={'initial_u':128})),
            source=request['source_files'])
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
        import hashlib
        value['request_sha256']=hashlib.sha256(raw).hexdigest()
        self.write(self.source/'request.json',value)

    def invocation(self,request,**changes):
        tensors=981 if request['GPU_arm']==5 else 537
        row=dict(status='COMPLETE',full_training=True,debug=False,actual_CUDA=True,completed_epochs=40,
            optimizer_updates=680,connected_parameter_tensors=tensors,expected_parameter_tensors=tensors)
        row.update(changes);(self.source/'training/invocations.jsonl').write_text(json.dumps(row)+'\n',encoding='utf8')

    def test_three_original_native_contracts_and_exact_external_score_paths(self):
        for gpu in (4,5,6):
            request=self.request(gpu);self.assertEqual(chain.validate_declaration(request),request)
            self.assertEqual(tuple(row['action'] for row in request['stages']),chain.ACTIONS)
            self.assertIn(str(self.scoring/'bank/score_overlay.json'),request['stages'][0]['command'])
            self.assertNotIn('prepare-current-bank',str(request['stages']))
            self.assertEqual((request['physical_batch'],request['epochs'],request['patch_size']),(2,250,[128]*3))

    def test_scale_change_or_rescoring_rejected(self):
        for key,value in dict(physical_batch=1,epochs=249,source_entries=641,train_cases=104,validation_cases=25,
                patch_size=[64]*3,rescore=True,debug=True,context_depth1=False,cp_probability=.4).items():
            with self.subTest(key=key):
                request=self.request();request[key]=value
                with self.assertRaises(ValueError):chain.validate_declaration(request)

    def test_output_overlap_and_tampered_cli_rejected(self):
        request=self.request();request['source_output']=request['chain_root']
        with self.assertRaisesRegex(ValueError,'overlap'):chain.validate_declaration(request)
        request=self.request();request['stages'][2]['command'].append('--resume')
        with self.assertRaisesRegex(ValueError,'no-resume'):chain.validate_declaration(request)

    def test_direct_cannot_borrow_gpu6_scheduler_or_stu_for_cnn(self):
        request=self.request(5);request['scheduler']=self.request(6)['scheduler']
        with self.assertRaises(ValueError):chain.validate_declaration(request)
        request=self.request(5);request['stunet_checkpoint']=str(self.base/'official_STU.pt')
        with self.assertRaisesRegex(ValueError,'CNN'):chain.validate_declaration(request)

    def test_full_invocation_required_and_partial_line_not_complete(self):
        request=self.request();self.scientific(request)
        path=self.source/'training/invocations.jsonl';path.write_text('{"status":"COMPLETE"}',encoding='utf8')
        self.assertIsNone(chain.source_complete(request))
        self.invocation(request,optimizer_updates=679)
        with self.assertRaisesRegex(ValueError,'680'):chain.source_complete(request)
        self.invocation(request);self.assertEqual(chain.source_complete(request)['completed_epochs'],40)

    def test_gpu5_connected_full_model_and_source_hash_guard(self):
        request=self.request(5);self.scientific(request);self.invocation(request,connected_parameter_tensors=537)
        with self.assertRaises(ValueError):chain.source_complete(request)
        self.invocation(request);self.assertEqual(chain.source_complete(request)['connected_parameter_tensors'],981)
        value=chain.read(self.source/'request.json');value['config']['seed']=0;self.write(self.source/'request.json',value)
        with self.assertRaisesRegex(ValueError,'scientific'):chain.source_complete(request)

    def test_partial_metadata_retry_identity_changes(self):
        path=self.base/'partial.json';path.write_bytes(b'{')
        with self.assertRaises(chain.PendingMetadata) as first:chain.read(path)
        path.write_bytes(b'{ ')
        with self.assertRaises(chain.PendingMetadata) as second:chain.read(path)
        self.assertNotEqual(first.exception.signature,second.exception.signature)
        self.write(path,dict(real='complete'));self.assertEqual(chain.read(path),dict(real='complete'))

    def test_cache_hash_once_and_detect_same_size_stat_mutation(self):
        path=self.base/'immutable';path.write_bytes(b'first')
        cache=chain.FileProofCache();proof=cache.capture(path)
        with patch.object(chain,'sha',wraps=chain.sha) as hashed:
            cache.verify(proof);cache.check();self.assertEqual(hashed.call_count,0)
            path.write_bytes(b'other')
            with self.assertRaisesRegex(ValueError,'identity'):cache.verify(proof)
        self.assertNotIn(str(path),cache.records)

    def test_exclusive_claim_or_output_never_overwrites(self):
        path=self.base/'claim.json';chain.publish(path,dict(original=True))
        with self.assertRaises(FileExistsError):chain.publish(path,dict(original=False))
        self.assertEqual(chain.read(path),dict(original=True))

    def storage_fixture(self,request):
        admission=self.write(request['storage_admission'],dict(CPU_UNIT=True))
        proof=chain.FileProofCache().capture(admission)
        budget=dict(global_peak_checkpoint_slots=7,minimum_runtime_free_bytes=chain.RESERVE,
            all_three_new_writable_bytes=6126028659,required_free_bytes=6126028659+chain.RESERVE)
        document=dict(arm_roots=request['arm_roots'],inventory={'path':request['inventory']},
            baseline={'preprocessed':request['baseline_preprocessed']},historical_basic_bank={'path':request['historical_basic_bank']},
            aggregate_budget=budget)
        binding=dict(format=chain.STORAGE_FORMAT,admission=proof,arm_roots=request['arm_roots'],
            required_free_bytes=budget['required_free_bytes'],minimum_runtime_free_bytes=chain.RESERVE,model_data_scale_preserved=True)
        self.write(request['storage_binding'],binding)
        return document,binding

    def test_missing_storage_is_explicit_wait_before_any_gpu_work(self):
        request=self.request()
        with self.assertRaises(chain.WaitingForStorage):chain.storage_ready(request,chain.FileProofCache())

    def test_aggregate_reserve_counted_once_and_double_add_rejected(self):
        request=self.request();document,binding=self.storage_fixture(request)
        fake=types.SimpleNamespace(verify_admission=lambda path:document,check_aggregate_disk=lambda *a,**k:{'CPU_UNIT':True})
        with patch.dict(sys.modules,{'hiercp_v1x.v24_matched_basic_cp_pipeline':fake}):
            value=chain.storage_ready(request,chain.FileProofCache())
            self.assertEqual(value['required_free_bytes'],17400317811)
            document['aggregate_budget']['required_free_bytes']+=chain.RESERVE
            binding['required_free_bytes']=document['aggregate_budget']['required_free_bytes'];self.write(request['storage_binding'],binding)
            with self.assertRaisesRegex(ValueError,'once only'):chain.storage_ready(request,chain.FileProofCache())

    def test_only_capacity_shortage_waits_but_corrupt_admission_fails(self):
        request=self.request();document,_=self.storage_fixture(request)
        fake=types.SimpleNamespace(verify_admission=lambda path:document,check_aggregate_disk=lambda *a,**k:(_ for _ in ()).throw(OSError(chain.CAPACITY_ERROR)))
        with patch.dict(sys.modules,{'hiercp_v1x.v24_matched_basic_cp_pipeline':fake}):
            with self.assertRaises(chain.WaitingForStorage):chain.storage_ready(request,chain.FileProofCache())
            fake.check_aggregate_disk=lambda *a,**k:(_ for _ in ()).throw(OSError('actual private arm overspend'))
            with self.assertRaisesRegex(OSError,'overspend'):chain.storage_ready(request,chain.FileProofCache())

    def test_failed_scorer_never_becomes_pending_or_complete(self):
        request=self.request();self.write(self.scoring/'failed.json',dict(error='CPU UNIT actual failure'))
        with self.assertRaisesRegex(RuntimeError,'actual failure'):chain.scoring_complete(request,chain.FileProofCache())

    def controller_fixture(self):
        controller=object.__new__(chain.Controller);controller.request=self.request();controller.root=self.group/'gpu4'
        controller.scoring=None;controller.storage=None;controller.cache=chain.FileProofCache();controller.state={}
        controller.update=lambda **fields:controller.state.update(fields)
        return controller

    def test_no_gpu_launch_before_gnn_or_cp_or_storage(self):
        controller=self.controller_fixture()
        with patch.object(chain,'preserve_predecessors'),patch.object(chain,'source_complete',return_value=None),patch.object(chain,'resources_ready') as gpu:
            self.assertFalse(controller.ready('prepare-native'));gpu.assert_not_called()
            self.assertEqual(controller.state['status'],'WAITING_FOR_FULL_GNN')
        with patch.object(chain,'preserve_predecessors'),patch.object(chain,'source_complete',return_value={'complete':True}),patch.object(chain,'scoring_complete',return_value=None),patch.object(chain,'resources_ready') as gpu:
            self.assertFalse(controller.ready('prepare-native'));gpu.assert_not_called()
            self.assertEqual(controller.state['status'],'WAITING_FOR_FULL642_SCORING')
        controller.scoring={'CPU_UNIT':'admitted'}
        with patch.object(chain,'preserve_predecessors'),patch.object(chain,'storage_ready',side_effect=chain.WaitingForStorage('real shortage')),patch.object(chain,'resources_ready') as gpu:
            with self.assertRaises(chain.WaitingForStorage):controller.ready('prepare-native')
            gpu.assert_not_called()

    def test_gpu5_wait_template_preserves_full_scoring_and_cnn(self):
        request=self.request(5);path=self.write(self.group/'gpu5/request.json',request)
        doc=waiter.build_request(path,host_container_lifetime={'CPU_UNIT':True},GPU_UUID=request['GPU_UUID'])
        self.assertEqual(doc['scoring_template']['physical_candidate_chunk'],32)
        self.assertEqual(doc['scoring_template']['source_entries'],642)
        self.assertIsNone(doc['scoring_template']['stunet_checkpoint'])
        self.assertNotIn('preserved_source_hashes',doc['scoring_template'])
        with self.assertRaises(ValueError):waiter.build_request(path,host_container_lifetime={},GPU_UUID='GPU-wrong')

    def test_gpu5_scoring_only_after_full_source_uses_fresh_request_binding(self):
        request=self.request(5);native_path=self.write(self.group/'gpu5/request.json',request)
        doc=waiter.build_request(native_path,host_container_lifetime={'CPU_UNIT':True},GPU_UUID=request['GPU_UUID'])
        for name in ('request.json','calibration.json','training/training_identity.json','training/invocations.jsonl','training/checkpoint_latest.pt','training/checkpoint_best.pt'):
            (self.source/name).write_bytes(b'CPU UNIT bytes')
        bank=self.base/'Basic';bank.mkdir();(bank/'index.json').write_bytes(b'CPU UNIT index')
        for name in ('direct_training_request.json','direct_training_registration.json'):(self.source.parent/name).write_bytes(b'CPU UNIT witness')
        guard=self.scoring/'direct_GPU45_access_guard.py';guard.write_bytes(b'CPU UNIT exact guard')
        with patch.object(scorer,'GUARD_SHA256',chain.sha(guard)),patch.object(scorer,'verify_binding') as verify:
            doc['scoring_template']['direct_guard_sha256']=scorer.GUARD_SHA256
            template,path=waiter.prepare_scoring(doc,request)
            proof=chain.read(self.scoring/'direct_GPU_access_proof.json')
            self.assertEqual(proof['request'],dict(path=str(path),sha256=chain.sha(path)))
            self.assertFalse(proof['PBS_owned_reservation']);verify.assert_called_once()
            self.assertEqual(len(template['preserved_source_hashes']),9)

    def test_same_source_cannot_duplicate_native_using_new_scoring_namespace(self):
        request=self.request();original=chain.claim_path(request)
        request['scoring_root']=str(self.base/'another_scoring_namespace')
        self.assertEqual(chain.claim_path(request),original)

    def old_controller(self):
        request=self.request();root=self.base/'old_native';root.mkdir()
        values=dict(request={'source_output':request['source_output']},status=dict(status='FAILED',child_pid=55,
            child_create_time=123.,command=['CPU_UNIT_old_child']),registration=dict(pid=44,create_time=122.,command=['CPU_UNIT_old_controller']))
        cache=chain.FileProofCache()
        request['native_predecessors']=[dict(root=str(root),**{key:cache.capture(self.write(root/(key+'.json'),value)) for key,value in values.items()})]
        import hashlib
        stem=hashlib.sha256(str(self.source).encode()).hexdigest()
        registry=self.source.parent/'gnn_native_chain_claims_20261010';registry.mkdir()
        claim=self.write(registry/(stem+'.json'),dict(chain_root=str(root),source_output=str(self.source)))
        request['source_claim_witnesses']=[cache.capture(claim)]
        return request,cache,root

    def test_prior_dead_failed_child_allowed_but_live_child_or_outputs_refused(self):
        request,cache,root=self.old_controller()
        with patch.object(chain,'process_snapshot',return_value=None):
            chain.preserve_predecessors(request,cache)
        with patch.object(chain,'process_snapshot',side_effect=lambda pid,*args:object() if pid==55 else None):
            with self.assertRaisesRegex(ValueError,'old native/scoring child'):chain.preserve_predecessors(request,cache)
        (root/'native').mkdir()
        with patch.object(chain,'process_snapshot',return_value=None):
            with self.assertRaisesRegex(ValueError,'still active'):chain.preserve_predecessors(request,cache)

    def test_prior_proof_or_source_claim_mutation_refused(self):
        request,cache,root=self.old_controller();(root/'request.json').write_text('{}',encoding='utf8')
        with patch.object(chain,'process_snapshot',return_value=None):
            with self.assertRaisesRegex(ValueError,'identity'):chain.preserve_predecessors(request,cache)

    def test_normal_child_exit_between_poll_and_rss_still_verifies_completion(self):
        import psutil
        controller=self.controller_fixture();controller.path=self.write(controller.root/'request.json',controller.request)
        controller.digest=chain.sha(controller.path);controller.storage={'sha256':'a'*64}
        controller.state.update(stages_completed=[],stage_proofs=[])
        stage=controller.request['stages'][0]
        polls=iter([None,0])
        child=types.SimpleNamespace(pid=55,poll=lambda:next(polls,0),wait=lambda:0)
        actor=types.SimpleNamespace(create_time=lambda:123.,memory_info=lambda:(_ for _ in ()).throw(psutil.NoSuchProcess(55)))
        proof=dict(action='prepare-native',CPU_UNIT='real proof route')
        with patch.object(chain.subprocess,'Popen',return_value=child),patch.object(psutil,'Process',return_value=actor),\
                patch.object(chain,'process_snapshot',return_value=actor),patch.object(chain,'preserve_predecessors'),\
                patch.object(chain,'native_stage_proof',return_value=proof) as verify:
            self.assertEqual(controller.stage(stage),proof);verify.assert_called_once_with(controller.request,'prepare-native')
            self.assertEqual(controller.state['stages_completed'],['prepare_native'])
            self.assertIsNone(controller.state['child_pid'])

    def test_native_only_direct_declaration_leaves_legacy_policy_untouched(self):
        from hiercp_v1x import v24_direct_gpu_access as direct
        request=self.request();stage=direct.stage_request_document(request)
        self.assertEqual(direct._stage_declaration(stage,Path(request['chain_root'])/'direct_GPU_stage_request.json',4,
            chain.AFFINITIES[4])['epochs'],250)
        with self.assertRaises(ValueError):direct._stage_declaration(stage,Path(request['chain_root'])/'direct_GPU_stage_request.json',5,chain.AFFINITIES[5])

    def test_standalone_entry_help_outside_repository(self):
        for name in ('watch_v24_scored_bank_to_nnunet.py','watch_v24_gnn_to_matched_scoring.py'):
            result=subprocess.run([sys.executable,'-B',str(self.repo/'tools'/name),'--help'],cwd=self.base,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--request',result.stdout)


if __name__=='__main__':unittest.main()
