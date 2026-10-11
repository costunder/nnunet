"""DEBUG-only safety and measured-calibration admission tests."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
from tools.run_v24_matched_basic_cp import parse


class MatchedNativeAdmissionDEBUG(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='matched_native_DEBUG_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native = dict(root=str(self.root),bank_sha256='a'*64)
        self.reports = []
        for batch in (2,4):
            report = dict(physical_batch=batch,bank_sha256='a'*64,accepted=True,
                clone_optimizer_updates=4,cp_events=2,measurements=[{}, {}, {}],
                original_source_rng_and_paste_unchanged=True)
            path=self.root/('trial_'+str(batch)+'.json')
            pipeline.publish(path,report)
            self.reports.append(dict(path=str(path),sha256=pipeline.sha(path),report=report))
        self.document=dict(format='v24_matched_Basic642_clone_calibration_v1',debug=True,
            bank_sha256='a'*64,production_updates=0,production_epochs=250,
            production_physical_batch=2,cp_probability=.5,trials=self.reports)

    def admission(self,document):
        path=self.root/'calibration.json'
        path.write_text(json.dumps(document),encoding='utf8')
        with patch.object(pipeline,'_guard_native',return_value=(self.native,{},{})):
            return pipeline.validate_calibration(self.root/'native_DEBUG.json')

    def test_both_complete_real_trial_receipts_are_admitted(self):
        self.assertEqual(self.admission(self.document),self.root/'calibration.json')

    def test_missing_physical_batch_measurement_is_rejected(self):
        document=copy.deepcopy(self.document);document['trials']=document['trials'][:1]
        with self.assertRaises(ValueError):self.admission(document)

    def test_no_actual_cp_event_is_rejected(self):
        document=copy.deepcopy(self.document)
        document['trials'][0]['report']['cp_events']=0
        path=Path(document['trials'][0]['path']);path.write_text(json.dumps(document['trials'][0]['report']))
        document['trials'][0]['sha256']=pipeline.sha(path)
        with self.assertRaises(ValueError):self.admission(document)

    def test_trial_receipt_cannot_be_relabelled_without_file_evidence(self):
        document=copy.deepcopy(self.document);document['trials'][0]['report']['clone_optimizer_updates']=3
        with self.assertRaises(ValueError):self.admission(document)

    def test_scale_or_production_updates_cannot_change(self):
        for key,value in (('production_epochs',2),('production_physical_batch',1),('production_updates',1),('cp_probability',1.)):
            with self.subTest(key=key):
                document=copy.deepcopy(self.document);document[key]=value
                with self.assertRaises(ValueError):self.admission(document)

    def test_only_larger_batch_actual_cuda_oom_can_be_rejected(self):
        for batch in (2,4):
            document=copy.deepcopy(self.document)
            for initial in document['trials']:
                Path(initial['path']).write_text(json.dumps(initial['report']))
                initial['sha256']=pipeline.sha(initial['path'])
            report=dict(physical_batch=batch,bank_sha256='a'*64,accepted=False,
                        rejection='actual_torch_CUDA_OutOfMemoryError')
            row=document['trials'][0 if batch==2 else 1]
            Path(row['path']).write_text(json.dumps(report));row.update(report=report,sha256=pipeline.sha(row['path']))
            if batch==2:
                with self.assertRaises(ValueError):self.admission(document)
            else:self.admission(document)

    def test_public_stages_cannot_change_batch_or_resume(self):
        for tail in (['--physical-batch','4'],['--resume']):
            with self.subTest(tail=tail),self.assertRaises(SystemExit):
                parse(['train','--native','native.json','--gpu','4','--storage-admission','admission.json',*tail])

    def test_matched_scoring_requires_original_bank_and_matching_official_weights(self):
        args=['prepare-current-bank','--pin','p.json','--inventory','i.json','--baseline-preprocessed','pre',
              '--input-cache','cache','--output','out','--gpu','6','--storage-admission','admission.json']
        with self.assertRaises(SystemExit):parse(args)
        with self.assertRaises(SystemExit):parse([*args,'--historical-basic-bank','bank'])
        parsed=parse([*args,'--historical-basic-bank','bank','--stunet-checkpoint','small_ep4k.model'])
        self.assertEqual(parsed.gpu,6)

    def test_another_completed_source_on_same_gpu_cannot_replace_chain_pin(self):
        action='pin-current-gnn'
        document=dict(format=pipeline.STAGE_FORMAT,status='COMPLETE',action=action,
            storage_profile=pipeline.PROFILE,storage_admission_sha256='b'*64,
            cached_inputs_written=False,model_data_scale_preserved=True,
            runtime_sources_sha256={name:pipeline.sha(pipeline.ROOT/name) for name in pipeline.FILES})
        pipeline.publish(self.root/'matched_storage_pin_current_gnn.json',document)
        pipeline.publish(self.root/'pin.json',dict(debug_fixture=True))
        request=dict(chain_root=str(self.root),storage_admission_sha256='b'*64,GPU_arm=6,
                     source_output='/DEBUG/original_GNN',source_code='/DEBUG/original_source')
        pin=dict(physical_GPU=6,source_output=request['source_output'],source_code=request['source_code'])
        from hiercp_v1x import v24_nnunet_cp as original
        with patch.object(original,'validate_current_pin',return_value=pin):
            self.assertEqual(pipeline.stage_proof(request,action)['physical_GPU'],6)
        for key in ('source_output','source_code'):
            changed=dict(pin,**{key:'/DEBUG/another_completed_source'})
            with patch.object(original,'validate_current_pin',return_value=changed),self.assertRaises(ValueError):
                pipeline.stage_proof(request,action)


class Depth1FullCoverageAdmissionCPUUnit(unittest.TestCase):
    """Explicit synthetic proof fixtures; never clinical output or CUDA claims."""
    def fixture(self):
        from hiercp_v1x.v24_context_depth1 import identity
        profile = dict(identity(), snapshot=str(pipeline.ROOT / 'work' / 'UNIT_snapshot'), scope_contract='a'*64)
        entries = {}
        for index in range(642):
            case = 'UNIT_case_' + str(index % 81)
            component = index + 1
            proof = dict(format=profile['format'], profile_sha256=profile['contract_sha256'],
                semantic_context_voxels=1007, semantic_liver_surface_voxels=4744, typed_overlap_voxels=1007,
                canonical_context_nodes=168, canonical_liver_surface_nodes=173,
                minimum_liver_depth_exclusive_mm=1., context_inner_radius_mm=2., context_outer_radius_mm=10.,
                roi_shape=[37,39,27], spacing_mm=[1.,1.,1.],
                full_fields_sha256={key:'b'*64 for key in ('footprint','organ','outside','depth')},
                canonical_coordinate_sha256={key:'c'*64 for key in ('surface','interior','context','liver_surface')},
                dense_CT_patch_sha256='d'*64)
            proof['geometry_sha256']=pipeline._canonical(proof)
            proof.update(canonical_nodes_sha256='e'*64,
                source_identity=dict(case_id=case,component_id=component,full_source_mask_sha256='f'*64))
            entries[str(index)] = dict(case_id=case,source_component=component,model_sha256='9'*64,
                all_original_P_in_context=True,all_original128_CP_scored=True,
                input_binding=dict(inference_profile=profile,source_context_geometry=proof,
                    canonical_source_counts=dict(source_context=168,source_liver_surface=173),
                    donor_case_id=case,donor_component=component,original_candidate_order_and_mapping_verified=True))
        overlay=dict(inference_profile=profile,inference_profile_sha256=profile['contract_sha256'],
            training_context_depth_exclusive_mm=4.,pin={'model_contract':{'scope_contract':'a'*64}},
            scoring_sources_sha256={'hiercp_v1x/v24_context_depth1.py':profile['module_sha256']},
            entries=entries,complete_sources=642,complete_positions=82176,debug=False,model_sha256='9'*64)
        return {'overlay':overlay}

    def test_whole642_real_context_proof_profile_is_required(self):
        binding=self.fixture()
        self.assertEqual(pipeline.validate_scoring_profile(binding,expected_depth1=True),binding['overlay']['inference_profile'])
        binding['overlay']['entries'].pop('641')
        with self.assertRaisesRegex(ValueError,'every642'):pipeline.validate_scoring_profile(binding)

    def test_empty_source_mixed_profile_omitted_query_or_foreign_source_rejected(self):
        for mutation in ('empty','mixed','omitted','foreign','changed_field','changed_CT','wrong_model'):
            binding=self.fixture();row=binding['overlay']['entries']['17'];inputs=row['input_binding'];proof=inputs['source_context_geometry']
            if mutation=='empty':proof['canonical_context_nodes']=0;inputs['canonical_source_counts']['source_context']=0
            elif mutation=='mixed':inputs['inference_profile']=None
            elif mutation=='omitted':row['all_original128_CP_scored']=False
            elif mutation=='foreign':proof['source_identity']['case_id']='UNIT_another_patient'
            elif mutation=='changed_field':proof['full_fields_sha256']['depth']='0'*64
            elif mutation=='changed_CT':proof['dense_CT_patch_sha256']='0'*64
            else:row['model_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):pipeline.validate_scoring_profile(binding)

    def test_original_training_depth_and_new_implementation_cannot_be_relabelled(self):
        for mutation in ('training','module','scope','profile_SHA','debug'):
            binding=self.fixture();overlay=binding['overlay']
            if mutation=='training':overlay['training_context_depth_exclusive_mm']=1.
            elif mutation=='module':overlay['scoring_sources_sha256']['hiercp_v1x/v24_context_depth1.py']='0'*64
            elif mutation=='scope':overlay['pin']['model_contract']['scope_contract']='0'*64
            elif mutation=='profile_SHA':overlay['inference_profile_sha256']='0'*64
            else:overlay['debug']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):pipeline.validate_scoring_profile(binding)

    def test_old_default_profile_remains_accepted_and_cannot_receive_depth1_rows(self):
        binding={'overlay':{'entries':{'UNIT_entry':{'scores':[0.]*128}}}}
        self.assertIsNone(pipeline.validate_scoring_profile(binding,expected_depth1=False))
        with self.assertRaises(ValueError):pipeline.validate_scoring_profile(binding,expected_depth1=True)
        binding['overlay']['entries']['UNIT_entry']['input_binding']={'inference_profile':{'mixed':True}}
        with self.assertRaises(ValueError):pipeline.validate_scoring_profile(binding)

    def test_explicit_CLI_profile_only_selects_full_scoring(self):
        args=['prepare-current-bank','--pin','p.json','--inventory','i.json','--baseline-preprocessed','pre',
            '--input-cache','cache','--output','out','--gpu','5','--storage-admission','admission.json',
            '--historical-basic-bank','bank']
        self.assertFalse(parse(args).context_depth1)
        self.assertTrue(parse([*args,'--context-depth1']).context_depth1)
        with self.assertRaises(SystemExit):
            parse(['train','--native','native.json','--gpu','5','--storage-admission','a.json','--context-depth1'])

    def test_native_training_retains_profile_and_exact_implementation_manifest(self):
        overlay=self.fixture()['overlay'];profile=overlay['inference_profile']
        native=dict(inference_profile=copy.deepcopy(profile),
            inference_profile_sha256=profile['contract_sha256'],GNN_training_context_depth_exclusive_mm=4.,
            source_files_sha256={name:pipeline.sha(pipeline.ROOT/name) for name in pipeline.FILES})
        pipeline.validate_native_scoring_profile(native,overlay)
        for mutation in ('missing_adapter','changed_adapter','training','profile','SHA'):
            changed=copy.deepcopy(native)
            if mutation=='missing_adapter':changed['source_files_sha256'].pop('hiercp_v1x/v24_context_depth1.py')
            elif mutation=='changed_adapter':changed['source_files_sha256']['hiercp_v1x/v24_context_depth1.py']='0'*64
            elif mutation=='training':changed['GNN_training_context_depth_exclusive_mm']=1.
            elif mutation=='profile':changed['inference_profile']=None
            else:changed['inference_profile_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                pipeline.validate_native_scoring_profile(changed,overlay)
        pipeline.validate_native_scoring_profile({}, {'entries':{}})


class Depth1OwnedAllocationCPUUnit(unittest.TestCase):
    """Controlled lease witnesses only; no scheduler command or CUDA execution."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='depth1_lease_UNIT_',dir=pipeline.ROOT/'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.path=self.root/'request.json'
        self.request=dict(context_depth1=True,GPU_arm=6,GPU_UUID='GPU-UNIT_owned',
            CPU_affinity=[32,33,34,35],chain_root=str(self.root),scheduler={'UNIT_only':True})
        pipeline.publish(self.path,self.request)
        self.wall=1000.;self.monotonic=500.;self.expiry=41000.
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ,
            V24_CONTEXT_DEPTH1_CHAIN_REQUEST=str(self.path),
            V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256=pipeline.sha(self.path),CUDA_VISIBLE_DEVICES='6'))
        self.stack.enter_context(patch.object(pipeline.time,'time',side_effect=lambda:self.wall))
        self.stack.enter_context(patch.object(pipeline.time,'monotonic',side_effect=lambda:self.monotonic))
        self.cpu=self.stack.enter_context(patch('psutil.Process',return_value=SimpleNamespace(
            cpu_affinity=lambda:[32,33,34,35])))
        self.scheduler=self.stack.enter_context(patch('tools.watch_v24_gnn_to_nnunet.scheduler_proof',
            side_effect=lambda request,stage_index:dict(expires_at_unix=self.expiry,
                ownership_and_allocation_checked=True)))

    def test_rechecks_live_PBS_thirty_seconds_and_no_DEBUG_lifetime_cap(self):
        guard=pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root)
        self.assertEqual(self.scheduler.call_count,2)
        guard();self.assertEqual(self.scheduler.call_count,2)
        self.wall+=1200;self.monotonic+=1200
        guard();self.assertEqual(self.scheduler.call_count,3)
        self.assertTrue(all(call.kwargs['stage_index']==1 for call in self.scheduler.call_args_list))

    def test_changed_request_CPU_or_CUDA_assignment_refuses_new_work(self):
        for mutation in ('request','CPU','CUDA'):
            with self.subTest(mutation=mutation):
                self.path.write_text(json.dumps(self.request),encoding='utf8')
                os.environ['V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256']=pipeline.sha(self.path)
                os.environ['CUDA_VISIBLE_DEVICES']='6'
                self.cpu.return_value.cpu_affinity=lambda:[32,33,34,35]
                guard=pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root)
                if mutation=='request':
                    self.path.write_text(json.dumps(dict(self.request,context_depth1=False)),encoding='utf8')
                    self.monotonic+=31;self.wall+=31
                elif mutation=='CPU':self.cpu.return_value.cpu_affinity=lambda:[0,1,2,3]
                else:os.environ['CUDA_VISIBLE_DEVICES']='5'
                with self.assertRaises((ValueError,RuntimeError)):guard()

    def test_monotonic_expiry_reserve_and_missing_seal_refuse_new_work(self):
        guard=pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root)
        # A backward wall-clock step must not extend the sealed GPU lease.
        self.wall=900.;self.monotonic+=40000-179
        with self.assertRaisesRegex(RuntimeError,'180-second'):guard()
        self.monotonic=500.;os.environ['V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256']='0'*64
        with self.assertRaises(ValueError):pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root)

    def test_missing_direct_assignment_proof_cannot_replace_owned_PBS(self):
        self.request['scheduler']=None
        self.path.write_text(json.dumps(self.request),encoding='utf8')
        os.environ['V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256']=pipeline.sha(self.path)
        with self.assertRaisesRegex(ValueError,'explicit proven owned allocation'):
            pipeline.scoring_deadline_guard(gpu=6,chain_root=self.root)


if __name__=='__main__':unittest.main()
