"""DEBUG-only safety and measured-calibration admission tests."""
import copy
import json
from pathlib import Path
import tempfile
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


if __name__=='__main__':unittest.main()
