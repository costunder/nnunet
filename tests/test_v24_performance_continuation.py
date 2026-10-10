"""CPU UNIT: exact native probe admission before production continuation."""
import copy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from tools import run_v24_performance_continuation as performance
from tools.run_v24_performance_continuation import compare_probes
from tests import test_v24_training_continuation as original_fixture


class NativeProbeAdmissionUnit(unittest.TestCase):
    def setUp(self):
        self.baseline=dict(variant='baseline',debug=True,production_optimizer_updates=0,
            active_U=128,all_P_included=True,physical_patient_batch=4,diagnostic_batches=1,
            original_training_identity_sha256='owned',initial_model_sha256='model',initial_RNG_sha256='rng',
            output_sha256='scores',loss=1.25,gradient_sha256='gradient',
            after_forward_model_sha256='model2',after_RNG_sha256='rng2',
            case_ids=['liver_6','liver_129','liver_123','liver_69'],physical_candidate_chunk=64,
            source_checkpoint=dict(raw_sha256='actual-snapshot'),
            debug_numerics=dict(debug_deterministic_numerics=True,production_config_modified=False,
                unsupported_operator_fallback=False,actual_forward_flags=dict(deterministic_algorithms=True,
                deterministic_warn_only=False,CUBLAS_WORKSPACE_CONFIG=':4096:8')),
            ordered_CPU_input_sha256='1'*64,
            input_tensor_proof=dict(complete=True,every_actual_value_hashed=True,both_sampled_views=True,
                ordered_records=524,content_sha256='1'*64,observed_native_chunks=9,expected_native_chunks=9,
                observed_upper_graphs=4,expected_upper_graphs=4,
                content=dict(local_chunks=[{} for _ in range(9)],upper_graphs=[{} for _ in range(4)])),
            gradient=dict(missing=[],finite=True),seconds=65.)
        self.optimized=copy.deepcopy(self.baseline);self.optimized.update(variant='optimized',seconds=30.)

    def test_native_proof_admits_exact_results(self):
        row=compare_probes(self.baseline,self.optimized,'actual-snapshot')
        self.assertEqual(row['status'],'ACTUAL_FULL128_OUTPUT_LOSS_GRADIENT_MODEL_RNG_EXACT_PASS')
        self.assertTrue(row['debug_single_batch_timing'])

    def test_each_numerical_change_is_rejected(self):
        for key in ('loss','output_sha256','gradient_sha256','initial_RNG_sha256','after_RNG_sha256',
                    'initial_model_sha256','after_forward_model_sha256','case_ids','physical_candidate_chunk'):
            changed=copy.deepcopy(self.optimized);changed[key]='different'
            with self.subTest(key=key),self.assertRaises(ValueError):
                compare_probes(self.baseline,changed,'actual-snapshot')

    def test_unmeasured_inputs_or_relaxed_debug_kernels_never_admit(self):
        for changed_field,value in (('complete',False),('ordered_records',523),
                                    ('every_actual_value_hashed',False),('both_sampled_views',False),
                                    ('observed_native_chunks',8),('expected_native_chunks',8),
                                    ('observed_upper_graphs',3),('expected_upper_graphs',3),
                                    ('content',dict(local_chunks=[{} for _ in range(8)],upper_graphs=[{} for _ in range(4)]))):
            changed=copy.deepcopy(self.optimized);changed['input_tensor_proof'][changed_field]=value
            with self.subTest(field=changed_field),self.assertRaisesRegex(ValueError,'CPU tensor proof'):
                compare_probes(self.baseline,changed,'actual-snapshot')
        for key,value in (('deterministic_algorithms',False),('deterministic_warn_only',True),
                          ('CUBLAS_WORKSPACE_CONFIG',None)):
            baseline=copy.deepcopy(self.baseline);optimized=copy.deepcopy(self.optimized)
            baseline['debug_numerics']['actual_forward_flags'][key]=value
            optimized['debug_numerics']['actual_forward_flags'][key]=value
            with self.subTest(field=key),self.assertRaisesRegex(ValueError,'Strict DEBUG'):
                compare_probes(baseline,optimized,'actual-snapshot')

    def test_wrong_snapshot_scope_or_missing_gradient_is_rejected(self):
        for patch in ({'active_U':23},{'all_P_included':False},{'physical_patient_batch':2},
                      {'production_optimizer_updates':1},{'source_checkpoint':{'raw_sha256':'wrong'}},
                      {'gradient':{'missing':['parameter'],'finite':True}},
                      {'gradient':{'missing':[],'finite':False}}):
            changed=copy.deepcopy(self.optimized);changed.update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                compare_probes(self.baseline,changed,'actual-snapshot')


class InterruptedSourceAdmissionDebug(unittest.TestCase):
    """CPU metadata only: actual checkpoint bytes, no model/CT/GPU launch."""
    def setUp(self):
        self.fixture=original_fixture.ExactContinuationDebug();self.fixture.setUp()
        self.args=self.fixture.args;self.old=self.fixture.old;self.code=original_fixture.ROOT
        self.job=self.fixture.root/'original_job';self.job.mkdir()
        self.live=set();self.host='ece-a6gpu6';self.uid=41833
        self.context=dict(uid=self.uid,hostname=self.host,pid_exists=lambda pid:pid in self.live,
            # Explicit CPU UNIT ownership metadata; production uses actual st_uid.
            owner_uid=lambda path:41833)
        self.status=dict(status='RUNNING',stage='train',worker_pid=121,worker_create_time=1234.25,
            child_pid=122,child_create_time=1235.75,request=dict(GPU=5,
                production_output=str(self.old.resolve()),code=str(self.code.resolve())))
        original_fixture._json(self.job/'status.json',self.status)
        self.proof_path=self.fixture.root/'interruption.json'
        self.reset_proof()
        self.patches=(patch.object(performance,'_interruption_context',return_value=self.context),
            patch.object(original_fixture.runtime,'_runtime_proof',return_value=original_fixture.RUNTIME_UNIT),
            patch.object(performance,'runtime_identity',return_value=dict(debug='CPU_UNIT_metadata_only',no_GPU=True)))
        for item in self.patches:item.start()

    def tearDown(self):
        for item in reversed(self.patches):item.stop()

    def reset_proof(self):
        witness=performance._checkpoint_witness(self.old/'training/checkpoint_latest.pt')
        now=time.time_ns()
        self.proof=dict(format=performance.INTERRUPTION_FORMAT,source_output=str(self.old.resolve()),
            source_code=str(self.code.resolve()),GPU=5,uid=41833,job=str(self.job.resolve()),
            worker=dict(pid=121,create_time=1234.25,absent=True),
            child=dict(pid=122,create_time=1235.75,absent=True),checkpoint=witness,
            snapshots=[dict(time_ns=now-3_000_000_000,checkpoint=copy.deepcopy(witness),
                worker_absent=True,child_absent=True),
                dict(time_ns=now-2_000_000_000,checkpoint=copy.deepcopy(witness),worker_absent=True,child_absent=True)],
            server_host='ece-a6gpu6',original_job_status='RUNNING',original_stage='train',detected_ns=now)
        self.write_proof()

    def write_proof(self):original_fixture._json(self.proof_path,self.proof)

    def latest(self):
        return original_fixture.runtime.inspect_continuation(self.old,self.code)['latest']

    def validate(self):
        return performance.validate_source_interruption(self.proof_path,self.old,self.code,self.latest(),gpu=5)

    def prepare(self):
        return performance.prepare(self.args,self.old,self.code,self.proof_path)

    def test_RUNNING_copy_is_exact_without_PAUSED_forgery_and_partial_BEST_preserved(self):
        saved=copy.deepcopy(self.fixture.latest)
        saved['state'].update(evaluation_position=8,evaluation_rows=[{'case_id':case} for case in self.fixture.val[:8]])
        original_fixture._save(self.old/'training/checkpoint_latest.pt',saved);self.reset_proof()
        old=original_fixture.runtime.inspect_continuation(self.old,self.code)
        old_latest=(self.old/'training/checkpoint_latest.pt').read_bytes()
        old_best=(self.old/'training/checkpoint_best.pt').read_bytes()
        old_job=(self.job/'status.json').read_bytes()
        document=self.prepare()
        self.assertEqual(document['source_admission']['status'],'INTERRUPTED_RUNNING')
        self.assertEqual(document['source']['latest']['status'],'RUNNING')
        _,admitted=performance.admit(self.args,self.old,self.code)
        self.assertEqual(admitted,document)
        fresh=original_fixture.runtime.inspect_continuation(self.args.output,self.code)
        self.assertEqual(fresh['latest']['numerical_state_sha256'],old['latest']['numerical_state_sha256'])
        self.assertEqual(fresh['latest']['state_sha256'],old['latest']['state_sha256'])
        self.assertEqual(fresh['latest']['evaluation_position'],8)
        self.assertEqual(fresh['latest']['status'],'RUNNING')
        self.assertEqual(old_latest,(self.args.output/'training/checkpoint_latest.pt').read_bytes())
        self.assertEqual(old_best,(self.args.output/'training/checkpoint_best.pt').read_bytes())
        self.assertEqual(old_latest,(self.old/'training/checkpoint_latest.pt').read_bytes())
        self.assertEqual(old_job,(self.job/'status.json').read_bytes())
        self.assertEqual(admitted['source_admission']['interruption']['raw_sha256'],performance._sha(self.proof_path))

    def test_RUNNING_without_proof_rejected_before_new_result_creation(self):
        with self.assertRaisesRegex(ValueError,'explicit verified'):
            performance.prepare(self.args,self.old,self.code)
        self.assertFalse(self.args.output.exists())

    def test_normal_PAUSED_source_still_works_without_interruption(self):
        saved=copy.deepcopy(self.fixture.latest);saved['state']['status']='PAUSED'
        original_fixture._save(self.old/'training/checkpoint_latest.pt',saved)
        document=performance.prepare(self.args,self.old,self.code)
        self.assertEqual(document['source_admission'],dict(status='PAUSED',original_checkpoint_status='PAUSED',interruption=None))
        _,admitted=performance.admit(self.args,self.old,self.code)
        self.assertEqual(admitted,document)
        with self.assertRaisesRegex(ValueError,'already PAUSED'):
            performance._source_admission(self.latest(),self.old,self.code,5,self.proof_path)

    def test_old_PID_alive_or_reused_is_rejected_on_prepare_and_later_admit(self):
        self.live.add(121)
        with self.assertRaisesRegex(ValueError,'PID still exists'):
            self.prepare()
        self.assertFalse(self.args.output.exists());self.live.clear();self.prepare();self.live.add(122)
        with self.assertRaisesRegex(ValueError,'PID still exists'):
            performance.admit(self.args,self.old,self.code)

    def test_two_stable_snapshots_and_closed_known_job_identity_required(self):
        good=copy.deepcopy(self.proof)
        mutations=(lambda p:p.update(GPU=6),lambda p:p.update(extra='unproved'),
            lambda p:p['worker'].update(create_time=1234.5),
            lambda p:p['child'].update(absent=False),
            lambda p:p['snapshots'][1].update(time_ns=p['snapshots'][0]['time_ns']+999_999_999),
            lambda p:p['snapshots'][1]['checkpoint'].update(sha256='0'*64),
            lambda p:p.update(snapshots=p['snapshots'][:1]),
            lambda p:p.update(original_job_status='FAILED'),
            lambda p:p['checkpoint'].update(mtime_ns=p['checkpoint']['mtime_ns']+1))
        for mutate in mutations:
            self.proof=copy.deepcopy(good);mutate(self.proof);self.write_proof()
            with self.subTest(proof=self.proof),self.assertRaises(ValueError):self.validate()
        self.proof=good;self.write_proof()
        wrong=copy.deepcopy(self.status);wrong['request']['production_output']=str(self.fixture.root/'other')
        original_fixture._json(self.job/'status.json',wrong)
        with self.assertRaisesRegex(ValueError,'job/launch identities'):self.validate()

    def test_wrong_actual_UID_host_or_file_ownership_rejected(self):
        for context in (dict(self.context,uid=0),dict(self.context,hostname='different-server'),
                        dict(self.context,owner_uid=lambda path:0)):
            with patch.object(performance,'_interruption_context',return_value=context),self.assertRaises(ValueError):
                self.validate()

    def test_proof_raw_bytes_and_explicit_attachment_path_bound_after_prepare(self):
        self.prepare();other=self.fixture.root/'otherproof.json'
        other.write_bytes(self.proof_path.read_bytes())
        with self.assertRaisesRegex(ValueError,'prepared attachment'):
            performance.admit(self.args,self.old,self.code,other)
        # Same valid fields in different raw bytes still cannot replace receipt.
        self.proof_path.write_text(json.dumps(self.proof,indent=2),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'proof bytes changed'):
            performance.admit(self.args,self.old,self.code)

    def test_changed_checkpoint_or_status_and_missing_BEST_never_admitted(self):
        expected=self.latest()
        changed=copy.deepcopy(self.fixture.latest);changed['model']['UNIT.weight']+=1
        original_fixture._save(self.old/'training/checkpoint_latest.pt',changed)
        with self.assertRaisesRegex(ValueError,'checkpoint SHA'):
            performance.validate_source_interruption(self.proof_path,self.old,self.code,expected,gpu=5)
        original_fixture._save(self.old/'training/checkpoint_latest.pt',self.fixture.latest);self.reset_proof()
        (self.old/'training/checkpoint_best.pt').rename(self.old/'training/actual_best_preserved.pt')
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.args.output.exists())


if __name__=='__main__':unittest.main()
