"""DEBUG CPU host-role fixtures; no actual reservation or model run is claimed."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_ablation_resources as common
from hiercp_v1x import basic_cp_seven_resources as seven
from tools import persist_basic_cp_seven as durable
from tests.test_basic_cp_seven_resources import Actor, SevenLeaseTests
from tests.test_persist_basic_cp_host_lifetime import DEBUG_documents


GPU5 = 'GPU-e41d3421-8819-0a49-4a97-6c4518a54f5f'


def DEBUG_proof(value=None):
    result = {} if value is None else copy.deepcopy(value)
    mapping = {str(i): 'GPU-0000000' + str(i) + '-0000-0000-0000-000000000000' for i in range(7)}
    mapping['5'] = GPU5
    environment = dict(CUDA_VISIBLE_DEVICES=GPU5, NVIDIA_VISIBLE_DEVICES='all', PBS_GPUFILE=None,
                       PBS_JOBID=result.get('job_id','129864.ECE-util1'))
    identity = dict(pid=900, create_time=9000.0, UID=41833, command=['DEBUG_actual_owned_holder'])
    result.update(gpu=5, GPU_UUID=GPU5, GPU_visibility_overridden=False, holder_identity=identity,
                  actual_driver_mapping=mapping, actual_scheduler_visibility={key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PBS_GPUFILE')})
    raw = '\n'.join(index + ', ' + uuid for index,uuid in mapping.items())
    return result, environment, raw


class GenericAllocationParserTests(unittest.TestCase):
    def test_actual_GPU5_host_role_does_not_claim_other_experiment_GPU_ownership(self):
        value, environment, raw = DEBUG_proof(dict(job_id='129864.ECE-util1'))
        with patch.object(common,'_query',return_value=raw), patch.object(common,'_exact',return_value=SimpleNamespace(environ=lambda:environment)):
            observation=common.validate_host_lifetime_GPU(value)
        self.assertEqual(observation['allocated_GPU'],5)
        self.assertEqual(observation['allocated_GPU_UUID'],GPU5)
        self.assertFalse(observation['experimental_GPU_ownership_claimed'])

    def test_original_UUID_absent_GPU6_contract_remains_readable(self):
        with patch.object(common,'_query') as querying:
            observation=common.validate_host_lifetime_GPU(dict(gpu=6))
        self.assertTrue(observation['legacy_original_GPU6_witness']);querying.assert_not_called()

    def test_GPU5_never_falls_back_to_an_unproved_legacy_GPU6_document(self):
        for value in ({'gpu':5},{'gpu':True},{'gpu':7},{'gpu':-1}):
            with self.subTest(value=value),self.assertRaises(ValueError): common.validate_host_lifetime_GPU(value)

    def test_current_driver_mismatch_is_fatal(self):
        value, environment, raw = DEBUG_proof(dict(job_id='129864.ECE-util1'))
        bad=raw.replace(GPU5,'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff')
        with patch.object(common,'_query',return_value=bad),self.assertRaises(RuntimeError): common.validate_host_lifetime_GPU(value)

    def test_real_holder_environment_must_match_the_sealed_scheduler_witness(self):
        value, environment, raw = DEBUG_proof(dict(job_id='129864.ECE-util1'))
        for change in ('job','UUID','multiGPU'):
            actual=copy.deepcopy(environment)
            if change=='job':actual['PBS_JOBID']='another_job.ECE-util1'
            elif change=='UUID':actual['CUDA_VISIBLE_DEVICES']=value['actual_driver_mapping']['6']
            else:actual['CUDA_VISIBLE_DEVICES']=GPU5+','+value['actual_driver_mapping']['6']
            with self.subTest(change=change),patch.object(common,'_query',return_value=raw),patch.object(common,'_exact',return_value=SimpleNamespace(environ=lambda:actual)),self.assertRaises(RuntimeError): common.validate_host_lifetime_GPU(value)

    def test_fake_single_GPU_visibility_and_override_are_rejected(self):
        value, environment, raw = DEBUG_proof(dict(job_id='129864.ECE-util1'))
        value['GPU_visibility_overridden']=True
        with self.assertRaises(ValueError):common.validate_host_lifetime_GPU(value)
        value['GPU_visibility_overridden']=False
        environment['CUDA_VISIBLE_DEVICES']='all';value['actual_scheduler_visibility']['CUDA_VISIBLE_DEVICES']='all'
        with patch.object(common,'_query',return_value=raw),patch.object(common,'_exact',return_value=SimpleNamespace(environ=lambda:environment)),self.assertRaises(RuntimeError):common.validate_host_lifetime_GPU(value)

    def test_persistence_GPU5_keeps_actual_running_PBS_deadline_guard(self):
        ref, old, actual=DEBUG_documents('129864.ECE-util1')
        value, environment, raw=DEBUG_proof(old)
        with patch.object(common,'_query',return_value=raw),patch.object(common,'_exact',return_value=SimpleNamespace(environ=lambda:environment)),patch.object(durable.time,'time',return_value=value['expires_at_unix']-1000):
            observation=durable.validate_host_lifetime(ref,value,json.dumps(actual))
            self.assertEqual(observation['job_id'],'129864.ECE-util1')
            actual['Jobs']['129864.ECE-util1']['job_state']='Q'
            with self.assertRaises(RuntimeError):durable.validate_host_lifetime(ref,value,json.dumps(actual))


class GenericHostNativeLeaseTests(SevenLeaseTests):
    # Reuse the real private-clone fixture and exact process adapter, not a
    # mocked final training result. Existing inherited resource tests still run.
    def test_GPU5_allocation_accepts_experimental_GPU0_with_unchanged_full_limits(self):
        original=json.loads(self.lifetime.read_bytes())
        value, environment, raw=DEBUG_proof(original)
        holder=Actor(900,['DEBUG_actual_owned_holder'],birth=9000.0)
        holder.environ=lambda:environment
        self.actors[900]=holder
        self.lifetime.write_text(json.dumps(value))
        self.value['host_lifetime']['sha256']=seven.sha(self.lifetime)
        original_query=self.query
        def query(argv,**kwargs):
            if '--query-gpu=index,uuid' in argv:return raw
            return original_query(argv,**kwargs)
        with patch.object(seven,'_query',side_effect=query):
            lease=seven.BasicLease(self.value);result=lease.verify(force=True,idle=True)
        self.assertEqual(lease.host_allocation_GPU_observation['allocated_GPU'],5)
        self.assertEqual(self.value['gpu'],0)
        self.assertEqual(result['own_GPU'],0)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],96*2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],704*2**30)
        self.assertFalse(result['PBS_GPU_owned_reservation'])


if __name__=='__main__':unittest.main()
