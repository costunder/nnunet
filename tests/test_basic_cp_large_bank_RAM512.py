"""CPU DEBUG512 provider routing and exact scientific/publication regressions."""
import copy,os,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from hiercp_v1x import basic_cp_large_bank_RAM512 as fresh
from hiercp_v1x import basic_cp_large_bank_recovery as canonical
from tests.test_basic_cp_large_bank import source_fixture

ROOT=Path(__file__).resolve().parents[1]

def request():
 return dict(memory_budget_revision=512,aggregate_memory_cap_GiB=512,
  effective_native_RSS_limit_GiB=32,effective_shared_bank_RSS_limit_GiB=320,
  owned_tmpfs_envelope_GiB=144,post_source_owned_tmpfs_envelope_GiB=272,auxiliary_RAM_envelope_GiB=16,
  shared_resource_guard_version=3,shared_bank_RSS_limit_GiB=384,RSS_limit_bytes=96*2**30,
  shared_bank_workers=24,code_sha256={fresh.PROVIDER_FILE:'DEBUG_PIN',fresh.RESOURCE_FILE:'DEBUG_PIN'},
  bank='DEBUG_original128',bank_sha256='DEBUG_originalSHA',candidate_binding='DEBUG_originalbinding',
  candidates_per_source=2048,full_sources=642,copies_per_CP_event=1)

class RAM512ProviderRoutingTests(unittest.TestCase):
 def setUp(self):
  self.old_state=(fresh._LEASE,fresh._REQUEST_IDENTITY,fresh._ROOT,fresh._RECOVERY_REF)
  fresh._LEASE=None;fresh._REQUEST_IDENTITY=None;fresh._ROOT=None;fresh._RECOVERY_REF=None
 def tearDown(self):fresh._LEASE,fresh._REQUEST_IDENTITY,fresh._ROOT,fresh._RECOVERY_REF=self.old_state

 def test_only_operational_source_execution_field_changes_and_real_request_is_unchanged(self):
  value=request();before=copy.deepcopy(value);result=fresh._source_execution_view(value)
  self.assertEqual(value,before);self.assertEqual(result.pop('shared_bank_RSS_limit_GiB'),320)
  expected=copy.deepcopy(before);expected.pop('shared_bank_RSS_limit_GiB');self.assertEqual(result,expected)

 def test_all_explicit512_flags_and_both_SHA_pins_are_required(self):
  for key in('memory_budget_revision','aggregate_memory_cap_GiB','effective_native_RSS_limit_GiB',
    'effective_shared_bank_RSS_limit_GiB','owned_tmpfs_envelope_GiB','auxiliary_RAM_envelope_GiB'):
   value=request();value[key]=True
   with self.subTest(key=key),self.assertRaises(canonical.base.BankContractError):fresh._validate_512_request(value)
  for key in(fresh.PROVIDER_FILE,fresh.RESOURCE_FILE):
   value=request();value['code_sha256'].pop(key)
   with self.subTest(pin=key),self.assertRaises(canonical.base.BankContractError):fresh._validate_512_request(value)

 def test_legacy_declarations_and_all24_workers_remain_mandatory(self):
  for key,value in(('shared_bank_RSS_limit_GiB',320),('RSS_limit_bytes',32*2**30),('shared_bank_workers',12)):
   settings=request();settings[key]=value
   with self.subTest(key=key),self.assertRaises(canonical.base.BankContractError):fresh._source_execution_view(settings)

 def test_quick_and_force_checks_share_exact_request_and_never_select_old800_lease(self):
  value=request();lease=Mock();lease.verify.return_value={'guard':'DEBUG512force'};lease.quick_verify.return_value={'guard':'DEBUG512quick'}
  with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(ROOT/'DEBUG_request.json')}),patch.object(fresh.base,'stat_identity',return_value=('DEBUG_inode',1)),patch.object(fresh.base,'read_json',return_value=value),patch.object(fresh,'_shared_lease_factory',return_value=lease)as factory:
   self.assertEqual(fresh.verify_resources(),{'guard':'DEBUG512quick'});self.assertEqual(fresh.verify_resources(force=True),{'guard':'DEBUG512force'})
  factory.assert_called_once_with(value);lease.quick_verify.assert_called_once_with();lease.verify.assert_called_once_with(force=True)
  self.assertEqual(value['shared_bank_RSS_limit_GiB'],384)

 def test_changed_live_request_identity_fails_before_another_guard_call(self):
  lease=Mock()
  with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(ROOT/'DEBUG_request.json')}),patch.object(fresh.base,'stat_identity',side_effect=[('DEBUG_inode',1),('DEBUG_inode',2)]),patch.object(fresh.base,'read_json',return_value=request()),patch.object(fresh,'_shared_lease_factory',return_value=lease):
   fresh.verify_resources()
   with self.assertRaisesRegex(canonical.base.BankContractError,'request changed'):fresh.verify_resources()
  self.assertEqual(lease.quick_verify.call_count,1)

 def test_unapproved_claim_failure_prevents_pool_builder_or_geometry_allocation(self):
  with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(ROOT/'DEBUG_request.json')}),patch.object(fresh.base,'read_json',return_value=request()),patch.object(fresh,'_snapshot'),patch.object(fresh,'verify_resources',side_effect=RuntimeError('DEBUG source claim absent')),patch.object(fresh,'_build')as build:
   with self.assertRaisesRegex(RuntimeError,'claim absent'):fresh.build_staged_banks({},ROOT/'tmp/DEBUG512',workers=24,rss_limit_gib=320,recovery_snapshot=ROOT/'DEBUG_snapshot.json',recovery_snapshot_sha256='DEBUG_SHA')
  build.assert_not_called()

 def test_claim_verification_occurs_before_private_pool_build_and_no_worker_reduction(self):
  events=[]
  with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(ROOT/'DEBUG_request.json')}),patch.object(fresh.base,'read_json',return_value=request()),patch.object(fresh,'_snapshot'),patch.object(fresh,'verify_resources',side_effect=lambda **kw:events.append(('guard',kw))),patch.object(fresh,'_build',side_effect=lambda *args,**kw:events.append(('build',kw))):
   fresh.build_staged_banks({},ROOT/'tmp/DEBUG512',workers=24,rss_limit_gib=320,recovery_snapshot=ROOT/'DEBUG_snapshot.json',recovery_snapshot_sha256='DEBUG_SHA')
  self.assertEqual(events[0],('guard',dict(force=True)));self.assertEqual(events[1][1]['workers'],24)
  self.assertEqual(events[1][1]['rss_limit_gib'],320)

 def test_build_only_request_reader_does_not_overlay_prior_requests_or_binding_arrays(self):
  value=request();other={'DEBUG':{'original_source':'UNCHANGED'}}
  current=ROOT/'DEBUG_request.json';prior=ROOT/'DEBUG_prior.json'
  with patch.dict(os.environ,{'BASIC_CP_LEASE_REQUEST':str(current)}),patch.object(fresh.base,'read_json',side_effect=lambda path:value if Path(path)==current else other):
   self.assertEqual(fresh._build_read_json(current)['shared_bank_RSS_limit_GiB'],320)
   self.assertIs(fresh._build_read_json(prior),other)
  self.assertEqual(value['shared_bank_RSS_limit_GiB'],384)

class RAM512ScientificCloneTests(unittest.TestCase):
 def test_frozen_scientific_bytecode_and_full_three_stage_build_are_identical(self):
  for name in('_classify','_reuse_entry','_build_stage_patient','_snapshot','snapshot_interrupted_outputs'):
   old,new=getattr(canonical,name),getattr(fresh,name)
   with self.subTest(function=name):self.assertEqual(new.__code__.co_code,old.__code__.co_code);self.assertEqual(new.__code__.co_consts,old.__code__.co_consts)
  self.assertEqual(fresh.FastGeometryProvider._full_proposals.__code__,canonical.FastGeometryProvider._full_proposals.__code__)
  self.assertEqual(fresh.FastFiniteLiverCenters.sampler.__code__,canonical.FastFiniteLiverCenters.sampler.__code__)
  self.assertEqual(fresh._build.__code__.co_code,canonical._build.__code__.co_code)
  differences=[(a,b)for a,b in zip(canonical._build.__code__.co_consts,fresh._build.__code__.co_consts)if a!=b]
  self.assertEqual(differences,[(384,320)])

 def test_worker_state_is_private_and_external_bank_format_is_preserved(self):
  self.assertIs(fresh._initialize.__globals__,vars(fresh));self.assertIs(fresh._build_stage_patient.__globals__,vars(fresh))
  self.assertIsNot(fresh._SNAPSHOTS,canonical._SNAPSHOTS);self.assertIsNot(fresh.base,canonical.base)
  self.assertEqual(fresh._publish_checked_prefix.__globals__['FORMAT'],canonical.original_base.FORMAT)
  self.assertNotEqual(fresh.FORMAT,canonical.original_base.FORMAT)
  self.assertIsNot(fresh.verify_resources,canonical.verify_resources)

 def test_original128_scores_and_all512_1024_2048_position_bytes_remain_unchanged(self):
  original,prefix=source_fixture(128),source_fixture(256);new_stream=source_fixture(2048)
  proposals=[(new_stream['candidate_raw_centers'][i],new_stream['candidate_centers'][i])for i in range(256,2048)]
  def run(base):
   results={};base.collect_prefixes(original,prefix,iter(proposals),targets=(512,1024,2048),publish=lambda count,data,proof:results.update({count:data}))
   return results
  old,new=run(canonical.base),run(fresh.base)
  for count in(512,1024,2048):
   self.assertEqual(set(old[count]),set(new[count]))
   for key in old[count]:self.assertEqual(old[count][key].tobytes(),new[count][key].tobytes())
   self.assertEqual(len(new[count]['scores']),128)

 def test_atomic_full_source_publication_bytes_equal_canonical_and_existing_paths_are_never_overwritten(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as directory:
   folder=Path(directory).resolve();old=folder/'DEBUG_old.npz';new=folder/'DEBUG_new.npz'
   payload=source_fixture(2048);canonical._atomic_npz(old,payload);fresh._atomic_npz(new,payload)
   self.assertEqual(old.read_bytes(),new.read_bytes());before=new.read_bytes()
   with self.assertRaises(FileExistsError):fresh._atomic_npz(new,source_fixture(1024))
   self.assertEqual(new.read_bytes(),before)

if __name__=='__main__':unittest.main()
