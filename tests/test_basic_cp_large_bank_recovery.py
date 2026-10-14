"""CPU DEBUG publication fixtures; no clinical subset or training results."""
import copy
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hiercp_v1x import basic_cp_large_bank as base
from hiercp_v1x import basic_cp_large_bank_fast as strict
from hiercp_v1x import basic_cp_large_bank_recovery as recovery
from tests.test_basic_cp_large_bank import source_fixture

ROOT=Path(__file__).resolve().parents[1]

class InterruptedPublicationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.root=Path(self.temp.name).resolve()
  self.old=self.root/'old';self.fresh=self.root/'fresh';self.fresh.mkdir()
  self.relative='entries/DEBUG_case__component_1.npz';self.original_path=self.root/'original.npz';self.admitted_path=self.root/'admitted256.npz'
  self.original=source_fixture(128);self.admitted=source_fixture(256)
  recovery._atomic_npz(self.original_path,self.original);recovery._atomic_npz(self.admitted_path,self.admitted)
  self.create(512)
 def tearDown(self):self.temp.cleanup()
 def create(self,count,payload=None):
  data=source_fixture(count)if payload is None else payload
  archive=self.old/str(count)/self.relative;archive.parent.mkdir(parents=True,exist_ok=True);recovery._atomic_npz(archive,data)
  prefix=self.admitted_path if count==512 else self.old/str(count//2)/self.relative
  proof=base.historical.verify_preservation(self.original,data,original_count=128,candidate_count=count)
  proof.update(admitted_prefix_count=256,all_admitted_prefix_centers_byte_preserved=True)
  row=dict(entry=self.relative,case_id='DEBUG_case',new_entry_sha256=base.file_sha(archive),original_entry_sha256=base.file_sha(self.original_path),proof=proof,admitted_prefix=dict(path=str(self.admitted_path),sha256=base.file_sha(self.admitted_path)),generation_prefix=dict(path=str(prefix),sha256=base.file_sha(prefix),count=len(base.read_entry(prefix)['candidate_centers'])),full_graph_geometry_cache=True)
  receipt=self.old/str(count)/'entry_receipts'/Path(self.relative).with_suffix('.json');recovery._atomic_json(receipt,row)
  return archive,receipt
 def classify(self,count):return recovery._classify(self.old,count,self.relative,'DEBUG_case',self.original_path,self.admitted_path)
 def reuse(self,row,count):
  snapshot={'records':{str(count):{self.relative:row}}};partial=dict(root=str(self.old),recovery_snapshot='DEBUG_SHA_PINNED_SNAPSHOT',recovery_snapshot_sha256='DEBUG_ONLY')
  with patch.object(recovery,'_snapshot',return_value=snapshot):
   return recovery._reuse_entry(partial,self.relative,count,self.original,self.admitted,self.fresh)
 def test_valid1024_publication_is_copied_byte_exact_and_old_sources_remain_unchanged(self):
  archive,receipt=self.create(1024);before=(archive.read_bytes(),receipt.read_bytes())
  row=self.classify(1024);self.assertEqual(row['status'],'VALID');self.assertTrue(self.reuse(row,1024))
  self.assertEqual((self.fresh/'1024'/self.relative).read_bytes(),before[0]);self.assertEqual((archive.read_bytes(),receipt.read_bytes()),before)
 def test_malformed_receipt_requires_explicit_snapshot_and_recomputes_full_source(self):
  archive,receipt=self.create(1024);receipt.write_text('{"entry":',encoding='utf8');before=(archive.read_bytes(),receipt.read_bytes())
  with self.assertRaises(Exception):strict._reuse_entry({'root':str(self.old)},self.relative,1024,self.original,self.admitted,self.fresh)
  row=self.classify(1024);self.assertEqual(row['status'],'INTERRUPTED');self.assertFalse(self.reuse(row,1024));self.assertEqual((archive.read_bytes(),receipt.read_bytes()),before)
 def test_unpublished_archive_is_preserved_and_not_admitted(self):
  archive,receipt=self.create(1024);receipt.unlink();before=archive.read_bytes();row=self.classify(1024)
  self.assertEqual(row['status'],'UNPUBLISHED');self.assertFalse(self.reuse(row,1024));self.assertEqual(archive.read_bytes(),before)
 def test_certified_valid_bytes_later_changing_are_fatal(self):
  archive,_=self.create(1024);row=self.classify(1024)
  with archive.open('ab')as stream:stream.write(b'DEBUG_CHANGED_AFTER_CERTIFICATION')
  with self.assertRaisesRegex(base.BankContractError,'changed after snapshot'):self.reuse(row,1024)
 def test_complete_source_array_contract_violation_is_fatal_not_interruption_fallback(self):
  payload=source_fixture(1024);payload['source_data'][0,0,0,0]+=100
  archive=self.old/'1024'/self.relative;archive.parent.mkdir(parents=True);recovery._atomic_npz(archive,payload)
  valid=base.read_json(self.old/'512/entry_receipts'/Path(self.relative).with_suffix('.json'))
  valid.update(new_entry_sha256=base.file_sha(archive),generation_prefix=dict(path=str(self.old/'512'/self.relative),sha256=base.file_sha(self.old/'512'/self.relative),count=512))
  recovery._atomic_json(self.old/'1024/entry_receipts'/Path(self.relative).with_suffix('.json'),valid)
  with self.assertRaisesRegex(base.BankContractError,'source_data'):self.classify(1024)
 def test_nested512_byte_change_is_fatal_even_with_complete1024_archive(self):
  self.create(1024);prefix=self.old/'512'/self.relative;data=base.read_entry(prefix);data['candidate_centers'][400,0]+=10000
  with prefix.open('wb')as stream:np.savez(stream,**data)
  with self.assertRaises(base.BankContractError):self.classify(1024)

class AtomicPublicationTests(unittest.TestCase):
 def test_atomic_json_npz_reject_existing_outputs_without_overwrite_or_stale_temporary(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as temporary:
   folder=Path(temporary).resolve();json_path=folder/'DEBUG.json';npz_path=folder/'DEBUG.npz'
   recovery._atomic_json(json_path,{'DEBUG':True});recovery._atomic_npz(npz_path,source_fixture(512));before=(json_path.read_bytes(),npz_path.read_bytes())
   with self.assertRaises(FileExistsError):recovery._atomic_json(json_path,{'changed':True})
   with self.assertRaises(FileExistsError):recovery._atomic_npz(npz_path,source_fixture(1024))
   self.assertEqual((json_path.read_bytes(),npz_path.read_bytes()),before);self.assertFalse(list(folder.glob('*.tmp')))
 def test_failed_promotion_exposes_no_partial_final_path(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as temporary:
   folder=Path(temporary).resolve();path=folder/'DEBUG.json'
   with patch.object(recovery,'_promote',side_effect=OSError('DEBUG interrupted before publication')):
    with self.assertRaises(OSError):recovery._atomic_json(path,{'DEBUG':True})
   self.assertFalse(path.exists());self.assertFalse(list(folder.glob('*.tmp')))
 def test_private_clone_keeps_all642_three_stage_scheduler_and_frozen_provider_predicates(self):
  self.assertIs(recovery._build.__code__,strict.build_staged_banks.__code__)
  self.assertIs(recovery.FastGeometryProvider._full_proposals.__code__,base.LargeGeometryProvider.proposals.__code__)
  self.assertIs(recovery.FastFiniteLiverCenters.sampler.__code__,base.FiniteLiverCenters.sampler.__code__)
  self.assertIs(base.write_json_exclusive,base.historical.write_json_exclusive)

class CompletedPrefixGuardCadenceTests(unittest.TestCase):
 def test_all642_actual_archive_copies_keep_full_array_SHA_checks_with_quick_per_entry_guards(self):
  # CPU DEBUG: all642 source identities with tiny fixture arrays. The sealed
  # certificate is isolated; the real per-entry SHA/array reader/copier runs.
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as temporary:
   folder=Path(temporary).resolve();old=folder/'old';fresh=folder/'fresh';fresh.mkdir()
   original_dir=folder/'original';admitted_dir=folder/'admitted';original_dir.mkdir();admitted_dir.mkdir()
   original_path=original_dir/'DEBUG_original.npz';admitted_path=admitted_dir/'DEBUG_256.npz';archive=old/'DEBUG_512.npz';old.mkdir()
   original,admitted,payload=source_fixture(128),source_fixture(256),source_fixture(512)
   for path,data in((original_path,original),(admitted_path,admitted),(archive,payload)):recovery._atomic_npz(path,data)
   index=original_dir/'index.json';cases={'DEBUG_case_'+str(i):[]for i in range(81)};records={}
   digest=base.file_sha(archive);archive_asset=recovery._asset(archive)
   proof=base.historical.verify_preservation(original,payload,original_count=128,candidate_count=512)
   proof.update(admitted_prefix_count=256,all_admitted_prefix_centers_byte_preserved=True)
   for i in range(642):
    case='DEBUG_case_'+str(i%81);relative='entries/DEBUG_'+str(i)+'.npz';cases[case].append(relative)
    receipt=old/'512/entry_receipts'/Path(relative).with_suffix('.json')
    row=dict(entry=relative,case_id=case,new_entry_sha256=digest,original_entry_sha256=base.file_sha(original_path),proof=proof)
    recovery._atomic_json(receipt,row)
    records[relative]=dict(status='VALID',assets=dict(archive=archive_asset,receipt=recovery._asset(receipt)))
   recovery._atomic_json(index,dict(entries_by_case=cases))
   for name in('complete.json','index.json','preservation_manifest.json','preparation_started.json'):recovery._atomic_json(old/'512'/name,dict(DEBUG=True))
   actual_safe_entry=base.safe_entry
   def resolve_entry(root,relative):
    if Path(root)==original_dir:return original_path
    if Path(root)==admitted_dir:return admitted_path
    return actual_safe_entry(root,relative)
   calls=[]
   def guard(*,force=False):calls.append(force)
   partial=dict(root=str(old),recovery_snapshot='DEBUG_CERTIFIED_FULL642',recovery_snapshot_sha256='DEBUG_ONLY')
   with patch.object(recovery,'_snapshot',return_value=dict(records={'512':records})),patch.object(recovery.base,'verify_prepared_bank',return_value=dict(DEBUG=True,full_sources=642))as certificate,patch.object(recovery.base,'safe_entry',side_effect=resolve_entry),patch.object(base.historical,'verify_preservation',wraps=base.historical.verify_preservation)as array_checks,patch.dict(recovery._reuse_completed_prefix.__globals__,verify_resources=guard):
    result=recovery._reuse_completed_prefix(partial,512,dict(bank_index=str(index)),folder/'DEBUG_binding.json','DEBUG_BINDING_SHA',admitted_dir,fresh)
   self.assertEqual(calls,[False]*642);self.assertEqual(array_checks.call_count,642);certificate.assert_called_once()
   self.assertEqual(result['source_entries'],642);self.assertEqual(set(result['entries']),set(records))
   targets=list((fresh/'512/entries').glob('*.npz'));self.assertEqual(len(targets),642)
   self.assertTrue(all(base.file_sha(path)==digest for path in targets));self.assertEqual(base.file_sha(archive),digest)
   self.assertEqual(len(list((fresh/'512/entry_receipts/entries').glob('*.json'))),642)
 def test_exact_private_replacement_preserves_all_build_force_boundaries_and_original_function(self):
  self.assertEqual(recovery._prefix_source,inspect.getsource(strict._reuse_completed_prefix))
  self.assertEqual(recovery._prefix_source.count('verify_resources(force=True)'),1)
  self.assertNotIn(('force',),recovery._reuse_completed_prefix.__code__.co_consts)
  self.assertIn(('force',),strict._reuse_completed_prefix.__code__.co_consts)
  self.assertIs(recovery._build.__code__,strict.build_staged_banks.__code__)
  original=inspect.getsource(strict.build_staged_banks)
  self.assertIn('verify_resources(force=True);original_index=',original)
  self.assertIn('if reused is not None:\n     verify_resources(force=True)',original)
  self.assertIs(recovery.verify_resources.__code__,strict.verify_resources.__code__)
 def test_guard_delegates642_sources_to_existing_quick_poll_and_keeps_forced_boundaries(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as temporary:
   path=Path(temporary).resolve()/'DEBUG_request.json';recovery._atomic_json(path,dict(DEBUG=True));calls=[]
   class Lease:
    def verify(self,*,force=False):calls.append(('verify',force))
    def quick_verify(self):calls.append(('quick',))
   with patch.object(recovery,'_ROOT',None),patch.object(recovery,'_LEASE',Lease()),patch.object(recovery,'_REQUEST_IDENTITY',(str(path),base.stat_identity(path))),patch.dict(os.environ,BASIC_CP_LEASE_REQUEST=str(path)):
    recovery.verify_resources(force=True)
    for _ in range(642):recovery.verify_resources()
    recovery.verify_resources(force=True)
   self.assertEqual(calls,[('verify',True)]+[('quick',)]*642+[('verify',True)])

if __name__=='__main__':unittest.main()
