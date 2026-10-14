"""CPU DEBUG publication fixtures; no clinical subset or training results."""
import copy
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

if __name__=='__main__':unittest.main()
