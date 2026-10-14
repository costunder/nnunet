"""Independent CPU DEBUG invariants for the approved physical512GiB ceiling."""
import copy,stat,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from hiercp_v1x import basic_cp_512_resources as guard
from hiercp_v1x import basic_cp_512_claims as claims

GIB=2**30

class IndependentRAM512AccountingTests(unittest.TestCase):
 def values(self,*,post=False):
  source=0 if post else 320*GIB
  natives={gpu:32*GIB for gpu in range(7)}if post else{6:32*GIB}
  return dict(store_bytes=(272 if post else 144)*GIB,native_rss=natives,source_rss=source,
   auxiliary_rss=16*GIB,source_pending=not post,host_total=1007*GIB,host_available=700*GIB,
   registered_rss=source+sum(natives.values()))

 def test_actual_source_and_prioritized_native_phase_maximum_is_exactly512(self):
  evidence=guard.enforce_accounting(**self.values())
  self.assertEqual(evidence['accounted_total_RAM_bytes'],512*GIB)
  self.assertEqual(evidence['source_process_tree_RSS_bytes'],320*GIB)
  self.assertEqual(evidence['native_RSS_by_GPU_bytes'],{'6':32*GIB})

 def test_all_seven_full_native_trees_and_independent_archives_fit_post_source512(self):
  evidence=guard.enforce_accounting(**self.values(post=True))
  self.assertEqual(evidence['accounted_total_RAM_bytes'],512*GIB)
  self.assertEqual(evidence['owned_tmpfs_limit_bytes'],272*GIB)
  self.assertEqual(len(evidence['native_RSS_by_GPU_bytes']),7)
  self.assertTrue(evidence['scientific_scale_preserved'])

 def test_store_even_one_byte_beyond144_or272_is_not_ignored(self):
  for post in(False,True):
   value=self.values(post=post);value['store_bytes']+=1
   with self.subTest(post=post),self.assertRaisesRegex(MemoryError,'tmpfs'):guard.enforce_accounting(**value)

 def test_source_one_byte_above320_does_not_shrink_workers_or_data(self):
  value=self.values();value['source_rss']+=1
  with self.assertRaisesRegex(MemoryError,'320GiB'):guard.enforce_accounting(**value)

 def test_every_native_tree_keeps_effective32_even_in_post_source_phase(self):
  for gpu in range(7):
   value=self.values(post=True);value['native_rss'][gpu]+=1
   with self.subTest(gpu=gpu),self.assertRaisesRegex(MemoryError,'32GiB'):guard.enforce_accounting(**value)

 def test_cache_helper_and_other_auxiliary_memory_is_not_fabricated_as_zero(self):
  value=self.values();value['auxiliary_rss']+=1
  with self.assertRaisesRegex(MemoryError,'16GiB'):guard.enforce_accounting(**value)

 def test_two_native_trees_are_forbidden_while_actual_source_is_pending(self):
  value=self.values();value['native_rss'][1]=GIB
  with self.assertRaisesRegex(MemoryError,'one complete native'):guard.enforce_accounting(**value)

 def test_candidate_native_preflight_debug_train_and_evaluation_cannot_overlap_source(self):
  for gpu in(3,4,5):
   value=self.values();value['native_rss']={gpu:GIB}
   with self.subTest(gpu=gpu),self.assertRaisesRegex(MemoryError,'Candidate native'):guard.enforce_accounting(**value)

 def test_source_cannot_silently_survive_post_source_store272_phase(self):
  value=self.values(post=True);value['source_rss']=1
  with self.assertRaisesRegex(RuntimeError,'live source'):guard.enforce_accounting(**value)

 def test_host_shortage_is_typed_at_exact_boundary_without_fake_free_bytes(self):
  value=self.values();value['host_available']=0
  evidence=guard.enforce_accounting(**value);self.assertEqual(evidence['measured_available_plus_owned_charge_bytes'],512*GIB)
  value['source_rss']-=1;value['registered_rss']-=1
  with self.assertRaises(guard.RAMAdmissionUnavailable)as caught:guard.enforce_accounting(**value)
  self.assertEqual(caught.exception.observation['shortfall_bytes'],1)

 def test_input_accounting_values_and_native_cohort_are_never_mutated(self):
  value=self.values(post=True);before=copy.deepcopy(value)
  guard.enforce_accounting(**value);self.assertEqual(value,before)

 def test_native_claim_forecast_reserves320_before_CPU_actor_is_visible(self):
  state=dict(claims={},source_pending=True,priority_complete=False)
  before=copy.deepcopy(state);plan=claims.reservation_bytes(state)
  self.assertEqual(plan['required_capacity_bytes'],480*GIB)
  state['claims']['DEBUG6']=dict(kind='native',gpu=6)
  self.assertEqual(claims.reservation_bytes(state)['required_capacity_bytes'],512*GIB)
  self.assertEqual(before,dict(claims={},source_pending=True,priority_complete=False))

 def test_no_second_claim_or_nonprioritized_claim_fits_during_source(self):
  for natives in((6,1),(0,)):
   state=dict(claims={str(gpu):dict(kind='native',gpu=gpu)for gpu in natives},source_pending=True,priority_complete=False)
   with self.subTest(natives=natives),self.assertRaises(claims.ResourceClaimUnavailable):claims.reservation_bytes(state)

 def test_after_actual_source_release_272_plus_all_native_reservations_stays512(self):
  state=dict(claims={str(gpu):dict(kind='native',gpu=gpu)for gpu in range(7)},source_pending=False,priority_complete=False)
  plan=claims.reservation_bytes(state);self.assertEqual(plan['required_capacity_bytes'],512*GIB)
  self.assertEqual(plan['source_reserved_bytes'],0);self.assertEqual(plan['owned_tmpfs_reserved_bytes'],272*GIB)

class IndependentTmpfsAndReleaseTests(unittest.TestCase):
 def setUp(self):
  self.root=Path(__file__).resolve().parents[1]/'tmp/DEBUG_SYNTHETIC_LINUX_BLOCKS'
  self.proc_root=self.root.parent/'DEBUG_PROC'
  self.process=Mock(pid=123)
  self.process.status.return_value='sleeping'
  self.metadata=dict(pid=123,create_time=1.0,UID=41833)

 def inode(self,number,blocks,mode=stat.S_IFREG,uid=41833):
  return types.SimpleNamespace(st_dev=27,st_ino=number,st_uid=uid,st_blocks=blocks,st_mode=mode|0o600,st_size=999999)

 def entries(self,rows):
  values=[]
  for name,number,blocks in rows:
   entry=Mock(path=str(self.root/name));entry.stat.return_value=self.inode(number,blocks);values.append(entry)
  context=Mock();context.__enter__=Mock(return_value=iter(values));context.__exit__=Mock(return_value=False)
  return context

 def test_hardlinks_and_live_alias_FD_count_once_but_open_deleted_payload_counts(self):
  # Synthetic Linux inode blocks are deliberate: Windows stat has no st_blocks.
  fd7=self.proc_root/'123/fd/7';fd8=self.proc_root/'123/fd/8'
  with patch.object(Path,'lstat',return_value=self.inode(1,8,stat.S_IFDIR)),patch.object(guard.os,'scandir',return_value=self.entries([('payload',2,16),('payload_hardlink',2,16),('other',3,4)])),patch.object(Path,'iterdir',return_value=iter([fd7,fd8])),patch.object(guard.os,'readlink',side_effect=[str(self.root)+'/payload',str(self.root)+'/old_checkpoint (deleted)']),patch.object(Path,'stat',autospec=True,side_effect=lambda p:self.inode(2,16)if p==fd7 else self.inode(4,32)),patch.object(guard.common,'_metadata',return_value=self.metadata):
   result=guard.allocated_tmpfs(self.root,{123:self.process},uid=41833,proc_root=self.proc_root)
  self.assertEqual(result['owned_tmpfs_allocated_bytes'],(8+16+4+32)*512)
  self.assertEqual(result['inode_count'],4);self.assertEqual(len(result['deleted_open_inode_observations']),1)
  self.assertEqual(result['deleted_open_inode_observations'][0]['inode'],4)

 def test_store_inventory_access_denied_is_fatal_and_never_fabricates_zero(self):
  with patch.object(Path,'lstat',return_value=self.inode(1,8,stat.S_IFDIR)),patch.object(guard.os,'scandir',side_effect=PermissionError('DEBUG strict inventory denial')),self.assertRaises(PermissionError):
   guard.allocated_tmpfs(self.root,{},uid=41833,proc_root=self.proc_root)

 def test_same_UID_process_memory_access_denied_is_fatal(self):
  process=Mock(pid=123);process.uids.return_value=types.SimpleNamespace(real=41833)
  process.status.return_value='sleeping';process.memory_info.side_effect=guard.psutil.AccessDenied(123)
  with patch.object(guard.psutil,'process_iter',return_value=[process]),patch.object(guard.common,'_metadata',return_value=self.metadata),self.assertRaises(guard.psutil.AccessDenied):guard.own_process_inventory()

 def test_live_actor_missing_FD_inventory_is_not_accepted_as_released(self):
  with patch.object(Path,'lstat',return_value=self.inode(1,8,stat.S_IFDIR)),patch.object(guard.os,'scandir',return_value=self.entries([])),patch.object(Path,'iterdir',side_effect=FileNotFoundError('DEBUG fd path vanished')),patch.object(guard.common,'_metadata',return_value=self.metadata),self.assertRaisesRegex(RuntimeError,'Live own process'):
   guard.allocated_tmpfs(self.root,{123:self.process},uid=41833,proc_root=self.proc_root)

 def test_original_PID_birth_release_and_reused_foreign_CUDA_do_not_retain_old_claim(self):
  expected=dict(pid=123,create_time=1.0,UID=41833,command=['DEBUG_owned_source'])
  process=Mock(pid=123);process.create_time.return_value=2.0
  resolver=guard.ProcessResolver(Mock())
  with patch.object(guard.psutil,'Process',return_value=process),patch.object(guard.common,'_metadata',return_value=dict(self.metadata,create_time=2.0)),patch.object(guard.common,'_query',return_value='123, GPU-new-foreign\n'):
   observation=resolver.inspect(expected)
   self.assertEqual(observation['state'],'gone');self.assertTrue(observation.get('original_actor_released_and_numeric_PID_reused'))
   self.assertFalse(observation['replacement_actor_admitted'])
   self.assertFalse(resolver.has_cuda([expected]))

 def test_empty_source_capture_cannot_authorize_post_source272_phase(self):
  ledger=object.__new__(claims.Ledger);ledger.resolver=Mock()
  state=dict(source_pending=True,source_released=False,source_actor_identity=None,source_captured=[],claims={})
  receipt=dict(candidate_counts={'512':642,'1024':642,'2048':642},producer_fully_released=True,captured=[])
  with patch.object(claims,'_receipt',return_value=receipt),self.assertRaises(RuntimeError):ledger._phases(state,dict(source_complete=True,source_completion_proof={'DEBUG':'receipt'}))
  self.assertTrue(state['source_pending']);self.assertFalse(state['source_released'])

if __name__=='__main__':unittest.main()
