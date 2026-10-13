"""CPU DEBUG weak-reference lifecycle and full642 reuse scheduling tests."""
from __future__ import annotations
import copy,gc,types,unittest,weakref
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hiercp_v1x import basic_cp_large_bank_fast as fast
from hiercp_v1x import basic_cp_large_geometry_cache as geometry


class Prepared:
 pass


class LifecycleTests(unittest.TestCase):
 def provider(self):
  provider=object.__new__(fast.FastGeometryProvider)
  provider.local=lambda *args,**kwargs:None
  provider.regions=types.SimpleNamespace(full_organ_mask=np.ones((3,3,3),dtype=bool))
  geometry.optimize_provider(provider,cache_limit_bytes=1024)
  return provider

 def test_old_edt_forbidden_flat_and_source_are_released_before_new_patient_loader(self):
  provider=self.provider();provider._fast_case_id='old'
  distance=np.arange(60,dtype=np.float64);forbidden=np.ones(60,dtype=bool);flat=np.arange(30,dtype=np.int64)
  references=[weakref.ref(value)for value in (distance,forbidden,flat)]
  provider.patient_cache={'distance':distance,'forbidden':forbidden};provider.flat_liver=flat
  # The old candidate pool namespace also owns the same cache; BOTH refs must go.
  provider.patient_pool=types.SimpleNamespace(cache=provider.patient_cache)
  prepared=Prepared();prepared.array=np.arange(40,dtype=np.float32)
  references.append(weakref.ref(prepared));provider._exact_geometry_cache_manager.prepared=prepared
  del distance,forbidden,flat,prepared
  expected=np.arange(125,dtype=np.int16).reshape(1,5,5,5)
  def actual_loader(self,case_id):
   gc.collect();self.asserted_before_loading_new_CT=all(reference()is None for reference in references)
   if not self.asserted_before_loading_new_CT:raise AssertionError('Old patient/source remained before loader')
   self.preseg=expected.copy();self.regions=types.SimpleNamespace(full_organ_mask=np.ones((5,5,5),dtype=bool))
   self.patient_cache={};self.flat_liver=np.arange(4);return self.preseg
  with patch.object(fast.base.LargeGeometryProvider,'load_patient',actual_loader):
   observed=provider.load_patient('new')
  self.assertTrue(provider.asserted_before_loading_new_CT)
  self.assertEqual(observed.tobytes(),expected.tobytes())
  self.assertEqual(provider._fast_case_id,'new')

 def test_previous_source_lru_released_before_next_full_source_prepare_with_patient_arrays_retained(self):
  provider=self.provider();manager=provider._exact_geometry_cache_manager
  prepared=Prepared();fields=np.arange(80,dtype=np.float32)
  prepared_reference=weakref.ref(prepared);fields_reference=weakref.ref(fields)
  manager.prepared=prepared;manager.cache=types.SimpleNamespace(records={'derived':fields})
  organ=manager.organ_uint8;organ_before=organ.tobytes()
  del prepared,fields
  expected=[(np.array([2,3,4]),np.array([2,3,4]))]
  def original_full_proposals(self,case_id,entry):
   gc.collect()
   if prepared_reference()is not None or fields_reference()is not None:
    raise AssertionError('Old source remained before full source.prepare')
   if self._exact_geometry_cache_manager.organ_uint8 is not organ:
    raise AssertionError('Patient static organ cache was discarded unnecessarily')
   yield from expected
  with patch.object(fast.FastGeometryProvider,'_full_proposals',original_full_proposals):
   observed=list(provider.proposals('DEBUG',{}))
  self.assertEqual(observed[0][0].tobytes(),expected[0][0].tobytes())
  self.assertEqual(manager.organ_uint8.tobytes(),organ_before)

 def test_same_patient_static_cache_is_reused_without_reload(self):
  provider=self.provider();provider._fast_case_id='same';provider.preseg=np.arange(27).reshape(1,3,3,3)
  provider.patient_cache={'distance':np.arange(27,dtype=np.float64)}
  provider.verify_patient_inputs=lambda:None
  original=provider.patient_cache
  with patch.object(fast.base.LargeGeometryProvider,'load_patient',side_effect=AssertionError('No reload')):
   self.assertIs(provider.load_patient('same'),provider.preseg)
  self.assertIs(provider.patient_cache,original)


class FullPrefixReuseSchedulingTests(unittest.TestCase):
 def test_full642_completed_prefix_copies_all_sources_without_any_patient_or_geometry_call(self):
  groups={f'case_{index}':[]for index in range(81)}
  for index in range(642):groups[f'case_{index%81}'].append(f'entries/source_{index:03d}.npz')
  original=types.SimpleNamespace()
  partial={'root':'/DEBUG/old'};destination=Path('/DEBUG/new');binding={'bank_index':'/DEBUG/original/index.json'}
  copied=[];written=[]
  def json_read(path):return {'entries_by_case':groups}
  with patch.object(Path,'exists',return_value=True),\
       patch.object(fast.base,'file_sha',return_value='f'*64),\
       patch.object(fast.base,'verify_prepared_bank',return_value={'source_entries':642,'candidate_count':512})as admitted,\
       patch.object(fast.base,'read_json',side_effect=json_read),\
       patch.object(fast.base,'read_entry',return_value=original),\
       patch.object(fast.base,'safe_entry',side_effect=lambda root,relative:Path(root)/relative),\
       patch.object(fast,'verify_resources'),\
       patch.object(fast,'_reuse_entry',side_effect=lambda p,r,c,o,a,d:copied.append(r)or True),\
       patch.object(fast.base,'write_json_exclusive',side_effect=lambda p,v:written.append(v)),\
       patch.object(fast.FastGeometryProvider,'load_patient',side_effect=AssertionError('No CT load')),\
       patch.object(fast.FastGeometryProvider,'proposals',side_effect=AssertionError('No graph generation')):
   record=fast._reuse_completed_prefix(partial,512,binding,Path('/DEBUG/binding.json'),'b'*64,
                                       Path('/DEBUG/256'),destination)
  self.assertEqual(len(copied),642);self.assertEqual(len(set(copied)),642)
  self.assertEqual(set(copied),{name for names in groups.values()for name in names})
  self.assertFalse(record['patient_CT_loaded']);self.assertFalse(record['full_graph_recomputed'])
  self.assertEqual(record['source_entries'],642);self.assertEqual(len(written),1)
  self.assertEqual(admitted.call_args.kwargs['candidate_count'],512)

 def test_missing_complete_receipt_uses_full_normal_generation_not_a_fake_completion(self):
  with patch.object(Path,'exists',return_value=False),patch.object(fast.base,'verify_prepared_bank')as verify:
   self.assertIsNone(fast._reuse_completed_prefix({'root':'/DEBUG/old'},512,{},Path('/DEBUG/binding.json'),
       'f'*64,Path('/DEBUG/256'),Path('/DEBUG/new')))
  verify.assert_not_called()


if __name__=='__main__':unittest.main()
