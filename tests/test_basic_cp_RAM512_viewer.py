"""CPU DEBUG: actual limit values change; UI layout and native formatter do not."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from tools import watch_basic_cp_RAM512 as view
from tools import watch_basic_cp_ablation as native


class RAM512ViewerTests(unittest.TestCase):
    def build(self,gpu,stage,request=None):
        q={'memory_budget_revision':512,'aggregate_memory_cap_GiB':512,
            'effective_native_RSS_limit_GiB':32,'effective_shared_bank_RSS_limit_GiB':320,
            'shared_banks':'DEBUG_shared','candidates_per_source':2048}
        if request is not None:q=request
        original={'physical_GPU':gpu,'controller_stage':stage,'RAM_limit_GiB':96,
            'status':'RUNNING','sources_done':10,'stage':'nnunet_training','stage_label':'DEBUG'}
        viewer=SimpleNamespace(main=native.main,read=lambda p:q if p.name=='request.json'else{})
        expanded=SimpleNamespace(expanded_snapshot=lambda root:dict(original))
        return view.build_entry(expanded,viewer),original

    def test_native_snapshot_changes_only_real_RAM_limit(self):
        entry,old=self.build(6,'train');new=entry.__globals__['snapshot'](Path('DEBUG_root'))
        self.assertEqual(new,dict(old,RAM_limit_GiB=32))
        self.assertIs(entry.__code__,native.main.__code__)

    def test_actual_CPU_source_uses320_without_changing_native_formatter(self):
        entry,old=self.build(5,'candidate_bank')
        from unittest.mock import patch
        with patch.object(view.admission,'candidate_progress_reason',return_value=('DEBUG source10/642',{})):
            new=entry.__globals__['snapshot'](Path('DEBUG_root'))
        self.assertEqual(new['RAM_limit_GiB'],320)
        self.assertEqual(new['sources_done'],old['sources_done'])
        self.assertIs(entry.__code__,native.main.__code__)

    def test_waiting_native5_uses_actual32_limit(self):
        entry,_=self.build(5,'waiting_candidate_bank')
        self.assertEqual(entry.__globals__['snapshot'](Path('DEBUG_root'))['RAM_limit_GiB'],32)

    def test_non_RAM512_request_is_explicit_error(self):
        entry,_=self.build(6,'train',{'memory_budget_revision':512,'aggregate_memory_cap_GiB':800})
        with self.assertRaisesRegex(ValueError,'actual approved request limits'):
            entry.__globals__['snapshot'](Path('DEBUG_root'))


if __name__=='__main__':unittest.main()
