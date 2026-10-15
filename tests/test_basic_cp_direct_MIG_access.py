import os
from pathlib import Path
import tempfile
import unittest

from hiercp_v1x import basic_cp_direct_MIG_access as access


class DirectMIGTests(unittest.TestCase):
    def accounting(self, **updates):
        values = dict(rows=[{'PID':1,'RSS_bytes':2},{'PID':2,'RSS_bytes':10},{'PID':3,'RSS_bytes':5}],
                      native_RSS_by_arm={'a':8,'b':2}, cohort_PIDs={1,2}, expected_arms=2,
                      native_reserve_bytes=20, auxiliary_reserve_bytes=10,
                      max_user_RAM_bytes=100, own_tmpfs_bytes=7, host_available_bytes=100)
        values.update(updates)
        return access._memory_accounting(**values)

    def mig(self, index, **changes):
        row = dict(MIG_UUID='MIG-'+str(index),parent_UUID='GPU-parent',GPU_instance_id=index,
                   compute_instance_id=0,compute_process_count=0,graphics_process_count=0,
                   total_bytes=10*access.GIB,free_bytes=10*access.GIB,used_bytes=0)
        row.update(changes)
        return row

    def test_cpu_ranges_are_exact(self):
        self.assertEqual(access.parse_CPU_list('0-3,7,11-12'), [0,1,2,3,7,11,12])
        with self.assertRaises(ValueError):access.parse_CPU_list('8-2')

    def test_current_native_and_auxiliary_are_not_double_reserved(self):
        proof = self.accounting()
        self.assertEqual(proof['current_uid_RSS_bytes'],17)
        self.assertEqual(proof['cohort_auxiliary_RSS_bytes'],2)
        self.assertEqual(proof['additional_reserved_bytes'],38)
        self.assertEqual(proof['prospective_user_RAM_bytes'],62)

    def test_actual_inconsistency_or_capacity_refuses(self):
        cases = [{'cohort_PIDs':{99}}, {'native_RSS_by_arm':{'a':15,'b':2}},
                 {'native_RSS_by_arm':{'a':30,'b':2}}, {'auxiliary_reserve_bytes':1},
                 {'host_available_bytes':37}, {'max_user_RAM_bytes':61}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):self.accounting(**changes)

    def test_same_parent_empty_exactly_six_and_explicit_exclusion(self):
        rows = [self.mig(i)for i in range(7)]+[self.mig(100,parent_UUID='GPU-other')]
        selected=access.choose_empty_MIGs(rows,'GPU-parent',exclude_UUIDs=['MIG-3'])
        self.assertEqual([x['GPU_instance_id']for x in selected],[0,1,2,4,5,6])

    def test_busy_foreign_or_undersized_is_not_admitted(self):
        cases = [{'parent_UUID':'GPU-other'},{'compute_process_count':1},{'graphics_process_count':1},
                 {'used_bytes':65*2**20},{'total_bytes':8*access.GIB},{'free_bytes':8*access.GIB}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                access.choose_empty_MIGs([self.mig(i)for i in range(5)]+[self.mig(9,**changes)],'GPU-parent')

    def test_overlapping_MIG_identity_is_rejected(self):
        with self.assertRaises(RuntimeError):
            access.choose_empty_MIGs([self.mig(i)for i in range(5)]+[self.mig(5,MIG_UUID='MIG-0')],'GPU-parent')

    @unittest.skipUnless(os.name=='posix','Linux tmpfs allocation accounting uses actual st_blocks')
    def test_tmpfs_hardlinks_are_charged_once(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);first=root/'checkpoint';first.write_bytes(b'checkpoint-data')
            (root/'same-checkpoint').hardlink_to(first);(root/'not-followed').symlink_to(first)
            proof=access._owned_tmpfs(first.stat().st_uid,[root])
            self.assertEqual(proof['regular_files'],1)
            self.assertEqual(proof['allocated_bytes'],first.stat().st_blocks*512)


if __name__=='__main__':unittest.main()
