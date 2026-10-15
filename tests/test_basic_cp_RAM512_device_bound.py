"""CPU DEBUG: conservative device allocation never understates owned tmpfs.

Mocked Linux metadata tests cover accounting algebra, not scientific results.
"""
from pathlib import Path, PurePosixPath
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from hiercp_v1x import basic_cp_512_resources as resource

GIB = resource.GIB


class WholeDeviceBoundTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent/'DEBUG_owned_tmpfs'
        self.metadata = SimpleNamespace(st_dev=7, st_ino=123, st_uid=41833, st_gid=2000,
            st_mode=stat.S_IFDIR|0o700, st_blocks=0)
        self.fs = SimpleNamespace(f_blocks=1024*GIB//4096, f_bfree=824*GIB//4096, f_frsize=4096)
        self.mounts = (('/',1,0,'8:1','/','ext4'),('/dev/shm',2,1,'0:7','/','tmpfs'))
        self.patches = [
            patch.object(Path,'lstat',return_value=self.metadata),
            patch.object(resource,'_device_mount_guard',return_value=self.mounts),
            patch.object(resource.os,'open',return_value=100),
            patch.object(resource.os,'fstat',return_value=self.metadata),
            patch.object(resource.os,'fstatvfs',return_value=self.fs,create=True),
            patch.object(resource.os,'major',return_value=0,create=True),
            patch.object(resource.os,'minor',return_value=7,create=True),
            patch.object(resource.os,'close'),
        ]
        self.mocks = [item.start() for item in self.patches]
        for item in self.patches:self.addCleanup(item.stop)

    def call(self, **kwargs):
        return resource.allocated_tmpfs(self.root,{},uid=41833,
            conservative_device_ceiling_bytes=272*GIB,**kwargs)

    def test_device_upper_bound_skips_inode_and_fd_scan_without_fabricating_inventory(self):
        with patch.object(resource.os,'scandir',side_effect=AssertionError('walked inode tree')):
            value = self.call()
        self.assertEqual(value['owned_tmpfs_allocated_bytes'],200*GIB)
        self.assertEqual(value['accounting'],'conservative_whole_tmpfs_device_allocated_upper_bound')
        self.assertIsNone(value['inode_count'])
        self.assertIsNone(value['deleted_open_inode_observations'])
        self.assertFalse(value['unknown_file_charge_is_actual_owned_attribution'])
        self.assertFalse(value['exact_inode_inventory_performed'])
        self.mocks[-1].assert_called_once_with(100)

    def test_sparse_logical_size_and_deleted_open_files_remain_covered_by_device_charge(self):
        # No logical filesize, outside owner or inaccessible FD is subtracted.
        process = Mock(pid=77)
        process.memory_info.side_effect = AssertionError('not a compute RSS scan')
        process.cmdline.side_effect = AssertionError('device charge already includes deleted files')
        value = resource.allocated_tmpfs(self.root,{77:process},uid=41833,
            conservative_device_ceiling_bytes=272*GIB)
        self.assertEqual(value['owned_tmpfs_allocated_bytes'],200*GIB)
        self.assertTrue(value['outside_root_files_conservatively_included'])
        self.assertTrue(value['deleted_open_inodes_included_in_device_charge'])

    def test_growth_between_observations_uses_larger_allocation(self):
        smaller = SimpleNamespace(f_blocks=1024*GIB//4096,f_bfree=924*GIB//4096,f_frsize=4096)
        self.mocks[4].side_effect = [smaller,self.fs]
        self.assertEqual(self.call()['owned_tmpfs_allocated_bytes'],200*GIB)

    def exact_empty(self):
        entries = MagicMock(); entries.__enter__.return_value = iter([])
        return patch.object(resource.os,'scandir',return_value=entries)

    def test_device_overcharge_above272_falls_back_to_exact_owned_scan(self):
        self.mocks[4].return_value = SimpleNamespace(f_blocks=1024*GIB//4096,
            f_bfree=500*GIB//4096,f_frsize=4096)
        with self.exact_empty() as walk:
            value = self.call()
        walk.assert_called_once_with(self.root)
        self.assertEqual(value['owned_tmpfs_allocated_bytes'],0)
        self.assertEqual(value['inode_count'],1)
        self.assertEqual(value['accounting'],'unique_owned_allocated_inode_blocks_including_open_deleted_FDs')

    def test_nested_or_other_filesystem_mount_selects_exact_scan(self):
        self.mocks[1].return_value = None
        with self.exact_empty() as walk:
            value = self.call()
        walk.assert_called_once_with(self.root)
        self.mocks[2].assert_not_called()
        self.assertEqual(value['inode_count'],1)

    def test_root_device_inode_or_owner_change_is_fatal(self):
        for key,new in [('st_dev',8),('st_ino',999),('st_uid',19535)]:
            with self.subTest(key=key):
                changed = SimpleNamespace(**vars(self.metadata)); setattr(changed,key,new)
                self.mocks[0].side_effect = [self.metadata,changed]
                with self.assertRaisesRegex(RuntimeError,'root or mount namespace changed'):
                    self.call()
                self.mocks[0].side_effect = None

    def test_opened_directory_fd_must_retain_exact_owned_root_identity(self):
        self.mocks[3].return_value = SimpleNamespace(**dict(vars(self.metadata),st_uid=999))
        with self.assertRaisesRegex(RuntimeError,'root FD changed'):
            self.call()

    def test_mount_change_during_observation_is_fatal(self):
        self.mocks[1].side_effect = [self.mounts,None]
        with self.assertRaisesRegex(RuntimeError,'mount namespace changed'):
            self.call()

    def test_malformed_device_counters_and_real_io_errors_never_become_zero(self):
        for fields in (dict(f_blocks=1,f_bfree=2,f_frsize=4096),
                       dict(f_blocks=1,f_bfree=0,f_frsize=0),
                       dict(f_blocks=True,f_bfree=0,f_frsize=4096)):
            with self.subTest(fields=fields):
                self.mocks[4].return_value = SimpleNamespace(**fields)
                with self.assertRaisesRegex(ValueError,'allocation counters'):
                    self.call()
        self.mocks[4].side_effect = PermissionError('DEBUG actual statvfs failure')
        with self.assertRaises(PermissionError):self.call()

    def test_optional_fast_path_cannot_change_the_approved_ceiling(self):
        for limit in (True,144*GIB,273*GIB,272.0*GIB):
            with self.subTest(limit=limit),self.assertRaisesRegex(ValueError,'272GiB'):
                resource.allocated_tmpfs(self.root,{},uid=41833,
                    conservative_device_ceiling_bytes=limit)

    def test_legacy_no_optin_never_uses_device_bound(self):
        with self.exact_empty():
            value = resource.allocated_tmpfs(self.root,{},uid=41833)
        self.mocks[1].assert_not_called()
        self.assertEqual(value['inode_count'],1)


class MountNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.root = PurePosixPath('/dev/shm/DEBUG_owned')
        self.mountinfo = ('1 0 8:1 / / rw - ext4 /dev/sda rw\n'
                          '2 1 0:7 / /dev/shm rw - tmpfs shm rw\n')
        for attribute,value in [('major',0),('minor',7)]:
            item = patch.object(resource.os,attribute,return_value=value,create=True)
            item.start();self.addCleanup(item.stop)

    def call(self,text):
        with patch.object(Path,'read_text',return_value=text):
            return resource._device_mount_guard(self.root,7,Path('/DEBUG_proc'))

    def test_covering_tmpfs_device_is_admitted(self):
        result = self.call(self.mountinfo)
        self.assertEqual(result[-1][0],'/dev/shm')
        self.assertEqual(result[-1][-1],'tmpfs')

    def test_nested_mount_cannot_hide_another_device_allocation(self):
        text = self.mountinfo+'3 2 8:4 / /dev/shm/DEBUG_owned/nested rw - ext4 /dev/sdb rw\n'
        self.assertIsNone(self.call(text))

    def test_unrelated_mount_does_not_disable_device_bound(self):
        text = self.mountinfo+'3 1 8:4 / /outside rw - ext4 /dev/sdb rw\n'
        self.assertIsNotNone(self.call(text))

    def test_bind_mount_below_root_is_rejected_even_if_device_number_matches(self):
        text = self.mountinfo+'3 2 0:7 /elsewhere /dev/shm/DEBUG_owned/nested rw - tmpfs shm rw\n'
        self.assertIsNone(self.call(text))

    def test_device_or_filesystem_mismatch_selects_exact_scan(self):
        self.assertIsNone(self.call(self.mountinfo.replace('0:7','0:8')))
        self.assertIsNone(self.call(self.mountinfo.replace('tmpfs shm','ext4 /dev/sdb')))

    def test_escaped_nested_mount_is_also_detected(self):
        self.root = PurePosixPath('/dev/shm/DEBUG owned')
        text = self.mountinfo+'3 2 8:4 / /dev/shm/DEBUG\\040owned/nested rw - ext4 /dev/sdb rw\n'
        self.assertIsNone(self.call(text))

    def test_malformed_or_unreadable_mountinfo_is_fatal(self):
        for text in ('', 'bad record', self.mountinfo+'3 2 8:4 / /bad\\999 rw - ext4 /dev/sdb rw\n'):
            with self.subTest(text=text),self.assertRaises((ValueError,RuntimeError)):
                self.call(text)
        with patch.object(Path,'read_text',side_effect=PermissionError('DEBUG mountinfo failure')):
            with self.assertRaises(PermissionError):
                resource._device_mount_guard(self.root,7,Path('/DEBUG_proc'))


if __name__ == '__main__':unittest.main()
