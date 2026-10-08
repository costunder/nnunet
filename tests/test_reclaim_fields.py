"""Storage-maintenance unit tests; synthetic files, no neural run or CT use."""
from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "reclaim_fields_under_test", Path(__file__).resolve().parents[1] / "tools" / "reclaim_fields.py")
reclaim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reclaim)


def silent(*args, **kwargs):
    pass


def npy_bytes(value):
    header = repr(dict(descr="<f4", fortran_order=False, shape=(2, 2, 2)))
    header += " " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + struct.pack("<8f", *([value] * 8))


class ReclaimFieldsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="reclaim-fields-UNIT-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = [self.publish(name) for name in reclaim.EXPERIMENTS[:2]]

    def publish(self, experiment, case="liver_1", *, binding_suffix=""):
        directory = self.root / experiment / "data" / "whole_case_fields" / case
        directory.mkdir(parents=True)
        binding = dict(image_sha256="a" * 64, label_sha256="b" * 64, common_sha256="c" * 64,
                       shape=[2, 2, 2], spacing=[1.0, 1.0, 1.0], test_identity=binding_suffix)
        fields = {}
        for name, value in (("depth", 1.0), ("occupied", 2.0)):
            payload = npy_bytes(value)
            path = directory / (name + ".npy")
            path.write_bytes(payload)
            fields[name] = dict(file=path.name, units="mm", shape=[2, 2, 2], dtype="<f4",
                                array_bytes=32, file_bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest(), fortran_order=False)
        metadata = dict(format=reclaim.FORMAT, binding=binding,
                        binding_sha256=reclaim.digest(binding), fields=fields)
        metadata["metadata_sha256"] = reclaim.digest(metadata)
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf8")
        return directory

    def verified(self):
        return reclaim.verify(reclaim.scan(self.root), progress=silent)

    def apply_plan(self, plan):
        with tempfile.TemporaryFile(mode="w+", encoding="utf8") as journal:
            result = reclaim.apply(plan, journal, progress=silent)
            journal.seek(0)
            events = [json.loads(line) for line in journal]
        return result, events

    def fingerprint(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_ino, p.stat().st_nlink)
                for p in self.root.rglob("*") if p.is_file()}

    def test_scan_and_verification_do_not_change_files(self):
        before = self.fingerprint()
        plan = self.verified()
        self.assertEqual(len(plan["pairs"]), 2)
        self.assertTrue(plan["verified"])
        self.assertEqual(before, self.fingerprint())

    def test_apply_preserves_bytes_metadata_paths_and_protected_outputs(self):
        protected = self.root / reclaim.EXPERIMENTS[0] / "checkpoint_latest.pt"
        protected.write_bytes(b"preserved model and optimizer test artifact")
        before = {p: v[0] for p, v in self.fingerprint().items()}
        results, events = self.apply_plan(self.verified())
        self.assertEqual(len(results), 2)
        self.assertEqual(len(events), 4)
        self.assertEqual(before, {p: v[0] for p, v in self.fingerprint().items()})
        for name in ("depth.npy", "occupied.npy"):
            left, right = (d / name for d in self.cases)
            self.assertTrue(os.path.samefile(left, right))
            self.assertEqual(left.stat().st_nlink, 2)
        self.assertEqual(reclaim.scan(self.root)["pairs"], [])

    def test_original_field_loader_accepts_both_paths_after_consolidation(self):
        import numpy as np
        from hiercp_v1x import u_bridge_fields
        self.apply_plan(self.verified())
        for case in self.cases:
            binding = json.loads((case / "metadata.json").read_text())["binding"]
            arrays, metadata = u_bridge_fields._load(case, binding, lambda: None)
            try:
                np.testing.assert_array_equal(arrays[0], np.ones((2, 2, 2), dtype=np.float32))
                np.testing.assert_array_equal(arrays[1], np.full((2, 2, 2), 2, dtype=np.float32))
                self.assertFalse(arrays[0].flags.writeable)
                self.assertFalse(arrays[1].flags.writeable)
                self.assertEqual(metadata["binding"], binding)
            finally:
                for array in arrays:
                    array._mmap.close()

    def test_existing_hardlinks_are_not_claimed_as_recoverable(self):
        for name in ("depth.npy", "occupied.npy"):
            destination = self.cases[1] / name
            destination.unlink()
            os.link(self.cases[0] / name, destination)
        self.assertEqual(reclaim.scan(self.root)["pairs"], [])

    def test_corrupt_actual_payload_is_refused_without_replacement(self):
        payload = self.cases[0] / "depth.npy"
        original = payload.read_bytes()
        payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        before = self.fingerprint()
        with self.assertRaisesRegex(ValueError, "Actual payload SHA256 differs"):
            self.verified()
        self.assertEqual(before, self.fingerprint())

    def test_changed_stat_after_verification_stops_all_replacements(self):
        plan = self.verified()
        last = Path(plan["pairs"][-1]["target"]["path"])
        info = last.stat()
        os.utime(last, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))
        before = self.fingerprint()
        with self.assertRaisesRegex(RuntimeError, "File changed since inventory"):
            self.apply_plan(plan)
        self.assertEqual(before, self.fingerprint())

    def test_metadata_change_after_verification_is_refused(self):
        plan = self.verified()
        metadata = Path(plan["pairs"][-1]["target"]["metadata_path"])
        metadata.write_text(metadata.read_text() + "\n")
        before = self.fingerprint()
        with self.assertRaisesRegex(RuntimeError, "Metadata changed"):
            self.apply_plan(plan)
        self.assertEqual(before, self.fingerprint())

    def test_link_quota_failure_keeps_every_original(self):
        plan = self.verified()
        before = self.fingerprint()
        with mock.patch.object(reclaim.os, "link", side_effect=OSError(errno.EDQUOT, "quota")):
            with self.assertRaises(OSError) as raised:
                self.apply_plan(plan)
        self.assertEqual(raised.exception.errno, errno.EDQUOT)
        self.assertEqual(before, self.fingerprint())
        self.assertEqual(list(self.root.rglob(".field-link-*")), [])

    def test_rename_failure_cleans_only_own_link_and_keeps_originals(self):
        plan = self.verified()
        before = self.fingerprint()
        with mock.patch.object(reclaim.os, "replace", side_effect=OSError(errno.EDQUOT, "quota")):
            with self.assertRaises(OSError):
                self.apply_plan(plan)
        self.assertEqual(before, self.fingerprint())
        self.assertEqual(list(self.root.rglob(".field-link-*")), [])

    def test_nfs_link_success_with_error_response_is_reconciled(self):
        plan = self.verified()
        original_link = os.link

        def completed_then_error(source, target):
            original_link(source, target)
            raise OSError(errno.EIO, "lost NFS response")

        with mock.patch.object(reclaim.os, "link", side_effect=completed_then_error):
            results, _ = self.apply_plan(plan)
        self.assertEqual(len(results), 2)
        self.assertEqual(reclaim.scan(self.root)["pairs"], [])
        self.assertEqual(list(self.root.rglob(".field-link-*")), [])

    def test_nfs_rename_success_with_error_response_is_reconciled(self):
        plan = self.verified()
        original_replace = os.replace

        def completed_then_error(source, target):
            original_replace(source, target)
            raise OSError(errno.EIO, "lost NFS response")

        with mock.patch.object(reclaim.os, "replace", side_effect=completed_then_error):
            results, _ = self.apply_plan(plan)
        self.assertEqual(len(results), 2)
        self.assertEqual(reclaim.scan(self.root)["pairs"], [])

    def test_partial_completion_rescans_and_finishes_only_remaining_duplicates(self):
        plan = self.verified()
        original = reclaim.replace_pair
        count = 0

        def fail_second(pair):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError(errno.EDQUOT, "quota after first replacement")
            return original(pair)

        with mock.patch.object(reclaim, "replace_pair", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.apply_plan(plan)
        resumed = self.verified()
        self.assertEqual(len(resumed["pairs"]), 1)
        results, _ = self.apply_plan(resumed)
        self.assertEqual(len(results), 1)
        self.assertEqual(reclaim.scan(self.root)["pairs"], [])

    def test_live_local_owner_blocks_without_signalling(self):
        plan = self.verified()
        lock = self.root / reclaim.EXPERIMENTS[0] / ".pipeline.lock"
        lock.write_text(json.dumps(dict(host=socket.gethostname(), pid=12345, token="unit")))
        before = self.fingerprint()
        with mock.patch.object(reclaim, "os", wraps=os) as posix_os:
            posix_os.name = "posix"
            posix_os.kill.return_value = None
            with self.assertRaisesRegex(RuntimeError, "inactive experiments"):
                self.apply_plan(plan)
        posix_os.kill.assert_called_once_with(12345, 0)
        self.assertEqual(before, self.fingerprint())

    def test_dead_local_owner_is_reported_inactive_without_deleting_lock(self):
        lock = self.root / reclaim.EXPERIMENTS[0] / ".pipeline.lock"
        content = json.dumps(dict(host=socket.gethostname(), pid=12345, token="unit"))
        lock.write_text(content)
        with mock.patch.object(reclaim, "os", wraps=os) as posix_os:
            posix_os.name = "posix"
            posix_os.kill.side_effect = ProcessLookupError
            self.assertEqual(reclaim.active_owners(self.root), [])
        self.assertEqual(lock.read_text(), content)

    def test_non_posix_owner_probe_never_calls_kill_zero(self):
        lock = self.root / reclaim.EXPERIMENTS[0] / ".pipeline.lock"
        lock.write_text(json.dumps(dict(host=socket.gethostname(), pid=12345, token="unit")))
        with mock.patch.object(reclaim, "os", wraps=os) as windows_os:
            windows_os.name = "nt"
            owners = reclaim.active_owners(self.root)
        windows_os.kill.assert_not_called()
        self.assertEqual(owners[0]["reason"], "non_posix_owner_check_unavailable")

    def test_signed_malformed_field_is_excluded_without_aborting_other_scan(self):
        path = self.cases[0] / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["fields"]["depth"] = []
        metadata.pop("metadata_sha256")
        metadata["metadata_sha256"] = reclaim.digest(metadata)
        path.write_text(json.dumps(metadata))
        plan = reclaim.scan(self.root)
        self.assertEqual(plan["pairs"], [])
        self.assertEqual(plan["metadata_valid_files"], 2)
        self.assertEqual(len(plan["metadata_errors"]), 1)

    def test_remote_owner_never_checks_unrelated_local_pid(self):
        lock = self.root / reclaim.EXPERIMENTS[0] / ".pipeline.lock"
        lock.write_text(json.dumps(dict(host=socket.gethostname() + "-remote", pid=12345, token="unit")))
        with mock.patch.object(reclaim.os, "kill") as exists:
            owners = reclaim.active_owners(self.root)
        exists.assert_not_called()
        self.assertEqual(owners[0]["reason"], "remote_or_unknown_owner")

    def test_persistent_reuse_lock_is_not_mistaken_for_training(self):
        (self.cases[0].parent / ".liver_1.reuse.lock").write_bytes(b"")
        with mock.patch.object(reclaim.os, "kill") as exists:
            self.assertEqual(reclaim.active_owners(self.root), [])
        exists.assert_not_called()

    def test_distinct_bindings_are_never_deduplicated(self):
        metadata_path = self.cases[1] / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["binding"]["spacing"] = [2.0, 1.0, 1.0]
        metadata["binding_sha256"] = reclaim.digest(metadata["binding"])
        metadata.pop("metadata_sha256")
        metadata["metadata_sha256"] = reclaim.digest(metadata)
        metadata_path.write_text(json.dumps(metadata))
        self.assertEqual(reclaim.scan(self.root)["pairs"], [])

    def test_field_symlink_is_excluded_and_never_followed_for_cleanup(self):
        target = self.cases[1] / "depth.npy"
        old_bytes = target.read_bytes()
        target.unlink()
        try:
            target.symlink_to(self.cases[0] / "depth.npy")
        except OSError as error:
            target.write_bytes(old_bytes)
            self.skipTest(f"Symlink creation unavailable in this Windows environment: {error}")
        plan = reclaim.scan(self.root)
        self.assertEqual(plan["pairs"], [])
        self.assertEqual(len(plan["metadata_errors"]), 1)
        self.assertTrue(target.is_symlink())

    def test_without_payload_verification_apply_is_refused(self):
        before = self.fingerprint()
        with self.assertRaisesRegex(ValueError, "payload verification"):
            self.apply_plan(reclaim.scan(self.root))
        self.assertEqual(before, self.fingerprint())

    def test_cancelled_hash_verification_stops_without_changing_any_file(self):
        record = reclaim.scan(self.root)["pairs"][0]["source"]
        cancelled = threading.Event()
        cancelled.set()
        before = self.fingerprint()
        with self.assertRaises((RuntimeError, InterruptedError)):
            reclaim.verify_record(record, cancelled)
        self.assertEqual(before, self.fingerprint())

    def test_another_users_file_ownership_is_refused_before_replacement(self):
        plan = self.verified()
        before = self.fingerprint()
        different_uid = plan["pairs"][0]["source"]["stat"]["uid"] + 1
        with mock.patch.object(reclaim.os, "getuid", create=True, return_value=different_uid):
            with self.assertRaisesRegex(PermissionError, "current user's own"):
                self.apply_plan(plan)
        self.assertEqual(before, self.fingerprint())

    def test_symlink_ancestor_rejection_branch_without_windows_symlink_privilege(self):
        original = Path.is_symlink
        blocked = self.cases[0].parent

        def reported_link(path):
            return path == blocked or original(path)

        before = self.fingerprint()
        with mock.patch.object(Path, "is_symlink", reported_link):
            with self.assertRaisesRegex(ValueError, "Symlink"):
                reclaim.checked_path(self.cases[0] / "depth.npy")
        self.assertEqual(before, self.fingerprint())

    def probe_fixture(self):
        root = self.root / "probe-only" / "experiments"
        root.mkdir(parents=True)
        git = root.parent / "HierCP-regions-8580e59" / ".git" / "objects"
        arm = root / reclaim.EXPERIMENTS[1] / "selected"
        git.mkdir(parents=True)
        arm.mkdir(parents=True)
        return root, git, arm

    def test_write_probes_touch_only_existing_locations_and_remove_own_files(self):
        root, git, arm = self.probe_fixture()
        preserved = arm / ".space-check-unrelated"
        preserved.write_bytes(b"not this invocation")
        before = self.fingerprint()
        results = reclaim.probe_writes(root)
        self.assertEqual([r["status"] for r in results], ["PASS", "PASS", "NOT_FOUND", "NOT_FOUND", "NOT_FOUND"])
        self.assertEqual(before, self.fingerprint())
        self.assertEqual(preserved.read_bytes(), b"not this invocation")
        self.assertFalse((root / reclaim.EXPERIMENTS[2]).exists())

    def test_write_probe_reports_exact_quota_errno_without_creating_directories(self):
        root, git, arm = self.probe_fixture()
        original = Path.open

        def quota_for_probe(path, *args, **kwargs):
            if path.name.startswith(".space-check-"):
                raise OSError(errno.EDQUOT, "Disk quota exceeded", str(path))
            return original(path, *args, **kwargs)

        before = self.fingerprint()
        with mock.patch.object(Path, "open", quota_for_probe):
            results = reclaim.probe_writes(root)
        self.assertEqual([r["status"] for r in results[:2]], ["FAIL", "FAIL"])
        self.assertEqual([r["errno"] for r in results[:2]], [errno.EDQUOT, errno.EDQUOT])
        self.assertEqual(before, self.fingerprint())


if __name__ == "__main__":
    unittest.main()
