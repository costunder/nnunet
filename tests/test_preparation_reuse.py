"""UNIT immutable publication fixtures; not CT or neural performance evidence."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
import gc
import json
import os
from pathlib import Path
import socket
import sys
import threading
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import torch

from hiercp_v1x import preparation_reuse as reuse
from hiercp_v1x import u_bridge_fields as fields
from hiercp_v1x import u_bridge_upper as upper


def binding():
    return dict(image_sha256="a" * 64, label_sha256="b" * 64, common_sha256="c" * 64,
                shape=[3, 4, 5], spacing=[.7, .7, 2.], fixture="UNIT_not_CT")


def close_mappings():
    for arrays, _ in list(fields._OPENED.values()):
        for array in arrays:
            array._mmap.close()
    fields._OPENED.clear()
    gc.collect()


@contextmanager
def directory():
    with TemporaryDirectory(prefix="UNIT_preparation_reuse_", dir=Path(__file__).resolve().parents[1]) as path:
        try:
            yield Path(path)
        finally:
            close_mappings()


def whole(source, signed=None):
    signed = binding() if signed is None else signed
    depth = np.arange(60, dtype=np.float32).reshape(3, 4, 5) / 8
    occupied = np.asfortranarray(np.arange(60, dtype=np.float64).reshape(3, 4, 5) / 3)
    fields.cached_fields(source / "whole_case_fields", "UNIT_case", signed,
                         lambda: depth, lambda: occupied, lambda: None)
    return depth, occupied


def inventory(root):
    return {p.relative_to(root).as_posix(): reuse.publication._sha(p)
            for p in root.rglob("*") if p.is_file()}


def own_namespace(root, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    owner = dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT_owned_namespace")
    owner.update(overrides)
    path = root / ".data.lock"
    path.write_text(json.dumps(owner), encoding="utf8")
    return path


@contextmanager
def unsupported_linux_rename(error=errno.EINVAL):
    """UNIT Linux API injection on Windows; not real Linux/NFS evidence."""
    fcntl = SimpleNamespace(LOCK_EX=2, LOCK_UN=8, lockf=Mock())
    with patch.object(reuse.sys, "platform", "linux"), \
            patch.dict(sys.modules, {"fcntl": fcntl}), \
            patch.object(reuse, "_linux_noreplace", side_effect=OSError(error, "UNIT unsupported no-replace")):
        yield fcntl


def upper_binding():
    return dict(case_id="UNIT_case", image_sha256="a" * 64, label_sha256="b" * 64,
        original_core_sha256="c" * 64, region_identity_sha256="d" * 64,
        source_component=1, source_voxels=2, upper_raw_dim=5, shape=[3, 4, 5], anchor=[1, 2, 3],
        spacing=[.7, .7, 2.], ct_clip=[-100., 200.], tumor_label=2, max_lesions=12)


def upper_arrays(kind):
    if kind == "source_raw":
        return np.arange(5, dtype=np.float32)
    return (np.arange(10, dtype=np.float32).reshape(2, 5),
            np.arange(6, dtype=np.float64).reshape(2, 3), np.asarray([1, 2], dtype=np.int64))


def static_upper(source, kind="source_raw", signed=None):
    signed = upper_binding() if signed is None else signed
    cache = upper.UpperCache(source / "upper_static", lambda: None)
    return cache._obtain(kind, signed, lambda: upper_arrays(kind))


class Provider:
    def __init__(self, root):
        self.graph_dir = root / "canonical_local"
        self.graph_dir.mkdir(parents=True)

    def _get(self, key, factory):
        return factory()


class PreparationReuseTests(unittest.TestCase):
    def test_whole_exact_hardlinks_no_factory_and_source_bytes_preserved_while_active(self):
        with directory() as root:
            source, target = root / "active", root / "own"
            expected = whole(source)
            (source / ".data.lock").write_text(json.dumps(dict(host="UNIT_active_host", pid=os.getpid())))
            before = inventory(source); events = []
            def forbid():
                self.fail("A completed field cache must not call its CPU factory")
            with reuse.preparation_reuse(Provider, [source], events.append):
                depth, occupied, receipt = fields.cached_fields(target / "whole_case_fields", "UNIT_case",
                    binding(), forbid, forbid, lambda: None)
            self.assertEqual(receipt["status"], "reopened")
            for name, actual, wanted in zip(("depth", "occupied"), (depth, occupied), expected):
                np.testing.assert_array_equal(actual, wanted)
                self.assertFalse(actual.flags.writeable)
                self.assertTrue(os.path.samefile(source / "whole_case_fields/UNIT_case" / (name + ".npy"),
                                                target / "whole_case_fields/UNIT_case" / (name + ".npy")))
            self.assertEqual(before, inventory(source))
            self.assertEqual([e["status"] for e in events], ["reused"])

    def test_local_completed_cache_loaded_without_graph_compute_or_extra_unpickle(self):
        with directory() as root:
            source = root / "source"; graph = source / "canonical_local"; graph.mkdir(parents=True)
            signed = dict(fixture="UNIT_graph_metadata")
            key = reuse.publication._digest(signed)
            old = graph / (key + ".pt")
            torch.save(dict(binding=signed, built=torch.arange(5)), old)
            before = inventory(source); events = []
            with reuse.preparation_reuse(Provider, [source], events.append) as Reusing:
                provider = Reusing(root / "own")
                path = provider.graph_dir / old.name
                def original_loader():
                    self.assertTrue(path.exists(), "Reuse must precede the original construction factory")
                    payload = torch.load(path, weights_only=False)
                    self.assertEqual(payload["binding"], signed)
                    return payload["built"]
                with patch.object(reuse.publication, "_torch_publication", wraps=reuse.publication._torch_publication) as verify:
                    actual = provider._get(("local", key), original_loader)
                    verify.assert_called_once()
            torch.testing.assert_close(actual, torch.arange(5))
            self.assertTrue(os.path.samefile(old, path))
            self.assertEqual(before, inventory(source))
            self.assertEqual(events[0]["artifact_sha256"], reuse.publication._sha(old))

    def test_interrupted_dot_publication_is_ignored_and_real_factory_runs(self):
        with directory() as root:
            source = root / "source"; attempt = source / "whole_case_fields/.UNIT_case.attempt.UNIT"
            attempt.mkdir(parents=True); (attempt / "depth.npy").write_bytes(b"UNIT incomplete")
            events = []; count = []
            def factory():
                count.append(1); return np.ones((3, 4, 5), dtype=np.float32)
            with reuse.preparation_reuse(Provider, [source], events.append):
                result = fields.cached_fields(root / "own/whole_case_fields", "UNIT_case", binding(),
                                              factory, factory, lambda: None)
            self.assertEqual(len(count), 2)
            self.assertEqual(result[2]["status"], "built")
            self.assertEqual(events[0]["status"], "cache_miss")
            self.assertEqual((attempt / "depth.npy").read_bytes(), b"UNIT incomplete")

    def test_exact_binding_mismatch_emits_miss_and_does_not_reuse_wrong_fields(self):
        with directory() as root:
            source = root / "source"; other = dict(binding(), image_sha256="d" * 64)
            whole(source, other); events = []; count = []
            def factory():
                count.append(1); return np.full((3, 4, 5), 8, dtype=np.float32)
            with reuse.preparation_reuse(Provider, [source], events.append):
                actual = fields.cached_fields(root / "own/whole_case_fields", "UNIT_case", binding(),
                                              factory, factory, lambda: None)
            self.assertEqual(len(count), 2)
            self.assertTrue(np.all(actual[0] == 8))
            self.assertEqual([e["status"] for e in events], ["binding_miss", "cache_miss"])

    def test_exact_binding_payload_tamper_rejected_not_recomputed(self):
        with directory() as root:
            source = root / "source"; whole(source); close_mappings()
            path = source / "whole_case_fields/UNIT_case/depth.npy"
            path.write_bytes(path.read_bytes() + b"UNIT tamper")
            def forbid():
                self.fail("Tamper is an error, not a miss or fallback")
            with reuse.preparation_reuse(Provider, [source]):
                with self.assertRaisesRegex(ValueError, "SHA/size differs"):
                    fields.cached_fields(root / "own/whole_case_fields", "UNIT_case", binding(),
                                         forbid, forbid, lambda: None)
            self.assertFalse((root / "own/whole_case_fields/UNIT_case").exists())

    def test_signed_metadata_tamper_rejected_even_if_binding_changed(self):
        with directory() as root:
            source = root / "source"; whole(source)
            path = source / "whole_case_fields/UNIT_case/metadata.json"
            value = json.loads(path.read_text()); value["binding"]["image_sha256"] = "d" * 64
            path.write_text(json.dumps(value))
            with reuse.preparation_reuse(Provider, [source]):
                with self.assertRaisesRegex(ValueError, "metadata was altered"):
                    fields.cached_fields(root / "own/whole_case_fields", "UNIT_case", binding(),
                        lambda: None, lambda: None, lambda: None)

    def test_existing_arm_publication_never_overwritten(self):
        with directory() as root:
            source, own = root / "source", root / "own"
            whole(source)
            fields.cached_fields(own / "whole_case_fields", "UNIT_case", binding(),
                lambda: np.full((3, 4, 5), 9, dtype=np.float32),
                lambda: np.full((3, 4, 5), 9, dtype=np.float32), lambda: None)
            before = inventory(own)
            with reuse.preparation_reuse(Provider, [source]):
                result = fields.cached_fields(own / "whole_case_fields", "UNIT_case", binding(),
                                             lambda: None, lambda: None, lambda: None)
            self.assertTrue(np.all(result[0] == 9))
            self.assertEqual(before, inventory(own))

    def test_existing_local_bytes_not_replaced(self):
        with directory() as root:
            source = root / "source"; source.mkdir(); key = "a" * 64
            with reuse.preparation_reuse(Provider, [source]) as Reusing:
                provider = Reusing(root / "own")
                path = provider.graph_dir / (key + ".pt"); path.write_bytes(b"UNIT existing own bytes")
                result = provider._get(("local", key), lambda: path.read_bytes())
            self.assertEqual(result, b"UNIT existing own bytes")

    def test_concurrent_field_ensure_one_publication_and_zero_factories(self):
        with directory() as root:
            source = root / "source"; whole(source); events = []
            def task(_):
                return fields.cached_fields(root / "own/whole_case_fields", "UNIT_case", binding(),
                    lambda: self.fail("CPU recompute"), lambda: self.fail("CPU recompute"), lambda: None)
            with reuse.preparation_reuse(Provider, [source], events.append):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(task, range(4)))
            self.assertTrue(all(result[0] is results[0][0] for result in results))
            self.assertEqual(sum(e["status"] == "reused" for e in events), 1)
            self.assertFalse(list((root / "own/whole_case_fields").glob(".*.reuse.*")))

    def test_cross_filesystem_copy_keeps_identical_bytes_and_source(self):
        with directory() as root:
            source = root / "source"; whole(source); before = inventory(source)
            with reuse.preparation_reuse(Provider, [source]), \
                    patch.object(reuse.os, "link", side_effect=OSError(errno.EXDEV, "UNIT different filesystem")):
                actual = fields.cached_fields(root / "own/whole_case_fields", "UNIT_case", binding(),
                    lambda: self.fail("factory"), lambda: self.fail("factory"), lambda: None)
            self.assertEqual(actual[2]["status"], "reopened")
            self.assertEqual(before, inventory(source))
            for name in ("depth", "occupied"):
                old = source / "whole_case_fields/UNIT_case" / (name + ".npy")
                new = root / "own/whole_case_fields/UNIT_case" / old.name
                self.assertEqual(old.read_bytes(), new.read_bytes())
                self.assertFalse(os.path.samefile(old, new))

    def test_context_restores_field_loader_on_error(self):
        original = fields.cached_fields
        original_upper = upper.UpperCache._obtain
        with directory() as root:
            with self.assertRaisesRegex(RuntimeError, "UNIT stop"):
                with reuse.preparation_reuse(Provider, [root]):
                    self.assertIsNot(fields.cached_fields, original)
                    self.assertIsNot(upper.UpperCache._obtain, original_upper)
                    raise RuntimeError("UNIT stop")
        self.assertIs(fields.cached_fields, original)
        self.assertIs(upper.UpperCache._obtain, original_upper)

    def test_symlink_root_rejected_without_publication(self):
        with directory() as root:
            source = root / "source"; source.mkdir()
            with patch.object(reuse.Path, "is_symlink", side_effect=lambda: True):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    with reuse.preparation_reuse(Provider, [source]):
                        self.fail("Invalid source context entered")

    def test_malformed_canonical_atomic_file_rejected_without_factory(self):
        with directory() as root:
            source = root / "source"; graph = source / "canonical_local"; graph.mkdir(parents=True)
            key = "a" * 64; (graph / (key + ".pt")).write_bytes(b"UNIT truncated non-ZIP")
            with reuse.preparation_reuse(Provider, [source]) as Reusing:
                provider = Reusing(root / "own")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    provider._get(("local", key), lambda: self.fail("No recompute on malformed publication"))

    def test_both_static_upper_helpers_reuse_active_exact_publications_without_factories(self):
        with directory() as root:
            source = root / "active"; expected = {kind: static_upper(source, kind) for kind in ("source_raw", "lesions")}
            (source / ".data.lock").write_text(json.dumps(dict(host="UNIT_active_host", pid=os.getpid())))
            before = inventory(source); events = []
            def forbid():
                self.fail("A completed exact helper publication must not recompute its original formula")
            with reuse.preparation_reuse(Provider, [source], events.append):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                for kind in ("source_raw", "lesions"):
                    actual = cache._obtain(kind, upper_binding(), forbid)
                    wanted = expected[kind]
                    for value, original in zip((actual,) if kind == "source_raw" else actual,
                                               (wanted,) if kind == "source_raw" else wanted):
                        np.testing.assert_array_equal(value, original)
                    old = source / "upper_static/UNIT_case/c1_a1_2_3" / kind / "arrays.npz"
                    new = root / "own/upper_static/UNIT_case/c1_a1_2_3" / kind / "arrays.npz"
                    self.assertTrue(os.path.samefile(old, new))
            self.assertEqual(before, inventory(source))
            self.assertEqual(cache.stats["original_helper_builds"], 0)
            self.assertEqual(cache.stats["disk_reopens"], 2)
            self.assertEqual([e["helper_kind"] for e in events], ["source_raw", "lesions"])
            self.assertTrue(all(e["status"] == "reused" for e in events))

    def test_static_upper_binding_miss_preserves_source_and_runs_real_factory(self):
        with directory() as root:
            source = root / "source"; static_upper(source, signed=dict(upper_binding(), image_sha256="e" * 64))
            before = inventory(source); calls = []; events = []
            def factory():
                calls.append(1); return np.full(5, 17, dtype=np.float32)
            with reuse.preparation_reuse(Provider, [source], events.append):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                actual = cache._obtain("source_raw", upper_binding(), factory)
            self.assertEqual(calls, [1]); self.assertTrue(np.all(actual == 17))
            self.assertEqual(before, inventory(source))
            self.assertEqual([e["status"] for e in events], ["binding_miss", "cache_miss"])

    def test_static_upper_payload_corruption_rejected_without_recompute(self):
        with directory() as root:
            source = root / "source"; static_upper(source)
            path = source / "upper_static/UNIT_case/c1_a1_2_3/source_raw/arrays.npz"
            path.write_bytes(path.read_bytes() + b"UNIT corruption")
            with reuse.preparation_reuse(Provider, [source]):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                with self.assertRaisesRegex(ValueError, "SHA/size differs"):
                    cache._obtain("source_raw", upper_binding(), lambda: self.fail("No corruption fallback"))
            self.assertFalse((root / "own/upper_static/UNIT_case/c1_a1_2_3/source_raw").exists())

    def test_static_upper_metadata_tamper_rejected_before_binding_miss(self):
        with directory() as root:
            source = root / "source"; static_upper(source)
            path = source / "upper_static/UNIT_case/c1_a1_2_3/source_raw/metadata.json"
            value = json.loads(path.read_text()); value["binding"]["image_sha256"] = "e" * 64
            path.write_text(json.dumps(value))
            with reuse.preparation_reuse(Provider, [source]):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                with self.assertRaisesRegex(ValueError, "metadata was altered"):
                    cache._obtain("source_raw", upper_binding(), lambda: self.fail("No metadata fallback"))

    def test_static_upper_concurrent_instances_import_once_and_preserve_original_loader(self):
        with directory() as root:
            source = root / "source"; static_upper(source); events = []
            def task(_):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                return cache._obtain("source_raw", upper_binding(), lambda: self.fail("No duplicate CPU helper"))
            with reuse.preparation_reuse(Provider, [source], events.append):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(task, range(4)))
            self.assertEqual(len(events), 1)
            for result in results:
                np.testing.assert_array_equal(result, upper_arrays("source_raw"))
            self.assertFalse(list((root / "own/upper_static/UNIT_case/c1_a1_2_3").glob(".*.reuse.*")))

    def test_static_upper_existing_arm_output_is_never_overwritten(self):
        with directory() as root:
            source = root / "source"; static_upper(source)
            own = root / "own"; cache = upper.UpperCache(own / "upper_static", lambda: None)
            cache._obtain("source_raw", upper_binding(), lambda: np.full(5, 19, dtype=np.float32))
            before = inventory(own)
            with reuse.preparation_reuse(Provider, [source]):
                actual = cache._obtain("source_raw", upper_binding(), lambda: self.fail("Own cache already present"))
            self.assertTrue(np.all(actual == 19)); self.assertEqual(before, inventory(own))

    def test_static_upper_original_numeric_validation_is_not_bypassed(self):
        with directory() as root:
            source = root / "source"; static_upper(source)
            folder = source / "upper_static/UNIT_case/c1_a1_2_3/source_raw"
            payload = folder / "arrays.npz"
            np.savez(payload, a0=np.ones(4, dtype=np.float32))
            path = folder / "metadata.json"; metadata = json.loads(path.read_text())
            metadata["payload_sha256"] = reuse.publication._sha(payload)
            metadata["payload_bytes"] = payload.stat().st_size
            metadata["arrays"] = [dict(shape=[4], dtype=np.dtype("float32").str, bytes=16)]
            metadata.pop("metadata_sha256"); metadata["metadata_sha256"] = reuse.publication._digest(metadata)
            path.write_text(json.dumps(metadata))
            with reuse.preparation_reuse(Provider, [source]):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                with self.assertRaisesRegex(ValueError, "shape/dtype/finite contract differs"):
                    cache._obtain("source_raw", upper_binding(), lambda: self.fail("Must call original numeric validation"))

    def test_static_upper_interrupted_attempt_ignored_and_real_factory_runs(self):
        with directory() as root:
            source = root / "source"; attempt = source / "upper_static/UNIT_case/c1_a1_2_3/.source_raw.attempt.UNIT"
            attempt.mkdir(parents=True); (attempt / "arrays.npz").write_bytes(b"UNIT incomplete publication")
            before = inventory(source); calls = []
            def factory():
                calls.append(1); return upper_arrays("source_raw")
            with reuse.preparation_reuse(Provider, [source]):
                cache = upper.UpperCache(root / "own/upper_static", lambda: None)
                actual = cache._obtain("source_raw", upper_binding(), factory)
            self.assertEqual(calls, [1]); self.assertEqual(before, inventory(source))
            np.testing.assert_array_equal(actual, upper_arrays("source_raw"))

    def test_Windows_normal_and_extended_paths_share_publication_lock(self):
        with directory() as root:
            ordinary = root / "own/upper_static/UNIT_case/c1_a1_2_3/source_raw"
            extended = "\\\\?\\" + str(ordinary)
            with patch.object(reuse.sys, "platform", "win32"):
                self.assertEqual(reuse._path_identity(ordinary), reuse._path_identity(extended))
                self.assertIs(reuse._lock(ordinary), reuse._lock(extended))

    def test_Linux_old_glibc_calls_identical_noreplace_syscall_on_known_ABIs(self):
        for architecture, number in (("x86_64", 316), ("aarch64", 276)):
            with self.subTest(architecture=architecture):
                syscall = Mock(return_value=0)
                library = SimpleNamespace(renameat2=None, syscall=syscall)
                with patch.object(reuse.sys, "platform", "linux"), \
                        patch.object(reuse.platform, "machine", return_value=architecture), \
                        patch.object(reuse.ctypes, "CDLL", return_value=library), \
                        patch.object(reuse.os, "rename", side_effect=AssertionError("Never weaken no-replace")):
                    reuse._rename_new(Path("UNIT_source"), Path("UNIT_target"))
                args = syscall.call_args.args
                self.assertEqual(args[0].value, number)
                self.assertEqual(args[1].value, -100)
                self.assertEqual(args[3].value, -100)
                self.assertEqual(args[5].value, 1)  # RENAME_NOREPLACE, not ordinary rename.

    def test_Linux_old_glibc_unknown_ABI_and_existing_target_preserve_errors(self):
        syscall = Mock(return_value=-1); library = SimpleNamespace(renameat2=None, syscall=syscall)
        with patch.object(reuse.sys, "platform", "linux"), \
                patch.object(reuse.ctypes, "CDLL", return_value=library), \
                patch.object(reuse.platform, "machine", return_value="UNIT_unknown"):
            with self.assertRaisesRegex(OSError, "ABI unavailable"):
                reuse._linux_noreplace(Path("UNIT_source"), Path("UNIT_target"))
            syscall.assert_not_called()
        with patch.object(reuse.sys, "platform", "linux"), \
                patch.object(reuse.ctypes, "CDLL", return_value=library), \
                patch.object(reuse.platform, "machine", return_value="x86_64"), \
                patch.object(reuse.ctypes, "get_errno", return_value=errno.EEXIST):
            with self.assertRaises(FileExistsError):
                reuse._rename_new(Path("UNIT_source"), Path("UNIT_target"))

    def test_unsupported_directory_rename_publishes_only_complete_owned_stage_for_each_errno(self):
        for error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            with self.subTest(errno=error), directory() as root:
                own_namespace(root)
                stage, target = root / ".UNIT_stage", root / "UNIT_complete"
                stage.mkdir(); (stage / "metadata.json").write_bytes(b"UNIT completed metadata")
                (stage / "payload.bin").write_bytes(b"UNIT completed payload")
                original = inventory(stage); rename = os.rename
                def checked(source, destination):
                    self.assertFalse(target.exists())
                    self.assertEqual(inventory(stage), original, "Complete staging must precede publication")
                    return rename(source, destination)
                with unsupported_linux_rename(error) as guard, patch.object(reuse.os, "rename", side_effect=checked):
                    method = reuse._rename_new(stage, target)
                self.assertEqual(method, "owned_namespace_posix_rename")
                self.assertEqual(inventory(target), original)
                self.assertFalse(stage.exists())
                self.assertEqual([call.args[1] for call in guard.lockf.call_args_list], [guard.LOCK_EX, guard.LOCK_UN])
                self.assertTrue((root / ".UNIT_complete.reuse.lock").is_file(), "Persistent lock inode must not be unlinked")

    def test_unsupported_directory_without_owned_namespace_or_wrong_owner_is_refused(self):
        for difference in ("missing", "host", "pid", "token"):
            with self.subTest(difference=difference), directory() as root:
                stage, target = root / ".UNIT_stage", root / "UNIT_complete"
                stage.mkdir(); (stage / "payload").write_bytes(b"UNIT preserve stage")
                if difference != "missing":
                    changes = {"host": "UNIT_other_host"} if difference == "host" else (
                        {"pid": os.getpid() + 1} if difference == "pid" else {"token": ""})
                    own_namespace(root, **changes)
                with unsupported_linux_rename(), patch.object(reuse.os, "rename") as rename:
                    with self.assertRaisesRegex(RuntimeError, "namespace lock"):
                        reuse._rename_new(stage, target)
                rename.assert_not_called()
                self.assertEqual((stage / "payload").read_bytes(), b"UNIT preserve stage")
                self.assertFalse(target.exists())

    def test_fallback_preserves_empty_directory_file_and_lexical_dangling_targets(self):
        for existing in ("empty_directory", "file", "symlink"):
            with self.subTest(existing=existing), directory() as root:
                own_namespace(root)
                stage, target = root / ".UNIT_stage", root / "UNIT_complete"
                stage.mkdir(); (stage / "payload").write_bytes(b"UNIT preserve stage")
                if existing == "empty_directory": target.mkdir()
                elif existing == "file": target.write_bytes(b"UNIT existing target")
                original_lexists = os.path.lexists
                def lexical_exists(path):
                    # UNIT injection for a dangling symlink endpoint. Windows
                    # link creation needs privileges that this test never asks for.
                    return (existing == "symlink" and Path(path) == target) or original_lexists(path)
                with unsupported_linux_rename(), patch.object(reuse.os.path, "lexists", side_effect=lexical_exists), \
                        patch.object(reuse.os, "rename") as rename:
                    with self.assertRaises(FileExistsError):
                        reuse._rename_new(stage, target)
                rename.assert_not_called()
                self.assertEqual((stage / "payload").read_bytes(), b"UNIT preserve stage")
                if existing == "empty_directory": self.assertEqual(list(target.iterdir()), [])
                elif existing == "file": self.assertEqual(target.read_bytes(), b"UNIT existing target")
                else: self.assertFalse(target.exists(), "Injected lexical endpoint must remain unpublished")

    def test_destination_created_after_native_rejection_is_preserved_before_fallback_rename(self):
        with directory() as root:
            own_namespace(root)
            stage, target = root / ".UNIT_stage", root / "UNIT_complete"
            stage.mkdir(); (stage / "payload").write_bytes(b"UNIT preserve stage")
            def collision(*unused):
                target.mkdir()
                raise OSError(errno.EINVAL, "UNIT unsupported filesystem with prior racing destination")
            with unsupported_linux_rename(), patch.object(reuse, "_linux_noreplace", side_effect=collision), \
                    patch.object(reuse.os, "rename") as rename:
                with self.assertRaises(FileExistsError): reuse._rename_new(stage, target)
            rename.assert_not_called(); self.assertEqual(list(target.iterdir()), [])
            self.assertEqual((stage / "payload").read_bytes(), b"UNIT preserve stage")

    def test_changed_namespace_owner_before_fallback_is_rejected(self):
        with directory() as root:
            owner = own_namespace(root)
            stage, target = root / ".UNIT_stage", root / "UNIT_complete"
            stage.mkdir(); (stage / "payload").write_bytes(b"UNIT preserve stage")
            with unsupported_linux_rename(), reuse._publication_guard(target), patch.object(reuse.os, "rename") as rename:
                own_namespace(root, token="UNIT_replaced_owner")
                with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                    reuse._rename_new(stage, target)
            rename.assert_not_called(); self.assertFalse(target.exists())
            self.assertIn("UNIT_replaced_owner", owner.read_text())

    def test_nested_guard_uses_one_descriptor_and_releases_only_outer_record_lock(self):
        with directory() as root:
            own_namespace(root); target = root / "UNIT_complete"
            with unsupported_linux_rename() as guard:
                with reuse._publication_guard(target) as outer:
                    with reuse._publication_guard(target) as inner:
                        self.assertIs(inner, outer)
                        self.assertEqual(guard.lockf.call_count, 1)
                    self.assertEqual(guard.lockf.call_count, 1)
                self.assertEqual(guard.lockf.call_count, 2)
            self.assertEqual(guard.lockf.call_args_list[0].args[0], guard.lockf.call_args_list[1].args[0])

    def test_fcntl_failure_is_not_downgraded_to_unlocked_rename(self):
        with directory() as root:
            own_namespace(root)
            stage, target = root / ".UNIT_stage", root / "UNIT_complete"
            stage.mkdir(); (stage / "payload").write_bytes(b"UNIT preserve stage")
            with unsupported_linux_rename() as guard, patch.object(reuse.os, "rename") as rename:
                guard.lockf.side_effect = OSError(errno.ENOLCK, "UNIT NFS record locking unavailable")
                with self.assertRaisesRegex(OSError, "locking unavailable"):
                    reuse._rename_new(stage, target)
            rename.assert_not_called(); self.assertFalse(target.exists())

    def test_coordinated_directory_race_keeps_winner_and_loser_complete_stage(self):
        with directory() as root:
            own_namespace(root); target = root / "UNIT_complete"
            stages = [root / (".UNIT_stage_" + str(i)) for i in range(2)]
            for index, stage in enumerate(stages):
                stage.mkdir(); (stage / "payload").write_text("UNIT winner " + str(index))
            ready = threading.Barrier(2)
            def unsupported(*unused):
                ready.wait(timeout=10)
                raise OSError(errno.EINVAL, "UNIT unsupported filesystem")
            def publish(stage):
                try: return reuse._rename_new(stage, target)
                except FileExistsError: return "collision_preserved"
            with unsupported_linux_rename(), patch.object(reuse, "_linux_noreplace", side_effect=unsupported):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(publish, stages))
            self.assertEqual(sorted(results), ["collision_preserved", "owned_namespace_posix_rename"])
            loser = next(stage for stage in stages if stage.exists())
            self.assertNotEqual((target / "payload").read_bytes(), (loser / "payload").read_bytes())
            self.assertEqual(len(list(target.iterdir())), 1)

    def test_copied_file_unsupported_rename_uses_exclusive_link_and_preserves_collision(self):
        for error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            with self.subTest(errno=error), directory() as root:
                stage, target = root / ".UNIT_file_stage", root / "UNIT_file_complete"
                stage.write_bytes(b"UNIT copied bytes")
                with unsupported_linux_rename(error), patch.object(reuse.os, "rename") as rename:
                    self.assertEqual(reuse._rename_new(stage, target), "exclusive_link")
                rename.assert_not_called(); self.assertFalse(stage.exists())
                self.assertEqual(target.read_bytes(), b"UNIT copied bytes")
                stage.write_bytes(b"UNIT different later stage")
                with unsupported_linux_rename(error), patch.object(reuse.os, "rename") as rename:
                    with self.assertRaises(FileExistsError): reuse._rename_new(stage, target)
                rename.assert_not_called(); self.assertEqual(target.read_bytes(), b"UNIT copied bytes")
                self.assertEqual(stage.read_bytes(), b"UNIT different later stage")

    def test_unrelated_native_errors_never_trigger_guarded_fallback(self):
        for error in (errno.EACCES, errno.EXDEV, errno.ENOSPC):
            with self.subTest(errno=error), unsupported_linux_rename(error), \
                    patch.object(reuse, "_guarded_posix_publication") as fallback:
                with self.assertRaises(OSError) as caught:
                    reuse._rename_new(Path("UNIT_stage"), Path("UNIT_complete"))
                self.assertEqual(caught.exception.errno, error); fallback.assert_not_called()

    def test_fields_and_both_upper_publications_copy_and_validate_under_forced_EINVAL(self):
        with directory() as root:
            source, target = root / "source", root / "own"
            expected = whole(source)
            for kind in ("source_raw", "lesions"): static_upper(source, kind)
            before = inventory(source); own_namespace(target); events = []
            original_link = os.link
            def source_cross_filesystem(old, new):
                if Path(old).is_relative_to(source):
                    raise OSError(errno.EXDEV, "UNIT source is a different filesystem")
                return original_link(old, new)
            with unsupported_linux_rename() as guard, reuse.preparation_reuse(Provider, [source], events.append), \
                    patch.object(reuse.os, "link", side_effect=source_cross_filesystem):
                actual = fields.cached_fields(target / "whole_case_fields", "UNIT_case", binding(),
                    lambda: self.fail("No field recomputation"), lambda: self.fail("No field recomputation"), lambda: None)
                cache = upper.UpperCache(target / "upper_static", lambda: None)
                for kind in ("source_raw", "lesions"):
                    cache._obtain(kind, upper_binding(), lambda: self.fail("No helper recomputation"))
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(before, inventory(source))
            self.assertEqual([event["kind"] for event in events], ["whole_case_fields", "upper_static", "upper_static"])
            self.assertTrue(all(event["publication_method"] == "owned_namespace_posix_rename" for event in events))
            self.assertTrue(all(method == "copy" for event in events for method in event["methods"]))
            for folder in (target / "whole_case_fields/UNIT_case",
                    target / "upper_static/UNIT_case/c1_a1_2_3/source_raw",
                    target / "upper_static/UNIT_case/c1_a1_2_3/lesions"):
                self.assertFalse(any(path.name.startswith(".") for path in folder.iterdir()), "Private file stages must not enter signed publication inventory")
            self.assertEqual(guard.lockf.call_count, 6, "One acquired/released record guard per whole publication; nested rename reuses descriptors")

    def test_guards_cover_original_field_upper_and_local_factories_on_cache_miss(self):
        with directory() as root:
            source, target = root / "source", root / "own"
            source.mkdir(); own_namespace(target)
            def protected(factory, path):
                def run():
                    self.assertIn(reuse._path_identity(path), reuse._GUARDS.active)
                    return factory()
                return run
            with unsupported_linux_rename(), reuse.preparation_reuse(Provider, [source]) as Reusing:
                field_path = target / "whole_case_fields/UNIT_case"
                array = lambda: np.ones((3, 4, 5), dtype=np.float32)
                fields.cached_fields(field_path.parent, field_path.name, binding(),
                    protected(array, field_path), protected(array, field_path), lambda: None)
                cache = upper.UpperCache(target / "upper_static", lambda: None)
                upper_path = target / "upper_static/UNIT_case/c1_a1_2_3/source_raw"
                cache._obtain("source_raw", upper_binding(), protected(lambda: upper_arrays("source_raw"), upper_path))
                provider = Reusing(target)
                key = "f" * 64
                result = provider._get(("local", key), protected(lambda: "UNIT original load", provider.graph_dir / (key + ".pt")))
                self.assertEqual(result, "UNIT original load")

    def test_immutable_hot_fields_upper_and_local_skip_record_lock_and_namespace_IO(self):
        with directory() as root:
            source, target = root / "source", root / "own"
            source.mkdir(); whole(target); static_upper(target)
            close_mappings(); own_namespace(target)
            with unsupported_linux_rename() as guard, reuse.preparation_reuse(Provider, [source]) as Reusing, \
                    patch.object(reuse, "_namespace_owner", side_effect=AssertionError("No hot-cache namespace IO")):
                guard.lockf.side_effect = AssertionError("No hot-cache record locking")
                provider = Reusing(target); key = "c" * 64
                path = provider.graph_dir / (key + ".pt")
                torch.save(dict(fixture="UNIT already published"), path)
                before = inventory(target)
                fields.cached_fields(target / "whole_case_fields", "UNIT_case", binding(),
                    lambda: self.fail("No hot field recompute"), lambda: self.fail("No hot field recompute"), lambda: None)
                cache = upper.UpperCache(target / "upper_static", lambda: None)
                cache._obtain("source_raw", upper_binding(), lambda: self.fail("No hot helper recompute"))
                result = provider._get(("local", key), lambda: torch.load(path, weights_only=False))
                self.assertEqual(result["fixture"], "UNIT already published")
                self.assertEqual(before, inventory(target))
            guard.lockf.assert_not_called()

    def test_preexisting_incomplete_directory_fast_path_keeps_original_rejection(self):
        with directory() as root:
            source, target = root / "source", root / "own"
            source.mkdir(); own_namespace(target)
            published = target / "whole_case_fields/UNIT_case"; published.mkdir(parents=True)
            before = inventory(target)
            with unsupported_linux_rename() as guard, reuse.preparation_reuse(Provider, [source]):
                guard.lockf.side_effect = AssertionError("No existing immutable-entry lock IO")
                with self.assertRaisesRegex(ValueError, "Incomplete whole-case field publication"):
                    fields.cached_fields(published.parent, published.name, binding(),
                        lambda: self.fail("Incomplete target must not be overwritten"), lambda: None, lambda: None)
            guard.lockf.assert_not_called(); self.assertEqual(before, inventory(target))
            self.assertEqual(list(published.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
