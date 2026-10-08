"""UNIT cache concurrency/identity fixtures; no CT or neural performance claim."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import multiprocessing
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest

import numpy as np

from hiercp_v1x import preparation_reuse as reuse
from hiercp_v1x import u_bridge_fields as fields


def binding():
    return dict(image_sha256="a" * 64, label_sha256="b" * 64,
                common_sha256="c" * 64, shape=[3, 4, 5],
                spacing=[.7, .7, 2.], fixture="UNIT_not_CT")


def close_mappings():
    for arrays, _ in list(fields._OPENED.values()):
        for array in arrays:
            array._mmap.close()
    fields._OPENED.clear()
    gc.collect()


@contextmanager
def directory():
    with TemporaryDirectory(prefix="UNIT_shared_fields_", dir=Path(__file__).resolve().parents[1]) as path:
        try:
            yield Path(path)
        finally:
            close_mappings()


def process_build(root, index, start, result):
    """Actual process worker for the Linux POSIX-lock regression."""
    root = Path(root)
    roots = [root / "arm_a/data", root / "arm_b/data"]
    try:
        if not start.wait(10):
            raise RuntimeError("UNIT concurrent worker start timed out")
        def depth():
            (root / ("factory_" + str(os.getpid()))).write_text("UNIT factory called")
            time.sleep(.2)
            return np.ones((3, 4, 5), dtype=np.float32)
        with reuse.preparation_reuse(object, list(reversed(roots)) if index else roots):
            arrays = fields.cached_fields(roots[index] / "whole_case_fields", "UNIT_case",
                binding(), depth, lambda: np.full((3, 4, 5), 2, dtype=np.float32), lambda: None)
        result.put((index, arrays[2]["status"], None))
    except Exception as error:
        result.put((index, None, repr(error)))
    finally:
        close_mappings()


class SharedFieldPublicationTests(unittest.TestCase):
    def test_two_cold_arm_destinations_build_once_and_hardlink_identical_arrays(self):
        with directory() as root:
            roots = [root / "arm_a/data", root / "arm_b/data"]
            barrier = threading.Barrier(2)
            calls = []
            events = []
            def factory(name):
                calls.append(name)
                time.sleep(.1)
                return np.full((3, 4, 5), 1 if name == "depth" else 2, dtype=np.float32)
            def obtain(index):
                barrier.wait(timeout=5)
                return fields.cached_fields(roots[index] / "whole_case_fields", "UNIT_case", binding(),
                    lambda: factory("depth"), lambda: factory("occupied"), lambda: None)
            with reuse.preparation_reuse(object, roots, events.append), ThreadPoolExecutor(2) as pool:
                results = list(pool.map(obtain, (0, 1)))
            self.assertEqual(sorted(calls), ["depth", "occupied"])
            self.assertEqual(sorted(result[2]["status"] for result in results), ["built", "reopened"])
            self.assertEqual(sorted(event["status"] for event in events), ["cache_miss", "reused"])
            for name in ("depth.npy", "occupied.npy", "metadata.json"):
                self.assertTrue(os.path.samefile(roots[0] / "whole_case_fields/UNIT_case" / name,
                                                roots[1] / "whole_case_fields/UNIT_case" / name))
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_different_cases_build_concurrently_not_under_a_global_lock(self):
        with directory() as root:
            roots = [root / "arm_a/data", root / "arm_b/data"]
            factories_entered = threading.Barrier(2)
            def factory():
                factories_entered.wait(timeout=5)
                return np.ones((3, 4, 5), dtype=np.float32)
            def obtain(index):
                return fields.cached_fields(roots[index] / "whole_case_fields", "UNIT_case" + str(index),
                    binding(), factory, lambda: np.ones((3, 4, 5), dtype=np.float32), lambda: None)
            with reuse.preparation_reuse(object, roots), ThreadPoolExecutor(2) as pool:
                results = list(pool.map(obtain, (0, 1)))
            self.assertEqual([result[2]["status"] for result in results], ["built", "built"])

    def test_shared_guard_same_identity_ignores_root_order_and_destination(self):
        with directory() as root:
            roots = [root / "arm_a/data", root / "arm_b/data"]
            acquired = []
            @contextmanager
            def record(target):
                acquired.append(target)
                yield
            from unittest.mock import patch
            with patch.object(reuse, "_publication_guard", record):
                with reuse._shared_fields_guard(roots[0] / "whole_case_fields", "UNIT_case", binding(), roots):
                    pass
                with reuse._shared_fields_guard(roots[1] / "whole_case_fields", "UNIT_case", binding(), roots[::-1]):
                    pass
            self.assertEqual(acquired[0], acquired[1])
            self.assertEqual(acquired[0].parent, root / ".field_publication")
            self.assertFalse(roots[0].exists())
            self.assertFalse(roots[1].exists())

    def test_existing_fields_do_not_enter_shared_guard_or_change_bytes(self):
        from unittest.mock import patch
        with directory() as root:
            roots = [root / "arm_a/data", root / "arm_b/data"]
            target = roots[0] / "whole_case_fields"
            fields.cached_fields(target, "UNIT_case", binding(),
                lambda: np.ones((3, 4, 5), dtype=np.float32),
                lambda: np.full((3, 4, 5), 2, dtype=np.float32), lambda: None)
            close_mappings()
            before = {p.name: p.read_bytes() for p in (target / "UNIT_case").iterdir()}
            with patch.object(reuse, "_publication_guard", side_effect=AssertionError("Shared lock on hot publication")):
                with reuse._shared_fields_guard(target, "UNIT_case", binding(), roots):
                    pass
            self.assertEqual(before, {p.name: p.read_bytes() for p in (target / "UNIT_case").iterdir()})

    def test_failed_first_factory_is_not_published_or_reused_by_second_arm(self):
        with directory() as root:
            roots = [root / "arm_a/data", root / "arm_b/data"]
            def failure():
                raise RuntimeError("UNIT factory failed")
            with reuse.preparation_reuse(object, roots):
                with self.assertRaisesRegex(RuntimeError, "UNIT factory failed"):
                    fields.cached_fields(roots[0] / "whole_case_fields", "UNIT_case", binding(),
                        failure, failure, lambda: None)
                result = fields.cached_fields(roots[1] / "whole_case_fields", "UNIT_case", binding(),
                    lambda: np.ones((3, 4, 5), dtype=np.float32),
                    lambda: np.full((3, 4, 5), 2, dtype=np.float32), lambda: None)
            self.assertEqual(result[2]["status"], "built")
            self.assertFalse((roots[0] / "whole_case_fields/UNIT_case").exists())
            self.assertTrue(list((roots[0] / "whole_case_fields").glob(".UNIT_case.attempt.*")))

    def test_different_bindings_do_not_share_wrong_arrays(self):
        with directory() as root:
            roots = [root / "arm_a/data", root / "arm_b/data"]
            def obtain(index):
                signed = dict(binding(), image_sha256=("a" if index == 0 else "d") * 64)
                return fields.cached_fields(roots[index] / "whole_case_fields", "UNIT_case", signed,
                    lambda: np.full((3, 4, 5), index + 1, dtype=np.float32),
                    lambda: np.full((3, 4, 5), index + 1, dtype=np.float32), lambda: None)
            with reuse.preparation_reuse(object, roots):
                results = [obtain(0), obtain(1)]
            self.assertTrue(np.all(results[0][0] == 1))
            self.assertTrue(np.all(results[1][0] == 2))
            self.assertFalse(os.path.samefile(roots[0] / "whole_case_fields/UNIT_case/depth.npy",
                                             roots[1] / "whole_case_fields/UNIT_case/depth.npy"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Actual Linux process locks; Windows thread checks above only")
    def test_two_processes_publish_only_one_field_copy(self):
        with directory() as root:
            ctx = multiprocessing.get_context("spawn")
            start, result = ctx.Event(), ctx.Queue()
            workers = [ctx.Process(target=process_build, args=(str(root), index, start, result)) for index in (0, 1)]
            for worker in workers:
                worker.start()
            start.set()
            rows = [result.get(timeout=20), result.get(timeout=20)]
            for worker in workers:
                worker.join(timeout=20)
                self.assertFalse(worker.is_alive())
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual([row[2] for row in rows], [None, None])
            self.assertEqual(sorted(row[1] for row in rows), ["built", "reopened"])
            self.assertEqual(len(list(root.glob("factory_*"))), 1)
            self.assertTrue(os.path.samefile(root / "arm_a/data/whole_case_fields/UNIT_case/depth.npy",
                                            root / "arm_b/data/whole_case_fields/UNIT_case/depth.npy"))


if __name__ == "__main__":
    unittest.main()
