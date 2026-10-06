"""UNIT exact original-helper memoization tests; no CT or neural execution."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x.u_bridge_upper import UpperCache


@contextmanager
def fixture():
    with TemporaryDirectory(prefix="UNIT_upper_", dir=Path(__file__).resolve().parents[1] / "work") as root:
        case = SimpleNamespace(paths=SimpleNamespace(case_id="UNIT_case"), shape=(2, 3, 4),
                               spacing=np.asarray([.7, .7, 2.], dtype=np.float32))
        source = SimpleNamespace(component_id=2, anchor_center=(1, 1, 2), voxel_count=8)
        regions = SimpleNamespace(region_id="exact UNIT regions")
        binding = dict(case_id="UNIT_case", image_sha256="a" * 64, label_sha256="b" * 64,
            original_core_sha256="c" * 64, region_identity_sha256="d" * 64,
            source_component=2, source_voxels=8, shape=list(case.shape),
            anchor=list(source.anchor_center), spacing=list(map(float, case.spacing)),
            ct_clip=[-200., 250.], tumor_label=2, max_lesions=None, upper_raw_dim=4)
        originals = (np.asarray([1., 2., 3., 4.], np.float32),
                     (np.arange(8, dtype=np.float32).reshape(2, 4),
                      np.arange(6, dtype=np.float32).reshape(2, 3), np.asarray([3, 5], np.int64)))
        counts = dict(source=0, lesions=0)
        def original_source(*args, **kwargs):
            counts["source"] += 1
            return originals[0]
        def original_lesions(*args, **kwargs):
            counts["lesions"] += 1
            return originals[1]
        module = SimpleNamespace(UPPER_RAW_DIM=4, _source_raw=original_source, _lesions=original_lesions)
        yield Path(root), case, source, regions, binding, originals, counts, module


def invoke(module, case, source, regions, binding):
    return (module._source_raw(case, source, regions, np.zeros(case.shape, dtype=bool),
                               ct_clip=tuple(binding["ct_clip"])),
            module._lesions(case, source, regions, tumor_label=2,
                            max_lesions=binding["max_lesions"], ct_clip=tuple(binding["ct_clip"])))


class UpperTests(unittest.TestCase):
    def test_original_runs_once_exact_arrays_and_defensive_copies(self):
        with fixture() as (root, case, source, regions, binding, originals, counts, module):
            cache = UpperCache(root, lambda: None)
            with cache.original_helpers(module, case, source, regions, binding):
                actual = invoke(module, case, source, regions, binding)
                for a, b in zip((actual[0], *actual[1]), (originals[0], *originals[1])):
                    self.assertEqual(a.dtype, b.dtype); np.testing.assert_array_equal(a, b)
                    a.flat[0] = 900
                again = invoke(module, case, source, regions, binding)
                for a, b in zip((again[0], *again[1]), (originals[0], *originals[1])):
                    np.testing.assert_array_equal(a, b)
            self.assertEqual(counts, dict(source=1, lesions=1))
            report = cache.report()
            self.assertEqual(report["original_helper_builds"], 2)
            self.assertEqual(report["resident_hits"], 2)
            self.assertEqual(report["entries"], 2)
            self.assertGreater(report["disk_bytes"], report["resident_bytes"])

    def test_cold_disk_reopen_runs_no_original_helpers(self):
        with fixture() as (root, case, source, regions, binding, originals, counts, module):
            first = UpperCache(root, lambda: None)
            with first.original_helpers(module, case, source, regions, binding):
                invoke(module, case, source, regions, binding)
            before = counts.copy(); second = UpperCache(root, lambda: None)
            with second.original_helpers(module, case, source, regions, binding):
                actual = invoke(module, case, source, regions, binding)
            self.assertEqual(counts, before)
            self.assertEqual(second.report()["disk_reopens"], 2)
            for a, b in zip((actual[0], *actual[1]), (originals[0], *originals[1])):
                self.assertEqual(a.tobytes(), b.tobytes())

    def test_payload_or_metadata_corruption_rejected_without_original_call(self):
        for kind in ("payload", "metadata"):
            with self.subTest(kind=kind), fixture() as (root, case, source, regions, binding, originals, counts, module):
                cache = UpperCache(root, lambda: None)
                with cache.original_helpers(module, case, source, regions, binding):
                    invoke(module, case, source, regions, binding)
                directory = root / "UNIT_case/c2_a1_1_2/source_raw"
                if kind == "payload":
                    with (directory / "arrays.npz").open("r+b") as stream:
                        stream.seek(-1, 2); stream.write(b"X")
                else:
                    path = directory / "metadata.json"
                    saved = json.loads(path.read_text()); saved["binding"]["ct_clip"] = [-100., 300.]
                    path.write_text(json.dumps(saved))
                before = counts.copy()
                with self.assertRaisesRegex(ValueError, "SHA256 differs|binding/metadata differs"):
                    with UpperCache(root, lambda: None).original_helpers(module, case, source, regions, binding):
                        invoke(module, case, source, regions, binding)
                self.assertEqual(counts, before)

    def test_changed_verified_binding_rejected_without_recompute(self):
        with fixture() as (root, case, source, regions, binding, originals, counts, module):
            with UpperCache(root, lambda: None).original_helpers(module, case, source, regions, binding):
                invoke(module, case, source, regions, binding)
            changed = copy.deepcopy(binding); changed["region_identity_sha256"] = "e" * 64
            before = counts.copy()
            with self.assertRaisesRegex(ValueError, "binding/metadata differs"):
                with UpperCache(root, lambda: None).original_helpers(module, case, source, regions, changed):
                    invoke(module, case, source, regions, changed)
            self.assertEqual(counts, before)

    def test_restore_originals_after_body_exception_and_wrong_source(self):
        with fixture() as (root, case, source, regions, binding, originals, counts, module):
            functions = module._source_raw, module._lesions
            cache = UpperCache(root, lambda: None)
            with self.assertRaisesRegex(RuntimeError, "UNIT interrupted"):
                with cache.original_helpers(module, case, source, regions, binding):
                    raise RuntimeError("UNIT interrupted")
            self.assertEqual((module._source_raw, module._lesions), functions)
            with self.assertRaisesRegex(ValueError, "another verified CT/source/region"):
                with cache.original_helpers(module, case, source, regions, binding):
                    module._source_raw(case, copy.copy(source), regions, np.zeros(case.shape, bool),
                                       ct_clip=tuple(binding["ct_clip"]))
            self.assertEqual((module._source_raw, module._lesions), functions)
            self.assertEqual(counts, dict(source=0, lesions=0))

    def test_original_factory_failure_restores_and_publishes_no_complete_output(self):
        with fixture() as (root, case, source, regions, binding, originals, counts, module):
            def fail(*args, **kwargs):
                raise RuntimeError("UNIT original lesion guard")
            module._lesions = fail; functions = module._source_raw, module._lesions
            cache = UpperCache(root, lambda: None)
            with self.assertRaisesRegex(RuntimeError, "UNIT original lesion guard"):
                with cache.original_helpers(module, case, source, regions, binding):
                    invoke(module, case, source, regions, binding)
            self.assertEqual((module._source_raw, module._lesions), functions)
            self.assertFalse((root / "UNIT_case/c2_a1_1_2/lesions").exists())
            self.assertEqual(cache.report()["original_helper_builds"], 1)


if __name__ == "__main__":
    unittest.main()
