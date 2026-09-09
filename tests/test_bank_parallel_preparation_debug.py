"""DEBUG synthetic CPU checks; no patient data, training, or GPU benchmark.

Local canonical geometry/edges are real. The inference integration test stubs
only the already-covered upper hierarchy so it isolates source reuse and the
ordered local graph mapper without creating a population-training fixture.
"""

from __future__ import annotations

import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy import ndimage as ndi

from hiercp import cache
from hiercp.common import CandidateInfo, CasePaths, LoadedCase, SourceTumor, synthetic_array_signature
from hiercp.curriculum import CandidateSpec
from hiercp.local import BuiltLocalGraph, build_local_graph, prepare_local_source
from hiercp.schema import GraphBuildConfig
from tools.online_bank_preparation import BankLocalGraphMapper


def debug_fixture():
    coordinates = np.indices((44, 44, 44))
    image = (coordinates[0] * 2 + coordinates[1] - coordinates[2] - 50).astype(np.float32)
    organ = np.zeros(image.shape, dtype=bool)
    organ[5:39, 5:39, 5:39] = True
    full_mask = np.sum((coordinates - np.asarray([16, 16, 16])[:, None, None, None]) ** 2, axis=0) <= 9
    label = organ.astype(np.uint8)
    label[full_mask] = 2
    case = LoadedCase(
        CasePaths("DEBUG_bank", Path("DEBUG_image"), Path("DEBUG_label")), image, label,
        np.eye(4), np.eye(4), None, None, np.ones(3),
        synthetic_array_signature(image), synthetic_array_signature(label),
    )
    slices = (slice(13, 20),) * 3
    source = SourceTumor(1, full_mask, full_mask[slices], image[slices], slices,
                         (16, 16, 16), (16., 16., 16.), int(full_mask.sum()))
    config = replace(
        GraphBuildConfig(), patch_size=16, context_radius_mm=6.,
        context_shells_mm=(2., 4., 6.), context_inner_radius_mm=1.,
        context_outer_radius_mm=6., adaptive_roi_margin_mm=12.,
    )
    regions = SimpleNamespace(full_organ_mask=organ, organ_depth=ndi.distance_transform_edt(organ).astype(np.float32))
    specs = [CandidateSpec(center, 1, 0, index, 0, 1., 5., 5., 40., 10.)
             for index, center in enumerate(((27, 27, 27), (28, 27, 27), (27, 28, 27), (27, 27, 28)))]
    return case, source, config, regions, specs


class BankParallelPreparationDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_nested_equal(self, left, right):
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_real_canonical_parallel_order_rng_and_shared_source_are_exact(self):
        case, source, config, regions, specs = debug_fixture()
        rng = np.random.default_rng(731)
        initial_rng = copy.deepcopy(rng.bit_generator.state)
        kwargs = dict(full_organ_mask=regions.full_organ_mask, organ_depth=regions.organ_depth,
                      config=config, rng=rng, ct_clip=(-200., 250.))
        prepared = prepare_local_source(case, source, **kwargs)
        snapshot = copy.deepcopy(vars(prepared))
        original_case = (case.image.copy(), case.label.copy())

        def build(spec):
            return build_local_graph(case, source, spec, prepared_source=prepared, **kwargs)

        serial = [build(spec) for spec in specs]
        second_finished = threading.Event()
        completion_order = []
        lock = threading.Lock()

        def inverted_first_wave(spec):
            index = specs.index(spec)
            if index == 0 and not second_finished.wait(timeout=15):
                raise RuntimeError("DEBUG second candidate did not run concurrently")
            result = build(spec)
            with lock:
                completion_order.append(index)
            if index == 1:
                second_finished.set()
            return result

        with tempfile.TemporaryDirectory(prefix="DEBUG_bank_parallel_") as root, redirect_stdout(io.StringIO()):
            mapper = BankLocalGraphMapper(root, workers=2)
            parallel = mapper(inverted_first_wave, specs, case_id=case.paths.case_id, source_component=1)
            self.assertEqual(completion_order[0], 1)
            self.assertEqual(mapper.last_report["completed_candidates"], len(specs))
            self.assertEqual(mapper.last_report["resources"]["configured_tasks"], len(specs))
            self.assertEqual(mapper.last_report["resources"]["applied_worker_counts"], [2, 2])
            for left, right in zip(serial, parallel):
                self.assert_nested_equal(vars(left), vars(right))
                self.assertIs(right.source_patch, prepared.source_patch)
            self.assertEqual(len(list((Path(root) / "preparation_resources").glob("*.summary.json"))), 1)
        self.assert_nested_equal(snapshot, vars(prepared))
        self.assert_nested_equal(initial_rng, rng.bit_generator.state)
        np.testing.assert_array_equal(original_case[0], case.image)
        np.testing.assert_array_equal(original_case[1], case.label)

    def test_inference_reuses_prepared_source_and_keeps_every_payload_tensor(self):
        case, source, config, regions, specs = debug_fixture()
        prepared = prepare_local_source(
            case, source, full_organ_mask=regions.full_organ_mask, organ_depth=regions.organ_depth,
            config=config, rng=np.random.default_rng(99), ct_clip=(-200., 250.),
        )
        candidates = [CandidateInfo(spec.center, source.patch_slices, 1., 5., 5., 40., 10.) for spec in specs]
        kwargs = dict(graph_config=config, liver_label=1, tumor_label=2,
                      ct_clip=(-200., 250.), seed=42, regions=regions)
        bank = SimpleNamespace(fingerprint=lambda: "DEBUG_not_a_trained_bank")
        upper_patient = {"DEBUG_upper_fixture": torch.arange(4)}
        upper_prototype = {"DEBUG_upper_fixture": torch.arange(2)}
        with ExitStack() as stack:
            stack.enter_context(patch.object(cache, "build_generation_specs", return_value=specs))
            stack.enter_context(patch.object(cache, "build_patient_graph", return_value=upper_patient))
            stack.enter_context(patch.object(cache, "build_prototype_graph", return_value=upper_prototype))
            serial, original_specs = cache.build_inference_sample(case, source, candidates, bank, **kwargs)
            stack.enter_context(patch.object(cache, "prepare_local_source", side_effect=AssertionError("source rebuilt")))
            with tempfile.TemporaryDirectory(prefix="DEBUG_bank_inference_") as root, redirect_stdout(io.StringIO()):
                parallel, mapped_specs = cache.build_inference_sample(
                    case, source, candidates, bank, prepared_source=prepared,
                    local_graph_map=BankLocalGraphMapper(root, workers=2), **kwargs,
                )
            self.assertEqual(original_specs, mapped_specs)
            self.assert_nested_equal(serial, parallel)
            with self.assertRaisesRegex(ValueError, "every BuiltLocalGraph"):
                cache.build_inference_sample(case, source, candidates, bank, prepared_source=prepared,
                                             local_graph_map=lambda *args, **kwargs: [], **kwargs)

    @staticmethod
    def bookkeeping_graph(index):
        # DEBUG scheduler fixture only, not a predicted or canonical graph.
        local = {"nodes": {"DEBUG": {"x": torch.tensor([[float(index)]])}},
                 "edges": {("DEBUG", "self", "DEBUG"): torch.tensor([[0], [0]])}}
        return BuiltLocalGraph(None, np.asarray([index]), np.asarray([index]), local, local)

    def test_auto_mapper_executes_all_128_original_tasks_once_and_reports_unique_files(self):
        called = []
        lock = threading.Lock()

        def build(index):
            with lock:
                called.append(index)
            return self.bookkeeping_graph(index)

        with tempfile.TemporaryDirectory(prefix="DEBUG_bank_all_candidates_") as root, redirect_stdout(io.StringIO()):
            mapper = BankLocalGraphMapper(root)
            result = mapper(build, range(128), case_id="DEBUG_128", source_component=7)
            self.assertEqual(sorted(called), list(range(128)))
            self.assertEqual([int(graph.target_patch[0]) for graph in result], list(range(128)))
            first = mapper.last_report
            self.assertEqual(first["resources"]["waves"][0]["tasks"], 1)
            self.assertEqual(first["resources"]["completed_tasks"], 128)
            self.assertEqual(first["resources"]["pending_tasks"], 0)
            self.assertEqual(first["target_nodes"], [1] * 128)
            mapper(build, [0, 1], case_id="DEBUG_128", source_component=7)
            reports = list((Path(root) / "preparation_resources").glob("*.summary.json"))
            self.assertEqual(len(reports), 2)
            self.assertEqual(sorted(json.loads(path.read_text())["completed_candidates"] for path in reports), [2, 128])

    def test_failure_is_raised_and_not_returned_as_partial_success(self):
        def build(index):
            if index == 1:
                raise RuntimeError("DEBUG candidate geometry failed")
            return self.bookkeeping_graph(index)

        with tempfile.TemporaryDirectory(prefix="DEBUG_bank_failure_") as root, redirect_stdout(io.StringIO()):
            mapper = BankLocalGraphMapper(root, workers=2)
            with self.assertRaisesRegex(RuntimeError, "candidate geometry failed"):
                mapper(build, range(4), case_id="DEBUG_failure", source_component=1)
            self.assertIsNone(mapper.last_report)
            self.assertEqual(list((Path(root) / "preparation_resources").glob("*.summary.json")), [])
            reports = list((Path(root) / "preparation_resources").glob("*.resources.*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_tasks"], 1)
            self.assertEqual(report["pending_tasks"], 2)


if __name__ == "__main__":
    unittest.main()
