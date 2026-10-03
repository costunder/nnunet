"""Call-boundary contracts for the native ROI diagnostic; no neural run."""
import unittest
from types import SimpleNamespace

from hiercp_v1x.roi_failure_replay import capture_failed_roi


class RoiFailureReplayTests(unittest.TestCase):
    def modules(self, *, operation="payload", phase="target", message="expected",
                final_error=None, complete=False):
        class RoiError(ValueError):
            pass
        spatial = SimpleNamespace(AdaptiveRoiBudgetError=RoiError)
        calls = []
        raw_payload, raw_footprint, forward = object(), object(), object()
        config = SimpleNamespace(adaptive_roi_max_voxels=8_000_000)
        source = SimpleNamespace(component_id=7, anchor_center=(5, 6, 7))
        spec = SimpleNamespace(center=(20, 30, 40), difficulty=3, corruption=1,
            scale_xyz=(1.5, 1.5, 1.5), rotation=(1., 0., 0., 0., 1., 0., 0., 0., 1.),
            region_id=2, prototype_id=4)
        case, organ, depth, prepared = object(), object(), object(), object()
        local = SimpleNamespace()

        def payload(**kwargs):
            calls.append(("payload", kwargs))
            if phase == "source" or operation == "payload":
                if final_error is not None:
                    raise final_error
                if not complete:
                    raise RoiError(message)
            return raw_payload

        def transform(footprint, forward_mm, spacing, config):
            calls.append(("transform", footprint, forward_mm, spacing, config))
            if operation == "transform" and not complete:
                raise RoiError(message)
            return footprint

        def target(case, spec, *, full_organ_mask, organ_depth, config,
                   ct_clip, prepared_source):
            calls.append(("target", case, spec, prepared_source))
            footprint = local.transform_footprint_physical(raw_footprint, forward,
                                                          (1., 1., 1.), config)
            return local.build_patch_payload(image=case, center=spec.center,
                footprint=footprint, full_organ=full_organ_mask,
                organ_depth=organ_depth, config=config, erase_target=True,
                spacing=(1., 1., 1.), ct_clip=ct_clip)

        local.build_patch_payload = payload
        local.transform_footprint_physical = transform
        local._prepare_local_target = target
        cache = SimpleNamespace()

        def prepare_source(case, source, *, config, token):
            calls.append(("source", case, source, config, token))
            if phase == "source":
                local.build_patch_payload(image=case, center=source.anchor_center,
                    footprint=raw_footprint, config=config, erase_target=False)
            return prepared

        cache.prepare_local_source = prepare_source
        token = object()
        sample_kwargs = dict(case=case, bank=object(), regions=object(),
                             graph_config=config, sample_index=1, seed=42, token=token)

        def sample(**kwargs):
            calls.append(("sample", kwargs))
            actual = cache.prepare_local_source(kwargs["case"], source,
                config=kwargs["graph_config"], token=kwargs["token"])
            if phase == "target":
                local._prepare_local_target(case, spec, full_organ_mask=organ,
                    organ_depth=depth, config=config, ct_clip=(-200., 250.),
                    prepared_source=actual)
            calls.append(("sample_completed",))
            return raw_payload

        cache.build_training_sample = sample
        originals = (cache.prepare_local_source, local._prepare_local_target,
                     local.build_patch_payload, local.transform_footprint_physical)
        return SimpleNamespace(cache=cache, local=local, spatial=spatial, calls=calls,
            sample_kwargs=sample_kwargs, originals=originals, footprint=raw_footprint,
            forward=forward, prepared=prepared, spec=spec,
            expected={"voxel_budget": 8_000_000, "tag": "expected"})

    def capture(self, fixture, expected=None):
        return capture_failed_roi(fixture.cache, fixture.local, fixture.spatial,
            sample_kwargs=fixture.sample_kwargs,
            expected_geometry=fixture.expected if expected is None else expected,
            parse_failure=lambda text: {"voxel_budget": 8_000_000, "tag": text})

    def assert_restored(self, fixture):
        self.assertEqual(fixture.originals,
            (fixture.cache.prepare_local_source, fixture.local._prepare_local_target,
             fixture.local.build_patch_payload, fixture.local.transform_footprint_physical))

    def test_exact_source_payload_capture(self):
        fixture = self.modules(phase="source")
        result = self.capture(fixture)
        self.assertEqual(result["phase"], "source")
        self.assertEqual(result["operation"], "build_patch_payload")
        self.assertFalse(result["payload_kwargs"]["erase_target"])
        self.assertEqual(result["source_component"], 7)
        self.assertEqual(result["source_anchor"], [5, 6, 7])
        self.assertIsNone(result["target_spec"])
        self.assert_restored(fixture)

    def test_exact_target_payload_and_curriculum_capture(self):
        fixture = self.modules()
        result = self.capture(fixture)
        self.assertEqual(result["phase"], "target")
        self.assertEqual(result["target_spec"]["center"], [20, 30, 40])
        self.assertEqual(result["target_spec"]["corruption"], 1)
        self.assertEqual(result["target_spec"]["scale_xyz"], [1.5, 1.5, 1.5])
        self.assertTrue(result["payload_kwargs"]["erase_target"])
        self.assertIs(result["payload_kwargs"]["footprint"], fixture.footprint)
        self.assertIs(result["target_context"]["spec"], fixture.spec)
        self.assertNotIn("prepared_source", result["target_context"]["kwargs"])
        self.assertNotIn("exception", result)
        self.assert_restored(fixture)

    def test_transform_preallocation_capture(self):
        fixture = self.modules(operation="transform")
        result = self.capture(fixture)
        self.assertEqual(result["operation"], "transform_footprint_physical")
        self.assertIs(result["transform_args"][0], fixture.footprint)
        self.assertIs(result["transform_args"][1], fixture.forward)
        self.assertIsNone(result["payload_kwargs"])
        self.assertEqual([x[0] for x in fixture.calls], ["sample", "source", "target", "transform"])
        self.assert_restored(fixture)

    def test_original_first_failure_must_match_manifest(self):
        fixture = self.modules(message="different")
        with self.assertRaisesRegex(ValueError, "differs from the failed manifest"):
            self.capture(fixture)
        self.assertNotIn("sample_completed", [x[0] for x in fixture.calls])
        self.assert_restored(fixture)

    def test_other_original_exception_propagates_unchanged(self):
        error = RuntimeError("original coordinate error")
        fixture = self.modules(final_error=error)
        with self.assertRaises(RuntimeError) as caught:
            self.capture(fixture)
        self.assertIs(caught.exception, error)
        self.assert_restored(fixture)

    def test_completed_sample_is_a_diagnostic_mismatch(self):
        fixture = self.modules(complete=True)
        with self.assertRaisesRegex(ValueError, "completed without reproducing"):
            self.capture(fixture)
        self.assertEqual(sum(c[0] == "sample_completed" for c in fixture.calls), 1)
        self.assert_restored(fixture)

    def test_complete_original_call_arguments_preserved(self):
        fixture = self.modules()
        self.capture(fixture)
        self.assertEqual(sum(c[0] == "sample" for c in fixture.calls), 1)
        received = fixture.calls[0][1]
        self.assertEqual(set(received), set(fixture.sample_kwargs))
        for key in received:
            self.assertIs(received[key], fixture.sample_kwargs[key])
        source = next(c for c in fixture.calls if c[0] == "source")
        self.assertIs(source[3], fixture.sample_kwargs["graph_config"])
        self.assert_restored(fixture)

    def test_original_guard_cannot_be_replaced_during_replay(self):
        fixture = self.modules()
        fixture.sample_kwargs["graph_config"].adaptive_roi_max_voxels = 12_000_000
        with self.assertRaisesRegex(ValueError, "exact failed original ROI budget"):
            self.capture(fixture)
        self.assertEqual(fixture.calls, [])
        self.assert_restored(fixture)

    def test_uncaptured_guard_cannot_be_claimed_reproduced(self):
        fixture = self.modules()
        def sample(**kwargs):
            raise fixture.spatial.AdaptiveRoiBudgetError("expected")
        fixture.cache.build_training_sample = sample
        with self.assertRaisesRegex(ValueError, "not captured"):
            self.capture(fixture)
        self.assert_restored(fixture)


if __name__ == "__main__":
    unittest.main()
