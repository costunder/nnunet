"""Half-B metadata UNIT and explicitly opt-in CUDA algebra DEBUG checks.

CPU tests exercise admission/identity/constructor mechanics, not neural quality.
The CUDA class uses synthetic fused128 vectors solely to compare the batched
upper equations and gradients with the actual tracked legacy native model.
It is separate from the real archived-CT/original-loss CUDA learning smoke.
Set RUN_HALF_B_CUDA_PARITY=1 and select HalfBCudaNativeParity explicitly to run.
"""
from __future__ import annotations

import copy
import os
import unittest

import torch

from hiercp_v1x.half_b_model import (
    HalfBUpper, MARKER_KEY, SUPPORT_POLICY, checkpoint_marker,
    half_b_identity, half_b_spec, install_half_b, model_contract, require_half_b_state,
)


def support_metadata(device="cpu", *, patients=3, generation="UNIT-generation-1"):
    """Synthetic labeled metadata only; this is never a real training cache."""
    cases = tuple(f"UNIT-train-{index}" for index in range(patients))
    samples = tuple(f"UNIT-signed-sample-{index}" for index in range(patients))
    generator = torch.Generator(device="cpu").manual_seed(713)
    return {
        "embeddings": torch.randn(patients * 8, 128, generator=generator).to(device),
        "owners": torch.arange(patients, device=device).repeat_interleave(8),
        "classes": (torch.arange(8, device=device) == 0).long().repeat(patients),
        "patient_case_ids": cases,
        "sample_ids": tuple(sample for sample in samples for _ in range(8)),
        "candidate_indices": tuple(range(8)) * patients,
        "expected_samples": dict(zip(samples, cases)),
        "training_case_ids": cases,
        "validation_case_ids": ("UNIT-validation-0",),
        "manifest_sha256": "b" * 64, "generation": generation,
        "epoch": 0, "fixed_view_epoch": 0, "support_policy": SUPPORT_POLICY,
        "debug": True, "full_signed_training_cache": False,
    }


class HalfBMetadataUnit(unittest.TestCase):
    def make_upper(self):
        # Constructor only: no L1/L2/CNN forward on CPU.
        return HalfBUpper(dropout=.1, scope_contract="a" * 64, debug_support=True)

    def test_contract_preserves_scale_and_declares_semantic_bridge(self):
        spec, contract = half_b_spec(), model_contract()
        self.assertEqual((spec["hidden_dim"], spec["heads"], spec["task_layers"],
                          spec["alignment_layers"]), (128, 4, 2, 2))
        self.assertEqual(spec["temperature"], .2)
        self.assertFalse(spec["native_v22_equivalence"])
        self.assertFalse(spec["query_GT_in_forward"])
        self.assertFalse(spec["native_alignment_CE_added"])
        self.assertFalse(spec["native_observation_CE_added"])
        self.assertIn("logit1_minus_logit0", spec["scorer"])
        self.assertEqual((contract["training_candidates_per_sample"], contract["candidate_pool_size"],
                          contract["configured_train_cases"], contract["configured_validation_cases"],
                          contract["production_epochs"]), (8, 128, 84, 21, 40))
        self.assertEqual(half_b_identity(), half_b_identity())

    def test_install_preserves_actual_local_owner_state_and_caller_cpu_rng(self):
        from hiercp.model import HierarchicalPyGPlacementModel
        native = HierarchicalPyGPlacementModel(hidden_dim=128, heads=4, local_layers=3,
                         patient_layers=2, prototype_layers=2, dense_batch_size=4)
        native.architecture_version += "|bounded_scope_" + "a" * 64
        local = native.local_encoder
        local_values = {name: value.clone() for name, value in local.state_dict().items()}
        old_upper_ids = {id(parameter) for name in (
            "patient_encoder", "prototype_encoder", "patient_readout", "population_readout", "score_head")
            for parameter in getattr(native, name).parameters()}
        rng = torch.get_rng_state().clone()
        original_forward = native.forward.__func__
        self.assertIs(install_half_b(native, debug_support=True), native)
        self.assertIs(native.local_encoder, local)
        self.assertIs(native.forward.__func__, original_forward)
        for name, value in local_values.items():
            self.assertTrue(torch.equal(local.state_dict()[name], value), name)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(old_upper_ids & {id(parameter) for parameter in native.parameters()})
        self.assertFalse(hasattr(native.half_b.core, "local"))
        self.assertEqual(len(native.half_b.core.l1), 2)
        self.assertEqual(len(native.half_b.core.l2), 2)
        self.assertTrue(all(layer.num_heads == 4 for layer in native.half_b.core.l2))
        require_half_b_state(native.state_dict(), debug_support=True)
        self.assertIs(install_half_b(native, debug_support=True).local_encoder, local)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        with self.assertRaises(ValueError):
            install_half_b(native)

    def test_marker_distinguishes_debug_production_and_rejects_another_arm(self):
        marker = checkpoint_marker(debug_support=True)
        require_half_b_state({MARKER_KEY: marker}, debug_support=True)
        require_half_b_state({"nested." + MARKER_KEY: marker}, prefix="nested.", debug_support=True)
        for state in ({}, {MARKER_KEY: marker.float()}, {MARKER_KEY: marker.roll(1)}):
            with self.assertRaises(RuntimeError):
                require_half_b_state(state, debug_support=True)
        with self.assertRaises(RuntimeError):
            require_half_b_state({MARKER_KEY: marker})

    def test_complete_support_admission_receipt_and_whole_query_patient_exclusion(self):
        upper = self.make_upper()
        memory = support_metadata()
        receipt = upper.bind_support(memory)
        self.assertEqual(receipt["support_candidates"], 24)
        self.assertEqual(receipt["support_source_samples"], 3)
        self.assertFalse(receipt["query_GT_labels_used"])
        self.assertFalse(receipt["live_upper_states_cached"])
        episode = upper._episode("UNIT-train-0")
        self.assertEqual(episode["patients"], 2)
        self.assertEqual(episode["patient_case_ids"], ("UNIT-train-1", "UNIT-train-2"))
        self.assertEqual(len(episode["indices"]), 16)
        self.assertTrue(torch.equal(episode["indices"], torch.arange(8, 24)))
        self.assertTrue(torch.equal(episode["owners"], torch.arange(2).repeat_interleave(8)))
        self.assertIs(upper._episode("UNIT-train-0"), episode)
        self.assertEqual(len(upper._episode(None)["indices"]), 24)

    def test_missing_incomplete_duplicated_wrong_class_and_validation_support_refused(self):
        upper = self.make_upper()
        with self.assertRaises(RuntimeError):
            upper.support_receipt()
        for changed in (
            lambda m: m.pop("expected_samples"),
            lambda m: m.update(candidate_indices=(0,) * 24),
            lambda m: m.update(classes=torch.zeros(24, dtype=torch.long)),
            lambda m: m.update(patient_case_ids=("UNIT-validation-0", "UNIT-train-1", "UNIT-train-2")),
            lambda m: m.update(support_policy="unsupported"),
            lambda m: m.update(embeddings=m["embeddings"].requires_grad_()),
            lambda m: m.update(embeddings=m["embeddings"].double()),
            lambda m: m.update(embeddings=torch.full((24, 128), torch.nan)),
            lambda m: m.update(full_signed_training_cache=True),
        ):
            with self.subTest(change=changed):
                memory = support_metadata()
                changed(memory)
                with self.assertRaises(ValueError):
                    upper.bind_support(memory)

    def test_production_refuses_debug_or_incomplete_signed_split_without_neural_call(self):
        upper = HalfBUpper(dropout=.1, scope_contract="a" * 64)
        with self.assertRaisesRegex(ValueError, "Production support"):
            upper.bind_support(support_metadata())

    def test_memory_mutation_duplicate_generation_and_loading_require_fresh_refresh(self):
        upper = self.make_upper()
        memory = support_metadata()
        upper.bind_support(memory)
        with self.assertRaisesRegex(ValueError, "Fresh unique-generation"):
            upper.bind_support(memory)
        memory["embeddings"][0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "changed after admission"):
            upper.support_receipt()
        upper.bind_support(support_metadata(generation="UNIT-generation-2"))
        state = upper.get_extra_state()
        self.assertEqual(state, {**half_b_spec(), "bounded_scope_contract": "a" * 64,
                                 "debug_support": True})
        upper.set_extra_state(state)
        with self.assertRaises(RuntimeError):
            upper.support_receipt()
        state["bounded_scope_contract"] = "c" * 64
        with self.assertRaises(RuntimeError):
            upper.set_extra_state(state)

    def test_query_admission_rejects_before_any_neural_call(self):
        upper = self.make_upper()
        upper.bind_support(support_metadata())
        for query, cases, counts in (
            (torch.zeros(7, 128), ("UNIT-train-0",), (7,)),
            (torch.zeros(8, 127), ("UNIT-train-0",), (8,)),
            (torch.zeros(8, 128), ("UNIT-validation-0",), (8,)),
            (torch.zeros(8, 128), ("unlisted-case",), (8,)),
            (torch.full((8, 128), torch.nan), ("UNIT-train-0",), (8,)),
        ):
            with self.assertRaises(ValueError):
                upper.score(query, cases, counts)


@unittest.skipUnless(os.environ.get("RUN_HALF_B_CUDA_PARITY") == "1",
                     "Explicit CUDA algebra DEBUG opt-in required; CPU metadata does not validate neural equations")
class HalfBCudaNativeParity(unittest.TestCase):
    """Full128/4heads/2+2 algebra DEBUG, actual native classes, CUDA only."""
    def setUp(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA parity has no CUDA device; no CPU fallback")
        self.device = torch.device("cuda:0")

    def compare_native(self, *, training, cases):
        # Dropout0 is deliberate algebra DEBUG. Production retains original .1.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            upper = HalfBUpper(dropout=0., scope_contract="a" * 64, debug_support=True)
        upper = upper.to(self.device).train(training)
        memory = support_metadata(self.device, patients=5)
        upper.bind_support(memory)
        reference = copy.deepcopy(upper.core).train(training)
        generator = torch.Generator(device="cpu").manual_seed(983)
        counts = (8,) * len(cases)
        q = torch.randn(sum(counts), 128, generator=generator).to(self.device).requires_grad_()
        reference_q = q.detach().clone().requires_grad_()
        score = torch.cat(upper.score(q, cases, counts))
        # Native reference: same entire eligible support and exact teacher plan;
        # only the adapter's episode batching/padding differs. No native CE used.
        keys = [case if case in memory["patient_case_ids"] else None for case in cases]
        native_states = {}
        for key in dict.fromkeys(keys):
            episode = upper._episode(key)
            support = memory["embeddings"][episode["indices"]]
            plan = copy.deepcopy(upper._plans[key])
            plan["support_embeddings"] = support.detach()
            native_states[key] = reference.prepare_support(support, episode["owners"],
                                      episode["classes"], cluster_plan=plan)
        reference_scores = []
        for query_piece, key in zip(reference_q.split(counts), keys):
            result = reference.predict_embeddings(query_piece, native_states[key])
            reference_scores.append(result["logits"][:, 1] - result["logits"][:, 0])
        reference_score = torch.cat(reference_scores)
        torch.testing.assert_close(score, reference_score, atol=1e-5, rtol=1e-4)
        weight = torch.linspace(.3, 1.7, len(q), device=self.device)
        objective = (score * weight).sum() + .17 * score.square().mean()
        reference_objective = (reference_score * weight).sum() + .17 * reference_score.square().mean()
        objective.backward()
        reference_objective.backward()
        torch.testing.assert_close(q.grad, reference_q.grad, atol=2e-5, rtol=1e-3)
        maximum_parameter_delta = 0.
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in upper.core.named_parameters():
            other = reference_parameters[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(other.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0., name)
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-4, rtol=2e-3,
                                       msg=lambda message, name=name: f"{name}: {message}")
            maximum_parameter_delta = max(maximum_parameter_delta,
                        float((parameter.grad - other.grad).abs().max()))
        self.assertIsNone(memory["embeddings"].grad)
        self.assertEqual(upper.support_receipt()["cached_exclusion_teacher_plans"], len(set(keys)))
        cached = {key: id(value) for key, value in upper._plans.items()}
        with torch.no_grad():
            upper.score(q.detach(), cases, counts)
        self.assertEqual(cached, {key: id(value) for key, value in upper._plans.items()})
        print(f"CUDA algebra DEBUG training={training}; native score maxabs="
              f"{float((score-reference_score).abs().max()):.9g}; query gradient maxabs="
              f"{float((q.grad-reference_q.grad).abs().max()):.9g}; parameter gradient "
              f"maxabs={maximum_parameter_delta:.9g}; every upper parameter connected; "
              "synthetic fused-vector parity only, no CT/classification-quality claim")

    def test_eval_mixed_exclusions_heldout_padding_and_repeated_case_native_output_gradient(self):
        self.compare_native(training=False, cases=("UNIT-train-0", "UNIT-train-1",
                                 "UNIT-validation-0", "UNIT-train-0"))

    def test_train_checkpointed_disjoint_episodes_native_output_gradient(self):
        self.compare_native(training=True, cases=("UNIT-train-0", "UNIT-train-1"))

    def test_install_preserves_already_initialized_cuda_rng(self):
        from hiercp.model import HierarchicalPyGPlacementModel
        # CUDA is already initialized by setUp. CPU construction/install must
        # preserve both caller generators even in this common runner sequence.
        torch.rand(2, device=self.device)
        native = HierarchicalPyGPlacementModel(hidden_dim=128, heads=4, local_layers=3,
                         patient_layers=2, prototype_layers=2, dense_batch_size=4)
        native.architecture_version += "|bounded_scope_" + "a" * 64
        cpu_state = torch.get_rng_state().clone()
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        install_half_b(native, debug_support=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
        for before, after in zip(cuda_states, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
