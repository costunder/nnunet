"""Debug numeric/plumbing fixtures, not medical data or trained-model evaluation."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import random
import types
import unittest
from pathlib import Path

import numpy as np

from custom_trainers.onlinecp_curriculum_policy import (
    CONFIG_FORMAT, FNV_OFFSET, CurriculumError, curriculum_config_sha256,
    eligible_candidate_indices, select_curriculum_candidate, stage_for_epoch,
    update_digest, validate_curriculum_config,
)
from custom_trainers import onlinecp_curriculum_policy as policy


ROOT = Path(__file__).resolve().parents[1]


def debug_config():
    return {
        "format": CONFIG_FORMAT, "experiment_id": "debug_numeric_fixture_only",
        "num_epochs": 250, "candidate_count": 128, "cp_probability": 0.5,
        "score_floor": 0.0, "max_score_drop": 1.0,
        "stages": [
            dict(start_epoch=0, end_epoch=100, top_rank=1, minimum_choices=1, temperature=None),
            dict(start_epoch=100, end_epoch=250, top_rank=32, minimum_choices=4, temperature=None),
        ],
    }


class PolicyDebugTests(unittest.TestCase):
    def setUp(self):
        self.config = debug_config()
        self.scores = np.linspace(1.0, 0.0, 128)

    def test_full_explicit_contract_and_hash(self):
        self.assertEqual(validate_curriculum_config(self.config)["num_epochs"], 250)
        self.assertEqual(curriculum_config_sha256(self.config), curriculum_config_sha256(copy.deepcopy(self.config)))
        changed = copy.deepcopy(self.config)
        changed["stages"][0]["end_epoch"] = 90
        changed["stages"][1]["start_epoch"] = 90
        self.assertNotEqual(curriculum_config_sha256(self.config), curriculum_config_sha256(changed))

    def test_no_hidden_reduction_or_probability_change(self):
        for field, value in (("num_epochs", 2), ("candidate_count", 8), ("cp_probability", 0.1)):
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = value
                with self.assertRaises(CurriculumError):
                    validate_curriculum_config(config)

    def test_missing_or_unknown_fields_fail(self):
        for operation in (lambda c: c.pop("max_score_drop"), lambda c: c.update(temperatur=0.5)):
            config = copy.deepcopy(self.config)
            operation(config)
            with self.assertRaises(CurriculumError):
                validate_curriculum_config(config)

    def test_stages_cannot_gap_overlap_shrink_or_remain_argmax(self):
        for field, value in (("start_epoch", 101), ("end_epoch", 249), ("top_rank", 1), ("minimum_choices", 1)):
            config = copy.deepcopy(self.config)
            config["stages"][1][field] = value
            with self.subTest(field=field), self.assertRaises(CurriculumError):
                validate_curriculum_config(config)

    def test_stage_boundaries_are_half_open(self):
        config = validate_curriculum_config(self.config)
        self.assertEqual(stage_for_epoch(config, 99)[0], 0)
        self.assertEqual(stage_for_epoch(config, 100)[0], 1)
        self.assertEqual(stage_for_epoch(config, 249)[0], 1)
        for epoch in (-1, 250, True):
            with self.assertRaises(CurriculumError):
                stage_for_epoch(config, epoch)

    def test_progression_reaches_non_argmax_choices(self):
        self.assertEqual(select_curriculum_candidate(self.scores, self.config, 0, .99), 0)
        selected = {select_curriculum_candidate(self.scores, self.config, 200, (i+.5)/32) for i in range(32)}
        self.assertEqual(selected, set(range(32)))

    def test_basic_control_keeps_entire_anatomical_pool(self):
        selected = {select_curriculum_candidate(self.scores, self.config, 0, (i+.5)/128,
                                               basic_control=True) for i in range(128)}
        self.assertEqual(selected, set(range(128)))

    def test_score_gate_failure_is_not_greedy_or_bad_candidate_fallback(self):
        config = copy.deepcopy(self.config)
        config["score_floor"] = 1.0
        with self.assertRaisesRegex(CurriculumError, "requires 4"):
            select_curriculum_candidate(self.scores, config, 150, .99)

    def test_bad_scores_draws_and_temperature_fail(self):
        for draw in (-.01, 1.0, float("nan")):
            with self.subTest(draw=draw), self.assertRaises(CurriculumError):
                select_curriculum_candidate(self.scores, self.config, 150, draw)
        for values in (self.scores[:8], np.full(128, np.nan)):
            with self.assertRaises(CurriculumError):
                select_curriculum_candidate(values, self.config, 150, .5)
        config = copy.deepcopy(self.config)
        config["stages"][1]["temperature"] = 1e-30
        with self.assertRaisesRegex(CurriculumError, "collapses"):
            select_curriculum_candidate(self.scores, config, 150, .5)

    def test_temperature_stable_ties_and_no_rng_side_effect(self):
        config = copy.deepcopy(self.config)
        config["stages"][1]["temperature"] = .5
        before = np.random.get_state()
        scores = np.ones(128)
        self.assertEqual(select_curriculum_candidate(scores, config, 150, .99), 31)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(scores, np.ones(128))


def _load_debug_class_definitions():
    """Execute real selection methods with framework scaffolding only.

    No nnU-Net crop, network, loss, optimizer, checkpoint file or medical image
    is constructed. This tests the legacy-hook-to-new-policy connection.
    """
    class DebugFrameworkBase:
        pass

    fake_torch = types.SimpleNamespace(device=lambda value: types.SimpleNamespace(type=value))
    namespace = dict(np=np, random=random, hashlib=hashlib, torch=fake_torch, os=os,
                     nnUNetDataLoader=DebugFrameworkBase, nnUNetTrainer=DebugFrameworkBase,
                     TRAINER_FORMAT="hiercp_online_trainer_v2")
    for name in ("CONFIG_FORMAT", "FNV_OFFSET", "RESUME_FORMAT", "SCHEDULE_FORMAT",
                 "CurriculumError", "canonical_sha256", "curriculum_config_sha256",
                 "schedule_token", "select_curriculum_candidate", "stage_for_epoch",
                 "update_digest", "validate_curriculum_config"):
        namespace[name] = getattr(policy, name)
    selections = [
        ("nnUNetTrainer_OnlinePairedCP.py", {
            "OnlineCPError", "_stable_seed", "_stable_u64", "_select_candidate_index",
            "nnUNetDataLoaderOnlineCP", "_nnUNetTrainer_250epochs_OnlineCP",
        }),
        ("nnUNetTrainer_OnlineCPCurriculum.py", {
            "nnUNetDataLoaderOnlineCPCurriculum", "_nnUNetTrainer_250epochs_OnlineCurriculum",
        }),
    ]
    for filename, names in selections:
        source = ROOT / "custom_trainers" / filename
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if {node.name for node in nodes} != names:
            raise AssertionError("A required production hook has disappeared")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class LoaderPlumbingDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = _load_debug_class_definitions()

    def stream(self, epoch, basic, final_rank=32):
        loader_class = self.namespace["nnUNetDataLoaderOnlineCPCurriculum"]
        loader = object.__new__(loader_class)
        config = debug_config()
        config["stages"][1]["top_rank"] = final_rank
        loader.curriculum_config = validate_curriculum_config(config)
        loader.curriculum_sha256 = curriculum_config_sha256(config)
        loader.basic_control = basic
        loader.online_epoch, loader.online_seed, loader.thread_id = epoch, 42, 3
        loader._cp_rng = np.random.default_rng(self.namespace["_stable_seed"](
            "hiercp_online_trainer_v2", 42, "cp", epoch, 3))
        loader._choice_tokens = []
        entry = {"candidate_centers": np.arange(384).reshape(128, 3),
                 "scores": np.linspace(1., 0., 128)}
        loader.online_bank = types.SimpleNamespace(
            entry_names=lambda case: ["debug_source"], load_for_case=lambda case, index: entry,
            cp_probability=.5, intensity_scale=(.95, 1.05), intensity_shift_hu=(-5., 5.),
            ct_mean=0., ct_std=1.,
        )
        sampled = [loader._sample_paste_plan("debug_case") for _ in range(100)]
        return sampled, loader._choice_tokens

    def test_actual_inherited_draws_and_new_selection_hook_are_paired(self):
        basic, basic_choices = self.stream(200, True)
        curriculum, curriculum_choices = self.stream(200, False)
        self.assertEqual([event for _, event in basic], [event for _, event in curriculum])
        self.assertEqual([plan is None for plan, _ in basic], [plan is None for plan, _ in curriculum])
        for (first, _), (second, _) in zip(basic, curriculum):
            if first is not None:
                self.assertEqual(first["scale"], second["scale"])
                self.assertEqual(first["normalized_offset"], second["normalized_offset"])
                self.assertLess(second["candidate_index"], 32)
        self.assertNotEqual(update_digest(FNV_OFFSET, basic_choices), update_digest(FNV_OFFSET, curriculum_choices))

    def test_epoch_replay_is_deterministic_and_choice_digest_replays(self):
        first, first_choices = self.stream(200, False)
        second, second_choices = self.stream(200, False)
        self.assertEqual([event for _, event in first], [event for _, event in second])
        self.assertEqual(first_choices, second_choices)
        self.assertNotEqual(first_choices, self.stream(201, False)[1])

    def test_equal_actual_choices_have_equal_digest_even_for_different_policies(self):
        _, basic_choices = self.stream(200, True, final_rank=128)
        _, curriculum_choices = self.stream(200, False, final_rank=128)
        self.assertEqual(basic_choices, curriculum_choices)

    def test_get_dataloaders_instantiates_the_new_loader(self):
        source = ROOT / "custom_trainers" / "nnUNetTrainer_OnlineCPCurriculum.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "_nnUNetTrainer_250epochs_OnlineCurriculum")
        bindings = [node.value for node in trainer.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "online_loader_class"
                            for target in node.targets)]
        self.assertEqual(len(bindings), 1)
        self.assertIsInstance(bindings[0], ast.Name)
        self.assertEqual(bindings[0].id, "nnUNetDataLoaderOnlineCPCurriculum")
        get_loaders = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                           and node.name == "get_dataloaders")
        factory_calls = [node for node in ast.walk(get_loaders) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "online_loader_class"
                         and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"]
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(ast.unparse(factory_calls[0].args[1]), "self.batch_size")
        self.assertTrue(any(keyword.arg is None and isinstance(keyword.value, ast.Call)
                            and ast.unparse(keyword.value.func) == "self._loader_policy_kwargs"
                            for keyword in factory_calls[0].keywords))
        direct_calls = [node.func.id for node in ast.walk(get_loaders)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("nnUNetDataLoaderOnlineCP", direct_calls)
        policy_helper = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                             and node.name == "_loader_policy_kwargs")
        isolated = {}
        exec(compile(ast.Module(body=[policy_helper], type_ignores=[]), str(source), "exec"), isolated)
        context = type("DebugLoaderPolicyContext", (), {})()
        context.curriculum_config = debug_config()
        for basic_control in (False, True):
            context.basic_control = basic_control
            kwargs = isolated["_loader_policy_kwargs"](context)
            self.assertEqual(set(kwargs), {"curriculum_config", "basic_control"})
            self.assertIs(kwargs["curriculum_config"], context.curriculum_config)
            self.assertIs(kwargs["basic_control"], basic_control)

    def test_legacy_or_incomplete_checkpoint_fails_without_starting_network(self):
        trainer = object.__new__(self.namespace["_nnUNetTrainer_250epochs_OnlineCurriculum"])
        with self.assertRaisesRegex(CurriculumError, "no fresh-start fallback"):
            trainer.load_checkpoint({"network_weights": {}, "current_epoch": 100})
        payload = {key: None for key in (
            "network_weights", "optimizer_state", "grad_scaler_state", "logging", "_best_ema",
            "current_epoch", "init_args", "trainer_name", "inference_allowed_mirroring_axes",
            "onlinecp_curriculum_resume",
        )}
        with self.assertRaisesRegex(CurriculumError, "Malformed or legacy"):
            trainer.load_checkpoint(payload)

    def test_resume_requires_complete_last_epoch_choice_and_event_audit(self):
        trainer_class = self.namespace["_nnUNetTrainer_250epochs_OnlineCurriculum"]
        valid = dict(epoch=199, stage=1, applied=193, samples=500,
                     event_digest="0123456789abcdef", choice_digest="fedcba9876543210")
        trainer_class._validate_epoch_record(valid, debug_config(), 200, 500)
        for field in valid:
            incomplete = dict(valid)
            del incomplete[field]
            with self.subTest(missing=field), self.assertRaises(CurriculumError):
                trainer_class._validate_epoch_record(incomplete, debug_config(), 200, 500)
        for field, value in (("epoch", 198), ("stage", 0), ("applied", 501),
                             ("samples", 499), ("samples", True),
                             ("event_digest", ""), ("choice_digest", "z" * 16)):
            corrupt = {**valid, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(CurriculumError):
                trainer_class._validate_epoch_record(corrupt, debug_config(), 200, 500)


if __name__ == "__main__":
    unittest.main()
