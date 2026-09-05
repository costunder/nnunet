"""Isolated debug/contract tests only: no medical data, nnU-Net, or training."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import downstream_level_ablation as d
from tools.online_eval_provenance import build_evaluation_contract


# These are user-supplied log lines, not medical inputs or a training execution.
USER_EPOCH_100_LINES = (
    "2026-09-03 22:12:40.553242: [OnlineCP] epoch=100 applied=193/500 rate=0.3860 schedule=41dce5c1165452df",
    "2026-09-03 23:41:21.821738: [OnlineCP] epoch=100 applied=193/500 rate=0.3860 schedule=41dce5c1165452df",
)
USER_EPOCH_100_TUPLE = (193, 500, "41dce5c1165452df")


class DownstreamScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="debug_downstream_audit_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trainers = (d.BASIC_TRAINER, d.FULL_TRAINER, d.NO_PATIENT_TRAINER, d.NO_POPULATION_TRAINER)
        self.results = {trainer: self.root / trainer for trainer in self.trainers}
        for result in self.results.values():
            result.mkdir()
            self.write_log(result)
        self.run = SimpleNamespace(report_root=self.root / "report", outer_fold=0, dataset_id=730)

    @staticmethod
    def record(epoch, applied=2, samples=500):
        return f"[OnlineCP] epoch={epoch} applied={applied}/{samples} rate=0.004 schedule={epoch:016x}"

    def write_log(self, result, records=None):
        rows = records if records is not None else [self.record(epoch, applied=0 if epoch == 0 else 2) for epoch in range(250)]
        path = result / "training_log_01.txt"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def audit(self, duplicate_policy="error"):
        d._reserve_evaluation_output(self.run, duplicate_policy=duplicate_policy)
        with patch.object(d, "_model_dir", side_effect=lambda run, trainer: self.results[trainer]):
            d._audit_schedules(self.run, duplicate_policy=duplicate_policy)
        return d._load_json(self.run.report_root / "schedule_audit.json")

    def write_user_duplicate_logs(self, result):
        """Retain 249 debug epochs plus the two verbatim user log occurrences."""
        self.write_log(result, [self.record(epoch, applied=0 if epoch == 0 else 2)
                                for epoch in range(250) if epoch != 100])
        paths = (result / "training_log_2026_9_3_19_50_00.txt",
                 result / "training_log_2026_9_3_23_39_23.txt")
        for path, line_number, text in zip(paths, (891, 26), USER_EPOCH_100_LINES):
            padding = ["DEBUG fixture padding; no training performed."] * (line_number - 1)
            path.write_text("\n".join([*padding, text]) + "\n", encoding="utf-8")
        return paths

    def prepare_matching_coalesced_runs(self):
        for result in self.results.values():
            self.write_log(result, [USER_EPOCH_100_LINES[0] if epoch == 100 else
                                   self.record(epoch, applied=0 if epoch == 0 else 2)
                                   for epoch in range(250)])
        self.write_user_duplicate_logs(self.results[d.NO_PATIENT_TRAINER])
        for trainer, repetitions in ((d.FULL_TRAINER, 1), (d.NO_POPULATION_TRAINER, 2)):
            with (self.results[trainer] / "training_log_01.txt").open("a", encoding="utf-8") as handle:
                handle.write((USER_EPOCH_100_LINES[1] + "\n") * repetitions)

    def prepare_debug_aggregate(self):
        """Create isolated publication fixtures, never clinical evaluations."""
        self.prepare_matching_coalesced_runs()
        audit = self.audit("coalesce-identical")
        case_ids = ["debug_case_2", "debug_case_10"]
        input_root = self.root / "debug_aggregate_input_bytes"

        def files(role):
            folder = input_root / role
            folder.mkdir(parents=True)
            paths = {}
            for case_id in case_ids:
                path = folder / f"{case_id}.bytes"
                path.write_text(f"DEBUG PUBLICATION FIXTURE ONLY: {role}/{case_id}", encoding="utf-8")
                paths[case_id] = path
            return paths

        ground_truth, full = files("ground_truth"), files("full")
        summaries = {}
        self.run.evaluation_for = lambda name: self.run.report_root / name
        for name, trainer in (("basic_vs_full", d.BASIC_TRAINER), ("no_patient_vs_full", d.NO_PATIENT_TRAINER),
                              ("no_population_vs_full", d.NO_POPULATION_TRAINER)):
            contract = build_evaluation_contract(
                cohort=case_ids, ground_truth_files=ground_truth,
                prediction_files={"basic": files(name), "hier": full},
                trainers={"basic": trainer, "hier": d.FULL_TRAINER},
                evaluation_definition={"outer_fold": 0, "dataset_id": 730, "debug_fixture": True},
            )
            # Deliberately undefined inference and absent size bins exercise N/A
            # formatting. These constants are not computed clinical metrics.
            statistic = {"difference": 0.0, "ci_low": None, "ci_high": None, "permutation_p": None}
            method = {"recall": 0.5, "precision": 0.5, "f1": 0.5, "fp_per_case": 0.5}
            criteria = {criterion: {"basic_cp": dict(method), "hiercp": dict(method), "by_size": {},
                                    "statistics": {metric: dict(statistic) for metric in method}}
                        for criterion in ("dice_ge_0p10", "dice_ge_0p25")}
            summary = {"debug_fixture": True, "actual_data_evaluation": False, "outer_fold": 0, "dataset_id": 730,
                       "evaluation_contract": contract, "criteria": criteria,
                       "case_tumor_dice": {"basic_mean": 0.5, "hier_mean": 0.5, "statistics": dict(statistic)},
                       "lesion_dice_quality": {"basic_mean": None, "hier_mean": None}}
            output = self.run.evaluation_for(name)
            output.mkdir()
            d._publish_report_json(output / "summary.json", summary)
            digest = d._sha256(output / "summary.json")
            d._publish_report_json(output / "completion.json", {
                "format": "online_eval_completion_v1", "complete": True,
                "debug_fixture": True, "actual_data_evaluation": False, "training_executed": False,
                "summary_sha256": digest, "evaluation_contract_sha256": contract["contract_sha256"],
                "outputs": {"summary.json": digest},
            })
            summaries[name] = summary
        return case_ids, audit, summaries, input_root

    def test_valid_exact_250_all_four_and_zero_event_epoch_allowed(self):
        audit = self.audit()
        self.assertIs(audit["matched"], True)
        self.assertEqual(audit["total_samples"], 125000)
        self.assertEqual(audit["total_cp_events"], 498)
        self.assertEqual(len(audit["log_inputs"]), 4)
        self.assertEqual(len(audit["epoch_records"]["basic"]), 250)
        self.assertEqual(audit["format"], "hiercp_downstream_ablation_schedule_audit_v3")
        self.assertEqual(audit["duplicate_policy"], "error")
        self.assertEqual(audit["training_resume_status"], "unverified")
        self.assertEqual(audit["epochs_kind"], "unique_recorded_epoch_indices")
        for details in audit["duplicate_audit"].values():
            self.assertEqual(details["status"], "no_duplicates")
            self.assertEqual(details["duplicate_epochs"], [])
            self.assertEqual(details["duplicate_occurrences"], 0)
            self.assertEqual(details["raw_logged_occurrences"], details["unique_epochs"])
        d._verify_schedule_audit(audit)

    def test_identical_and_conflicting_duplicate_have_both_locations(self):
        result = self.results[d.BASIC_TRAINER]
        for applied in (0, 3):
            with self.subTest(applied=applied):
                path = self.write_log(result)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(self.record(0, applied=applied) + "\n")
                with self.assertRaisesRegex(d.DownstreamAblationError, r"Duplicate.*:1, again at .*:251"):
                    d._schedule_records_from_result(result)

    def test_duplicate_across_files_is_not_resume_overwrite(self):
        result = self.results[d.BASIC_TRAINER]
        (result / "training_log_02.txt").write_text(self.record(1) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(d.DownstreamAblationError, "Duplicate OnlineCP epoch 1"):
            d._schedule_records_from_result(result)

    def test_user_epoch100_two_files_coalesce_only_with_explicit_policy(self):
        result = self.results[d.NO_PATIENT_TRAINER]
        paths = self.write_user_duplicate_logs(result)
        before = {path: path.read_bytes() for path in result.glob("training_log_*.txt")}
        with self.assertRaisesRegex(d.DownstreamAblationError, "Duplicate OnlineCP epoch 100"):
            d._schedule_records_from_result(result)
        observation = d._schedule_observations_from_result(result, duplicate_policy="coalesce-identical")
        self.assertEqual(observation["records"][100], USER_EPOCH_100_TUPLE)
        self.assertEqual(d._schedule_records_from_result(result, duplicate_policy="coalesce-identical"), observation["records"])
        details = observation["audit"]
        self.assertEqual(details["status"], "identical_duplicates_coalesced")
        self.assertEqual(details["duplicate_epochs"], [100])
        self.assertEqual(details["duplicate_occurrences"], 1)
        occurrences = [item for item in details["occurrences"] if item["epoch"] == 100]
        self.assertEqual([(Path(item["path"]), item["line_number"], item["text"]) for item in occurrences],
                         [(paths[0].resolve(), 891, USER_EPOCH_100_LINES[0]),
                          (paths[1].resolve(), 26, USER_EPOCH_100_LINES[1])])
        self.assertEqual(details["unique_epochs"], 250)
        self.assertEqual(details["raw_logged_occurrences"], 251)
        self.assertEqual(details["unique_total_samples"], 125000)
        self.assertEqual(details["raw_total_samples"], 125500)
        self.assertEqual(details["unique_total_cp_events"], 689)
        self.assertEqual(details["raw_total_cp_events"], 882)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_identical_occurrences_in_same_file_preserve_each_line(self):
        result = self.results[d.BASIC_TRAINER]
        for uppercase in (False, True):
            with self.subTest(uppercase_fingerprint=uppercase):
                second_line = USER_EPOCH_100_LINES[1]
                if uppercase:
                    second_line = second_line.replace(USER_EPOCH_100_TUPLE[2], USER_EPOCH_100_TUPLE[2].upper())
                source_lines = [USER_EPOCH_100_LINES[0], second_line]
                self.write_log(result, source_lines)
                observation = d._schedule_observations_from_result(result, duplicate_policy="coalesce-identical")
                self.assertEqual(observation["records"], {100: USER_EPOCH_100_TUPLE})
                occurrences = observation["audit"]["occurrences"]
                self.assertEqual([item["line_number"] for item in occurrences], [1, 2])
                self.assertEqual([item["schedule"] for item in occurrences], [USER_EPOCH_100_TUPLE[2]] * 2)
                self.assertEqual([item["text"] for item in occurrences], source_lines)
                self.assertEqual(observation["audit"]["duplicate_occurrences"], 1)

    def test_three_occurrences_are_all_preserved_and_not_tripled_as_unique(self):
        result = self.results[d.BASIC_TRAINER]
        self.write_log(result, [USER_EPOCH_100_LINES[0], USER_EPOCH_100_LINES[1], USER_EPOCH_100_LINES[1]])
        details = d._schedule_observations_from_result(result, duplicate_policy="coalesce-identical")["audit"]
        self.assertEqual(details["duplicate_occurrences"], 2)
        self.assertEqual(len(details["occurrences"]), 3)
        self.assertEqual(details["raw_logged_occurrences"], 3)
        self.assertEqual(details["raw_total_samples"], 1500)
        self.assertEqual(details["raw_total_cp_events"], 579)
        self.assertEqual(details["unique_epochs"], 1)
        self.assertEqual(details["unique_total_samples"], 500)
        self.assertEqual(details["unique_total_cp_events"], 193)

    def test_coalesce_rejects_conflict_in_each_semantic_tuple_field(self):
        result = self.results[d.BASIC_TRAINER]
        conflicts = [USER_EPOCH_100_LINES[1].replace("applied=193/500", "applied=192/500").replace("rate=0.3860", "rate=0.3840"),
                     USER_EPOCH_100_LINES[1].replace("applied=193/500", "applied=193/501").replace("rate=0.3860", "rate=0.3852"),
                     USER_EPOCH_100_LINES[1].replace("41dce5c1165452df", "41dce5c1165452de")]
        for repeat_count in (1, 2):
            for conflicting_line in conflicts:
                with self.subTest(prior_identical_occurrences=repeat_count, line=conflicting_line):
                    self.write_log(result, [USER_EPOCH_100_LINES[0]] * repeat_count + [conflicting_line])
                    with self.assertRaises(d.DownstreamAblationError):
                        d._schedule_observations_from_result(result, duplicate_policy="coalesce-identical")

    def test_coalesce_rejects_invalid_duplicate_counts(self):
        result = self.results[d.BASIC_TRAINER]
        for counts in ("193/0", "-1/500", "501/500"):
            with self.subTest(counts=counts):
                second_line = USER_EPOCH_100_LINES[1].replace("193/500", counts)
                self.write_log(result, [USER_EPOCH_100_LINES[0], second_line])
                with self.assertRaises(d.DownstreamAblationError):
                    d._schedule_records_from_result(result, duplicate_policy="coalesce-identical")

    def test_four_run_audit_separates_unique_and_raw_totals_and_preserves_logs(self):
        self.prepare_matching_coalesced_runs()
        before = {path: (path.read_bytes(), d._sha256(path)) for result in self.results.values()
                  for path in result.glob("training_log_*.txt")}
        audit = self.audit("coalesce-identical")
        self.assertIs(audit["matched"], True)
        self.assertEqual(audit["duplicate_policy"], "coalesce-identical")
        self.assertEqual(audit["total_samples"], 125000)
        self.assertEqual(audit["total_cp_events"], 689)
        self.assertEqual(audit["training_resume_status"], "unverified")
        self.assertEqual(audit["epochs_kind"], "unique_recorded_epoch_indices")
        self.assertTrue(audit["count_semantics"])
        for method, duplicates in (("basic", 0), ("full", 1), ("no_patient", 1), ("no_population", 2)):
            details = audit["duplicate_audit"][method]
            self.assertEqual(details["duplicate_occurrences"], duplicates)
            self.assertEqual(details["raw_logged_occurrences"], 250 + duplicates)
            self.assertEqual(details["raw_total_samples"], 125000 + 500 * duplicates)
            self.assertEqual(details["raw_total_cp_events"], 689 + 193 * duplicates)
            self.assertEqual(details["unique_total_samples"], 125000)
            self.assertEqual(details["unique_total_cp_events"], 689)
        self.assertEqual(before, {path: (path.read_bytes(), d._sha256(path)) for path in before})
        d._verify_schedule_audit(audit, duplicate_policy="coalesce-identical")

    def test_duplicate_metadata_mutation_or_omission_is_not_accepted(self):
        self.prepare_matching_coalesced_runs()
        audit = self.audit("coalesce-identical")
        variants = []
        for field, value in (("status", "no_duplicates"), ("duplicate_epochs", []), ("duplicate_occurrences", 0),
                             ("raw_logged_occurrences", 250), ("raw_total_samples", 125000),
                             ("raw_total_cp_events", 689), ("unique_epochs", 249),
                             ("unique_total_samples", 125500), ("unique_total_cp_events", 882)):
            changed = copy.deepcopy(audit)
            changed["duplicate_audit"]["no_patient"][field] = value
            variants.append((field, changed))
        for field, value in (("text", "altered original log line"), ("line_number", 999), ("path", "invented.txt"),
                             ("epoch", 249), ("applied", 123), ("samples", 501), ("schedule", "0" * 16)):
            changed = copy.deepcopy(audit)
            item = next(row for row in changed["duplicate_audit"]["no_patient"]["occurrences"] if row["epoch"] == 100)
            item[field] = value
            variants.append((f"occurrence_{field}", changed))
        changed = copy.deepcopy(audit)
        changed["duplicate_audit"]["no_patient"]["occurrences"].pop()
        variants.append(("removed_occurrence", changed))
        changed = copy.deepcopy(audit)
        del changed["duplicate_audit"]["no_patient"]["occurrences"]
        variants.append(("missing_occurrences", changed))
        for field, value in (("training_resume_status", "verified"), ("epochs_kind", "actual_executed_epochs"),
                             ("count_semantics", "tampered-count-definition"), ("duplicate_policy", "error")):
            changed = copy.deepcopy(audit)
            changed[field] = value
            variants.append((field, changed))
        for name, changed in variants:
            with self.subTest(field=name):
                with self.assertRaises(d.DownstreamAblationError):
                    d._verify_schedule_audit(changed, duplicate_policy="coalesce-identical")

    def test_verification_requires_same_explicit_duplicate_policy(self):
        self.prepare_matching_coalesced_runs()
        audit = self.audit("coalesce-identical")
        with self.assertRaises(d.DownstreamAblationError):
            d._verify_schedule_audit(audit)
        d._verify_schedule_audit(audit, duplicate_policy="coalesce-identical")
        changed = copy.deepcopy(audit)
        del changed["duplicate_policy"]
        with self.assertRaises(d.DownstreamAblationError):
            d._verify_schedule_audit(changed, duplicate_policy="coalesce-identical")

    def test_identical_duplicates_cannot_fill_a_missing_unique_epoch(self):
        target = self.results[d.NO_PATIENT_TRAINER]
        self.write_log(target, [self.record(epoch) for epoch in range(249)] + [self.record(100)])
        observation = d._schedule_observations_from_result(target, duplicate_policy="coalesce-identical")
        self.assertEqual(observation["audit"]["raw_logged_occurrences"], 250)
        self.assertEqual(observation["audit"]["unique_epochs"], 249)
        with self.assertRaises(d.DownstreamAblationError):
            self.audit("coalesce-identical")
        self.assertFalse((self.run.report_root / "schedule_audit.json").exists())

    def test_invalid_duplicate_policy_is_rejected_before_reserving_output(self):
        result = self.results[d.BASIC_TRAINER]
        for policy in ("last-wins", "", None, 1):
            with self.subTest(policy=policy):
                for parser in (d._schedule_observations_from_result, d._schedule_records_from_result):
                    with self.assertRaises(d.DownstreamAblationError):
                        parser(result, duplicate_policy=policy)
                with self.assertRaises(d.DownstreamAblationError):
                    d._reserve_evaluation_output(self.run, duplicate_policy=policy)
                self.assertFalse(self.run.report_root.exists())

    def test_cli_policy_and_started_manifest_default_strict_and_explicit_opt_in(self):
        parser = d.build_parser()
        self.assertEqual(parser.parse_args(["evaluate"]).schedule_duplicate_policy, "error")
        args = parser.parse_args(["evaluate", "--schedule-duplicate-policy", "coalesce-identical"])
        self.assertEqual(args.schedule_duplicate_policy, "coalesce-identical")
        action = next(action for action in parser._actions if action.dest == "schedule_duplicate_policy")
        self.assertEqual(set(action.choices), {"error", "coalesce-identical"})
        for policy in ("error", "coalesce-identical"):
            self.run.report_root = self.root / f"started_{policy}"
            d._reserve_evaluation_output(self.run, duplicate_policy=policy)
            started = d._load_json(self.run.report_root / "evaluation_started.json")
            self.assertEqual(started["schedule_duplicate_policy"], policy)
            self.assertIs(started["complete"], False)

    def test_evaluate_forwards_explicit_policy_to_reservation_audit_and_aggregate(self):
        args = SimpleNamespace(dry_run=False, schedule_duplicate_policy="coalesce-identical")
        run = SimpleNamespace(evaluation_for=lambda name: self.root / name)
        with patch.object(d, "_reserve_evaluation_output") as reserve, patch.object(d, "_audit_schedules") as audit, \
                patch.object(d, "_run_quality_evaluation", return_value={}) as evaluate, \
                patch.object(d, "_aggregate_evaluation") as aggregate:
            d._evaluate(run, args)
        reserve.assert_called_once_with(run, duplicate_policy="coalesce-identical")
        audit.assert_called_once_with(run, duplicate_policy="coalesce-identical")
        self.assertEqual(evaluate.call_count, 3)
        aggregate.assert_called_once_with(run, {"basic_vs_full": {}, "no_patient_vs_full": {}, "no_population_vs_full": {}},
                                         duplicate_policy="coalesce-identical")

    def test_aggregate_verifies_policy_twice_and_publishes_bound_duplicate_metadata(self):
        case_ids, audit, summaries, input_root = self.prepare_debug_aggregate()
        source_paths = [path for path in input_root.rglob("*") if path.is_file()]
        source_paths += [path for result in self.results.values() for path in result.glob("training_log_*.txt")]
        before = {path: path.read_bytes() for path in source_paths}
        with patch.object(d, "_validation_ids", return_value=case_ids), \
                patch.object(d, "_verify_schedule_audit", wraps=d._verify_schedule_audit) as verify:
            d._aggregate_evaluation(self.run, summaries, duplicate_policy="coalesce-identical")
        self.assertEqual(verify.call_count, 2)
        self.assertEqual([call.kwargs for call in verify.call_args_list], [{"duplicate_policy": "coalesce-identical"}] * 2)
        summary = d._load_json(self.run.report_root / "summary.json")
        completion = d._load_json(self.run.report_root / "completion.json")
        report = (self.run.report_root / "comparison.md").read_text(encoding="utf-8")
        self.assertEqual(summary["schedule_duplicate_policy"], "coalesce-identical")
        self.assertEqual(summary["training_resume_status"], "unverified")
        self.assertEqual(summary["schedule_duplicate_summary"], {
            name: {key: value for key, value in item.items() if key != "occurrences"}
            for name, item in audit["duplicate_audit"].items()
        })
        self.assertIs(completion["complete"], True)
        self.assertEqual(completion["schedule_duplicate_policy"], "coalesce-identical")
        self.assertEqual(completion["coalesced_schedule_occurrences"], 4)
        self.assertEqual(completion["training_resume_status"], "unverified")
        for field, filename in (("summary_sha256", "summary.json"), ("comparison_sha256", "comparison.md"),
                                ("schedule_audit_sha256", "schedule_audit.json")):
            self.assertEqual(completion[field], d._sha256(self.run.report_root / filename))
        self.assertEqual(completion["pair_summary_sha256"], {
            name: d._sha256(self.run.evaluation_for(name) / "summary.json") for name in summaries})
        self.assertIn("Schedule duplicate policy: `coalesce-identical`", report)
        self.assertIn("4 extra identical log occurrences coalesced", report)
        self.assertIn("Training resume continuity, optimizer steps and checkpoint lineage remain unverified.", report)
        self.assertIn("Neither proves the number of executed optimizer steps.", report)
        self.assertIn("| M3 w/o Level 2 | 250 | 252 | 2 | 689 / 1075 | 125000 / 126000 |", report)
        self.assertIn("N/A", report)
        self.assertEqual(before, {path: path.read_bytes() for path in source_paths})

    def test_aggregate_default_strict_rejects_coalesced_audit_without_publication(self):
        case_ids, _, summaries, _ = self.prepare_debug_aggregate()
        with patch.object(d, "_validation_ids", return_value=case_ids):
            with self.assertRaises(d.DownstreamAblationError):
                d._aggregate_evaluation(self.run, summaries)
        for filename in ("comparison.md", "summary.json", "completion.json"):
            self.assertFalse((self.run.report_root / filename).exists())

    def test_aggregate_second_verification_blocks_publication_after_log_mutation(self):
        case_ids, _, summaries, _ = self.prepare_debug_aggregate()
        real_verify = d._verify_schedule_audit
        observed_policies = []

        def verify_with_late_debug_mutation(payload, *, duplicate_policy="error"):
            observed_policies.append(duplicate_policy)
            if len(observed_policies) == 2:
                with (self.results[d.FULL_TRAINER] / "training_log_01.txt").open("a", encoding="utf-8") as handle:
                    handle.write("DEBUG: simulated late input mutation\n")
            return real_verify(payload, duplicate_policy=duplicate_policy)

        with patch.object(d, "_validation_ids", return_value=case_ids), \
                patch.object(d, "_verify_schedule_audit", side_effect=verify_with_late_debug_mutation):
            with self.assertRaisesRegex(d.DownstreamAblationError, "logs changed"):
                d._aggregate_evaluation(self.run, summaries, duplicate_policy="coalesce-identical")
        self.assertEqual(observed_policies, ["coalesce-identical", "coalesce-identical"])
        for filename in ("comparison.md", "summary.json", "completion.json"):
            self.assertFalse((self.run.report_root / filename).exists())

    def test_invalid_epoch_and_count_domains(self):
        result = self.results[d.BASIC_TRAINER]
        for epoch, applied, samples in [(250, 1, 500), (-1, 1, 500), (1, 0, 0), (1, -1, 500), (1, 501, 500)]:
            with self.subTest(epoch=epoch, applied=applied, samples=samples):
                self.write_log(result, [self.record(epoch, applied, samples)])
                with self.assertRaises(d.DownstreamAblationError):
                    d._schedule_records_from_result(result)

    def test_missing_epoch_and_zero_total_and_schedule_mismatch_fail(self):
        for variant in ("missing", "zero", "mismatch"):
            with self.subTest(variant=variant):
                for result in self.results.values():
                    self.write_log(result)
                self.run.report_root = self.root / variant
                target = self.results[d.NO_PATIENT_TRAINER]
                if variant == "missing":
                    self.write_log(target, [self.record(epoch) for epoch in range(249)])
                elif variant == "zero":
                    self.write_log(target, [self.record(epoch, 0) for epoch in range(250)])
                else:
                    self.write_log(target, [self.record(epoch, 3) for epoch in range(250)])
                with self.assertRaises(d.DownstreamAblationError):
                    self.audit()
                self.assertFalse((self.run.report_root / "schedule_audit.json").exists())

    def test_log_mutation_and_added_log_are_detected(self):
        result = self.results[d.BASIC_TRAINER]
        contract = d._schedule_log_contract(result)
        path = result / "training_log_01.txt"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("a later diagnostic line\n")
        with self.assertRaisesRegex(d.DownstreamAblationError, "logs changed"):
            d._schedule_records_from_result(result, log_contract=contract)
        contract = d._schedule_log_contract(result)
        (result / "training_log_02.txt").write_text("later log\n", encoding="utf-8")
        with self.assertRaisesRegex(d.DownstreamAblationError, "logs changed"):
            d._verify_schedule_log_contract(result, contract)

    def test_audit_content_and_post_audit_source_mutation_fail(self):
        audit = self.audit()
        altered = copy.deepcopy(audit)
        altered["total_cp_events"] += 1
        with self.assertRaisesRegex(d.DownstreamAblationError, "total-count"):
            d._verify_schedule_audit(altered)
        path = self.results[d.FULL_TRAINER] / "training_log_01.txt"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("changed after audit\n")
        with self.assertRaisesRegex(d.DownstreamAblationError, "logs changed"):
            d._verify_schedule_audit(audit)

    def test_output_reservation_and_report_publication_never_clobber(self):
        d._reserve_evaluation_output(self.run)
        sentinel = self.run.report_root / "comparison.md"
        sentinel.write_text("existing user result", encoding="utf-8")
        with self.assertRaises(d.DownstreamAblationError):
            d._reserve_evaluation_output(self.run)
        with self.assertRaises(d.DownstreamAblationError):
            d._publish_report_text(sentinel, "replacement")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing user result")
        self.assertFalse(list(self.run.report_root.glob("*.tmp")))

    def test_new_cli_output_only_redirects_evaluation_and_keeps_dataset730(self):
        output = self.root / "new_reports"
        args = d.build_parser().parse_args(["evaluate", "--project", str(d.PROJECT_ROOT), "--evaluation-output", str(output)])
        run = d._make_layout(args)
        self.assertEqual(run.report_root, output.resolve())
        self.assertEqual(run.evaluation_for("basic_vs_full"), output / "basic_vs_full")
        self.assertEqual(run.dataset_id, 730)
        self.assertNotIn(str(output), str(run.bank_for("no_patient")))
        self.assertFalse(output.exists())

    def test_missing_size_bin_and_undefined_value_are_na(self):
        summary = {"case_tumor_dice": {"basic_mean": None}, "lesion_dice_quality": {"basic_mean": None},
                   "criteria": {"dice_ge_0p10": {"basic_cp": {"precision": None}, "by_size": {}}}}
        value = d._method_from_pair(summary, "basic")
        self.assertIsNone(value["criteria"]["dice_ge_0p10"]["le_10mm_recall"])
        self.assertEqual(d._format_number(value["tumor_dice"]), "N/A")
        with self.assertRaises(d.DownstreamAblationError):
            d._format_number(float("nan"))


class DownstreamPairProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="debug_downstream_provenance_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = ["case_2", "case_10"]
        self.definition = {"outer_fold": 0, "dataset_id": 730, "thresholds": [0.1, 0.25]}
        self.gt = self.files("gt")
        self.full = self.files("full")
        self.pairs = {}
        for name, trainer in (("basic_vs_full", d.BASIC_TRAINER), ("no_patient_vs_full", d.NO_PATIENT_TRAINER), ("no_population_vs_full", d.NO_POPULATION_TRAINER)):
            self.pairs[name] = {"evaluation_contract": self.contract(trainer, self.files(name))}

    def files(self, label):
        folder = self.root / label
        folder.mkdir()
        result = {}
        for case_id in self.ids:
            path = folder / f"{case_id}.bytes"
            path.write_text(f"ISOLATED DEBUG FIXTURE ONLY {label} {case_id}", encoding="utf-8")
            result[case_id] = path
        return result

    def contract(self, trainer, basic, *, full=None, ids=None, definition=None):
        return build_evaluation_contract(cohort=self.ids if ids is None else ids,
            ground_truth_files=self.gt, prediction_files={"basic": basic, "hier": self.full if full is None else full},
            trainers={"basic": trainer, "hier": d.FULL_TRAINER}, evaluation_definition=self.definition if definition is None else definition)

    def test_three_pairs_share_verified_full_inputs(self):
        identity = d._validate_pairwise_provenance(self.pairs)
        self.assertEqual(identity["cohort"], self.ids)

    def test_changed_full_hash_is_rejected_even_with_equal_metric_means(self):
        changed = copy.deepcopy(self.pairs)
        changed["no_patient_vs_full"] = {"evaluation_contract": self.contract(d.NO_PATIENT_TRAINER, self.files("other_basic"), full=self.files("other_full"))}
        for summary in changed.values():
            summary["case_tumor_dice"] = {"hier_mean": 0.6722}
        with self.assertRaisesRegex(d.DownstreamAblationError, "identity mismatch"):
            d._validate_pairwise_provenance(changed)

    def test_cohort_order_definition_and_trainer_mismatch_fail(self):
        for variant in ("order", "definition", "trainer"):
            with self.subTest(variant=variant):
                altered = copy.deepcopy(self.pairs)
                kwargs = {"ids": list(reversed(self.ids))} if variant == "order" else ({"definition": {**self.definition, "thresholds": [0.5]}} if variant == "definition" else {})
                trainer = d.BASIC_TRAINER if variant == "trainer" else d.NO_PATIENT_TRAINER
                altered["no_patient_vs_full"]["evaluation_contract"] = self.contract(trainer, self.files(variant), **kwargs)
                with self.assertRaises(d.DownstreamAblationError):
                    d._validate_pairwise_provenance(altered)

    def test_original_prediction_mutation_is_rejected(self):
        self.full[self.ids[0]].write_text("changed debug fixture", encoding="utf-8")
        with self.assertRaises(ValueError):
            d._validate_pairwise_provenance(self.pairs)

    def test_missing_legacy_provenance_is_not_accepted(self):
        altered = copy.deepcopy(self.pairs)
        altered["no_patient_vs_full"].pop("evaluation_contract")
        with self.assertRaisesRegex(d.DownstreamAblationError, "Missing evaluation provenance"):
            d._validate_pairwise_provenance(altered)

    def test_pair_completion_and_artifact_hashes_required(self):
        output = self.root / "completed_pair"
        output.mkdir()
        summary = self.pairs["basic_vs_full"]
        path = output / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(d.DownstreamAblationError):
            d._verified_pair_summary(output)
        completion = {"format": "online_eval_completion_v1", "complete": True,
                      "summary_sha256": d._sha256(path), "evaluation_contract_sha256": summary["evaluation_contract"]["contract_sha256"],
                      "outputs": {"summary.json": d._sha256(path)}}
        (output / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
        self.assertEqual(d._verified_pair_summary(output), summary)
        path.write_text(json.dumps({**summary, "tampered": True}), encoding="utf-8")
        with self.assertRaisesRegex(d.DownstreamAblationError, "incomplete or changed"):
            d._verified_pair_summary(output)


if __name__ == "__main__":
    unittest.main()
