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

    def audit(self):
        d._reserve_evaluation_output(self.run)
        with patch.object(d, "_model_dir", side_effect=lambda run, trainer: self.results[trainer]):
            d._audit_schedules(self.run)
        return d._load_json(self.run.report_root / "schedule_audit.json")

    def test_valid_exact_250_all_four_and_zero_event_epoch_allowed(self):
        audit = self.audit()
        self.assertIs(audit["matched"], True)
        self.assertEqual(audit["total_samples"], 125000)
        self.assertEqual(audit["total_cp_events"], 498)
        self.assertEqual(len(audit["log_inputs"]), 4)
        self.assertEqual(len(audit["epoch_records"]["basic"]), 250)
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
