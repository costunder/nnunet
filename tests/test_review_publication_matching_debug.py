"""CPU DEBUG regression of publication crash windows, early guards and matching.

No patient dataset, actual training, external command, or original output is used.
The contract fixture has genuine small numeric preprocessing NPZs; checkpoint
files are hash-only DEBUG artifacts and are never deserialized as trained models.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_curriculum_bank_contract as fixtures
from custom_trainers import onlinecp_curriculum_contract as contract_module
from tools import online_cp_curriculum as publisher
from tools import online_eval_v2 as evaluator
from tools import nnunet as offline
import run as entrypoint


class PublicationTransactionDebugTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CurriculumBankContractDebugTests("test_exact_contract")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.output = self.fixture.bank / "feedback_contract.json"

    def verify(self, payload):
        return contract_module.verify_curriculum_contract_payload(payload, self.fixture.index,
            curriculum_sha256="debug-policy", expected_candidate_count=128,
            dataset_name="Dataset900_Debug", nnunet_fold=0,
            preprocessed_root=self.fixture.pre.parent)

    def publish(self, verify=None):
        return publisher._publish_verified_contract(self.output, self.fixture.contract, verify or self.verify)

    def snapshot(self, path):
        return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}

    def test_payload_can_be_verified_before_final_and_environment_is_untouched(self):
        with patch.dict(os.environ, {"nnUNet_preprocessed": "DEBUG_UNRELATED_CONTEXT"}):
            result = self.verify(self.fixture.contract)
            self.assertEqual(result["verified_configuration"], "3d_fullres")
            self.publish()
            self.assertEqual(os.environ["nnUNet_preprocessed"], "DEBUG_UNRELATED_CONTEXT")
        self.assertTrue(self.output.is_file())

    def test_identical_final_reuse_leaves_all_original_bytes_unchanged(self):
        self.publish()
        before = self.snapshot(self.fixture.bank)
        self.assertEqual(self.publish(), self.output)
        self.assertEqual(before, self.snapshot(self.fixture.bank))
        receipt = next((self.fixture.bank / "contract_publication_history").glob("*/publication.json"))
        self.assertEqual(json.loads(receipt.read_text())["outcome"], "published")

    def test_existing_final_changed_payload_or_dependency_is_not_overwritten(self):
        self.publish()
        original = self.output.read_bytes()
        changed = copy.deepcopy(self.fixture.contract)
        changed["curriculum_sha256"] = "another-policy"
        with self.assertRaisesRegex(ValueError, "different inputs"):
            publisher._publish_verified_contract(self.output, changed, self.verify)
        self.assertEqual(self.output.read_bytes(), original)
        target = self.fixture.data_root / "a.pkl"
        target.write_bytes(target.read_bytes() + b"DEBUG changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.publish()
        self.assertEqual(self.output.read_bytes(), original)

    def test_malformed_unowned_final_is_preserved(self):
        self.output.write_bytes(b'{"format":')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.output.read_bytes(), b'{"format":')
        self.assertFalse((self.fixture.bank / "contract_publication_history").exists())

    def test_strict_payload_reader_rejects_duplicate_and_nonfinite_json(self):
        for data in ('{"a":1,"a":2}', '{"a":NaN}', '[]'):
            with self.subTest(data=data):
                self.output.write_text(data, encoding="utf-8")
                with self.assertRaises(ValueError):
                    contract_module.read_curriculum_contract_payload(self.output)

    def test_serialization_and_sync_failures_do_not_create_final(self):
        original = publisher._write_exclusive_json
        for failure in ("partial_write", "after_staged_sync"):
            def fail(path, payload):
                if path.name == "contract.json":
                    if failure == "partial_write":
                        with path.open("x") as handle:
                            handle.write('{"format":')
                    else:
                        original(path, payload)
                    raise OSError("DEBUG staged write failure")
                return original(path, payload)
            with self.subTest(failure=failure), patch.object(publisher, "_write_exclusive_json", side_effect=fail):
                with self.assertRaisesRegex(OSError, "DEBUG"):
                    self.publish()
            self.assertFalse(self.output.exists())
        original_attempts = self.snapshot(self.fixture.bank / "contract_publication_history")
        self.publish()
        after = self.snapshot(self.fixture.bank / "contract_publication_history")
        self.assertTrue(original_attempts.items() <= after.items())

    def test_staging_validation_failure_is_not_published_and_is_retryable(self):
        with self.assertRaisesRegex(ValueError, "DEBUG verifier"):
            self.publish(verify=lambda payload: (_ for _ in ()).throw(ValueError("DEBUG verifier")))
        self.assertFalse(self.output.exists())
        self.publish()
        self.assertTrue(self.output.exists())

    def test_no_clobber_install_failure_preserves_stage_without_final(self):
        with patch.object(publisher.os, "link", side_effect=OSError("DEBUG unsupported filesystem / ENOSPC")):
            with self.assertRaisesRegex(OSError, "DEBUG"):
                self.publish()
        self.assertFalse(self.output.exists())
        staged = next((self.fixture.bank / "contract_publication_history").glob("*/contract.json"))
        before = staged.read_bytes()
        self.publish()
        self.assertEqual(staged.read_bytes(), before)

    def test_installed_final_survives_lost_ack_and_reconciles_without_rewrite(self):
        original_link = os.link
        def installed_then_failed(source, destination):
            original_link(source, destination)
            raise OSError("DEBUG lost acknowledgement after install")
        with patch.object(publisher.os, "link", side_effect=installed_then_failed):
            with self.assertRaisesRegex(OSError, "DEBUG"):
                self.publish()
        before = self.snapshot(self.fixture.bank)
        self.publish()
        self.assertEqual(before, self.snapshot(self.fixture.bank))

    def test_final_verification_failure_and_missing_receipt_can_reconcile(self):
        calls = []
        def fail_final(payload):
            calls.append(True)
            if len(calls) == 3:
                raise OSError("DEBUG transient final verification")
            return self.verify(payload)
        with self.assertRaisesRegex(OSError, "DEBUG"):
            self.publish(fail_final)
        before = self.output.read_bytes()
        self.publish()
        self.assertEqual(before, self.output.read_bytes())
        self.assertFalse(list((self.fixture.bank / "contract_publication_history").glob("*/publication.json")))

    def test_receipt_write_failure_leaves_verified_final_retryable(self):
        original = publisher._write_exclusive_json
        def fail(path, payload):
            if path.name == "publication.json":
                raise OSError("DEBUG receipt ENOSPC")
            return original(path, payload)
        with patch.object(publisher, "_write_exclusive_json", side_effect=fail):
            with self.assertRaisesRegex(OSError, "DEBUG"):
                self.publish()
        before = self.snapshot(self.fixture.bank)
        self.publish()
        self.assertEqual(before, self.snapshot(self.fixture.bank))

    def test_two_concurrent_identical_writers_commit_one_final(self):
        barrier = threading.Barrier(2)
        local = threading.local()
        def verify(payload):
            local.calls = getattr(local, "calls", 0) + 1
            if local.calls == 2:
                barrier.wait(timeout=15)
            return self.verify(payload)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: self.publish(verify), range(2)))
        self.assertEqual(outcomes, [self.output, self.output])
        receipts = [json.loads(p.read_text()) for p in
                    (self.fixture.bank / "contract_publication_history").glob("*/publication.json")]
        self.assertEqual(sorted(r["outcome"] for r in receipts), ["concurrent_verified_reuse", "published"])
        self.verify(contract_module.read_curriculum_contract_payload(self.output))


class ValidOnlyMatchingDebugTests(unittest.TestCase):
    @staticmethod
    def matrices(gt, pred, intersections):
        gt, pred, inter = np.asarray(gt), np.asarray(pred), np.asarray(intersections).reshape(len(gt), len(pred))
        return evaluator.PairMatrices(gt, pred, inter, 2 * inter / (gt[:, None] + pred[None, :]),
            inter / (gt[:, None] + pred[None, :] - inter), inter / gt[:, None], inter / pred[None, :])

    def test_feasible_counterexample_all_permutations_and_size_bin_attribution(self):
        sizes = np.array([100, 200])
        inter = np.array([[45, 36], [45, 0]])
        criterion = evaluator.Criterion("debug", "DEBUG", "dice", .25)
        for rows in ([0, 1], [1, 0]):
            for cols in ([0, 1], [1, 0]):
                matrices = self.matrices(sizes[rows], sizes[cols], inter[np.ix_(rows, cols)])
                result = evaluator.match_components(matrices, criterion)
                mapped = [(rows[i], cols[j]) for i, j in result.gt_to_pred.items()]
                self.assertEqual(mapped, [(0, 0)])
                self.assertEqual(result.tp, 1)
                self.assertAlmostEqual(sum(matrices.dice[i, j] for i, j in result.gt_to_pred.items()), .45)
        diameters = (6 * sizes * 5 / np.pi) ** (1 / 3)
        self.assertLess(diameters[0], 10)
        self.assertGreater(diameters[1], 10)

    def test_exact_threshold_and_zero_overlap(self):
        matrices = self.matrices([100], [100], [[25]])
        for threshold, tp in ((.25, 1), (np.nextafter(.25, 1), 0)):
            self.assertEqual(evaluator.match_components(matrices, evaluator.Criterion("d", "D", "dice", threshold)).tp, tp)
        zero = self.matrices([100], [100], [[0]])
        self.assertEqual(evaluator.match_components(zero, evaluator.Criterion("d", "D", "dice", 0)).tp, 0)

    def test_empty_and_rectangular(self):
        for gt, pred in (([], [100]), ([100], []), ([], [])):
            self.assertEqual(evaluator.match_components(self.matrices(gt, pred, []), evaluator.CRITERIA[1]).tp, 0)
        result = evaluator.match_components(self.matrices([100, 100], [100], [[50], [25]]), evaluator.CRITERIA[2])
        self.assertEqual(dict(result.gt_to_pred), {0: 0})

    def test_primary_cardinality_beats_better_single_pair(self):
        matrices = self.matrices([100, 100], [100, 100], [[80, 10], [10, 0]])
        self.assertEqual(evaluator.match_components(matrices, evaluator.CRITERIA[1]).tp, 2)

    def test_legacy_and_max_dice_semantics_unchanged(self):
        matrices = self.matrices([100, 200], [100, 200], [[45, 36], [45, 0]])
        self.assertEqual(dict(evaluator.match_components(matrices, evaluator.CRITERIA[0]).gt_to_pred), {0: 1, 1: 0})
        self.assertEqual(dict(evaluator.max_dice_match(matrices).gt_to_pred), {0: 1, 1: 0})

    def test_equal_quality_tie_only_requires_optimal_cardinality_and_quality(self):
        matrices = self.matrices([100, 100], [100, 100], [[25, 25], [25, 25]])
        result = evaluator.match_components(matrices, evaluator.CRITERIA[2])
        self.assertEqual(result.tp, 2)
        self.assertEqual(sum(matrices.dice[i, j] for i, j in result.gt_to_pred.items()), .5)
        self.assertIn("nonunique", evaluator.MATCHING_TIE_POLICY)
        self.assertEqual(evaluator.VERSION, "online_basic_hiercp_evaluation_v5")


class UnsupportedOfflineDebugTests(unittest.TestCase):
    def test_direct_train_rejects_before_planning_or_baseline(self):
        with patch.object(offline, "plan_ready") as planning, patch.object(offline, "train_one") as training:
            with self.assertRaisesRegex(offline.PipelineError, "Unsupported legacy offline"):
                offline.train(None, None, {}, [0], "cpu", False)
            planning.assert_not_called()
            training.assert_not_called()

    def test_native_cli_rejects_before_reading_configuration(self):
        for target in ("train", "all"):
            argv = ["DEBUG", target, "--project-root", "DEBUG", "--medical-root", "DEBUG",
                    "--workspace", "DEBUG", "--config", "DEBUG"]
            with self.subTest(target=target), patch("sys.argv", argv), patch.object(offline, "load_json") as read:
                with self.assertRaisesRegex(offline.PipelineError, "Unsupported legacy offline"):
                    offline.main()
                read.assert_not_called()

    def test_top_level_rejects_before_work_directory_or_subprocess(self):
        for target in ("full", "nnunet-all", "nnunet-train"):
            parser = SimpleNamespace(parse_args=lambda: SimpleNamespace(target=target))
            with self.subTest(target=target), patch.object(entrypoint, "parser", return_value=parser), \
                 patch.object(entrypoint, "medical_root") as medical, patch.object(entrypoint, "execute") as execute:
                with self.assertRaises(SystemExit):
                    entrypoint.main()
                medical.assert_not_called()
                execute.assert_not_called()

    def test_historical_readonly_evaluation_not_rejected(self):
        for target in ("evaluate", "status", "check", "plan", "prepare"):
            offline.require_supported_offline_target(target)


if __name__ == "__main__":
    unittest.main()
