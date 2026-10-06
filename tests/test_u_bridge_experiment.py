"""UNIT execution-contract tests; no CT load, neural model, or GPU run."""
from contextlib import redirect_stderr, redirect_stdout
import copy
from io import StringIO
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hiercp_v1x.u_bridge_experiment import (
    digest, joint_calibration, lock, read, write_new, prepared_data_path,
)
from tools.run_v18_u_bridge import parse


def unit_directory():
    return TemporaryDirectory(prefix="UNIT_u_bridge_contract_",
                              dir=Path(__file__).resolve().parents[1])


def arguments():
    return ["--gpu", "0", "--arm", "both", "--experiment", "UNIT_new_experiment",
            "--inventory", "UNIT_signed_inventory.json", "--workers", "4",
            "--batch-candidates", "2", "4", "8", "--cuda-gib", "12",
            "--rss-gib", "32", "--resident-gib", "8",
            "--validation-local-chunk", "16"]


def debug_arguments():
    return ["--debug", "--debug-fixture", "UNIT_fixture", "--debug-config", "UNIT_config.json",
            "--debug-source", "UNIT_original_source", "--debug-bank", "UNIT_bank.pt",
            "--debug-epochs", "2"]


class ExperimentContractTests(unittest.TestCase):
    def assert_rejected(self, argv, message):
        stream = StringIO()
        with redirect_stderr(stream), self.assertRaises(SystemExit) as raised:
            parse(argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, stream.getvalue())

    def test_production_and_debug_require_explicit_separate_inputs(self):
        production = parse(arguments() + ["--baseline", "UNIT_preserved_baseline"])
        self.assertFalse(production.debug)
        self.assertIsNone(production.debug_epochs)
        self.assertEqual(production.batch_candidates, [2, 4, 8])
        debug = parse(arguments() + debug_arguments())
        self.assertTrue(debug.debug)
        self.assertEqual(debug.debug_epochs, 2)
        self.assertIsNone(debug.baseline)
        self.assert_rejected(arguments(), "Production needs --baseline")
        self.assert_rejected(arguments() + ["--baseline", "UNIT_baseline", "--debug-epochs", "1"],
                             "cannot use DEBUG inputs/epochs")
        self.assert_rejected(arguments() + ["--debug"], "DEBUG requires")
        self.assert_rejected(arguments() + debug_arguments()[:-1] + ["40"], "explicit1..2epochs")

    def test_explicit_physical_batches_and_resources_cannot_silently_fallback(self):
        for option, values, message in (
            ("--workers", ["1"], "workers>=2"),
            ("--rss-gib", ["8"], "resident<RSS"),
            ("--cuda-gib", ["0"], "positive limits"),
            ("--validation-local-chunk", ["0"], "L0 chunk"),
            ("--batch-candidates", ["2", "2"], "Unique explicit positive physical batches"),
            ("--batch-candidates", ["0"], "Unique explicit positive physical batches"),
        ):
            with self.subTest(option=option, values=values):
                self.assert_rejected(arguments() + ["--baseline", "UNIT_baseline", option, *values], message)

    def test_matched_batch_is_common_accepted_and_maximizes_slower_arm_throughput(self):
        # Selected-only winner4 is rejected by native; selected's fast1 loses
        # to batch2 once native's measured bottleneck is included.
        reports = {
            "selected": {"reports": [
                dict(physical_batch=1, accepted=True, samples_per_second=100.),
                dict(physical_batch=2, accepted=True, samples_per_second=12.),
                dict(physical_batch=4, accepted=True, samples_per_second=150.)]},
            "native": {"reports": [
                dict(physical_batch=1, accepted=True, samples_per_second=10.),
                dict(physical_batch=2, accepted=True, samples_per_second=19.),
                dict(physical_batch=4, accepted=False, reason="UNIT measured budget rejection")]},
        }
        untouched = copy.deepcopy(reports)
        result = joint_calibration(reports, [1, 2, 4])
        self.assertEqual(result["physical_batch"], 2)
        self.assertEqual(result["accepted_common"], [1, 2])
        self.assertEqual(result["explicit_candidates"], [1, 2, 4])
        self.assertEqual(result["reports"], untouched)
        self.assertEqual(reports, untouched)

    def test_no_common_accepted_batch_fails_without_inventing_batch_one(self):
        reports = {
            "selected": {"reports": [dict(physical_batch=2, accepted=True, samples_per_second=4.)]},
            "native": {"reports": [dict(physical_batch=4, accepted=True, samples_per_second=5.)]},
        }
        original = copy.deepcopy(reports)
        with self.assertRaisesRegex(RuntimeError, "No explicit physical sample batch passed both arms"):
            joint_calibration(reports, [2, 4])
        self.assertEqual(reports, original)
        with self.assertRaisesRegex(ValueError, "Both real-arm measurements"):
            joint_calibration({"selected": reports["selected"]}, [2])

    def test_live_same_host_lock_is_refused_without_touching_owner(self):
        with unit_directory() as directory:
            path = Path(directory) / ".setup.lock"
            owner = dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT_live")
            write_new(path, owner)
            before = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "owner is still alive"):
                with lock(path):
                    self.fail("A live owner was displaced")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob("*.stale.*")), [])

    def test_unknown_host_or_pid_lock_is_refused_without_process_probe(self):
        with unit_directory() as directory:
            for position, owner in enumerate((
                dict(host=socket.gethostname() + "_UNIT_other", pid=424242, token="UNIT_foreign"),
                dict(host=socket.gethostname(), pid="424242", token="UNIT_unknown"),
            )):
                path = Path(directory) / (str(position) + ".lock")
                write_new(path, owner)
                before = path.read_bytes()
                with patch("psutil.pid_exists") as probe:
                    with self.assertRaisesRegex(RuntimeError, "another/unknown host"):
                        with lock(path):
                            self.fail("An unknown lock owner was displaced")
                    probe.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_dead_local_owner_is_preserved_as_receipt_and_only_own_lock_removed(self):
        with unit_directory() as directory:
            path = Path(directory) / ".setup.lock"
            stale = dict(host=socket.gethostname(), pid=424242, token="UNIT_dead")
            write_new(path, stale)
            receipt = path.with_name(path.name + ".stale.UNIT_dead")
            with patch("psutil.pid_exists", return_value=False) as probe, redirect_stdout(StringIO()):
                with lock(path):
                    probe.assert_called_once_with(424242)
                    self.assertEqual(read(receipt), stale)
                    active = read(path)
                    self.assertEqual(active["pid"], os.getpid())
                    self.assertNotEqual(active["token"], stale["token"])
            self.assertFalse(path.exists())
            self.assertEqual(read(receipt), stale)

    def test_existing_stale_receipt_cannot_be_overwritten(self):
        with unit_directory() as directory:
            path = Path(directory) / ".setup.lock"
            stale = dict(host=socket.gethostname(), pid=424242, token="UNIT_dead")
            write_new(path, stale)
            receipt = path.with_name(path.name + ".stale.UNIT_dead")
            write_new(receipt, {"UNIT_preserved_receipt": True})
            before = path.read_bytes(), receipt.read_bytes()
            with patch("psutil.pid_exists", return_value=False):
                with self.assertRaisesRegex(FileExistsError, "receipt already exists"):
                    with lock(path):
                        self.fail("An existing receipt was replaced")
            self.assertEqual((path.read_bytes(), receipt.read_bytes()), before)

    def test_lock_cleanup_preserves_replacement_owner_and_handles_body_exception(self):
        with unit_directory() as directory:
            path = Path(directory) / ".setup.lock"
            with self.assertRaisesRegex(ValueError, "UNIT deliberate body failure"):
                with lock(path):
                    raise ValueError("UNIT deliberate body failure")
            self.assertFalse(path.exists())
            replacement = dict(host=socket.gethostname(), pid=424242, token="UNIT_replacement")
            with lock(path):
                # Deliberate UNIT replacement of our own test file.
                path.write_text(json.dumps(replacement), encoding="utf8")
            self.assertEqual(read(path), replacement)

    def test_contract_hash_and_equality_survive_json_resume_roundtrip(self):
        content = dict(debug=False, epochs=40,
            samples=[dict(case_id="UNIT_case", positive_center=(4, 5, 6),
                          selected_centers=((0, 1, 2), (2, 3, 4)))],
            scope=dict(optional_roles=("source_liver_surface", "target_liver_surface")))
        canonical = json.loads(json.dumps(content, allow_nan=False))
        self.assertEqual(digest(content), digest(canonical))
        signed = dict(canonical, sha256=digest(canonical))
        with unit_directory() as directory:
            path = Path(directory) / "experiment.json"
            write_new(path, signed)
            self.assertEqual(read(path), signed)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, dict(signed, epochs=1))
            self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            digest({"UNIT_invalid_nonfinite_contract": float("nan")})

    def test_prepared_cache_is_explicit_disjoint_and_does_not_touch_old_results(self):
        with unit_directory() as directory:
            root = Path(directory)
            new = root / 'new'
            old = root / 'old'
            cache = old / 'data'
            cache.mkdir(parents=True)
            receipt = old / 'results.json'
            write_new(receipt, {'UNIT_preserved': True})
            before = receipt.read_bytes()
            self.assertEqual(prepared_data_path(new), new.resolve() / 'data')
            self.assertEqual(prepared_data_path(new, cache), cache.resolve())
            self.assertEqual(receipt.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                prepared_data_path(new, root)
            with self.assertRaisesRegex(ValueError, 'baseline'):
                prepared_data_path(new, cache, old)

    def test_prepared_cache_live_or_foreign_owner_refused_dead_local_owner_left_intact(self):
        with unit_directory() as directory:
            root = Path(directory)
            cache = root / 'old' / 'data'
            cache.mkdir(parents=True)
            owner_path = cache.parent / '.pipeline.lock'
            for owner in (
                dict(host=socket.gethostname(), pid=os.getpid(), token='UNIT_live'),
                dict(host='UNIT_foreign_host', pid=424242, token='UNIT_foreign'),
            ):
                owner_path.write_text(json.dumps(owner), encoding='utf8')
                with self.assertRaisesRegex(RuntimeError, 'may still be active'):
                    prepared_data_path(root / 'new', cache)
                self.assertEqual(read(owner_path), owner)
            dead = dict(host=socket.gethostname(), pid=424242, token='UNIT_dead')
            owner_path.write_text(json.dumps(dead), encoding='utf8')
            with patch('psutil.pid_exists', return_value=False):
                self.assertEqual(prepared_data_path(root / 'new', cache), cache.resolve())
            self.assertEqual(read(owner_path), dead)


if __name__ == "__main__":
    unittest.main(verbosity=2)
