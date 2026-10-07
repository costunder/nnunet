"""UNIT independent-arm ownership; no CT, model execution or quality claims."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hiercp_v1x import owned_continuation as owned
from hiercp_v1x import u_bridge_continuation as publication
from tests.test_u_bridge_continuation import fixture, write_json, byte_inventory


def reseal_receipt(path, receipt):
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = publication._digest(receipt)
    write_json(path, receipt)


class OwnedContinuationTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix="UNIT_owned_continuation_", dir=Path(__file__).resolve().parents[1])

    def clone(self, base, arm="selected"):
        source = fixture(base)
        target = base / (arm + "_independent")
        publication.prepare_continuation(source, target, arm)
        return source, target

    def test_existing_owner_resumes_while_historical_source_lock_is_active(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            write_json(source / ".pipeline.lock", dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT"))
            write_json(source / "data/.data.lock", dict(host="UNIT_other_host", pid=123, token="UNIT"))
            before = byte_inventory(source)
            with patch.object(publication, "_inactive", side_effect=AssertionError("Old lock must not be inspected")):
                report = owned.verify_owned_continuation(source, target, "selected")
            self.assertTrue(report["resumed"])
            self.assertFalse(report["source_lock_dependency"])
            self.assertTrue(report["own_immutable_bytes_verified"])
            self.assertEqual(byte_inventory(source), before)

    def test_advanced_checkpoint_is_kept_and_never_replaced_from_old_source(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            original_source = (source / "selected/checkpoint_latest.pt").read_bytes()
            latest = target / "selected/checkpoint_latest.pt"
            latest.write_bytes(b"UNIT advanced independent checkpoint")
            original_prepare = publication.prepare_continuation
            with owned.independent_continuation_execution():
                self.assertIsNot(publication.prepare_continuation, original_prepare)
                report = publication.prepare_continuation(source, target, "selected")
            self.assertIs(publication.prepare_continuation, original_prepare)
            self.assertTrue(report["reused"])
            self.assertEqual(latest.read_bytes(), b"UNIT advanced independent checkpoint")
            self.assertEqual((source / "selected/checkpoint_latest.pt").read_bytes(), original_source)

    def test_fresh_migration_still_rejects_active_original_source(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "new_independent"
            write_json(source / ".pipeline.lock", dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT"))
            with owned.independent_continuation_execution(), self.assertRaisesRegex(RuntimeError, "still active"):
                publication.prepare_continuation(source, target, "native")
            self.assertFalse(target.exists())

    def test_selected_and_native_own_independent_namespaces_then_resume_concurrently(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base)
            selected, native = base / "selected_independent", base / "native_independent"
            publication.prepare_continuation(source, selected, "selected")
            publication.prepare_continuation(source, native, "native")
            write_json(source / ".pipeline.lock", dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT"))
            write_json(selected / ".pipeline.lock", dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT_selected"))
            write_json(native / ".pipeline.lock", dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT_native"))
            with owned.independent_continuation_execution():
                one = publication.prepare_continuation(source, selected, "selected")
                two = publication.prepare_continuation(source, native, "native")
            self.assertNotEqual(one["data_root"], two["data_root"])
            self.assertNotEqual(one["destination_root"], two["destination_root"])
            self.assertFalse((native / "selected").exists())
            self.assertTrue((selected / ".pipeline.lock").is_file())
            self.assertTrue((native / ".pipeline.lock").is_file())

    def test_immutable_initial_cache_and_arm_identity_tamper_are_rejected(self):
        for name in ("initial.pt", "selected/training_identity.json",
                     "data/canonical_local/" + "a" * 64 + ".pt"):
            with self.subTest(name=name), self.directory() as directory:
                source, target = self.clone(Path(directory))
                (target / name).write_bytes(b"UNIT tampered immutable bytes")
                with self.assertRaisesRegex(ValueError, "immutable bytes"):
                    owned.verify_owned_continuation(source, target, "selected")

    def test_missing_latest_never_falls_back_to_initial_state(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            (target / "selected/checkpoint_latest.pt").unlink()
            with self.assertRaisesRegex(ValueError, "saved progress is missing"):
                owned.verify_owned_continuation(source, target, "selected")

    def test_wrong_arm_source_or_destination_identity_is_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source, target = self.clone(base)
            for expected_source, arm in ((source, "native"), (base / "different_source", "selected")):
                with self.subTest(source=expected_source, arm=arm), self.assertRaisesRegex(ValueError, "identity differs"):
                    owned.verify_owned_continuation(expected_source, target, arm)

    def test_receipt_tamper_or_unsafe_resealed_inventory_is_rejected(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            path = target / publication.RECEIPT
            receipt = publication._read(path)
            receipt["resumed"] = False
            write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, "receipt identity"):
                owned.verify_owned_continuation(source, target, "selected")
            receipt["files"][0]["path"] = "../outside.txt"
            reseal_receipt(path, receipt)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                owned.verify_owned_continuation(source, target, "selected")

    def test_resealed_receipt_cannot_omit_required_initial_or_calibration_record(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            path = target / publication.RECEIPT
            receipt = publication._read(path)
            receipt["files"] = [record for record in receipt["files"] if record["path"] != "initial.pt"]
            reseal_receipt(path, receipt)
            with self.assertRaisesRegex(ValueError, "missing sealed"):
                owned.verify_owned_continuation(source, target, "selected")

    def test_execution_alias_restores_when_owned_checkpoint_validation_fails(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            (target / "calibration.json").write_bytes(b"UNIT corrupt calibration")
            original_prepare = publication.prepare_continuation
            with self.assertRaisesRegex(ValueError, "immutable bytes"):
                with owned.independent_continuation_execution():
                    publication.prepare_continuation(source, target, "selected")
            self.assertIs(publication.prepare_continuation, original_prepare)

    def test_mutable_checkpoint_symlink_is_rejected_even_when_arm_can_progress(self):
        with self.directory() as directory:
            source, target = self.clone(Path(directory))
            latest = target / "selected/checkpoint_latest.pt"
            original = Path.is_symlink
            def simulated(path):
                return path == latest or original(path)
            with patch.object(Path, "is_symlink", simulated), self.assertRaisesRegex(ValueError, "symlink"):
                owned.verify_owned_continuation(source, target, "selected")

    def test_new_native_checkpoint_is_checked_even_when_not_in_original_clone_receipt(self):
        with self.directory() as directory:
            base = Path(directory); source, target = self.clone(base, "native")
            (target / "native").mkdir()
            latest = target / "native/checkpoint_latest.pt"
            latest.write_bytes(b"UNIT newly saved native checkpoint")
            self.assertTrue(owned.verify_owned_continuation(source, target, "native")["resumed"])
            original = Path.is_symlink
            def simulated(path):
                return path == latest or original(path)
            with patch.object(Path, "is_symlink", simulated), self.assertRaisesRegex(ValueError, "symlink"):
                owned.verify_owned_continuation(source, target, "native")


if __name__ == "__main__":
    unittest.main()
