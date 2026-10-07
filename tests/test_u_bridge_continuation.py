"""UNIT metadata/file-lifecycle fixtures, not CT or neural quality evidence."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from hiercp_v1x import u_bridge_continuation as migration


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def fixture(base, arm="selected"):
    source = base / "source"
    source.mkdir()
    data = source / "data"
    data.mkdir()
    baseline = base / "preserved_baseline"
    baseline.mkdir()
    contract = dict(format="v18_u_bridge_matched_experiment_v1", debug=True,
                    baseline=dict(baseline=str(baseline)), prepared_data_root=str(data),
                    unit_fixture="metadata_only_not_a_model")
    contract["sha256"] = migration._digest(contract)
    write_json(source / "experiment.json", contract)
    (source / "initial.pt").write_bytes(b"UNIT opaque copied initial state; not neural evidence")
    write_json(source / "calibration.json", dict(contract_sha256=contract["sha256"], physical_batch=1,
                                               reports=dict(selected={}, native={})))
    write_json(source / "calibration_selected.json", dict(unit=True))
    if arm is not None:
        (source / arm).mkdir()
        write_json(source / arm / "training_identity.json", dict(unit=True, identity_sha256="a" * 64))
        (source / arm / "checkpoint_latest.pt").write_bytes(b"UNIT opaque latest checkpoint")
        (source / arm / "checkpoint_best.pt").write_bytes(b"UNIT opaque best checkpoint")
        (source / arm / "curve.jsonl").write_text('{"unit":true}\n')
        (source / arm / "STOP_AFTER_BATCH").write_text("UNIT prior pause request")
    canonical = data / "canonical_local"
    canonical.mkdir()
    with zipfile.ZipFile(canonical / ("a" * 64 + ".pt"), "w") as archive:
        archive.writestr("archive/data.pkl", b"UNIT serialization publication fixture only")
        archive.writestr("archive/version", b"3\n")
    (canonical / ("b" * 64 + ".unfinished.tmp")).write_bytes(b"UNIT interrupted unpublished bytes")
    fields = data / "whole_case_fields/UNIT_case"
    fields.mkdir(parents=True)
    binding = dict(unit="metadata_only", case="UNIT_case")
    records = {}
    for name in ("depth", "occupied"):
        path = fields / (name + ".npy")
        path.write_bytes(b"UNIT exact payload bytes; not an actual CT array: " + name.encode())
        records[name] = dict(file=path.name, units="mm", file_bytes=path.stat().st_size,
                             sha256=migration._sha(path))
    metadata = dict(format="v18_exact_whole_case_distance_fields_v1", binding=binding,
                    binding_sha256=migration._digest(binding), fields=records)
    metadata["metadata_sha256"] = migration._digest(metadata)
    write_json(fields / "metadata.json", metadata)
    attempt = fields.parent / ".UNIT_case.attempt.UNIT"
    attempt.mkdir()
    (attempt / "depth.npy").write_bytes(b"UNIT incomplete attempt")
    upper = data / "upper_static/UNIT_case/c1_a1_1_1/source_raw"
    upper.mkdir(parents=True)
    (upper / "arrays.npz").write_bytes(b"UNIT exact static payload bytes")
    metadata = dict(format="v18_exact_original_static_upper_v1", kind="source_raw", binding=binding,
                    binding_sha256=migration._digest(binding), payload_bytes=(upper / "arrays.npz").stat().st_size,
                    payload_sha256=migration._sha(upper / "arrays.npz"))
    metadata["metadata_sha256"] = migration._digest(metadata)
    write_json(upper / "metadata.json", metadata)
    return source


def byte_inventory(root):
    return {p.relative_to(root).as_posix(): migration._sha(p)
            for p in root.rglob("*") if p.is_file()}


class ContinuationTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix="UNIT_continuation_", dir=Path(__file__).resolve().parents[1])

    def test_exact_sealed_bytes_and_mutable_arm_copies_preserve_source(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); before = byte_inventory(source)
            target = base / "independent"
            report = migration.prepare_continuation(source, target, "selected")
            self.assertTrue(report["resumed"])
            self.assertFalse(report["reused"])
            self.assertEqual(byte_inventory(source), before)
            self.assertEqual((target / "experiment.json").read_bytes(), (source / "experiment.json").read_bytes())
            self.assertEqual((target / "initial.pt").read_bytes(), (source / "initial.pt").read_bytes())
            self.assertFalse(os.path.samefile(target / "selected/checkpoint_latest.pt", source / "selected/checkpoint_latest.pt"))
            self.assertFalse((target / "selected/STOP_AFTER_BATCH").exists())
            self.assertFalse((target / "native").exists())
            self.assertFalse(list(target.rglob("*.tmp")))
            self.assertFalse((target / "data/whole_case_fields/.UNIT_case.attempt.UNIT").exists())

    def test_completed_cache_hardlinks_and_new_namespace_are_independent(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            report = migration.prepare_continuation(source, target, "selected")
            old = source / "data/canonical_local" / ("a" * 64 + ".pt")
            new = target / "data/canonical_local" / old.name
            self.assertTrue(os.path.samefile(old, new))
            self.assertGreater(report["methods"]["hardlink"], 0)
            (target / "data/canonical_local" / ("c" * 64 + ".pt")).write_bytes(b"UNIT new destination-only publication")
            self.assertFalse((source / "data/canonical_local" / ("c" * 64 + ".pt")).exists())

    def test_idempotent_repeat_does_not_replace_advanced_checkpoint_or_logs(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            migration.prepare_continuation(source, target, "selected")
            latest = target / "selected/checkpoint_latest.pt"
            latest.write_bytes(b"UNIT newly advanced checkpoint")
            (target / "selected/curve.jsonl").write_text('{"unit":"advanced"}\n')
            write_json(target / "memory_execution.json", dict(unit=True))
            repeated = migration.prepare_continuation(source, target, "selected")
            self.assertTrue(repeated["reused"])
            self.assertEqual(latest.read_bytes(), b"UNIT newly advanced checkpoint")
            self.assertEqual((source / "selected/checkpoint_latest.pt").read_bytes(), b"UNIT opaque latest checkpoint")

    def test_native_without_saved_arm_uses_same_initial_and_calibration(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "native_independent"
            report = migration.prepare_continuation(source, target, "native")
            self.assertFalse(report["resumed"])
            self.assertFalse((target / "selected").exists())
            self.assertFalse((target / "native").exists())
            self.assertEqual((source / "calibration.json").read_bytes(), (target / "calibration.json").read_bytes())

    def test_live_source_lock_rejected_without_destination_writes(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            write_json(source / ".pipeline.lock", dict(host=socket.gethostname(), pid=os.getpid(), token="UNIT"))
            with self.assertRaisesRegex(RuntimeError, "still active"):
                migration.prepare_continuation(source, target, "selected")
            self.assertFalse(target.exists())

    def test_other_host_data_lock_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            write_json(source / "data/.data.lock", dict(host="UNIT_other_host", pid=123, token="UNIT"))
            with self.assertRaisesRegex(RuntimeError, "another/unknown host"):
                migration.prepare_continuation(source, target, "selected")
            self.assertFalse(target.exists())

    def test_dead_local_lock_is_preserved_byte_for_byte(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            owner = source / ".selected.lock"
            write_json(owner, dict(host=socket.gethostname(), pid=123456789, token="UNIT"))
            original = owner.read_bytes()
            with patch.object(migration, "_pid_alive", return_value=False):
                migration.prepare_continuation(source, target, "selected")
            self.assertEqual(owner.read_bytes(), original)
            self.assertFalse((target / ".selected.lock").exists())

    def test_source_or_baseline_overlap_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base)
            for target in (source, source / "child", base, base / "preserved_baseline/child"):
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "disjoint"):
                    migration.prepare_continuation(source, target, "selected")

    def test_old_contract_or_calibration_tamper_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base)
            contract = migration._read(source / "experiment.json")
            contract["unit_fixture"] = "UNIT tampered"
            write_json(source / "experiment.json", contract)
            with self.assertRaisesRegex(ValueError, "sealed"):
                migration.prepare_continuation(source, base / "independent", "selected")

    def test_published_field_payload_tamper_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base)
            (source / "data/whole_case_fields/UNIT_case/depth.npy").write_bytes(b"UNIT changed payload")
            with self.assertRaisesRegex(ValueError, "SHA/size"):
                migration.prepare_continuation(source, base / "independent", "selected")

    def test_truncated_canonical_publication_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base)
            (source / "data/canonical_local" / ("a" * 64 + ".pt")).write_bytes(b"UNIT truncated ZIP")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                migration.prepare_continuation(source, base / "independent", "selected")

    def test_destination_immutable_or_receipt_tamper_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            migration.prepare_continuation(source, target, "selected")
            (target / "initial.pt").write_bytes(b"UNIT tampered initial")
            with self.assertRaisesRegex(ValueError, "immutable bytes"):
                migration.prepare_continuation(source, target, "selected")

    def test_cross_device_copy_fallback_but_not_general_error_fallback(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            with patch.object(migration.os, "link", side_effect=OSError(errno.EXDEV, "UNIT cross-device")):
                report = migration.prepare_continuation(source, target, "selected")
            self.assertEqual(report["methods"]["hardlink"], 0)
            self.assertGreater(report["methods"]["copy"], 3)
            with patch.object(migration.os, "link", side_effect=OSError(errno.EACCES, "UNIT permission failure")):
                with self.assertRaises(OSError):
                    migration.prepare_continuation(source, base / "permission_error", "native")

    def test_unowned_destination_and_second_arm_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            target.mkdir(); (target / "user_result.txt").write_text("UNIT user-owned file")
            with self.assertRaisesRegex(FileExistsError, "Unowned"):
                migration.prepare_continuation(source, target, "selected")
            other = base / "owned"
            migration.prepare_continuation(source, other, "selected")
            with self.assertRaisesRegex(ValueError, "different source/arm"):
                migration.prepare_continuation(source, other, "native")

    def test_arm_files_without_latest_checkpoint_are_not_invented_as_resume(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base)
            (source / "selected/checkpoint_latest.pt").unlink()
            with self.assertRaisesRegex(ValueError, "without a latest checkpoint"):
                migration.prepare_continuation(source, base / "independent", "selected")

    def test_interrupted_owned_copy_retries_without_overwriting_existing_bytes(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            damaged = source / "data/whole_case_fields/UNIT_case/depth.npy"
            original = damaged.read_bytes()
            damaged.write_bytes(b"UNIT interrupted verification fixture")
            with self.assertRaises(ValueError):
                migration.prepare_continuation(source, target, "selected")
            self.assertTrue((target / migration.PENDING).is_file())
            self.assertFalse((target / migration.RECEIPT).exists())
            preserved_initial = (target / "initial.pt").read_bytes()
            damaged.write_bytes(original)
            report = migration.prepare_continuation(source, target, "selected")
            self.assertEqual((target / "initial.pt").read_bytes(), preserved_initial)
            self.assertGreater(report["methods"]["existing"], 0)

    def test_altered_receipt_or_unsafe_path_is_rejected(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            migration.prepare_continuation(source, target, "selected")
            path = target / migration.RECEIPT
            receipt = migration._read(path)
            receipt["methods"]["copy"] += 1
            write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, "receipt identity"):
                migration.prepare_continuation(source, target, "selected")
            receipt["files"][0]["path"] = "../outside.txt"
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = migration._digest(receipt)
            write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, "path/role"):
                migration.prepare_continuation(source, target, "selected")

    def test_existing_destination_mutable_symlink_is_not_followed_on_repeat(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            migration.prepare_continuation(source, target, "selected")
            suspect = target / "selected/checkpoint_latest.pt"
            original_check = Path.is_symlink
            def simulate_symlink(path):
                return path == suspect or original_check(path)
            with patch.object(Path, "is_symlink", simulate_symlink):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    migration.prepare_continuation(source, target, "selected")

    def test_original_static_upper_lesions_publication_kind_is_preserved(self):
        with self.directory() as directory:
            base = Path(directory); source = fixture(base); target = base / "independent"
            parent = source / "data/upper_static/UNIT_case/c1_a1_1_1"
            lesions = parent / "lesions"
            lesions.mkdir()
            payload = lesions / "arrays.npz"
            payload.write_bytes(b"UNIT original lesions schema publication, not a neural result")
            metadata = migration._read(parent / "source_raw/metadata.json")
            metadata.update(kind="lesions", payload_bytes=payload.stat().st_size,
                            payload_sha256=migration._sha(payload), arrays=[
                                dict(shape=[7, 14], dtype="<f4", bytes=392),
                                dict(shape=[7, 3], dtype="<f4", bytes=84),
                                dict(shape=[7], dtype="<i8", bytes=56)])
            metadata.pop("metadata_sha256")
            metadata["metadata_sha256"] = migration._digest(metadata)
            write_json(lesions / "metadata.json", metadata)
            migration.prepare_continuation(source, target, "selected")
            cloned = target / "data/upper_static/UNIT_case/c1_a1_1_1/lesions/arrays.npz"
            self.assertTrue(os.path.samefile(payload, cloned))
            self.assertEqual((lesions / "metadata.json").read_bytes(),
                             cloned.with_name("metadata.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
