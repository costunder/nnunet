#!/usr/bin/env python3
"""Validate outputs against an explicit, reproducible cohort contract.

Production mode is the default.  It requires the candidate and reference case
ID sets to match exactly and treats every rejection as a failed validation.
Intentional subset validation is available only through the explicitly
non-production ``--run-mode nonproduction`` profile.

``--resume`` never trusts an ``ok`` flag alone.  A row is reusable only when
SHA-256 fingerprints for all four input files, the validation contract, and
the complete cohort still match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from hiercp.common import CasePaths, discover_cases, output_paths, write_manifest


VALIDATION_SCHEMA_VERSION = 2
RUN_MODES = ("production", "nonproduction")


def load_array(path: Path):
    nii = nib.load(str(path))
    array = np.asarray(nii.dataobj)
    if array.ndim == 4 and array.shape[-1] <= 4:
        array = array[..., 0]
    elif array.ndim == 4 and array.shape[0] <= 4:
        array = array[0]
    return nii, array


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_previous_report(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle)
            if row.get("case_id")
        }


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a content fingerprint without loading a volume into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_contract_sha256(
    *,
    tumor_label: int,
    allowed: set[int],
    expect_augmentation: bool,
    run_mode: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "tumor_label": int(tumor_label),
            "allowed_labels": sorted(int(value) for value in allowed),
            "expect_augmentation": bool(expect_augmentation),
            "run_mode": run_mode,
            "checks": (
                "shape,image_affine,label_affine,allowed_labels,finite_image,"
                "tumor_delta"
            ),
        }
    )


def cohort_sha256(
    *, reference_ids: set[str], candidate_ids: set[str], run_mode: str
) -> str:
    return canonical_sha256(
        {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "run_mode": run_mode,
            "reference_case_ids": sorted(reference_ids),
            "candidate_case_ids": sorted(candidate_ids),
        }
    )


def case_input_sha256(
    *, candidate: CasePaths, reference: CasePaths
) -> dict[str, str]:
    return {
        "candidate_image_sha256": sha256_file(candidate.image_path),
        "candidate_label_sha256": sha256_file(candidate.label_path),
        "reference_image_sha256": sha256_file(reference.image_path),
        "reference_label_sha256": sha256_file(reference.label_path),
    }


def atomic_write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_manifest(rows, temporary)
        os.replace(temporary, path)
    finally:
        if _exists(temporary):
            temporary.unlink()


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if _exists(temporary):
            temporary.unlink()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _same_file(source: Path, destination: Path) -> bool:
    try:
        return destination.samefile(source)
    except OSError:
        return False


def _same_content(source: Path, destination: Path) -> bool:
    if _same_file(source, destination):
        return True
    if not destination.is_file():
        return False
    try:
        if source.stat().st_size != destination.stat().st_size:
            return False
        return sha256_file(source) == sha256_file(destination)
    except OSError:
        return False


def _refuse_replacement(paths: list[Path], *, action: str) -> None:
    rendered = ", ".join(str(path) for path in paths)
    raise FileExistsError(
        f"Refusing to {action} without explicit overwrite authorization: {rendered}. "
        "Confirm the paths, then re-run with --overwrite, or choose a new output directory."
    )


def materialize(src: Path, dst: Path, mode: str, *, overwrite: bool = False) -> None:
    if _exists(dst) and _same_content(src, dst):
        return
    if _exists(dst) and not overwrite:
        _refuse_replacement([dst], action="replace an existing validated output")
    dst.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if mode in {"symlink", "hardlink"}:
            temporary.unlink()
            if mode == "symlink":
                temporary.symlink_to(src.resolve())
            else:
                os.link(src, temporary)
        elif mode == "copy":
            shutil.copy2(src, temporary)
        else:
            raise ValueError(f"Unsupported materialization mode: {mode}")
        if not _same_content(src, temporary):
            raise ValueError(f"Materialization content mismatch: {src}")
        if overwrite:
            os.replace(temporary, dst)
        elif mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            os.link(temporary, dst)
    finally:
        if _exists(temporary):
            temporary.unlink()


def remove_materialized(
    valid_output_dir: Path,
    case_id: str,
    *,
    overwrite: bool = False,
) -> None:
    image_path, label_path = output_paths(valid_output_dir, case_id)
    existing = [path for path in (image_path, label_path) if _exists(path)]
    if existing and not overwrite:
        _refuse_replacement(existing, action="remove rejected validated outputs")
    for path in existing:
        if _exists(path):
            path.unlink()


def materialize_case(
    case: CasePaths,
    valid_output_dir: Path,
    mode: str,
    *,
    overwrite: bool = False,
) -> None:
    image_dst, label_dst = output_paths(valid_output_dir, case.case_id)
    pairs = ((case.image_path, image_dst), (case.label_path, label_dst))
    conflicts = [
        destination
        for source, destination in pairs
        if _exists(destination) and not _same_content(source, destination)
    ]
    if conflicts and not overwrite:
        _refuse_replacement(conflicts, action="replace existing validated outputs")
    for source, destination in pairs:
        materialize(source, destination, mode, overwrite=overwrite)


def _materialized_inventory(valid_output_dir: Path) -> set[Path]:
    if not _exists(valid_output_dir):
        return set()
    if valid_output_dir.is_symlink() or not valid_output_dir.is_dir():
        raise ValueError(
            f"Validated output root must be a real directory, not a file or symlink: "
            f"{valid_output_dir}"
        )

    inventory: set[Path] = set()
    allowed_directories = {"image", "labels"}
    for child in valid_output_dir.iterdir():
        if child.name not in allowed_directories:
            inventory.add(child)
    for name in sorted(allowed_directories):
        folder = valid_output_dir / name
        if not _exists(folder):
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(
                f"Validated output component must be a real directory: {folder}"
            )
        inventory.update(folder.iterdir())
    return inventory


def materialize_and_verify(
    cases: list[CasePaths],
    valid_output_dir: Path,
    mode: str,
    *,
    overwrite: bool = False,
    source_sha256: dict[Path, str] | None = None,
) -> str:
    valid_output_dir = valid_output_dir.absolute()
    expected: dict[Path, Path] = {}
    for case in cases:
        image_dst, label_dst = output_paths(valid_output_dir, case.case_id)
        for source, destination in (
            (case.image_path, image_dst),
            (case.label_path, label_dst),
        ):
            if destination in expected:
                raise ValueError(f"Duplicate validated output path: {destination}")
            expected[destination] = source

    hashes = {
        source: sha256_file(source) if source_sha256 is None else source_sha256[source]
        for source in expected.values()
    }
    for source, fingerprint in hashes.items():
        if source.resolve().is_relative_to(valid_output_dir.resolve()):
            raise ValueError(f"Validated output overlaps its source: {source}")
        if sha256_file(source) != fingerprint:
            raise ValueError(f"Accepted input changed after validation: {source}")
    relative = {
        destination.relative_to(valid_output_dir): hashes[source]
        for destination, source in expected.items()
    }
    fingerprint = canonical_sha256(
        {"files": {path.as_posix(): value for path, value in relative.items()}}
    )

    def verify(root: Path) -> bool:
        actual = _materialized_inventory(root)
        return actual == {root / path for path in relative} and all(
            (root / path).is_file() and sha256_file(root / path) == value
            for path, value in relative.items()
        )

    def check_tree(root: Path) -> None:
        unsafe = [
            path for path in _materialized_inventory(root)
            if not path.is_file() and not path.is_symlink()
        ]
        if unsafe:
            raise IsADirectoryError(
                f"Refusing to replace unexpected nested output directories: {unsafe}"
            )

    def remove_owned_tree(root: Path) -> None:
        if root.parent != valid_output_dir.parent or not root.name.startswith(
            f".{valid_output_dir.name}.publish-"
        ):
            raise ValueError(f"Unsafe publication cleanup path: {root}")
        check_tree(root)
        for path in _materialized_inventory(root):
            path.unlink()
        for name in ("image", "labels"):
            folder = root / name
            if folder.is_dir():
                folder.rmdir()
        root.rmdir()

    if _exists(valid_output_dir):
        check_tree(valid_output_dir)
        if verify(valid_output_dir):
            return fingerprint
        if not overwrite:
            _refuse_replacement([valid_output_dir], action="replace a validated dataset")

    valid_output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = valid_output_dir.parent / f".{valid_output_dir.name}.publish.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    staging: Path | None = None
    backup: Path | None = None
    published = False
    try:
        # Recheck after taking the writer lock; no existing output is removed
        # before a complete replacement has passed the same content checks.
        if _exists(valid_output_dir):
            check_tree(valid_output_dir)
            if verify(valid_output_dir):
                return fingerprint
            if not overwrite:
                _refuse_replacement([valid_output_dir], action="replace a validated dataset")
        staging = Path(tempfile.mkdtemp(
            prefix=f".{valid_output_dir.name}.publish-stage-", dir=valid_output_dir.parent
        ))
        for destination, source in expected.items():
            materialize(source, staging / destination.relative_to(valid_output_dir), mode)
        if not verify(staging) or any(
            sha256_file(source) != value for source, value in hashes.items()
        ):
            raise ValueError("Accepted inputs or staged dataset changed during publication")
        if _exists(valid_output_dir):
            backup = Path(tempfile.mkdtemp(
                prefix=f".{valid_output_dir.name}.publish-backup-", dir=valid_output_dir.parent
            ))
            backup.rmdir()
            valid_output_dir.rename(backup)
        try:
            staging.rename(valid_output_dir)
            published = True
            if not verify(valid_output_dir):
                raise ValueError("Published validated dataset failed content verification")
        except Exception:
            if published:
                valid_output_dir.rename(staging)
                published = False
            if backup is not None and backup.is_dir():
                backup.rename(valid_output_dir)
                backup = None
            raise
        if backup is not None:
            remove_owned_tree(backup)
            print(f"[Replace] verified dataset published; previous dataset removed: {valid_output_dir}")
        return fingerprint
    finally:
        if staging is not None and staging.is_dir():
            remove_owned_tree(staging)
        lock.unlink()


def previous_valid_row_is_reusable(
    row: dict[str, str] | None,
    *,
    candidate: CasePaths,
    reference: CasePaths,
    input_sha256: dict[str, str],
    contract_sha256: str,
    expected_cohort_sha256: str,
) -> bool:
    if not row or not parse_bool(row.get("ok", False)):
        return False
    if not candidate.image_path.is_file() or not candidate.label_path.is_file():
        return False
    if not reference.image_path.is_file() or not reference.label_path.is_file():
        return False
    expected = {
        **input_sha256,
        "validation_contract_sha256": contract_sha256,
        "cohort_sha256": expected_cohort_sha256,
        "validation_schema_version": str(VALIDATION_SCHEMA_VERSION),
    }
    return all(str(row.get(key, "")) == str(value) for key, value in expected.items())


def validate_case(
    reference: CasePaths,
    candidate: CasePaths,
    *,
    tumor_label: int,
    allowed: set[int],
    expect_augmentation: bool,
) -> dict[str, object]:
    ref_image_nii, ref_image = load_array(reference.image_path)
    ref_label_nii, ref_label = load_array(reference.label_path)
    out_image_nii, out_image = load_array(candidate.image_path)
    out_label_nii, out_label = load_array(candidate.label_path)

    shape_ok = ref_image.shape == out_image.shape == ref_label.shape == out_label.shape
    image_affine_ok = np.allclose(ref_image_nii.affine, out_image_nii.affine, atol=1e-5)
    label_affine_ok = np.allclose(ref_label_nii.affine, out_label_nii.affine, atol=1e-5)
    labels = set(np.unique(out_label).astype(int).tolist())
    labels_ok = labels.issubset(allowed)
    finite_ok = bool(np.isfinite(out_image).all())
    ref_tumor = int(np.sum(ref_label == tumor_label))
    out_tumor = int(np.sum(out_label == tumor_label))
    tumor_delta = out_tumor - ref_tumor
    tumor_ok = tumor_delta > 0 if expect_augmentation else tumor_delta == 0

    structural_ok = all(
        (shape_ok, image_affine_ok, label_affine_ok, labels_ok, finite_ok)
    )
    ok = structural_ok and tumor_ok
    if not structural_ok:
        reason = "structural_validation_failed"
    elif not tumor_ok and expect_augmentation:
        reason = "no_tumor_added"
    elif not tumor_ok:
        reason = "unexpected_tumor_change"
    else:
        reason = "accepted"

    return {
        "case_id": candidate.case_id,
        "ok": ok,
        "reason": reason,
        "expect_augmentation": expect_augmentation,
        "shape_ok": shape_ok,
        "image_affine_ok": image_affine_ok,
        "label_affine_ok": label_affine_ok,
        "labels_ok": labels_ok,
        "finite_ok": finite_ok,
        "reference_tumor_voxels": ref_tumor,
        "output_tumor_voxels": out_tumor,
        "tumor_delta": tumor_delta,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--valid-output-dir",
        default=None,
        help="Optional dataset containing only accepted pairs.",
    )
    parser.add_argument(
        "--materialization",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument("--tumor-label", type=int, default=2)
    parser.add_argument("--allowed-label", type=int, action="append", default=[0, 1, 2])
    parser.add_argument("--expect-augmentation", action="store_true")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="production",
        help=(
            "production requires the exact full reference cohort and zero "
            "rejections; nonproduction explicitly permits a candidate subset"
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse only rows whose SHA-256 input and contract fingerprints match.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero on any rejection in nonproduction mode. Production "
            "mode is always strict."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly authorize replacement of validation reports and existing "
            "materialized outputs."
        ),
    )
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    valid_output_dir = (
        Path(args.valid_output_dir).resolve() if args.valid_output_dir else None
    )
    if valid_output_dir is not None:
        for source_root in (reference_dir, candidate_dir):
            if valid_output_dir.is_relative_to(source_root) or source_root.is_relative_to(valid_output_dir):
                raise ValueError(
                    f"Validated output and input roots must not overlap: "
                    f"{valid_output_dir} / {source_root}"
                )
    report = candidate_dir / "validation.csv"
    summary_path = candidate_dir / "validation_summary.json"
    existing_reports = [path for path in (report, summary_path) if _exists(path)]
    if existing_reports and not (args.resume or args.overwrite):
        rendered = ", ".join(str(path) for path in existing_reports)
        raise FileExistsError(
            "Existing validation reports require an explicit safe action: "
            f"{rendered}. Use --resume to fingerprint-check reusable rows, "
            "--overwrite to revalidate and atomically replace them, or choose "
            "a new candidate directory."
        )

    reference_cases = discover_cases(reference_dir)
    reference = {case.case_id: case for case in reference_cases}
    candidate_cases = discover_cases(candidate_dir)
    reference_ids = set(reference)
    candidate_ids = {case.case_id for case in candidate_cases}
    missing_candidate_ids = sorted(reference_ids - candidate_ids)
    unexpected_candidate_ids = sorted(candidate_ids - reference_ids)
    if unexpected_candidate_ids:
        raise RuntimeError(
            "Candidate cohort contains cases absent from the reference cohort: "
            f"{unexpected_candidate_ids}. Correct --reference-dir/--candidate-dir; "
            "unmatched cases cannot be validated."
        )
    if args.run_mode == "production" and missing_candidate_ids:
        raise RuntimeError(
            "Production validation requires the exact full reference cohort. "
            f"reference_cases={len(reference_ids)}, candidate_cases={len(candidate_ids)}, "
            f"missing_candidate_ids={missing_candidate_ids}. Regenerate the missing "
            "cases; use --run-mode nonproduction only for an explicitly non-final "
            "subset check."
        )

    allowed = set(args.allowed_label)
    contract_fingerprint = validation_contract_sha256(
        tumor_label=args.tumor_label,
        allowed=allowed,
        expect_augmentation=args.expect_augmentation,
        run_mode=args.run_mode,
    )
    cohort_fingerprint = cohort_sha256(
        reference_ids=reference_ids,
        candidate_ids=candidate_ids,
        run_mode=args.run_mode,
    )
    previous = read_previous_report(report) if args.resume else {}

    rows: list[dict[str, object]] = []
    accepted = 0
    rejected = 0
    resumed = 0
    rechecked = 0

    for candidate in candidate_cases:
        ref = reference[candidate.case_id]
        input_fingerprints = case_input_sha256(candidate=candidate, reference=ref)

        old_row = previous.get(candidate.case_id)
        if args.resume and previous_valid_row_is_reusable(
            old_row,
            candidate=candidate,
            reference=ref,
            input_sha256=input_fingerprints,
            contract_sha256=contract_fingerprint,
            expected_cohort_sha256=cohort_fingerprint,
        ):
            row = dict(old_row or {})
            row["resumed"] = True
            rows.append(row)
            accepted += 1
            resumed += 1
            print(f"[Resume] {candidate.case_id}: already valid")
            continue

        rechecked += 1
        try:
            row = validate_case(
                ref,
                candidate,
                tumor_label=args.tumor_label,
                allowed=allowed,
                expect_augmentation=args.expect_augmentation,
            )
        except Exception as exc:
            row = {
                "case_id": candidate.case_id,
                "ok": False,
                "reason": "exception",
                "message": str(exc),
                "expect_augmentation": args.expect_augmentation,
                "validated_at": datetime.now().isoformat(timespec="seconds"),
            }

        row.update(
            {
                **input_fingerprints,
                "validation_contract_sha256": contract_fingerprint,
                "cohort_sha256": cohort_fingerprint,
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "run_mode": args.run_mode,
                "resumed": False,
            }
        )

        rows.append(row)
        if parse_bool(row.get("ok", False)):
            accepted += 1
            print(f"[OK] {candidate.case_id} tumor_delta={row.get('tumor_delta', '')}")
        else:
            rejected += 1
            print(
                f"[REJECT] {candidate.case_id} reason={row.get('reason')} "
                f"tumor_delta={row.get('tumor_delta', '')}"
            )

    # Bind every result to the bytes checked above, including resumed rows.
    for candidate, row in zip(candidate_cases, rows, strict=True):
        current = case_input_sha256(candidate=candidate, reference=reference[candidate.case_id])
        if any(current[key] != row.get(key) for key in current):
            raise RuntimeError(f"Validation input changed while checking: {candidate.case_id}")

    strict_effective = args.run_mode == "production" or args.strict
    successful = accepted > 0 and not (strict_effective and rejected)
    if args.run_mode == "production":
        successful = successful and accepted == len(reference_ids)

    accepted_ids = {
        str(row["case_id"])
        for row in rows
        if parse_bool(row.get("ok", False))
    }
    materialized_verified = False
    materialized_inventory_sha256: str | None = None

    # Never materialize a partial production cohort.  Non-production mode may
    # intentionally materialize its explicitly labelled accepted subset.
    if successful and valid_output_dir is not None:
        accepted_cases = [
            candidate
            for candidate in candidate_cases
            if candidate.case_id in accepted_ids
        ]
        materialized_inventory_sha256 = materialize_and_verify(
            accepted_cases,
            valid_output_dir,
            args.materialization,
            overwrite=args.overwrite,
            source_sha256={
                source: str(row[key])
                for case, row in zip(candidate_cases, rows, strict=True)
                if case.case_id in accepted_ids
                for source, key in (
                    (case.image_path, "candidate_image_sha256"),
                    (case.label_path, "candidate_label_sha256"),
                )
            },
        )
        materialized_verified = True

    summary = {
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "valid_output_dir": str(valid_output_dir) if valid_output_dir else None,
        "expect_augmentation": args.expect_augmentation,
        "run_mode": args.run_mode,
        "reference_total": len(reference_ids),
        "candidate_total": len(candidate_ids),
        "cohort_exact": reference_ids == candidate_ids,
        "missing_candidate_ids": missing_candidate_ids,
        "unexpected_candidate_ids": unexpected_candidate_ids,
        "validation_contract_sha256": contract_fingerprint,
        "cohort_sha256": cohort_fingerprint,
        "total": len(candidate_ids),
        "accepted": accepted,
        "rejected": rejected,
        "resumed_without_loading": resumed,
        "rechecked": rechecked,
        "strict_requested": args.strict,
        "strict_effective": strict_effective,
        "validation_complete": successful,
        "production_complete": args.run_mode == "production" and successful,
        "materialized": materialized_verified,
        "materialized_case_ids": sorted(accepted_ids) if materialized_verified else [],
        "materialized_inventory_sha256": materialized_inventory_sha256,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_manifest(rows, report)
    atomic_write_json(summary, summary_path)

    print(
        "[ValidationSummary] "
        f"run_mode={args.run_mode} accepted={accepted} rejected={rejected} "
        f"total={len(candidate_ids)} reference_total={len(reference_ids)} "
        f"resumed={resumed} rechecked={rechecked} report={report} "
        f"validation_complete={successful} "
        f"production_complete={summary['production_complete']} "
        f"cohort_exact={summary['cohort_exact']} "
        f"materialized={materialized_verified} "
        f"materialized_inventory_sha256={materialized_inventory_sha256}"
    )
    if successful and materialized_verified:
        print(f"Accepted dataset: {valid_output_dir}")

    if accepted == 0:
        raise SystemExit(2)
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
