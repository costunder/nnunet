#!/usr/bin/env python3
"""Safely assemble an exact original/validated-augmentation cohort.

Synthetic cases are renamed to ``<case_id>__<tag>``.  The full source,
destination, validation, and collision contract is checked before writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hiercp.common import discover_cases, write_manifest


RUN_MODES = ("production", "nonproduction")
SUPPORTED_VALIDATION_SCHEMA_VERSION = "2"
REQUIRED_VALIDATION_FIELDS = frozenset(
    {
        "case_id",
        "ok",
        "run_mode",
        "validation_schema_version",
        "validation_contract_sha256",
        "cohort_sha256",
        "candidate_image_sha256",
        "candidate_label_sha256",
        "reference_image_sha256",
        "reference_label_sha256",
    }
)


@dataclass(frozen=True)
class PlannedCase:
    output_case_id: str
    source_case_id: str
    source_kind: str
    image_source: Path
    label_source: Path
    image_destination: Path
    label_destination: Path
    validation_contract_sha256: str = ""
    validation_report_sha256: str = ""

    def manifest_row(self, *, run_mode: str) -> dict[str, object]:
        return {
            "output_case_id": self.output_case_id,
            "source_case_id": self.source_case_id,
            "source": self.source_kind,
            "source_image": str(self.image_source),
            "source_label": str(self.label_source),
            "image": str(self.image_destination),
            "label": str(self.label_destination),
            "run_mode": run_mode,
            "validation_contract_sha256": self.validation_contract_sha256,
            "validation_report_sha256": self.validation_report_sha256,
        }


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _same_file(source: Path, destination: Path) -> bool:
    try:
        return destination.samefile(source)
    except OSError:
        return False


def _path_key(path: Path) -> str:
    # Destination identity is lexical here. Resolving would make two distinct
    # symlinks collide merely because they currently point at the same source.
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if _exists(dst):
        if _same_file(src, dst):
            return
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace non-identical destination: {dst}. Existing "
                "outputs are reusable only when Path.samefile(source, destination) "
                "is true; pass --overwrite after confirming this exact path."
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        if mode == "symlink":
            temporary.symlink_to(src.resolve())
        elif mode == "hardlink":
            os.link(src, temporary)
        elif mode == "copy":
            shutil.copy2(src, temporary)
        else:  # pragma: no cover - argparse prevents this
            raise ValueError(f"Unsupported materialization mode: {mode}")
        os.replace(temporary, dst)
    finally:
        if _exists(temporary):
            temporary.unlink()


def _plan_cases(
    cases: list,
    *,
    out_dir: Path,
    suffix: str,
    source_kind: str,
    validation_rows: dict[str, dict[str, str]] | None = None,
    validation_report_sha256: str = "",
) -> list[PlannedCase]:
    planned: list[PlannedCase] = []
    for case in cases:
        output_id = f"{case.case_id}{suffix}"
        validation_row = (validation_rows or {}).get(case.case_id, {})
        planned.append(
            PlannedCase(
                output_case_id=output_id,
                source_case_id=case.case_id,
                source_kind=source_kind,
                image_source=case.image_path,
                label_source=case.label_path,
                image_destination=out_dir / "image" / f"{output_id}_0000.nii.gz",
                label_destination=out_dir / "labels" / f"{output_id}.nii.gz",
                validation_contract_sha256=str(
                    validation_row.get("validation_contract_sha256", "")
                ),
                validation_report_sha256=validation_report_sha256,
            )
        )
    return planned


def _read_validation_report(
    path: Path,
    *,
    augmented_cases: list,
    reference_cases: list,
    run_mode: str,
) -> tuple[dict[str, dict[str, str]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation report does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_fields = sorted(REQUIRED_VALIDATION_FIELDS - fields)
        if missing_fields:
            raise RuntimeError(
                f"Validation report lacks required fingerprint fields: {missing_fields}. "
                "Re-run tools.validate with the current validation schema."
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Validation report is empty: {path}")

    by_id: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise RuntimeError(f"Validation report contains a row without case_id: {path}")
        if case_id in by_id:
            duplicates.append(case_id)
        by_id[case_id] = row
    if duplicates:
        raise RuntimeError(
            f"Validation report contains duplicate case IDs: {sorted(set(duplicates))}"
        )
    invalid_schema_ids = sorted(
        case_id
        for case_id, row in by_id.items()
        if str(row.get("validation_schema_version", ""))
        != SUPPORTED_VALIDATION_SCHEMA_VERSION
    )
    if invalid_schema_ids:
        raise RuntimeError(
            "Validation report was not produced by the supported fingerprint schema. "
            f"invalid_schema_case_ids={invalid_schema_ids}; re-run tools.validate."
        )

    accepted = {
        case_id: row
        for case_id, row in by_id.items()
        if _parse_bool(row.get("ok", False))
    }
    augmented_by_id = {case.case_id: case for case in augmented_cases}
    reference_by_id = {case.case_id: case for case in reference_cases}
    accepted_ids = set(accepted)
    augmented_ids = set(augmented_by_id)
    if accepted_ids != augmented_ids:
        raise RuntimeError(
            "Augmentation directory must equal the validation report's accepted cohort. "
            f"missing_accepted_outputs={sorted(accepted_ids - augmented_ids)}, "
            f"unvalidated_or_rejected_outputs={sorted(augmented_ids - accepted_ids)}."
        )

    if run_mode == "production":
        rejected_ids = sorted(set(by_id) - accepted_ids)
        nonproduction_ids = sorted(
            case_id
            for case_id, row in by_id.items()
            if row.get("run_mode") != "production"
        )
        if rejected_ids or nonproduction_ids:
            raise RuntimeError(
                "Production assembly requires a full production validation report "
                "with zero rejected rows. "
                f"rejected_ids={rejected_ids}, nonproduction_rows={nonproduction_ids}."
            )

    contract_fingerprints = {
        str(row.get("validation_contract_sha256", "")) for row in accepted.values()
    }
    cohort_fingerprints = {
        str(row.get("cohort_sha256", "")) for row in accepted.values()
    }
    if len(contract_fingerprints) != 1 or "" in contract_fingerprints:
        raise RuntimeError("Accepted validation rows do not share one contract fingerprint")
    if len(cohort_fingerprints) != 1 or "" in cohort_fingerprints:
        raise RuntimeError("Accepted validation rows do not share one cohort fingerprint")

    stale_inputs: list[str] = []
    for case_id, case in augmented_by_id.items():
        row = accepted[case_id]
        if sha256_file(case.image_path) != row.get("candidate_image_sha256"):
            stale_inputs.append(f"{case_id}:image")
        if sha256_file(case.label_path) != row.get("candidate_label_sha256"):
            stale_inputs.append(f"{case_id}:label")
        reference_case = reference_by_id[case_id]
        if sha256_file(reference_case.image_path) != row.get("reference_image_sha256"):
            stale_inputs.append(f"{case_id}:reference_image")
        if sha256_file(reference_case.label_path) != row.get("reference_label_sha256"):
            stale_inputs.append(f"{case_id}:reference_label")
    if stale_inputs:
        raise RuntimeError(
            "Validated augmentation inputs changed after validation: "
            f"{stale_inputs}. Re-run tools.validate before assembly."
        )
    return accepted, sha256_file(path)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not _exists(current) and current.parent != current:
        current = current.parent
    return current


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _preflight(
    plan: list[PlannedCase],
    *,
    out_dir: Path,
    manifest_path: Path,
    manifest_rows: list[dict[str, object]],
    mode: str,
    overwrite: bool,
) -> bool:
    """Validate the complete plan before materializing any destination."""

    output_ids: set[str] = set()
    destination_owners: dict[str, str] = {}
    conflicts: list[str] = []

    for item in plan:
        if item.output_case_id in output_ids:
            raise RuntimeError(f"Output case ID collision: {item.output_case_id}")
        output_ids.add(item.output_case_id)
        for source, destination, role in (
            (item.image_source, item.image_destination, "image"),
            (item.label_source, item.label_destination, "label"),
        ):
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing {role} source for {item.source_case_id}: {source}"
                )
            key = _path_key(destination)
            owner = destination_owners.get(key)
            if owner is not None:
                raise RuntimeError(
                    f"Destination collision at {destination}: {owner} and "
                    f"{item.output_case_id}:{role}"
                )
            destination_owners[key] = f"{item.output_case_id}:{role}"
            parent = _nearest_existing_parent(destination.parent)
            if not parent.is_dir():
                raise NotADirectoryError(
                    f"Destination parent is blocked by a non-directory path: {parent}"
                )
            if mode == "hardlink" and source.stat().st_dev != parent.stat().st_dev:
                raise RuntimeError(
                    f"Hardlink crosses filesystems for {source} -> {destination}; "
                    "choose --mode symlink or copy."
                )
            if _exists(destination) and destination.is_dir():
                raise IsADirectoryError(
                    f"Destination file path is occupied by a directory: {destination}"
                )
            if _exists(destination) and not _same_file(source, destination):
                conflicts.append(str(destination))

    expected_destinations = set(destination_owners)
    stale: list[str] = []
    for subdir, pattern in (("image", "*_0000.nii.gz"), ("labels", "*.nii.gz")):
        directory = out_dir / subdir
        if _exists(directory) and not directory.is_dir():
            raise NotADirectoryError(f"Assembly output path is not a directory: {directory}")
        if directory.is_dir():
            stale.extend(
                str(path)
                for path in directory.glob(pattern)
                if _path_key(path) not in expected_destinations
            )
    if stale:
        raise RuntimeError(
            "Assembly output contains unplanned stale files. They are never deleted "
            f"implicitly, even with --overwrite: {sorted(stale)}. Use a clean output "
            "directory or explicitly remove the confirmed files."
        )
    if conflicts and not overwrite:
        raise FileExistsError(
            "Assembly destinations already exist but are not the requested source "
            f"files: {sorted(conflicts)}. Existing files are reusable only through "
            "Path.samefile; pass --overwrite after confirming these exact targets."
        )

    all_destinations_reusable = all(
        _same_file(source, destination)
        for item in plan
        for source, destination in (
            (item.image_source, item.image_destination),
            (item.label_source, item.label_destination),
        )
    )
    if _exists(manifest_path):
        if not manifest_path.is_file():
            raise FileExistsError(f"Manifest path is not a regular file: {manifest_path}")
        existing_rows = _read_manifest(manifest_path)
        expected_rows = [
            {key: str(value) for key, value in row.items()} for row in manifest_rows
        ]
        manifest_matches = existing_rows == expected_rows
        if not overwrite and not manifest_matches:
            raise FileExistsError(
                f"Existing manifest does not match the requested assembly: {manifest_path}. "
                "Pass --overwrite only after reviewing the requested cohort."
            )
        if not overwrite and manifest_matches and not all_destinations_reusable:
            raise RuntimeError(
                "The manifest matches, but one or more materialized files are missing "
                "or stale. Pass --overwrite to repair the verified destinations."
            )
        if manifest_matches and all_destinations_reusable:
            return True
    return False


def _atomic_write_manifest(rows: list[dict[str, object]], path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--aug-dir", default=None)
    parser.add_argument(
        "--validation-report",
        default=None,
        help="Required validation.csv whose accepted cohort exactly equals --aug-dir.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="aug")
    parser.add_argument("--mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="production",
        help="Only nonproduction permits an augmented subset of the original cohort.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.include_original and args.aug_dir is None:
        raise ValueError("Nothing to assemble: original disabled and --aug-dir omitted")
    if args.aug_dir is not None and args.validation_report is None:
        raise ValueError(
            "--aug-dir requires --validation-report; unvalidated augmentation output "
            "cannot be assembled into a final dataset."
        )
    if args.aug_dir is None and args.validation_report is not None:
        raise ValueError("--validation-report is only valid together with --aug-dir")

    original_dir = Path(args.original_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    original_cases = discover_cases(original_dir)
    original_ids = {case.case_id for case in original_cases}
    plan: list[PlannedCase] = []
    if args.include_original:
        plan.extend(
            _plan_cases(
                original_cases,
                out_dir=out_dir,
                suffix="",
                source_kind="original",
            )
        )
    if args.aug_dir is not None:
        safe_tag = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in args.tag)
        if not any(character.isalnum() for character in safe_tag):
            raise ValueError(
                f"--tag must contain at least one alphanumeric character: {args.tag!r}"
            )
        augmented_cases = discover_cases(Path(args.aug_dir).resolve())
        augmented_ids = {case.case_id for case in augmented_cases}
        unexpected_augmented_ids = sorted(augmented_ids - original_ids)
        missing_augmented_ids = sorted(original_ids - augmented_ids)
        if unexpected_augmented_ids:
            raise RuntimeError(
                "Augmentation cohort contains cases absent from --original-dir: "
                f"{unexpected_augmented_ids}"
            )
        if args.run_mode == "production" and missing_augmented_ids:
            raise RuntimeError(
                "Production assembly requires augmentation for the exact full original "
                f"cohort. missing_augmented_ids={missing_augmented_ids}. Use "
                "--run-mode nonproduction only for a clearly non-final subset artifact."
            )
        validation_path = Path(args.validation_report).resolve()
        validation_rows, validation_report_fingerprint = _read_validation_report(
            validation_path,
            augmented_cases=augmented_cases,
            reference_cases=original_cases,
            run_mode=args.run_mode,
        )
        plan.extend(
            _plan_cases(
                augmented_cases,
                out_dir=out_dir,
                suffix=f"__{safe_tag}",
                source_kind="augmented",
                validation_rows=validation_rows,
                validation_report_sha256=validation_report_fingerprint,
            )
        )

    rows = [item.manifest_row(run_mode=args.run_mode) for item in plan]
    manifest_path = out_dir / "manifest.csv"
    if _preflight(
        plan,
        out_dir=out_dir,
        manifest_path=manifest_path,
        manifest_rows=rows,
        mode=args.mode,
        overwrite=args.overwrite,
    ):
        print(
            f"[Resume] Assembly already matches exactly: cases={len(rows)} "
            f"run_mode={args.run_mode} out={out_dir}"
        )
        return

    for item in plan:
        materialize(item.image_source, item.image_destination, args.mode, args.overwrite)
        materialize(item.label_source, item.label_destination, args.mode, args.overwrite)
    _atomic_write_manifest(rows, manifest_path)
    print(
        f"[OK] Assembly complete: cases={len(rows)} run_mode={args.run_mode} "
        f"out={out_dir} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
