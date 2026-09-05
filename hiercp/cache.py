"""Prototype-bank and optimized hierarchical PyG cache preparation.

The optimized path preserves the graph/loss definitions while removing repeated
patient-region and source-local preprocessing.  Every cache directory also owns
a lightweight JSON index, so training does not deserialize every ``.pt`` file
just to validate metadata or recover the train/validation split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from hiercp.common import (
    LoadedCase,
    build_candidate_pool,
    choose_source_tumor,
    discover_cases,
    load_case,
    stable_case_seed,
    verify_loaded_case_source_signatures,
    write_manifest,
)
from hiercp.curriculum import build_generation_specs, build_training_specs
from hiercp.local import build_local_graph, prepare_local_source
from hiercp.prototype import PrototypeBank, build_prototype_bank
from hiercp.region import (
    REGION_CACHE_FORMAT,
    REGION_CACHE_SEED_SALT,
    PatientRegionData,
    load_or_build_patient_regions,
)
from hiercp.schema import GraphBuildConfig
from hiercp.spatial import (
    AdaptiveRoiBudgetError,
    CanonicalGraphUnavailable,
    EmptyCanonicalNodeError,
)
from hiercp.hierarchy import build_patient_graph, build_prototype_graph


CACHE_FORMAT = "full-cache"
CACHE_INDEX_FORMAT = "full-index"
CACHE_COMPLETE_FORMAT = "full-complete"
PROTOTYPE_METADATA_FORMAT = "hiercp_prototype_metadata_v3_sha256"
CACHE_PROGRESS_FORMAT = "hiercp_cache_progress_csv_v2_sha256"
CACHE_PROGRESS_COLUMNS = (
    "case_id",
    "sample_index",
    "split",
    "status",
    "path",
    "candidates",
    "easy",
    "inter",
    "intra_corrupted",
    "message",
    "config_fingerprint",
    "source_image_sha256",
    "source_label_sha256",
    "artifact_sha256",
    "file_size",
    "updated_at",
)

CACHE_RUN_MODES = ("production", "ablation", "benchmark", "debug")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_case_ids(
    values: Sequence[object],
    *,
    context: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, value in enumerate(values):
        case_id = str(value)
        if not case_id or case_id != case_id.strip():
            raise ValueError(
                f"{context} contains a blank or whitespace-padded case ID at "
                f"index {index}: {value!r}"
            )
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
        result.append(case_id)
    if duplicates:
        raise ValueError(f"{context} contains duplicate case IDs: {sorted(duplicates)}")
    if not allow_empty and not result:
        raise ValueError(f"{context} must not be empty")
    return tuple(result)


def _source_contract(
    case_paths: Sequence[object],
    ordered_case_ids: Sequence[str],
) -> list[dict[str, str]]:
    by_id: dict[str, object] = {}
    for paths in case_paths:
        case_id = str(paths.case_id)
        if case_id in by_id:
            raise ValueError(f"Discovered duplicate case ID: {case_id!r}")
        by_id[case_id] = paths
    expected = set(str(value) for value in ordered_case_ids)
    actual = set(by_id)
    if actual != expected:
        raise ValueError(
            "Discovered source cohort mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return [
        {
            "case_id": str(case_id),
            "image_sha256": _sha256_file(Path(by_id[str(case_id)].image_path)),
            "label_sha256": _sha256_file(Path(by_id[str(case_id)].label_path)),
        }
        for case_id in ordered_case_ids
    ]


def _assert_file_sha256(path: Path, expected: object, *, context: str) -> None:
    fingerprint = str(expected or "")
    if _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"{context} has no valid SHA-256 contract for {path}")
    if not path.is_file():
        raise ValueError(f"{context} artifact is missing: {path}")
    actual = _sha256_file(path)
    if actual != fingerprint:
        raise ValueError(
            f"{context} SHA-256 mismatch for {path}: "
            f"expected={fingerprint}, actual={actual}"
        )


def _validate_prototype_manifest(
    path: Path,
    sources: Sequence[dict[str, str]],
) -> None:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Prototype manifest has no header: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"Cannot read prototype manifest {path}: {exc}") from exc
    ids = _validated_case_ids(
        [row.get("case_id", "") for row in rows],
        context=f"Prototype manifest {path}",
        allow_empty=False,
    )
    expected = {row["case_id"]: row for row in sources}
    if set(ids) != set(expected) or len(rows) != len(expected):
        raise ValueError(
            "Prototype manifest cohort mismatch: "
            f"expected={sorted(expected)}, actual={sorted(ids)}"
        )
    for row in rows:
        case_id = str(row["case_id"])
        source = expected[case_id]
        if row.get("status") != "ok":
            raise ValueError(
                f"Prototype manifest contains a non-success row for {case_id}: "
                f"{row.get('status')!r}"
            )
        for key in ("image_sha256", "label_sha256"):
            if row.get(key) != source[key]:
                raise ValueError(
                    f"Prototype manifest source mismatch for {case_id}: {key}"
                )


def _atomic_torch_save(
    payload: object,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    "Refusing to replace an existing hierarchical-cache artifact "
                    f"without explicit overwrite authorization: {path}"
                ) from exc
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _atomic_manifest_save(rows: list[dict[str, object]], path: Path) -> None:
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
        if temporary.exists():
            temporary.unlink()


def _load_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata root must be an object: {path}")
    return payload


def _assert_metadata_equal(actual: dict, expected: dict, *, context: str) -> None:
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        details = ", ".join(
            f"{key}: actual={actual.get(key)!r} expected={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(
            f"{context} metadata does not match the current request ({details}). "
            "Use --overwrite or a separate workspace."
        )



def _torch_load_cpu(path: Path) -> object:
    """Load one cache payload on CPU across supported PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_recoverable_partial_cache(
    files: Sequence[Path],
    *,
    selected_case_ids: Sequence[str],
    split_lookup: dict[str, str],
    samples_per_case: int,
    total_candidates: int,
    prototype_fingerprint: str,
    config_fingerprint: str,
    source_by_id: dict[str, dict[str, str]],
    graph_config: dict,
    ct_clip: Sequence[float],
) -> None:
    """Validate orphan ``.pt`` files before adopting an interrupted cache.

    Older code wrote ``config.json`` only after every case finished. A process
    interruption could therefore leave valid atomic ``.pt`` files without the
    metadata file needed for resume. This function verifies the metadata and
    tensor cardinalities embedded in every existing payload before a new
    ``config.json`` is created. No file is changed when validation fails.
    """

    selected = {str(value) for value in selected_case_ids}
    expected_graph = dict(graph_config)
    expected_clip = tuple(float(value) for value in ct_clip)
    expected_fingerprint = str(prototype_fingerprint)
    problems: list[str] = []

    for path in files:
        local: list[str] = []
        try:
            payload = _torch_load_cpu(path)
        except Exception as exc:
            problems.append(f"{path.name}: cannot load ({type(exc).__name__}: {exc})")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{path.name}: payload root is not a dict")
            continue

        case_id = str(payload.get("case_id", ""))
        try:
            sample_index = int(payload.get("sample_index", -1))
        except (TypeError, ValueError):
            sample_index = -1
        split_name = str(payload.get("split", ""))

        if payload.get("format") != CACHE_FORMAT:
            local.append(
                f"format={payload.get('format')!r}, expected={CACHE_FORMAT!r}"
            )
        if payload.get("prototype_fingerprint") != expected_fingerprint:
            local.append("prototype fingerprint mismatch")
        if payload.get("config_fingerprint") != str(config_fingerprint):
            local.append("config fingerprint mismatch")
        if case_id not in selected:
            local.append(f"case_id={case_id!r} is outside the selected split")
        elif split_name != split_lookup.get(case_id):
            local.append(
                f"split={split_name!r}, expected={split_lookup.get(case_id)!r}"
            )
        if case_id in source_by_id:
            source = source_by_id[case_id]
            if payload.get("source_image_sha256") != source["image_sha256"]:
                local.append("source image SHA-256 mismatch")
            if payload.get("source_label_sha256") != source["label_sha256"]:
                local.append("source label SHA-256 mismatch")
        if not (0 <= sample_index < int(samples_per_case)):
            local.append(
                f"sample_index={sample_index}, expected 0..{int(samples_per_case) - 1}"
            )
        elif case_id:
            expected_name = f"{case_id}__{sample_index:03d}.pt"
            if path.name != expected_name:
                local.append(f"filename={path.name!r}, expected={expected_name!r}")
        if payload.get("graph_config") != expected_graph:
            local.append("graph_config mismatch")
        try:
            actual_clip = tuple(float(value) for value in payload.get("ct_clip", ()))
        except (TypeError, ValueError):
            actual_clip = ()
        if actual_clip != expected_clip:
            local.append(f"ct_clip={actual_clip!r}, expected={expected_clip!r}")

        target_locals = payload.get("target_locals")
        source_local = payload.get("source_local")
        if not isinstance(source_local, dict) or source_local.get("format") != "canonical-full-v22":
            local.append("source_local is missing or has an unsupported canonical-full format")
        elif not isinstance(source_local.get("nodes"), dict) or not isinstance(source_local.get("edges"), dict):
            local.append("source_local has no canonical nodes/edges")
        if not isinstance(target_locals, list) or len(target_locals) != int(total_candidates):
            local.append(
                "target_local count="
                f"{len(target_locals) if isinstance(target_locals, list) else 'invalid'}, "
                f"expected={int(total_candidates)}"
            )
        elif any(
            not isinstance(item, dict)
            or item.get("format") != "canonical-full-v22"
            or not isinstance(item.get("nodes"), dict)
            or not isinstance(item.get("edges"), dict)
            for item in target_locals
        ):
            local.append("one or more target_locals have no canonical full topology")

        for key in (
            "target_patches",
            "difficulties",
            "corruptions",
            "candidate_centers",
            "candidate_regions",
            "candidate_prototypes",
        ):
            value = payload.get(key)
            try:
                count = int(value.shape[0])
            except (AttributeError, IndexError, TypeError, ValueError):
                local.append(f"{key} has no valid leading dimension")
                continue
            if count != int(total_candidates):
                local.append(f"{key} count={count}, expected={int(total_candidates)}")

        if local:
            problems.append(f"{path.name}: " + "; ".join(local))

        del payload

    if problems:
        preview = "\n  - ".join(problems[:10])
        suffix = "" if len(problems) <= 10 else f"\n  ... and {len(problems) - 10} more"
        raise ValueError(
            "Hierarchical cache artifact validation failed. No cache file was modified.\n"
            f"  - {preview}{suffix}\n"
            "Use a separate workspace if these files came from a different configuration."
        )

def prepare_prototype_bank(
    *,
    data_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    region_cache_dir: str | os.PathLike[str],
    training_case_ids: Sequence[str],
    graph_config: GraphBuildConfig,
    liver_label: int,
    tumor_label: int,
    ct_clip: tuple[float, float],
    seed: int,
    overwrite: bool,
    workers: int,
) -> PrototypeBank:
    """Build population prototypes from training cases only.

    Patient-region data are stored once per case under ``region_cache_dir`` and
    then reused by hierarchical cache preparation.
    """

    output = Path(output_path)
    raw_requested = _validated_case_ids(
        training_case_ids,
        context="Prototype training cohort",
        allow_empty=False,
    )
    requested = tuple(sorted(raw_requested))
    case_paths = discover_cases(data_dir, case_ids=requested)
    sources = _source_contract(case_paths, requested)
    source_by_id = {row["case_id"]: row for row in sources}
    expected_metadata = {
        "format": PROTOTYPE_METADATA_FORMAT,
        "integrity_format": "sha256_v1",
        "training_cases": list(requested),
        "source_cases": sources,
        "seed": int(seed),
        "graph_config": graph_config.to_dict(),
        "labels": {"liver": int(liver_label), "tumor": int(tumor_label)},
        "ct_clip": [float(value) for value in ct_clip],
        "region_cache_format": REGION_CACHE_FORMAT,
    }
    metadata_path = output.parent / "metadata.json"
    manifest_path = output.parent / "manifest.csv"
    if (output.exists() or output.is_symlink()) and not output.is_file():
        raise FileExistsError(f"Prototype output is not a regular file: {output}")
    if output.is_file() and not overwrite:
        metadata = _load_json_object(metadata_path)
        _assert_metadata_equal(metadata, expected_metadata, context="Prototype bank")
        if metadata.get("state") != "ready":
            raise ValueError(
                "Prototype metadata is not in a published ready state; use "
                "--overwrite after confirming the exact output path, or use a "
                "separate workspace"
            )
        _assert_file_sha256(
            output,
            metadata.get("prototype_sha256"),
            context="Prototype bank",
        )
        _assert_file_sha256(
            manifest_path,
            metadata.get("manifest_sha256"),
            context="Prototype manifest",
        )
        _validate_prototype_manifest(manifest_path, sources)
        bank = PrototypeBank.load(output)
        stored_cases = tuple(
            sorted(
                _validated_case_ids(
                    bank.training_case_ids,
                    context="Stored prototype training cohort",
                    allow_empty=False,
                )
            )
        )
        if stored_cases != requested:
            raise ValueError(
                "Existing prototype bank was fitted on different training cases; "
                "use --overwrite or a separate workspace"
            )
        if metadata.get("prototype_fingerprint") != bank.fingerprint():
            raise ValueError("Prototype metadata fingerprint does not match bank contents")
        print(f"[Skip] Prototype bank exists: {output}")
        return bank
    stale_sidecars = [
        path
        for path in (metadata_path, manifest_path)
        if path.exists() or path.is_symlink()
    ]
    if not output.is_file() and stale_sidecars and not overwrite:
        raise FileExistsError(
            "Prototype output is missing but stale sidecars already exist: "
            f"{stale_sidecars}. Use --overwrite after confirming these exact paths, "
            "or choose a separate workspace."
        )

    groups: list[tuple[str, np.ndarray]] = []
    rows: list[dict[str, object]] = []

    def prepare_case(paths):
        source = source_by_id[paths.case_id]
        try:
            case = load_case(paths)
            region_seed = stable_case_seed(seed, paths.case_id, REGION_CACHE_SEED_SALT)
            regions = load_or_build_patient_regions(
                case,
                cache_dir=region_cache_dir,
                liver_label=liver_label,
                tumor_label=tumor_label,
                config=graph_config,
                seed=region_seed,
                ct_clip=ct_clip,
                overwrite=overwrite,
            )
            return (
                paths.case_id,
                np.asarray(regions.region_features, dtype=np.float32).copy(),
                {
                    "case_id": paths.case_id,
                    "status": "ok",
                    "regions": regions.num_regions,
                    "image_sha256": source["image_sha256"],
                    "label_sha256": source["label_sha256"],
                },
                "",
            )
        except Exception as exc:
            return (
                paths.case_id,
                None,
                {
                    "case_id": paths.case_id,
                    "status": "error",
                    "message": str(exc),
                    "image_sha256": source["image_sha256"],
                    "label_sha256": source["label_sha256"],
                },
                str(exc),
            )

    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError(f"Prototype preparation workers must be positive, got {workers}")
    if worker_count > 1:
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="hiercp-prototype"
        ) as executor:
            results = executor.map(prepare_case, case_paths)
            for case_id, features, row, message in results:
                rows.append(row)
                if features is not None:
                    groups.append((case_id, features))
                    print(f"[OK] prototype descriptors {case_id} regions={features.shape[0]}")
                else:
                    print(f"[Error] prototype descriptors {case_id}: {message}")
    else:
        for paths in case_paths:
            case_id, features, row, message = prepare_case(paths)
            rows.append(row)
            if features is not None:
                groups.append((case_id, features))
                print(f"[OK] prototype descriptors {case_id} regions={features.shape[0]}")
            else:
                print(f"[Error] prototype descriptors {case_id}: {message}")

    successful = {case_id for case_id, _ in groups}
    missing = sorted(set(requested) - successful)
    if missing:
        raise RuntimeError(
            "Prototype fitting requires every training case to produce region descriptors; "
            f"failed cases: {missing}"
        )
    if not groups:
        raise RuntimeError("No training-case region descriptors were produced")

    bank = build_prototype_bank(
        groups,
        config=graph_config,
        rng=np.random.default_rng(seed + 1009),
    )
    current_sources = _source_contract(case_paths, requested)
    if current_sources != sources:
        raise RuntimeError(
            "Prototype source files changed while descriptors were being prepared; "
            "no prototype publication was written. Retry with stable inputs."
        )
    bank.save(output, overwrite=overwrite)
    _atomic_manifest_save(rows, manifest_path)
    _atomic_json_save(
        {
            **expected_metadata,
            "data_dir": str(Path(data_dir).resolve()),
            "prototypes": bank.num_prototypes,
            "prototype_fingerprint": bank.fingerprint(),
            "prototype_sha256": _sha256_file(output),
            "manifest_sha256": _sha256_file(manifest_path),
            "state": "ready",
        },
        metadata_path,
    )
    print(f"[OK] prototype bank saved: {output} prototypes={bank.num_prototypes}")
    return bank


def build_training_sample(
    case: LoadedCase,
    bank: PrototypeBank,
    regions: PatientRegionData,
    *,
    sample_index: int,
    split_name: str,
    graph_config: GraphBuildConfig,
    liver_label: int,
    tumor_label: int,
    source_selection: str,
    source_pad: int,
    total_candidates: int,
    candidate_pool_size: int,
    easy_fraction: float,
    inter_fraction: float,
    intra_fraction: float,
    max_draws: int,
    min_liver_coverage: float,
    occupied_clearance_vox: int,
    min_center_separation_mm: float,
    ct_clip: tuple[float, float],
    seed: int,
) -> dict | None:
    rng = np.random.default_rng(
        stable_case_seed(seed, case.paths.case_id, f"sample_{sample_index}")
    )
    source, _, _ = choose_source_tumor(
        case.image,
        case.label,
        tumor_label=tumor_label,
        rng=rng,
        selection=source_selection,
        pad=source_pad,
    )
    placement_mask = case.label == int(liver_label)
    occupied = case.label == int(tumor_label)
    candidates, _ = build_candidate_pool(
        case,
        source,
        placement_mask=placement_mask,
        full_organ_mask=regions.full_organ_mask,
        occupied_mask=occupied,
        organ_distance=regions.organ_depth,
        rng=rng,
        num_candidates=max(int(candidate_pool_size), int(total_candidates) - 1),
        max_draws=max_draws,
        min_liver_coverage=min_liver_coverage,
        occupied_clearance_vox=occupied_clearance_vox,
        min_center_separation_mm=min_center_separation_mm,
    )
    source_rng = np.random.default_rng(
        stable_case_seed(seed, case.paths.case_id, f"local_{sample_index}")
    )
    prepared_source = prepare_local_source(
        case,
        source,
        full_organ_mask=regions.full_organ_mask,
        organ_depth=regions.organ_depth,
        config=graph_config,
        rng=source_rng,
        ct_clip=ct_clip,
    )

    # Candidate-pool constraints are cheaper than canonical graph construction,
    # so one anatomically valid center can still lack a required deep-parenchyma
    # context set.  Reject only that center and deterministically rebuild the
    # curriculum from the remaining pool; never fabricate context nodes or
    # rebuild a different topology after sampling.
    viable_candidates = list(candidates)
    rejected_centers: set[tuple[int, int, int]] = set()
    specs = None
    built_local = None
    while len(viable_candidates) >= int(total_candidates) - 1:
        specs = build_training_specs(
            case,
            source,
            viable_candidates,
            regions,
            bank,
            total_candidates=total_candidates,
            easy_fraction=easy_fraction,
            inter_fraction=inter_fraction,
            intra_fraction=intra_fraction,
            tumor_label=tumor_label,
            config=graph_config,
            rng=rng,
        )
        if specs is None:
            return None

        current_locals = []
        failed_center: tuple[int, int, int] | None = None
        for spec_index, spec in enumerate(specs):
            try:
                current_locals.append(
                    build_local_graph(
                        case,
                        source,
                        spec,
                        full_organ_mask=regions.full_organ_mask,
                        organ_depth=regions.organ_depth,
                        config=graph_config,
                        rng=source_rng,
                        ct_clip=ct_clip,
                        prepared_source=prepared_source,
                    )
                )
            except EmptyCanonicalNodeError:
                # Index zero is the real positive location.  If that semantic
                # graph is impossible, the complete sample is unusable.
                if spec_index == 0:
                    return None
                failed_center = tuple(int(value) for value in spec.center)
                break
        if failed_center is None:
            built_local = current_locals
            break
        rejected_centers.add(failed_center)
        viable_candidates = [
            candidate
            for candidate in viable_candidates
            if tuple(int(value) for value in candidate.center) not in rejected_centers
        ]

    if specs is None or built_local is None:
        return None
    patient_graph = build_patient_graph(
        case,
        source,
        specs,
        regions,
        tumor_label=tumor_label,
        config=graph_config,
        ct_clip=ct_clip,
    )
    prototype_graph = build_prototype_graph(
        specs,
        patient_graph,
        regions,
        bank,
        config=graph_config,
    )
    return {
        "format": CACHE_FORMAT,
        "prototype_fingerprint": bank.fingerprint(),
        "case_id": case.paths.case_id,
        "sample_index": int(sample_index),
        "split": str(split_name),
        "source_component": int(source.component_id),
        # Keep half precision in RAM and during H2D transfer; conversion, when
        # required, happens on the accelerator inside the model.
        "source_patch": torch.from_numpy(prepared_source.source_patch.astype(np.float16)),
        "target_patches": torch.from_numpy(
            np.stack([item.target_patch for item in built_local]).astype(np.float16)
        ),
        "source_local": built_local[0].source_local,
        "target_locals": [item.target_local for item in built_local],
        "patient_graph": patient_graph,
        "prototype_graph": prototype_graph,
        "difficulties": torch.tensor([spec.difficulty for spec in specs], dtype=torch.long),
        "corruptions": torch.tensor([spec.corruption for spec in specs], dtype=torch.long),
        "candidate_centers": torch.tensor([spec.center for spec in specs], dtype=torch.long),
        "candidate_regions": torch.tensor([spec.region_id for spec in specs], dtype=torch.long),
        "candidate_prototypes": torch.tensor(
            [spec.prototype_id for spec in specs], dtype=torch.long
        ),
        "graph_config": graph_config.to_dict(),
        "ct_clip": tuple(float(value) for value in ct_clip),
    }


def _cache_config_fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _progress_key(case_id: str, sample_index: int | None) -> tuple[str, int | None]:
    return str(case_id), None if sample_index is None else int(sample_index)


def _parse_progress_sample_index(value: object) -> int | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid sample_index in cache manifest: {value!r}") from exc


def _progress_row_key(row: dict[str, object]) -> tuple[str, int | None]:
    return _progress_key(
        str(row.get("case_id", "")),
        _parse_progress_sample_index(row.get("sample_index")),
    )


def _load_progress_manifest(path: Path) -> dict[tuple[str, int | None], dict[str, str]]:
    if not path.is_file():
        return {}
    records: dict[tuple[str, int | None], dict[str, str]] = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return {}
            for row_number, raw in enumerate(reader, start=2):
                case_id = str(raw.get("case_id", "")).strip()
                if not case_id:
                    raise ValueError(
                        f"Blank case_id in cache manifest {path}:{row_number}"
                    )
                sample_index = _parse_progress_sample_index(raw.get("sample_index"))
                key = _progress_key(case_id, sample_index)
                if key in records:
                    raise ValueError(
                        f"Duplicate cache manifest key {key!r} at "
                        f"{path}:{row_number}"
                    )
                normalized = {
                    column: str(raw.get(column, "") or "")
                    for column in CACHE_PROGRESS_COLUMNS
                }
                normalized["case_id"] = case_id
                normalized["sample_index"] = (
                    "" if sample_index is None else str(sample_index)
                )
                records[key] = normalized
    except (OSError, csv.Error) as exc:
        raise ValueError(f"Cannot read hierarchical cache manifest {path}: {exc}") from exc
    return records


def _progress_sort_key(row: dict[str, str]) -> tuple[str, int]:
    sample_index = _parse_progress_sample_index(row.get("sample_index"))
    return str(row.get("case_id", "")), -1 if sample_index is None else sample_index


def _atomic_progress_manifest_save(
    records: dict[tuple[str, int | None], dict[str, str]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CACHE_PROGRESS_COLUMNS))
        writer.writeheader()
        for row in sorted(records.values(), key=_progress_sort_key):
            writer.writerow({column: row.get(column, "") for column in CACHE_PROGRESS_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _progress_row(
    *,
    case_id: str,
    sample_index: int | None,
    split_name: str,
    status: str,
    config_fingerprint: str,
    path: str = "",
    candidates: int | str = "",
    easy: int | str = "",
    inter: int | str = "",
    intra_corrupted: int | str = "",
    message: str = "",
    source_image_sha256: str = "",
    source_label_sha256: str = "",
    artifact_sha256: str = "",
    file_size: int | str = "",
) -> dict[str, str]:
    return {
        "case_id": str(case_id),
        "sample_index": "" if sample_index is None else str(int(sample_index)),
        "split": str(split_name),
        "status": str(status),
        "path": str(path),
        "candidates": str(candidates),
        "easy": str(easy),
        "inter": str(inter),
        "intra_corrupted": str(intra_corrupted),
        "message": str(message),
        "config_fingerprint": str(config_fingerprint),
        "source_image_sha256": str(source_image_sha256),
        "source_label_sha256": str(source_label_sha256),
        "artifact_sha256": str(artifact_sha256),
        "file_size": str(file_size),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _row_matches_current_config(row: dict[str, str], fingerprint: str) -> bool:
    recorded = str(row.get("config_fingerprint", ""))
    return bool(recorded) and recorded == str(fingerprint)


def _validate_progress_contract(
    records: dict[tuple[str, int | None], dict[str, str]],
    *,
    selected_case_ids: Sequence[str],
    split_lookup: dict[str, str],
    samples_per_case: int,
    fingerprint: str,
    source_by_id: dict[str, dict[str, str]],
) -> None:
    selected = set(str(value) for value in selected_case_ids)
    for (case_id, sample_index), row in records.items():
        if case_id not in selected:
            raise ValueError(
                f"Cache manifest contains out-of-cohort case {case_id!r}"
            )
        if not _row_matches_current_config(row, fingerprint):
            raise ValueError(
                "Cache manifest contains a stale or incomplete contract for "
                f"{case_id!r}; use --overwrite or a separate workspace"
            )
        if str(row.get("split", "")) != split_lookup[case_id]:
            raise ValueError(f"Cache manifest split mismatch for {case_id!r}")
        if sample_index is not None and not 0 <= sample_index < int(samples_per_case):
            raise ValueError(
                f"Cache manifest sample index is out of range: "
                f"{(case_id, sample_index)!r}"
            )
        source = source_by_id[case_id]
        expected_hashes = {
            "source_image_sha256": source["image_sha256"],
            "source_label_sha256": source["label_sha256"],
        }
        for key, expected in expected_hashes.items():
            if str(row.get(key, "")) != expected:
                raise ValueError(
                    f"Cache manifest source SHA mismatch for {case_id!r}: {key}"
                )


def _sample_row_is_complete(
    row: dict[str, str] | None,
    *,
    root: Path,
    expected_output: Path,
    fingerprint: str,
) -> bool:
    if row is None or not _row_matches_current_config(row, fingerprint):
        return False
    status = str(row.get("status", ""))
    if status != "ok":
        return False
    path_text = str(row.get("path", "")) or expected_output.name
    actual = root / path_text
    if actual != expected_output or not actual.is_file():
        return False
    try:
        size = int(actual.stat().st_size)
    except OSError:
        return False
    if size <= 0:
        return False
    recorded_size = str(row.get("file_size", "")).strip()
    if recorded_size:
        try:
            if int(recorded_size) != size:
                return False
        except ValueError:
            return False
    recorded_sha256 = str(row.get("artifact_sha256", ""))
    if _SHA256_RE.fullmatch(recorded_sha256) is None:
        return False
    try:
        if _sha256_file(actual) != recorded_sha256:
            return False
    except OSError:
        return False
    return True


def _adopt_existing_cache_files(
    *,
    root: Path,
    records: dict[tuple[str, int | None], dict[str, str]],
    selected_case_ids: Sequence[str],
    split_lookup: dict[str, str],
    samples_per_case: int,
    fingerprint: str,
) -> int:
    """Fail closed on artifacts that lack an exact current manifest record."""

    selected = {str(value) for value in selected_case_ids}
    pattern = re.compile(r"^(?P<case>.+)__(?P<sample>\d{3})\.pt$")
    problems: list[str] = []
    for path in root.glob("*.pt"):
        match = pattern.match(path.name)
        if match is None:
            problems.append(f"{path.name}: unsupported cache filename")
            continue
        case_id = match.group("case")
        sample_index = int(match.group("sample"))
        if case_id not in selected or not (0 <= sample_index < int(samples_per_case)):
            problems.append(
                f"{path.name}: outside the exact selected case/sample cohort"
            )
            continue
        key = _progress_key(case_id, sample_index)
        existing = records.get(key)
        if not _sample_row_is_complete(
            existing,
            root=root,
            expected_output=path,
            fingerprint=fingerprint,
        ):
            problems.append(
                f"{path.name}: missing or mismatched current SHA-256 manifest record"
            )
    if problems:
        preview = "; ".join(problems[:20])
        remainder = len(problems) - min(len(problems), 20)
        raise FileExistsError(
            "Untracked or unverifiable hierarchical-cache artifacts were found. "
            "Automatic adoption is forbidden because their source provenance cannot "
            f"be proven: {preview}"
            + (f"; additional={remainder}" if remainder else "")
            + ". Pass --overwrite only after confirming this exact cache directory, "
            "or use a separate workspace."
        )
    return 0


def _progress_entries(
    *,
    root: Path,
    records: dict[tuple[str, int | None], dict[str, str]],
    selected_case_ids: Sequence[str],
    split_lookup: dict[str, str],
    samples_per_case: int,
    fingerprint: str,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for case_id in selected_case_ids:
        for sample_index in range(int(samples_per_case)):
            output = root / f"{case_id}__{sample_index:03d}.pt"
            row = records.get(_progress_key(case_id, sample_index))
            if not _sample_row_is_complete(
                row,
                root=root,
                expected_output=output,
                fingerprint=fingerprint,
            ):
                continue
            entries.append(
                {
                    "path": output.name,
                    "case_id": str(case_id),
                    "sample_index": int(sample_index),
                    "split": split_lookup[str(case_id)],
                    "source_image_sha256": str(row["source_image_sha256"]),
                    "source_label_sha256": str(row["source_label_sha256"]),
                    "artifact_sha256": str(row["artifact_sha256"]),
                    "file_size": int(row["file_size"]),
                }
            )
    return entries


def _progress_is_complete(
    *,
    root: Path,
    records: dict[tuple[str, int | None], dict[str, str]],
    selected_case_ids: Sequence[str],
    samples_per_case: int,
    fingerprint: str,
) -> bool:
    for case_id in selected_case_ids:
        case_row = records.get(_progress_key(str(case_id), None))
        if (
            case_row is None
            or not _row_matches_current_config(case_row, fingerprint)
            or str(case_row.get("status", "")) != "ok"
        ):
            return False
        for sample_index in range(int(samples_per_case)):
            output = root / f"{case_id}__{sample_index:03d}.pt"
            if not _sample_row_is_complete(
                records.get(_progress_key(case_id, sample_index)),
                root=root,
                expected_output=output,
                fingerprint=fingerprint,
            ):
                return False
    return True


def validate_cache_publication(
    cache_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Load an exact, SHA-bound cache publication or fail closed."""

    root = Path(cache_dir)
    config_path = root / "config.json"
    manifest_path = root / "manifest.csv"
    index_path = root / "index.json"
    complete_path = root / "complete.json"
    config = _load_json_object(config_path)
    index = _load_json_object(index_path)
    complete = _load_json_object(complete_path)
    if config.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache config: {config_path}")
    if config.get("integrity_format") != "sha256_v1":
        raise ValueError(
            f"Cache config has no supported integrity contract: {config_path}"
        )
    if config.get("index_format") != CACHE_INDEX_FORMAT:
        raise ValueError(f"Cache config index format is unsupported: {config_path}")
    if config.get("progress_format") != CACHE_PROGRESS_FORMAT:
        raise ValueError(f"Cache progress format is unsupported: {config_path}")
    run_mode = config.get("run_mode")
    if run_mode not in CACHE_RUN_MODES:
        raise ValueError(
            f"Cache config has an unsupported run_mode={run_mode!r}: {config_path}"
        )
    expected_state = "ready" if run_mode == "production" else "ready_nonproduction"
    if config.get("state") != expected_state:
        raise ValueError(
            "Cache config state/run_mode contract is invalid: "
            f"run_mode={run_mode!r}, expected_state={expected_state!r}, "
            f"actual_state={config.get('state')!r}"
        )
    subset_active = config.get("subset_active")
    if type(subset_active) is not bool:
        raise ValueError(f"Cache config subset_active must be a boolean: {config_path}")
    if index.get("format") != CACHE_INDEX_FORMAT or index.get("cache_format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache index: {index_path}")
    if complete.get("format") != CACHE_COMPLETE_FORMAT or complete.get("cache_format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache completion marker: {complete_path}")

    fingerprint = str(config.get("config_fingerprint", ""))
    if _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"Cache config fingerprint is missing or invalid: {config_path}")
    if index.get("config_fingerprint") != fingerprint:
        raise ValueError("Cache index/config fingerprint mismatch")
    if complete.get("config_fingerprint") != fingerprint:
        raise ValueError("Cache complete/config fingerprint mismatch")
    for key in ("prototype_fingerprint", "run_mode"):
        if index.get(key) != config.get(key):
            raise ValueError(f"Cache index/config mismatch: {key}")
        if complete.get(key) != config.get(key):
            raise ValueError(f"Cache complete/config mismatch: {key}")
    _assert_file_sha256(
        index_path,
        complete.get("index_sha256"),
        context="Cache publication index",
    )
    _assert_file_sha256(
        manifest_path,
        complete.get("manifest_sha256"),
        context="Cache publication manifest",
    )
    _assert_file_sha256(
        config_path,
        complete.get("config_sha256"),
        context="Cache publication config",
    )

    train_ids = _validated_case_ids(
        config.get("train_case_ids", ()),
        context="Published cache training split",
        allow_empty=False,
    )
    val_ids = _validated_case_ids(
        config.get("val_case_ids", ()),
        context="Published cache validation split",
    )
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Published cache splits overlap: {overlap}")
    selected_ids = _validated_case_ids(
        config.get("selected_case_ids", ()),
        context="Published cache selected cohort",
        allow_empty=False,
    )
    split_lookup = {case_id: "train" for case_id in train_ids}
    split_lookup.update({case_id: "val" for case_id in val_ids})
    unknown_selected = sorted(set(selected_ids) - set(split_lookup))
    if unknown_selected:
        raise ValueError(
            f"Published cache selected cohort is outside configured splits: {unknown_selected}"
        )
    if run_mode == "production":
        expected_selected = (*train_ids, *val_ids)
        if subset_active:
            raise ValueError("Production cache publication forbids an active case subset")
        if selected_ids != expected_selected:
            raise ValueError(
                "Production cache publication must contain the exact ordered train+val "
                "cohort: "
                f"expected={list(expected_selected)!r}, actual={list(selected_ids)!r}"
            )
    try:
        samples_per_case = int(config["samples_per_case"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Published cache has invalid samples_per_case") from exc
    if samples_per_case <= 0:
        raise ValueError("Published cache samples_per_case must be positive")

    raw_sources = config.get("source_cases")
    if not isinstance(raw_sources, list):
        raise ValueError("Published cache has no source_cases integrity contract")
    source_ids = _validated_case_ids(
        [row.get("case_id", "") if isinstance(row, dict) else "" for row in raw_sources],
        context="Published cache source contract",
        allow_empty=False,
    )
    if tuple(source_ids) != tuple(selected_ids):
        raise ValueError(
            "Published cache source contract does not exactly match selected cohort"
        )
    source_by_id: dict[str, dict[str, str]] = {}
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError("Published cache source contract contains a non-object row")
        case_id = str(raw["case_id"])
        source = {
            "case_id": case_id,
            "image_sha256": str(raw.get("image_sha256", "")),
            "label_sha256": str(raw.get("label_sha256", "")),
        }
        if any(
            _SHA256_RE.fullmatch(source[key]) is None
            for key in ("image_sha256", "label_sha256")
        ):
            raise ValueError(f"Published cache source SHA is invalid for {case_id}")
        source_by_id[case_id] = source

    records = _load_progress_manifest(manifest_path)
    _validate_progress_contract(
        records,
        selected_case_ids=selected_ids,
        split_lookup=split_lookup,
        samples_per_case=samples_per_case,
        fingerprint=fingerprint,
        source_by_id=source_by_id,
    )
    for case_id in selected_ids:
        case_row = records.get(_progress_key(case_id, None))
        if (
            case_row is None
            or not _row_matches_current_config(case_row, fingerprint)
            or str(case_row.get("status", "")) != "ok"
        ):
            status = None if case_row is None else case_row.get("status")
            raise ValueError(
                "Published cache has no successful terminal case row for "
                f"{case_id!r}: status={status!r}"
            )
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Cache index has no entries list: {index_path}")
    expected_keys = {
        (case_id, sample_index)
        for case_id in selected_ids
        for sample_index in range(samples_per_case)
    }
    actual_keys: set[tuple[str, int]] = set()
    actual_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Cache index contains a non-object entry")
        case_id = str(entry.get("case_id", ""))
        try:
            sample_index = int(entry.get("sample_index"))
            recorded_size = int(entry.get("file_size"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cache index entry has invalid numeric fields: {entry}") from exc
        key = (case_id, sample_index)
        if key in actual_keys:
            raise ValueError(f"Cache index contains duplicate entry key: {key!r}")
        actual_keys.add(key)
        path_text = str(entry.get("path", ""))
        if (
            not path_text
            or Path(path_text).is_absolute()
            or Path(path_text).name != path_text
        ):
            raise ValueError(f"Cache index contains an unsafe artifact path: {path_text!r}")
        if path_text in actual_names:
            raise ValueError(f"Cache index contains duplicate artifact path: {path_text}")
        actual_names.add(path_text)
        if key not in expected_keys:
            raise ValueError(f"Cache index contains an out-of-cohort entry: {key!r}")
        if entry.get("split") != split_lookup[case_id]:
            raise ValueError(f"Cache index split mismatch for {key!r}")
        source = source_by_id[case_id]
        if entry.get("source_image_sha256") != source["image_sha256"]:
            raise ValueError(f"Cache index source image SHA mismatch for {key!r}")
        if entry.get("source_label_sha256") != source["label_sha256"]:
            raise ValueError(f"Cache index source label SHA mismatch for {key!r}")
        artifact_sha256 = str(entry.get("artifact_sha256", ""))
        if _SHA256_RE.fullmatch(artifact_sha256) is None:
            raise ValueError(f"Cache index artifact SHA is invalid for {key!r}")
        artifact = root / path_text
        if not artifact.is_file():
            raise FileNotFoundError(f"Indexed cache artifact is missing: {artifact}")
        if recorded_size <= 0 or artifact.stat().st_size != recorded_size:
            raise ValueError(f"Indexed cache artifact size mismatch: {artifact}")
        _assert_file_sha256(
            artifact,
            artifact_sha256,
            context="Indexed cache artifact",
        )
        row = records.get(_progress_key(case_id, sample_index))
        if row is None or row.get("status") != "ok":
            raise ValueError(f"Cache index has no successful manifest row for {key!r}")
        expected_row_fields = {
            "path": path_text,
            "split": split_lookup[case_id],
            "source_image_sha256": source["image_sha256"],
            "source_label_sha256": source["label_sha256"],
            "artifact_sha256": artifact_sha256,
            "file_size": str(recorded_size),
        }
        mismatches = [
            name
            for name, value in expected_row_fields.items()
            if str(row.get(name, "")) != str(value)
        ]
        if mismatches:
            raise ValueError(
                f"Cache index/manifest mismatch for {key!r}: {mismatches}"
            )
    if actual_keys != expected_keys:
        raise ValueError(
            "Cache index entry cohort is not exact: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    disk_names = {path.name for path in root.glob("*.pt")}
    if disk_names != actual_names:
        raise ValueError(
            "Published cache artifact cohort is not exact: "
            f"missing={sorted(actual_names - disk_names)}, "
            f"extra={sorted(disk_names - actual_names)}"
        )
    expected_count = len(expected_keys)
    if index.get("expected_entries") != expected_count:
        raise ValueError("Cache index expected_entries is inconsistent")
    marker_expected = {
        "selected_case_ids": list(selected_ids),
        "samples_per_case": samples_per_case,
        "expected_entries": expected_count,
        "entries": expected_count,
    }
    marker_mismatches = [
        key for key, value in marker_expected.items() if complete.get(key) != value
    ]
    if marker_mismatches:
        raise ValueError(
            f"Cache completion marker is inconsistent: {marker_mismatches}"
        )
    return index


def prepare_hierarchical_cache(
    *,
    data_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    region_cache_dir: str | os.PathLike[str],
    bank_path: str | os.PathLike[str],
    train_case_ids: Sequence[str],
    val_case_ids: Sequence[str],
    graph_config: GraphBuildConfig,
    liver_label: int,
    tumor_label: int,
    source_selection: str,
    source_pad: int,
    samples_per_case: int,
    total_candidates: int,
    candidate_pool_size: int,
    easy_fraction: float,
    inter_fraction: float,
    intra_fraction: float,
    max_draws: int,
    min_liver_coverage: float,
    occupied_clearance_vox: int,
    min_center_separation_mm: float,
    ct_clip: tuple[float, float],
    seed: int,
    max_cases: int | None,
    overwrite: bool,
    workers: int,
    run_mode: str = "production",
) -> list[dict[str, object]]:
    graph_config.validate()
    mode = str(run_mode).strip().lower()
    if mode not in CACHE_RUN_MODES:
        raise ValueError(f"Unsupported cache run_mode={run_mode!r}; expected {CACHE_RUN_MODES}")
    train_ids = list(
        _validated_case_ids(
            train_case_ids,
            context="Cache training split",
            allow_empty=False,
        )
    )
    val_ids = list(
        _validated_case_ids(
            val_case_ids,
            context="Cache validation split",
        )
    )
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Training and validation cache splits overlap: {overlap}")
    if int(samples_per_case) <= 0:
        raise ValueError(f"samples_per_case must be positive, got {samples_per_case}")

    bank_file = Path(bank_path)
    bank = PrototypeBank.load(bank_file)
    bank_case_ids = _validated_case_ids(
        bank.training_case_ids,
        context="Prototype-bank training cohort",
        allow_empty=False,
    )
    if tuple(sorted(bank_case_ids)) != tuple(sorted(train_ids)):
        raise ValueError("Prototype bank training cases do not match requested split")

    all_case_ids = [*train_ids, *val_ids]
    if max_cases is not None:
        if mode == "production":
            raise ValueError(
                "Production cache preparation forbids max_cases. Use the full configured "
                "cohort, or explicitly select a non-production run_mode and separate cache."
            )
        case_limit = int(max_cases)
        if case_limit <= 0:
            raise ValueError(f"max_cases must be positive when supplied, got {max_cases}")
        all_case_ids = all_case_ids[:case_limit]
    if not all_case_ids:
        raise ValueError("Hierarchical cache selected case cohort is empty")
    split_lookup = {value: "train" for value in train_ids}
    split_lookup.update({value: "val" for value in val_ids})
    case_paths = discover_cases(data_dir, case_ids=all_case_ids, run_mode=mode)
    case_paths_by_id = {paths.case_id: paths for paths in case_paths}
    sources = _source_contract(case_paths, all_case_ids)
    source_by_id = {row["case_id"]: row for row in sources}

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected_cache_config = {
        "format": CACHE_FORMAT,
        "integrity_format": "sha256_v1",
        "index_format": CACHE_INDEX_FORMAT,
        "region_cache_format": REGION_CACHE_FORMAT,
        "prototype_fingerprint": bank.fingerprint(),
        "prototype_artifact_sha256": _sha256_file(bank_file),
        "source_cases": sources,
        "train_case_ids": train_ids,
        "val_case_ids": val_ids,
        "selected_case_ids": list(all_case_ids),
        "run_mode": mode,
        "subset_active": max_cases is not None,
        "samples_per_case": int(samples_per_case),
        "total_candidates": int(total_candidates),
        "candidate_pool_size": int(candidate_pool_size),
        "difficulty_fractions": {
            "easy": float(easy_fraction),
            "inter": float(inter_fraction),
            "intra_corrupted": float(intra_fraction),
        },
        "source_selection": str(source_selection),
        "source_pad": int(source_pad),
        "max_draws": int(max_draws),
        "min_liver_coverage": float(min_liver_coverage),
        "occupied_clearance_vox": int(occupied_clearance_vox),
        "min_center_separation_mm": float(min_center_separation_mm),
        "labels": {"liver": int(liver_label), "tumor": int(tumor_label)},
        "graph_config": graph_config.to_dict(),
        "ct_clip": [float(value) for value in ct_clip],
        "seed": int(seed),
    }
    fingerprint = _cache_config_fingerprint(expected_cache_config)
    config_path = root / "config.json"
    manifest_path = root / "manifest.csv"
    index_path = root / "index.json"
    complete_path = root / "complete.json"
    config_payload = {
        **expected_cache_config,
        "data_dir": str(Path(data_dir).resolve()),
        "prototype_bank": str(Path(bank_path).resolve()),
        "region_cache_dir": str(Path(region_cache_dir).resolve()),
        "progress_format": CACHE_PROGRESS_FORMAT,
        "config_fingerprint": fingerprint,
    }

    existing_files = sorted(root.glob("*.pt"))
    if overwrite:
        for path in existing_files:
            path.unlink()
        for name in ("manifest.csv", "config.json", "index.json", "complete.json"):
            target = root / name
            if target.exists() or target.is_symlink():
                target.unlink()
        existing_files = []
    sidecar_paths = (config_path, manifest_path, index_path, complete_path)
    invalid_sidecars = [
        path
        for path in sidecar_paths
        if (path.exists() or path.is_symlink()) and not path.is_file()
    ]
    if invalid_sidecars:
        raise FileExistsError(
            "Cache metadata paths must be regular files; refusing to replace: "
            f"{invalid_sidecars}"
        )
    publication_present_at_start = any(
        path.exists() or path.is_symlink()
        for path in (index_path, complete_path)
    )

    if config_path.is_file():
        _assert_metadata_equal(
            _load_json_object(config_path),
            expected_cache_config,
            context="Hierarchical cache",
        )
    if not config_path.is_file() and existing_files:
        raise FileExistsError(
            "Hierarchical cache artifacts exist without an exact SHA-256 config "
            f"contract: {root}. Automatic adoption is forbidden. Pass --overwrite "
            "only after confirming this exact cache directory, or use a separate "
            "workspace."
        )
    elif not config_path.is_file():
        _atomic_json_save({**config_payload, "state": "building"}, config_path)
        print(f"[Init] Hierarchical cache metadata: {config_path}")

    records = _load_progress_manifest(manifest_path)
    _validate_progress_contract(
        records,
        selected_case_ids=all_case_ids,
        split_lookup=split_lookup,
        samples_per_case=samples_per_case,
        fingerprint=fingerprint,
        source_by_id=source_by_id,
    )
    adopted = _adopt_existing_cache_files(
        root=root,
        records=records,
        selected_case_ids=all_case_ids,
        split_lookup=split_lookup,
        samples_per_case=samples_per_case,
        fingerprint=fingerprint,
    )
    if existing_files:
        _validate_recoverable_partial_cache(
            existing_files,
            selected_case_ids=all_case_ids,
            split_lookup=split_lookup,
            samples_per_case=samples_per_case,
            total_candidates=total_candidates,
            prototype_fingerprint=bank.fingerprint(),
            config_fingerprint=fingerprint,
            source_by_id=source_by_id,
            graph_config=graph_config.to_dict(),
            ct_clip=ct_clip,
        )
    if adopted or not manifest_path.is_file():
        _atomic_progress_manifest_save(records, manifest_path)
    if adopted:
        print(f"[Recover] Registered existing cache files in CSV: {adopted}")

    expected_entry_count = len(all_case_ids) * int(samples_per_case)

    def complete_entries() -> list[dict[str, object]] | None:
        if not _progress_is_complete(
            root=root,
            records=records,
            selected_case_ids=all_case_ids,
            samples_per_case=samples_per_case,
            fingerprint=fingerprint,
        ):
            return None
        entries = _progress_entries(
            root=root,
            records=records,
            selected_case_ids=all_case_ids,
            split_lookup=split_lookup,
            samples_per_case=samples_per_case,
            fingerprint=fingerprint,
        )
        if len(entries) != expected_entry_count:
            raise RuntimeError(
                "Cache progress claimed completion with an inexact artifact cohort: "
                f"expected={expected_entry_count}, actual={len(entries)}"
            )
        expected_paths = [root / str(entry["path"]) for entry in entries]
        expected_names = {path.name for path in expected_paths}
        actual_names = {path.name for path in root.glob("*.pt")}
        if actual_names != expected_names:
            raise RuntimeError(
                "Hierarchical cache artifact cohort is not exact: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
        _validate_recoverable_partial_cache(
            expected_paths,
            selected_case_ids=all_case_ids,
            split_lookup=split_lookup,
            samples_per_case=samples_per_case,
            total_candidates=total_candidates,
            prototype_fingerprint=bank.fingerprint(),
            config_fingerprint=fingerprint,
            source_by_id=source_by_id,
            graph_config=graph_config.to_dict(),
            ct_clip=ct_clip,
        )
        current_sources = _source_contract(case_paths, all_case_ids)
        if current_sources != sources:
            raise RuntimeError(
                "Hierarchical-cache source files changed during preparation; "
                "no complete publication was written. Retry with stable inputs."
            )
        _assert_file_sha256(
            bank_file,
            expected_cache_config["prototype_artifact_sha256"],
            context="Prototype bank used by hierarchical cache",
        )
        return sorted(entries, key=lambda item: str(item["path"]))

    def cache_index_payload(entries: list[dict[str, object]]) -> dict[str, object]:
        return {
            "format": CACHE_INDEX_FORMAT,
            "cache_format": CACHE_FORMAT,
            "prototype_fingerprint": bank.fingerprint(),
            "config_fingerprint": fingerprint,
            "run_mode": mode,
            "expected_entries": expected_entry_count,
            "entries": entries,
        }

    def publish_complete(entries: list[dict[str, object]]) -> None:
        if len(entries) != expected_entry_count:
            raise RuntimeError(
                f"Refusing to publish partial cache index: expected={expected_entry_count}, "
                f"actual={len(entries)}"
            )
        payload = cache_index_payload(entries)
        _atomic_json_save(payload, index_path)
        ready_state = "ready" if mode == "production" else "ready_nonproduction"
        _atomic_json_save({**config_payload, "state": ready_state}, config_path)
        _atomic_json_save(
            {
                "format": CACHE_COMPLETE_FORMAT,
                "cache_format": CACHE_FORMAT,
                "prototype_fingerprint": bank.fingerprint(),
                "config_fingerprint": fingerprint,
                "run_mode": mode,
                "selected_case_ids": list(all_case_ids),
                "samples_per_case": int(samples_per_case),
                "expected_entries": expected_entry_count,
                "entries": len(entries),
                "index_sha256": _sha256_file(index_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "config_sha256": _sha256_file(config_path),
            },
            complete_path,
        )
        published = validate_cache_publication(root)
        if published != payload:
            raise RuntimeError(
                "Published cache index does not match the verified in-memory cohort"
            )

    def invalidate_publication(*, state: str, message: str = "") -> None:
        if publication_present_at_start:
            raise FileExistsError(
                "Refusing to remove or replace a pre-existing cache publication "
                "without explicit --overwrite authorization"
            )
        for published_path in (complete_path, index_path):
            if published_path.exists():
                published_path.unlink()
        payload = {**config_payload, "state": state}
        if message:
            payload["failure_message"] = message
        _atomic_json_save(payload, config_path)

    try:
        verified_entries = complete_entries()
    except Exception as exc:
        if not publication_present_at_start:
            invalidate_publication(state="failed", message=str(exc))
        raise
    if verified_entries is not None:
        if publication_present_at_start:
            published = validate_cache_publication(root)
            expected_index = cache_index_payload(verified_entries)
            if published != expected_index:
                raise ValueError(
                    "Existing cache publication does not exactly match the current "
                    "verified cohort; pass --overwrite after confirming this cache "
                    "directory, or use a separate workspace"
                )
            action = "Skip"
        else:
            publish_complete(verified_entries)
            action = "Publish"
        print(
            f"[{action}] Hierarchical cache complete from CSV: {root} "
            f"entries={len(verified_entries)}"
        )
        return []
    if publication_present_at_start:
        raise FileExistsError(
            "A pre-existing cache index/completion marker is not backed by the exact "
            "current artifact cohort. It was left untouched. Pass --overwrite only "
            "after confirming this exact cache directory, or use a separate workspace."
        )
    invalidate_publication(state="building")

    pending_by_case: dict[str, list[int]] = {}
    for case_id in all_case_ids:
        case_id = str(case_id)
        pending: list[int] = []
        for sample_index in range(int(samples_per_case)):
            output = root / f"{case_id}__{sample_index:03d}.pt"
            if not _sample_row_is_complete(
                records.get(_progress_key(case_id, sample_index)),
                root=root,
                expected_output=output,
                fingerprint=fingerprint,
            ):
                pending.append(sample_index)
        case_row = records.get(_progress_key(case_id, None))
        case_terminal_ok = (
            case_row is not None
            and _row_matches_current_config(case_row, fingerprint)
            and str(case_row.get("status", "")) == "ok"
        )
        if pending or not case_terminal_ok:
            pending_by_case[case_id] = pending

    if not pending_by_case:
        raise RuntimeError(
            "CSV progress table found no pending work but cache completeness check failed"
        )

    pending_case_paths = [case_paths_by_id[case_id] for case_id in pending_by_case]

    def prepare_case(paths):
        local_rows: list[dict[str, str]] = []
        split_name = split_lookup[paths.case_id]
        pending_indices = list(pending_by_case[paths.case_id])
        source_record = source_by_id[paths.case_id]

        def case_progress_row(**kwargs) -> dict[str, str]:
            return _progress_row(
                source_image_sha256=source_record["image_sha256"],
                source_label_sha256=source_record["label_sha256"],
                **kwargs,
            )

        try:
            case = load_case(paths)
            if not np.any(case.label == int(tumor_label)):
                local_rows.append(
                    case_progress_row(
                        case_id=paths.case_id,
                        sample_index=None,
                        split_name=split_name,
                        status="no_tumor",
                        config_fingerprint=fingerprint,
                        message="tumor label is absent",
                    )
                )
                return local_rows
            region_seed = stable_case_seed(seed, paths.case_id, REGION_CACHE_SEED_SALT)
            regions = load_or_build_patient_regions(
                case,
                cache_dir=region_cache_dir,
                liver_label=liver_label,
                tumor_label=tumor_label,
                config=graph_config,
                seed=region_seed,
                ct_clip=ct_clip,
                overwrite=overwrite,
            )
        except AdaptiveRoiBudgetError as exc:
            local_rows.append(
                case_progress_row(
                    case_id=paths.case_id,
                    sample_index=None,
                    split_name=split_name,
                    status="resource_budget_error",
                    config_fingerprint=fingerprint,
                    message=str(exc),
                )
            )
            return local_rows
        except Exception as exc:
            local_rows.append(
                case_progress_row(
                    case_id=paths.case_id,
                    sample_index=None,
                    split_name=split_name,
                    status="load_error",
                    config_fingerprint=fingerprint,
                    message=str(exc),
                )
            )
            return local_rows

        for sample_index in pending_indices:
            output = root / f"{paths.case_id}__{sample_index:03d}.pt"
            if (output.exists() or output.is_symlink()) and not overwrite:
                local_rows.append(
                    case_progress_row(
                        case_id=paths.case_id,
                        sample_index=sample_index,
                        split_name=split_name,
                        status="invalid_existing_artifact",
                        config_fingerprint=fingerprint,
                        message=(
                            "cache artifact appeared without a committed current "
                            "SHA-256 manifest row; automatic adoption is forbidden"
                        ),
                    )
                )
                continue
            try:
                sample = build_training_sample(
                    case,
                    bank,
                    regions,
                    sample_index=sample_index,
                    split_name=split_name,
                    graph_config=graph_config,
                    liver_label=liver_label,
                    tumor_label=tumor_label,
                    source_selection=source_selection,
                    source_pad=source_pad,
                    total_candidates=total_candidates,
                    candidate_pool_size=candidate_pool_size,
                    easy_fraction=easy_fraction,
                    inter_fraction=inter_fraction,
                    intra_fraction=intra_fraction,
                    max_draws=max_draws,
                    min_liver_coverage=min_liver_coverage,
                    occupied_clearance_vox=occupied_clearance_vox,
                    min_center_separation_mm=min_center_separation_mm,
                    ct_clip=ct_clip,
                    seed=seed,
                )
                if sample is None:
                    local_rows.append(
                        case_progress_row(
                            case_id=paths.case_id,
                            sample_index=sample_index,
                            split_name=split_name,
                            status="insufficient_curriculum_candidates",
                            config_fingerprint=fingerprint,
                            message="deterministic candidate curriculum could not be formed",
                        )
                    )
                    continue
                sample["config_fingerprint"] = fingerprint
                sample["source_image_sha256"] = source_record["image_sha256"]
                sample["source_label_sha256"] = source_record["label_sha256"]
                verify_loaded_case_source_signatures(case)
                _atomic_torch_save(sample, output, overwrite=overwrite)
                difficulties = sample["difficulties"].tolist()
                local_rows.append(
                    case_progress_row(
                        case_id=paths.case_id,
                        sample_index=sample_index,
                        split_name=split_name,
                        status="ok",
                        path=output.name,
                        candidates=len(sample["target_locals"]),
                        easy=difficulties.count(1),
                        inter=difficulties.count(2),
                        intra_corrupted=difficulties.count(3),
                        config_fingerprint=fingerprint,
                        artifact_sha256=_sha256_file(output),
                        file_size=output.stat().st_size,
                    )
                )
            except AdaptiveRoiBudgetError as exc:
                local_rows.append(
                    case_progress_row(
                        case_id=paths.case_id,
                        sample_index=sample_index,
                        split_name=split_name,
                        status="resource_budget_error",
                        config_fingerprint=fingerprint,
                        message=str(exc),
                    )
                )
            except CanonicalGraphUnavailable as exc:
                local_rows.append(
                    case_progress_row(
                        case_id=paths.case_id,
                        sample_index=sample_index,
                        split_name=split_name,
                        status="unrepresentable_local_geometry",
                        config_fingerprint=fingerprint,
                        message=str(exc),
                    )
                )
            except Exception as exc:
                local_rows.append(
                    case_progress_row(
                        case_id=paths.case_id,
                        sample_index=sample_index,
                        split_name=split_name,
                        status="error",
                        config_fingerprint=fingerprint,
                        message=str(exc),
                    )
                )
        sample_rows = {
            int(row["sample_index"]): row
            for row in local_rows
            if str(row.get("sample_index", "")) != ""
        }
        missing_rows = sorted(set(pending_indices) - set(sample_rows))
        failed_rows = [
            row for row in sample_rows.values() if str(row.get("status", "")) != "ok"
        ]
        if missing_rows or failed_rows:
            failures = ", ".join(
                f"{row['sample_index']}={row.get('status', 'unknown')}"
                for row in sorted(
                    failed_rows, key=lambda item: int(item["sample_index"])
                )
            )
            message_parts = []
            if missing_rows:
                message_parts.append(f"missing progress rows for samples {missing_rows}")
            if failures:
                message_parts.append(f"failed samples: {failures}")
            case_status = "sample_failure"
            case_message = "; ".join(message_parts)
        else:
            case_status = "ok"
            case_message = (
                f"all {len(pending_indices)} pending cache samples completed"
            )
        local_rows.append(
            case_progress_row(
                case_id=paths.case_id,
                sample_index=None,
                split_name=split_name,
                status=case_status,
                config_fingerprint=fingerprint,
                message=case_message,
            )
        )
        return local_rows

    emitted_rows: list[dict[str, object]] = []

    def commit_rows(local_rows: list[dict[str, str]]) -> None:
        for row in local_rows:
            records[_progress_row_key(row)] = row
            emitted_rows.append(dict(row))
        # The main thread is the sole CSV writer. Each completed case is committed
        # atomically. If a worker saves a .pt just before a crash but its SHA row
        # is not committed, the next run fails closed and requires explicit
        # overwrite authorization instead of guessing its provenance.
        _atomic_progress_manifest_save(records, manifest_path)
        for row in local_rows:
            status = row.get("status")
            case_id = row.get("case_id")
            sample_index = row.get("sample_index") or None
            if status == "ok":
                print(f"[OK] cache {case_id} sample={sample_index}")
            elif status == "no_tumor":
                print(f"[Invalid] {case_id}: configured case has no tumor label")
            elif status == "insufficient_curriculum_candidates":
                print(f"[Invalid] {case_id} sample={sample_index}: insufficient candidates")
            elif status == "unrepresentable_local_geometry":
                print(
                    f"[Invalid] {case_id} sample={sample_index}: "
                    f"unrepresentable local geometry ({row.get('message', '')})"
                )
            elif status == "resource_budget_error":
                print(
                    f"[ResourceError] {case_id} sample={sample_index}: "
                    f"{row.get('message', '')}"
                )
            elif status == "load_error":
                print(f"[Error] load {case_id}: {row.get('message', '')}")
            else:
                print(f"[Error] cache {case_id} sample={sample_index}: {row.get('message', status)}")

    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError(f"Cache preparation workers must be positive, got {workers}")
    if worker_count > 1:
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="hiercp-cache"
        ) as executor:
            future_to_case = {
                executor.submit(prepare_case, paths): paths.case_id
                for paths in pending_case_paths
            }
            for future in as_completed(future_to_case):
                case_id = future_to_case[future]
                try:
                    local_rows = future.result()
                except Exception as exc:
                    row = _progress_row(
                        case_id=case_id,
                        sample_index=None,
                        split_name=split_lookup[case_id],
                        status="load_error",
                        config_fingerprint=fingerprint,
                        source_image_sha256=source_by_id[case_id]["image_sha256"],
                        source_label_sha256=source_by_id[case_id]["label_sha256"],
                        message=f"worker failure: {exc}",
                    )
                    commit_rows([row])
                else:
                    # Keep manifest persistence outside the worker-result catch;
                    # a failed atomic commit must propagate, not be relabelled.
                    commit_rows(local_rows)
    else:
        for paths in pending_case_paths:
            commit_rows(prepare_case(paths))

    try:
        verified_entries = complete_entries()
    except Exception as exc:
        invalidate_publication(state="failed", message=str(exc))
        raise
    if verified_entries is None:
        failure_details: list[str] = []
        status_counts: dict[str, int] = {}
        resource_failure = False
        for case_id in all_case_ids:
            case_row = records.get(_progress_key(case_id, None))
            if case_row is not None and _row_matches_current_config(case_row, fingerprint):
                status = str(case_row.get("status", "case_error"))
                if status != "ok":
                    message = " ".join(str(case_row.get("message", "")).split())
                    status_counts[status] = status_counts.get(status, 0) + 1
                    resource_failure = resource_failure or status == "resource_budget_error"
                    failure_details.append(f"{case_id}[*]={status}: {message}")
            for sample_index in range(int(samples_per_case)):
                output = root / f"{case_id}__{sample_index:03d}.pt"
                row = records.get(_progress_key(case_id, sample_index))
                if _sample_row_is_complete(
                    row,
                    root=root,
                    expected_output=output,
                    fingerprint=fingerprint,
                ):
                    continue
                if row is None or not _row_matches_current_config(row, fingerprint):
                    status = "missing"
                    message = "no current progress row"
                else:
                    status = str(row.get("status", "incomplete"))
                    message = " ".join(str(row.get("message", "")).split())
                status_counts[status] = status_counts.get(status, 0) + 1
                resource_failure = resource_failure or status == "resource_budget_error"
                failure_details.append(
                    f"{case_id}[{sample_index}]={status}: {message}"
                )
        valid_entries = _progress_entries(
            root=root,
            records=records,
            selected_case_ids=all_case_ids,
            split_lookup=split_lookup,
            samples_per_case=samples_per_case,
            fingerprint=fingerprint,
        )
        preview = "; ".join(failure_details[:20])
        remainder = max(0, len(failure_details) - 20)
        failure_message = (
            "Hierarchical cache is incomplete; no cache index/ready marker was "
            f"published. expected_valid_entries={expected_entry_count}, "
            f"actual_valid_entries={len(valid_entries)}, status_counts={status_counts}, "
            f"first_failures=[{preview}]"
            + (f"; additional_failures={remainder}" if remainder else "")
            + f". Full diagnostics remain in {manifest_path}."
        )
        invalidate_publication(state="failed", message=failure_message)
        if resource_failure:
            raise AdaptiveRoiBudgetError(failure_message)
        raise RuntimeError(failure_message)

    publish_complete(verified_entries)
    print(
        f"[OK] Hierarchical cache complete: {complete_path} "
        f"entries={len(verified_entries)}/{expected_entry_count} run_mode={mode}"
    )
    return emitted_rows

def build_inference_sample(
    case: LoadedCase,
    source: object,
    candidates: Sequence[object],
    bank: PrototypeBank,
    *,
    graph_config: GraphBuildConfig,
    liver_label: int,
    tumor_label: int,
    ct_clip: tuple[float, float],
    seed: int,
    regions: PatientRegionData | None = None,
) -> tuple[dict, list]:
    """Build an uncached hierarchy for candidate scoring during generation."""

    from hiercp.common import CandidateInfo, SourceTumor

    if not isinstance(source, SourceTumor):
        raise TypeError("source must be SourceTumor")
    if not all(isinstance(candidate, CandidateInfo) for candidate in candidates):
        raise TypeError("candidates must contain CandidateInfo objects")
    typed_candidates = list(candidates)
    if regions is None:
        regions = load_or_build_patient_regions(
            case,
            liver_label=liver_label,
            tumor_label=tumor_label,
            config=graph_config,
            ct_clip=ct_clip,
            seed=stable_case_seed(
                seed, case.paths.case_id, REGION_CACHE_SEED_SALT
            ),
            cache_dir=None,
            overwrite=False,
            mmap=False,
        )
    specs = build_generation_specs(
        typed_candidates,
        regions,
        bank,
        config=graph_config,
    )
    source_rng = np.random.default_rng(
        stable_case_seed(seed, case.paths.case_id, "infer_local")
    )
    prepared_source = prepare_local_source(
        case,
        source,
        full_organ_mask=regions.full_organ_mask,
        organ_depth=regions.organ_depth,
        config=graph_config,
        rng=source_rng,
        ct_clip=ct_clip,
    )
    built_local = [
        build_local_graph(
            case,
            source,
            spec,
            full_organ_mask=regions.full_organ_mask,
            organ_depth=regions.organ_depth,
            config=graph_config,
            rng=source_rng,
            ct_clip=ct_clip,
            prepared_source=prepared_source,
        )
        for spec in specs
    ]
    patient_graph = build_patient_graph(
        case,
        source,
        specs,
        regions,
        tumor_label=tumor_label,
        config=graph_config,
        ct_clip=ct_clip,
    )
    prototype_graph = build_prototype_graph(
        specs,
        patient_graph,
        regions,
        bank,
        config=graph_config,
    )
    sample = {
        "format": CACHE_FORMAT,
        "prototype_fingerprint": bank.fingerprint(),
        "case_id": case.paths.case_id,
        "sample_index": -1,
        "split": "inference",
        "source_component": int(source.component_id),
        "source_patch": torch.from_numpy(prepared_source.source_patch.astype(np.float16)),
        "target_patches": torch.from_numpy(
            np.stack([item.target_patch for item in built_local]).astype(np.float16)
        ),
        "source_local": built_local[0].source_local,
        "target_locals": [item.target_local for item in built_local],
        "patient_graph": patient_graph,
        "prototype_graph": prototype_graph,
        "difficulties": torch.ones(len(specs), dtype=torch.long),
        "corruptions": torch.zeros(len(specs), dtype=torch.long),
        "candidate_centers": torch.tensor([spec.center for spec in specs], dtype=torch.long),
        "candidate_regions": torch.tensor([spec.region_id for spec in specs], dtype=torch.long),
        "candidate_prototypes": torch.tensor(
            [spec.prototype_id for spec in specs], dtype=torch.long
        ),
        "graph_config": graph_config.to_dict(),
        "ct_clip": tuple(float(value) for value in ct_clip),
    }
    return sample, specs
