"""Explicit, non-destructive migration of SHA-verified failed donor caches.

Only the donor eligibility contract and an increased ROI allocation ceiling may
change. This certifies byte/schema identities, not the unrecorded historical
producer code. Source files are never written. An interrupted copy remains
incomplete and requires a fresh destination, not automatic overwrite.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from hiercp import cache as c
from hiercp.region import graph_config_budget_compatible, load_patient_regions

MIGRATION_FORMAT = "hiercp_failed_cache_migration_v1"


def _snapshot(roots, source_paths):
    paths = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Migration source directory is missing or symlinked: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Migration refuses source symlinks: {path}")
            if path.is_file():
                paths.add(path.resolve())
    for paths_row in source_paths:
        for name in ("image_path", "label_path"):
            path = Path(getattr(paths_row, name))
            if path.is_symlink():
                raise ValueError(f"Migration refuses source symlinks: {path}")
            paths.add(path.resolve())
    return [{"path": str(path), "sha256": c._sha256_file(path), "file_size": path.stat().st_size}
            for path in sorted(paths, key=str)]


def _verify_inventory(inventory):
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Migration has no source inventory")
    names = set()
    for row in inventory:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "file_size"}:
            raise ValueError("Invalid migration inventory row")
        path = Path(row["path"])
        if str(path) in names or not path.is_absolute() or path.is_symlink():
            raise ValueError("Migration source inventory has duplicate/unsafe paths")
        names.add(str(path))
        c._assert_file_sha256(path, row["sha256"], context="Migration original input")
        if type(row["file_size"]) is not int or path.stat().st_size != row["file_size"]:
            raise ValueError(f"Migration original size mismatch: {path}")


def _semantic_sha(payload):
    """Hash all tensors and semantic metadata before/after metadata-only copy."""
    digest = hashlib.sha256()

    def feed(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                raise ValueError("Migration payload contains nonfinite tensors")
            digest.update(f"torch:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject or (np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array))):
                raise ValueError("Migration payload contains object/nonfinite arrays")
            digest.update(f"numpy:{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict[")
            for key in sorted(value, key=repr):
                feed(key)
                feed(value[key])
            digest.update(b"]")
        elif isinstance(value, (tuple, list)):
            digest.update((type(value).__name__ + "[").encode())
            for item in value:
                feed(item)
            digest.update(b"]")
        elif hasattr(value, "to_dict"):
            digest.update(type(value).__qualname__.encode())
            feed(value.to_dict())
        elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
            if isinstance(value, np.generic):
                value = value.item()
            digest.update(json.dumps(value, allow_nan=False, sort_keys=True).encode())
            digest.update(b";")
        else:
            raise ValueError(f"Unsupported migration payload object: {type(value).__qualname__}")

    semantic = dict(payload)
    semantic.pop("config_fingerprint", None)
    semantic.pop("migration_source_sha256", None)
    semantic["graph_config"] = {key: value for key, value in payload["graph_config"].items()
                                if key != "adaptive_roi_max_voxels"}
    feed(semantic)
    return digest.hexdigest()


def _context(*, source_cache_dir, destination_cache_dir, prepare_kwargs):
    source, destination = Path(source_cache_dir).resolve(), Path(destination_cache_dir).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Migration source and destination must be disjoint directories")
    bound = inspect.signature(c.prepare_hierarchical_cache).bind(**prepare_kwargs)
    bound.apply_defaults()
    request = dict(bound.arguments)
    if Path(request["cache_dir"]).resolve() != destination or request["overwrite"] is not False:
        raise ValueError("Migration requires the exact destination and overwrite=False")
    if request["max_cases"] is not None:
        raise ValueError("Recovery migration preserves the full configured cohort, not max_cases")
    request["graph_config"].validate()
    old_config = c._load_json_object(source / "config.json")
    if old_config.get("state") != "failed" or any((source / name).exists() for name in ("index.json", "complete.json")):
        raise ValueError("Migration requires a failed cache without a publication marker")
    if "donor_eligibility" in old_config or "donor_contract_sha256" in old_config:
        raise ValueError("This migration is for legacy failed caches without donor eligibility")
    selected = [*request["train_case_ids"], *request["val_case_ids"]]
    c._validated_case_ids(selected, context="Migration full train+val cohort", allow_empty=False)
    if request["run_mode"] not in c.CACHE_RUN_MODES:
        raise ValueError("Migration run mode is invalid")
    paths = c.discover_cases(request["data_dir"], case_ids=selected, run_mode=request["run_mode"])
    sources = c._source_contract(paths, selected)
    bank_path = Path(request["bank_path"])
    original_bank = Path(old_config.get("prototype_bank", ""))
    original_regions = Path(old_config.get("region_cache_dir", ""))
    for protected in (source, original_bank.parent.resolve(), original_regions.resolve()):
        if destination == protected or destination in protected.parents or protected in destination.parents:
            raise ValueError(f"Migration destination overlaps a protected original root: {protected}")
    for path in (bank_path, original_bank):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Migration requires original and copied prototype artifacts: {path}")
        c._assert_file_sha256(path, old_config.get("prototype_artifact_sha256"), context="Migration prototype")
        # The bank already exists: this path performs only strict read validation.
        c.prepare_prototype_bank(data_dir=request["data_dir"], output_path=path,
            region_cache_dir=request["region_cache_dir"], training_case_ids=request["train_case_ids"],
            graph_config=request["graph_config"], liver_label=request["liver_label"],
            tumor_label=request["tumor_label"], ct_clip=request["ct_clip"], seed=request["seed"],
            overwrite=False, workers=request["workers"])
    bank = c.PrototypeBank.load(bank_path)
    donor = request["donor_eligibility"]
    if donor is None:
        donor = c.build_donor_eligibility(case_paths=paths, selected_case_ids=selected, source_cases=sources,
            liver_label=request["liver_label"], tumor_label=request["tumor_label"], workers=request["workers"])
    donor_ids = c.validate_donor_eligibility(donor, selected_case_ids=selected, source_cases=sources,
        labels={"liver": int(request["liver_label"]), "tumor": int(request["tumor_label"])})
    if not donor_ids:
        raise ValueError("Migration has no eligible self-donors")
    expected = c._cache_expected_metadata(request, bank=bank, sources=sources,
                                          selected_case_ids=selected, donor_eligibility=donor)
    legacy_keys = set(expected) - {"donor_eligibility", "donor_contract_sha256"}
    if any(key not in old_config for key in legacy_keys):
        raise ValueError("Legacy cache is missing required contract fields")
    legacy_contract = {key: old_config[key] for key in legacy_keys}
    if c._cache_config_fingerprint(legacy_contract) != old_config.get("config_fingerprint"):
        raise ValueError("Legacy cache config fingerprint does not match its exact contract")
    differences = [key for key in legacy_keys if key != "graph_config" and expected[key] != old_config[key]]
    if differences or not graph_config_budget_compatible(old_config["graph_config"], expected["graph_config"]):
        raise ValueError(f"Migration permits only donor eligibility and increased ROI budget; mismatches={differences}")
    if old_config.get("progress_format") != c.CACHE_PROGRESS_FORMAT:
        raise ValueError("Legacy progress contract is not SHA-bound")
    if not (source / "manifest.csv").is_file():
        raise ValueError("Legacy cache manifest is missing")
    records = c._load_progress_manifest(source / "manifest.csv")
    split = {case_id: "train" for case_id in request["train_case_ids"]}
    split.update({case_id: "val" for case_id in request["val_case_ids"]})
    source_by_id = {row["case_id"]: row for row in sources}
    c._validate_progress_contract(records, selected_case_ids=selected, split_lookup=split,
        samples_per_case=request["samples_per_case"], fingerprint=old_config["config_fingerprint"], source_by_id=source_by_id)
    files = sorted(source.glob("*.pt"))
    c._adopt_existing_cache_files(root=source, records=records, selected_case_ids=donor_ids,
        split_lookup=split, samples_per_case=request["samples_per_case"], fingerprint=old_config["config_fingerprint"])
    for (case_id, index), row in records.items():
        if index is not None and row.get("status") == "ok":
            if case_id not in donor_ids or not c._sample_row_is_complete(row, root=source,
                expected_output=source / f"{case_id}__{index:03d}.pt", fingerprint=old_config["config_fingerprint"]):
                raise ValueError(f"Legacy success row has no exact eligible artifact: {case_id}/{index}")
        if case_id not in donor_ids and index is not None:
            raise ValueError("Legacy ineligible donor has unexpected per-sample rows")
        if case_id in donor_ids and row.get("status") == "no_tumor":
            raise ValueError("Legacy no_tumor row disagrees with current raw label evidence")
    c._validate_recoverable_partial_cache(files, selected_case_ids=donor_ids, split_lookup=split,
        samples_per_case=request["samples_per_case"], total_candidates=request["total_candidates"],
        prototype_fingerprint=bank.fingerprint(), config_fingerprint=old_config["config_fingerprint"],
        source_by_id=source_by_id, graph_config=old_config["graph_config"], ct_clip=request["ct_clip"])
    copied_regions = Path(request["region_cache_dir"])
    if not original_regions.is_dir() or not copied_regions.is_dir():
        raise ValueError("Migration requires original and copied region caches")
    required_regions = set(request["train_case_ids"]) | {str(row["case_id"]) for row in records.values()
                                                       if row.get("sample_index") and row.get("status") == "ok"}
    for case_id in required_regions:
        region, metadata = load_patient_regions(original_regions / case_id, mmap=True)
        del region
        source_row = source_by_id[case_id]
        if metadata.get("case_id") != case_id or metadata.get("image", {}).get("sha256") != source_row["image_sha256"] or metadata.get("label", {}).get("sha256") != source_row["label_sha256"]:
            raise ValueError(f"Migration region source mismatch: {case_id}")
        if not graph_config_budget_compatible(metadata.get("graph_config"), expected["graph_config"]) or metadata.get("labels") != expected["labels"] or metadata.get("ct_clip") != expected["ct_clip"]:
            raise ValueError(f"Migration region configuration mismatch: {case_id}")
        if metadata.get("seed") != c.stable_case_seed(request["seed"], case_id, c.REGION_CACHE_SEED_SALT):
            raise ValueError(f"Migration region seed mismatch: {case_id}")
        for original in (original_regions / case_id).iterdir():
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"Unexpected region artifact: {original}")
            c._assert_file_sha256(copied_regions / case_id / original.name,
                                 c._sha256_file(original), context="Copied region bytes")
    inventory = _snapshot([source, original_bank.parent, original_regions], paths)
    return dict(source=source, destination=destination, request=request, old=old_config,
                expected=expected, fingerprint=c._cache_config_fingerprint(expected), records=records,
                files=files, sources=sources, paths=paths, split=split, donors=donor_ids, inventory=inventory)


def migrate_failed_hierarchical_cache(*, source_cache_dir, destination_cache_dir, prepare_kwargs):
    destination = Path(destination_cache_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Migration requires an entirely new cache destination; existing results were not touched")
    context = _context(source_cache_dir=source_cache_dir, destination_cache_dir=destination,
                       prepare_kwargs=prepare_kwargs)
    destination = context["destination"]
    destination.mkdir(parents=True, exist_ok=False)
    certificate_path = destination / "migration.json"
    base = {"format": MIGRATION_FORMAT, "source_cache_dir": str(context["source"]),
            "destination_cache_dir": str(destination), "source_config_fingerprint": context["old"]["config_fingerprint"],
            "target_config_fingerprint": context["fingerprint"], "source_inventory": context["inventory"],
            "donor_contract_sha256": context["expected"]["donor_contract_sha256"],
            "allowed_changes": ["explicit_donor_eligibility", "increased_adaptive_roi_max_voxels"],
            "historical_producer_code_identity": "unverified_not_recorded_by_legacy_cache",
            "actual_training_executed": False, "source_mutation_allowed": False}
    c._atomic_json_save({**base, "state": "copying"}, certificate_path)
    config = {**context["expected"], "config_fingerprint": context["fingerprint"], "state": "building",
              "data_dir": str(Path(context["request"]["data_dir"]).resolve()),
              "prototype_bank": str(Path(context["request"]["bank_path"]).resolve()),
              "region_cache_dir": str(Path(context["request"]["region_cache_dir"]).resolve()),
              "progress_format": c.CACHE_PROGRESS_FORMAT}
    c._atomic_json_save(config, destination / "config.json")
    records = {key: {**row, "config_fingerprint": context["fingerprint"]} for key, row in context["records"].items()}
    for source_row in context["sources"]:
        case_id = source_row["case_id"]
        if case_id not in context["donors"]:
            records[(case_id, None)] = c._progress_row(case_id=case_id, sample_index=None,
                split_name=context["split"][case_id], status="donor_ineligible",
                config_fingerprint=context["fingerprint"], source_image_sha256=source_row["image_sha256"],
                source_label_sha256=source_row["label_sha256"],
                message="configured_tumor_label_absent; retained in full patient/source split")
    copied = []

    def copy_artifact(path):
        original_sha = c._sha256_file(path)
        payload = c._torch_load_cpu(path)
        semantic_sha = _semantic_sha(payload)
        payload["config_fingerprint"] = context["fingerprint"]
        payload["graph_config"] = context["expected"]["graph_config"]
        payload["migration_source_sha256"] = original_sha
        output = destination / path.name
        c._atomic_torch_save(payload, output, overwrite=False)
        case_id, sample_index = payload["case_id"], int(payload["sample_index"])
        del payload
        if _semantic_sha(c._torch_load_cpu(output)) != semantic_sha:
            raise ValueError(f"Metadata-only migration changed tensor/semantic contents: {path.name}")
        return {"path": path.name, "source_sha256": original_sha,
                "artifact_sha256": c._sha256_file(output), "file_size": output.stat().st_size,
                "semantic_sha256": semantic_sha, "case_id": case_id,
                "sample_index": sample_index}

    def commit(row):
        copied.append(row)
        records[(row["case_id"], row["sample_index"])].update(
            artifact_sha256=row["artifact_sha256"], file_size=str(row["file_size"]))
        c._atomic_progress_manifest_save(records, destination / "manifest.csv")

    from hiercp.preparation_runtime import run_case_jobs
    run_case_jobs(tasks=context["files"], function=copy_artifact, commit=commit,
                  workers=context["request"]["workers"], report_path=c._runtime_report_path(destination, "migration"))
    c._atomic_progress_manifest_save(records, destination / "manifest.csv")
    _verify_inventory(context["inventory"])
    if sorted(row["path"] for row in copied) != sorted(path.name for path in context["files"]):
        raise ValueError("Migration copy cohort is incomplete")
    body = {**base, "state": "ready_for_prepare", "copied_artifacts": sorted(copied, key=lambda row: row["path"]),
            "copied_count": len(copied), "expected_donor_entries": len(context["donors"]) * context["request"]["samples_per_case"]}
    certificate = {**body, "contract_sha256": c._cache_config_fingerprint(body)}
    c._atomic_json_save(certificate, certificate_path)
    try:
        validate_cache_migration(source_cache_dir=source_cache_dir, destination_cache_dir=destination, prepare_kwargs=prepare_kwargs)
    except Exception as exc:
        # This certificate is owned by this new migration, never an old result.
        failed = {**body, "state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        c._atomic_json_save({**failed, "contract_sha256": c._cache_config_fingerprint(failed)}, certificate_path)
        raise
    return certificate


def validate_cache_migration(*, source_cache_dir, destination_cache_dir, prepare_kwargs):
    destination = Path(destination_cache_dir).resolve()
    certificate = c._load_json_object(destination / "migration.json")
    body = {key: value for key, value in certificate.items() if key != "contract_sha256"}
    if certificate.get("format") != MIGRATION_FORMAT or certificate.get("state") != "ready_for_prepare" or certificate.get("contract_sha256") != c._cache_config_fingerprint(body):
        raise ValueError("Migration is incomplete or its certificate changed; use a fresh output root")
    if certificate.get("source_cache_dir") != str(Path(source_cache_dir).resolve()) or certificate.get("destination_cache_dir") != str(destination):
        raise ValueError("Migration certificate source/destination mismatch")
    _verify_inventory(certificate.get("source_inventory"))
    context = _context(source_cache_dir=source_cache_dir, destination_cache_dir=destination, prepare_kwargs=prepare_kwargs)
    if certificate.get("target_config_fingerprint") != context["fingerprint"] or certificate.get("source_inventory") != context["inventory"]:
        raise ValueError("Migration no longer matches the exact original input or target request")
    config = c._load_json_object(destination / "config.json")
    c._assert_metadata_equal(config, context["expected"], context="Migrated target cache")
    if config.get("config_fingerprint") != context["fingerprint"]:
        raise ValueError("Migrated target fingerprint mismatch")
    copies = certificate.get("copied_artifacts")
    if not isinstance(copies, list) or certificate.get("copied_count") != len(copies) or [row.get("path") for row in copies] != sorted(path.name for path in context["files"]):
        raise ValueError("Migration copied artifact cohort is not exact")
    records = c._load_progress_manifest(destination / "manifest.csv")
    for row in copies:
        name = row["path"]
        path = destination / name
        if Path(name).name != name or path.is_symlink():
            raise ValueError("Migration copied artifact path is unsafe")
        c._assert_file_sha256(context["source"] / name, row["source_sha256"], context="Migrated original artifact")
        c._assert_file_sha256(path, row["artifact_sha256"], context="Migrated retained artifact")
        if path.stat().st_size != row["file_size"] or _semantic_sha(c._torch_load_cpu(path)) != row["semantic_sha256"]:
            raise ValueError(f"Migrated artifact content mismatch: {name}")
        record = records.get((row["case_id"], row["sample_index"]))
        if not c._sample_row_is_complete(record, root=destination, expected_output=path, fingerprint=context["fingerprint"]):
            raise ValueError(f"Migrated artifact lost its exact progress record: {name}")
    return certificate
