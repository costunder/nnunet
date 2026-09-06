"""Preserve a failed preparation and recover into a separate, verified workspace.

The resource pilot uses full-size real samples. It is calibration, not training
or a performance evaluation. Original graphs, trainers and results are never
modified. No '--overwrite' command is issued.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant {value}: {path}")

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    with tempfile.NamedTemporaryFile(prefix=path.name + ".writing.", dir=path.parent, delete=False) as handle:
        staging = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(staging, path)
    finally:
        staging.unlink()  # our exact private temporary file, never an old result


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def preparation_code_identity(project):
    project = Path(project).resolve()
    paths = sorted((project / "hiercp").rglob("*.py")) + [project / "tools/feedback_preparation_recovery.py"]
    return {str(path.relative_to(project)): digest(path) for path in paths}


def validate_pilot(result, request):
    required = {"format": "hiercp_full_size_resource_pilot_v1", "calibration_only": True,
                "training_performed": False, "case_id": request["case_id"],
                "sample_index": request["sample_index"], "request_sha256": value_sha(request),
                "roi_budget": request["config"]["graph"]["adaptive_roi_max_voxels"],
                "candidate_count": request["config"]["cache"]["total_candidates"]}
    if any(result.get(key) != expected for key, expected in required.items()):
        raise ValueError("Resource pilot does not bind the entire current request/sample/code")
    measurement = result.get("measurement", {})
    if measurement.get("status") != "complete" or measurement.get("sampling_error") is not None:
        raise ValueError("Resource pilot did not successfully measure the full sample")
    for field in ("sampled_peak_rss_bytes", "elapsed_seconds"):
        value = measurement.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Resource pilot lacks a finite positive measurement: {field}")


def _flag(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 == len(argv):
        raise ValueError(f"Original launch command does not bind {flag} exactly once")
    return argv[argv.index(flag) + 1]


def validate_recovery_source(plan, source_root):
    """Read-only, stdlib-only validation; safe to call from --dry-run."""
    project = Path(plan["project_root"]).resolve()
    source = Path(source_root).resolve()
    target = Path(plan["run_root"]).resolve()
    work = project / "work"
    if source == work or not source.is_relative_to(work) or source == target:
        raise ValueError("Recovery source must be a distinct experiment beneath this checkout/work")
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Recovery source and destination must not overlap")
    old = read_json(source / "launch_plan.json")
    for key, expected in (("project_root", project), ("run_root", source),
                          ("medical_root", Path(plan["medical_root"]).resolve())):
        if Path(old[key]).resolve() != expected:
            raise ValueError(f"Original launch identity mismatch: {key}")
    commands = old.get("commands", [])
    names = [item.get("name") for item in commands]
    if len(names) != len(set(names)):
        raise ValueError("Original launch contains duplicate stage names")
    by_name = {item["name"]: item["argv"] for item in commands}
    bindings = (("gnn-prepare", "--outer-fold", plan["outer_fold"]),
                ("plan", "--dataset-id", plan["dataset_id"]),
                ("train_full", "--seed", plan["seed"]))
    for stage, flag, expected in bindings:
        if int(_flag(by_name[stage], flag)) != int(expected):
            raise ValueError(f"Original experiment {flag} differs from recovery request")
    fold = int(plan["outer_fold"])
    gnn = source / f"paired/folds/fold_{fold}/gnn"
    config = read_json(gnn / "graphs/config.json")
    split = read_json(gnn / "split.json")
    if config.get("state") != "failed":
        raise ValueError("Only an explicitly failed graph preparation is recoverable here")
    if (gnn / "graphs/complete.json").exists() or (gnn / "graphs/index.json").exists():
        raise ValueError("Failed source unexpectedly has a published graph cache")
    for name in ("train", "val"):
        if config.get(f"{name}_case_ids") != split.get(name):
            raise ValueError(f"Original graph/source split mismatch: {name}")
    if config.get("selected_case_ids") != split["train"] + split["val"]:
        raise ValueError("Original graph preparation did not select the entire configured cohort")
    if config.get("subset_active") or config.get("run_mode") != "benchmark":
        raise ValueError("Recovery does not adopt debug/subset or unrelated cache contracts")
    excluded = set(split.get("outer_validation_excluded", []))
    if excluded & set(config["selected_case_ids"]):
        raise ValueError("Outer validation leaked into the graph preparation cohort")
    for checkpoint in (gnn / "model.pt", gnn / "model.last.pt"):
        if checkpoint.exists():
            raise ValueError("This is preparation recovery, not a trained-model migration")
    required = [source / "launch_plan.json", source / "paired/outer_splits.json",
                source / "paired/case_profiles.csv", gnn / "split.json", gnn / "prototype.pt",
                gnn / "metadata.json", gnn / "manifest.csv", gnn / "graphs/config.json",
                gnn / "graphs/manifest.csv"]
    hashes = {str(path.relative_to(source)): digest(path) for path in required}
    return {"format": "hiercp_preparation_source_v1", "source_root": str(source),
            "outer_fold": fold, "dataset_id": int(plan["dataset_id"]),
            "seed": int(plan["seed"]), "files": hashes}


def verify_identity(identity):
    source = Path(identity["source_root"])
    for name, expected in identity["files"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Recovery source changed: {source / name}")


def geometry_envelope(eligibility, config):
    """Analytic enclosure, NOT a measured memory estimate or a graph crop.

    Every lesion component is included. The current corruption contract has
    maximum scale 1.60 and physical rotations. A sphere enclosing the padded
    box encloses every rotated/scaled footprint. The unchanged 64-mm search
    bound encloses every permitted native ROI and liver-surface extension.
    """
    graph = config["graph"]
    radius = float(graph["adaptive_roi_max_radius_mm"])
    if radius < max(float(graph["adaptive_roi_margin_mm"]),
                    float(graph["context_outer_radius_mm"]) + 2.0):
        raise ValueError("Configured physical-radius contract is internally inconsistent")
    pad = int(config["cache"]["source_pad"])
    # Match load_case's float32 spacing and the corruption transform's
    # float32 anisotropic scale before taking the conservative enclosure.
    maximum_scale = math.nextafter(struct.unpack("f", struct.pack("f", 1.60))[0], math.inf)
    rows = []
    for case in eligibility["cases"]:
        spacing = [struct.unpack("f", struct.pack("f", float(v)))[0] for v in case["spacing_mm"]]
        if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
            raise ValueError("Invalid source spacing in donor evidence")
        maximum, shape = 0, None
        for bbox in case["component_bbox_shapes"]:
            # +1 accounts for the integer-anchor asymmetry and voxel cells.
            half_mm = [((int(n) + 2 * pad + 1) / 2) * s for n, s in zip(bbox, spacing)]
            radius_mm = math.nextafter(maximum_scale * math.sqrt(sum(v * v for v in half_mm)), math.inf)
            candidate = [2 * math.ceil(radius_mm / s) + 1 + 2 * math.ceil(radius / s) for s in spacing]
            volume = math.prod(candidate)
            if volume > maximum:
                maximum, shape = volume, candidate
        rows.append({"case_id": case["case_id"], "roi_envelope_voxels": maximum,
                     "roi_envelope_shape": shape, "components": len(case["component_bbox_shapes"])})
    maximum = max((row["roi_envelope_voxels"] for row in rows), default=0)
    if maximum < 1:
        raise ValueError("No eligible real tumor source exists in this cohort")
    return {"format": "hiercp_roi_envelope_v1", "maximum_voxels": maximum,
            "maximum_scale": maximum_scale, "physical_search_radius_mm": radius,
            "basis": "all real lesion bounding boxes, full padding, physical rotation enclosure; no sampling",
            "memory_measured": False, "cases": rows}


def prepare_kwargs(config, split, root, medical):
    from hiercp.schema import graph_config_from_dict
    cache = config["cache"]
    return {"data_dir": Path(medical) / "Data", "cache_dir": Path(root) / "graphs",
            "region_cache_dir": Path(root) / "regions", "bank_path": Path(root) / "prototype.pt",
            "train_case_ids": split["train"], "val_case_ids": split["val"],
            "graph_config": graph_config_from_dict(config["graph"]),
            "liver_label": int(config["labels"]["liver"]), "tumor_label": int(config["labels"]["tumor"]),
            **{key: cache[key] for key in ("source_selection", "source_pad", "samples_per_case",
                "total_candidates", "candidate_pool_size", "easy_fraction", "inter_fraction",
                "intra_fraction", "max_draws", "min_liver_coverage", "occupied_clearance_vox",
                "min_center_separation_mm")},
            "ct_clip": tuple(config["ct_clip"]), "seed": int(config["seed"]),
            "max_cases": None, "overwrite": False, "workers": config["runtime"]["prepare_workers"],
            "run_mode": "benchmark"}


def _copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Recovery expects a regular, non-symlink file: {source}")
    fingerprint = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or digest(destination) != fingerprint:
            raise ValueError(f"Existing recovery copy differs; preserved: {destination}")
        return
    # Publish only a complete copy. An interrupted unique staging file is kept
    # as evidence and never mistaken for an already verified destination.
    with tempfile.NamedTemporaryFile(prefix=destination.name + ".copying.", dir=destination.parent, delete=False) as outgoing:
        staging = Path(outgoing.name)
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if digest(staging) != fingerprint or digest(source) != fingerprint:
        raise ValueError(f"Source changed while copying; staging evidence preserved: {staging}")
    os.link(staging, destination)  # exclusive publication; never replace a target
    staging.unlink()  # only our exact, successfully published temporary copy


def _profile(request, output):
    """One actual full-size sample per isolated process for clean RSS evidence."""
    from hiercp.cache import build_training_sample
    from hiercp.common import discover_cases, load_case, stable_case_seed
    from hiercp.prototype import PrototypeBank
    from hiercp.region import load_or_build_patient_regions, REGION_CACHE_SEED_SALT
    from hiercp.preparation_runtime import Measurement
    config = request["config"]
    kwargs = prepare_kwargs(config, request["split"], request["gnn_root"], request["medical_root"])
    case_id, index = request["case_id"], int(request["sample_index"])
    measurement = Measurement()
    report = {"format": "hiercp_full_size_resource_pilot_v1", "calibration_only": True,
              "training_performed": False, "case_id": case_id, "sample_index": index,
              "request_sha256": value_sha(request),
              "roi_budget": config["graph"]["adaptive_roi_max_voxels"]}
    try:
        if preparation_code_identity(Path(__file__).resolve().parents[1]) != request["code_sha256"]:
            raise ValueError("Preparation code changed after the pilot request was bound")
        with measurement:
            case = load_case(discover_cases(kwargs["data_dir"], case_ids=[case_id])[0])
            if (digest(case.paths.image_path) != request["source"]["image_sha256"]
                    or digest(case.paths.label_path) != request["source"]["label_sha256"]
                    or digest(kwargs["bank_path"]) != request["prototype_sha256"]):
                raise ValueError("Pilot source or prototype SHA changed")
            regions = load_or_build_patient_regions(case, cache_dir=kwargs["region_cache_dir"],
                liver_label=kwargs["liver_label"], tumor_label=kwargs["tumor_label"],
                config=kwargs["graph_config"], seed=stable_case_seed(kwargs["seed"], case_id, REGION_CACHE_SEED_SALT),
                ct_clip=kwargs["ct_clip"], overwrite=False)
            bank = PrototypeBank.load(kwargs["bank_path"])
            fields = ("graph_config", "liver_label", "tumor_label", "source_selection", "source_pad",
                "total_candidates", "candidate_pool_size", "easy_fraction", "inter_fraction", "intra_fraction",
                "max_draws", "min_liver_coverage", "occupied_clearance_vox", "min_center_separation_mm", "ct_clip", "seed")
            sample = build_training_sample(case, bank, regions, sample_index=index,
                split_name="train" if case_id in request["split"]["train"] else "val",
                **{key: kwargs[key] for key in fields})
            if sample is None:
                raise RuntimeError("Full-size pilot returned no sample; no fabricated output is accepted")
            report["candidate_count"] = len(sample["target_locals"])
    finally:
        report["measurement"] = getattr(measurement, "report", {
            "status": "failed", "error": "Resource measurement did not initialize; no peak RSS available"})
        write_new(output, report)
        print("[FullSizeResourcePilot] " + json.dumps(report, allow_nan=False), flush=True)


def prepare_recovery(plan, source_root, *, runner, env):
    from hiercp.cache import (build_donor_eligibility, migrate_failed_hierarchical_cache,
                              validate_donor_eligibility, validate_cache_migration,
                              _source_contract, validate_cache_publication)
    from hiercp.common import discover_cases
    from hiercp.donor_preflight import audit_donor_headers
    from hiercp.preparation_runtime import snapshot
    source, target = Path(source_root).resolve(), Path(plan["run_root"]).resolve()
    identity = validate_recovery_source(plan, source)
    folder = target / "recovery"
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / "source_identity.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("Recovery source no longer matches its initial identity")
    else:
        write_new(identity_path, identity)
    fold = int(plan["outer_fold"])
    relative_gnn = Path(f"paired/folds/fold_{fold}/gnn")
    old_gnn, gnn = source / relative_gnn, target / relative_gnn
    split = read_json(old_gnn / "split.json")
    old_cache = read_json(old_gnn / "graphs/config.json")
    current = read_json(Path(plan["project_root"]) / "config/train.json")
    # Only orchestration/eligibility/resource policy changes are authorized.
    # Compare the original graph/cache contract before doing any expensive work.
    expected = {"graph": old_cache["graph_config"], "labels": old_cache["labels"],
                "ct_clip": old_cache["ct_clip"]}
    for name, value in expected.items():
        if current[name] != value:
            raise ValueError(f"Checked-in {name} differs from original experiment; not a budget-only recovery")
    cache = current["cache"]
    for name in ("source_selection", "source_pad", "samples_per_case", "total_candidates", "candidate_pool_size",
                 "max_draws", "min_liver_coverage", "occupied_clearance_vox", "min_center_separation_mm"):
        if cache[name] != old_cache[name]:
            raise ValueError(f"Original cache setting changed: {name}")
    for current_name, old_name in (("easy_fraction", "easy"), ("inter_fraction", "inter"), ("intra_fraction", "intra_corrupted")):
        if cache[current_name] != old_cache["difficulty_fractions"][old_name]:
            raise ValueError("Original curriculum fractions changed")
    if int(current["seed"]) + fold != int(old_cache["seed"]):
        raise ValueError("Original fold-specific quality-GNN seed changed")
    selected = split["train"] + split["val"]
    paths = discover_cases(Path(plan["medical_root"]) / "Data", case_ids=selected, run_mode="benchmark")
    receipt_path = folder / "complete.json"
    if not receipt_path.exists():
        # Discover every geometry failure before full source hashing, cache
        # copying, raw-label classification or full-size resource pilots.
        audit_donor_headers(case_paths=paths, selected_case_ids=selected, workers="auto",
                            report_path=folder / "header_geometry.json")
    sources = _source_contract(paths, selected)
    if sources != old_cache["source_cases"]:
        raise ValueError("Actual source images/labels differ from original cache contract")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        verify_identity(identity)
        for relative, sha in receipt["files"].items():
            if digest(target / relative) != sha:
                raise ValueError(f"Recovery completion artifact changed: {relative}")
        restored_config = read_json(Path(plan["train_config"]))
        restored_config["seed"] = int(restored_config["seed"]) + fold
        restored_kwargs = prepare_kwargs(restored_config, split, gnn, plan["medical_root"])
        restored_kwargs["donor_eligibility"] = read_json(folder / "donor_eligibility.json")
        validate_cache_migration(source_cache_dir=old_gnn / "graphs",
                                 destination_cache_dir=gnn / "graphs", prepare_kwargs=restored_kwargs)
        validate_cache_publication(gnn / "graphs")
        return receipt
    copy_sources = [path for path in (old_gnn / "regions").rglob("*") if path.is_file()]
    copy_sources.extend((old_gnn / "graphs").glob("*.pt"))
    required_disk = sum(path.stat().st_size for path in copy_sources) + int(plan["minimum_free_bytes"])
    if shutil.disk_usage(target).free < required_disk:
        raise OSError("Insufficient free storage for independent cache copies plus the configured preprocessing reserve")
    for name in ("outer_splits.json", "case_profiles.csv"):
        _copy_verified(source / "paired" / name, target / "paired" / name)
    for name in ("split.json", "prototype.pt", "metadata.json", "manifest.csv"):
        _copy_verified(old_gnn / name, gnn / name)
    for path in sorted((old_gnn / "regions").rglob("*")):
        if path.is_symlink():
            raise ValueError("Region-copy symlinks are not adopted by recovery")
        if path.is_file():
            _copy_verified(path, gnn / "regions" / path.relative_to(old_gnn / "regions"))
    eligibility_path = folder / "donor_eligibility.json"
    if eligibility_path.exists():
        eligibility = read_json(eligibility_path)
    else:
        eligibility = build_donor_eligibility(case_paths=paths, selected_case_ids=selected,
            source_cases=sources, liver_label=current["labels"]["liver"], tumor_label=current["labels"]["tumor"],
            workers="auto", report_path=folder / "donor_resources.json")
        write_new(eligibility_path, eligibility)
    validate_donor_eligibility(eligibility, selected_case_ids=selected, source_cases=sources, labels=current["labels"])
    envelope = geometry_envelope(eligibility, current)
    proposed = copy.deepcopy(current)
    proposed["graph"]["adaptive_roi_max_voxels"] = max(int(current["graph"]["adaptive_roi_max_voxels"]), envelope["maximum_voxels"])
    proposed["runtime"]["prepare_workers"] = "auto"
    # seed in the config remains the base seed; paired_benchmark adds the fold.
    profile_config = copy.deepcopy(proposed)
    profile_config["seed"] = int(old_cache["seed"])
    resources = snapshot()
    code_identity = preparation_code_identity(plan["project_root"])
    library_versions = {name: importlib.metadata.version(name) for name in
                        ("torch", "numpy", "scipy", "nibabel", "torch-geometric")}
    environment_identity = {"libraries": library_versions,
                            "cpu_capacity": resources["cpu_capacity"],
                            "host_total_memory_bytes": resources["host_total_memory_bytes"],
                            "python_executable": str(plan["python_executable"])}
    sources_by_id = {row["case_id"]: row for row in sources}
    # An allocation precheck, explicitly an estimate (128 bytes/voxel for
    # simultaneous dense ROI fields). It is not substituted for measured RSS.
    estimate = int(proposed["graph"]["adaptive_roi_max_voxels"]) * 128
    if estimate > 0.8 * resources["available_memory_bytes"]:
        raise MemoryError("Full geometric ROI envelope exceeds conservative pilot memory headroom; no context/graph was reduced")
    with (old_gnn / "graphs/manifest.csv").open(newline="", encoding="utf-8") as handle:
        failed = [row for row in csv.DictReader(handle) if row["sample_index"] != "" and row["status"] != "ok"]
    if not failed:
        raise ValueError("Recovery expected failed full-size samples to measure")
    pilot_reports = []
    pilot_files = []
    for row in failed:
        request = {"config": profile_config, "split": split, "gnn_root": str(gnn),
                   "medical_root": str(plan["medical_root"]), "case_id": row["case_id"],
                   "sample_index": int(row["sample_index"]), "code_sha256": code_identity,
                   "source": sources_by_id[row["case_id"]], "prototype_sha256": digest(gnn / "prototype.pt"),
                   "donor_contract_sha256": eligibility["contract_sha256"],
                   "environment": environment_identity}
        stem = f"{row['case_id']}__{int(row['sample_index']):03d}"
        stem += "." + value_sha(request)[:16]
        request_path = folder / "pilots" / f"{stem}.request.json"
        output = folder / "pilots" / f"{stem}.measurement.json"
        if request_path.exists():
            if read_json(request_path) != request:
                raise ValueError("Existing resource pilot request differs; preserved")
        else:
            write_new(request_path, request)
        if output.exists():
            prior = read_json(output)
            if prior["measurement"]["status"] != "complete":
                # Preserve the failed evidence and create a new attempt, never
                # mistake it for a reusable successful measurement.
                output = output.with_name(f"{stem}.{uuid.uuid4().hex}.measurement.json")
            else:
                validate_pilot(prior, request)
                pilot_reports.append(prior)
                pilot_files.extend((request_path, output))
                continue
        print(f"[CALIBRATION full-size] {stem}; analytic ROI ceiling={proposed['graph']['adaptive_roi_max_voxels']}", flush=True)
        runner([plan["python_executable"], "-B", "-m", "tools.feedback_preparation_recovery", "profile",
                "--request", str(request_path), "--output", str(output)],
               cwd=Path(plan["project_root"]), env=env, check=True)
        result = read_json(output)
        validate_pilot(result, request)
        pilot_reports.append(result)
        pilot_files.extend((request_path, output))
    resource_report = {"format": "hiercp_measured_roi_policy_v1", "envelope": envelope,
        "pilot_memory_estimate_bytes": estimate, "estimate_is_measurement": False,
        "resources_before": resources, "pilots": pilot_reports,
        "maximum_measured_peak_rss_bytes": max(item["measurement"]["sampled_peak_rss_bytes"] for item in pilot_reports),
        "graph_geometry_changed": False, "dataset_reduced": False,
        "limitation": "full-size failed samples measured, not every future online placement; analytic ROI bound is not a measured worst-case RSS guarantee"}
    if resource_report["maximum_measured_peak_rss_bytes"] > 0.8 * snapshot()["available_memory_bytes"]:
        raise MemoryError("Measured full-size peak RSS does not leave safe preparation headroom")
    policy_path = folder / f"resource_policy.{value_sha(resource_report)[:16]}.json"
    if policy_path.exists():
        if read_json(policy_path) != resource_report:
            raise ValueError("Existing measured resource policy is incompatible")
    else:
        write_new(policy_path, resource_report)
    config_path = Path(plan["train_config"])
    if config_path.exists():
        if read_json(config_path) != proposed:
            raise ValueError("Existing recovery configuration differs; preserved")
    else:
        write_new(config_path, proposed)
    kwargs = prepare_kwargs(profile_config, split, gnn, plan["medical_root"])
    kwargs["donor_eligibility"] = eligibility
    if not (gnn / "graphs").exists():
        migrate_failed_hierarchical_cache(source_cache_dir=old_gnn / "graphs",
                                         destination_cache_dir=gnn / "graphs", prepare_kwargs=kwargs)
    else:
        migration = read_json(gnn / "graphs/migration.json")
        if migration.get("state") != "ready_for_prepare":
            raise ValueError("Interrupted migration preserved; use another --experiment-name, never --overwrite")
        validate_cache_migration(source_cache_dir=old_gnn / "graphs", destination_cache_dir=gnn / "graphs",
                                 prepare_kwargs=kwargs)
    preparation_env = {**env, "HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES": str(resource_report["maximum_measured_peak_rss_bytes"])}
    runner([plan["python_executable"], "-B", "-m", "tools.paired_benchmark", "gnn-prepare",
            "--project-root", str(plan["project_root"]), "--medical-root", str(plan["medical_root"]),
            "--work", str(target / "paired"), "--outer-fold", str(fold),
            "--train-config", str(config_path), "--device", "cuda:0"],
           cwd=Path(plan["project_root"]), env=preparation_env, check=True)
    validate_cache_publication(gnn / "graphs")
    verify_identity(identity)
    files = [config_path, policy_path, eligibility_path, identity_path,
             target / "paired/outer_splits.json", target / "paired/case_profiles.csv",
             gnn / "split.json", gnn / "prototype.pt", gnn / "metadata.json", gnn / "manifest.csv",
             gnn / "graphs/migration.json", gnn / "graphs/config.json",
             gnn / "graphs/manifest.csv", gnn / "graphs/index.json", gnn / "graphs/complete.json", *pilot_files]
    receipt = {"format": "hiercp_preparation_recovery_complete_v1", "train_config": str(config_path),
               "source_identity": identity, "files": {str(path.relative_to(target)): digest(path) for path in files},
               "original_results_preserved": True, "training_performed": False}
    write_new(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("profile",))
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _profile(read_json(args.request), args.output)


if __name__ == "__main__":
    main()
