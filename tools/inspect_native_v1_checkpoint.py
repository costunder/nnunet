"""Read known server V1 checkpoint metadata on CPU without selecting a model.

Every missing file, load failure and preservation failure is reported. Saved
filenames are never parsed to invent training patient identities. Adjacent JSON
is evidence of its current bytes, not proof that its configuration was trained.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
FORMAT = "native_v1_checkpoint_CPU_inspection_v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def known_candidates(medical_root, *, include_recent_native=False):
    """Literal paths from preserved launchers/terminal records, not existence claims."""
    medical = Path(medical_root).resolve()
    legacy = medical / "HierCP"
    native = medical / "experiments/v1_native_nested416_seed42_20261003/native"
    full = legacy / "work/full"
    paired = legacy / "work/paired_basic_vs_hiercp/folds/fold_0/gnn"
    candidates = [
        dict(label="historical_full", checkpoint=full / "model.pt", preserved_root=legacy,
             prototype=full / "prototype.pt",
             sidecars={"config": legacy / "config/train.json", "split": full / "split.json",
                       "preflight": full / "model.pt.preflight.json",
                       "cache_config": full / "graphs/config.json",
                       "cache_index": full / "graphs/index.json"},
             recorded_origin="recovered terminal f4e3b499-a931-484a-832c-d2aac078fd54; historical train105/val26"),
        dict(label="historical_paired_fold0", checkpoint=paired / "model.pt", preserved_root=legacy,
             prototype=paired / "prototype.pt",
             sidecars={"config": legacy / "config/train.json", "split": paired / "split.json",
                       "preflight": paired / "model.pt.preflight.json",
                       "cache_config": paired / "graphs/config.json",
                       "cache_index": paired / "graphs/index.json"},
             recorded_origin="recovered terminal ee0258b7-a9e4-40b2-a0e7-506017907681; model.last.pt completed40/40 at that recorded time"),
    ]
    if include_recent_native:
        candidates.append(dict(label="native_suite_20261003", checkpoint=native / "results/v1.0/checkpoint_best.pt", preserved_root=native,
             prototype=native / "shared/prototype_bank.pt",
             sidecars={"config": native / "configs/v1.0.json", "split": native / "shared/split.json",
                       "manifest": native / "manifest.json",
                       "resolved_config": native / "results/v1.0/resolved_config.json",
                       "launch_contract": native / "results/v1.0/launch_contract.json",
                       "preflight": native / "results/v1.0/checkpoint_best.pt.preflight.json",
                       "cache_config": native / "shared/cache/config.json",
                       "cache_index": native / "shared/cache/index.json"},
             recorded_origin="run_v1_native_nested416_server.sh default native path; launcher contract alone does not establish training completion"))
    return candidates


def _ids(value, label):
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Explicit unique nonempty case-id list required: {label}")
    return list(value)


def _json_metadata(value, label):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Saved metadata is not finite JSON data: {label}") from error


def _at(value, path):
    for field in path:
        if not isinstance(value, Mapping) or field not in value:
            return None
        value = value[field]
    return value


def _scope(graph):
    roi = graph.get("adaptive_roi_margin_mm") if isinstance(graph, Mapping) else None
    outer = graph.get("context_outer_radius_mm") if isinstance(graph, Mapping) else None
    shells = graph.get("context_shells_mm") if isinstance(graph, Mapping) else None
    if type(roi) not in (int, float) or type(outer) not in (int, float):
        kind = "UNKNOWN"
    elif roi == 30 and outer == 28 and shells in ([4, 12, 28], (4, 12, 28)):
        kind = "ORIGINAL_NATIVE_ROI30_CONTEXT28"
    elif roi == 30 and outer == 30:
        kind = "BOUNDED30_CONTEXT30_NOT_ORIGINAL_NATIVE"
    else:
        kind = "OTHER_SCOPE"
    return dict(classification=kind, adaptive_roi_margin_mm=roi,
                context_outer_radius_mm=outer, context_shells_mm=shells)


def extract_metadata(payload, *, heldout_case_ids=None):
    """Extract actual saved fields; unknown training identities stay UNKNOWN."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a metadata mapping")
    heldout = set(_ids(heldout_case_ids, "reference held-out cases")) if heldout_case_ids is not None else None
    fields = ("format", "method", "architecture_version", "geometry_contract", "framework",
              "upper_feature_policy", "model_kwargs", "graph_config", "ct_clip", "epoch",
              "best_epoch", "completed_epoch", "target_epochs", "training_complete",
              "selection", "best_selection", "best_mrr", "selection_mrr", "prototype_bank",
              "prototype_fingerprint", "cache_publication", "training_signature", "validation_policy")
    saved = {field: _json_metadata(payload[field], field) for field in fields if field in payload}
    signature = payload.get("training_signature")
    evidence = []
    for path in (("training_case_ids",), ("train_case_ids",),
                 ("training_signature", "training_case_ids"), ("training_signature", "train_case_ids"),
                 ("preflight_calibration", "training_case_ids"), ("preflight_calibration", "train_case_ids"),
                 ("preflight_calibration", "split", "train"), ("split", "train")):
        value = _at(payload, path)
        if value is not None:
            evidence.append(dict(source="checkpoint." + ".".join(path), case_ids=_ids(value, ".".join(path))))
    prototype = payload.get("prototype_training_cases")
    prototype_ids = _ids(prototype, "prototype_training_cases") if prototype is not None else None
    state = payload.get("state_dict")
    result = dict(saved_fields=saved, saved_field_names=sorted(saved),
        state_dict_present=isinstance(state, Mapping) and bool(state),
        state_dict_tensor_entries=len(state) if isinstance(state, Mapping) else None,
        state_dict_values_printed=False, scope=_scope(payload.get("graph_config")),
        seed_evidence={"checkpoint.seed": _json_metadata(payload["seed"], "seed")} if "seed" in payload else {},
        training_case_evidence=evidence, prototype_training_cases=prototype_ids,
        prototype_cases_are_optimizer_training_proof=False,
        heldout_reference_available=heldout is not None)
    result["architecture_identity"] = dict(
        status="SAVED_ARCHITECTURE_TAG" if "architecture_version" in saved else "ARCHITECTURE_TAG_NOT_RECORDED",
        saved_architecture_tag=saved.get("architecture_version"), saved_method=saved.get("method"),
        saved_framework=saved.get("framework"), model_kwargs_present=isinstance(saved.get("model_kwargs"), Mapping),
        model_kwargs=saved.get("model_kwargs"), strict_model_load_executed=False,
        model_operator_compatibility_verified=False)
    if isinstance(signature, Mapping) and "seed" in signature:
        result["seed_evidence"]["checkpoint.training_signature.seed"] = _json_metadata(signature["seed"], "signature seed")
    _update_training_status(result, heldout)
    result["prototype_heldout_overlap"] = (sorted(set(prototype_ids) & heldout)
        if prototype_ids is not None and heldout is not None else None)
    result["prototype_heldout_independence"] = ("UNKNOWN" if result["prototype_heldout_overlap"] is None
        else "OVERLAP_DETECTED" if result["prototype_heldout_overlap"] else "NO_OVERLAP_IN_SAVED_PROTOTYPE_IDS")
    return result


def _update_training_status(metadata, heldout):
    evidence = metadata["training_case_evidence"]
    groups = {tuple(sorted(row["case_ids"])) for row in evidence}
    detected = sorted({case for row in evidence for case in row["case_ids"]})
    metadata.update(training_case_ids=detected if evidence else None,
        training_case_status="UNKNOWN" if not evidence else "CONFLICT" if len(groups) != 1 else "EXPLICIT_SAVED_IDS",
        training_heldout_overlap=sorted(set(detected) & heldout) if evidence and heldout is not None else None)
    overlap = metadata["training_heldout_overlap"]
    metadata["heldout_independence"] = ("UNKNOWN" if overlap is None else "OVERLAP_DETECTED" if overlap
        else "CONFLICTING_TRAINING_EVIDENCE" if len(groups) > 1 else "NO_OVERLAP_IN_EXPLICIT_SAVED_IDS")


def _error(phase, error):
    return dict(phase=phase, type=type(error).__name__, message=str(error))


def _read_sidecar(path):
    result = dict(path=str(path), status="MISSING", errors=[])
    content = None
    try:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Regular adjacent JSON file unavailable: {path}")
        result["sha256_before"] = sha(path)
        content = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(content, Mapping):
            raise ValueError("Adjacent JSON must contain a metadata mapping")
        result.update(status="READ", observed_current_JSON_only=True,
                      current_config_is_training_proof=False,
                      field_names=sorted(content))
        for field in ("format", "graph", "model", "seed", "split", "train", "val", "training_case_ids", "train_case_ids"):
            if field in content:
                result.setdefault("selected_fields", {})[field] = _json_metadata(content[field], field)
    except Exception as error:
        result["status"] = "MISSING" if isinstance(error, FileNotFoundError) else "ERROR"
        result["errors"].append(_error("read_adjacent_JSON", error))
    finally:
        if "sha256_before" in result:
            try:
                result["sha256_after"] = sha(path)
                result["file_preserved"] = result["sha256_after"] == result["sha256_before"]
                if not result["file_preserved"]:
                    raise ValueError("Adjacent JSON bytes changed during metadata inspection")
            except Exception as error:
                result["status"] = "ERROR"
                result["errors"].append(_error("verify_adjacent_JSON_preservation", error))
                content = None
    return result, content


def _literal_cache_case_ids(index, files):
    """Match saved literal filenames to explicit current index case_id fields."""
    files = _ids(files, "training_signature.train_cache_files")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Cache index lacks actual entry metadata")
    by_file = {}
    for entry in entries:
        if (not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str)
                or not entry["path"] or entry["path"] in by_file):
            raise ValueError("Cache index has invalid/duplicate literal entry path")
        by_file[entry["path"]] = entry
    selected = []
    for filename in files:
        entry = by_file.get(filename)
        if (entry is None or entry.get("split") != "train"
                or not isinstance(entry.get("case_id"), str) or not entry["case_id"]):
            raise ValueError("Saved train_cache_files lacks matching explicit training case_id entry")
        selected.append(entry["case_id"])
    return sorted(set(selected)), len(files)


def _bound_cache_ids(payload, sidecars, contents):
    """Read literal case_id fields only when the saved checkpoint binds index bytes."""
    index = contents.get("cache_index")
    expected_sha = _at(payload, ("cache_publication", "index_sha256"))
    files = _at(payload, ("training_signature", "train_cache_files"))
    if index is None or expected_sha is None or files is None:
        return None
    sidecar = sidecars["cache_index"]
    if sidecar.get("status") != "READ" or not sidecar.get("file_preserved"):
        raise ValueError("SHA-bound cache index was not successfully read and preserved")
    observed_sha = sidecar.get("sha256_before")
    if observed_sha != expected_sha:
        raise ValueError("Adjacent cache index differs from checkpoint's saved index SHA256")
    case_ids, count = _literal_cache_case_ids(index, files)
    return dict(source="checkpoint.train_cache_files + SHA-bound cache_index.entries.case_id",
                case_ids=case_ids, index_sha256=observed_sha,
                training_cache_files=count, filename_patient_inference=False)


def _reconstructed_cache_ids(payload, sidecars, contents, heldout):
    """Separate current mapping evidence; never establish historical independence."""
    result = dict(status="UNAVAILABLE", training_time_hash_binding=False,
                  historical_optimizer_training_verified=False, filename_patient_inference=False,
                  used_for_heldout_independence=False, case_ids=None, current_mapping_reference_overlap=None)
    if _at(payload, ("cache_publication", "index_sha256")) is not None:
        result.update(status="NOT_APPLICABLE", reason="Saved cache index SHA is handled as separate bound evidence")
        return result
    files = _at(payload, ("training_signature", "train_cache_files"))
    index = contents.get("cache_index")
    sidecar = sidecars.get("cache_index", {})
    if files is None or index is None or sidecar.get("status") != "READ" or not sidecar.get("file_preserved"):
        result["reason"] = "Saved train_cache_files and successfully read current cache index are both required"
        return result
    case_ids, count = _literal_cache_case_ids(index, files)
    result.update(status="RECONSTRUCTED_CURRENT_MAPPING_NO_TRAINING_TIME_HASH_BINDING",
                  source="checkpoint.train_cache_files + current cache_index.entries.case_id",
                  case_ids=case_ids, current_index_sha256=sidecar["sha256_before"], training_cache_files=count,
                  current_mapping_reference_overlap=sorted(set(case_ids) & heldout) if heldout is not None else None)
    return result


def _cpu_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def inspect_candidate(spec, *, heldout_case_ids=None, load_checkpoint=None):
    """Inspect one literal candidate, report failures, and verify original bytes."""
    path = Path(spec["checkpoint"])
    result = dict(label=spec["label"], path=str(path), preserved_root=str(spec['preserved_root']), recorded_origin=spec["recorded_origin"],
                  status="MISSING", errors=[], selected_for_evaluation=False,
                  checkpoint_file_present=False, checkpoint_metadata_read=False)
    loader = _cpu_load if load_checkpoint is None else load_checkpoint
    try:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Known regular checkpoint unavailable: {path}")
        result["checkpoint_file_present"] = True
        result.update(sha256_before=sha(path), size_bytes=path.stat().st_size)
        payload = loader(path)
        metadata = extract_metadata(payload, heldout_case_ids=heldout_case_ids)
        result.update(status="READ_CPU_METADATA", load_settings=dict(map_location="cpu", weights_only=False, mmap=True),
                      metadata=metadata, checkpoint_metadata_read=True)
        sidecars, contents = {}, {}
        for label, sidecar in spec["sidecars"].items():
            sidecars[label], contents[label] = _read_sidecar(Path(sidecar))
        result["adjacent_JSON"] = sidecars
        evidence = _bound_cache_ids(payload, sidecars, contents)
        if evidence is not None:
            metadata["training_case_evidence"].append(evidence)
            _update_training_status(metadata, set(heldout_case_ids) if heldout_case_ids is not None else None)
        try:
            metadata["reconstructed_current_training_mapping"] = _reconstructed_cache_ids(
                payload, sidecars, contents, set(heldout_case_ids) if heldout_case_ids is not None else None)
        except Exception as error:
            metadata["reconstructed_current_training_mapping"] = dict(
                status="ERROR", errors=[_error("reconstruct_current_cache_mapping", error)],
                training_time_hash_binding=False, used_for_heldout_independence=False,
                historical_optimizer_training_verified=False, filename_patient_inference=False,
                case_ids=None, current_mapping_reference_overlap=None)
        prototype = Path(spec["prototype"])
        result["known_adjacent_prototype"] = dict(path=str(prototype), exists=prototype.is_file(),
                                                  loaded=False, role="recorded adjacent artifact; filename alone does not bind trained bank")
        saved_path = metadata["saved_fields"].get("prototype_bank")
        result["saved_prototype_path_matches_known_adjacent"] = (str(prototype) == str(saved_path)
            if isinstance(saved_path, str) else None)
    except Exception as error:
        result["status"] = "MISSING" if isinstance(error, FileNotFoundError) else "ERROR"
        result["errors"].append(_error("CPU_checkpoint_metadata_inspection", error))
    finally:
        if "sha256_before" in result:
            try:
                result["sha256_after"] = sha(path)
                result["file_preserved"] = result["sha256_after"] == result["sha256_before"]
                if not result["file_preserved"]:
                    raise ValueError("Original checkpoint bytes changed during metadata inspection")
            except Exception as error:
                result["status"] = "ERROR"
                result["file_preserved"] = False
                result["errors"].append(_error("verify_checkpoint_preservation", error))
    return result


def inspect_reference_summary(path, *, debug=False):
    # Reuse all four sealed report/cohort/denominator/order/endpoint checks.
    from tools.analyze_full128_competition import analyze_summary
    checked = analyze_summary(path, debug=debug)
    cohorts = [set(case["case_id"] for case in arm["cases"]) for arm in checked["arms"].values()]
    if any(cohort != cohorts[0] for cohort in cohorts) or (not debug and len(cohorts[0]) != 21):
        raise ValueError("Reference summary does not preserve every same held-out case in all arms")
    return dict(path=str(Path(path).resolve()), status="VERIFIED_SAVED_REPORTS", debug=debug,
                case_ids=sorted(cohorts[0]), cases=len(cohorts[0]),
                original_files_sha256=checked["original_files_sha256"], neural_forward_executed=False)


def inspect_known(medical_root, *, reference_summary=None, debug=False, include_recent_native=False, load_checkpoint=None):
    reference = dict(status="UNKNOWN", reason="No reference summary supplied; held-out identity overlap cannot be determined")
    heldout = None
    if reference_summary is not None:
        try:
            reference = inspect_reference_summary(reference_summary, debug=debug)
            heldout = reference["case_ids"]
        except Exception as error:
            reference = dict(path=str(reference_summary), status="ERROR", errors=[_error("verify_reference_summary", error)])
    candidates = [inspect_candidate(spec, heldout_case_ids=heldout, load_checkpoint=load_checkpoint)
                  for spec in known_candidates(medical_root, include_recent_native=include_recent_native)]
    return dict(format=FORMAT, medical_root=str(Path(medical_root).resolve()), debug=debug,
                reference=reference, candidates=candidates, candidate_count=len(candidates),
                include_recent_native=include_recent_native,
                checkpoint_files_present=sum(row["checkpoint_file_present"] for row in candidates),
                checkpoint_metadata_read=sum(row["checkpoint_metadata_read"] for row in candidates),
                checkpoint_state_dicts_present=sum(row.get("metadata", {}).get("state_dict_present", False) for row in candidates),
                known_paths_are_existence_claims=False, checkpoint_selected=False,
                evaluation_started=False, neural_forward_executed=False, training_started=False,
                GPU_used=False, optimizer_instantiated=False, optimizer_state_restored=False, quality_verified=False,
                read_only=True, originals_written=False,
                inspection_errors=sum(row["status"] != "READ_CPU_METADATA" for row in candidates)
                                  + int(reference["status"] == "ERROR"))


def write_new_report(path, report):
    path = Path(path).resolve()
    protected = [Path(row["path"]).resolve() for row in report["candidates"]]
    reference = report["reference"].get("original_files_sha256", {})
    protected.extend(Path(item).resolve() for item in reference)
    for row in report["candidates"]:
        protected.extend(Path(item["path"]).resolve() for item in row.get("adjacent_JSON", {}).values())
    directories=[Path(row['preserved_root']).resolve() for row in report['candidates']]
    if 'path' in report['reference']:
        directories.append(Path(report['reference']['path']).resolve().parent)
    if path in protected or any(path.is_relative_to(old) for old in directories):
        raise ValueError("Inspection output must be a new file outside original artifacts")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-root", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--include-recent-native", action="store_true",
                        help="Also inspect the separately launched 20261003 native suite; default is the two historical checkpoints")
    args = parser.parse_args()
    report = inspect_known(args.medical_root, reference_summary=args.reference_summary, debug=args.debug,
                           include_recent_native=args.include_recent_native)
    write_new_report(args.output, report)
    print("V1 checkpoint inspection | CPU metadata only | no selection / forward / training", flush=True)
    print("Reference: " + report["reference"]["status"], flush=True)
    count = report["candidate_count"]
    print(f"Checkpoint files: {report['checkpoint_files_present']}/{count} present "
          f"| metadata: {report['checkpoint_metadata_read']}/{count} read "
          f"| saved state_dict weights: {report['checkpoint_state_dicts_present']}/{count}", flush=True)
    for row in report["candidates"]:
        metadata = row.get("metadata", {})
        saved=metadata.get('saved_fields',{})
        print(f"{row['label']} | {row['status']} | scope={metadata.get('scope', {}).get('classification', 'UNKNOWN')} "
              f"| best_epoch={saved.get('epoch','UNKNOWN')} | completed={saved.get('training_complete','UNKNOWN')} "
              f"| train IDs={metadata.get('training_case_status', 'UNKNOWN')} "
              f"| held-out={metadata.get('heldout_independence', 'UNKNOWN')} | {row['path']}", flush=True)
        if metadata:
            architecture = metadata["architecture_identity"]
            print('  architecture tag='+architecture['status']
                  +' | saved tag='+str(architecture['saved_architecture_tag'])
                  +' | saved method='+str(architecture['saved_method'])
                  +' | saved model_kwargs='+json.dumps(architecture['model_kwargs'],sort_keys=True)
                  +' | strict model load / operator compatibility: unverified'
                  +' | train overlap='+json.dumps(metadata.get('training_heldout_overlap'))
                  +' | prototype overlap='+json.dumps(metadata.get('prototype_heldout_overlap')),flush=True)
            mapping = metadata.get("reconstructed_current_training_mapping", {})
            print('  Current cache mapping: '+mapping.get('status','UNAVAILABLE')
                  +' | mapped_count='+str(len(mapping['case_ids']) if mapping.get('case_ids') is not None else 'UNKNOWN')
                  +' | reference overlap='+json.dumps(mapping.get('current_mapping_reference_overlap'))
                  +' | auxiliary evidence; excluded from held-out independence verdict',flush=True)
            for error in mapping.get("errors", []):
                print(f"  MAPPING ERROR: {error['type']}: {error['message']}", flush=True)
        for error in row["errors"]:
            print(f"  ERROR {error['phase']}: {error['type']}: {error['message']}", flush=True)
        for label, sidecar in row.get("adjacent_JSON", {}).items():
            for error in sidecar["errors"]:
                if sidecar['status']!='MISSING':
                    print(f"  ADJACENT {label}: {error['type']}: {error['message']}", flush=True)
        missing=[label for label,item in row.get('adjacent_JSON',{}).items() if item['status']=='MISSING']
        if missing:print('  Optional adjacent JSON absent: '+', '.join(missing)
                         +' (does not mean checkpoint weights are missing; details in REPORT)',flush=True)
    for error in report["reference"].get("errors", []):
        print(f"REFERENCE ERROR: {error['type']}: {error['message']}", flush=True)
    print("REPORT: " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
