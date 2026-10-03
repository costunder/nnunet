"""Bounded, read-only CPU geometry diagnostics for failed native-v1 ROI requests.

This executes no neural model, cache publication or training. Exact target
replay follows the original sample/graph preparation until its first ROI error.
The proposed voxel ceiling exists only in the child process's copy of
GraphBuildConfig. Existing source, config, masks and prepared results stay intact.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FORMAT = "v1_roi_budget_probe_v1"
SCOPE = "debug_geometry_only"


def parse_roi_failure(message: str) -> dict:
    """Extract only the original allocation-guard diagnostic, without eval/code."""
    if not message.startswith("Adaptive ROI request exceeds graph.adaptive_roi_max_voxels;"):
        raise ValueError("Only the original adaptive-native-shape voxel failure can be replayed")
    result = {}
    for name in ("footprint_shape", "spacing_mm", "requested_shape", "effective_shape"):
        found = re.search(rf"\b{name}=(\([^)]*\))", message)
        if found is None:
            raise ValueError(f"ROI diagnostic lacks {name}")
        values = ast.literal_eval(found.group(1))
        if not isinstance(values, tuple) or len(values) != 3:
            raise ValueError(f"Invalid ROI diagnostic {name}")
        if name == "spacing_mm":
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Invalid ROI spacing")
        elif any(type(v) is not int or v <= 0 for v in values):
            raise ValueError(f"Invalid ROI dimensions: {name}")
        result[name] = list(values)
    for name in ("requested_margin_mm", "requested_voxels", "effective_voxels", "voxel_budget"):
        found = re.search(rf"\b{name}=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", message)
        if found is None:
            raise ValueError(f"ROI diagnostic lacks {name}")
        value = float(found.group(1)) if name == "requested_margin_mm" else int(found.group(1))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid ROI diagnostic {name}")
        result[name] = value
    if result["requested_shape"] != result["effective_shape"]:
        raise ValueError("Historical ROI diagnostic reports reduced geometry")
    if math.prod(result["requested_shape"]) != result["requested_voxels"]:
        raise ValueError("ROI shape/voxel diagnostic mismatch")
    if result["requested_voxels"] != result["effective_voxels"]:
        raise ValueError("Historical ROI diagnostic reports reduced voxel count")
    return result


def failure_requests(path: Path) -> list[dict]:
    """Replay every sample-level ROI resource failure; never silently skip one."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    requests = []
    identities = set()
    for row in rows:
        if row.get("status") != "resource_budget_error":
            continue
        if row.get("sample_index", "") == "":
            raise ValueError("Case-level resource failure cannot be replayed as a source ROI")
        try:
            index = int(row["sample_index"])
        except (ValueError, KeyError) as error:
            raise ValueError("Invalid failed sample index") from error
        case_id = row.get("case_id", "")
        if index < 0 or re.fullmatch(r"liver_[0-9]+", case_id) is None:
            raise ValueError("Invalid failed sample identity")
        identity = (case_id, index)
        if identity in identities:
            raise ValueError("Duplicate failed sample identity")
        identities.add(identity)
        requests.append({"case_id": case_id, "sample_index": index,
                         "split_name": row.get("split", ""),
                         "original_failure": row["message"],
                         "geometry": parse_roi_failure(row["message"]),
                         "source_image_sha256": row.get("source_image_sha256"),
                         "source_label_sha256": row.get("source_label_sha256")})
    if not requests:
        raise ValueError("No failed sample-level ROI allocation requests in manifest")
    return sorted(requests, key=lambda r: (r["case_id"], r["sample_index"]))


def _write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def summarize_probe_failure(row: dict, *, message_limit: int = 480) -> dict:
    """Summarize recorded child output; never infer an exception from ROI shape.

    The full traceback and stdout remain in ``row['error']`` / ``row['stdout']``.
    Only this terminal summary is bounded. Unclassified child output is labeled
    explicitly, and a resource stop is distinguished from a child exception.
    """
    if type(message_limit) is not int or message_limit < 1:
        raise ValueError("Failure display message limit must be a positive integer")
    if row.get("status") == "PASS":
        raise ValueError("A successful probe has no failure summary")
    stderr, stdout = row.get("error", ""), row.get("stdout", "")
    if not isinstance(stderr, str) or not isinstance(stdout, str):
        raise TypeError("Recorded child stderr/stdout must be text")
    source, error_class, message = None, None, None
    marker = "Traceback (most recent call last):"
    if marker in stderr:
        # In a chained traceback the final block contains the propagated error.
        lines = stderr.rsplit(marker, 1)[1].strip().splitlines()
        pattern = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?::(?:\s*(.*))?)?$")
        matches = [(index, pattern.match(line)) for index, line in enumerate(lines)]
        matches = [(index, match) for index, match in matches
                   if match is not None and (":" in lines[index] or
                       re.fullmatch(r"(?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*(?:Error|Exception)|KeyboardInterrupt|SystemExit)", match.group(1)))]
        if matches:
            # Later lines belong to a multiline exception message. A line such
            # as ``Tip: ...`` must not replace the actual exception header.
            index, match = matches[0]
            error_class = match.group(1)
            message = "\n".join([match.group(2) or "", *lines[index+1:]]).strip()
            source = "stderr_traceback"
    if source is None:
        if stderr.strip():
            source, message = "stderr_unclassified", stderr.strip().splitlines()[-1]
        elif stdout.strip():
            source, message = "stdout_unclassified", stdout.strip().splitlines()[-1]
        elif row.get("status") in ("RSS_BUDGET", "TIME_BUDGET"):
            source = "diagnostic_resource_budget"
            message = "Child stopped at the explicitly configured " + row["status"]
        elif row.get("answer_missing"):
            source = "child_exit_without_answer"
            message = f"Child returned {row.get('returncode')} without an answer file or diagnostic output"
        else:
            source = "child_exit_without_output"
            message = f"Child returned {row.get('returncode')} without diagnostic output; cause is unknown"
    # Collapse whitespace for one terminal line; this does not alter raw output.
    compact = " ".join(message.split())
    truncated = len(compact) > message_limit
    if truncated:
        compact = compact[:message_limit] + " ... [full output in report]"
    return dict(source=source, error_class=error_class, message=compact,
                message_truncated=truncated)


def format_probe_result(row: dict) -> str:
    """Format one result, including an honest failure reason when available."""
    line = (f"CPU ROI DEBUG | {row['case_id']}[{row['sample_index']}] | {row['status']} | "
            f"RSS={row['peak_rss_bytes']/2**30:.3f} GiB | wall={row['wall_seconds']:.2f}s")
    if row["status"] != "PASS":
        details = summarize_probe_failure(row)
        label = details["error_class"] or details["source"]
        line += f" | {label}"
        if details["message"]:
            line += ": " + details["message"]
    return line


def worker(request_path: str) -> None:
    """Child entry: import the verified frozen snapshot before any hiercp module."""
    from dataclasses import replace
    import hashlib
    import numpy as np
    import psutil
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    snapshot = Path(request["snapshot"]).resolve()
    sys.path.insert(0, str(snapshot))
    from hiercp.common import (CasePaths, load_case,
                               stable_case_seed, ct_normalize, erase_mask_with_context,
                               verify_loaded_case_source_signatures)
    from hiercp.region import (load_patient_regions, _region_cache_metadata,
                              _metadata_equal, REGION_CACHE_SEED_SALT)
    from hiercp.schema import graph_config_from_dict
    from hiercp.spatial import (adaptive_native_shape, build_patch_payload, center_crop_or_pad)
    import hiercp.spatial
    from hiercp import cache, local
    from hiercp.prototype import PrototypeBank
    from .roi_failure_replay import capture_failed_roi
    if Path(hiercp.spatial.__file__).resolve() != snapshot / "hiercp/spatial.py":
        raise ValueError("Worker imported a non-frozen spatial implementation")
    spatial_hash = hashlib.sha256((snapshot / "hiercp/spatial.py").read_bytes()).hexdigest()
    if spatial_hash != request["spatial_sha256"]:
        raise ValueError("Frozen spatial source changed before child execution")
    cfg = request["config"]
    original = graph_config_from_dict(cfg["graph"])
    proposed = replace(original, adaptive_roi_max_voxels=request["candidate_roi_max_voxels"])
    proposed.validate()
    row = request["failure"]
    case_id, index = row["case_id"], row["sample_index"]
    data = Path(request["medical_root"]) / "Data"
    load_start = time.perf_counter()
    case = load_case(CasePaths(case_id, data / "image" / f"{case_id}_0000.nii.gz",
                              data / "labels" / f"{case_id}.nii.gz"))
    for name in ("image", "label"):
        if getattr(case, f"{name}_source_signature")["sha256"] != row[f"source_{name}_sha256"]:
            raise ValueError(f"Failed manifest {name} content does not match actual CT")
    region_root = Path(request["shared"]) / "regions" / case_id
    regions, metadata = load_patient_regions(region_root, mmap=True)
    expected = _region_cache_metadata(case, liver_label=cfg["labels"]["liver"],
        tumor_label=cfg["labels"]["tumor"], config=original, ct_clip=tuple(cfg["ct_clip"]),
        seed=stable_case_seed(cfg["seed"], case_id, REGION_CACHE_SEED_SALT))
    if not _metadata_equal(metadata, expected):
        raise ValueError("Existing region cache does not match original case/config contract")
    bank_path = Path(request['shared'])/'prototype_bank.pt'
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != request['prototype_sha256']:
        raise ValueError('Original prepared prototype changed before replay')
    bank = PrototypeBank.load(bank_path)
    prepared_contract = json.loads((Path(request['shared'])/'cache/config.json').read_text(encoding='utf-8'))
    if prepared_contract.get('prototype_fingerprint') != bank.fingerprint():
        raise ValueError('Replay prototype does not match the failed cache preparation contract')
    c = cfg['cache']
    sample_kwargs = dict(case=case, bank=bank, regions=regions, sample_index=index,
        split_name=row['split_name'], graph_config=original,
        liver_label=int(cfg['labels']['liver']), tumor_label=int(cfg['labels']['tumor']),
        source_selection=str(c['source_selection']), source_pad=int(c['source_pad']),
        total_candidates=int(c['total_candidates']), candidate_pool_size=int(c['candidate_pool_size']),
        easy_fraction=float(c['easy_fraction']), inter_fraction=float(c['inter_fraction']),
        intra_fraction=float(c['intra_fraction']), max_draws=int(c['max_draws']),
        min_liver_coverage=float(c['min_liver_coverage']),
        occupied_clearance_vox=int(c['occupied_clearance_vox']),
        min_center_separation_mm=float(c['min_center_separation_mm']),
        min_center_separation_vox=float(c.get('min_center_separation_vox', 0.0)),
        ct_clip=tuple(float(v) for v in cfg['ct_clip']), seed=int(cfg['seed']))
    geometry = row["geometry"]
    load_seconds = time.perf_counter() - load_start
    start = time.perf_counter()
    captured = capture_failed_roi(cache, local, hiercp.spatial,
        sample_kwargs=sample_kwargs, expected_geometry=geometry, parse_failure=parse_roi_failure)
    replay_seconds = time.perf_counter() - start
    # The replay uses every original candidate/coordinate/graph check. Its
    # exception is consumed only after exact first-error equality is proven.
    # No preceding graph tensor is retained for the measured ROI field phase.
    del sample_kwargs, bank
    import gc
    gc.collect()
    transform_seconds = 0.0
    if captured['operation'] == 'build_patch_payload':
        kwargs = dict(captured['payload_kwargs'])
        if kwargs['config'].to_dict() != original.to_dict():
            raise ValueError('Captured payload used a different original graph contract')
        kwargs['config'] = proposed
        footprint = kwargs['footprint']
    elif captured['operation'] == 'transform_footprint_physical':
        args, tkw = captured['transform_args'], captured['transform_kwargs']
        if len(args) != 4 or tkw or args[3].to_dict() != original.to_dict():
            raise ValueError('Unexpected original transform call contract')
        start = time.perf_counter()
        footprint = local.transform_footprint_physical(args[0], args[1], args[2], proposed)
        transform_seconds = time.perf_counter() - start
        if list(footprint.shape) != geometry['footprint_shape']:
            raise ValueError('Original transformed mask differs from the captured allocation request')
        context = captured['target_context']
        if context is None or context['case'] is not case:
            raise ValueError('Failed physical transform has no exact target context')
        k = context['kwargs']
        kwargs = dict(image=case.image, center=context['spec'].center, footprint=footprint,
            full_organ=k['full_organ_mask'], organ_depth=k['organ_depth'], spacing=case.spacing,
            config=proposed, erase_target=True, ct_clip=k['ct_clip'],
            ct_normalize_fn=ct_normalize, erase_fn=erase_mask_with_context)
    else:
        raise ValueError('Unsupported captured ROI operation')
    center = tuple(int(v) for v in kwargs['center'])
    if any(v < 0 or v >= size for v, size in zip(center, case.image.shape)):
        raise ValueError('Captured original ROI center is out of bounds')
    if list(footprint.shape) != geometry['footprint_shape']:
        raise ValueError('Captured full footprint does not match the original error')
    depth = float(kwargs['organ_depth'][center])
    payload_initial_shape = adaptive_native_shape(footprint.shape, case.spacing, proposed,
                                                 center_liver_depth_mm=depth)
    original_count = int(np.count_nonzero(footprint))
    mask_sha_before = hashlib.sha256(np.ascontiguousarray(footprint).view(np.uint8)).hexdigest()
    before_rss = psutil.Process().memory_info().rss
    start = time.perf_counter()
    payload = build_patch_payload(**kwargs)
    elapsed = time.perf_counter() - start
    actual_shape = list(payload["footprint"].shape)
    geometry_equal = actual_shape == geometry["requested_shape"]
    if (any(a < b for a, b in zip(actual_shape, payload_initial_shape))
            or any(a < b for a, b in zip(payload_initial_shape, geometry['requested_shape']))
            or math.prod(actual_shape) > request['candidate_roi_max_voxels']):
        raise ValueError('Original ROI expansion reduced geometry or exceeded its explicit budget')
    if hashlib.sha256(np.ascontiguousarray(footprint).view(np.uint8)).hexdigest() != mask_sha_before:
        raise ValueError('ROI measurement mutated the captured full mask')
    if int(np.count_nonzero(payload["footprint"])) != original_count:
        raise ValueError("ROI probe changed original full paste footprint")
    expected_mask = center_crop_or_pad(footprint, actual_shape, pad_value=False)
    if not np.array_equal(expected_mask, payload['footprint']):
        raise ValueError('Measured native ROI changed the exact captured mask positions')
    del expected_mask
    unique_arrays = {id(value): value for value in payload.values() if isinstance(value, np.ndarray)}
    if any(not np.all(np.isfinite(value)) for value in unique_arrays.values()):
        raise ValueError("Non-finite original ROI geometry payload")
    verify_loaded_case_source_signatures(case)
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != request['prototype_sha256']:
        raise ValueError('Prepared prototype changed during replay')
    peak_rss = None
    if sys.platform != "win32":
        import resource
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    answer = dict(case_id=case_id, sample_index=index, status="PASS",
        cpu_geometry_only=True, source_component=captured['source_component'],
        source_anchor=captured['source_anchor'], original_guard_reproduced=True,
        replay_contract='original_sample_first_roi_failure_v1',
        original_failure_phase=captured['phase'], failed_operation=captured['operation'],
        target_spec=captured['target_spec'], actual_center=list(center),
        erase_target=bool(kwargs['erase_target']), replayed_geometry=captured['replayed_geometry'],
        replay_seconds=replay_seconds, transform_seconds=transform_seconds,
        original_requested_shape=geometry["requested_shape"], requested_shape=geometry['requested_shape'],
        payload_initial_shape=list(payload_initial_shape), geometry_preserved=True, full_mask_preserved=True,
        effective_shape=actual_shape, geometry_equal=geometry_equal,
        original_requested_voxels=geometry["requested_voxels"],
        roi_voxels=math.prod(actual_shape), full_mask_voxels=original_count,
        model_input_shape=list(payload["model_input"].shape), load_seconds=load_seconds,
        geometry_seconds=elapsed, retained_payload_bytes=sum(v.nbytes for v in unique_arrays.values()),
        rss_before_geometry_bytes=before_rss,
        rss_after_geometry_bytes=psutil.Process().memory_info().rss,
        process_peak_rss_bytes=peak_rss,
        spatial_sha256=spatial_hash, source_image_sha256=row["source_image_sha256"],
        source_label_sha256=row["source_label_sha256"],
        limitation="Exact first failed source/target ROI cost; earlier original graphs may be replayed; no full-cache admission or training quality claim")
    _write_new(Path(request["answer_path"]), answer)


def run_probe(experiment: str, output: str, *, candidate_voxels: int,
              rss_bytes: int, case_timeout_seconds: float, reuse_report=None) -> dict:
    from .contracts import V1_ARCHIVE_SHA256
    from .experiment import load_suite, preparation_root, digest
    import psutil
    if type(candidate_voxels) is not int or candidate_voxels <= 0:
        raise ValueError("Candidate ROI voxel budget must be an explicit positive integer")
    if type(rss_bytes) is not int or rss_bytes <= 0:
        raise ValueError("Explicit RSS resource budget must be positive")
    if not math.isfinite(case_timeout_seconds) or case_timeout_seconds <= 0:
        raise ValueError("Explicit per-case timeout must be finite and positive")
    root = Path(experiment).resolve()
    manifest = load_suite(root)
    if manifest.get("sampling_contract", {}).get("mode") != "native":
        raise ValueError("ROI diagnostic must use the original native preparation owner")
    shared = preparation_root(root, manifest)
    progress = shared / "cache" / "manifest.csv"
    requests = failure_requests(progress)
    for row in requests:
        case_id = row['case_id']
        expected_split = ('train' if case_id in manifest['split']['train'] else
                          'val' if case_id in manifest['split']['val'] else None)
        if expected_split is None or row['split_name'] not in (expected_split, ''):
            raise ValueError('Failed ROI request is not bound to the original train/val split')
        row['split_name'] = expected_split
    cfg_path = root / "configs" / "v1.0.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    original_voxels = cfg["graph"]["adaptive_roi_max_voxels"]
    if original_voxels != 8_000_000 or candidate_voxels <= original_voxels:
        raise ValueError("Probe requires original 8000000 guard and an explicit larger candidate")
    if any(r["geometry"]["voxel_budget"] != original_voxels for r in requests):
        raise ValueError("Failed manifest belongs to a different ROI allocation guard")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Existing probe report is preserved; choose a new output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(destination.stem + "_requests")
    work.mkdir(exist_ok=False)
    snapshot = root / "source" / "v1.0"
    spatial_sha = digest(snapshot / "hiercp/spatial.py")
    bank_path = shared/'prototype_bank.pt'
    watched = [root / "manifest.json", cfg_path, progress, bank_path, shared/'cache/config.json']
    for row in requests:
        for name, folder, suffix in (('image', 'image', '_0000.nii.gz'), ('label', 'labels', '.nii.gz')):
            path = Path(manifest['medical_root'])/'Data'/folder/(row['case_id']+suffix)
            if digest(path) != row[f'source_{name}_sha256']:
                raise ValueError('Failed ROI manifest raw CT bytes changed')
            if path not in watched:
                watched.append(path)
    signatures = {str(p): digest(p) for p in watched}
    reuse, reuse_provenance = {}, None
    if reuse_report is not None:
        from .roi_probe_reuse import verified_source_reuse
        reuse, reuse_provenance = verified_source_reuse(reuse_report, experiment=root,
            requests=requests, config=cfg, snapshot=snapshot, spatial_sha256=spatial_sha,
            medical_root=manifest['medical_root'],
            failed_manifest_sha256=signatures[str(progress)], candidate_voxels=candidate_voxels,
            rss_bytes=rss_bytes, case_timeout_seconds=case_timeout_seconds)
        watched_reuse = [Path(reuse_provenance['report']),
                         *map(Path, reuse_provenance['bound_sidecars'])]
        signatures.update({str(p): digest(p) for p in watched_reuse})
    rows = []
    tool = Path(__file__).resolve().parents[1] / "tools" / "probe_v1_roi_budget.py"
    for ordinal, failure in enumerate(requests):
        previous = reuse.get((failure['case_id'], failure['sample_index']))
        if previous is not None:
            rows.append(previous)
            print('REUSED verified source measurement | '+format_probe_result(previous), flush=True)
            continue
        request_path = work / f"{ordinal:03d}_request.json"
        answer_path = work / f"{ordinal:03d}_answer.json"
        _write_new(request_path, dict(snapshot=str(snapshot), config=cfg,
            candidate_roi_max_voxels=candidate_voxels, failure=failure,
            medical_root=manifest["medical_root"], shared=str(shared),
            spatial_sha256=spatial_sha, prototype_sha256=signatures[str(bank_path)],
            answer_path=str(answer_path)))
        command = [sys.executable, "-u", str(tool), "--worker-request", str(request_path)]
        print(f"CPU ROI DEBUG | {ordinal+1}/{len(requests)} | {failure['case_id']}[{failure['sample_index']}]", flush=True)
        started = time.perf_counter()
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace")
        observed_peak = 0
        stopped = None
        try:
            while proc.poll() is None:
                try:
                    observed_peak = max(observed_peak, psutil.Process(proc.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    break  # Collect the child's actual return code/output below.
                elapsed = time.perf_counter() - started
                if observed_peak > rss_bytes or elapsed > case_timeout_seconds:
                    stopped = "RSS_BUDGET" if observed_peak > rss_bytes else "TIME_BUDGET"
                    print(f"Stopping owned diagnostic child PID {proc.pid}: {stopped}; command={command!r}", flush=True)
                    proc.terminate()
                    break
                time.sleep(0.05)
            stdout, stderr = proc.communicate()
        except KeyboardInterrupt:
            if proc.poll() is None:
                print(f"Stopping owned diagnostic child PID {proc.pid}: user interrupted ROI probe", flush=True)
                proc.terminate()
            proc.communicate()
            raise
        if stopped or proc.returncode or not answer_path.is_file():
            row = dict(case_id=failure["case_id"], sample_index=failure["sample_index"],
                       status=stopped or "FAILED", returncode=proc.returncode,
                       peak_rss_bytes=observed_peak, wall_seconds=time.perf_counter()-started,
                       error=stderr, stdout=stdout, answer_missing=not answer_path.is_file())
        else:
            row = json.loads(answer_path.read_text(encoding="utf-8"))
            row["peak_rss_bytes"] = max(observed_peak, row["process_peak_rss_bytes"] or 0)
            row["wall_seconds"] = time.perf_counter()-started
            if row["peak_rss_bytes"] > rss_bytes:
                row["status"] = "RSS_BUDGET"
        if row["status"] != "PASS":
            row["failure_summary"] = summarize_probe_failure(row)
        rows.append(row)
        print(format_probe_result(row), flush=True)
    preserved = all(digest(Path(name)) == sha for name, sha in signatures.items())
    report = dict(format=FORMAT, scope=SCOPE, archived_source_sha256=V1_ARCHIVE_SHA256,
        replay_protocol='original_sample_first_roi_failure_v1',
        implementation_sha256={name: digest(Path(__file__).parent/name) for name in
            ('roi_budget_probe.py', 'roi_failure_replay.py', 'roi_probe_reuse.py')},
        original_roi_max_voxels=original_voxels, candidate_roi_max_voxels=candidate_voxels,
        experiment=str(root), failed_manifest_sha256=signatures[str(progress)],
        resource_budget=dict(rss_bytes=rss_bytes, case_timeout_seconds=case_timeout_seconds,
                             rss_sampling_interval_seconds=0.05, concurrent_cases=1,
                             concurrency_reason="Isolated whole-process peak measurement per full failing source ROI"),
        failed_sample_requests=len(requests), measurements=rows,
        reused_source_measurements=len(reuse), new_measurements=len(requests)-len(reuse),
        reuse_provenance=reuse_provenance,
        completed=preserved and all(row["status"] == "PASS" for row in rows),
        originals_preserved=preserved, training_started=False, cache_publication_created=False,
        production_ready=False, quality_verified=False,
        limitation="Exact failed ROI costs only; original graph preparation may be replayed to reach target errors; no full-cache capacity or training-quality validation")
    _write_new(destination, report)
    return report
