"""Explicit server-only, full-validation native/strict-nested paired evaluation.

Both completed 40-epoch runs must pass collect_result before CUDA evaluation.
This re-evaluates their selected best states; it never trains, writes checkpoints,
or labels a graph quality PASS. Statistical helpers perform no neural execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mrr", "top1", "margin")


def score_metrics(scores):
    """Original v1 metric algebra, batched on CUDA; positive is column zero.

    Ties count against the anchor, exactly as original ranking_metric_sums.
    Margin is positive minus the hardest negative, as original epoch_pass.
    """
    import torch
    if not scores.is_cuda:
        raise ValueError("Neural score metric probe requires CUDA; no CPU fallback")
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] != 8:
        raise ValueError("Original v1 evaluation requires eight candidates per sample")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Nonfinite original ranking scores")
    rank = 1 + (scores[:, 1:] >= scores[:, :1]).sum(dim=1)
    margin = scores[:, 0] - scores[:, 1:].max(dim=1).values
    return torch.stack((rank.float().reciprocal(), (rank == 1).float(), margin), dim=1)


def aggregate_cases(rows, cohort):
    """Average sample metrics within each case; candidates/views are not units."""
    if not cohort or len(set(cohort)) != len(cohort):
        raise ValueError("Case cohort must be nonempty and unique")
    groups = {case: [] for case in cohort}
    sample_ids = set()
    for row in rows:
        case, sample_id = row["case_id"], row["sample_id"]
        if case not in groups or not sample_id or sample_id in sample_ids:
            raise ValueError("Unknown case or duplicate/missing sample identity")
        sample_ids.add(sample_id)
        if any(isinstance(row[key], bool) or not isinstance(row[key], (int, float))
               or not math.isfinite(row[key]) for key in METRICS):
            raise ValueError("Every metric must be a finite measured value")
        if any(not 0 <= row[key] <= 1 for key in ("mrr", "top1")):
            raise ValueError("MRR/top1 outside [0,1]")
        groups[case].append(row)
    if any(not values for values in groups.values()):
        raise ValueError("Every indexed validation case must have measured samples")
    return [dict(case_id=case, samples=len(values), sample_ids=sorted(r["sample_id"] for r in values),
                 **{key: math.fsum(r[key] for r in values) / len(values) for key in METRICS})
            for case, values in sorted(groups.items())]


def paired_summary(native, nested, *, resamples, confidence, seed):
    """Percentile paired bootstrap of case-mean deltas, never candidate rows."""
    import numpy as np
    if type(resamples) is not int or resamples < 2 or not 0 < confidence < 1:
        raise ValueError("Explicit bootstrap resamples >=2 and confidence in (0,1) are required")
    def index(rows):
        result = {row["case_id"]: row for row in rows}
        if len(result) != len(rows) or len(result) < 2:
            raise ValueError("At least two distinct paired cases are required")
        return result
    old, new = index(native), index(nested)
    if old.keys() != new.keys():
        raise ValueError("Paired evaluation has different case cohorts")
    cases = sorted(old)
    for case in cases:
        if old[case]["samples"] != new[case]["samples"] or old[case]["sample_ids"] != new[case]["sample_ids"]:
            raise ValueError("Paired evaluation changed sample identities/denominators")
    delta = np.array([[new[case][key] - old[case][key] for key in METRICS] for case in cases], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired metrics")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cases), size=(resamples, len(cases)))
    distributions = delta[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    intervals = np.quantile(distributions, (alpha, 1.0-alpha), axis=0)
    return dict(resampling_unit="case_id", case_ids=cases, cases=len(cases),
                bootstrap_resamples=resamples, confidence=confidence, bootstrap_seed=seed,
                method="paired percentile bootstrap of equally weighted case means",
                metrics={key: dict(delta_nested_minus_native=float(delta[:, i].mean()),
                                   interval=[float(intervals[0, i]), float(intervals[1, i])])
                         for i, key in enumerate(METRICS)},
                graph_quality_passed=False, quality_tolerance=None, automatic_promotion=False,
                scope="Selected best checkpoints on fixed validation; conditional uncertainty, not an independent test or CP efficacy")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def child(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    source = Path(request["source"]).resolve(strict=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(source))
    import torch
    import psutil
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from hiercp import sample as sample_module
    from hiercp.data import HierarchicalCacheDataset, collate_samples, CudaPrefetchLoader
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import configure_runtime, set_seed, load_checkpoint
    from hiercp.loss import ranking_metric_sums
    from hiercp_v1x.sampling_runtime import install_from_environment, load_environment_contract
    if Path(sample_module.__file__).resolve() != source / "hiercp/sample.py":
        raise RuntimeError("Evaluator imported another sampler snapshot")
    install_from_environment()  # Native verifies snapshot without replacing its factory.
    contract = load_environment_contract()
    if contract != request["sampling_contract"] or contract["mode"] != request["mode"]:
        raise ValueError("Child request differs from its immutable sampler contract")
    cohort = request["cohort"]
    if len(cohort) != 21 or len(set(cohort)) != 21:
        raise ValueError("Internal evaluator requires the entire 21-case cohort")
    index_path = Path(request["files"][0]).parent / "index.json"
    indexed = json.loads(index_path.read_text(encoding="utf-8"))["entries"]
    validation = sorted((e for e in indexed if e["split"] == "val"), key=lambda e:e["path"])
    if validation != request["entries"] or {e["case_id"] for e in validation} != set(cohort):
        raise ValueError("Internal request omits/changes indexed validation samples")
    if [str(index_path.parent / e["path"]) for e in validation] != request["files"]:
        raise ValueError("Internal request substitutes validation files")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One selected CUDA device is required; no CPU fallback")
    cfg, runtime = request["config"], request["config"]["runtime"]
    set_seed(42, deterministic=runtime["deterministic"])
    configure_runtime(deterministic=runtime["deterministic"], allow_tf32=runtime["allow_tf32"],
                      cudnn_benchmark=runtime["cudnn_benchmark"])
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(request["cuda_gib"] * 2**30)
    if not 0 < budget < total:
        raise ValueError("Explicit CUDA budget must leave device headroom")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    process = psutil.Process()
    def resources():
        rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
        if rss > int(request["rss_gib"] * 2**30):
            raise MemoryError("Explicit evaluator parent+worker RSS budget exceeded")
        return rss
    protected = request["protected_sha256"]
    def unchanged():
        for path, expected in protected.items():
            if _digest(path) != expected:
                raise ValueError(f"Evaluation input/checkpoint changed: {path}")
    unchanged()
    payload = load_checkpoint(request["checkpoint"], torch.device("cpu"))
    if (payload.get("training_complete") is not True or payload.get("target_epochs") != 40
            or payload.get("completed_epoch") != 40
            or payload.get("training_signature", {}).get("run_mode") != "production"
            or payload.get("training_signature", {}).get("val_cache_files") != [e["path"] for e in validation]):
        raise ValueError("Internal evaluator cannot use a partial/DEBUG or different-cohort checkpoint")
    net = HierarchicalPyGPlacementModel(**payload["model_kwargs"]).cuda()
    net.load_state_dict(payload["state_dict"], strict=True)
    del payload
    net.eval()
    ds = HierarchicalCacheDataset(request["files"], mmap=True, training=False, seed=42)
    ds.set_epoch(29)  # Curriculum epoch29, while original validation local views use effective epoch0.
    training = cfg["training"]
    workers = request["workers"]
    options = dict(num_workers=workers, pin_memory=training["pin_memory"])
    if workers:
        options.update(persistent_workers=training["persistent_workers"], prefetch_factor=training["prefetch_factor"])
    generator = torch.Generator().manual_seed(42 + 4001)
    loader = DataLoader(ds, batch_size=request["physical_batch"], shuffle=False, drop_last=False,
                        generator=generator, collate_fn=collate_samples, **options)
    if training["cuda_prefetch"]:
        loader = CudaPrefetchLoader(loader, torch.device("cuda:0"))
    rows, offset, peak_rss = [], 0, resources()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"FULL 21-case {request['mode']}", unit="batch"):
            peak_rss = max(peak_rss, resources())
            count = batch.sample_count
            expected = request["entries"][offset:offset+count]
            if list(batch.case_ids) != [e["case_id"] for e in expected] or any(c != 8 for c in batch.counts):
                raise ValueError("Original validation order/candidate contract changed")
            if batch.local_batch_view2 is None:
                raise ValueError("Original two-view validation is required")
            batch = batch.to("cuda:0", non_blocking=training["pin_memory"])
            with torch.autocast("cuda", enabled=training["amp"]):
                output = net(batch)
            scores = torch.stack([score.detach() for score in output.scores])
            values = score_metrics(scores)
            top1, rr, n = ranking_metric_sums(output.scores)
            if not (torch.equal(top1, values[:, 1].contiguous().sum()) and torch.equal(rr, values[:, 0].contiguous().sum())
                    and int(n) == count):
                raise AssertionError("Metric algebra disagrees with original ranking_metric_sums")
            for entry, metrics in zip(expected, values.cpu().tolist()):
                rows.append(dict(case_id=entry["case_id"], sample_id=entry["path"], **dict(zip(METRICS, metrics))))
            offset += count
            del output, batch, scores, values, top1, rr, n
    torch.cuda.synchronize()
    if offset != len(request["files"]):
        raise ValueError("Validation evaluator did not consume every indexed sample")
    unchanged()
    report = dict(format="hiercp_v1x_full_case_evaluation_v1", mode=request["mode"], debug=False,
                  measured_new_evaluation=True, validation_cases=21, validation_samples=offset,
                  per_case=aggregate_cases(rows, request["cohort"]), per_sample=rows,
                  physical_batch=request["physical_batch"], workers=workers, seed=42,
                  curriculum_epoch=29, effective_local_view_epoch=0, checkpoint=request["checkpoint"],
                  checkpoint_sha256=protected[request["checkpoint"]], sampling_contract=request["sampling_contract"],
                  parameters=sum(p.numel() for p in net.parameters()), gpu=torch.cuda.get_device_name(0),
                  precision="CUDA autocast" if training["amp"] else "CUDA float32",
                  evaluation_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_parent_and_worker_rss_bytes=peak_rss, checkpoint_bytes_preserved=True,
                  training_started=False, optimizer_created=False, nnunet_started=False,
                  graph_quality_passed=False, automatic_promotion=False)
    _write(request["report"], report)


def evaluate(args):
    if args.bootstrap_resamples < 2 or not 0 < args.confidence < 1:
        raise ValueError("Explicit bootstrap resamples >=2 and confidence in (0,1) are required")
    if args.cuda_gib <= 0 or args.rss_gib <= 0:
        raise ValueError("Positive explicit GPU/RSS budgets are required")
    sys.path.insert(0, str(ROOT))
    from tools.local_cnn_device import select
    from hiercp_v1x.contracts import compare_reports, resolve_execution_config
    from hiercp_v1x.experiment import load_suite, preparation_root, digest, read
    from hiercp_v1x.results import collect_result
    select(args.gpu)
    native, nested = (Path(p).resolve(strict=True) for p in (args.native_experiment, args.nested_experiment))
    # These guards run before any forward. Partial/debug/mismatched results cannot be promoted by this tool.
    completed = [collect_result(path, "v1.0") for path in (native, nested)]
    comparison = compare_reports(*completed)
    if any(report.get("native_reference_experiment") != str(native) for report in completed):
        raise ValueError("Both arms must use this exact explicit native reference")
    manifests = [load_suite(path) for path in (native, nested)]
    shared = preparation_root(native, manifests[0])
    if preparation_root(nested, manifests[1]) != shared:
        raise ValueError("Paired arms do not share canonical preparation")
    cohort = manifests[0]["split"]["val"]
    entries = sorted((e for e in read(shared / "cache/index.json")["entries"] if e["split"] == "val"), key=lambda e:e["path"])
    if len(cohort) != 21 or {e["case_id"] for e in entries} != set(cohort):
        raise ValueError("Requires the entire 21-case indexed validation cohort; no subset/skip")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = read(shared / "execution_lock.json")
    protected = {str(shared / "cache" / e["path"]): e["artifact_sha256"] for e in entries}
    protected.update({str(shared / name): digest(shared / name) for name in
                      ("cache/config.json", "cache/index.json", "cache/complete.json", "prototype_bank.pt")})
    reports = []
    for path, manifest, result, mode in zip((native, nested), manifests, completed, ("native", "strict_nested")):
        cfg = resolve_execution_config(read(path / "configs/v1.0.json"), lock)
        if cfg["training"]["fixed_validation_epoch"] != 29 or cfg["seed"] != 42:
            raise ValueError("Original seed42/fixed-validation29 contract changed")
        best = result["checkpoints"]["best"]
        contract_path = path / "sampling/v1.0.json"
        request = dict(source=str(path / "source/v1.0"), config=cfg, mode=mode, cohort=cohort,
                       files=[str(shared / "cache" / e["path"]) for e in entries], entries=entries,
                       physical_batch=lock["selected_batch_size"], workers=lock["selected_num_workers"],
                       checkpoint=best["path"], sampling_contract=result["sampling_contract"],
                       cuda_gib=args.cuda_gib, rss_gib=args.rss_gib,
                       protected_sha256={**protected, best["path"]:best["sha256"], str(contract_path):digest(contract_path)},
                       report=str(output / f"{mode}.json"))
        request_path = output / f"{mode}_request.json"
        _write(request_path, request)
        env = dict(os.environ, HIERCP_V1X_SAMPLING_CONTRACT=str(contract_path), PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child-request", str(request_path)],
                                   env=env, cwd=request["source"])
        try:
            code = process.wait()
        except KeyboardInterrupt:
            print(f"Interrupted; waiting for directly launched evaluator PID {process.pid}; no session termination.", flush=True)
            process.wait()
            raise
        if code:
            raise RuntimeError(f"{mode} validation failed ({code}); outputs preserved, comparison not published")
        reports.append(read(request["report"]))
    summary = dict(format="hiercp_v1x_case_paired_sampling_evaluation_v1", completed_runs=completed,
                   matched_contract_comparison=comparison,
                   paired=paired_summary(*(r["per_case"] for r in reports), resamples=args.bootstrap_resamples,
                                         confidence=args.confidence, seed=args.bootstrap_seed),
                   arm_reports=[str(output / f"{mode}.json") for mode in ("native", "strict_nested")],
                   training_started=False, nnunet_started=False, graph_quality_passed=False, automatic_promotion=False)
    _write(output / "paired_summary.json", summary)
    print(f"REPORT: {output / 'paired_summary.json'} | paired case deltas only; no quality PASS", flush=True)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--child-request":
        if len(argv) != 2:
            raise ValueError("Internal child request requires exactly one frozen request")
        return child(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-experiment", "nested-experiment", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cuda-gib", type=float, required=True)
    parser.add_argument("--rss-gib", type=float, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Statistical resampling seed; training seed remains42")
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
