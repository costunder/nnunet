"""Actual-cache CPU DEBUG comparison of D preparation execution costs.

Reads completed real-CT canonical records; never reconstructs raw CT geometry,
trains a model, publishes production readiness or overwrites original evidence.
The tested post-build stage is two genuine epoch0 view materializations plus
lossless serialization/compression. All input/output SHA and sampled N/E must
match before a cost comparison is reported.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


class RSS:
    def __init__(self, limit):
        import psutil
        self.process = psutil.Process()
        self.limit = limit
        self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
    def _watch(self):
        while not self.done.wait(.02):
            self.peak = max(self.peak, self.process.memory_info().rss)
    def check(self):
        current = self.process.memory_info().rss
        self.peak = max(self.peak, current)
        if current > self.limit:
            raise MemoryError("Actual preparation DEBUG RSS budget exceeded; no data reduction")
    def close(self):
        self.done.set()
        self.thread.join()


def old_memory_functions():
    """Execute only the two preserved accounting functions, not old pipeline."""
    result = subprocess.run(["git", "-c", "safe.directory=" + str(ROOT), "show",
                             "c32158ef1bb75c78a7b657d7c95da36d2f01cf33:hiercp_v1x/transition_v1_data.py"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    source = result.stdout
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_bytes")
    provider = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OriginalInputProvider")
    measurement = next(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name == "_memory_measurement")
    import numpy as np
    import psutil
    import torch
    from collections.abc import Mapping
    from hiercp_v1x import transition_v1_local as local
    namespace = dict(torch=torch, np=np, psutil=psutil, Mapping=Mapping, local=local)
    module = ast.Module(body=[function, measurement], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<preserved-c32158e-accounting-functions>", "exec"), namespace)
    return namespace["_memory_measurement"], hashlib.sha256(source.encode()).hexdigest()


def accounting_benchmark(records, repeats, rss):
    from hiercp_v1x import transition_v1_data as data
    old_measurement, old_source_sha = old_memory_functions()
    results = []
    for count in (128, 256, 537):
        selected = records[:count]
        if len(selected) != count:
            raise ValueError("All537 real records required for accounting growth measurement")
        legacy = SimpleNamespace(_records=OrderedDict((str(index), (value, 0)) for index, value in enumerate(selected)),
                                 _donors=OrderedDict(), _raw=None, resident_bytes=16 * 1024**3, rss_bytes=rss.limit)
        current = object.__new__(data.OriginalInputProvider)
        current._resident = data._ResidentStorageLedger()
        current._records = data._AccountedCache(current._resident, "record", lambda value: value[0])
        current._donors = data._AccountedCache(current._resident, "donor", lambda value: value[:2])
        current._raw = None
        current.resident_bytes, current.rss_bytes, current._cached_bytes = legacy.resident_bytes, legacy.rss_bytes, 0
        start = time.perf_counter()
        for index, value in enumerate(selected):
            current._records[str(index)] = value, 0
        registration_seconds = time.perf_counter() - start
        active = selected[:32]
        before = old_measurement(legacy, active)
        after = current._memory_measurement(active)
        keys = ("raw_resident_bytes", "canonical_record_resident_bytes", "prepared_donor_resident_bytes", "canonical_resident_bytes")
        mismatches = {key: dict(old=before[key], new=after[key]) for key in keys if before[key] != after[key]}
        if mismatches:
            raise ValueError("Actual tensor-only canonical physical byte accounting changed: " + str(mismatches))
        old_times, new_times = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            measured = old_measurement(legacy, active)
            old_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            renewed = current._memory_measurement(active)
            new_times.append(time.perf_counter() - start)
            if measured["canonical_resident_bytes"] != renewed["canonical_resident_bytes"]:
                raise ValueError("Actual repeated accounting byte parity failed")
        row = dict(actual_cached_observations=count, active_complete_records=len(active), repeats=repeats,
                   canonical_physical_bytes=after["canonical_resident_bytes"], physical_byte_parity=True,
                   ledger_registration_seconds=registration_seconds,
                   old_scan_seconds_mean=sum(old_times) / repeats,
                   new_incremental_report_seconds_mean=sum(new_times) / repeats,
                   improvement_ratio=(sum(old_times) / sum(new_times)))
        results.append(row)
        rss.check()
        print("ACTUAL CPU DEBUG accounting | " + json.dumps(row), flush=True)
    return dict(preserved_old_commit="c32158ef1bb75c78a7b657d7c95da36d2f01cf33",
                preserved_old_source_normalized_text_sha256=old_source_sha,
                scope="actual canonical tensors only; raw CT/prepared donor objects absent; per-report32active records",
                measurements=results)


def execute_stage(output, rows, records, *, parallel, workers, rss):
    from hiercp_v1x import transition_v1_data as data
    from hiercp_v1x import transition_v1_local as local
    from hiercp_v22 import storage
    from hiercp_v1x.transition_preparation_storage import GraphWriter
    output.mkdir()
    writer = GraphWriter(output, minimum_free_bytes=1) if parallel else storage.GraphWriter(output, minimum_free_bytes=1)
    def one(index):
        row, record = rows[index], records[index]
        start = time.perf_counter()
        payload = local.materialize_pair(record, epoch=0)
        counts = data._sampled_measurement(payload)
        del payload
        sampled_seconds = time.perf_counter() - start
        for key in ("sampled_view_nodes", "sampled_view_edges", "sampled_two_view_nodes", "sampled_two_view_edges"):
            if counts[key] != row[key]:
                raise ValueError("Actual original epoch0 sampled topology changed: " + key)
        start = time.perf_counter()
        stored = writer.write(f"graphs/{index:06d}.pt.gz", record,
                              f"{row['donor_case_id']}:{row['donor_component']}")
        written_seconds = time.perf_counter() - start
        return dict(observation_id=row["id"], actual_sampled_measurements=counts, stored=stored,
                    materialize_two_views_seconds=sampled_seconds, lossless_write_seconds=written_seconds)
    start = time.perf_counter()
    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, range(len(records))))
    else:
        results = []
        for index in range(len(records)):
            results.append(one(index))
            if (index + 1) % 8 == 0:
                print(f"ACTUAL CPU DEBUG serial stage | {index + 1}/{len(records)}", flush=True)
            rss.check()
    seconds = time.perf_counter() - start
    rss.check()
    return dict(complete=True, actual_observations=len(results), actual_sampled_graphs=2 * len(results),
                workers=workers if parallel else 1, wall_seconds=seconds,
                source_encode_calls=writer.source_encode_calls if parallel else len(results),
                source_memo_hits=writer.source_memo_hits if parallel else 0,
                shared_source_files=len(list((output / "shared_sources").glob("*.pt.gz"))),
                materialize_seconds_sum=sum(row["materialize_two_views_seconds"] for row in results),
                write_seconds_sum=sum(row["lossless_write_seconds"] for row in results), records=results)


def run(a):
    import psutil
    import torch
    if not a.debug or a.observations != 32 or a.workers != 16:
        raise ValueError("Explicit actual32-observation/parallel16 DEBUG measurement required")
    if a.output.exists():
        raise FileExistsError("Fresh DEBUG output required; existing measurements preserved")
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    from tools.run_v17_crossed_training import prepare_source
    proof, scope = prepare_source(a.baseline / "source/v1.0")
    from hiercp_v1x import transition_v1_local as local
    from hiercp_v1x import transition_v1_data as data
    from hiercp_v22 import storage
    index_before = sha(a.cache)
    meta = json.loads(a.cache.read_text(encoding="utf8"))
    if meta.get("debug") is not True or meta.get("complete") is not True or len(meta["records"]) != 537:
        raise ValueError("Existing complete real537-observation DEBUG preparation required")
    first_case = meta["records"][0]["case_id"]
    selected = [row for row in meta["records"] if row["case_id"] == first_case][:a.observations]
    if len(selected) != a.observations or len({(row["donor_case_id"], row["donor_component"]) for row in selected}) != 1:
        raise ValueError("Exactly32 same-recipient/same-donor existing observations required")
    rss = RSS(int(a.rss_gib * 1024**3))
    report = dict(debug=True, scope="actual existing canonical cache post-build execution and accounting only",
                  raw_CT_geometry_rebuilt=False, GPU_neural_run=False, training_started=False,
                  production_ready=False, quality_verified=False, original_inputs_preserved=False,
                  input_index=str(a.cache), input_index_sha256=index_before,
                  source_snapshot_proof=proof, scope_contract=scope,
                  resources=dict(cpu_logical=os.cpu_count(), available_RAM_bytes=psutil.virtual_memory().available,
                                 torch_version=torch.__version__, torch_CPU_threads=2,
                                 CPU_post_stage_workers=a.workers, rss_limit_bytes=rss.limit,
                                 GPU_used=False), selected_case=first_case,
                  actual_subset_observations=len(selected), total_existing_DEBUG_observations=len(meta["records"]))
    try:
        inputs = []
        records, records_by_id = [], {}
        started = time.perf_counter()
        for ordinal, row in enumerate(meta["records"]):
            root = (a.cache.parent / row.get("segment", ".")).resolve(strict=True)
            graph = (root / row["path"]).resolve(strict=True)
            source = (root / row["shared_source"]["path"]).resolve(strict=True)
            if (not graph.is_relative_to(a.cache.parent.resolve()) or not source.is_relative_to(a.cache.parent.resolve())
                    or sha(graph) != row["sha256"] or sha(source) != row["shared_source"]["sha256"]):
                raise ValueError("Preserved actual canonical input byte identity changed")
            record = storage.load_record(root, row["path"])
            inputs.append(dict(id=row["id"], graph=str(graph), graph_sha256=row["sha256"],
                               source=str(source), source_sha256=row["shared_source"]["sha256"]))
            records.append(record)
            records_by_id[row["id"]] = record
            if (ordinal + 1) % 128 == 0:
                rss.check()
                print(f"ACTUAL CPU DEBUG input checks | {ordinal + 1}/537", flush=True)
        report["verified_input_load_seconds"] = time.perf_counter() - started
        chosen = [records_by_id[row["id"]] for row in selected]
        # Warm only byte/metadata validation; neither timed arm inherits graph
        # construction or a prior sampled output. Both start with verified data.
        for record in chosen:
            local.validate_record(record)
        report["old_serial"] = execute_stage(a.output / "old_serial", selected, chosen,
                                             parallel=False, workers=1, rss=rss)
        print("ACTUAL CPU DEBUG old serial wall | " + str(report["old_serial"]["wall_seconds"]), flush=True)
        report["new_parallel16"] = execute_stage(a.output / "new_parallel16", selected, chosen,
                                                 parallel=True, workers=a.workers, rss=rss)
        pairs = []
        for before, after in zip(report["old_serial"]["records"], report["new_parallel16"]["records"]):
            if (before["observation_id"] != after["observation_id"]
                    or before["actual_sampled_measurements"] != after["actual_sampled_measurements"]
                    or before["stored"] != after["stored"]):
                raise ValueError("Actual post-build compressed file SHA/reference/bounds/topology parity failed")
            pairs.append(dict(observation_id=before["observation_id"],
                              compressed_graph_sha256=before["stored"]["sha256"],
                              compressed_shared_source_sha256=before["stored"]["shared_source"]["sha256"],
                              byte_exact_parity=True, sampled_N_E_parity=True))
        report["parity"] = dict(complete=True, actual_observations=len(pairs), records=pairs)
        report["post_stage_improvement_ratio"] = report["old_serial"]["wall_seconds"] / report["new_parallel16"]["wall_seconds"]
        report["memory_accounting"] = accounting_benchmark(records, a.accounting_repeats, rss)
        if sha(a.cache) != index_before:
            raise ValueError("Original input index changed during DEBUG")
        for item in inputs:
            if sha(item["graph"]) != item["graph_sha256"] or sha(item["source"]) != item["source_sha256"]:
                raise ValueError("Original canonical graph/source changed during DEBUG")
        report["original_inputs_preserved"] = True
        report["complete"] = True
        report["source_files"] = {name: sha(ROOT / name) for name in (
            "tools/benchmark_v17_preparation_execution.py", "hiercp_v1x/transition_v1_data.py",
            "hiercp_v1x/transition_preparation_storage.py", "hiercp_v1x/transition_v1_local.py",
            "hiercp_v22/storage.py")}
        report["limitations"] = ["No raw CT donor/recipient canonical graph construction was timed",
            "This does not estimate total14102 preparation, optimizer/epoch time or ranking quality",
            "First32 actual same-case observations are explicitly DEBUG; no production subset applied",
            "Parallel speed depends on hardware and contention; threaded per-row durations overlap",
            "Memory benchmark contains actual canonical records, no raw CT or prepared donor buffers"]
        report["resources"]["peak_process_RSS_bytes"] = rss.peak
        save_new(a.output / "report.json", report)
        summary = dict(report=str(a.output / "report.json"), complete=True,
                       old_serial_wall_seconds=report["old_serial"]["wall_seconds"],
                       new_parallel16_wall_seconds=report["new_parallel16"]["wall_seconds"],
                       post_stage_improvement_ratio=report["post_stage_improvement_ratio"],
                       source_encode_calls=dict(old=report["old_serial"]["source_encode_calls"], new=report["new_parallel16"]["source_encode_calls"]),
                       actual32_byte_SHA_and_sampled_N_E_parity=True,
                       original_inputs_preserved=True, training_started=False,
                       scope="CPU actual-cache post-build DEBUG only")
        save_new(a.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        rss.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--rss-gib", type=float, required=True)
    parser.add_argument("--accounting-repeats", type=int, required=True)
    args = parser.parse_args()
    if not args.rss_gib > 0 or args.accounting_repeats < 1:
        raise ValueError("Explicit positive RSS/repeat measurement budgets required")
    run(args)


if __name__ == "__main__":
    main()
