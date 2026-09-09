"""Measured, ordered full-candidate CPU graph preparation for online CP banks.

Calibration waves perform real pending work, once per candidate. They never
sample or shrink a graph, change a CP candidate, or replace GPU graph batching.
The existing preparation scheduler measures RSS/CPU/throughput and admits later
waves against the allocated CPU capacity and available-RAM headroom.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from hiercp.preparation_runtime import run_case_jobs


BANK_GRAPH_PREPARATION_FORMAT = "hiercp_bank_ordered_graph_preparation_v1"


def _local_counts(local: dict) -> tuple[int, int]:
    return (
        sum(int(node["x"].shape[0]) for node in local["nodes"].values()),
        sum(int(edges.shape[1]) for edges in local["edges"].values()),
    )


class BankLocalGraphMapper:
    """An ordered map over complete canonical targets, sharing a read-only source.

    ``workers='auto'`` uses full real-candidate waves of 1, 2, 4, ... workers,
    bounded by allocation and measured memory. Timing comparisons are across
    different pending candidates, not repeated identical-input benchmarks.
    Every result, including calibration work, is retained in its original slot.
    """

    def __init__(self, bank_root: str | Path, *, workers: int | str = "auto"):
        self.report_root = Path(bank_root) / "preparation_resources"
        self.workers = workers
        self.last_report: dict | None = None

    def __call__(self, build_one, specs, *, case_id: str, source_component: int):
        tasks = list(enumerate(specs))
        if not tasks:
            raise ValueError("Bank local graph preparation requires nonempty candidate specs")
        self.last_report = None
        case_token = re.sub(r"[^A-Za-z0-9_.-]", "_", str(case_id))
        prefix = f"candidate_graphs.{case_token}.{int(source_component)}.{uuid.uuid4().hex}"
        self.report_root.mkdir(parents=True, exist_ok=True)
        context = {
            "format": BANK_GRAPH_PREPARATION_FORMAT,
            "case_id": str(case_id),
            "source_component": int(source_component),
            "candidate_count": len(tasks),
            "measurement_scope": "full canonical target graphs; all actual pending candidates once",
            "shared_source": "read-only prepared canonical source; no second source construction",
            "output_order": "original candidate/spec index",
            "debug_subset_or_graph_reduction": False,
        }
        print("[BankGraphPreparation] " + json.dumps(context, sort_keys=True), flush=True)
        completed = {}

        def execute(task):
            index, spec = task
            return index, build_one(spec)

        def commit(result):
            index, graph = result
            if type(index) is not int or not 0 <= index < len(tasks) or index in completed:
                raise RuntimeError("Bank graph preparation returned a duplicate or invalid candidate index")
            completed[index] = graph

        resources = run_case_jobs(
            tasks=tasks, function=execute, commit=commit, workers=self.workers,
            report_path=self.report_root / f"{prefix}.resources.json",
        )
        if set(completed) != set(range(len(tasks))):
            raise RuntimeError("Bank graph preparation did not resolve every original candidate")
        ordered = [completed[index] for index in range(len(tasks))]
        source_nodes, source_edges = _local_counts(ordered[0].source_local)
        target_counts = [_local_counts(item.target_local) for item in ordered]
        report = {
            **context,
            "status": "complete",
            "completed_candidates": len(ordered),
            "source_nodes_shared": source_nodes,
            "source_edges_shared": source_edges,
            "target_nodes": [counts[0] for counts in target_counts],
            "target_edges_including_cross_relations": [counts[1] for counts in target_counts],
            "resources": resources,
        }
        report_path = self.report_root / f"{prefix}.summary.json"
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.last_report = report
        print(f"[BankGraphPreparationComplete] candidates={len(ordered)} report={report_path}", flush=True)
        return ordered
