"""Read-only source-cost ordering for the v1.8 CUDA batch calibration.

This wrapper is for calibration only. Actual training keeps the original
provider and its original example order. It changes neither candidate keys nor
graphs, labels, views, sampling, or optimizer work. Original canonical storage
is a stress proxy for new native-U targets, not a future-cohort memory proof.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .u_bridge_data import _resident_size


FORMAT = "v18_signed_source_calibration_proxy_v1"
SOURCE_COPIES = 8 * 2  # Eight queries and two genuine original local views.


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: explicit integer >= {minimum} required")
    return value


def _source_inventory(local):
    if not isinstance(local, dict) or local.get("format") != "canonical-full-v22":
        raise ValueError("Calibration ordering requires original canonical source tables")
    nodes, edges = local.get("nodes"), local.get("edges")
    declared_nodes, declared_edges = local.get("counts"), local.get("edge_counts")
    if (not isinstance(nodes, dict) or not isinstance(edges, dict)
            or not isinstance(declared_nodes, dict) or not isinstance(declared_edges, dict)
            or set(nodes) != set(declared_nodes) or set(edges) != set(declared_edges)):
        raise ValueError("Original source node/edge metadata coverage differs")
    node_counts, edge_counts = {}, {}
    for role, table in nodes.items():
        count = _integer(declared_nodes[role], str(role) + " node count")
        if not isinstance(table, dict) or not torch.is_tensor(table.get("x")):
            raise ValueError("Original source node feature table required")
        if table["x"].ndim != 2 or table["x"].shape[0] != count:
            raise ValueError("Original source node count differs from tensor shape")
        node_counts[str(role)] = count
    for role, index in edges.items():
        count = _integer(declared_edges[role], str(role) + " edge count")
        if not torch.is_tensor(index) or index.ndim != 2 or tuple(index.shape) != (2, count):
            raise ValueError("Original source edge count differs from tensor shape")
        name = "|".join(role) if isinstance(role, tuple) else str(role)
        if name in edge_counts:
            raise ValueError("Original source relation names are ambiguous")
        edge_counts[name] = count
    return dict(source_nodes=sum(node_counts.values()), source_edges=sum(edge_counts.values()),
                source_node_type_counts=node_counts, source_relation_edge_counts=edge_counts,
                source_footprint_voxels=_integer(local.get("footprint_voxels"),
                                                "source footprint", minimum=1))


def _read_source(entry, examples):
    """Use the base provider's existing SHA verification, without another hash scan."""
    if isinstance(entry, dict) and "candidate_centers" in entry:
        return entry, None, None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), (str, Path)):
        raise ValueError("Signed {path, sha256} entries or original canonical mappings required")
    expected = entry.get("sha256")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("Source entry needs the already-verified SHA256 receipt")
    candidate = Path(entry["path"])
    if candidate.is_symlink():
        raise ValueError("Calibration source cannot be a symlink")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Calibration source must remain a regular original file")
    # The provider has already read and verified this digest. Require that
    # receipt before loading even the mmap metadata; no second file hash pass.
    if expected not in {row.get("original_sample_sha256") for row in examples.values()}:
        raise ValueError("Signed source was not verified by the original provider")
    sample = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(sample, dict):
        raise ValueError("Original canonical source mapping required")
    return sample, expected, str(path)


class CalibrationProvider:
    """Rank original train IDs by canonical cost; forward all data APIs exactly.

    ``stress_receipts`` returns detached copies of internally immutable JSON
    receipts. Ranking uses source storage duplicated sixteen times plus whole
    original canonical storage. The combined value is a conservative ordering
    proxy, not an activation bound or a guarantee for new native-U geometry.
    """

    def __init__(self, base, source_samples):
        self._base = base
        train = list(base.examples("train"))
        val = list(base.examples("val"))
        original = train + val
        examples = {}
        indices = set()
        for row in original:
            identity = (row["case_id"], _integer(row["sample_index"], "sample index"))
            index = _integer(row["index"], "source index")
            if identity in examples or index in indices:
                raise ValueError("Base provider source identities and indices must be unique")
            examples[identity] = row
            indices.add(index)
        if not train:
            raise ValueError("Complete original training source population required")
        seen, receipts = set(), {}
        train_indices = {row["index"] for row in train}
        for entry in source_samples:
            sample, source_sha, path = _read_source(entry, examples)
            identity = (sample["case_id"], _integer(sample["sample_index"], "sample index"))
            if identity not in examples or identity in seen:
                raise ValueError("Source inventory differs from original provider coverage")
            row = examples[identity]
            if source_sha != row.get("original_sample_sha256"):
                raise ValueError("Source SHA256 receipt belongs to another original example")
            centers = sample["candidate_centers"]
            positive = tuple(int(x) for x in centers[0])
            if (len(centers) != 8 or positive != tuple(row["positive_center"])
                    or sample["source_component"] != row["source_component"]):
                raise ValueError("Calibration source component/anchor/query count differs")
            seen.add(identity)
            if row["index"] not in train_indices:
                continue
            local = sample["source_local"]
            stats = _source_inventory(local)
            source_bytes = _resident_size(local)
            canonical_bytes = _resident_size(sample)
            if min(source_bytes, canonical_bytes) <= 0:
                raise ValueError("Original source has no measurable CPU canonical storage")
            repeated = SOURCE_COPIES * source_bytes
            receipts[row["index"]] = dict(format=FORMAT, source_index=row["index"],
                source_id=row["id"], case_id=row["case_id"], sample_index=row["sample_index"],
                source_component=row["source_component"], positive_center=list(positive),
                original_sample_sha256=source_sha, original_sample_path=path,
                original_source_storage_bytes=source_bytes,
                original_canonical_storage_bytes=canonical_bytes,
                source_copies_in_two_view_eight_query_batch=SOURCE_COPIES,
                repeated_source_storage_proxy_bytes=repeated,
                stress_ordering_proxy_bytes=repeated + canonical_bytes,
                proxy_scope="original source and selected canonical tables; new native-U targets unmeasured",
                future_cohort_worst_case_verified=False, samples_or_candidates_changed=False,
                **stats)
        if seen != set(examples):
            raise ValueError("Original source inventory is missing provider examples")
        self._train = tuple(copy.deepcopy(row) for row in sorted(train, key=lambda row: (
            -receipts[row["index"]]["stress_ordering_proxy_bytes"],
            -receipts[row["index"]]["repeated_source_storage_proxy_bytes"], row["index"])))
        self._receipts = tuple(json.dumps(receipts[row["index"]], sort_keys=True, allow_nan=False)
                               for row in self._train)

    @property
    def stress_receipts(self):
        return tuple(json.loads(receipt) for receipt in self._receipts)

    def examples(self, partition):
        if partition in ("train", "inner_train"):
            return copy.deepcopy(list(self._train))
        return self._base.examples(partition)

    def batch(self, *args, **kwargs):
        return self._base.batch(*args, **kwargs)

    def candidate_keys(self, *args, **kwargs):
        return self._base.candidate_keys(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)
