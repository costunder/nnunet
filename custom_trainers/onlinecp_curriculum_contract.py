"""Standalone fail-closed reader of a publisher-verified curriculum bank.

Installed beside the trainer; no dependency on the HierCP checkout. A contract
is provenance, not a signature or proof against maliciously rewritten inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "onlinecp_curriculum_bank_contract_v1"
ARCHITECTURE = "hiercp_conditioned_readout_v3"
GEOMETRY = "level0_physical_closure_v2"
STREAM_BYTES = 1024 * 1024  # I/O buffer bound, never a dataset/sample limit.


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(STREAM_BYTES), b""):
            digest.update(block)
        finished = os.fstat(handle.fileno())
    after = path.stat()
    # Windows path.stat/fstat timestamps can legitimately differ. Compare each
    # API before/after, and only inode/device/size across those two APIs.
    if (_signature(before) != _signature(after) or _signature(opened) != _signature(finished)
            or _signature(before)[:3] != _signature(opened)[:3]):
        raise ValueError(f"Provenance input changed while hashing: {path}")
    return digest.hexdigest()


def value_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(value, name):
    if (not isinstance(value, list) or not value
            or any(not isinstance(v, str) or not v.strip() for v in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Invalid or duplicate {name} patient IDs")
    for item in value:
        _component(item, f"{name} patient ID")
    return value


def _component(value, name):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or any(char in value for char in "/\\:\x00")):
        raise ValueError(f"Unsafe {name}: {value!r}")
    return value


def _npy_header(handle):
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported preprocessing NPY header version: {version}")
    if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
            or dtype.kind not in "biufc" or dtype.itemsize <= 0
            or not shape or any(type(dim) is not int or dim <= 0 for dim in shape)):
        raise ValueError("Unsupported nonnumeric, object or empty preprocessing NPY layout")
    return tuple(shape), bool(fortran), dtype, math.prod(shape) * dtype.itemsize


def _compare_unpacked_cache(stream, header, cache_path):
    """Compare every canonical element byte without materializing a volume."""
    shape, fortran, dtype, byte_count = header
    before = cache_path.stat()
    with cache_path.open("rb") as handle:
        cache_header = _npy_header(handle)
        offset = handle.tell()
    if cache_header != header or before.st_size != offset + byte_count:
        raise ValueError(f"Unpacked cache shape/dtype/order/length differs from canonical NPZ: {cache_path}")
    mapped = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    flat = None
    try:
        flat = mapped.reshape(-1, order="F" if fortran else "C")
        if not np.shares_memory(flat, mapped):
            raise ValueError(f"Unsupported non-contiguous cache layout: {cache_path}")
        octets = memoryview(flat).cast("B")
        try:
            position = 0
            while position < byte_count:
                count = min(STREAM_BYTES, byte_count - position)
                block = stream.read(count)
                if len(block) != count:
                    raise ValueError("Truncated canonical NPZ array payload")
                if block != octets[position:position + count].tobytes():
                    raise ValueError(f"Unpacked cache bytes differ from canonical NPZ: {cache_path}")
                position += count
            if stream.read(1):
                raise ValueError("Canonical NPZ member has trailing array bytes")
        finally:
            octets.release()
    finally:
        del flat
        mapped._mmap.close()
    if _signature(before) != _signature(cache_path.stat()):
        raise ValueError(f"Unpacked cache changed while verifying: {cache_path}")


def _verify_npz_caches(npz_path, data_root, case_id):
    before = npz_path.stat()
    with zipfile.ZipFile(npz_path, "r") as archive:
        if sorted(archive.namelist()) != ["data.npy", "seg.npy"]:
            raise ValueError(f"Canonical NPZ requires exactly data.npy and seg.npy: {npz_path}")
        headers = {}
        for member, suffix in (("data.npy", ".npy"), ("seg.npy", "_seg.npy")):
            with archive.open(member, "r") as stream:
                header = _npy_header(stream)
                if archive.getinfo(member).file_size != stream.tell() + header[3]:
                    raise ValueError(f"Canonical NPZ member length disagrees with its header: {member}")
                headers[member] = header
                cache_path = data_root / (case_id + suffix)
                if cache_path.exists():
                    _compare_unpacked_cache(stream, header, cache_path)
        data_shape, seg_shape = headers["data.npy"][0], headers["seg.npy"][0]
        if (len(data_shape) != 4 or len(seg_shape) != 4 or seg_shape[0] != 1
                or data_shape[1:] != seg_shape[1:]):
            raise ValueError(f"Preprocessing data/segmentation channel shapes disagree: {npz_path}")
    if _signature(before) != _signature(npz_path.stat()):
        raise ValueError(f"Canonical NPZ changed while verifying caches: {npz_path}")


def _verify_live_preprocessing(contract, live_dataset, train, val, marker):
    files = contract["files"]
    for name in ("preprocessed_split", "preprocess_marker", "plans", "dataset"):
        recorded = Path(files[name]["path"])
        if recorded.resolve() != (live_dataset / recorded.name).resolve():
            raise ValueError(f"Live nnU-Net {name} is not the verified dataset file")
    nn_config = _json(files["nnunet_config"]["path"])
    configuration = _component(nn_config["dataset"]["configuration"], "nnU-Net configuration")
    if Path(files["plans"]["path"]).stem != nn_config["dataset"]["plans"]:
        raise ValueError("Verified plans filename and nnU-Net configuration disagree")
    plans = _json(files["plans"]["path"])
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Preprocessing marker lacks actual output inventories")
    identifier = _component(outputs.get("data_identifier"), "preprocessed data identifier")
    if plans["configurations"][configuration]["data_identifier"] != identifier:
        raise ValueError("Verified configuration and preprocessing data_identifier disagree")
    storage = outputs.get("storage_format")
    if storage not in {"npz", "blosc2"}:
        raise ValueError("Preprocessing marker requires an explicit supported storage_format")
    rows = outputs.get("cases")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Preprocessing marker lacks complete case output hashes")
    row_ids = _ids([row.get("case_id") for row in rows], "preprocessed")
    if set(row_ids) != set(train) | set(val):
        raise ValueError("Preprocessing output cohort is not exactly training plus validation")
    data_root = live_dataset / identifier
    suffixes = (".npz", ".pkl") if storage == "npz" else (".b2nd", "_seg.b2nd", ".pkl")
    required = {case_id + suffix for case_id in row_ids for suffix in suffixes}
    optional = ({case_id + suffix for case_id in row_ids for suffix in (".npy", "_seg.npy")}
                if storage == "npz" else set())
    if len(required) != len(row_ids) * len(suffixes) or (optional and len(optional) != 2 * len(row_ids)):
        raise ValueError("Patient IDs create ambiguous preprocessing file names")
    present = {path.name for path in data_root.iterdir()}
    if not required.issubset(present) or not present.issubset(required | optional):
        raise ValueError("Live preprocessing inventory is missing, extra, or mixed-backend")
    if any(not (data_root / name).is_file() for name in present):
        raise ValueError("Live preprocessing inventory contains a non-file input")
    signatures = {name: _signature((data_root / name).stat()) for name in present}
    for row in rows:
        expected_keys = {"case_id", "data_sha256", "properties_sha256"}
        if storage == "blosc2":
            expected_keys.add("segmentation_sha256")
        if set(row) != expected_keys:
            raise ValueError("Preprocessing case record has missing or unexpected hashes")
        case_id = row["case_id"]
        paths = {"data_sha256": data_root / (case_id + (".npz" if storage == "npz" else ".b2nd")),
                 "properties_sha256": data_root / (case_id + ".pkl")}
        if storage == "blosc2":
            paths["segmentation_sha256"] = data_root / (case_id + "_seg.b2nd")
        for field, source in paths.items():
            if file_sha256(source) != row[field]:
                raise ValueError(f"Live preprocessing bytes changed: {case_id} {field}")
        if storage == "npz":
            _verify_npz_caches(paths["data_sha256"], data_root, case_id)
    if {path.name for path in data_root.iterdir()} != present:
        raise ValueError("Live preprocessing inventory changed during verification")
    if any(_signature((data_root / name).stat()) != signature
           for name, signature in signatures.items()):
        raise ValueError("Live preprocessing bytes changed during verification")
    return configuration


def verify_curriculum_bank_contract(bank_path, *, curriculum_sha256,
                                    expected_candidate_count, dataset_name, nnunet_fold,
                                    contract_filename="curriculum_contract.json"):
    index_path = Path(bank_path).resolve()
    if contract_filename not in {"curriculum_contract.json", "feedback_contract.json"}:
        raise ValueError("Unsupported bank contract filename")
    path = index_path.parent / contract_filename
    contract = _json(path)
    expected = {"format": FORMAT, "architecture_version": ARCHITECTURE,
                "geometry_contract": GEOMETRY, "dataset_name": dataset_name,
                "nnunet_fold": nnunet_fold, "curriculum_sha256": curriculum_sha256,
                "candidate_count": expected_candidate_count}
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"Curriculum bank contract mismatch: {name}")
    files = contract.get("files", {})
    required = {"index", "config", "manifest", "complete", "gnn_checkpoint",
                "gnn_split", "prototype", "graph_complete", "outer_splits",
                "preprocessed_split", "preprocess_marker", "plans", "dataset",
                "train_config", "nnunet_config"}
    if set(files) != required:
        raise ValueError("Missing or unexpected curriculum provenance files")
    for name, record in files.items():
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Curriculum provenance file changed: {name}")
    if Path(files["index"]["path"]).resolve() != index_path:
        raise ValueError("Contract references a different bank index")
    train = _ids(contract.get("train_case_ids"), "training")
    val = _ids(contract.get("validation_case_ids"), "validation")
    gnn_train = _ids(contract.get("gnn_train_case_ids"), "GNN training")
    gnn_val = _ids(contract.get("gnn_validation_case_ids"), "GNN validation")
    proto = _ids(contract.get("prototype_training_case_ids"), "prototype fitting")
    if set(train) & set(val) or set(gnn_train) & set(gnn_val):
        raise ValueError("Patient leakage across training and held-out cohorts")
    if set(gnn_train) | set(gnn_val) != set(train) or set(proto) != set(gnn_train):
        raise ValueError("GNN/prototype cohorts violate the outer training boundary")
    gnn_split = _json(files["gnn_split"]["path"])
    if (gnn_split.get("train") != gnn_train or gnn_split.get("val") != gnn_val
            or gnn_split.get("outer_validation_excluded") != val):
        raise ValueError("GNN split and contract cohorts disagree")
    pre_root = os.environ.get("nnUNet_preprocessed")
    if not pre_root:
        raise ValueError("nnUNet_preprocessed is required for live split verification")
    live_dataset = Path(pre_root) / _component(dataset_name, "dataset name")
    live_split = live_dataset / "splits_final.json"
    if live_split.resolve() != Path(files["preprocessed_split"]["path"]).resolve():
        raise ValueError("Live nnU-Net split is not the verified preprocessing split")
    splits = _json(live_split)
    if not isinstance(nnunet_fold, int) or isinstance(nnunet_fold, bool) or not 0 <= nnunet_fold < len(splits):
        raise ValueError("Invalid nnU-Net fold")
    if splits[nnunet_fold] != {"train": train, "val": val}:
        raise ValueError("Live nnU-Net split differs from the bank training cohort")
    marker = _json(files["preprocess_marker"]["path"])
    if marker.get("input_contract", {}).get("planning_cohort") != "outer_train_only_v1":
        raise ValueError("Fingerprint/plans were not fitted to outer training patients only")
    configuration = _verify_live_preprocessing(contract, live_dataset, train, val, marker)
    index = _json(index_path)
    if index.get("dataset_name") != dataset_name or index.get("candidate_count") != expected_candidate_count:
        raise ValueError("Bank dataset/candidate count mismatch")
    entries = index.get("entries_by_case", {})
    if not entries or not set(entries).issubset(set(train)):
        raise ValueError("CP donor/recipient is outside the outer training patients")
    names = [name for values in entries.values() for name in values]
    if len(names) != len(set(names)) or set(names) != set(contract.get("entry_sha256", {})):
        raise ValueError("Bank entry inventory differs from the verified contract")
    for name in names:
        entry = (index_path.parent / name).resolve()
        if not entry.is_relative_to(index_path.parent) or file_sha256(entry) != contract["entry_sha256"][name]:
            raise ValueError(f"Bank entry escaped root or changed: {name}")
    return {**contract, "contract_sha256": value_sha256(contract),
            "verified_configuration": configuration}
