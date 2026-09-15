"""Read-only, ordered Basic CP input equivalence; never checkpoint admission.

Both arguments name an index.json (a bank directory is also accepted). The
caller must authenticate the original identities independently. In particular,
this module never tries a current-model guard and then accepts an old model.
Cold verification audits all typed raw payloads. Warm calls reuse only native
stat/SHA witnesses; no tensors, wall times, or cache counters enter the receipt.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from custom_trainers.onlinecp_raw_bank import PASTE_CONTRACT, RawBankStore, _relative_path

FORMAT = "basic_bank_semantic_equivalence_v1"
PROOF_FORMAT = "basic_bank_integrity_binding_v1"
PROOF_KEYS = {"format", "index_sha256", "config_sha256", "feedback_contract_sha256",
              "original_identity_sha256"}
# Only these *producer* dependencies are absent from Basic's uniform selection.
# The preprocess bindings are not declared GNN-only: independent full native
# preprocessing equivalence is an explicit prerequisite outside this sub-proof.
GNN_ONLY = frozenset({"checkpoint_sha256", "prototype_sha256", "gnn_split_sha256",
    "gnn_training_cases_sha256", "gnn_causality_sha256", "gnn_causality_preflight_sha256",
    "gnn_graph_complete_sha256", "hier_top_k"})
EXTERNAL_PREPROCESS = frozenset({"train_config_sha256", "preprocess_marker_sha256",
                               "preprocess_contract_sha256"})
REQUIRED_EXTERNAL_CHECKS = (
    "original_bank_checkpoint_history_arm_and_runtime_authenticity",
    "full_native_preprocessing_images_segmentations_properties_splits_and_raw_case_hashes_equal_before_full",
    "ordered_train_validation_cohort_and_split_equal",
    "basic_loader_selection_raw_paste_native_augmentation_implementation_equal_or_independently_proven",
    "online_seed_epoch_resume_progress_worker_rng_initialization_assignment_and_count_equal",
    "physical_batch_gradient_accumulation_iterations_epochs_dataloader_order_and_epoch_warmup_equal",
    "native_plans_dataset_configuration_normalization_numpy_torch_and_transform_runtime_equal",
    "original_basic_evaluation_valid_and_full_uses_fresh_current_gnn_not_old_learned_artifacts",
)
_CACHE: dict[tuple, dict] = {}
_NPY_PROOFS: dict[tuple, dict] = {}
_LOG = logging.getLogger(__name__)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path):
    def bad_constant(value):
        raise ValueError(f"Non-finite JSON constant in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=bad_constant)


def _state(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _digest(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA-256 at {field}")
    return value


def _bind(path: Path, proof: Mapping) -> tuple[Path, dict]:
    path = Path(path)
    index = (path / "index.json" if path.is_dir() else path).resolve(strict=True)
    if index.name != "index.json" or not index.is_file():
        raise ValueError("A Basic bank must name its index.json")
    if not isinstance(proof, Mapping) or set(proof) != PROOF_KEYS or proof.get("format") != PROOF_FORMAT:
        raise ValueError(f"Expected exact {PROOF_FORMAT} proof fields")
    proof = dict(proof)
    for field in PROOF_KEYS - {"format"}:
        _digest(proof[field], field)
    for filename, field in (("index.json", "index_sha256"), ("config.json", "config_sha256"),
                            ("feedback_contract.json", "feedback_contract_sha256")):
        actual = _relative_path(index.parent, filename)
        if _sha(actual) != proof[field]:
            raise ValueError(f"Basic bank integrity differs: {filename}/{field}")
        _json(actual)
    return index, proof


def _array_bytes(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    # Bounded copying, including non-contiguous arrays; never materialize a
    # complete patient volume merely to compute its logical-content digest.
    iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C", buffersize=131072)
    for chunk in iterator:
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


class _SemanticStore(RawBankStore):
    """Native verifier plus serialization-independent typed payload identities."""

    def __init__(self, root):
        super().__init__(root)
        self.array_semantics = {}
        self.cases_semantics = {}
        self.candidates_semantics = {}

    def _check(self, relative, digest):
        path = _relative_path(self.root, relative)
        key = (str(relative), digest)
        # Preserve the native reject-on-changed-witness policy verbatim.
        if key in self._witnesses or path.suffix != ".npy":
            return super()._check(relative, digest)
        _digest(digest, str(relative))
        before = _state(path)
        stat = path.stat()
        physical = (stat.st_dev, stat.st_ino, *before, digest)
        # An unavailable inode is not a safe alias-sharing identity.
        if not stat.st_ino:
            physical = (str(path.resolve()), *physical)
        semantic = _NPY_PROOFS.get(physical)
        if semantic is None:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if array.dtype.hasobject:
                    raise ValueError(f"Object NPY payload: {relative}")
                offset, length = int(array.offset), int(array.nbytes)
                # Storage-order identity is conservative for non-C NPYs. A
                # C/F reserialization is explicitly not certified by v1.
                order = "C" if array.flags.c_contiguous else "F"
                semantic = {"dtype": array.dtype.str, "shape": list(array.shape), "order": order}
                full, data = hashlib.sha256(), hashlib.sha256()
                position = 0
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        full.update(chunk)
                        start, stop = max(0, offset - position), min(len(chunk), offset + length - position)
                        if stop > start:
                            data.update(chunk[start:stop])
                        position += len(chunk)
                if full.hexdigest() != digest:
                    raise ValueError(f"Raw payload SHA-256 mismatch: {relative}")
                semantic["data_sha256"] = data.hexdigest()
            finally:
                array._mmap.close()
            if _state(path) != before:
                raise ValueError(f"Raw payload changed during verification: {relative}")
            _NPY_PROOFS[physical] = semantic
        if _state(path) != before:
            raise ValueError(f"Raw payload changed during verification: {relative}")
        self.array_semantics[str(path.resolve())] = semantic
        self._witnesses[key] = before
        return super()._check(relative, digest)

    def canonical(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("Object arrays are not Basic payloads")
            if isinstance(value, np.memmap):
                semantic = self.array_semantics.get(str(Path(value.filename).resolve()))
                if semantic is not None and semantic["shape"] == list(value.shape) and semantic["dtype"] == value.dtype.str:
                    return {"array": dict(semantic)}
            return {"array": {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C",
                              "data_sha256": _array_bytes(value)}}
        if isinstance(value, np.generic):
            return self.canonical(value.item())
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Payload metadata keys must be strings")
            return {key: self.canonical(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [self.canonical(item) for item in value]
        if value is None or type(value) in {str, bool, int, float}:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite scalar payload metadata")
            return value
        raise ValueError(f"Unsupported Basic payload value: {type(value).__name__}")

    def load_case(self, relative, digest):
        self._check(relative, digest)
        _json(_relative_path(self.root, relative))
        value = super().load_case(relative, digest)
        if digest not in self.cases_semantics:
            self.cases_semantics[digest] = self.canonical(value)
        return value

    def load_candidate(self, relative, digest):
        self._check(relative, digest)
        _json(_relative_path(self.root, relative))
        value = super().load_candidate(relative, digest)
        if digest not in self.candidates_semantics:
            case_digest = value.get("case_reference_sha256")
            if case_digest not in self.cases_semantics:
                raise ValueError("Candidate's raw-case identity was not verified first")
            semantic = dict(value)
            semantic["case_reference_sha256"] = _json_sha(self.cases_semantics[case_digest])
            self.candidates_semantics[digest] = self.canonical(semantic)
        return value


def _difference(left, right, path="basic"):
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys differ {sorted(set(left) ^ set(right))}"
        for key in sorted(left):
            result = _difference(left[key], right[key], f"{path}.{key}")
            if result:
                return result
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            result = _difference(a, b, f"{path}[{index}]")
            if result:
                return result
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _implementation():
    root = Path(__file__).resolve().parents[1]
    names = ("tools/basic_bank_equivalence.py", "tools/online_cp_benchmark.py",
             "custom_trainers/onlinecp_raw_bank.py", "custom_trainers/onlinecp_raw_resampling.py",
             "custom_trainers/nnUNetTrainer_OnlinePairedCP.py", "custom_trainers/nnUNetTrainer_OnlineCPFeedback.py",
             "custom_trainers/onlinecp_feedback_policy.py")
    return {name: _sha(root / name) for name in names}


def _entry_inventory(root: Path, expected: set[str]):
    entries = root / "entries"
    if not entries.is_dir() or entries.is_symlink():
        raise ValueError("Invalid bank entries directory")
    actual = set()
    for path in entries.rglob("*"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Non-regular/nested bank entry artifact: {path}")
        actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(f"Bank entry file inventory differs: {sorted(actual ^ expected)}")


def _metadata_integrity(root, metadata, config, rows, bank):
    from tools.online_cp_benchmark import (_source_slot_metadata, _validate_no_placement_row,
                                           natural_key, value_sha256)
    inventory = metadata.get("eligible_sources_by_case")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Missing full eligible-source inventory")
    for case_id, components in inventory.items():
        if (not isinstance(case_id, str) or not case_id or not isinstance(components, list)
                or any(type(value) is not int or value < 1 for value in components)
                or len(set(components)) != len(components)):
            raise ValueError(f"Malformed eligible-source inventory: {case_id}")
    extras = {"scoring_execution", "entries_by_case", "eligible_sources_by_case", "no_eligible_cases",
              "eligible_cases", "source_entries", "total_candidates", "manifest", "source_schedule_format",
              "source_slots_by_case", "eligible_source_slots", "no_placement_sources"}
    if set(metadata) - set(config) - extras:
        raise ValueError(f"Unknown index-only dependency: {sorted(set(metadata) - set(config) - extras)}")
    expected_keys = {(case_id, component) for case_id, components in inventory.items()
                     for component in (components if components else [-1])}
    row_map, manifest = {}, {}
    for row in rows:
        key = (row.get("case_id"), int(row["source_component"]))
        if key in row_map:
            raise ValueError(f"Duplicate manifest source: {key}")
        row_map[key] = row
        if key[1] == -1:
            if row.get("status") != "no_eligible_source" or row.get("entry") or int(row["candidate_count"]) != 0:
                raise ValueError(f"Invalid no-eligible-source sentinel: {key}")
        elif row.get("status") == "no_placement":
            if metadata.get("no_placement_policy") != "retain_original":
                raise ValueError("No-placement source requires explicit retain_original policy")
            json.loads(row.get("candidate_search_json", ""), object_pairs_hook=_pairs)
            _validate_no_placement_row(row, 128)
        elif row.get("status") == "ok" and row.get("entry") and int(row["candidate_count"]) == 128:
            if row["entry"] in manifest:
                raise ValueError("Duplicate manifest entry")
            manifest[row["entry"]] = row
        else:
            raise ValueError(f"Unsupported/incomplete manifest source status: {key}")
    if set(row_map) != expected_keys:
        raise ValueError(f"Full manifest/source inventory differs: {sorted(set(row_map) ^ expected_keys)}")
    expected_entries = {name for names in bank.entries_by_case.values() for name in names}
    if len(expected_entries) != sum(map(len, bank.entries_by_case.values())) or set(manifest) != expected_entries:
        raise ValueError("Manifest/index entry inventory differs or duplicates entries")
    for case_id, names in bank.entries_by_case.items():
        if not names or any(manifest[name]["case_id"] != case_id for name in names):
            raise ValueError(f"Index/source ownership differs: {case_id}")
    _entry_inventory(root, expected_entries)
    no_eligible = sorted([key for key, value in inventory.items() if not value], key=natural_key)
    if metadata.get("no_eligible_cases") != no_eligible:
        raise ValueError("No-eligible patient inventory differs")
    counts = {"eligible_cases": sum(bool(value) for value in inventory.values()),
              "source_entries": len(manifest), "total_candidates": len(manifest) * 128}
    if any(type(metadata.get(key)) is not int or metadata[key] != value for key, value in counts.items()):
        raise ValueError("Bank aggregate source inventory counts differ")
    if metadata.get("no_placement_policy", "error") == "retain_original":
        expected_slots = _source_slot_metadata(rows, inventory, 128)
        if any(metadata.get(key) != value for key, value in expected_slots.items()):
            raise ValueError("No-placement/source-slot schedule differs from complete manifest")
    elif any(key in metadata for key in ("source_schedule_format", "source_slots_by_case",
                                        "eligible_source_slots", "no_placement_sources")):
        raise ValueError("Source slots require explicit retain_original policy")
    wanted_complete = {"format": metadata["format"], "index_sha256": _sha(root / "index.json"),
        "manifest_sha256": _sha(root / "manifest.csv"), "config_sha256": _sha(root / "config.json"),
        "eligible_inventory_sha256": value_sha256(inventory), "eligible_cases": counts["eligible_cases"],
        "source_entries": counts["source_entries"], "candidate_count": 128}
    if _json(root / "complete.json") != wanted_complete:
        raise ValueError("Bank complete.json does not bind current index/config/manifest/full inventory")
    return manifest, row_map, inventory, expected_entries


def _audit(index: Path, proof: dict, implementation: dict):
    from custom_trainers.nnUNetTrainer_OnlinePairedCP import OnlineCPBank
    from tools.online_cp_benchmark import _audit_bank_entries, _validate_no_placement_row

    cache_key = (str(index), _json_sha(proof), _json_sha(implementation))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        _entry_inventory(index.parent, cached["entries"])
        for (relative, digest) in list(cached["store"]._witnesses):
            cached["store"]._check(relative, digest)
        _LOG.info("Basic bank audit warm native stat/SHA witnesses: %s", index)
        return cached
    config, metadata = _json(index.parent / "config.json"), _json(index)
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("Bank config/index must be JSON objects")
    for key, value in config.items():
        if key not in metadata or metadata[key] != value:
            raise ValueError(f"Index/config disagree at {key}")
    if metadata.get("paste_contract") != PASTE_CONTRACT or metadata.get("candidate_count") != 128:
        raise ValueError("Basic equivalence requires the typed raw-target contract and all 128 candidates")
    bank = OnlineCPBank(index)
    if not all(key in config for key in ("cp_probability", "intensity_scale_range", "intensity_shift_range_hu",
                                         "normalization", "network_patch_size", "tumor_label", "liver_label")):
        raise ValueError("Incomplete Basic CP/normalization configuration")
    with (index.parent / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Duplicate or absent manifest CSV headers")
        rows = list(reader)
    manifest, row_map, inventory, expected_entries = _metadata_integrity(index.parent, metadata, config, rows, bank)
    row_keys = set(row_map)
    store = _SemanticStore(index.parent)
    for filename in ("index.json", "config.json", "feedback_contract.json", "manifest.csv", "complete.json"):
        path = _relative_path(index.parent, filename)
        store._check(filename, _sha(path))
    for relative, row in manifest.items():
        store._check(relative, _digest(row["entry_sha256"], relative))
    try:
        _audit_bank_entries(index.parent, metadata["entries_by_case"], 128, manifest,
                            raw_store=store, expected_paste_contract=PASTE_CONTRACT)
        # Metadata consumer fields not repeated in config are still retained;
        # explicit schedules below normalize paths into audited logical content.
        cases = []
        schedule = bank.source_slots_by_case
        case_ids = list(inventory)
        used_rows = set()
        for case_id in case_ids:
            names = schedule[case_id] if schedule is not None else bank.entries_by_case.get(case_id, ())
            slots = []
            if not inventory[case_id]:
                used_rows.add((case_id, -1))
            for slot_index, name in enumerate(names):
                component = (metadata["source_slots_by_case"][case_id][slot_index]["source_component"]
                             if schedule is not None else int(manifest[name]["source_component"]))
                used_rows.add((case_id, component))
                if not name:
                    row = row_map.get((case_id, component))
                    if row is None or row.get("status") != "no_placement":
                        raise ValueError(f"Unproven no-placement source slot: {case_id}/{component}")
                    slots.append({"source_component": component, "status": "no_placement",
                                  "no_placement_proof": _validate_no_placement_row(row, 128)})
                    continue
                entry = bank._load(name)
                case_digest = str(entry["raw_case_reference_sha256"][0])
                excluded = {"scores", "raw_case_reference", "raw_case_reference_sha256",
                            "candidate_payloads", "candidate_payload_sha256"}
                slots.append({"source_component": component, "status": "ok",
                    "entry": store.canonical({key: value for key, value in entry.items() if key not in excluded}),
                    "raw_case": store.cases_semantics[case_digest],
                    "candidates": [store.candidates_semantics[str(value)] for value in entry["candidate_payload_sha256"]]})
            cases.append({"case_id": case_id, "source_slots": slots})
        if used_rows != row_keys:
            raise ValueError(f"Manifest sources outside ordered Basic schedule: {sorted(row_keys ^ used_rows)}")
        retained = {key: value for key, value in config.items() if key not in GNN_ONLY | EXTERNAL_PREPROCESS}
        result = {"manifest": {"config": retained, "cases": cases}, "store": store, "entries": expected_entries,
                  "excluded": {**{key: config[key] for key in sorted(GNN_ONLY | EXTERNAL_PREPROCESS) if key in config},
                               "scoring_execution": metadata.get("scoring_execution"),
                               "manifest_publication_path": metadata.get("manifest")}}
        # Close mmap descriptors now; only immutable digests and native witnesses
        # survive. Revalidate after the audit to reject concurrent mutation.
        for (relative, digest) in list(store._witnesses):
            store._check(relative, digest)
        _CACHE[cache_key] = result
        _LOG.info("Basic bank audit cold full typed-content verification: %s", index)
        return result
    finally:
        store.close()


def compare_basic_banks(source_bank: Path, target_bank: Path, *, source_proof: Mapping,
                        target_proof: Mapping) -> dict:
    """Return a deterministic bank-only equality sub-proof, or a precise error.

    Source/target proof identities are independently authenticated by the caller.
    This function does not authorize a trainer, checkpoint, metric, preprocessing
    cache, or old GNN. Every ``required_external_checks`` condition remains due.
    """
    source, old = _bind(source_bank, source_proof)
    target, new = _bind(target_bank, target_proof)
    implementation = _implementation()
    left, right = _audit(source, old, implementation), _audit(target, new, implementation)
    difference = _difference(left["manifest"], right["manifest"])
    if difference:
        raise ValueError(f"Basic bank semantic mismatch: {difference}")
    # Bind originals again after reading every referenced payload.
    _bind(source, old)
    _bind(target, new)
    manifest = left["manifest"]
    ordered = []
    for case in manifest["cases"]:
        slots = []
        for slot in case["source_slots"]:
            record = {"source_component": slot["source_component"], "status": slot["status"]}
            if slot["status"] == "ok":
                record.update(entry_sha256=_json_sha(slot["entry"]), raw_case_sha256=_json_sha(slot["raw_case"]),
                              candidate_sha256=[_json_sha(candidate) for candidate in slot["candidates"]])
            else:
                record["no_placement_proof_sha256"] = _json_sha(slot["no_placement_proof"])
            slots.append(record)
        ordered.append({"case_id": case["case_id"], "source_slots": slots})
    return {"format": FORMAT, "scope": "bank_input_subproof_only", "equal": True,
            "checkpoint_reuse_authorized": False,
            "source_binding": {"index": str(source), **old}, "target_binding": {"index": str(target), **new},
            "implementation": implementation, "semantic_manifest_sha256": _json_sha(manifest),
            "retained_config": manifest["config"], "ordered_cases": ordered,
            "excluded_producer_dependencies": {"source": left["excluded"], "target": right["excluded"]},
            "excluded_score_values": "all 128 finite scores independently validated; Basic uses uniform CDF",
            "serialization_policy": "typed metadata and numerical bytes; NPZ timestamps/paths ignored; NPY C/F storage-order changes rejected",
            "required_external_checks": list(REQUIRED_EXTERNAL_CHECKS)}
