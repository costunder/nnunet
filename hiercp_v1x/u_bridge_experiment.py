"""Immutable inputs and matched execution for the v1.8 comparison-position trial."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "v18_u_bridge_matched_experiment_v1"
FILES = ("hiercp_v1x/u_bridge_experiment.py", "hiercp_v1x/u_bridge_data.py",
         "hiercp_v1x/u_bridge_fields.py", "hiercp_v1x/u_bridge_calibration.py",
         "hiercp_v1x/u_bridge_upper.py",
         "hiercp_v1x/u_bridge_training.py", "hiercp_v1x/bounded_scope.py",
         "hiercp_v1x/scope_probe_support.py", "hiercp_v1x/scope_learning_loop.py",
         "tools/run_v18_u_bridge.py", "tools/local_cnn_device.py", "config/v18_u_bridge.json")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepared_data_path(experiment, prepared=None, baseline=None):
    """Choose a verified cache namespace without replacing old experiments.

    Canonical entries validate their exact CT/source/config binding on access.
    A reused namespace can gain missing entries and exact distance fields;
    existing graph files are never overwritten. Its own lock serializes users.
    """
    import psutil
    root = Path(experiment).resolve()
    if prepared is None:
        return root / "data"
    candidate = Path(prepared)
    if candidate.is_symlink():
        raise ValueError("Prepared cache must be a regular directory, not a symlink")
    path = candidate.resolve(strict=True)
    if not path.is_dir() or path == root or root.is_relative_to(path) or path.is_relative_to(root):
        raise ValueError("Reused prepared cache must be disjoint from the new experiment")
    if baseline is not None:
        preserved = Path(baseline).resolve(strict=True)
        if path.is_relative_to(preserved) or preserved.is_relative_to(path):
            raise ValueError("Preserved baseline cannot be a writable preparation cache")
    owner_path = path.parent / ".pipeline.lock"
    if owner_path.exists():
        owner = read(owner_path)
        if (owner.get("host") != socket.gethostname() or type(owner.get("pid")) is not int
                or psutil.pid_exists(owner["pid"])):
            raise RuntimeError(f"Prepared cache's original experiment may still be active: {owner_path}; {owner}")
    return path


@contextmanager
def lock(path):
    """Only remove a positively identified stale owner on this same host."""
    import psutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    if path.exists():
        owner = read(path)
        if owner.get("host") != socket.gethostname() or type(owner.get("pid")) is not int:
            raise RuntimeError(f"Experiment lock belongs to another/unknown host: {path}; {owner}")
        if psutil.pid_exists(owner["pid"]):
            raise RuntimeError(f"Experiment owner is still alive: {path}; {owner}")
        if read(path) != owner:
            raise RuntimeError("Lock changed during stale-owner inspection")
        recovered = path.with_name(path.name + ".stale." + owner.get("token", uuid.uuid4().hex))
        if recovered.exists():
            raise FileExistsError(f"Preserved stale-lock receipt already exists: {recovered}")
        path.rename(recovered)
        print(f"Recovered dead local owner PID={owner['pid']}; receipt={recovered}", flush=True)
    owner = dict(host=socket.gethostname(), pid=os.getpid(), token=token)
    write_new(path, owner)
    try:
        yield
    finally:
        if path.exists() and read(path) == owner:
            path.unlink()


class Budget:
    def __init__(self, cuda_bytes, rss_bytes):
        if min(cuda_bytes, rss_bytes) <= 0:
            raise ValueError("Explicit positive resource budgets required")
        self.cuda_bytes, self.rss_bytes = int(cuda_bytes), int(rss_bytes)

    def check(self):
        import psutil
        import torch
        if torch.cuda.memory_allocated() > self.cuda_bytes:
            raise MemoryError("Declared CUDA allocation budget exceeded")
        if psutil.Process().memory_info().rss > self.rss_bytes:
            raise MemoryError("Declared process RSS budget exceeded")


def prepare_inputs(a):
    """Retain signed source identities; no new anchors or candidate GT."""
    if a.debug:
        fixture = read(a.debug_fixture / "fixture_manifest.json")
        if fixture.get("debug") is not True:
            raise ValueError("An explicit real-CT DEBUG fixture is required")
        config = copy.deepcopy(read(a.debug_config)["configuration"])
        samples = [dict(path=str((a.debug_fixture / item["name"]).resolve()), sha256=item["sha256"])
                   for item in fixture["files"]]
        source = a.debug_source.resolve(strict=True)
        proof = dict(debug=True, fixture_sha256=sha(a.debug_fixture / "fixture_manifest.json"),
                     configuration_sha256=sha(a.debug_config), bank_fixture_sha256=sha(a.debug_bank),
                     training_case_ids=sorted({r["case_id"] for r in fixture["source_records"]
                                              if any(f["name"].startswith(r["case_id"] + ".") and f["split"] == "train"
                                                     for f in fixture["files"])}),
                     validation_case_ids=sorted({r["case_id"] for r in fixture["source_records"]
                                                if any(f["name"].startswith(r["case_id"] + ".") and f["split"] == "val"
                                                       for f in fixture["files"])}))
        bank_path, regions = a.debug_bank.resolve(strict=True), None
    else:
        from .half_a_training import baseline_proof
        native, baseline = baseline_proof(a.baseline)
        signature = baseline["neural_baseline"]["training_signature"]
        cache = a.baseline / "shared/cache"
        index = read(cache / "index.json")
        entries = {r["path"]: r for r in index["entries"]}
        names = signature["train_cache_files"] + signature["val_cache_files"]
        if set(entries) != set(names) or len(set(names)) != len(names):
            raise ValueError("Signed original materialized sample list differs")
        samples = [dict(path=str((cache / name).resolve()), sha256=entries[name]["artifact_sha256"])
                   for name in names]
        config = copy.deepcopy(native["config"])
        source = (a.baseline / "source/v1.0").resolve(strict=True)
        bank_path = (a.baseline / "shared/prototype_bank.pt").resolve(strict=True)
        regions = (a.baseline / "shared/regions").resolve(strict=True)
        actual_val = sorted({entries[n]["case_id"] for n in signature["val_cache_files"]})
        proof = dict(debug=False, baseline_proof_sha256=digest(baseline),
            cache_index_sha256=sha(cache / "index.json"), baseline=str(a.baseline.resolve()),
            training_case_ids=native["split"]["train"], validation_case_ids=native["split"]["val"],
            actual_train_cases=sorted({entries[n]["case_id"] for n in signature["train_cache_files"]}),
            actual_validation_cases=actual_val,
            configured_but_not_materialized_validation_cases=sorted(set(native["split"]["val"]) - set(actual_val)),
            actual_train_samples=len(signature["train_cache_files"]),
            actual_validation_samples=len(signature["val_cache_files"]),
            measured_baseline_execution=baseline["execution"], bank_sha256=sha(bank_path))
    if (config["seed"] != 42 or config["training"]["epochs"] != 40
            or config["cache"]["total_candidates"] != 8 or config["cache"]["candidate_pool_size"] != 128
            or config["model"]["hidden_dim"] != 128 or config["model"]["heads"] != 4
            or [config["model"][k] for k in ("local_layers", "patient_layers", "prototype_layers")] != [3, 2, 2]):
        raise ValueError("Original full v1 model/seed42/40epoch/8-of128 contract required")
    config["graph"]["adaptive_roi_margin_mm"] = 10.
    config["graph"]["context_outer_radius_mm"] = 10.
    if config["training"]["consistency_weight"] != .1:
        raise ValueError("Original two-view consistency .1 required")
    inventory = read(a.inventory)
    if not a.debug and (inventory.get("debug") is not False or inventory.get("complete") is not True):
        raise ValueError("Production requires the completed original native128 inventory, not DEBUG")
    raw = inventory["raw_records"]
    proof.update(inventory_sha256=sha(a.inventory), source_samples=samples)
    return samples, raw, config, source, bank_path, regions, proof


def joint_calibration(reports, candidates):
    """Choose one physical sample batch accepted by BOTH comparison arms."""
    # The neural calibration adapter supplies a normalized candidates list.
    tables = {arm: {int(r["physical_batch"]): r for r in report["reports"]}
              for arm, report in reports.items()}
    if set(tables) != {"selected", "native"}:
        raise ValueError("Both real-arm measurements are required")
    accepted = [n for n in candidates if all(tables[arm].get(n, {}).get("accepted") is True
                                           for arm in tables)]
    if not accepted:
        raise RuntimeError("No explicit physical sample batch passed both arms; measured reports preserved")
    winner = max(accepted, key=lambda n: min(tables[arm][n]["samples_per_second"] for arm in tables))
    return dict(physical_batch=winner, accepted=True, arm="matched_both",
                explicit_candidates=list(candidates), accepted_common=accepted,
                selection="max minimum measured samples/sec across both arms", reports=reports)
