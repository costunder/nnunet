"""Read existing own-task BEST weights without starting a historical runner.

Each arm is loaded in a separate worker process: the archived ``hiercp``
package and its bound geometry/constructor adapters are process-global.
No optimizer state is loaded, no old checkpoint is written, and the trained
score path (including B's original anchor/curriculum support labels) is kept.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib
from pathlib import Path
import sys

from .contracts import canonical_hash
from .experiment import digest, read


RESULT_DIRS = {"V1": "v1.0", "A": "half_A", "B": "half_B"}
_COMMON_FIELDS = (
    "architecture_version", "model_kwargs", "graph_config", "geometry_contract",
    "ct_clip", "cache_publication", "training_signature", "validation_policy",
    "preflight_calibration",
)


@dataclass
class HistoricalModel:
    arm: str
    model: object
    config: dict
    prototype_bank: object
    checkpoint: dict
    receipt: dict
    scope: dict
    source: Path
    baseline: Path
    experiment: Path


def validate_checkpoint_metadata(best, last, *, config, checkpoint_path,
                                 scope_digest, arm, experiment_digest=None):
    """Bind a selected model to its completed run; never substitute LAST.

    Kept separate from construction so wrong epoch/scope/arm/config can be
    checked without importing the archived implementation or touching CUDA.
    """
    import torch
    if arm not in RESULT_DIRS or not isinstance(best, dict) or not isinstance(last, dict):
        raise ValueError("Explicit historical V1/A/B BEST and LAST mappings required")
    if (last.get("format") != "hiercp_training_state_v1"
            or last.get("epoch") != 40 or last.get("target_epochs") != 40
            or last.get("training_complete") is not True
            or best.get("training_complete") is not True
            or best.get("completed_epoch") != 40 or best.get("target_epochs") != 40
            or type(best.get("epoch")) is not int or not 1 <= best["epoch"] <= 40
            or best["epoch"] != last.get("best_epoch")
            or best.get("best_epoch") != last.get("best_epoch")
            or best.get("best_selection") != last.get("best_selection")
            or Path(last.get("best_checkpoint", "")).resolve() != Path(checkpoint_path).resolve()):
        raise ValueError("Historical BEST is not selected by this complete40-epoch run")
    for key in _COMMON_FIELDS:
        if key not in best or key not in last or best[key] != last[key]:
            raise ValueError(f"Historical BEST/LAST metadata differs: {key}")
    if (best["model_kwargs"] != config["model"]
            or best["graph_config"] != config["graph"]
            or best["geometry_contract"] != config["graph"]["geometry_contract"]
            or tuple(best["ct_clip"]) != tuple(float(v) for v in config["ct_clip"])
            or best["training_signature"].get("seed") != 42
            or best["training_signature"].get("target_epochs") != 40
            or best["training_signature"].get("run_mode") != "production"
            or best.get("gradient_connectivity", {}).get("verified") is not True):
        raise ValueError("Historical selected model configuration/production contract differs")
    if (not isinstance(scope_digest, str) or len(scope_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in scope_digest)):
        raise ValueError("Exact archived10mm scope digest required")
    marker_names = {"v1x_bounded_scope_digest": scope_digest}
    if arm in ("A", "B"):
        if (not isinstance(experiment_digest, str) or len(experiment_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in experiment_digest)):
            raise ValueError("Historical half experiment digest required")
        marker_names[f"v1x_half_{arm.lower()}_digest"] = experiment_digest
    for payload in (best, last):
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("Historical actual model state is missing")
        for name, identity in marker_names.items():
            expected = torch.tensor(list(bytes.fromhex(identity)), dtype=torch.uint8)
            actual = state.get(name)
            if (not isinstance(actual, torch.Tensor) or actual.dtype != torch.uint8
                    or actual.shape != (32,) or not torch.equal(actual.detach().cpu(), expected)):
                raise ValueError(f"Historical actual weights lack their bound identity: {name}")
        forbidden = {"V1": ("v1x_half_a_digest", "v1x_half_b_digest"),
                     "A": ("v1x_half_b_digest",), "B": ("v1x_half_a_digest",)}[arm]
        if any(name in state for name in forbidden):
            raise ValueError("Historical model state identifies another experimental arm")
    return {"selected_epoch": best["epoch"], "completed_epochs": 40,
            "selection": copy.deepcopy(best["best_selection"]), "selection_task": "original8_candidate_curriculum",
            "optimizer_imported": False, "last_used_as_best": False}


def _activate(source, activated_scope):
    from .scope_probe_support import activate_original
    from . import bounded_scope
    if activated_scope is None:
        proof = activate_original(source)
        scope = bounded_scope.install(10, expected_snapshot_root=source)
    else:
        # A caller may need the geometry to build query inputs first. It must
        # supply the actual already installed adapter and archived module path.
        scope = copy.deepcopy(activated_scope)
        package = sys.modules.get("hiercp")
        filename = getattr(package, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(source):
            raise ValueError("Already activated hiercp is not this exact historical source")
        actual = copy.deepcopy(bounded_scope._ACTIVE)
        if actual != scope:
            raise ValueError("Already activated historical scope differs")
        proof = {"source": str(source), "already_activated_exact_scope": True}
    # Validate/cache the exact original geometry runtime before constructing a
    # PyG convolution. PyG creates runtime propagation modules when a model is
    # instantiated; they must not be mistaken for imported archive modules by
    # a later first-time geometry activation. Source verification remains exact.
    from . import transition_v1_local
    transition_v1_local._runtime(expected_snapshot_root=source,
        scope_contract=scope["contract_sha256"])
    return proof, scope


def load_historical(arm, experiment, *, budget, device="cuda", activated_scope=None):
    """Load V1/A/B own BEST and retain all trained modules/score semantics.

    ``budget`` is the caller's measured explicit resource guard. A fresh worker
    owns one arm, so historical constructor and tensor contracts cannot leak
    into another model. All existing experiment files remain read-only.
    """
    import torch
    if arm not in RESULT_DIRS or not callable(budget):
        raise ValueError("Explicit V1/A/B arm and callable resource budget required")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Historical full candidate evaluation requires actual CUDA")
    root = Path(experiment).resolve(strict=True)
    if arm == "V1":
        from .half_a_training import baseline_proof
        receipt, proof = baseline_proof(root)
        baseline = root
        source = baseline / "source/v1.0"
    else:
        module = importlib.import_module(f"hiercp_v1x.half_{arm.lower()}_training")
        receipt = module.verify(root)
        proof = receipt["baseline_proof"]
        baseline = Path(receipt["baseline_experiment"]).resolve(strict=True)
        source = Path(receipt["source"]).resolve(strict=True)
        if source != baseline / "source/v1.0":
            raise ValueError("Historical arm source is not its preserved baseline archive")
    config = copy.deepcopy(receipt["config"])
    if (config["seed"] != 42 or config["training"]["epochs"] != 40
            or config["cache"]["total_candidates"] != 8
            or config["cache"]["candidate_pool_size"] != 128
            or config["graph"]["adaptive_roi_margin_mm"] != 10
            or config["graph"]["context_outer_radius_mm"] != 10):
        raise ValueError("Historical exact10mm/seed42/full40/8sample/128pool required")
    source_proof, scope = _activate(source, activated_scope)
    if scope["contract_sha256"] != proof["neural_baseline"]["scope_digest"]:
        raise ValueError("Historical geometry adapter does not reproduce the trained10mm contract")
    if arm == "V1":
        from .scope_training_entry import install_checkpoint_binding
        install_checkpoint_binding(scope)
    else:
        importlib.import_module(f"hiercp_v1x.half_{arm.lower()}_entry").install_identity(receipt, scope)
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.contracts import require_current_checkpoint
    from hiercp.prototype import PrototypeBank
    checkpoint_path = root / "results" / RESULT_DIRS[arm] / "checkpoint_best.pt"
    last_path = checkpoint_path.with_name("checkpoint_best.last.pt")
    paths = [root / "manifest.json", root / "config.json", checkpoint_path, last_path,
             baseline / "shared/prototype_bank.pt"]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Historical regular immutable evidence is missing: {path}")
    before = {str(path): digest(path) for path in paths}
    budget()
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    last = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint = validate_checkpoint_metadata(best, last, config=config, checkpoint_path=checkpoint_path,
        scope_digest=scope["contract_sha256"], arm=arm,
        experiment_digest=receipt["contract_sha256"] if arm != "V1" else None)
    require_current_checkpoint(best)
    if best.get("prototype_bank") != str((baseline / "shared/prototype_bank.pt").resolve()):
        raise ValueError("Historical selected model prototype bank differs from its baseline")
    model = HierarchicalPyGPlacementModel(**best["model_kwargs"])
    if model.architecture_version != best["architecture_version"]:
        raise ValueError("Historical constructed architecture differs from selected actual weights")
    model.load_state_dict(best["state_dict"], strict=True)
    for value in model.state_dict().values():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("Historical selected weights contain nonfinite values")
    model.to(device).eval()
    bank = PrototypeBank.load(baseline / "shared/prototype_bank.pt")
    baseline_manifest = read(baseline / "manifest.json")
    if set(bank.training_case_ids) != set(baseline_manifest["split"]["train"]):
        raise ValueError("Historical prototype bank is not the preserved84 training patients")
    if {str(path): digest(path) for path in paths} != before:
        raise ValueError("Historical weights/evidence changed during read-only loading")
    checkpoint.update(path=str(checkpoint_path), sha256=before[str(checkpoint_path)],
        architecture=best["architecture_version"], prototype_bank_sha256=before[str(baseline / "shared/prototype_bank.pt")],
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        source_proof=source_proof, files_preserved=before)
    del best, last
    budget()
    return HistoricalModel(arm, model, config, bank, checkpoint, receipt, scope, source, baseline, root)


def rebuild_historical_support(bundle, *, output, budget):
    """B-only: rebuild its complete original anchor-label bank from BEST.

    This is an inference refresh using all151 signed training samples (each
    with8 candidates and2 views), not observed P/U labels. Receipt output is
    confined to the caller's separate evaluation directory.
    """
    if bundle.arm != "B" or not hasattr(bundle.model, "half_b"):
        raise ValueError("Only historical B owns an original anchor/curriculum support bank")
    output = Path(output).resolve()
    for preserved in (bundle.baseline, bundle.experiment, bundle.source):
        if output == preserved or output.is_relative_to(preserved) or preserved.is_relative_to(output):
            raise ValueError("Historical support output must be disjoint from every preserved experiment/source")
    output.mkdir(parents=True, exist_ok=True)
    from .half_b_support import SupportManager
    from .scope_training_entry import install_loader_hook, ScopeWorkerInitializer
    pipeline = importlib.import_module("hiercp.pipeline")
    install_loader_hook(pipeline, ScopeWorkerInitializer(
        str(bundle.source), 10, bundle.scope["contract_sha256"], False))
    manager = SupportManager(pipeline, bundle.receipt, bundle.scope, output, budget)
    manager.refresh(bundle.model, epoch=bundle.checkpoint["selected_epoch"], device="cuda",
        use_amp=bool(bundle.config["training"]["amp"]), trigger="read_only_existing_best_full128_evaluation")
    # Keep manager lifetime alongside the bound bank; callers may release its
    # loader after scoring by close(). close() intentionally clears model bank.
    return manager


def load_debug_historical(arm, source, fixture, support_fixture=None, *, budget,
                          device="cuda", workers=4):
    """Separate actual-CT/CUDA mechanical fixture with fresh full-size weights.

    The past A/B DEBUG tools saved initialization audit tensors and final SHA,
    not their trained final weights. Consequently this helper explicitly uses
    fresh seed42 weights. It does not relax ``load_historical``'s full40 guard
    and cannot produce a historical quality comparison or production asset.
    """
    import torch
    if arm not in RESULT_DIRS or not callable(budget) or type(workers) is not int or workers < 1:
        raise ValueError("Explicit V1/A/B DEBUG arm/resource guard/parallel workers required")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Historical DEBUG mechanical smoke requires actual CUDA")
    source = Path(source).resolve(strict=True)
    fixture = Path(fixture).resolve(strict=True)
    manifest_path = fixture.with_name("fixture_manifest.json")
    manifest = read(manifest_path)
    if (manifest.get("debug") is not True or manifest.get("quality_verified", False) is not False
            or manifest.get("full_training") is not False or manifest.get("production_ready") is not False
            or manifest.get("fixture_sha256") != digest(fixture)):
        raise ValueError("Actual signed DEBUG query fixture required")
    before = {str(fixture): digest(fixture), str(manifest_path): digest(manifest_path)}
    proof, scope = _activate(source, None)
    payload = torch.load(fixture, map_location="cpu", weights_only=False, mmap=True)
    if payload["config"] != manifest["config"] or payload["train_cases"] != manifest["train_cases"]:
        raise ValueError("Actual DEBUG configuration/cohort differs from its signed fixture")
    config = copy.deepcopy(payload["config"])
    if (config["seed"] != 42 or config["cache"]["total_candidates"] != 8
            or config["cache"]["candidate_pool_size"] != 128
            or config["training"]["epochs"] != 40):
        raise ValueError("Actual full-size DEBUG model must preserve production recipe")
    from . import bounded_scope
    config["graph"] = bounded_scope.configure(config["graph"], 10).to_dict()
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import set_seed
    set_seed(42, deterministic=config["runtime"]["deterministic"])
    model = HierarchicalPyGPlacementModel(**config["model"])
    if arm == "A":
        from .half_a_model import install_half_a
        install_half_a(model)
    elif arm == "B":
        from .half_b_model import install_half_b
        install_half_b(model, debug_support=True)
    model.to(device).eval()
    budget()
    receipt = {"debug": True, "actual_CT": True, "full_training": False,
        "full_evaluation": False, "quality_verified": False, "production_ready": False,
        "training_started": False, "checkpoint_written": False,
        "weight_origin": "fresh_seed42_full_model_untrained_mechanical_smoke_only",
        "fixture_sha256": before[str(fixture)], "source_proof": proof,
        "graph_config": config["graph"]}
    if arm == "B":
        if support_fixture is None:
            raise ValueError("B DEBUG requires its actual training-only support extension")
        support_fixture = Path(support_fixture).resolve(strict=True)
        extension_path = support_fixture.with_name("support_manifest_DEBUG.json")
        extension = read(extension_path)
        if (extension.get("debug") is not True or extension.get("status") != "COMPLETE"
                or extension.get("quality_verified") is not False
                or extension.get("support_fixture_sha256") != digest(support_fixture)
                or extension.get("original_fixture_sha256") != before[str(fixture)]
                or extension.get("configuration") != payload["config"]
                or canonical_hash({key: value for key, value in extension.items() if key != "identity_sha256"})
                   != extension.get("identity_sha256")):
            raise ValueError("B DEBUG support extension does not match actual fixed query fixture")
        before[str(support_fixture)] = digest(support_fixture)
        before[str(extension_path)] = digest(extension_path)
        from .scope_learning_inputs import rebuild_scope, supervision_digest
        training = [sample for sample in payload["samples"] if sample["split"] == "train"]
        bounded, _ = rebuild_scope(training, manifest, payload["config"], bounded_scope,
            10, workers, budget, profile_payload=False)
        extra_payload = torch.load(support_fixture, map_location="cpu", weights_only=False, mmap=True)
        extra = extra_payload["bounded_samples"]
        if supervision_digest(extra) != extension["native_supervision_sha256"]:
            raise ValueError("B actual support-only case supervision changed")
        canonical = bounded + extra
        cases = tuple(sample["case_id"] for sample in canonical)
        held = tuple(payload["validation_cases"])
        if (len(cases) < 3 or len(set(cases)) != len(cases) or set(cases) & set(held)
                or list(cases) != extension["support_train_cases"]):
            raise ValueError("B DEBUG full fixture bank requires3 distinct training-only cases")
        from concurrent.futures import ThreadPoolExecutor
        from hiercp.sample import materialize_sample_views
        from hiercp.data import collate_samples
        from .half_b_support import encode_local_support, SUPPORT_POLICY
        with ThreadPoolExecutor(max_workers=workers) as pool:
            views = list(pool.map(lambda sample: materialize_sample_views(
                copy.deepcopy(sample), training=False, epoch=0, global_seed=42), canonical))
        cpu = collate_samples(views)
        embeddings = encode_local_support(model, cpu, device=device,
            use_amp=bool(config["training"]["amp"]))
        sample_ids = tuple(f"DEBUG:{sample['case_id']}:{sample['sample_index']}" for sample in canonical)
        memory = dict(embeddings=embeddings, owners=torch.arange(len(cases), device=device).repeat_interleave(8),
            classes=torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], device=device).repeat(len(cases)),
            patient_case_ids=cases, sample_ids=tuple(name for name in sample_ids for _ in range(8)),
            candidate_indices=tuple(range(8)) * len(cases), expected_samples=dict(zip(sample_ids, cases)),
            training_case_ids=cases, validation_case_ids=held,
            manifest_sha256=before[str(extension_path)], generation="DEBUG_fresh_mechanical_bank",
            epoch=0, fixed_view_epoch=0, support_policy=SUPPORT_POLICY,
            debug=True, full_signed_training_cache=False)
        receipt["support"] = model.half_b.bind_support(memory)
        del embeddings, cpu, views, canonical, extra_payload
    if {name: digest(name) for name in before} != before:
        raise ValueError("Actual historical DEBUG fixture changed during read-only construction")
    checkpoint = {"debug": True, "selected_epoch": None, "completed_epochs": 0,
        "optimizer_imported": False, "weight_origin": receipt["weight_origin"],
        "architecture": model.architecture_version,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "files_preserved": before, "checkpoint_written": False}
    return HistoricalModel(arm, model, config, payload["prototype_bank"], checkpoint,
        receipt, scope, source, fixture.parent, fixture.parent)
