"""Read a real legacy native30 BEST with its explicitly supplied source.

This loader starts no training, restores no optimizer and writes no checkpoint.
Missing old architecture metadata stays UNKNOWN. Strict loading verifies the
chosen source's state schema, not historical training-time source provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


FORMAT = "legacy_native30_best_readonly_loader_v1"
_ACTIVE = None
_SOURCE_REQUIRED = ("__init__.py", "schema.py", "local.py", "spatial.py", "sample.py",
                    "common.py", "hierarchy.py", "prototype.py", "region.py", "curriculum.py")
_MODEL_FIELDS = ("hidden_dim", "heads", "local_layers", "patient_layers", "prototype_layers",
                 "dropout", "dense_base_channels", "dense_feature_dim", "dense_batch_size",
                 "channels_last_3d", "checkpoint_local_blocks", "checkpoint_dense_encoder")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _json(value, label):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Finite saved JSON metadata required: " + label) from error


def _native_configuration(kwargs, graph, clip):
    if (not isinstance(kwargs, Mapping) or any(key not in kwargs for key in _MODEL_FIELDS)
            or kwargs.get("ablation_mode", "full") != "full" or not isinstance(graph, Mapping)):
        raise ValueError("Complete saved full-model kwargs and native graph configuration required")
    if (graph.get("adaptive_roi_margin_mm") != 30
            or graph.get("context_outer_radius_mm") != 28
            or tuple(graph.get("context_shells_mm", ())) != (4, 12, 28)
            or graph.get("context_radius_mm") != 28 or graph.get("patch_size") != 48
            or graph.get("canonical_full_graph") is not True):
        raise ValueError("Original native ROI30/context28/shell4,12,28/CNN48 required; no bounded30 substitution")
    if not isinstance(clip, (tuple, list)) or tuple(clip) != (-200., 250.):
        raise ValueError("Saved original CT clipping contract required")
    _json(kwargs, "model_kwargs")
    _json(graph, "graph_config")


def _model_class(source):
    import torch
    name = "hiercp.model" if (source / "hiercp/model.py").is_file() else "hiercp.pyg_models"
    model_class = getattr(importlib.import_module(name), "HierarchicalPyGPlacementModel", None)
    if not isinstance(model_class, type) or not issubclass(model_class, torch.nn.Module):
        raise ValueError("Actual selected source has no HierarchicalPyGPlacementModel class")
    return model_class


def _cuda(device, budget):
    import torch
    if not callable(budget):
        raise ValueError("Explicit callable native30 resource budget required")
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Native30 neural evaluation requires actual CUDA; no CPU fallback")
    return selected


def _inventory(source):
    directory = source / "hiercp"
    result = {}
    for path in sorted(directory.rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Legacy original Python source escapes the selected directory")
        result[path.relative_to(source).as_posix()] = sha(path)
    if not result:
        raise ValueError("Explicit actual legacy hiercp Python source is missing")
    return result


def activate_native30_source(original_source):
    """Bind and activate one actual original source in a fresh worker process.

    No v5 archive or bounded10/bounded30 adapter is installed. All original
    importable hiercp Python files are hashed, so later imports remain bound.
    """
    global _ACTIVE
    source = Path(original_source).resolve(strict=True)
    if _ACTIVE is not None:
        if source != Path(_ACTIVE["source"]):
            raise RuntimeError("A different original source requires a fresh native30 worker")
        return verify_native30_source()
    if any(name == "hiercp" or name.startswith("hiercp.") for name in sys.modules):
        raise RuntimeError("Activate legacy native30 source before every hiercp import")
    if not source.is_dir() or any(not (source / "hiercp" / name).is_file() for name in _SOURCE_REQUIRED):
        raise ValueError("Complete explicit actual legacy hiercp source directory required")
    if not any((source / "hiercp" / name).is_file() for name in ("model.py", "pyg_models.py")):
        raise ValueError("Actual legacy model.py or pyg_models.py is required")
    config = source / "config/train.json"
    if config.is_symlink() or not config.is_file():
        raise ValueError("Actual source/config/train.json execution settings required")
    inventory = _inventory(source)
    selected = dict(format=FORMAT, source=str(source), source_python_sha256=inventory,
                    source_config_path=str(config), source_config_sha256=sha(config),
                    source_provenance_at_training="UNKNOWN",
                    archive_equivalence_asserted=False, bounded_scope_installed=False,
                    source_files_written=False)
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    importlib.import_module("hiercp")
    _ACTIVE = selected
    return verify_native30_source()


def _generated_modules():
    """Identify PyG generated code through verified original class methods."""
    found = {}
    for module_name, module in list(sys.modules.items()):
        if not (module_name == "hiercp" or module_name.startswith("hiercp.")):
            continue
        path = getattr(module, "__file__", None)
        if path is None or not Path(path).resolve().is_relative_to(Path(_ACTIVE["source"])):
            continue
        for value in vars(module).values():
            if not isinstance(value, type) or value.__module__ != module_name:
                continue
            for method in ("propagate", "edge_updater"):
                operation = getattr(value, method, None)
                generated = getattr(operation, "__module__", "")
                if generated.startswith(module_name + "_") and generated.endswith("_" + method):
                    found[generated] = dict(original_class=value.__qualname__, original_module=module_name,
                                            bound_method=method)
    return found


def verify_native30_source(proof=None):
    """Check all selected Python bytes and every currently imported hiercp file."""
    if _ACTIVE is None:
        raise RuntimeError("Explicit native30 source has not been activated")
    source = Path(_ACTIVE["source"])
    if (_inventory(source) != _ACTIVE["source_python_sha256"]
            or sha(_ACTIVE["source_config_path"]) != _ACTIVE["source_config_sha256"]):
        raise ValueError("Selected legacy source/config bytes changed during evaluation")
    if proof is not None and any(proof.get(key) != _ACTIVE[key] for key in
                                ("source", "source_python_sha256", "source_config_sha256")):
        raise ValueError("Legacy source proof identifies another source or changed bytes")
    imported, generated = {}, {}
    allowed_generated = _generated_modules()
    for name, module in list(sys.modules.items()):
        if not (name == "hiercp" or name.startswith("hiercp.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise ValueError("Imported legacy hiercp module has no bound Python file: " + name)
        path = Path(filename).resolve(strict=True)
        if path.is_relative_to(source):
            relative = path.relative_to(source).as_posix()
            expected = _ACTIVE["source_python_sha256"].get(relative)
            if expected is None or sha(path) != expected:
                raise ValueError("Imported legacy hiercp Python file is not in selected source: " + name)
            imported[name] = dict(path=str(path), sha256=expected)
        elif name in allowed_generated:
            generated[name] = {**allowed_generated[name], "path": str(path), "sha256": sha(path),
                               "PyG_generated_runtime_code": True}
        else:
            raise ValueError("Imported hiercp module belongs to another source: " + name)
    if proof is not None:
        for name, previous in proof.get("generated_runtime_modules", {}).items():
            if generated.get(name) != previous:
                raise ValueError("Bound generated PyG runtime code changed: " + name)
    return {**copy.deepcopy(_ACTIVE), "imported_source_modules": imported,
            "generated_runtime_modules": generated, "source_preserved": True}


def validate_native30_metadata(payload, checkpoint_path):
    """Validate saved BEST metadata and real tensors without constructing a model."""
    import torch
    path = Path(checkpoint_path)
    if path.name != "model.pt":
        raise ValueError("Explicit existing model.pt BEST required; LAST/latest is not BEST")
    if not isinstance(payload, Mapping):
        raise ValueError("Saved legacy native30 checkpoint metadata mapping required")
    if (payload.get("method") != "hiercp-full" or payload.get("framework") != "torch_geometric"
            or payload.get("format") == "hiercp_training_state_v1"
            or payload.get("training_complete") is not True
            or payload.get("completed_epoch") != 40 or payload.get("target_epochs") != 40
            or type(payload.get("epoch")) is not int or not 1 <= payload["epoch"] <= 40
            or payload.get("best_epoch") != payload["epoch"]):
        raise ValueError("Real selected native30 BEST from a complete40-epoch run required")
    kwargs, graph, clip = payload.get("model_kwargs"), payload.get("graph_config"), payload.get("ct_clip")
    _native_configuration(kwargs, graph, clip)
    state = payload.get("state_dict")
    if (not isinstance(state, Mapping) or not state or any(not isinstance(name, str) or not name
                                                         for name in state)):
        raise ValueError("Actual nonempty saved native30 model state required")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if not tensors or any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("Actual finite native30 state tensors required")
    if any(name.startswith("v1x_") for name in state):
        raise ValueError("Bounded/half/sampling experiment weights are not original native30")
    cases = payload.get("prototype_training_cases")
    if (not isinstance(cases, (list, tuple)) or not cases
            or any(not isinstance(case, str) or not case for case in cases) or len(set(cases)) != len(cases)):
        raise ValueError("Explicit saved unique prototype training case IDs required")
    fingerprint = payload.get("prototype_fingerprint")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)):
        raise ValueError("Saved exact prototype fingerprint required")
    selection = payload.get("best_selection")
    if not isinstance(selection, Mapping) or not selection:
        raise ValueError("Saved own-task BEST selection metadata required")
    signature = payload.get("training_signature")
    if (not isinstance(signature, Mapping) or type(signature.get("seed")) is not int
            or signature["seed"] != 42 or signature.get("run_mode") != "production"):
        raise ValueError("Saved original production training_signature seed42 required")
    return dict(format=FORMAT, selected_epoch=payload["epoch"], completed_epochs=40,
                checkpoint_selection="explicit existing own-task BEST model.pt",
                selection_task="original_source_anchor_vs_curriculum_candidates",
                selection=_json(selection, "best_selection"),
                saved_architecture_version=payload.get("architecture_version", "UNKNOWN"),
                saved_geometry_contract=payload.get("geometry_contract", "UNKNOWN"),
                saved_model_kwargs=_json(kwargs, "model_kwargs"), saved_graph_config=_json(graph, "graph_config"),
                ct_clip=list(map(float, clip)), prototype_training_cases=list(cases),
                prototype_fingerprint=fingerprint, original_native_roi30_context28=True,
                seed=signature["seed"], seed_origin="saved checkpoint training_signature",
                optimizer_imported=False, optimizer_instantiated=False, training_started=False,
                checkpoint_written=False, quality_verified=False, production_ready=False,
                strict_loading_is_training_source_equivalence_proof=False,
                source_provenance_at_training="UNKNOWN")


def load_native30(checkpoint, prototype, original_source, budget, device="cuda"):
    """Construct the supplied original model and strictly read its real BEST."""
    import torch
    selected_device = _cuda(device, budget)
    source = Path(original_source).resolve(strict=True)
    source_proof = activate_native30_source(source)
    checkpoint, prototype = Path(checkpoint).resolve(strict=True), Path(prototype).resolve(strict=True)
    for path in (checkpoint, prototype):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Explicit regular native30 checkpoint and prototype files required")
    before = {str(path): sha(path) for path in (checkpoint, prototype)}
    budget()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    receipt = validate_native30_metadata(payload, checkpoint)
    source_config = json.loads((source / "config/train.json").read_text(encoding="utf8"))
    amp = source_config.get("training", {}).get("amp")
    if type(amp) is not bool:
        raise ValueError("Explicit actual source/config/train.json training.amp execution setting required")
    cache_config = source_config.get("cache")
    if (not isinstance(cache_config, Mapping) or type(cache_config.get("source_pad")) is not int
            or cache_config["source_pad"] < 0):
        raise ValueError("Explicit actual original source/config/train.json cache.source_pad required")
    saved_cache = payload["training_signature"].get("cache", {})
    saved_pad = saved_cache.get("source_pad") if isinstance(saved_cache, Mapping) else None
    if saved_pad is not None and saved_pad != cache_config["source_pad"]:
        raise ValueError("Saved source padding differs from explicitly supplied original source config")
    model_class = _model_class(source)
    model = model_class(**copy.deepcopy(payload["model_kwargs"]))
    # No partial load, injected revision buffer or retry with another model.
    model.load_state_dict(payload["state_dict"], strict=True)
    bank_class = getattr(importlib.import_module("hiercp.prototype"), "PrototypeBank")
    bank = bank_class.load(prototype)
    if (bank.fingerprint() != receipt["prototype_fingerprint"]
            or set(bank.training_case_ids) != set(receipt["prototype_training_cases"])):
        raise ValueError("Actual native30 prototype fingerprint/case IDs differ from saved BEST")
    bank.validate()
    config = {"seed": receipt["seed"], "model": copy.deepcopy(payload["model_kwargs"]), "graph": copy.deepcopy(payload["graph_config"]),
              "ct_clip": list(receipt["ct_clip"]), "training": {"amp": amp},
              "cache": copy.deepcopy(cache_config)}
    graph_config = importlib.import_module("hiercp.schema").graph_config_from_dict(config["graph"])
    graph_config.validate()
    model.to(selected_device).eval()
    budget()
    after = {str(path): sha(path) for path in (checkpoint, prototype)}
    if after != before:
        raise ValueError("Original native30 checkpoint/prototype bytes changed during read-only loading")
    proof = verify_native30_source(source_proof)
    receipt.update(path=str(checkpoint), sha256=before[str(checkpoint)], prototype_path=str(prototype),
                   prototype_sha256=before[str(prototype)], files_preserved=before,
                   source_proof=proof, actual_model_class=model_class.__module__ + "." + model_class.__qualname__,
                   loaded_model_architecture_version=getattr(model, "architecture_version", "UNKNOWN"),
                   state_loaded_strict=True, state_schema_compatible=True,
                   total_parameters=sum(parameter.numel() for parameter in model.parameters()),
                   trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                   AMP_execution_setting=amp, AMP_setting_origin="current explicitly selected source/config/train.json",
                   source_pad=cache_config["source_pad"],
                   source_pad_setting_origin="current explicitly selected original source/config/train.json cache.source_pad",
                   source_pad_at_training=saved_pad if saved_pad is not None else "UNKNOWN",
                   AMP_setting_is_training_provenance=False, actual_CUDA=True)
    del payload
    return SimpleNamespace(model=model, config=config, bank=bank, prototype_bank=bank,
                           receipt=receipt, checkpoint=receipt, source=source, source_proof=proof)


def load_debug_native30(original_source, config, prototype, budget, device="cuda"):
    """Fresh full original model for an explicitly UNTRAINED actual CT/CUDA smoke.

    This separate path reads no production checkpoint and creates none. Its
    original model configuration cannot be replaced with a smaller smoke model.
    """
    selected_device = _cuda(device, budget)
    source = Path(original_source).resolve(strict=True)
    source_proof = activate_native30_source(source)
    original_config = json.loads((source / "config/train.json").read_text(encoding="utf8"))
    if _json(config, "DEBUG config") != original_config:
        raise ValueError("DEBUG smoke must preserve the exact original source/config/train.json model and graph")
    _native_configuration(config.get("model"), config.get("graph"), config.get("ct_clip"))
    if config.get("seed") != 42 or type(config.get("training", {}).get("amp")) is not bool:
        raise ValueError("DEBUG original full-model seed42 and explicit AMP setting required")
    deterministic = config.get("runtime", {}).get("deterministic")
    if type(deterministic) is not bool:
        raise ValueError("DEBUG original explicit deterministic execution setting required")
    prototype = Path(prototype).resolve(strict=True)
    if not prototype.is_file():
        raise ValueError("Actual DEBUG training-case-fitted prototype file required")
    prototype_sha = sha(prototype)
    budget()
    importlib.import_module("hiercp.tensor").set_seed(42, deterministic=deterministic)
    model_class = _model_class(source)
    model = model_class(**copy.deepcopy(config["model"]))
    bank = importlib.import_module("hiercp.prototype").PrototypeBank.load(prototype)
    bank.validate()
    graph_config = importlib.import_module("hiercp.schema").graph_config_from_dict(config["graph"])
    graph_config.validate()
    model.to(selected_device).eval()
    budget()
    if sha(prototype) != prototype_sha:
        raise ValueError("Actual DEBUG prototype changed during read-only load")
    proof = verify_native30_source(source_proof)
    receipt = dict(format="native30_fresh_full_model_actual_CT_CUDA_DEBUG_v1", debug=True,
                   weight_origin="fresh_seed42_original_full_model_UNTRAINED",
                   trained_weights=False, full_training=False, full_evaluation=False,
                   training_started=False, optimizer_instantiated=False, optimizer_imported=False,
                   checkpoint_written=False, quality_verified=False, production_ready=False,
                   original_native_roi30_context28=True, source_proof=proof, seed=42,
                   source_provenance_at_training="NOT_APPLICABLE_UNTRAINED_DEBUG",
                   actual_model_class=model_class.__module__ + "." + model_class.__qualname__,
                   total_parameters=sum(parameter.numel() for parameter in model.parameters()),
                   trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                   prototype_path=str(prototype), prototype_sha256=prototype_sha,
                   prototype_training_cases=list(bank.training_case_ids), prototype_fingerprint=bank.fingerprint(),
                   AMP_execution_setting=config["training"]["amp"], actual_CUDA=True)
    return SimpleNamespace(model=model, config=copy.deepcopy(config), bank=bank, prototype_bank=bank,
                           receipt=receipt, checkpoint=receipt, source=source, source_proof=proof)
