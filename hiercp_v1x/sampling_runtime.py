"""Install an explicitly bound sampler in an isolated preserved-v1 snapshot.

Only immutable CPU graph metadata is reduced. CNN/GAT outputs are never cached.
The append-only snapshot hook runs in the parent and spawned DataLoader workers.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Mapping


FORMAT = "hiercp_v1_strict_nested_runtime_v1"
ENVIRONMENT_KEY = "HIERCP_V1X_SAMPLING_CONTRACT"
DIGEST_BUFFER = "v1x_sampling_digest"
ARCHITECTURE_SEPARATOR = "|sampling_"
SAMPLING_IMPORT_HOOK = (
    b"\n\n# Explicit v1.x sampler; reinstall in spawned DataLoader workers.\n"
    b"from hiercp_v1x.sampling_runtime import install_from_environment as _v1x_install_sampler\n"
    b"_v1x_install_sampler()\n"
)
_ACTIVE = None
_ORIGINAL_BUILD = None
_ORIGINAL_MATERIALIZE = None


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_contract(mode, graph_config, source_identity, profile=None):
    """Create a JSON identity; callers must supply all strict role budgets."""
    if mode not in ("native", "strict_nested"):
        raise ValueError("Sampling mode must be native or strict_nested")
    if hasattr(graph_config, "to_dict"):
        graph_config = graph_config.to_dict()
    if not isinstance(graph_config, Mapping) or not graph_config:
        raise ValueError("Exact original graph_config is required")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise ValueError("Explicit snapshot source identity is required")
    if mode == "native":
        if profile is not None:
            raise ValueError("Native sampling cannot adopt a nested profile")
        selected = None
    else:
        from hiercp_v1x.graph_size import GraphSeedProfile
        if isinstance(profile, GraphSeedProfile):
            selected = dict(profile.budgets)
        elif isinstance(profile, Mapping):
            selected = dict(GraphSeedProfile(profile).budgets)
        else:
            raise ValueError("strict_nested requires explicit six-role seed budgets")
    contract = dict(format=FORMAT, mode=mode, graph_config=_plain(graph_config),
                    source_identity=_plain(source_identity), profile=selected)
    contract["contract_sha256"] = canonical_hash(contract)
    return validate_contract(contract)


def validate_contract(contract):
    if not isinstance(contract, Mapping):
        raise ValueError("Sampling contract must be a mapping")
    required = {"format", "mode", "graph_config", "source_identity", "profile", "contract_sha256"}
    if set(contract) != required or contract["format"] != FORMAT:
        raise ValueError("Sampling contract format/fields differ")
    plain = dict(contract)
    expected = plain.pop("contract_sha256")
    if expected != canonical_hash(plain):
        raise ValueError("Sampling contract digest mismatch")
    if contract["mode"] not in ("native", "strict_nested"):
        raise ValueError("Invalid sampling mode")
    if not isinstance(contract["graph_config"], Mapping) or not contract["graph_config"]:
        raise ValueError("Missing sampling graph_config")
    identity = contract["source_identity"]
    if not isinstance(identity, Mapping) or not isinstance(identity.get("files"), Mapping):
        raise ValueError("Sampling source_identity.files is required")
    required_files = {
        "hiercp/sample.py", "hiercp/schema.py", "hiercp/spatial.py",
        "hiercp_v1x/sampling_runtime.py", "hiercp_v1x/sampling_entry.py",
        "hiercp_v1x/graph_size.py", "hiercp_v1x/nested_graph_size.py",
    }
    if not required_files <= set(identity["files"]):
        raise ValueError("Sampling source identity lacks a required implementation")
    for name, digest in identity["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Sampling source identity must use contained POSIX paths")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Sampling source SHA256 malformed")
    native_hash = identity.get("native_sample_sha256")
    if (not isinstance(native_hash, str) or len(native_hash) != 64
            or any(c not in "0123456789abcdef" for c in native_hash)):
        raise ValueError("Exact original native sample SHA256 is required")
    from hiercp_v1x.graph_size import GraphSeedProfile
    if contract["mode"] == "native":
        if contract["profile"] is not None:
            raise ValueError("Native contract cannot carry nested seed budgets")
    else:
        GraphSeedProfile(contract["profile"])
    return copy.deepcopy(dict(contract))


def load_environment_contract():
    raw = os.environ.get(ENVIRONMENT_KEY)
    if not raw:
        raise ValueError("Explicit sampler contract environment is missing")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("Sampler contract must be an absolute JSON path")
    return validate_contract(json.loads(path.read_text(encoding="utf-8")))


def _verify_snapshot(sample_module, contract):
    source = Path(sample_module.__file__).resolve()
    root = source.parent.parent
    for name, expected in contract["source_identity"]["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"Sampler snapshot implementation mismatch: {name}")
    for name in ("hiercp.schema", "hiercp.spatial"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(f"Sampler imported another source snapshot: {name}")
    content = source.read_bytes()
    if contract["mode"] == "strict_nested":
        if not content.endswith(SAMPLING_IMPORT_HOOK):
            raise ValueError("Strict snapshot lacks the spawned-worker sampler hook")
        content = content[:-len(SAMPLING_IMPORT_HOOK)]
    if hashlib.sha256(content).hexdigest() != contract["source_identity"]["native_sample_sha256"]:
        raise ValueError("Native sampler prefix is not the preserved original source")


def _check_config(config, contract):
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    if _plain(payload) != contract["graph_config"]:
        raise ValueError("Sampling graph_config differs from the bound original configuration")


def runtime_build_local_view(source_local, target_local, config, *, seed):
    if _ACTIVE is None or _ORIGINAL_BUILD is None:
        raise RuntimeError("Strict sampler was not installed in this process")
    _check_config(config, _ACTIVE)
    native = _ORIGINAL_BUILD(source_local, target_local, config, seed=seed)
    from hiercp_v1x.graph_size import GraphSeedProfile
    from hiercp_v1x.nested_graph_size import build_local_view
    selected, audit = build_local_view(native, config, seed=seed,
                                      profile=GraphSeedProfile(_ACTIVE["profile"]))
    selected.v1x_sampling_contract_sha256 = _ACTIVE["contract_sha256"]
    # The sampler already validates the exact induced subset. Directed BFS is
    # a separate topology diagnostic and is not repeated in every update.
    selected.v1x_sampling_seconds = float(audit["preparation_seconds"])
    return selected


def _view_identity(sample, *, training, epoch, global_seed):
    return dict(case_id=str(sample.get("case_id", "")),
                sample_index=int(sample.get("sample_index", -1)),
                effective_epoch=int(epoch) if training else 0,
                global_seed=int(global_seed))


def _validate_materialized(sample, contract, *, training, epoch, global_seed):
    expected = contract["contract_sha256"]
    first, second = sample.get("local_graphs"), sample.get("local_graphs_view2")
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        raise ValueError("Both materialized original views are required")
    if not first or len(first) != len(second):
        raise ValueError("Materialized candidate/view cardinality mismatch")
    if sample.get("v1x_sampling_contract_sha256") != expected:
        raise ValueError("Pre-materialized sample belongs to another sampler contract")
    identity = _view_identity(sample, training=training, epoch=epoch, global_seed=global_seed)
    if sample.get("v1x_sampling_view_identity") != identity:
        raise ValueError("Pre-materialized view belongs to another case/sample/epoch/global seed")
    from hiercp.sample import stable_view_seed
    for view_index, graphs in enumerate((first, second)):
        for candidate_index, graph in enumerate(graphs):
            if graph.get("v1x_sampling_contract_sha256") != expected:
                raise ValueError("Pre-materialized graph is native, unbound, or from another sampler")
            seed = stable_view_seed(identity["global_seed"], identity["case_id"],
                                    identity["sample_index"], candidate_index,
                                    identity["effective_epoch"], view_index) & 0x7FFFFFFF
            actual = graph.get("view_seed")
            if actual is None or tuple(actual.shape) != (1,) or int(actual[0]) != seed:
                raise ValueError("Pre-materialized candidate/order/view seed differs")


def runtime_materialize_sample_views(sample, *, training, epoch, global_seed):
    if _ACTIVE is None or _ORIGINAL_MATERIALIZE is None:
        raise RuntimeError("Strict materializer was not installed in this process")
    _check_config(sample.get("graph_config"), _ACTIVE)
    if "local_graphs" in sample or "local_graphs_view2" in sample:
        _validate_materialized(sample, _ACTIVE, training=training, epoch=epoch, global_seed=global_seed)
        return sample
    result = _ORIGINAL_MATERIALIZE(sample, training=training, epoch=epoch,
                                   global_seed=global_seed)
    result["v1x_sampling_contract_sha256"] = _ACTIVE["contract_sha256"]
    result["v1x_sampling_view_identity"] = _view_identity(result, training=training, epoch=epoch,
                                                        global_seed=global_seed)
    _validate_materialized(result, _ACTIVE, training=training, epoch=epoch, global_seed=global_seed)
    return result


def install_from_environment():
    """No environment means exact native preparation; strict hook is idempotent."""
    global _ACTIVE, _ORIGINAL_BUILD, _ORIGINAL_MATERIALIZE
    if not os.environ.get(ENVIRONMENT_KEY):
        return None
    contract = load_environment_contract()
    sample = sys.modules.get("hiercp.sample")
    if sample is None:
        sample = importlib.import_module("hiercp.sample")
    if _ACTIVE is not None:
        if _ACTIVE != contract:
            raise ValueError("A different sampling contract is already installed")
        return copy.deepcopy(_ACTIVE)
    _verify_snapshot(sample, contract)
    from hiercp.schema import graph_config_from_dict
    config = graph_config_from_dict(dict(contract["graph_config"]))
    config.validate()
    _check_config(config, contract)
    if contract["mode"] == "native":
        # Leave native function identities, output fields and sampling untouched.
        return contract
    _ORIGINAL_BUILD = sample.build_local_view
    _ORIGINAL_MATERIALIZE = sample.materialize_sample_views
    _ACTIVE = contract
    sample.build_local_view = runtime_build_local_view
    sample.materialize_sample_views = runtime_materialize_sample_views
    # A manual installation after hiercp.data import also updates its alias.
    data = sys.modules.get("hiercp.data")
    if data is not None and getattr(data, "materialize_sample_views", None) is _ORIGINAL_MATERIALIZE:
        data.materialize_sample_views = runtime_materialize_sample_views
    return copy.deepcopy(contract)


def apply_sampling_identity(model, contract):
    """Persist strict topology identity without changing any learned parameter."""
    contract = validate_contract(contract)
    import torch
    architecture = getattr(model, "architecture_version", None)
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("Original model architecture identity is missing")
    if contract["mode"] == "native":
        if ARCHITECTURE_SEPARATOR in architecture or hasattr(model, DIGEST_BUFFER):
            raise ValueError("Native model cannot adopt a strict sampling identity")
        return model
    suffix = ARCHITECTURE_SEPARATOR + contract["contract_sha256"]
    value = torch.tensor(list(bytes.fromhex(contract["contract_sha256"])), dtype=torch.uint8)
    if hasattr(model, DIGEST_BUFFER):
        if not architecture.endswith(suffix) or not torch.equal(getattr(model, DIGEST_BUFFER).cpu(), value):
            raise ValueError("Model already carries a different sampling identity")
        return model
    if ARCHITECTURE_SEPARATOR in architecture:
        raise ValueError("Model architecture already names another sampling identity")
    model.register_buffer(DIGEST_BUFFER, value, persistent=True)
    model.architecture_version = architecture + suffix
    model.register_load_state_dict_pre_hook(_require_model_state_sampling_identity)
    return model


def _require_model_state_sampling_identity(module, state_dict, prefix, *unused):
    """Direct state_dict loads cannot replace or omit the bound identity."""
    import torch
    architecture = getattr(module, "architecture_version", "")
    if ARCHITECTURE_SEPARATOR not in architecture:
        raise ValueError("Strict model architecture lost its sampling identity")
    encoded = architecture.rsplit(ARCHITECTURE_SEPARATOR, 1)[1]
    expected = torch.tensor(list(bytes.fromhex(encoded)), dtype=torch.uint8)
    incoming = state_dict.get(prefix + DIGEST_BUFFER)
    if (not torch.is_tensor(incoming) or incoming.dtype != torch.uint8
            or tuple(incoming.shape) != (32,)
            or not torch.equal(incoming.detach().cpu(), expected)):
        raise ValueError("Model state_dict has another or missing sampling identity")


def validate_checkpoint_identity(payload, contract):
    """Reject sampler changes before optimizer/model state is adopted.

    Returns a shallow metadata copy with the base architecture for the original
    checkpoint validator. The caller's payload and persisted digest are intact.
    """
    contract = validate_contract(contract)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Actual neural checkpoint state_dict is required")
    architecture = payload.get("architecture_version")
    if not isinstance(architecture, str):
        raise ValueError("Checkpoint architecture identity is required")
    state = payload["state_dict"]
    if contract["mode"] == "native":
        if ARCHITECTURE_SEPARATOR in architecture or DIGEST_BUFFER in state:
            raise ValueError("Strict checkpoint cannot be loaded as native sampling")
        return dict(payload)
    import torch
    suffix = ARCHITECTURE_SEPARATOR + contract["contract_sha256"]
    digest = state.get(DIGEST_BUFFER)
    expected = torch.tensor(list(bytes.fromhex(contract["contract_sha256"])), dtype=torch.uint8)
    if (not architecture.endswith(suffix) or not torch.is_tensor(digest)
            or digest.dtype != torch.uint8 or tuple(digest.shape) != (32,)
            or not torch.equal(digest.detach().cpu(), expected)):
        raise ValueError("Checkpoint sampler source/config/profile identity differs")
    if ARCHITECTURE_SEPARATOR in architecture[:-len(suffix)]:
        raise ValueError("Checkpoint has multiple sampling architecture markers")
    _check_config(payload.get("graph_config"), contract)
    result = dict(payload)
    result["architecture_version"] = architecture[:-len(suffix)]
    return result
