"""Exact-v1, one-factor experiment contracts; no model or training side effects.

The source-anchor curriculum in these stages is distinct from v2.2's observed
P/U target.  A stage transition cannot relabel either experiment's observations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping
from zipfile import ZipFile


V1_REVISION = "74dcc2cf03d2d40d1f582223321d96004333f661"
V1_ARCHIVE_SHA256 = "5157bafe641e9189824826532a3b055ea560f5c5dfc299374842a3bd1d20a22e"
V1_MANIFEST_SHA256 = "82813025615bec9a1294974fedf36077d45447d0317cbd1b0733d4d6fab612f3"
TARGET_CONTRACT = "v1_source_original_anchor_vs_curriculum_candidates"
LOCAL_SAMPLING_MODES = ("native", "strict_nested")
LOCAL_SAMPLING_ROLES = (
    "tumor_surface", "tumor_interior", "source_context",
    "source_liver_surface", "target_context", "target_liver_surface",
)


def local_sampling_spec(mode: str, profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Require an explicit selection policy, without changing the model stages."""
    if mode not in LOCAL_SAMPLING_MODES:
        raise ContractError(f"Unknown local sampling mode {mode!r}")
    if mode == "native":
        if profile is not None:
            raise ContractError("Native sampling does not accept a thinning profile")
        return {"mode": mode, "profile": None,
                "runtime_rng_boundary": "worker_preflight_capture_restore_v1"}
    if not isinstance(profile, Mapping) or set(profile) != set(LOCAL_SAMPLING_ROLES):
        raise ContractError("Strict nested sampling requires all six explicit role seed budgets")
    if any(type(profile[key]) is not int or profile[key] <= 0 for key in LOCAL_SAMPLING_ROLES):
        raise ContractError("Strict nested role seed budgets must be positive integers")
    return {"mode": mode, "profile": {key: profile[key] for key in LOCAL_SAMPLING_ROLES},
            "runtime_rng_boundary": "worker_preflight_capture_restore_v1"}


class ContractError(ValueError):
    """An experiment changed an unapproved factor or evidence is incomparable."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    predecessor: str | None
    changed_factor: str
    description: str
    local_features: str
    local_operator: str
    activation_storage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STAGES = MappingProxyType({
    "v1.0": StageSpec("v1.0", None, "none", "Exact preserved v1 baseline",
                     "cnn_and_handcrafted", "gat", "original_checkpointing"),
    "v1.1": StageSpec("v1.1", "v1.0", "activation_storage",
                     "Retain CNN activations; keep local graph checkpointing and identical mathematics",
                     "cnn_and_handcrafted", "gat", "cnn_retained_local_checkpointed"),
    "v1.2": StageSpec("v1.2", "v1.1", "local_features",
                     "Remove only handcrafted inputs of the local node projection",
                     "cnn_only", "gat", "cnn_retained_local_checkpointed"),
    "v1.3": StageSpec("v1.3", "v1.2", "local_operator",
                     "Replace only local GAT by relation-separated mean GraphSAGE",
                     "cnn_only", "relation_mean_sage", "cnn_retained_local_checkpointed"),
})


def get_stage(stage: str) -> StageSpec:
    try:
        return STAGES[stage]
    except KeyError as exc:
        raise ContractError(f"Unknown stage {stage!r}; expected {tuple(STAGES)}") from exc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ContractError("Contract values must be finite JSON values") from exc
    return _sha(data)


def _root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parents[1]


def verify_archive(root: str | Path | None = None) -> dict[str, Any]:
    """Verify the pinned manifest and every exact byte, without extracting files."""
    directory = _root(root) / "versions" / "v1"
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if _sha(manifest_bytes) != V1_MANIFEST_SHA256:
        raise ContractError("Preserved v1 manifest differs from the pinned original")
    manifest = json.loads(manifest_bytes)
    archive = directory / "pipeline_v1_source.zip"
    archive_sha = _sha(archive.read_bytes())
    if (manifest.get("revision") != V1_REVISION
            or manifest.get("archive_sha256") != V1_ARCHIVE_SHA256
            or archive_sha != V1_ARCHIVE_SHA256):
        raise ContractError("Preserved v1 archive identity mismatch")
    expected = manifest["files"]
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ContractError("Archive has missing, extra, or duplicate source files")
        for info in zipped.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename:
                raise ContractError(f"Unsafe archive member {info.filename!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ContractError(f"Archive symbolic link {info.filename!r}")
            if _sha(zipped.read(info)) != expected[info.filename]:
                raise ContractError(f"Archive source hash mismatch: {info.filename}")
        corrupt = zipped.testzip()
        if corrupt is not None:
            raise ContractError(f"Archive CRC mismatch: {corrupt}")
        base = json.loads(zipped.read("config/train.json"))
    return {
        "revision": V1_REVISION,
        "archive_sha256": archive_sha,
        "manifest_sha256": _sha(manifest_bytes),
        "verified_files": len(expected),
        "base_config": base,
        "base_config_sha256": canonical_hash(base),
        "file_hashes": copy.deepcopy(expected),
    }


def _base(root: str | Path | None = None) -> dict[str, Any]:
    return verify_archive(root)["base_config"]


def config_diff(before: Any, after: Any, path: str = "") -> dict[str, dict[str, Any]]:
    """Leaf changes, including added/removed keys (no silent schema relaxation)."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed: dict[str, dict[str, Any]] = {}
        for key in sorted(set(before) | set(after)):
            here = f"{path}.{key}" if path else key
            if key not in before:
                changed[here] = {"kind": "added", "after": after[key]}
            elif key not in after:
                changed[here] = {"kind": "removed", "before": before[key]}
            else:
                changed.update(config_diff(before[key], after[key], here))
        return changed
    # bool is an int subclass; type changes must not be hidden by Python equality.
    if type(before) is not type(after) or before != after:
        return {path: {"kind": "changed", "before": before, "after": after}}
    return {}


def make_stage_config(base: Mapping[str, Any], stage: str,
                      *, root: str | Path | None = None) -> dict[str, Any]:
    spec = get_stage(stage)
    original = _base(root)
    delta = config_diff(original, dict(base))
    if delta:
        raise ContractError(f"Base must be exact archived v1; altered keys: {tuple(delta)}")
    result = copy.deepcopy(original)
    if spec.activation_storage == "cnn_retained_local_checkpointed":
        result["model"]["checkpoint_dense_encoder"] = False
    return result


def resolve_execution_config(config: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze measured auto calibration; do not tune each stage independently.

    The suite must use the baseline's same calibration identity for all stages.
    Resolving auto to a measured integer is recorded separately from stage design.
    """
    if not isinstance(calibration, Mapping):
        raise ContractError("Execution calibration must be explicitly supplied")
    batch = calibration.get("selected_batch_size")
    workers = calibration.get("selected_num_workers")
    if type(batch) is not int or batch <= 0:
        raise ContractError("Calibration selected_batch_size must be a measured positive integer")
    if type(workers) is not int or workers < 0:
        raise ContractError("Calibration selected_num_workers must be a measured nonnegative integer")
    result = copy.deepcopy(dict(config))
    result["training"]["batch_size"] = batch
    result["training"]["num_workers"] = workers
    return result


def validate_stage_config(config: Mapping[str, Any], stage: str,
                          *, root: str | Path | None = None,
                          execution_lock: Mapping[str, Any] | None = None) -> None:
    expected = make_stage_config(_base(root), stage, root=root)
    if execution_lock is not None:
        expected = resolve_execution_config(expected, execution_lock)
    delta = config_diff(expected, dict(config))
    if delta:
        raise ContractError(f"{stage} contains undeclared changes: {tuple(delta)}")


def validate_transition(before_stage: str, before: Mapping[str, Any],
                        after_stage: str, after: Mapping[str, Any],
                        *, root: str | Path | None = None,
                        execution_lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Require the declared predecessor and precisely one experiment factor."""
    old, new = get_stage(before_stage), get_stage(after_stage)
    if new.predecessor != old.name:
        raise ContractError(f"{after_stage} must be compared to {new.predecessor}, not {before_stage}")
    validate_stage_config(before, before_stage, root=root, execution_lock=execution_lock)
    validate_stage_config(after, after_stage, root=root, execution_lock=execution_lock)
    fields = ("local_features", "local_operator", "activation_storage")
    altered = [key for key in fields if getattr(old, key) != getattr(new, key)]
    if altered != [new.changed_factor]:
        raise ContractError(f"Stage registry changes multiple factors: {altered}")
    return {"from": old.name, "to": new.name, "changed_factor": new.changed_factor,
            "config_diff": config_diff(dict(before), dict(after)),
            "unchanged_target_contract": TARGET_CONTRACT,
            "source_archive_sha256": V1_ARCHIVE_SHA256}


def _identity(identity: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        raise ContractError(f"{name} needs a non-empty immutable identity")
    result = copy.deepcopy(dict(identity))
    canonical_hash(result)
    return result


def make_run_contract(stage: str, config: Mapping[str, Any], *,
                      cache_identity: Mapping[str, Any],
                      source_identity: Mapping[str, Any],
                      evaluation_identity: Mapping[str, Any],
                      physical_batch_size: int, debug: bool,
                      root: str | Path | None = None,
                      execution_lock: Mapping[str, Any] | None = None,
                      sampling_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind exact resume to one stage, config, cache, source, and physical batch."""
    spec = get_stage(stage)
    validate_stage_config(config, stage, root=root, execution_lock=execution_lock)
    if type(physical_batch_size) is not int or physical_batch_size <= 0:
        raise ContractError("Physical batch must be the actual measured positive integer")
    if type(debug) is not bool:
        raise ContractError("debug must be explicit")
    if execution_lock is not None and physical_batch_size != execution_lock["selected_batch_size"]:
        raise ContractError("Physical batch differs from the shared baseline calibration")
    binding = {
        "format": "hiercp_v1x_run_v1", "stage": spec.to_dict(),
        "target_contract": TARGET_CONTRACT,
        "config": copy.deepcopy(dict(config)),
        "config_sha256": canonical_hash(config),
        "baseline_archive_sha256": V1_ARCHIVE_SHA256,
        "cache_identity": _identity(cache_identity, "cache"),
        "source_identity": _identity(source_identity, "source"),
        "evaluation_identity": _identity(evaluation_identity, "evaluation"),
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "debug": debug,
        "execution_lock": copy.deepcopy(dict(execution_lock)) if execution_lock is not None else None,
    }
    if sampling_contract is not None:
        binding["sampling_contract"] = _identity(sampling_contract, "local sampling")
    binding["contract_sha256"] = canonical_hash(binding)
    return binding


def validate_resume(expected: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    """No cross-stage exact resume or replacement of an existing run contract."""
    for label, value in (("expected", expected), ("saved", saved)):
        raw = dict(value)
        digest = raw.pop("contract_sha256", None)
        if digest is None or canonical_hash(raw) != digest:
            raise ContractError(f"{label} run contract hash mismatch")
    delta = config_diff(dict(expected), dict(saved))
    if delta:
        raise ContractError(f"Exact resume contract mismatch: {tuple(delta)}")


EVALUATION_IDENTITY_FIELDS = (
    "target_contract", "candidate_contract", "split_sha256",
    "candidate_ids_sha256", "mask_contract", "source_archive_sha256",
    "evaluation_source_sha256", "physical_batch_size", "effective_batch_size", "target_epochs", "denominator",
)


def _evaluation(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    get_stage(report.get("stage"))
    if report.get("complete") is not True or type(report.get("debug")) is not bool:
        raise ContractError("Only explicitly completed, scoped evaluations can be compared")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ContractError("Missing evaluation provenance")
    for field in EVALUATION_IDENTITY_FIELDS:
        if field not in evaluation or evaluation[field] in (None, "", {}):
            raise ContractError(f"Missing evaluation identity: {field}")
    if (evaluation["target_contract"] != TARGET_CONTRACT
            or evaluation["source_archive_sha256"] != V1_ARCHIVE_SHA256):
        raise ContractError("Evaluation changed v1's target or original source identity")
    denominator = evaluation["denominator"]
    if not isinstance(denominator, dict) or not denominator:
        raise ContractError("Evaluation denominators must be explicitly reported")
    if any(type(x) is not int or x <= 0 for x in denominator.values()):
        raise ContractError("Evaluation denominators must be positive integers")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ContractError("Missing measured metrics")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(f"Metric {name} must be a finite measured number")
        if name.lower() in ("mrr", "top1", "top5", "top10", "pair_win", "recall_at_1"):
            if not 0 <= value <= 1:
                raise ContractError(f"Metric {name} must lie in [0,1]")
    return evaluation, metrics


def compare_reports(baseline: Mapping[str, Any], candidate: Mapping[str, Any],
                    predecessor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Produce differences only after matching exact evaluation contracts.

    This function never labels a DEBUG result as a production promotion.  Raw
    report authors retain responsibility for measured metrics and identities.
    """
    if baseline.get("stage") != "v1.0":
        raise ContractError("The baseline report must be the exact v1.0 evaluation")
    spec = get_stage(candidate.get("stage"))
    if spec.name == 'v1.0':
        if predecessor is not None:
            raise ContractError('Native/strict-nested comparison has one matched baseline, no stage predecessor')
        sampling = []
        for label, report, mode in (('baseline', baseline, 'native'),
                                    ('candidate', candidate, 'strict_nested')):
            contract = report.get('sampling_contract')
            if not isinstance(contract, Mapping) or contract.get('mode') != mode:
                raise ContractError(f'{label} needs its explicit {mode} sampling contract')
            raw = dict(contract)
            digest = raw.pop('contract_sha256', None)
            if digest is None or canonical_hash(raw) != digest:
                raise ContractError(f'{label} sampling contract hash mismatch')
            local_sampling_spec(mode, contract.get('profile'))
            sampling.append(copy.deepcopy(dict(contract)))
        if sampling[0].get('graph_config') != sampling[1].get('graph_config'):
            raise ContractError('Sampling comparison changed original physical graph configuration')
        if any('native_reference_experiment' in report for report in (baseline, candidate)):
            reference = baseline.get('native_reference_experiment')
            if not reference or candidate.get('native_reference_experiment') != reference:
                raise ContractError('Sampling comparison belongs to another native reference experiment')
        old_identity, old_metrics = _evaluation(baseline)
        identity, metrics = _evaluation(candidate)
        delta = config_diff({key: old_identity[key] for key in EVALUATION_IDENTITY_FIELDS},
                            {key: identity[key] for key in EVALUATION_IDENTITY_FIELDS})
        if delta or baseline['debug'] != candidate['debug']:
            raise ContractError(f'Sampling evaluation is incomparable: {tuple(delta)}')
        if set(old_metrics) != set(metrics):
            raise ContractError('Sampling metric set differs; missing metrics are not zeros')
        return dict(format='hiercp_v1x_sampling_comparison_v1', stage='v1.0',
                    changed_factor='local_sampling_selection_policy', debug=candidate['debug'],
                    evaluation=copy.deepcopy(identity),
                    sampling_contracts=dict(native=sampling[0], strict_nested=sampling[1]),
                    metric_deltas={'baseline': {key: metrics[key] - old_metrics[key] for key in metrics}},
                    quality_claim='measured matched-contract deltas only; no tolerance or quality verdict inferred',
                    graph_quality_passed=False, automatic_promotion=False, training_started=False)
    if predecessor is None:
        if spec.predecessor != "v1.0":
            raise ContractError(f"Supply the {spec.predecessor} predecessor evaluation")
        predecessor = baseline
    if predecessor.get("stage") != spec.predecessor:
        raise ContractError("Incorrect predecessor evaluation")
    current_identity, current_metrics = _evaluation(candidate)
    differences: dict[str, Any] = {}
    for label, report in (("baseline", baseline), ("predecessor", predecessor)):
        identity, metrics = _evaluation(report)
        identity_delta = config_diff(
            {k: current_identity[k] for k in EVALUATION_IDENTITY_FIELDS},
            {k: identity[k] for k in EVALUATION_IDENTITY_FIELDS})
        if identity_delta or report["debug"] != candidate["debug"]:
            raise ContractError(f"{label} evaluation is incomparable: {tuple(identity_delta)}")
        if set(metrics) != set(current_metrics):
            raise ContractError(f"{label} metric set differs; missing metrics are not zeros")
        differences[label] = {k: current_metrics[k] - metrics[k] for k in metrics}
    return {"format": "hiercp_v1x_comparison_v1", "stage": spec.name,
            "predecessor": spec.predecessor, "debug": candidate["debug"],
            "evaluation": copy.deepcopy(current_identity), "metric_deltas": differences,
            "quality_claim": "measured same-contract comparison only",
            "automatic_promotion": False, "training_started": False}
