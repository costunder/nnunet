#!/usr/bin/env python3
"""Train and summarize independent leave-one-level-out HierCP ablations.

An existing Full M3 checkpoint is a reference only after its completed training
and cache/config contracts are verified. Comparisons with missing or different
contracts are explicitly unavailable. Each variant freezes its removed encoder and gives
the score head only the feature blocks from active hierarchy levels:

- no_local:       remove Level 0 only; Levels 1 and 2 remain active
- no_patient:     remove Level 1 only; Levels 0 and 2 remain active
- no_population:  remove Level 2 only; Levels 0 and 1 remain active

Direct level contributions are measured as Full M3 minus the corresponding
one-factor ablation. They are leave-one-component-out effects, not percentages.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ABLATION_ORDER = ("no_local", "no_patient", "no_population", "full")
TRAINABLE_MODES = ("no_local", "no_patient", "no_population")
MODE_LABELS = {
    "no_local": "M3 w/o Level 0 — Local",
    "no_patient": "M3 w/o Level 1 — Patient",
    "no_population": "M3 w/o Level 2 — Population",
    "full": "Full M3 — Levels 0+1+2",
}
MODE_DESCRIPTIONS = {
    "no_local": (
        "Level 0 encoder, readout features, and Level-0 projection columns are removed; "
        "patient and population graphs remain active from raw graph features."
    ),
    "no_patient": (
        "Level 1 encoder, readout features, and Level-1 projection columns are removed; "
        "local and population graphs remain active."
    ),
    "no_population": (
        "Level 2 encoder and conditioned population readout are removed; local and patient graphs remain active."
    ),
    "full": "Full M3 reference; completion and comparability are verified separately.",
}


def _resolve_project(value: str | None) -> Path:
    if value:
        root = Path(value).expanduser().resolve()
    else:
        root = PROJECT_ROOT
    if not (root / "hiercp" / "model.py").is_file():
        raise FileNotFoundError(f"HierCP project not found: {root}")
    return root


def _resolve_work(project: Path, value: str | None) -> Path:
    return (
        Path(value).expanduser().resolve()
        if value
        else (project / "work" / "full").resolve()
    )


def _parse_modes(value: str) -> tuple[str, ...]:
    text = str(value).strip().lower()
    if text in {"all", "ablations"}:
        return TRAINABLE_MODES
    modes: list[str] = []
    for token in text.split(","):
        mode = token.strip()
        if not mode:
            continue
        if mode == "full":
            raise ValueError(
                "Full M3 is the existing reference and is not retrained by this tool."
            )
        if mode not in TRAINABLE_MODES:
            raise ValueError(
                f"Unknown mode={mode!r}; expected one of {TRAINABLE_MODES} or all"
            )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("No ablation mode was selected")
    return tuple(modes)


def _checkpoint_path(work: Path, mode: str) -> Path:
    if mode == "full":
        return work / "model.pt"
    return work / "ablation_independent" / mode / "model.pt"


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    from hiercp.tensor import torch_load_compat

    payload = torch_load_compat(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def _checkpoint_mode(payload: dict[str, Any]) -> str:
    kwargs = payload.get("model_kwargs")
    if not isinstance(kwargs, dict):
        return "unknown"
    return str(kwargs.get("ablation_mode", "full"))


def _selection(payload: dict[str, Any]) -> dict[str, float | None]:
    raw = payload.get("selection")
    if not isinstance(raw, dict):
        raw = payload.get("best_selection")
    if not isinstance(raw, dict):
        raw = {}

    def number(key: str, fallback: Any = None) -> float | None:
        try:
            value = float(raw.get(key, fallback))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    return {
        "mrr": number("mrr", payload.get("best_mrr")),
        "acc": number("acc"),
        "margin": number("margin"),
        "ranking": number("ranking"),
        "consistency": number("consistency"),
    }


def _comparison_contract(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract verifiable contracts; missing legacy metadata is not equality."""
    from hiercp.contracts import ARCHITECTURE_VERSION, GEOMETRY_CONTRACT
    from hiercp.schema import UPPER_FEATURE_POLICY

    issues: list[str] = []
    contract: dict[str, Any] = {}
    expected = {"architecture_version": ARCHITECTURE_VERSION,
                "geometry_contract": GEOMETRY_CONTRACT,
                "upper_feature_policy": UPPER_FEATURE_POLICY}
    for key, value in expected.items():
        if payload.get(key) != value:
            issues.append(f"missing_or_incompatible:{key}")
    for key in ("method", "framework", "architecture_version", "geometry_contract",
                "upper_feature_policy", "model_kwargs", "graph_config", "ct_clip",
                "prototype_training_cases", "prototype_fingerprint", "validation_policy",
                "training_signature", "cache_publication", "runtime"):
        if key not in payload:
            issues.append(f"missing:{key}")
        else:
            contract[key] = copy.deepcopy(payload[key])
    if payload.get("training_complete") is not True:
        issues.append("training_not_complete")
    epochs = payload.get("target_epochs")
    completed = payload.get("completed_epoch")
    if (type(epochs) is not int or epochs < 1 or type(completed) is not int
            or completed != epochs):
        issues.append("completed_epoch_does_not_match_target")
    connectivity = payload.get("gradient_connectivity")
    if (not isinstance(connectivity, dict) or connectivity.get("verified") is not True
            or connectivity.get("missing_parameters") != []):
        issues.append("gradient_connectivity_not_verified")
    elif (not isinstance(connectivity.get("connected_parameters"), list)
          or not all(isinstance(name, str) for name in connectivity["connected_parameters"])
          or connectivity.get("connected_parameter_count") != len(connectivity["connected_parameters"])
          or connectivity.get("expected_parameter_count") != len(connectivity["connected_parameters"])
          or len(set(connectivity["connected_parameters"])) != len(connectivity["connected_parameters"])):
        issues.append("gradient_connectivity_counts_inconsistent")
    signature = contract.get("training_signature")
    mode = _checkpoint_mode(payload)
    if not isinstance(signature, dict):
        issues.append("invalid:training_signature")
    else:
        required = ("format", "run_mode", "target_epochs", "seed", "batch_setting",
                    "batch_size", "worker_setting", "num_workers",
                    "gradient_accumulation_setting", "gradient_accumulation_steps",
                    "target_effective_batch_size", "resolved_effective_batch_size",
                    "calibration_resource_fingerprint", "consistency_weight", "optimizer",
                    "scheduler", "amp", "grad_clip", "deterministic", "allow_tf32",
                    "curriculum", "train_cache_files", "val_cache_files")
        issues.extend(f"missing:training_signature.{key}" for key in required if key not in signature)
        if signature.get("format") != "hiercp_training_signature_v1":
            issues.append("invalid:training_signature.format")
        if signature.get("target_epochs") != epochs:
            issues.append("training_signature.target_epochs_mismatch")
        for key in ("target_epochs", "batch_size", "gradient_accumulation_steps", "resolved_effective_batch_size"):
            if type(signature.get(key)) is not int or signature[key] < 1:
                issues.append(f"invalid:training_signature.{key}")
        for key in ("train_cache_files", "val_cache_files"):
            if not isinstance(signature.get(key), list) or not signature[key]:
                issues.append(f"missing_cohort:training_signature.{key}")
        if signature.get("ablation_mode", "full") != mode:
            issues.append("training_signature.ablation_mode_mismatch")
        role = signature.get("run_mode")
        role_valid = isinstance(role, str) and (
            (mode == "full" and role in {"production", "benchmark"})
            or (mode in TRAINABLE_MODES and role == "ablation")
        )
        if not role_valid:
            issues.append("unsupported_full_or_ablation_run_role")
        else:
            # These are the explicit experiment roles, not optimization choices.
            # Batch/effective-batch, epochs, seeds, cache membership, AMP, runtime,
            # and every other signature field remain strictly compared.
            signature["run_mode"] = "verified_full_vs_ablation_role"
        signature.pop("ablation_mode", None)
    kwargs = contract.get("model_kwargs")
    if not isinstance(kwargs, dict):
        issues.append("invalid:model_kwargs")
    else:
        kwargs.pop("ablation_mode", None)
        for key in ("hidden_dim", "heads", "local_layers", "patient_layers", "prototype_layers",
                    "dense_base_channels", "dense_feature_dim"):
            if type(kwargs.get(key)) is not int or kwargs[key] < 1:
                issues.append(f"invalid:model_kwargs.{key}")
    graph = contract.get("graph_config")
    if not isinstance(graph, dict) or graph.get("geometry_contract") != GEOMETRY_CONTRACT:
        issues.append("invalid:graph_config.geometry_contract")
    clip = contract.get("ct_clip")
    if isinstance(clip, (tuple, list)) and len(clip) == 2:
        contract["ct_clip"] = list(clip)  # Same two values; normalize serialization only.
    else:
        issues.append("invalid:ct_clip")
    fingerprint = contract.get("prototype_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        issues.append("invalid:prototype_fingerprint")
    if not isinstance(contract.get("prototype_training_cases"), list) or not contract["prototype_training_cases"]:
        issues.append("missing:prototype_training_cases")
    if not isinstance(contract.get("validation_policy"), dict) or not contract["validation_policy"]:
        issues.append("invalid:validation_policy")
    if not isinstance(contract.get("runtime"), dict) or not contract["runtime"]:
        issues.append("invalid:runtime")
    publication = contract.get("cache_publication")
    for key in ("config_sha256", "index_sha256", "complete_sha256"):
        value = publication.get(key) if isinstance(publication, dict) else None
        if (not isinstance(value, str) or len(value) != 64
                or any(letter not in "0123456789abcdef" for letter in value)):
            issues.append(f"missing_or_invalid:cache_publication.{key}")
    if any(value is None for value in _selection(payload).values()):
        issues.append("incomplete_or_nonfinite_selection_metrics")
    return contract, sorted(set(issues))


def _contract_differences(first: Any, second: Any, prefix: str = "") -> list[str]:
    if isinstance(first, dict) and isinstance(second, dict):
        result: list[str] = []
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                result.append(path)
            else:
                result.extend(_contract_differences(first[key], second[key], path))
        return result
    return [] if first == second else [prefix]


def _pairwise_comparability(records: dict[str, dict[str, Any]], ablated: str) -> dict[str, Any]:
    reasons: list[str] = []
    for mode in ("full", ablated):
        record = records.get(mode, {})
        if record.get("status") != "complete":
            reasons.append(f"{mode}:status={record.get('status', 'missing')}")
        reasons.extend(f"{mode}:{issue}" for issue in record.get("comparison_issues", []))
        if not isinstance(record.get("comparison_contract"), dict):
            reasons.append(f"{mode}:missing_comparison_contract")
    if not reasons:
        differences = _contract_differences(records["full"]["comparison_contract"],
                                            records[ablated]["comparison_contract"])
        reasons.extend(f"contract_mismatch:{name}" for name in differences)
    return {"status": "incomparable" if reasons else "comparable", "reasons": reasons}


def _checkpoint_record(work: Path, mode: str) -> dict[str, Any]:
    path = _checkpoint_path(work, mode)
    payload = _load_checkpoint(path)
    if payload is None:
        return {
            "mode": mode,
            "label": MODE_LABELS[mode],
            "path": str(path),
            "status": "missing",
        }
    recorded_mode = _checkpoint_mode(payload)
    if recorded_mode != mode:
        raise ValueError(
            f"Checkpoint mode mismatch at {path}: recorded={recorded_mode!r}, "
            f"expected={mode!r}"
        )
    selection = _selection(payload)
    contract, issues = _comparison_contract(payload)
    return {
        "mode": mode,
        "label": MODE_LABELS[mode],
        "description": MODE_DESCRIPTIONS[mode],
        "path": str(path),
        "status": "complete" if not issues else (
            "partial" if payload.get("training_complete") is not True else "incomparable"
        ),
        "epoch": int(payload.get("completed_epoch", payload.get("epoch", 0))),
        "best_epoch": int(payload.get("best_epoch", payload.get("epoch", 0))),
        "comparison_contract": contract,
        "comparison_issues": issues,
        "original_run_mode": (
            payload["training_signature"].get("run_mode")
            if isinstance(payload.get("training_signature"), dict) else None
        ),
        **selection,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_shared_assets(work: Path) -> dict[str, Any]:
    required = [
        work / "graphs" / "config.json",
        work / "graphs" / "index.json",
        work / "graphs" / "complete.json",
        work / "prototype.pt",
        work / "split.json",
        work / "model.pt",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required full-M3/shared assets are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    full = _checkpoint_record(work, "full")
    if full["status"] != "complete":
        raise ValueError("Full reference is not verified/comparable: " + ", ".join(full["comparison_issues"]))
    contract = full["comparison_contract"]
    current_publication = {
        f"{name}_sha256": _sha256_file(work / "graphs" / f"{name}.json")
        for name in ("config", "index", "complete")
    }
    if contract["cache_publication"] != current_publication:
        raise ValueError("Full reference cache publication differs from current config/index/complete SHA256")
    cache_config = json.loads((work / "graphs" / "config.json").read_text(encoding="utf-8"))
    split = json.loads((work / "split.json").read_text(encoding="utf-8"))
    if cache_config.get("prototype_artifact_sha256") != _sha256_file(work / "prototype.pt"):
        raise ValueError("Current prototype artifact differs from the bound cache publication")
    if contract["prototype_fingerprint"] != cache_config.get("prototype_fingerprint"):
        raise ValueError("Full reference prototype fingerprint differs from current cache")
    if contract["graph_config"] != cache_config.get("graph_config"):
        raise ValueError("Full reference graph configuration differs from current cache")
    for name in ("train", "val"):
        if sorted(split.get(name, [])) != sorted(cache_config.get(f"{name}_case_ids", [])):
            raise ValueError(f"Current split.{name} differs from the bound cache cohort")
    return full


def _reference_request_issues(reference: dict[str, Any], config: dict[str, Any], args) -> list[str]:
    from hiercp.schema import graph_config_from_dict

    contract = reference["comparison_contract"]
    training = config["training"]
    signature = contract["training_signature"]
    requested_model = dict(config["model"])
    requested_model.pop("ablation_mode", None)
    expected = {
        "seed": int(args.seed if args.seed is not None else config["seed"]),
        "target_epochs": int(args.epochs if args.epochs is not None else training["epochs"]),
        "batch_setting": args.batch_size if args.batch_size is not None else training["batch_size"],
        "worker_setting": args.num_workers if args.num_workers is not None else training["num_workers"],
        "gradient_accumulation_setting": training["gradient_accumulation_steps"],
        "target_effective_batch_size": training["target_effective_batch_size"],
        "consistency_weight": training["consistency_weight"],
        "grad_clip": training["grad_clip"],
    }
    issues = [f"training_signature.{key}" for key, value in expected.items() if signature.get(key) != value]
    issues.extend(_contract_differences(contract["model_kwargs"], requested_model, "model_kwargs"))
    issues.extend(_contract_differences(contract["graph_config"], graph_config_from_dict(config["graph"]).to_dict(), "graph_config"))
    if contract["ct_clip"] != config["ct_clip"]:
        issues.append("ct_clip")
    for key in ("lr", "weight_decay"):
        if signature["optimizer"].get(key) != training[key]:
            issues.append(f"optimizer.{key}")
    for key, value in signature["curriculum"].items():
        if training.get(key) != value:
            issues.append(f"curriculum.{key}")
    if contract["validation_policy"].get("epoch") != training["fixed_validation_epoch"]:
        issues.append("validation_policy.epoch")
    if ("metric_precision" in contract["validation_policy"]
            and contract["validation_policy"]["metric_precision"] != training["checkpoint_metric_precision"]):
        issues.append("validation_policy.metric_precision")
    return sorted(set(issues))


def _run(command: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=str(cwd), env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}"
        )


def command_train(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    work = _resolve_work(project, args.work)
    reference = _require_shared_assets(work)
    modes = _parse_modes(args.modes)
    config = Path(args.config).expanduser().resolve() if args.config else project / "config" / "train.json"
    requested = json.loads(config.read_text(encoding="utf-8"))
    mismatches = _reference_request_issues(reference, requested, args)
    if mismatches:
        raise ValueError("Ablation request differs from completed Full reference: " + ", ".join(mismatches))

    for mode in modes:
        checkpoint = _checkpoint_path(work, mode)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "hiercp.pipeline",
            "train",
            "--run-mode",
            "ablation",
            "--config",
            str(config),
            "--cache-dir",
            str(work / "graphs"),
            "--prototype-bank",
            str(work / "prototype.pt"),
            "--checkpoint",
            str(checkpoint),
            "--device",
            str(args.device),
            "--ablation-mode",
            mode,
        ]
        if args.epochs is not None:
            command += ["--epochs", str(int(args.epochs))]
        if args.batch_size is not None:
            command += ["--batch-size", str(int(args.batch_size))]
        if args.num_workers is not None:
            command += ["--num-workers", str(int(args.num_workers))]
        if args.seed is not None:
            command += ["--seed", str(int(args.seed))]
        if args.overwrite:
            command.append("--overwrite")
        print(f"\n=== {MODE_LABELS[mode]} ===", flush=True)
        _run(command, cwd=project)

    command_summarize(
        argparse.Namespace(project=str(project), work=str(work), output=None, overwrite=bool(args.overwrite))
    )


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _complete_records(work: Path) -> list[dict[str, Any]]:
    return [_checkpoint_record(work, mode) for mode in ABLATION_ORDER]


def command_status(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    work = _resolve_work(project, args.work)
    print("HierCP M3-aligned ablation status")
    print(f"  project: {project}")
    print(f"  work:    {work}")
    for record in _complete_records(work):
        print(
            f"  {record['mode']:<14} {record['status']:<8} "
            f"mrr={_fmt(record.get('mrr'))} "
            f"acc={_fmt(record.get('acc'))} "
            f"margin={_fmt(record.get('margin'))}"
        )


def _difference(
    records: dict[str, dict[str, Any]], metric: str, ablated: str
) -> float | None:
    if _pairwise_comparability(records, ablated)["status"] != "comparable":
        return None
    try:
        full_value = float(records["full"][metric])
        ablated_value = float(records[ablated][metric])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(full_value) or not math.isfinite(ablated_value):
        return None
    return full_value - ablated_value


def command_summarize(args: argparse.Namespace) -> None:
    project = _resolve_project(args.project)
    work = _resolve_work(project, args.work)
    records = _complete_records(work)
    output = (
        Path(args.output).expanduser().resolve()
        if getattr(args, "output", None)
        else work / "ablation_independent" / "summary"
    )
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ("summary.json", "summary.csv", "comparison.md")]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not getattr(args, "overwrite", False):
        raise FileExistsError("Ablation reports already exist; use a new output directory or explicit --overwrite: " + ", ".join(existing))

    by_mode = {record["mode"]: record for record in records}
    comparisons = {mode: _pairwise_comparability(by_mode, mode) for mode in TRAINABLE_MODES}
    contributions = {
        "level0_local": {
            metric: _difference(by_mode, metric, "no_local")
            for metric in ("mrr", "acc", "margin")
        },
        "level1_patient": {
            metric: _difference(by_mode, metric, "no_patient")
            for metric in ("mrr", "acc", "margin")
        },
        "level2_population": {
            metric: _difference(by_mode, metric, "no_population")
            for metric in ("mrr", "acc", "margin")
        },
    }

    payload = {
        "format": "hiercp_m3_independent_leave_one_level_out_v2",
        "work": str(work),
        "records": records,
        "comparability": comparisons,
        "direct_level_contributions": contributions,
        "definition": {
            "level0_local": "Full M3 - M3 w/o Level 0",
            "level1_patient": "Full M3 - M3 w/o Level 1",
            "level2_population": "Full M3 - M3 w/o Level 2",
        },
        "interpretation": (
            "Each value is a one-factor leave-one-level-out effect under the full M3 "
            "architecture. It is not an independent causal percentage and interactions "
            "between levels may remain."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    columns = [
        "mode", "label", "status", "epoch", "best_epoch",
        "mrr", "acc", "margin", "ranking", "consistency", "path",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in columns})

    lines = [
        "# HierCP Full-M3 Independent Level Ablation",
        "",
        "Each ablation removes exactly one level from Full M3 while keeping the other two active.",
        "A numerical contribution is reported only for completed checkpoints with matching verified cache, split, prototype, model/geometry, seed, curriculum and resolved optimization contracts.",
        "The documented Full production/benchmark versus variant ablation execution roles, and the intended removed-level mode, are the only normalized differences. Missing legacy metadata or different resolved batch sizes are incomparable.",
        "",
    ]
    removed = {
        "no_local": "Level 0 — Local",
        "no_patient": "Level 1 — Patient",
        "no_population": "Level 2 — Population",
        "full": "None",
    }
    lines.extend(["", "## Comparability", ""])
    for mode, comparison in comparisons.items():
        reasons = "; ".join(comparison["reasons"]) or "completed contracts match"
        lines.append(f"- {mode}: {comparison['status']} — {reasons}")
    lines.extend(["", "| Model | Removed level | Status | Top-1 | MRR | Margin | Best epoch |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for record in records:
        lines.append(
            f"| {record['label']} | {removed[record['mode']]} | {record['status']} | "
            f"{_fmt(record.get('acc'))} | {_fmt(record.get('mrr'))} | "
            f"{_fmt(record.get('margin'))} | {record.get('best_epoch', '—')} |"
        )
    lines += [
        "",
        "## Direct contribution of each level",
        "",
        "| Level | Definition | ΔTop-1 | ΔMRR | ΔMargin |",
        "|---|---|---:|---:|---:|",
        f"| Level 0 — Local | Full − w/o L0 | {_fmt(contributions['level0_local']['acc'])} | {_fmt(contributions['level0_local']['mrr'])} | {_fmt(contributions['level0_local']['margin'])} |",
        f"| Level 1 — Patient | Full − w/o L1 | {_fmt(contributions['level1_patient']['acc'])} | {_fmt(contributions['level1_patient']['mrr'])} | {_fmt(contributions['level1_patient']['margin'])} |",
        f"| Level 2 — Population | Full − w/o L2 | {_fmt(contributions['level2_population']['acc'])} | {_fmt(contributions['level2_population']['mrr'])} | {_fmt(contributions['level2_population']['margin'])} |",
        "",
        "> Positive values mean Full M3 performed better when that level was present.",
        "> These are leave-one-level-out effects, not percentage shares; level interactions are not separated.",
        "",
    ]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[OK] summary: {output}")


def _module_has_nonzero_grad(module: Any) -> bool:
    import torch

    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool(torch.any(parameter.grad != 0))
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def command_self_test(args: argparse.Namespace) -> None:
    import torch

    from hiercp.data import collate_samples
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.model import ABLATION_MODES, MODEL_ARCHITECTURE_VERSION, HierarchicalPyGPlacementModel
    from hiercp.schema import UPPER_RAW_DIM
    from hiercp.tensor import resolve_device
    from tools.smoke import _assert_sampled_views, _canonical_sample

    expected_modes = ("full", "no_local", "no_patient", "no_population")
    if tuple(ABLATION_MODES) != expected_modes:
        raise RuntimeError(
            f"Unexpected model ablation modes: {ABLATION_MODES}; expected={expected_modes}"
        )

    device = resolve_device(args.device)
    canonical, config = _canonical_sample()
    materialized, _ = _assert_sampled_views(canonical, config)
    hidden_dim = 16
    expected_score_blocks = {
        "full": 12,
        "no_local": 7,
        "no_patient": 9,
        "no_population": 8,
    }
    expected_usage = {
        "full": (True, True, True),
        "no_local": (False, True, True),
        "no_patient": (True, False, True),
        "no_population": (True, True, False),
    }

    for mode in expected_modes:
        torch.manual_seed(20260831)
        model = HierarchicalPyGPlacementModel(
            hidden_dim=hidden_dim,
            heads=4,
            local_layers=3,
            patient_layers=2,
            prototype_layers=2,
            dropout=0.0,
            dense_base_channels=4,
            dense_feature_dim=8,
            dense_batch_size=3,
            channels_last_3d=True,
            checkpoint_local_blocks=True,
            checkpoint_dense_encoder=True,
            ablation_mode=mode,
        ).to(device)
        # v3 removes disabled-level projection columns and query inputs, so
        # ablations intentionally have different state schemas/RNG consumption.
        # Equal seeds are reproducibility, not a claim of identical shared init.
        if model.architecture_version != MODEL_ARCHITECTURE_VERSION:
            raise RuntimeError(f"Unexpected architecture for {mode}")

        expected_score_input = hidden_dim * expected_score_blocks[mode] + UPPER_RAW_DIM
        if model.score_input_dim != expected_score_input:
            raise RuntimeError(
                f"Score input width mismatch for mode={mode}: "
                f"actual={model.score_input_dim}, expected={expected_score_input}"
            )
        first_score_layer = model.score_head[0]
        if int(first_score_layer.in_features) != expected_score_input:
            raise RuntimeError(
                f"Score-head layer width mismatch for mode={mode}: "
                f"actual={first_score_layer.in_features}, expected={expected_score_input}"
            )

        encoders = (
            model.local_encoder,
            model.patient_encoder,
            model.prototype_encoder,
        )
        actual_trainability: list[bool] = []
        for encoder in encoders:
            flags = {bool(parameter.requires_grad) for parameter in encoder.parameters()}
            if len(flags) != 1:
                raise RuntimeError(
                    f"Mixed trainability inside encoder for mode={mode}: {flags}"
                )
            actual_trainability.append(flags.pop())
        if tuple(actual_trainability) != expected_usage[mode]:
            raise RuntimeError(
                f"Encoder trainability mismatch for mode={mode}: "
                f"actual={tuple(actual_trainability)}, expected={expected_usage[mode]}"
            )
        trainable_ids = {id(parameter) for parameter in model.trainable_parameters()}
        expected_trainable_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        if trainable_ids != expected_trainable_ids:
            raise RuntimeError(f"trainable_parameters() contract mismatch for mode={mode}")

        batch = collate_samples([copy.deepcopy(materialized), copy.deepcopy(materialized)])
        optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.01)
        before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        model.train()
        output = model(batch)
        if len(output.scores) != 2 or any(tuple(score.shape) != (6,) for score in output.scores):
            raise RuntimeError(f"Bad score shape for {mode}: {output.scores}")
        ranking, _ = curriculum_ranking_loss(
            output.scores,
            batch.difficulty_list(),
            epoch=30,
            config=CurriculumConfig(
                easy_epochs=2,
                inter_epochs=4,
                intra_epochs=6,
                model_mine_start_epoch=7,
            ),
        )
        loss = ranking + 0.1 * output.consistency
        loss.backward()
        empty_lesions = int(batch.patient_batch["lesion"].num_nodes) == 0
        conditional_missing: list[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise RuntimeError(f"Disabled parameter received a gradient: {mode}:{name}")
                continue
            if parameter.grad is None:
                if empty_lesions and "lesion" in name:
                    conditional_missing.append(name)
                    continue
                raise RuntimeError(f"Disconnected active parameter: {mode}:{name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Non-finite active gradient: {mode}:{name}")
        if not _module_has_nonzero_grad(model.score_head):
            raise RuntimeError(f"No score-head gradient in mode={mode}")
        actual_usage = (
            _module_has_nonzero_grad(model.local_encoder),
            _module_has_nonzero_grad(model.patient_encoder),
            _module_has_nonzero_grad(model.prototype_encoder),
        )
        if actual_usage != expected_usage[mode]:
            raise RuntimeError(
                f"Encoder-gradient usage mismatch for mode={mode}: "
                f"actual={actual_usage}, expected={expected_usage[mode]}"
            )
        if mode == "no_local" and float(output.consistency.detach().cpu()) != 0.0:
            raise RuntimeError("no_local must have zero consistency loss")
        optimizer.step()
        for level, removed in (("local", "no_local"), ("patient", "no_patient"), ("prototype", "no_population")):
            if mode == removed:
                continue
            encoder = getattr(model, f"{level}_encoder")
            for block_index in range(len(encoder.blocks)):
                prefix = f"{level}_encoder.blocks.{block_index}."
                if not any(not torch.equal(before[name], parameter.detach())
                           for name, parameter in model.named_parameters() if name.startswith(prefix)):
                    raise RuntimeError(f"No optimizer update in {mode}:{prefix}")

        model.eval()
        inference_batch = collate_samples([copy.deepcopy(materialized)])
        with torch.inference_mode():
            normal_scores = model(copy.deepcopy(inference_batch)).scores[0].float()
            chunked_scores = model.score_inference_chunked(
                copy.deepcopy(inference_batch), local_chunk_size=2
            )[0].float()
        if not torch.allclose(normal_scores, chunked_scores, rtol=1e-5, atol=1e-5):
            error = float((normal_scores - chunked_scores).abs().max().cpu())
            raise RuntimeError(
                f"Chunked/full mismatch for mode={mode}: max_error={error}"
            )
        print(
            f"[OK] mode={mode} score_shape={tuple(normal_scores.shape)} "
            f"encoder_grads={actual_usage} physical_debug_batch=2 "
            f"conditionally_absent_lesion_parameters={len(conditional_missing)}"
        )

    print("[OK] Versioned active score widths, per-parameter gradient presence/finiteness, and per-block updates verified")
    print("[OK] Independent one-level-out forward/backward and chunked inference smoke complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train one or more ablations")
    train.add_argument("--work", default=None)
    train.add_argument("--config", default=None)
    train.add_argument("--modes", default="all")
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(func=command_train)

    status = commands.add_parser("status", help="Show checkpoint status")
    status.add_argument("--work", default=None)
    status.set_defaults(func=command_status)

    summarize = commands.add_parser("summarize", help="Write comparison tables")
    summarize.add_argument("--work", default=None)
    summarize.add_argument("--output", default=None)
    summarize.add_argument("--overwrite", action="store_true", help="Explicitly replace the three report files")
    summarize.set_defaults(func=command_summarize)

    self_test = commands.add_parser("self-test", help="Run synthetic CUDA/CPU smoke")
    self_test.add_argument("--device", default="cpu")
    self_test.set_defaults(func=command_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
