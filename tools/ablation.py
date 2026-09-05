#!/usr/bin/env python3
"""Train and summarize independent leave-one-level-out HierCP ablations.

The existing Full M3 checkpoint is the fixed reference. Three variants are
trained from scratch with the same graph cache, split, prototype bank, seed,
curriculum, and optimizer. Each variant freezes its removed encoder and gives
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
        "Level 0 local embeddings and Level-0 inputs to Level 1 are exact zeros; "
        "patient and population graphs remain active from raw graph features."
    ),
    "no_patient": (
        "Level 1 embeddings and Level-1 inputs to Level 2 are exact zeros; "
        "local and population graphs remain active."
    ),
    "no_population": (
        "Level 2 population embeddings are exact zeros; local and patient graphs remain active."
    ),
    "full": "Existing complete M3 hierarchy.",
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
    return {
        "mode": mode,
        "label": MODE_LABELS[mode],
        "description": MODE_DESCRIPTIONS[mode],
        "path": str(path),
        "status": "complete" if bool(payload.get("training_complete", False)) else "partial",
        "epoch": int(payload.get("epoch", payload.get("completed_epoch", 0))),
        "best_epoch": int(payload.get("best_epoch", payload.get("epoch", 0))),
        **selection,
    }


def _require_shared_assets(work: Path) -> None:
    required = [
        work / "graphs" / "config.json",
        work / "graphs" / "index.json",
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
    full = _load_checkpoint(work / "model.pt")
    assert full is not None
    if _checkpoint_mode(full) != "full":
        raise ValueError(f"Reference checkpoint is not full M3: {work / 'model.pt'}")


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
    _require_shared_assets(work)
    modes = _parse_modes(args.modes)
    config = Path(args.config).expanduser().resolve() if args.config else project / "config" / "train.json"

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
        argparse.Namespace(project=str(project), work=str(work), output=None)
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

    by_mode = {record["mode"]: record for record in records}
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
        "All ablations are trained from scratch with the same cache, split, prototype, seed, and curriculum. Each score head uses only the active hierarchy widths.",
        "",
        "| Model | Removed level | Status | Top-1 | MRR | Margin | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    removed = {
        "no_local": "Level 0 — Local",
        "no_patient": "Level 1 — Patient",
        "no_population": "Level 2 — Population",
        "full": "None",
    }
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
    from hiercp.model import ABLATION_MODES, HierarchicalPyGPlacementModel
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
    shared_initial_state: dict[str, torch.Tensor] | None = None
    hidden_dim = 16
    expected_score_blocks = {
        "full": 9,
        "no_local": 4,
        "no_patient": 7,
        "no_population": 7,
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
            local_layers=2,
            patient_layers=1,
            prototype_layers=1,
            dropout=0.0,
            dense_base_channels=4,
            dense_feature_dim=8,
            dense_batch_size=3,
            channels_last_3d=True,
            checkpoint_local_blocks=True,
            checkpoint_dense_encoder=True,
            ablation_mode=mode,
        ).to(device)
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if shared_initial_state is None:
            shared_initial_state = state
        else:
            if state.keys() != shared_initial_state.keys():
                raise RuntimeError(f"State-dict schema changed in mode={mode}")
            for key in state:
                if key.startswith("score_head."):
                    continue
                if not torch.equal(state[key], shared_initial_state[key]):
                    raise RuntimeError(
                        "Same-seed shared parameter changed outside the mode-specific "
                        f"score head for mode={mode}, key={key}"
                    )

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

        batch = collate_samples([copy.deepcopy(materialized)])
        model.train()
        output = model(batch)
        if len(output.scores) != 1 or tuple(output.scores[0].shape) != (6,):
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
            f"encoder_grads={actual_usage}"
        )

    print("[OK] Shared parameter initialization, active score widths, and frozen encoders verified")
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
