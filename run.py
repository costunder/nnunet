#!/usr/bin/env python3
"""HierCP training, generation, validation, and nnU-Net evaluation runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

from hiercp.common import discover_cases
from hiercp.split import load_case_split, make_case_split, save_case_split
from hiercp.schema import UPPER_FEATURE_POLICY
from hiercp.tensor import torch_load_compat, training_state_path

ROOT = Path(__file__).resolve().parent
TRAIN_CONFIG = ROOT / "config" / "train.json"
NNUNET_CONFIG = ROOT / "config" / "nnunet.json"
DEFAULT_WORK = ROOT / "work"
RUN_NAME = "placement"
RUN_MODES = ("production", "ablation", "benchmark", "debug")
UNAVAILABLE = "unavailable"

NNUNET_TARGETS = {
    "nnunet-check": "check",
    "nnunet-prepare": "prepare",
    "nnunet-plan": "plan",
    "nnunet-train": "train",
    "nnunet-evaluate": "evaluate",
    "nnunet-all": "all",
    "nnunet-status": "status",
}


class RunError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunError(f"Config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunError(f"Config root must be a JSON object: {path}")
    return payload


def medical_root(requested: str | None) -> Path:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    env_value = os.environ.get("MEDICAL_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend([ROOT.parent, Path.cwd()])
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if (resolved / "Data" / "image").is_dir() and (resolved / "Data" / "labels").is_dir():
            return resolved
    raise RunError("Could not locate Medical/Data/{image,labels}. Checked: " + ", ".join(checked))


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_work(work: Path, medical: Path, *, create: bool = True) -> None:
    work = work.resolve()
    if work == medical.resolve() or work == ROOT.resolve():
        raise RunError(f"Refusing protected work directory: {work}")
    for candidate in (
        medical / "Data",
        medical / "Data_aug",
        medical / "Task03_Liver",
        medical / "archive",
        medical / "liver_copy_paste_experiments",
    ):
        resolved = candidate.resolve()
        if work == resolved or is_within(work, resolved):
            raise RunError(f"Refusing protected work directory: {work}")
    project_work = (ROOT / "work").resolve()
    if is_within(work, ROOT.resolve()) and not (work == project_work or is_within(work, project_work)):
        raise RunError(f"Work inside the source tree must stay under {project_work}: {work}")
    if create:
        work.mkdir(parents=True, exist_ok=True)


def _validation_run_mode(run_mode: str) -> str:
    return "production" if run_mode == "production" else "nonproduction"


def _guard_run_contract(args: argparse.Namespace, work: Path) -> None:
    run_mode = str(args.run_mode)
    reduction_options = {
        "--max-cases": args.max_cases,
        "--case-id": args.case_id,
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--num-workers": args.num_workers,
    }
    supplied = [name for name, value in reduction_options.items() if value not in (None, [])]
    generation_target = args.target in {"generate", "all", "full"}

    if args.ablation_mode != "full" and args.run_mode != "ablation":
        raise RunError(
            f"--ablation-mode {args.ablation_mode} requires --run-mode ablation."
        )
    if args.run_mode == "ablation" and args.target != "train":
        raise RunError("--run-mode ablation is valid only for the train target.")

    if (args.skip_validation or args.skip_assemble) and not generation_target:
        raise RunError(
            "--skip-validation/--skip-assemble apply only to generate workflows."
        )

    if args.skip_validation and not args.skip_assemble:
        raise RunError(
            "--skip-validation also requires --skip-assemble; unvalidated output "
            "cannot be assembled into a dataset."
        )
    if run_mode == "production":
        if supplied:
            raise RunError(
                "Production mode forbids scale/subset overrides: "
                + ", ".join(supplied)
                + ". Use the complete configured workload or explicitly select a "
                "non-production --run-mode with a separate --work directory."
            )
        if generation_target and (args.skip_validation or args.skip_assemble):
            raise RunError(
                "Production generation requires validation and assembly; "
                "--skip-validation/--skip-assemble are non-production-only controls."
            )
        return

    if args.target in {"all", "full", *NNUNET_TARGETS.keys()}:
        raise RunError(
            f"--run-mode {run_mode} cannot target {args.target!r}; aggregate and "
            "nnU-Net final workflows are production-only. Run the individual "
            "non-production stage instead."
        )
    if args.target in {"prepare", "train", "generate"}:
        if args.work is None or work.resolve() == DEFAULT_WORK.resolve():
            raise RunError(
                f"--run-mode {run_mode} requires an explicit, separate --work "
                "directory so reduced artifacts cannot be mistaken for production."
            )


def _report_overwrite_scope(ctx: dict, args: argparse.Namespace) -> None:
    if not args.overwrite:
        return
    p = ctx["paths"]
    targets_by_stage = {
        "prepare": [p["split"], p["prototype"], p["regions"], p["graphs"]],
        "train": [
            p["checkpoint"],
            p["last"],
            p["checkpoint"].with_suffix(p["checkpoint"].suffix + ".preflight.json"),
        ],
        "generate": [p["output"], p["valid"], p["dataset"], p["meta"]],
    }
    stages: list[str] = []
    if args.target in {"prepare", "all", "full"}:
        stages.append("prepare")
    if args.target in {"train", "all", "full"}:
        stages.append("train")
    if args.target in {"generate", "all", "full"}:
        stages.append("generate")
    targets = [str(path.resolve()) for stage in stages for path in targets_by_stage[stage]]
    print(
        "[OverwriteScope] "
        + json.dumps(
            {"authorized": True, "stage_count": len(stages), "targets": targets},
            sort_keys=True,
        )
    )


def module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def execute(command: Sequence[object], *, log: Path | None = None, dry_run: bool = False) -> None:
    values = [str(value) for value in command]
    printable = " ".join(shlex.quote(value) for value in values)
    print(f"\n$ {printable}")
    if dry_run:
        return

    handle = None
    if log is not None and os.environ.get("HIERCP_DISABLE_FILE_LOG", "0").lower() not in {
        "1", "true", "yes", "on"
    }:
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8", buffering=1)
            handle.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] $ {printable}\n")
        except OSError as exc:
            print(f"[Warn] File logging disabled for {log}: {exc}")
            handle = None

    process = subprocess.Popen(
        values,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if handle is not None:
            try:
                handle.write(line)
            except OSError as exc:
                print(f"[Warn] File logging disabled: {exc}")
                try:
                    handle.close()
                except OSError as close_exc:
                    print(f"[Warn] Could not close disabled log {log}: {close_exc}")
                handle = None
    return_code = process.wait()
    if handle is not None:
        try:
            handle.close()
        except OSError as exc:
            print(f"[Warn] Could not close log {log}: {exc}")
    if return_code != 0:
        raise RunError(f"Command failed with exit code {return_code}: {printable}")


def paths(work: Path) -> dict[str, Path]:
    output = work / "output"
    checkpoint = work / "model.pt"
    return {
        "split": work / "split.json",
        "prototype": work / "prototype.pt",
        "regions": work / "regions",
        "graphs": work / "graphs",
        "checkpoint": checkpoint,
        "last": training_state_path(checkpoint),
        "log": work / "run.log",
        "output": output / "data",
        "valid": output / "valid",
        "dataset": work / "dataset",
        "meta": output / "run.json",
        "validation_report": output / "data" / "validation.csv",
        "validation_summary": output / "data" / "validation_summary.json",
    }


def ensure_split(ctx: dict, args: argparse.Namespace) -> dict[str, object]:
    target = ctx["paths"]["split"]
    cases = discover_cases(
        ctx["data"], max_cases=args.max_cases, run_mode=args.run_mode
    )
    ids = [case.case_id for case in cases]
    if target.is_file() and not args.overwrite:
        split = load_case_split(target)
        recorded = set(split["train"]) | set(split["val"])
        if recorded != set(ids):
            raise RunError("Existing split differs from the selected cases; use a new work directory or --overwrite")
        if int(split.get("seed", -1)) != ctx["seed"]:
            raise RunError("Existing split uses a different seed")
        expected = float(ctx["config"]["training"]["val_fraction"])
        if abs(float(split.get("val_fraction", -1.0)) - expected) > 1e-12:
            raise RunError("Existing split uses a different validation fraction")
        print(f"[Reuse] split train={len(split['train'])} val={len(split['val'])}: {target}")
        return split
    split = make_case_split(
        ids,
        val_fraction=float(ctx["config"]["training"]["val_fraction"]),
        seed=ctx["seed"],
    )
    if args.dry_run:
        print(
            f"[Plan] split train={len(split['train'])} val={len(split['val'])}: "
            f"would write {target}"
        )
    else:
        save_case_split(split, target)
        print(f"[OK] split train={len(split['train'])} val={len(split['val'])}: {target}")
    return split


def _human_bytes(value: int) -> str:
    amount = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _reference_graph_sizes(target_work: Path) -> list[int]:
    values: list[int] = []
    work_root = ROOT / "work"
    if not work_root.is_dir():
        return values
    for path in work_root.rglob("graphs/*.pt"):
        if is_within(path.resolve(), target_work.resolve()):
            continue
        try:
            size = int(path.stat().st_size)
        except OSError:
            continue
        if size > 0:
            values.append(size)
    return values


def _prepare_storage_preflight(ctx: dict, split: dict[str, object]) -> None:
    """Refuse a full cache build before it exhausts the target filesystem."""

    work = Path(ctx["work"]).resolve()
    selected_cases = len(split["train"]) + len(split["val"])
    samples_per_case = int(ctx["config"]["cache"]["samples_per_case"])
    graph_files = selected_cases * samples_per_case
    reference = sorted(_reference_graph_sizes(work))
    if reference:
        index = min(len(reference) - 1, int(round(0.90 * (len(reference) - 1))))
        graph_bytes_each = max(8 * 1024**2, int(reference[index]))
        graph_source = f"p90 of {len(reference)} existing graph files"
    else:
        graph_bytes_each = 16 * 1024**2
        graph_source = "16 MiB conservative fallback"
    # Cropped compressed region caches vary with liver bounding-box size. The
    # estimate is deliberately conservative and includes all train/val cases.
    region_bytes = selected_cases * 48 * 1024**2
    graph_bytes = graph_files * graph_bytes_each
    headroom = 2 * 1024**3
    estimated = int((region_bytes + graph_bytes) * 1.20 + headroom)
    usage_target = work
    while not usage_target.exists() and usage_target != usage_target.parent:
        usage_target = usage_target.parent
    free = int(shutil.disk_usage(usage_target).free)
    print(
        "[Storage] "
        f"free={_human_bytes(free)} estimated_required={_human_bytes(estimated)} "
        f"regions≈{_human_bytes(region_bytes)} graphs≈{_human_bytes(graph_bytes)} "
        f"({graph_source})"
    )
    if free < estimated and os.environ.get("HIERCP_ALLOW_LOW_DISK", "0").lower() not in {
        "1", "true", "yes", "on"
    }:
        raise RunError(
            "Insufficient free space for prepare. No cache files were created. "
            f"Need approximately {_human_bytes(estimated)}, available {_human_bytes(free)}. "
            "Use a work directory on a larger filesystem or explicitly set "
            "HIERCP_ALLOW_LOW_DISK=1 after independently verifying capacity."
        )


def prepare(ctx: dict, args: argparse.Namespace) -> None:
    p = ctx["paths"]
    split = ensure_split(ctx, args)
    if not split["train"]:
        raise RunError("The split has no training cases")
    _prepare_storage_preflight(ctx, split)
    prototype = module_command(
        "hiercp.pipeline",
        "prepare-prototypes",
        "--config", ctx["config_path"],
        "--data-dir", ctx["data"],
        "--split-file", p["split"],
        "--region-cache-dir", p["regions"],
        "--output", p["prototype"],
        "--seed", ctx["seed"],
        "--run-mode", args.run_mode,
    )
    graphs = module_command(
        "hiercp.pipeline",
        "prepare",
        "--config", ctx["config_path"],
        "--data-dir", ctx["data"],
        "--split-file", p["split"],
        "--region-cache-dir", p["regions"],
        "--prototype-bank", p["prototype"],
        "--cache-dir", p["graphs"],
        "--seed", ctx["seed"],
        "--run-mode", args.run_mode,
    )
    if args.overwrite:
        prototype.append("--overwrite")
        graphs.append("--overwrite")
    execute(prototype, log=p["log"], dry_run=args.dry_run)
    execute(graphs, log=p["log"], dry_run=args.dry_run)


def train(ctx: dict, args: argparse.Namespace) -> None:
    p = ctx["paths"]
    if not args.dry_run:
        if not p["prototype"].is_file():
            raise RunError(f"Prototype bank is missing: {p['prototype']}; run prepare first")
        if not list(p["graphs"].glob("*.pt")):
            raise RunError(f"Graph cache is empty: {p['graphs']}; run prepare first")
    command = module_command(
        "hiercp.pipeline",
        "train",
        "--config", ctx["config_path"],
        "--cache-dir", p["graphs"],
        "--prototype-bank", p["prototype"],
        "--checkpoint", p["checkpoint"],
        "--device", args.device,
        "--seed", ctx["seed"],
        "--run-mode", args.run_mode,
        "--ablation-mode", args.ablation_mode,
    )
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.overwrite:
        command.append("--overwrite")
    execute(command, log=p["log"], dry_run=args.dry_run)


def write_metadata(
    ctx: dict,
    args: argparse.Namespace,
    *,
    validation_status: str,
    assembly_status: str,
) -> None:
    p = ctx["paths"]
    p["meta"].parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "method": "hiercp-full",
        "framework": "torch_geometric",
        "upper_feature_policy": UPPER_FEATURE_POLICY,
        "run_mode": args.run_mode,
        "medical_root": str(ctx["medical"]),
        "input_data": str(ctx["data"]),
        "work": str(ctx["work"]),
        "config": str(ctx["config_path"]),
        "prototype": str(p["prototype"]),
        "checkpoint": str(p["checkpoint"]),
        "generated_output": str(p["output"]),
        "validation_report": str(p["validation_report"]),
        "generation_status": "complete",
        "validation_status": validation_status,
        "assembly_status": assembly_status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if p["meta"].exists() and not args.overwrite:
        existing = load_json(p["meta"])
        stable_existing = {key: value for key, value in existing.items() if key != "updated_at"}
        stable_requested = {key: value for key, value in payload.items() if key != "updated_at"}
        if stable_existing == stable_requested:
            print(f"[Reuse] run metadata already matches exactly: {p['meta']}")
            return
        raise FileExistsError(
            f"Run metadata already exists with different content: {p['meta']}. "
            "Use --overwrite or a separate --work directory."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{p['meta'].name}.", suffix=".tmp", dir=p["meta"].parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, p["meta"])
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[OK] run metadata: {p['meta']}")


def generate(ctx: dict, args: argparse.Namespace) -> None:
    p = ctx["paths"]
    if not args.dry_run:
        if not p["prototype"].is_file():
            raise RunError(f"Prototype bank is missing: {p['prototype']}")
        if not p["checkpoint"].is_file():
            raise RunError(f"Checkpoint is missing: {p['checkpoint']}; run train first")
    command = module_command(
        "hiercp.pipeline",
        "generate",
        "--config", ctx["config_path"],
        "--data-dir", ctx["data"],
        "--region-cache-dir", p["regions"],
        "--prototype-bank", p["prototype"],
        "--checkpoint", p["checkpoint"],
        "--out-dir", p["output"],
        "--device", args.device,
        "--seed", ctx["seed"],
        "--run-mode", args.run_mode,
    )
    if args.max_cases is not None:
        command.extend(["--max-cases", str(args.max_cases)])
    for case_id in args.case_id or []:
        command.extend(["--case-id", case_id])
    if args.overwrite:
        command.append("--overwrite")
    execute(command, log=p["log"], dry_run=args.dry_run)

    validation_status = "skipped_nonproduction"
    if not args.skip_validation:
        command = module_command(
            "tools.validate",
            "--reference-dir", ctx["data"],
            "--candidate-dir", p["output"],
            "--valid-output-dir", p["valid"],
            "--materialization", args.materialization,
            "--expect-augmentation",
            "--run-mode", _validation_run_mode(args.run_mode),
        )
        if args.overwrite:
            command.append("--overwrite")
        else:
            command.append("--resume")
        execute(command, log=p["log"], dry_run=args.dry_run)
        validation_status = "planned" if args.dry_run else "complete"

    assembly_status = "skipped_nonproduction"
    if not args.skip_assemble:
        command = module_command(
            "tools.assemble",
            "--original-dir", ctx["data"],
            "--aug-dir", p["valid"],
            "--validation-report", p["validation_report"],
            "--out-dir", p["dataset"],
            "--tag", RUN_NAME,
            "--mode", args.materialization,
            "--run-mode", _validation_run_mode(args.run_mode),
        )
        if args.overwrite:
            command.append("--overwrite")
        execute(command, log=p["log"], dry_run=args.dry_run)
        assembly_status = "planned" if args.dry_run else "complete"

    if not args.dry_run:
        write_metadata(
            ctx,
            args,
            validation_status=validation_status,
            assembly_status=assembly_status,
        )


def run_nnunet(ctx: dict, args: argparse.Namespace, target: str) -> None:
    config = Path(args.nnunet_config).expanduser().resolve()
    if not config.is_file():
        raise RunError(f"nnU-Net config is missing: {config}")
    command = module_command(
        "tools.nnunet",
        target,
        "--project-root", ROOT,
        "--medical-root", ctx["medical"],
        "--workspace", ctx["work"],
        "--config", config,
        "--device", args.device,
        "--materialization", args.materialization,
    )
    if args.nnunet_root:
        command.extend(["--nnunet-root", str(Path(args.nnunet_root).expanduser().resolve())])
    if args.folds:
        command.extend(["--folds", args.folds])
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    execute(command, log=ctx["work"] / "nnunet.log", dry_run=args.dry_run)


def status(ctx: dict) -> None:
    p = ctx["paths"]
    print("HierCP status")
    print("  project:    ", ROOT)
    print("  data:       ", ctx["data"])
    print("  work:       ", ctx["work"])
    print("  config:     ", ctx["config_path"])
    if p["split"].is_file():
        split = load_case_split(p["split"])
        print(f"  split:       train={len(split['train'])} val={len(split['val'])}")
    else:
        print("  split:       missing")
    print("  prototype:  ", "present (integrity not checked by status)" if p["prototype"].is_file() else "missing")
    print("  regions:    ", len([x for x in p["regions"].iterdir() if x.is_dir()]) if p["regions"].is_dir() else 0)
    print("  graph cache:", len(list(p["graphs"].glob("*.pt"))))
    if p["checkpoint"].is_file():
        try:
            checkpoint = torch_load_compat(p["checkpoint"], map_location="cpu")
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            best_epoch = int(checkpoint.get("best_epoch", checkpoint_epoch))
            selection = checkpoint.get("best_selection")
            if isinstance(selection, dict):
                def metric(name: str) -> str:
                    try:
                        value = float(selection[name])
                    except (KeyError, TypeError, ValueError):
                        return UNAVAILABLE
                    return f"{value:.4f}" if math.isfinite(value) else UNAVAILABLE

                print(
                    "  checkpoint:  readable (provenance not checked by status) "
                    f"epoch={checkpoint_epoch} best_epoch={best_epoch} "
                    f"mrr={metric('mrr')} margin={metric('margin')}"
                )
            else:
                print(
                    "  checkpoint:  readable (provenance not checked by status) "
                    f"epoch={checkpoint_epoch} best_epoch={best_epoch}"
                )
        except Exception as exc:
            print(f"  checkpoint:  unreadable ({exc})")
    else:
        print("  checkpoint:  missing")
    if p["last"].is_file():
        try:
            state = torch_load_compat(p["last"], map_location="cpu")
            epoch = int(state.get("epoch", 0))
            target = int(state.get("target_epochs", 0))
            marker = state.get("training_complete")
            complete = marker is True and target > 0 and epoch == target
            if marker is True and not complete:
                status = "inconsistent"
            else:
                status = "completion_marker_present_unverified" if complete else "partial_state_unverified"
            rendered_target = str(target) if target > 0 else "unknown"
            print(
                f"  training:    {status} epoch={epoch}/{rendered_target} "
                f"training_complete_marker={marker!r}"
            )
        except Exception as exc:
            print(f"  training:    unreadable ({exc})")
    else:
        print("  training:    not started")
    for label, root in (("generated", p["output"]), ("validated", p["valid"]), ("dataset", p["dataset"])):
        images = {path.name.removesuffix("_0000.nii.gz") for path in (root / "image").glob("*_0000.nii.gz")}
        labels = {path.name.removesuffix(".nii.gz") for path in (root / "labels").glob("*.nii.gz")}
        if images != labels:
            print(f"  {label}: invalid pairs; missing_labels={sorted(images - labels)} missing_images={sorted(labels - images)}")
        else:
            print(f"  {label}: pairs={len(images)}; content/manifest integrity not checked by status")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "target",
        choices=(
            "prepare", "train", "generate", "all", "full",
            "smoke", "audit", "causality", "env", "case", "status",
            *NNUNET_TARGETS.keys(),
        ),
    )
    result.add_argument("--medical-root")
    result.add_argument("--work")
    result.add_argument("--config", default=str(TRAIN_CONFIG))
    result.add_argument("--nnunet-config", default=str(NNUNET_CONFIG))
    result.add_argument("--nnunet-root")
    result.add_argument("--folds")
    result.add_argument("--device", default="auto")
    result.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="production",
        help="production rejects scale/subset overrides; reduced runs must be labelled",
    )
    result.add_argument(
        "--ablation-mode",
        choices=("full", "no_local", "no_patient", "no_population"),
        default="full",
    )
    result.add_argument("--seed", type=int)
    result.add_argument("--max-cases", type=int)
    result.add_argument("--case-id", action="append")
    result.add_argument("--epochs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--num-workers", type=int)
    result.add_argument("--checkpoint")
    result.add_argument(
        "--audit-split",
        choices=("all", "train", "val"),
        default="all",
    )
    result.add_argument("--audit-batches", type=int, default=0)
    result.add_argument("--strict", action="store_true")
    result.add_argument("--label", help="NIfTI label for the case audit target")
    result.add_argument("--components", type=int, default=3)
    result.add_argument("--output")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--skip-validation", action="store_true")
    result.add_argument("--skip-assemble", action="store_true")
    result.add_argument("--materialization", choices=("symlink", "hardlink", "copy"), default="symlink")
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        medical = medical_root(args.medical_root)
        work = Path(args.work).expanduser().resolve() if args.work else DEFAULT_WORK.resolve()
        _guard_run_contract(args, work)
        read_only_targets = {"status", "audit", "env", "case", "nnunet-status"}
        guard_work(
            work,
            medical,
            create=not args.dry_run and args.target not in read_only_targets,
        )
        config_path = Path(args.config).expanduser().resolve()
        config = load_json(config_path)
        seed = int(args.seed if args.seed is not None else config["seed"])
        ctx = {
            "medical": medical,
            "data": (medical / "Data").resolve(),
            "work": work,
            "config_path": config_path,
            "config": config,
            "seed": seed,
            "paths": paths(work),
        }
        _report_overwrite_scope(ctx, args)

        if args.target == "status":
            status(ctx)
            return
        if args.target == "audit":
            execute(module_command("tools.audit"), dry_run=args.dry_run)
            return
        if args.target == "smoke":
            execute(module_command("tools.smoke", "--device", args.device), log=ctx["paths"]["log"], dry_run=args.dry_run)
            return
        if args.target == "causality":
            checkpoint = (
                Path(args.checkpoint).expanduser().resolve()
                if args.checkpoint
                else ctx["paths"]["checkpoint"]
            )
            if not args.dry_run:
                if not checkpoint.is_file():
                    raise RunError(f"Checkpoint is missing: {checkpoint}")
                if not ctx["paths"]["graphs"].is_dir():
                    raise RunError(
                        f"Graph cache is missing: {ctx['paths']['graphs']}; run prepare first"
                    )
            report = (
                Path(args.output).expanduser().resolve()
                if args.output
                else ctx["work"] / "causality.json"
            )
            command = module_command(
                "tools.causality",
                "--cache-dir", ctx["paths"]["graphs"],
                "--checkpoint", checkpoint,
                "--prototype-bank", ctx["paths"]["prototype"],
                "--run-mode", args.run_mode,
                "--seed", ctx["seed"],
                "--device", args.device,
                "--split", args.audit_split,
                "--output", report,
            )
            if args.audit_batches > 0:
                command.extend(["--max-batches", str(args.audit_batches)])
            if args.strict:
                command.append("--strict")
            if args.overwrite:
                command.append("--overwrite")
            execute(command, log=ctx["paths"]["log"], dry_run=args.dry_run)
            return
        if args.target == "env":
            execute(module_command("tools.env", "--medical-root", medical, "--require-pyg"), dry_run=args.dry_run)
            return
        if args.target == "case":
            if not args.label:
                raise RunError("case target requires --label /path/to/label.nii.gz")
            command = module_command("tools.case", args.label, "--components", args.components)
            if args.output:
                command.extend(["--output", args.output])
            execute(command, dry_run=args.dry_run)
            return
        if args.target in NNUNET_TARGETS:
            run_nnunet(ctx, args, NNUNET_TARGETS[args.target])
            return

        if args.target in {"prepare", "train", "generate", "all", "full"}:
            environment_command = module_command(
                "tools.env", "--medical-root", medical, "--require-pyg"
            )
            if args.target in {"train", "generate", "all", "full"} and str(
                args.device
            ).lower() != "cpu":
                environment_command.append("--require-cuda")
            execute(environment_command, dry_run=args.dry_run)

        if args.target in {"prepare", "all", "full"}:
            prepare(ctx, args)
        if args.target in {"train", "all", "full"}:
            train(ctx, args)
        if args.target in {"generate", "all", "full"}:
            generate(ctx, args)
        if args.target == "full":
            run_nnunet(ctx, args, "all")
        executed_stages = [
            stage
            for stage, selected in (
                ("prepare", args.target in {"prepare", "all", "full"}),
                ("train", args.target in {"train", "all", "full"}),
                ("generate", args.target in {"generate", "all", "full"}),
                ("validate", args.target in {"generate", "all", "full"} and not args.skip_validation),
                ("assemble", args.target in {"generate", "all", "full"} and not args.skip_assemble),
                ("nnunet_full_evaluation", args.target == "full"),
            )
            if selected
        ]
        print(
            "[RunSummary] "
            + json.dumps(
                {
                    "status": "planned" if args.dry_run else "requested_stages_complete",
                    "completion_scope": (
                        "dry-run command plan"
                        if args.dry_run
                        else "only the explicitly requested stages"
                    ),
                    "target": args.target,
                    "run_mode": args.run_mode,
                    "stages": executed_stages,
                    "stage_status": {
                        stage: "planned" if args.dry_run else "completed_or_exactly_reused"
                        for stage in executed_stages
                    },
                    "skipped_nonproduction_stages": [
                        stage for stage, skipped in (
                            ("validate", args.skip_validation),
                            ("assemble", args.skip_assemble),
                        ) if skipped and args.target in {"generate", "all", "full"}
                    ],
                    "actual_execution_evidence": (
                        "none; dry-run only"
                        if args.dry_run
                        else "see each stage PreRun/PostRun report and artifact manifest"
                    ),
                    "full_training_requested": args.target in {"train", "all", "full"},
                    "full_evaluation_requested": args.target == "full",
                    "runner_infers_reuse_as_execution": False,
                },
                sort_keys=True,
            )
        )
        if not args.dry_run:
            status(ctx)
    except (RunError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
