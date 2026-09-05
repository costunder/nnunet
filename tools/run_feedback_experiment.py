"""Start a NEW full-size feedback experiment without replacing existing trainers/results.

This is a launcher, not a resume or evaluation command. It preserves the checked-in
40-epoch quality-GNN and 250-epoch nnU-Net contracts. Only one allocated visible GPU
is supported; this program never changes CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_trainers.install_onlinecp_custom_trainers import (
    MODULES, audit_sources, locate_nnunet_root,
)


def build_plan(project_root, medical_root, *, outer_fold=0, dataset_id=760,
               seed=42, python_executable=None):
    if type(outer_fold) is not int or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be one of 0, 1, 2, 3, 4")
    if type(dataset_id) is not int or not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be an integer in 1..999")
    if type(seed) is not int or seed < 0:
        raise ValueError("nnU-Net seed must be a nonnegative integer")
    project, medical = Path(project_root).resolve(), Path(medical_root).resolve()
    nnconfig = json.loads((project / "config/nnunet.json").read_text(encoding="utf-8"))
    root = project / "work/feedback_experiment"
    pair, online = "feedback_experiment/paired", "feedback_experiment/online"
    online_path = root / "online"
    raw = online_path / "nnunetv2/nnUNet_raw"
    py = str(python_executable or sys.executable)
    environment = {name: str(online_path / "nnunetv2" / name) for name in
                   ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}
    environment.update(PYTHONUNBUFFERED="1",
                       nnUNet_n_proc_DA=str(nnconfig["training"]["nnunet_n_proc_DA"]))
    commands = []

    def add(name, *argv):
        commands.append({"name": name, "argv": [str(arg) for arg in argv]})

    add("install_private_trainers", py, project / "custom_trainers/install_onlinecp_custom_trainers.py",
        "apply", "--nnunet-root", root / "runtime/nnunetv2")
    add("environment", py, project / "run.py", "env", "--medical-root", medical)
    paired = ["--project-root", project, "--medical-root", medical,
              "--work", root / "paired", "--outer-fold", outer_fold, "--device", "cuda:0"]
    for stage in ("split", "gnn-prepare", "gnn-train"):
        add(stage, py, "-m", "tools.paired_benchmark", stage, *paired)
    online_args = ["--project-root", project, "--medical-root", medical,
                   "--paired-root", pair, "--online-root", online,
                   "--outer-fold", outer_fold, "--dataset-id", dataset_id]
    for stage in ("plan", "bank"):
        add(stage, py, "-m", "tools.online_cp_benchmark", stage, *online_args,
            "--device", "cuda:0", "--candidate-count", 128)
    policy = project / "config/online_cp_feedback.json"
    add("feedback_contract", py, "-m", "tools.online_cp_curriculum", *online_args,
        "--curriculum-config", policy)
    training = ["--bank", online_path / f"folds/fold_{outer_fold}/bank/index.json",
                "--feedback-config", policy,
                "--configuration", nnconfig["dataset"]["configuration"],
                "--device", "cuda", "--seed", seed]
    extra = ["--feedback-gnn-config", project / "config/online_cp_feedback_gnn.json",
             "--feedback-raw-root", raw / f"Dataset{dataset_id:03d}_LiverOnlineCP_OF{outer_fold}"]
    for dry_run in (True, False):
        for arm in ("full", "basic"):
            add(("check_" if dry_run else "train_") + arm,
                py, "-m", "tools.train_online_feedback", *training, "--arm", arm,
                *(extra if arm == "full" else []), *( ["--dry-run"] if dry_run else []))
    return {"project_root": project, "medical_root": medical, "run_root": root,
            "package_destination": root / "runtime/nnunetv2", "python_executable": py,
            "env_updates": environment, "commands": commands,
            "minimum_free_bytes": int(nnconfig["runtime"]["minimum_free_gb_before_preprocess"] * 1024**3),
            "seed_note": f"nnU-Net seed={seed}; quality GNN/bank inherit their checked-in fold-specific configuration",
            "scope": "New experiment: quality GNN, Full, then Basic. Not downstream comparison/evaluation or resume."}


def copy_nnunet_package(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Existing package copy preserved: {destination}")
    destination = destination.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Private nnU-Net copy must not overlap its original package")
    if not (source / "training/nnUNetTrainer").is_dir():
        raise ValueError(f"Not an nnunetv2 package: {source}")

    def ignore(path, names):
        excluded = {"__pycache__", ".git", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"}
        if Path(path) == source / "training/nnUNetTrainer":
            excluded.update(MODULES)
        return excluded.intersection(names)

    shutil.copytree(source, destination, ignore=ignore)


def execute_plan(plan, *, runner=None, package_root=None):
    runner = subprocess.run if runner is None else runner
    root, medical, project = (plan[key] for key in ("run_root", "medical_root", "project_root"))
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Existing experiment preserved: {root}. This launcher does not resume or overwrite it.")
    if not root.resolve().is_relative_to(project):
        raise ValueError("Experiment output escapes this source checkout")
    for folder in ("Data/image", "Data/labels"):
        if not (medical / folder).is_dir():
            raise FileNotFoundError(f"Required real-data directory is missing: {medical / folder}")
    # Existing ancestors may include a separate work volume. Inspect that volume.
    storage = root.parent
    while not storage.exists():
        storage = storage.parent
    if shutil.disk_usage(storage).free < plan["minimum_free_bytes"]:
        raise RuntimeError("Insufficient free space for the configured preprocessing contract")
    source = Path(package_root).resolve() if package_root is not None else locate_nnunet_root(None)
    if not (source / "training/nnUNetTrainer").is_dir():
        raise ValueError(f"Not an nnunetv2 package: {source}")
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Experiment and original nnU-Net package overlap")
    audit_sources()
    env = {**os.environ, **plan["env_updates"]}
    env["PYTHONPATH"] = os.pathsep.join([str(root / "runtime"), str(project), env.get("PYTHONPATH", "")])
    gpu_check = ("import torch\n"
                 "n = torch.cuda.device_count()\n"
                 "if n != 1:\n"
                 "    raise RuntimeError(f'Expected one allocated visible GPU, found {n}; GPU visibility was not changed')\n")
    runner([plan["python_executable"], "-c", gpu_check], cwd=project, env=env, check=True)
    root.mkdir(parents=True, exist_ok=False)
    print(f"[NEW EXPERIMENT] {root}", flush=True)
    with (root / "launch_plan.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, default=str, indent=2)
    copy_nnunet_package(source, plan["package_destination"])
    for command in plan["commands"]:
        print(f"\n[RUN {command['name']}] {shlex.join(command['argv'])}", flush=True)
        runner(command["argv"], cwd=project, env=env, check=True)
    print(f"[TRAINING COMMANDS COMPLETED] {plan['env_updates']['nnUNet_results']}", flush=True)
    print("Downstream comparison and statistical evaluation have not been run.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-root", required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--dataset-id", type=int, default=760)
    parser.add_argument("--seed", type=int, default=42, help="nnU-Net seed; quality model uses its configured fold-specific seed")
    parser.add_argument("--dry-run", action="store_true", help="Print only: no directories, copying, GPU checks or child commands")
    args = parser.parse_args(argv)
    plan = build_plan(PROJECT_ROOT, args.medical_root, outer_fold=args.outer_fold,
                      dataset_id=args.dataset_id, seed=args.seed)
    print(f"[SCOPE] {plan['scope']}\n[SEEDS] {plan['seed_note']}", flush=True)
    if args.dry_run:
        print(f"[DRY RUN ONLY] Would create {plan['run_root']} and a private nnU-Net package copy")
        for command in plan["commands"]:
            print(f"[{command['name']}] {shlex.join(command['argv'])}")
        return
    try:
        execute_plan(plan)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        print(f"[FAILED] Further stages were not launched. Existing files and partial outputs are preserved: {plan['run_root']}",
              file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
