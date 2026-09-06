"""Start a NEW full-size feedback experiment without replacing existing trainers/results.

Preparation recovery uses a new work directory and never resumes a training
checkpoint. The launcher preserves the checked-in
40-epoch quality-GNN and 250-epoch nnU-Net contracts. Only one allocated visible GPU
is supported; this program never changes CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_trainers.install_onlinecp_custom_trainers import (
    MODULES, audit_sources, locate_nnunet_root,
)


def build_plan(project_root, medical_root, *, outer_fold=0, dataset_id=760,
               seed=42, python_executable=None, run_root=None, experiment_name=None,
               train_config=None, recover_from=None):
    if type(outer_fold) is not int or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be one of 0, 1, 2, 3, 4")
    if type(dataset_id) is not int or not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be an integer in 1..999")
    if type(seed) is not int or seed < 0:
        raise ValueError("nnU-Net seed must be a nonnegative integer")
    project, medical = Path(project_root).resolve(), Path(medical_root).resolve()
    nnconfig = json.loads((project / "config/nnunet.json").read_text(encoding="utf-8"))
    if run_root is not None and experiment_name is not None:
        raise ValueError("Choose run_root or experiment_name, not both")
    work = project / "work"
    if experiment_name is not None:
        name = Path(experiment_name)
        if name.is_absolute() or not name.parts or any(part in (".", "..") for part in name.parts):
            raise ValueError("experiment_name must be a relative name below work")
    else:
        name = Path("feedback_experiment_recovered" if recover_from is not None else "feedback_experiment")
    root = Path(run_root) if run_root is not None else work / name
    root = (project / root if not root.is_absolute() else root).resolve()
    if root == work.resolve() or not root.is_relative_to(work.resolve()):
        raise ValueError("Experiment output escapes this checkout's work directory")
    relative = root.relative_to(work.resolve())
    pair, online = (relative / "paired").as_posix(), (relative / "online").as_posix()
    recovery_source = None
    if recover_from is not None:
        recovery_source = Path(recover_from)
        recovery_source = (project / recovery_source if not recovery_source.is_absolute() else recovery_source).resolve()
        if not recovery_source.is_relative_to(work.resolve()) or recovery_source == work.resolve():
            raise ValueError("Recovery source must belong to this checkout's work directory")
        if root == recovery_source or root.is_relative_to(recovery_source) or recovery_source.is_relative_to(root):
            raise ValueError("Recovery output must not overlap its preserved source")
    explicit_train_config = train_config is not None or recovery_source is not None
    train_config = Path(train_config) if train_config is not None else (
        root / "recovery/train_config.json" if recovery_source is not None else project / "config/train.json")
    train_config = (project / train_config if not train_config.is_absolute() else train_config).resolve()
    if recovery_source is not None and not train_config.is_relative_to(root):
        raise ValueError("Recovery train_config must be a new artifact inside its run root")
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
    if explicit_train_config:
        paired.extend(["--train-config", train_config])
    for stage in (("gnn-train",) if recovery_source is not None else ("split", "gnn-prepare", "gnn-train")):
        add(stage, py, "-m", "tools.paired_benchmark", stage, *paired)
    online_args = ["--project-root", project, "--medical-root", medical,
                   "--paired-root", pair, "--online-root", online,
                   "--outer-fold", outer_fold, "--dataset-id", dataset_id]
    if explicit_train_config:
        online_args.extend(["--train-config", train_config])
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
            "outer_fold": outer_fold, "dataset_id": dataset_id, "seed": seed,
            "experiment_name": relative.as_posix(), "train_config": train_config,
            "recovery_source_root": recovery_source,
            "package_destination": root / "runtime/nnunetv2", "python_executable": py,
            "env_updates": environment, "commands": commands,
            "minimum_free_bytes": int(nnconfig["runtime"]["minimum_free_gb_before_preprocess"] * 1024**3),
            "seed_note": f"nnU-Net seed={seed}; quality GNN/bank inherit their checked-in fold-specific configuration",
            "scope": ("Preparation recovery into a new directory; new quality GNN, Full, then Basic. Not training-checkpoint resume or evaluation."
                      if recovery_source is not None else
                      "New experiment: quality GNN, Full, then Basic. Not downstream comparison/evaluation or resume.")}


def validate_recovery_source(plan, source_root):
    # Kept lazy and read-only so a dry run does not import torch or nnU-Net.
    from tools.feedback_preparation_recovery import validate_recovery_source as validate
    return validate(plan, source_root)


def prepare_recovery(plan, source_root, *, runner, env):
    from tools.feedback_preparation_recovery import prepare_recovery as prepare
    return prepare(plan, source_root, runner=runner, env=env)


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False, default=str).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_inventory(destination):
    destination = Path(destination)
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("The verified private runtime is missing or was replaced")
    result = {}
    for path in sorted(destination.rglob("*")):
        relative = path.relative_to(destination)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Private runtime contains an unverified symlink: {path}")
        if path.is_file():
            result[relative.as_posix()] = _file_sha256(path)
    for name, expected in MODULES.items():
        if result.get("training/nnUNetTrainer/" + name) != expected:
            raise ValueError(f"Private runtime trainer is missing or changed: {name}")
    if "__init__.py" not in result or "training/nnUNetTrainer/nnUNetTrainer.py" not in result:
        raise ValueError("Private runtime lacks its native nnU-Net package evidence")
    return result


def _save_journal(root, journal):
    path = root / "execution_journal.json"
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlinked execution journal")
    payload = {**journal, "journal_sha256": _json_sha256(journal)}
    descriptor, name = tempfile.mkstemp(prefix=".execution_journal.", suffix=".tmp", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_preparation_journal(plan, source_identity):
    root = plan["run_root"]
    path = root / "execution_journal.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Preparation resume needs its own verified execution_journal.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = {"format", "plan_sha256", "source_identity", "runtime_inventory", "preparation_receipt",
            "stages", "training_started", "complete", "journal_sha256"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Malformed preparation execution journal")
    checksum = payload.pop("journal_sha256")
    if (payload["format"] != "feedback_preparation_execution_v1" or checksum != _json_sha256(payload)
            or payload["plan_sha256"] != _json_sha256(plan)
            or payload["source_identity"] != source_identity):
        raise ValueError("Preparation plan, source identity or journal checksum changed")
    saved_plan = root / "launch_plan.json"
    if saved_plan.is_symlink() or not saved_plan.is_file() or _json_sha256(json.loads(saved_plan.read_text(encoding="utf-8"))) != _json_sha256(plan):
        raise ValueError("The saved launch plan is missing or changed")
    if type(payload["training_started"]) is not bool or type(payload["complete"]) is not bool:
        raise ValueError("Malformed preparation execution status")
    if payload["training_started"] or payload["complete"]:
        raise ValueError("Training already started. --resume-preparation does not resume or restart training checkpoints; preserve this run and inspect its stage-specific training state.")
    if not isinstance(payload["stages"], list) or any(not isinstance(row, dict) or row.get("name") not in
            {"copy_private_runtime", "install_private_trainers", "environment", "recover_preparation"}
            or row.get("status") not in {"running", "completed", "failed"} for row in payload["stages"]):
        raise ValueError("Preparation journal contains unrecognized execution stages")
    if not payload["runtime_inventory"]:
        raise ValueError("Runtime setup was interrupted before verifiable completion. Preserve this directory and choose a new --run-root; it will not be overwritten.")
    if _runtime_inventory(plan["package_destination"]) != payload["runtime_inventory"]:
        raise ValueError("Private runtime bytes changed since verified installation")
    return payload


def _execute_recovery(plan, source, source_identity, *, runner, env, journal=None):
    root = plan["run_root"]
    token = uuid.uuid4().hex
    lock = root / "recovery_execution.lock"
    lock_payload = json.dumps({"pid": os.getpid(), "token": token, "run_root": str(root)})
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(lock_payload)
    except FileExistsError as exc:
        raise FileExistsError(f"Recovery execution lock already exists: {lock}. Check whether its recorded process is still active before retrying; no process was terminated.") from exc
    try:
        continuing = journal is not None
        if continuing:
            # Another launcher may have completed between preflight and lock
            # acquisition. Recheck under the exclusive lock before any reuse.
            journal = _load_preparation_journal(plan, source_identity)
        if journal is None:
            journal = dict(format="feedback_preparation_execution_v1", plan_sha256=_json_sha256(plan),
                           source_identity=source_identity, runtime_inventory=None, preparation_receipt=None,
                           stages=[], training_started=False, complete=False)
            _save_journal(root, journal)

        def stage(name, action, *, training=False):
            if training:
                journal["training_started"] = True
            record = {"name": name, "status": "running"}
            journal["stages"].append(record)
            _save_journal(root, journal)
            try:
                value = action()
            except (Exception, KeyboardInterrupt) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                _save_journal(root, journal)
                raise
            record["status"] = "completed"
            _save_journal(root, journal)
            return value

        commands = {command["name"]: command for command in plan["commands"]}

        def child(name):
            command = commands[name]
            print(f"\n[RUN {name}] {shlex.join(command['argv'])}", flush=True)
            return runner(command["argv"], cwd=plan["project_root"], env=env, check=True)

        if not continuing:
            stage("copy_private_runtime", lambda: copy_nnunet_package(source, plan["package_destination"]))
            def install():
                child("install_private_trainers")
                journal["runtime_inventory"] = _runtime_inventory(plan["package_destination"])
            stage("install_private_trainers", install)
        # A diagnostic is rerun, not inferred from an old zero return code.
        stage("environment", lambda: child("environment"))

        def recover():
            receipt = prepare_recovery(plan, plan["recovery_source_root"], runner=runner, env=env)
            if not isinstance(receipt, dict) or not isinstance(receipt.get("train_config"), str):
                raise ValueError("Recovery helper must return a verified receipt with train_config")
            config = Path(receipt["train_config"])
            if not config.is_absolute() or config.resolve() != plan["train_config"] or not config.is_file() or config.is_symlink():
                raise ValueError("Recovery receipt train_config does not match the exact planned new artifact")
            # Strict JSON plus helper re-verification, never existence-only reuse.
            json.dumps(receipt, allow_nan=False)
            previous = journal["preparation_receipt"]
            if previous is not None and previous != receipt:
                raise ValueError("Verified preparation receipt changed on resume")
            journal["preparation_receipt"] = receipt
            return receipt
        stage("recover_preparation", recover)
        for command in plan["commands"]:
            if command["name"] not in {"install_private_trainers", "environment"}:
                stage(command["name"], lambda name=command["name"]: child(name), training=True)
        journal["complete"] = True
        _save_journal(root, journal)
    finally:
        if lock.read_text(encoding="utf-8") != lock_payload:
            raise RuntimeError("Recovery execution lock ownership changed; it was not removed")
        lock.unlink()


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


def execute_plan(plan, *, runner=None, package_root=None, resume_preparation=False):
    runner = subprocess.run if runner is None else runner
    root, medical, project = (plan[key] for key in ("run_root", "medical_root", "project_root"))
    recovering = plan.get("recovery_source_root") is not None
    if resume_preparation and not recovering:
        raise ValueError("--resume-preparation requires --recover-from; it is not training resume")
    if root.is_symlink() or ((root.exists()) and not resume_preparation):
        raise FileExistsError(f"Existing experiment preserved: {root}. Use a new --run-root, or explicit --resume-preparation only for a verified interrupted preparation recovery.")
    if not root.resolve().is_relative_to(project):
        raise ValueError("Experiment output escapes this source checkout")
    source_identity = None
    journal = None
    if recovering:
        recovery_source = plan["recovery_source_root"]
        if not recovery_source.is_dir() or not (recovery_source / "launch_plan.json").is_file():
            raise FileNotFoundError(f"Recovery source launch_plan.json is missing: {recovery_source}")
        if root == recovery_source or root.is_relative_to(recovery_source) or recovery_source.is_relative_to(root):
            raise ValueError("Recovery output overlaps its preserved source")
        source_identity = validate_recovery_source(plan, recovery_source)
        if not isinstance(source_identity, dict):
            raise ValueError("Recovery source validation did not return an identity")
        json.dumps(source_identity, allow_nan=False)
        if resume_preparation:
            journal = _load_preparation_journal(plan, source_identity)
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
    if recovering:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(root / "runtime"), str(project), env.get("PYTHONPATH", "")])
    gpu_check = ("import torch\n"
                 "n = torch.cuda.device_count()\n"
                 "if n != 1:\n"
                 "    raise RuntimeError(f'Expected one allocated visible GPU, found {n}; GPU visibility was not changed')\n")
    runner([plan["python_executable"], "-c", gpu_check], cwd=project, env=env, check=True)
    if not resume_preparation:
        root.mkdir(parents=True, exist_ok=False)
        print(f"[NEW EXPERIMENT] {root}", flush=True)
        with (root / "launch_plan.json").open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, default=str, indent=2)
    if recovering:
        _execute_recovery(plan, source, source_identity, runner=runner, env=env, journal=journal)
    else:
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
    parser.add_argument("--recover-from", help="Preserve this existing preparation and recover into a new work directory in the same checkout")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--run-root", help="Output path inside this checkout's work directory; relative paths start at the checkout")
    outputs.add_argument("--experiment-name", help="Relative experiment name below work (nested names supported)")
    parser.add_argument("--train-config", help="Explicit quality-training config; recovery creates this artifact inside its new run root")
    parser.add_argument("--resume-preparation", action="store_true", help="Reverify and continue this recovery's preparation journal only, never training checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print only: no directories, copying, GPU checks or child commands")
    args = parser.parse_args(argv)
    plan = build_plan(PROJECT_ROOT, args.medical_root, outer_fold=args.outer_fold,
                      dataset_id=args.dataset_id, seed=args.seed, run_root=args.run_root,
                      experiment_name=args.experiment_name, train_config=args.train_config,
                      recover_from=args.recover_from)
    if args.resume_preparation and args.recover_from is None:
        raise ValueError("--resume-preparation requires --recover-from")
    print(f"[SCOPE] {plan['scope']}\n[SEEDS] {plan['seed_note']}", flush=True)
    if args.dry_run:
        if plan["recovery_source_root"] is not None:
            identity = validate_recovery_source(plan, plan["recovery_source_root"])
            if args.resume_preparation:
                _load_preparation_journal(plan, identity)
            print(f"[recover_preparation] Read-only source validation passed: {plan['recovery_source_root']}; helper will prepare {plan['train_config']}")
        if args.resume_preparation:
            print(f"[DRY RUN ONLY] Would reverify and continue preparation in {plan['run_root']}; no runtime copy or training-checkpoint resume")
        else:
            print(f"[DRY RUN ONLY] Would create {plan['run_root']} and a private nnU-Net package copy")
        for command in plan["commands"]:
            print(f"[{command['name']}] {shlex.join(command['argv'])}")
        return
    try:
        execute_plan(plan, resume_preparation=args.resume_preparation)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        print(f"[FAILED] Further stages were not launched. Existing files and partial outputs are preserved: {plan['run_root']}",
              file=sys.stderr, flush=True)
        if plan["recovery_source_root"] is not None:
            print("Preparation-only retry: repeat with --resume-preparation after inspecting execution_journal.json. If training started or runtime setup was incomplete, automatic retry is refused; no checkpoints are restarted.", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
