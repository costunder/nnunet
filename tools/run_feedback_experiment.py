"""Start a NEW full-size feedback experiment without replacing existing trainers/results.

Preparation recovery uses a new work directory. Explicit --resume-experiment
continues its verified journal without repeating preparation. The launcher preserves the checked-in
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


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant)


def _bound_files(root, files):
    """Hash exact regular artifacts; never follow an escaping/symlinked receipt."""
    root = Path(root).resolve()
    result = {}
    for value in files:
        path = Path(value)
        if ".." in path.parts:
            raise ValueError(f"Parent traversal in experiment artifact: {path}")
        if not path.is_absolute():
            path = root / path
        if (not path.is_relative_to(root) or not path.resolve().is_relative_to(root)
                or any(part.is_symlink() for part in (path, *path.parents) if part != root.parent)
                or not path.is_file()):
            raise ValueError(f"Missing or unsafe experiment artifact: {path}")
        result[path.relative_to(root).as_posix()] = _file_sha256(path)
    return result


def _verify_preparation_receipt(plan, journal):
    root = plan["run_root"]
    receipt = journal["preparation_receipt"]
    if (not isinstance(receipt, dict)
            or receipt.get("format") != "hiercp_preparation_recovery_complete_v1"
            or receipt.get("train_config") != str(plan["train_config"])
            or receipt.get("source_identity") != journal["source_identity"]
            or receipt.get("original_results_preserved") is not True
            or receipt.get("training_performed") is not False
            or not isinstance(receipt.get("files"), dict) or not receipt["files"]):
        raise ValueError("Experiment resume requires the complete preparation receipt")
    receipt_path = root / "recovery/complete.json"
    _bound_files(root, [receipt_path])
    if _read_json(receipt_path) != receipt:
        raise ValueError("Preparation receipt changed since publication")
    # Legacy receipts used platform-native separators. Compare paths normalized
    # here but retain the original receipt/journal bytes unmodified.
    expected = {Path(name).as_posix(): sha for name, sha in receipt["files"].items()}
    if _bound_files(root, receipt["files"]) != expected:
        raise ValueError("Preparation input/config/cache receipt bytes changed")
    config_key = plan["train_config"].relative_to(root).as_posix()
    graphs = root / f"paired/folds/fold_{plan['outer_fold']}/gnn/graphs"
    required = [config_key, *[(graphs / name).relative_to(root).as_posix()
                              for name in ("config.json", "manifest.csv", "index.json", "complete.json")]]
    if any(name not in expected for name in required):
        raise ValueError("Preparation receipt does not bind the configured graph publication")
    from hiercp.cache import validate_cache_publication
    validate_cache_publication(graphs)  # all materialized/no-placement entries, no rebuild


def _resume_inputs(plan):
    files = [plan["train_config"], *[plan["project_root"] / "config" / name for name in
             ("nnunet.json", "online_cp_feedback.json", "online_cp_feedback_gnn.json")]]
    return _bound_files(plan["project_root"], files)


def _load_experiment_journal(plan, source_identity):
    root = plan["run_root"]
    path = root / "execution_journal.json"
    _bound_files(root, [path, root / "launch_plan.json"])
    payload = _read_json(path)
    keys = {"format", "plan_sha256", "source_identity", "runtime_inventory", "preparation_receipt",
            "stages", "training_started", "complete", "journal_sha256"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Malformed experiment execution journal")
    checksum = payload.pop("journal_sha256")
    if (payload["format"] != "feedback_preparation_execution_v1"
            or checksum != _json_sha256(payload) or payload["plan_sha256"] != _json_sha256(plan)
            or payload["source_identity"] != source_identity
            or _json_sha256(_read_json(root / "launch_plan.json")) != _json_sha256(plan)):
        raise ValueError("Experiment plan, source identity or journal checksum changed")
    if any(type(payload[key]) is not bool for key in ("training_started", "complete")):
        raise ValueError("Malformed experiment execution status")
    if not payload["runtime_inventory"] or _runtime_inventory(plan["package_destination"]) != payload["runtime_inventory"]:
        raise ValueError("Runtime setup is incomplete or private runtime bytes changed")
    stages = payload["stages"]
    if not isinstance(stages, list):
        raise ValueError("Malformed experiment stage history")
    sequence = [item["name"] for item in plan["commands"]
                if item["name"] not in {"install_private_trainers", "environment"}]
    preparation = {"copy_private_runtime", "install_private_trainers", "environment", "recover_preparation"}
    completed, entered_training = set(), False
    inputs = None
    for row in stages:
        if (not isinstance(row, dict) or row.get("name") not in preparation | set(sequence)
                or row.get("status") not in {"completed", "failed"}):
            raise ValueError("Unrecognized or still-running stage; active/interrupted attempts are ambiguous")
        name = row["name"]
        if name in preparation:
            if entered_training:
                raise ValueError("Preparation was re-entered after training began")
            if row["status"] == "completed":
                completed.add(name)
            continue
        entered_training = True
        position = len(completed - preparation)
        if (not preparation.issubset(completed) or position >= len(sequence)
                or name != sequence[position]):
            raise ValueError("Experiment stage order or completed preparation is invalid")
        if "input_files" in row:
            inputs = _resume_inputs(plan) if inputs is None else inputs
            if row["input_files"] != inputs:
                raise ValueError("Experiment configuration changed after its recorded attempt")
        if row["status"] == "completed":
            evidence = row.get("completion_evidence")
            if not isinstance(evidence, dict) or evidence != _stage_evidence(plan, name):
                raise ValueError(f"Completed stage lacks unchanged verified output evidence: {name}")
            completed.add(name)
    if not preparation.issubset(completed) or not entered_training or not payload["training_started"]:
        raise ValueError("Preparation is not complete; use the separate --resume-preparation workflow")
    if payload["complete"] and not set(sequence).issubset(completed):
        raise ValueError("Experiment completion flag disagrees with completed stages")
    _verify_preparation_receipt(plan, payload)
    return payload


def _training_folder(plan, arm):
    from tools.train_online_feedback import TRAINERS
    config = _read_json(plan["project_root"] / "config/nnunet.json")
    dataset = f"Dataset{plan['dataset_id']:03d}_LiverOnlineCP_OF{plan['outer_fold']}"
    return (Path(plan["env_updates"]["nnUNet_results"]) / dataset /
            f"{TRAINERS[arm]}__{config['dataset']['plans']}__{config['dataset']['configuration']}" / "fold_0")


def _checkpoint_payload(path):
    path = Path(path)
    if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"Missing or symlinked checkpoint is not a trusted experiment artifact: {path}")
    from hiercp.tensor import torch_load_compat
    payload = torch_load_compat(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a full state dictionary: {path}")
    return payload


def _resume_command(plan, command, *, previously_attempted):
    """Only select native resume paths; native loaders retain final identity checks."""
    name, argv = command["name"], list(command["argv"])
    root = plan["run_root"]
    if name == "gnn-train":
        gnn = root / f"paired/folds/fold_{plan['outer_fold']}/gnn"
        last, best, preflight = gnn / "model.last.pt", gnn / "model.pt", gnn / "model.pt.preflight.json"
        existing = [path for path in (last, best, preflight) if path.exists() or path.is_symlink()]
        _bound_files(root, existing)
        if existing and not previously_attempted:
            raise ValueError("Unjournaled GNN output exists; its ownership/attempt is ambiguous")
        if last.is_file():
            state = _checkpoint_payload(last)
            required = {"state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                        "rng_state", "train_shuffle_generator_state", "train_worker_generator_state",
                        "val_worker_generator_state", "training_signature", "preflight_calibration", "epoch", "target_epochs"}
            if state.get("format") != "hiercp_training_state_v1" or not required.issubset(state):
                raise ValueError("GNN checkpoint lacks complete optimizer/scheduler/RNG resume state")
            epochs = _read_json(plan["train_config"])["training"]["epochs"]
            if (type(state["epoch"]) is not int or not 1 <= state["epoch"] <= epochs
                    or state["target_epochs"] != epochs or not best.is_file() or not preflight.is_file()):
                raise ValueError("GNN checkpoint lacks its exact epoch, best model or saved calibration evidence")
        elif existing:
            raise ValueError("GNN partial checkpoint/preflight has no resumable model.last.pt; all files preserved")
        elif previously_attempted:
            training = _read_json(plan["train_config"])["training"]
            if training.get("batch_size") != "auto":
                raise ValueError("No GNN checkpoint and no mandatory auto-calibration boundary; cannot prove safe retry")
            # Auto calibration is durably published before real optimization.
            # No checkpoint AND no preflight is the failed-calibration boundary.
    elif name in {"train_full", "train_basic"}:
        arm = name.removeprefix("train_")
        folder = _training_folder(plan, arm)
        if folder.is_symlink():
            raise ValueError("nnU-Net output folder must not be a symlink")
        if folder.exists() and any(folder.iterdir()):
            if not previously_attempted:
                raise ValueError("Unjournaled nnU-Net output exists; its ownership/attempt is ambiguous")
            choices = [folder / f"checkpoint_{suffix}.pth" for suffix in ("final", "latest", "best")]
            checkpoint = next((path for path in choices if path.is_file()), None)
            if checkpoint is None:
                raise ValueError(f"Partial nnU-Net run has no checkpoint; no fresh-start fallback: {folder}")
            _bound_files(root, [checkpoint])
            state = _checkpoint_payload(checkpoint)
            extension = state.get("onlinecp_curriculum_resume", {})
            required = {"network_weights", "optimizer_state", "grad_scaler_state", "current_epoch"}
            resume_required = {"next_epoch", "config", "config_sha256", "bank_identity", "runtime_identity",
                               "last_epoch", "python_rng", "numpy_rng", "cpu_rng", "cuda_rng",
                               "lr_scheduler_state", "extension"}
            if (not required.issubset(state) or not isinstance(extension, dict)
                    or not resume_required.issubset(extension)
                    or extension.get("format") != "onlinecp_segmentation_feedback_resume_v1"):
                raise ValueError("nnU-Net checkpoint lacks complete feedback/optimizer/RNG resume state")
            argv.append("--resume")
        elif previously_attempted:
            raise ValueError("Attempted nnU-Net training has no checkpoint; fresh restart is not authorized")
    return argv


def _stage_evidence(plan, name):
    """Durable output evidence captured only after the native stage succeeds."""
    root, fold = plan["run_root"], plan["outer_fold"]
    gnn = root / f"paired/folds/fold_{fold}/gnn"
    bank = root / f"online/folds/fold_{fold}/bank"
    dataset = f"Dataset{plan['dataset_id']:03d}_LiverOnlineCP_OF{fold}"
    if name == "gnn-train":
        files = [gnn / value for value in ("model.pt", "model.last.pt", "model.pt.preflight.json",
                                          "causality.json", "causality.json.preflight.json")]
        state = _checkpoint_payload(gnn / "model.last.pt")
        epochs = _read_json(plan["train_config"])["training"]["epochs"]
        if state.get("training_complete") is not True or state.get("epoch") != epochs:
            raise ValueError("GNN stage returned without a completed full training checkpoint")
        if _read_json(gnn / "causality.json").get("status") != "complete":
            raise ValueError("GNN causality audit did not complete")
    elif name == "plan":
        _verify_online_artifacts(plan, name)
        pre = Path(plan["env_updates"]["nnUNet_preprocessed"]) / dataset
        nnconfig = _read_json(plan["project_root"] / "config/nnunet.json")
        files = [pre / value for value in ("online_cp_preprocess_complete.json", "splits_final.json",
                                            "dataset.json", f"{nnconfig['dataset']['plans']}.json")]
    elif name == "bank":
        _verify_online_artifacts(plan, name)
        files = [bank / value for value in ("index.json", "config.json", "manifest.csv", "complete.json")]
    elif name == "feedback_contract":
        _verify_online_artifacts(plan, name)
        files = [bank / "feedback_contract.json"]
    elif name in {"check_full", "check_basic"}:
        _verify_online_artifacts(plan, "feedback_contract")
        files = [bank / "feedback_contract.json"]
    elif name in {"train_full", "train_basic"}:
        identity = _verify_online_artifacts(plan, "feedback_contract")
        checkpoint = _training_folder(plan, name.removeprefix("train_")) / "checkpoint_final.pth"
        state = _checkpoint_payload(checkpoint)
        if (state.get("current_epoch") != 250
                or state.get("onlinecp_curriculum_resume", {}).get("bank_identity") != identity):
            raise ValueError("nnU-Net stage returned without the full 250-epoch final checkpoint")
        files = [checkpoint]
    else:
        raise ValueError(f"No completion evidence contract for stage {name}")
    return {"format": "feedback_stage_completion_v1", "files": _bound_files(root, files)}


def _verify_online_artifacts(plan, name):
    """Read-only native contract audit, including files bound by completion markers."""
    from tools import online_cp_benchmark as online
    relative = plan["run_root"].relative_to(plan["project_root"] / "work")
    layout = online.make_layout(argparse.Namespace(
        project_root=str(plan["project_root"]), medical_root=str(plan["medical_root"]),
        paired_root=(relative / "paired").as_posix(), online_root=(relative / "online").as_posix(),
        train_config=str(plan["train_config"])))
    train_cfg = _read_json(plan["train_config"])
    nn_cfg = _read_json(plan["project_root"] / "config/nnunet.json")
    if name == "plan":
        return online._verified_preprocess_contract(layout, plan["outer_fold"], train_cfg, nn_cfg, plan["dataset_id"])
    if name == "bank":
        return online._verified_bank_identity(layout, plan["outer_fold"], train_cfg, nn_cfg, plan["dataset_id"])
    from custom_trainers.onlinecp_curriculum_contract import verify_curriculum_bank_contract
    from custom_trainers.onlinecp_feedback_policy import validate_feedback_config, feedback_config_sha256
    config = validate_feedback_config(_read_json(plan["project_root"] / "config/online_cp_feedback.json"))
    previous = os.environ.get("nnUNet_preprocessed")
    try:
        os.environ["nnUNet_preprocessed"] = plan["env_updates"]["nnUNet_preprocessed"]
        return verify_curriculum_bank_contract(
            layout.bank(plan["outer_fold"]) / "index.json", curriculum_sha256=feedback_config_sha256(config),
            expected_candidate_count=128,
            dataset_name=f"Dataset{plan['dataset_id']:03d}_LiverOnlineCP_OF{plan['outer_fold']}",
            nnunet_fold=0, contract_filename="feedback_contract.json")
    finally:
        if previous is None:
            os.environ.pop("nnUNet_preprocessed", None)
        else:
            os.environ["nnUNet_preprocessed"] = previous


def _execute_experiment_resume(plan, identity, *, runner, env):
    root = plan["run_root"]
    lock = root / "recovery_execution.lock"
    lock_payload = json.dumps({"pid": os.getpid(), "token": uuid.uuid4().hex, "run_root": str(root)})
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(lock_payload)
    try:
        journal = _load_experiment_journal(plan, identity)  # race-safe recheck under lock
        history = root / "recovery/journal_history"
        if history.is_symlink():
            raise ValueError("Journal history must not be a symlink")
        history.mkdir(exist_ok=True)
        original = (root / "execution_journal.json").read_bytes()
        archive = history / (hashlib.sha256(original).hexdigest() + ".json")
        if archive.exists():
            if archive.is_symlink() or archive.read_bytes() != original:
                raise ValueError("Archived execution journal changed")
        else:
            with archive.open("xb") as handle:
                handle.write(original)
        for command in plan["commands"]:
            name = command["name"]
            if name in {"install_private_trainers", "environment"}:
                continue
            records = [row for row in journal["stages"] if row["name"] == name]
            if any(row["status"] == "completed" for row in records):
                print(f"[VERIFIED SKIP {name}] completed artifacts unchanged", flush=True)
                continue
            argv = _resume_command(plan, command, previously_attempted=bool(records))
            record = {"name": name, "status": "running", "input_files": _resume_inputs(plan),
                      "attempt_id": uuid.uuid4().hex, "argv": argv}
            journal["stages"].append(record)
            journal["training_started"] = True
            _save_journal(root, journal)
            try:
                print(f"\n[CONTINUE {name}] {shlex.join(argv)}", flush=True)
                runner(argv, cwd=plan["project_root"], env=env, check=True)
                record["completion_evidence"] = _stage_evidence(plan, name)
            except (Exception, KeyboardInterrupt) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                _save_journal(root, journal)
                raise
            record["status"] = "completed"
            _save_journal(root, journal)
        journal["complete"] = True
        _save_journal(root, journal)
    finally:
        if lock.read_text(encoding="utf-8") != lock_payload:
            raise RuntimeError("Recovery execution lock ownership changed; it was not removed")
        lock.unlink()


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
            if training:
                record["input_files"] = _resume_inputs(plan)
            journal["stages"].append(record)
            _save_journal(root, journal)
            try:
                value = action()
                if training:
                    record["completion_evidence"] = _stage_evidence(plan, name)
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
    """Copy native nnU-Net into a new runtime, preserving an older CP install.

    Source CP helpers may belong to the old experiment and need not match the
    current MODULES hashes. Exclude those exact helper names only; the installer
    writes the current helpers exclusively inside this new destination. Resume
    of an existing runtime still requires its original verified inventory.
    """
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


def execute_plan(plan, *, runner=None, package_root=None, resume_preparation=False, resume_experiment=False):
    runner = subprocess.run if runner is None else runner
    root, medical, project = (plan[key] for key in ("run_root", "medical_root", "project_root"))
    recovering = plan.get("recovery_source_root") is not None
    if resume_preparation and resume_experiment:
        raise ValueError("Choose --resume-preparation or --resume-experiment, not both")
    if resume_experiment and not recovering:
        raise ValueError("--resume-experiment requires the original --recover-from and experiment identity")
    if resume_preparation and not recovering:
        raise ValueError("--resume-preparation requires --recover-from; it is not training resume")
    if root.is_symlink() or ((root.exists()) and not (resume_preparation or resume_experiment)):
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
        if resume_experiment:
            if (root / "recovery_execution.lock").exists():
                raise FileExistsError("Experiment execution lock exists; no active or ambiguous attempt is resumed")
            journal = _load_experiment_journal(plan, source_identity)
    for folder in ("Data/image", "Data/labels"):
        if not (medical / folder).is_dir():
            raise FileNotFoundError(f"Required real-data directory is missing: {medical / folder}")
    # Existing ancestors may include a separate work volume. Inspect that volume.
    storage = root.parent
    while not storage.exists():
        storage = storage.parent
    if shutil.disk_usage(storage).free < plan["minimum_free_bytes"]:
        raise RuntimeError("Insufficient free space for the configured preprocessing contract")
    source = (plan["package_destination"] if resume_experiment else
              Path(package_root).resolve() if package_root is not None else locate_nnunet_root(None))
    if not (source / "training/nnUNetTrainer").is_dir():
        raise ValueError(f"Not an nnunetv2 package: {source}")
    if not resume_experiment and (root == source or root.is_relative_to(source) or source.is_relative_to(root)):
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
    if not (resume_preparation or resume_experiment):
        root.mkdir(parents=True, exist_ok=False)
        print(f"[NEW EXPERIMENT] {root}", flush=True)
        with (root / "launch_plan.json").open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, default=str, indent=2)
    if resume_experiment:
        _execute_experiment_resume(plan, source_identity, runner=runner, env=env)
    elif recovering:
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
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-preparation", action="store_true", help="Reverify and continue this recovery's preparation journal only, never training checkpoints")
    resume.add_argument("--resume-experiment", action="store_true", help="Continue the same verified experiment after preparation; preserve caches and use only native checkpoint resume")
    parser.add_argument("--dry-run", action="store_true", help="Print only: no directories, copying, GPU checks or child commands")
    args = parser.parse_args(argv)
    plan = build_plan(PROJECT_ROOT, args.medical_root, outer_fold=args.outer_fold,
                      dataset_id=args.dataset_id, seed=args.seed, run_root=args.run_root,
                      experiment_name=args.experiment_name, train_config=args.train_config,
                      recover_from=args.recover_from)
    if args.resume_preparation and args.recover_from is None:
        raise ValueError("--resume-preparation requires --recover-from")
    if args.resume_experiment and args.recover_from is None:
        raise ValueError("--resume-experiment requires the original --recover-from")
    scope = ("Continue this verified experiment: preserve completed preparation, quality GNN then Full/Basic; no overwrite or fresh training fallback."
             if args.resume_experiment else plan["scope"])
    print(f"[SCOPE] {scope}\n[SEEDS] {plan['seed_note']}", flush=True)
    if args.dry_run:
        if plan["recovery_source_root"] is not None:
            identity = validate_recovery_source(plan, plan["recovery_source_root"])
            if args.resume_preparation:
                _load_preparation_journal(plan, identity)
            if args.resume_experiment:
                journal = _load_experiment_journal(plan, identity)
                print("[VERIFIED PREPARATION] Existing cache/receipt/runtime checked; no preparation commands will run")
            else:
                print(f"[recover_preparation] Read-only source validation passed: {plan['recovery_source_root']}; helper will prepare {plan['train_config']}")
        if args.resume_experiment:
            print(f"[DRY RUN ONLY] Would continue {plan['run_root']}; no writes or child commands")
        elif args.resume_preparation:
            print(f"[DRY RUN ONLY] Would reverify and continue preparation in {plan['run_root']}; no runtime copy or training-checkpoint resume")
        else:
            print(f"[DRY RUN ONLY] Would create {plan['run_root']} and a private nnU-Net package copy")
        for command in plan["commands"]:
            if args.resume_experiment:
                if command["name"] in {"install_private_trainers", "environment"}:
                    continue
                records = [row for row in journal["stages"] if row["name"] == command["name"]]
                if any(row["status"] == "completed" for row in records):
                    print(f"[VERIFIED SKIP {command['name']}]")
                    continue
                command = {**command, "argv": _resume_command(plan, command, previously_attempted=bool(records))}
            print(f"[{command['name']}] {shlex.join(command['argv'])}")
        return
    try:
        execute_plan(plan, resume_preparation=args.resume_preparation, resume_experiment=args.resume_experiment)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        print(f"[FAILED] Further stages were not launched. Existing files and partial outputs are preserved: {plan['run_root']}",
              file=sys.stderr, flush=True)
        if plan["recovery_source_root"] is not None:
            print("Retry options: --resume-preparation is only for incomplete preparation. Once preparation completed, use the SAME original arguments plus --resume-experiment; it verifies cache/receipt/runtime and refuses ambiguous attempts or missing/incompatible training checkpoints. Do not edit the journal, delete preflight files, or use --overwrite. Partial runtime setup still requires a separate new experiment.", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
