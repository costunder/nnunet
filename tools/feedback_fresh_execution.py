"""Journaled fresh feedback execution, using the shared native stage executor.

Existing legacy roots without this journal are not silently adopted. Installed
runtime, every completed stage, input configuration, and native child receipts
are reverified; no missing training checkpoint is converted into a fresh run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import uuid

from tools.feedback_stage_execution import (
    execute_command_stage, owner_identity, run_lock, verified_child_result,
)

FORMAT = "feedback_fresh_execution_v1"


def _launch():
    from tools import run_feedback_experiment
    return run_feedback_experiment


def native_inventory(package):
    """Copied native package bytes, excluding only the designated CP helpers."""
    launch = _launch()
    package = Path(package)
    if not package.is_dir() or package.is_symlink():
        raise ValueError("Missing native package for fresh execution")
    output = {}
    excluded = {"__pycache__", ".git", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if any(part in excluded for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Unverified native package symlink: {path}")
        if (relative.parent.as_posix() == "training/nnUNetTrainer"
                and relative.name in launch.MODULES):
            continue
        if path.is_file():
            output[relative.as_posix()] = launch._file_sha256(path)
    if not {"__init__.py", "training/nnUNetTrainer/nnUNetTrainer.py"}.issubset(output):
        raise ValueError("Native package inventory lacks required implementation files")
    return output


def _sequence(plan):
    return ["copy_private_runtime", *[item["name"] for item in plan["commands"]]]


def _evidence(plan, name):
    launch = _launch()
    if name == "copy_private_runtime":
        return {"format": "feedback_native_package_copy_v1",
                "native_files": native_inventory(plan["package_destination"])}
    return launch._stage_evidence(plan, name)


def load_journal(plan, *, allow_debug=False, reconcile=False):
    launch = _launch()
    root = Path(plan["run_root"])
    launch._bound_files(root, [root / "launch_plan.json", root / "execution_journal.json"])
    saved = launch._read_json(root / "launch_plan.json")
    journal = launch._read_json(root / "execution_journal.json")
    keys = {"format", "plan_sha256", "source_identity", "input_files", "stages",
            "runtime_inventory", "training_started", "complete", "journal_sha256"}
    if not isinstance(journal, dict) or set(journal) != keys:
        raise ValueError("Fresh resume needs its own complete journal; legacy unjournaled roots are preserved")
    checksum = journal.pop("journal_sha256")
    if (journal["format"] != FORMAT or checksum != launch._json_sha256(journal)
            or journal["plan_sha256"] != launch._json_sha256(plan)
            or launch._json_sha256(saved) != launch._json_sha256(plan)
            or journal["input_files"] != launch._resume_inputs(plan)):
        raise ValueError("Fresh plan/configuration/journal changed; no inputs were overwritten")
    if (not isinstance(journal["source_identity"], dict)
            or set(journal["source_identity"]) != {"native_package", "native_files"}
            or not isinstance(journal["source_identity"]["native_files"], dict)
            or not isinstance(journal["stages"], list)
            or any(type(journal[key]) is not bool for key in ("training_started", "complete"))):
        raise ValueError("Malformed fresh source/stage status")
    sequence, position = _sequence(plan), 0
    for row in journal["stages"]:
        if (not isinstance(row, dict) or position >= len(sequence)
                or row.get("name") != sequence[position]
                or row.get("status") not in {"running", "failed", "completed"}
                or row.get("input_files") != journal["input_files"]):
            raise ValueError("Fresh stage ordering, configuration, or status is invalid")
        if row.get("execution_backend") == "injected_debug_runner" and not allow_debug:
            raise ValueError("DEBUG runner receipts cannot authorize a production experiment")
        if row["status"] == "running":
            if not reconcile:
                raise ValueError("Unfinished stage needs locked child-receipt reconciliation, not another native launch")
            if row["name"] == "copy_private_runtime":
                destination = Path(plan["package_destination"])
                if destination.exists():
                    evidence = _evidence(plan, row["name"])
                    if evidence["native_files"] != journal["source_identity"]["native_files"]:
                        raise ValueError("Incomplete or changed native copy preserved")
                    row.update(status="completed", completion_evidence=evidence,
                               reconciliation="verified atomic native copy")
                else:
                    row.update(status="failed", error="Local copy interrupted before publication; staging preserved")
            else:
                terminal = verified_child_result(plan, row)
                if terminal["returncode"]:
                    row.update(status="failed", error=f"Native child exited with {terminal['returncode']}")
                else:
                    row.update(status="completed", completion_evidence=_evidence(plan, row["name"]),
                               reconciliation="verified native exit receipt and current complete artifacts")
            launch._save_journal(root, journal)
        if row["status"] == "completed":
            if row.get("execution_backend") == "native_receipted" and verified_child_result(plan, row)["returncode"] != 0:
                raise ValueError("Completed fresh stage has a failed native child receipt")
            evidence = _evidence(plan, row["name"])
            if row.get("completion_evidence") != evidence:
                raise ValueError(f"Completed fresh stage proof changed: {row['name']}")
            if row["name"] == "copy_private_runtime" and evidence["native_files"] != journal["source_identity"]["native_files"]:
                raise ValueError("Copied native implementation differs from its recorded source")
            position += 1
    installed = any(row["name"] == "install_private_trainers" and row["status"] == "completed"
                    for row in journal["stages"])
    if installed:
        actual = launch._runtime_inventory(plan["package_destination"])
        if journal["runtime_inventory"] != actual:
            # A publisher may have completed before the journal field was set.
            if reconcile and journal["runtime_inventory"] is None:
                journal["runtime_inventory"] = actual
                launch._save_journal(root, journal)
            else:
                raise ValueError("Installed private runtime changed")
    if journal["complete"] and position != len(sequence):
        raise ValueError("Fresh completion flag lacks all stage proofs")
    return journal


def _copy_runtime(plan, source, journal):
    launch = _launch()
    root, destination = Path(plan["run_root"]), Path(plan["package_destination"])
    if native_inventory(source) != journal["source_identity"]["native_files"]:
        raise ValueError("Original native runtime changed before its copy completed")
    row = {"name": "copy_private_runtime", "status": "running", "attempt_id": uuid.uuid4().hex,
           "owner": owner_identity(), "input_files": journal["input_files"]}
    journal["stages"].append(row)
    launch._save_journal(root, journal)
    staging = root / "execution_attempts" / row["attempt_id"] / "native_package"
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Uncommitted runtime destination is preserved: {destination}")
        launch.copy_nnunet_package(source, staging)
        if native_inventory(staging) != journal["source_identity"]["native_files"]:
            raise ValueError("Staged runtime copy failed complete content verification")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Runtime destination appeared during publication")
        # Within this exclusively locked, newly-owned run root, no cooperating
        # writer can create destination between verification and rename.
        os.rename(staging, destination)
        row["completion_evidence"] = _evidence(plan, row["name"])
    except (Exception, KeyboardInterrupt) as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        launch._save_journal(root, journal)
        raise
    row["status"] = "completed"
    launch._save_journal(root, journal)


def execute_fresh(plan, source, *, runner, env, resume=False):
    launch = _launch()
    root = Path(plan["run_root"])
    with run_lock(root):
        if resume:
            journal = load_journal(plan, allow_debug=runner is not subprocess.run, reconcile=True)
        else:
            if (root / "execution_journal.json").exists():
                raise FileExistsError("Existing fresh journal must be resumed explicitly")
            journal = {"format": FORMAT, "plan_sha256": launch._json_sha256(plan),
                       "source_identity": {"native_package": str(source), "native_files": native_inventory(source)},
                       "input_files": launch._resume_inputs(plan), "stages": [],
                       "runtime_inventory": None, "training_started": False, "complete": False}
            launch._save_journal(root, journal)
        complete = {row["name"] for row in journal["stages"] if row["status"] == "completed"}
        if "copy_private_runtime" not in complete:
            original = Path(journal["source_identity"]["native_package"])
            _copy_runtime(plan, original, journal)
        for command in plan["commands"]:
            name = command["name"]
            records = [row for row in journal["stages"] if row["name"] == name]
            if any(row["status"] == "completed" for row in records):
                print(f"[VERIFIED SKIP {name}] unchanged complete artifacts", flush=True)
                continue
            # A failed attempt with a real permitted child must have a terminal
            # receipt before any native resume command can be considered.
            for row in records:
                attempt = root / "execution_attempts" / row["attempt_id"]
                if row.get("execution_backend") != "injected_debug_runner" and (attempt / "permit.json").exists():
                    verified_child_result(plan, row)
            execute_command_stage(plan, journal, command, runner=runner, env=env,
                                  previously_attempted=bool(records))
            if name == "install_private_trainers":
                journal["runtime_inventory"] = launch._runtime_inventory(plan["package_destination"])
                launch._save_journal(root, journal)
        journal["complete"] = True
        launch._save_journal(root, journal)
