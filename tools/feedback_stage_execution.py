"""Durable stage attempts shared by fresh/recovery/upgrade feedback runs.

The child writes an immutable exit receipt itself. A dead launcher is not proof
that its native training child stopped; missing/ambiguous receipts fail closed.
No process is killed, no checkpoint is replaced, and no environment is serialized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time
import uuid

import psutil


def _read(path):
    from tools.run_feedback_experiment import _read_json
    if any(item.is_symlink() for item in (Path(path), *Path(path).parents)):
        raise ValueError(f"Symlinked stage receipt: {path}")
    return _read_json(path)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def publish_new_json(path, value):
    """Complete self-owned staging, then atomic no-clobber publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlinked publication path: {path}")
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".staging")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    if _read(temporary) != value:
        raise ValueError(f"Staged JSON failed readback: {temporary}")
    os.link(temporary, path)  # Never replace a user's final artifact.
    # Keep the attempt's staged evidence, including on failed publication.


def owner_identity():
    return {"host": socket.gethostname(), "pid": os.getpid(),
            "process_started": psutil.Process().create_time()}


@contextmanager
def run_lock(root, *, create=True):
    """OS-owned advisory lock: process death releases it, no stale-lock deletion.

    This does not authorize resuming an orphan child: exit receipts below are
    checked separately. Unsupported filesystem locking is an explicit error.
    """
    path = Path(root) / "feedback_execution.lock"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Execution lock path must not contain symlinks")
    # Read-only source verification must never create/initialize a source lock.
    # Existing lock bytes are untouched; only the OS advisory lock is acquired.
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "r+b", buffering=0) as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Execution lock is not a regular file")
        if os.fstat(handle.fileno()).st_size == 0:
            if not create:
                raise ValueError("Existing execution lock is empty; source verification cannot initialize it")
            handle.write(b" ")
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _attempt_path(plan, row):
    root = Path(plan["run_root"]).resolve()
    token = row.get("attempt_id")
    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid stage attempt identity")
    path = root / "execution_attempts" / token
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Stage attempt path contains a symlink")
    return path


def verified_child_result(plan, row):
    """Return only producer-written terminal state, never infer it from a PID."""
    path = _attempt_path(plan, row)
    spec = _read(path / "attempt.json")
    if spec.get("argv") != row.get("argv") or spec.get("attempt_id") != row["attempt_id"]:
        raise ValueError("Stage argv and attempt receipt disagree")
    result_path = path / "child_complete.json"
    if not result_path.is_file():
        raise ValueError(f"Stage child has no terminal receipt; it may still be running. Preserved: {path}")
    result = _read(result_path)
    started = _read(path / "child_started.json")
    permit = _read(path / "permit.json")
    if (set(result) != {"format", "attempt_sha256", "child", "returncode"}
            or result["format"] != "feedback_child_complete_v1"
            or result["attempt_sha256"] != _sha(spec)
            or result["child"] != started
            or permit != {"attempt_sha256": _sha(spec), "child_pid": started["pid"]}
            or type(result["returncode"]) is not int):
        raise ValueError("Malformed or mismatched stage child completion")
    return result


def run_owned_command(plan, row, *, runner, env):
    if runner is not subprocess.run:
        # Explicit dependency injection is only for DEBUG tests. This marker
        # cannot serve as a real child receipt on a production resume.
        row["execution_backend"] = "injected_debug_runner"
        return runner(row["argv"], cwd=plan["project_root"], env=env, check=True)
    path = _attempt_path(plan, row)
    path.mkdir(parents=True, exist_ok=False)
    spec = {"format": "feedback_child_attempt_v1", "attempt_id": row["attempt_id"],
            "argv": row["argv"], "cwd": str(plan["project_root"]), "parent": owner_identity()}
    publish_new_json(path / "attempt.json", spec)
    command = [plan["python_executable"], "-B", "-m", "tools.feedback_stage_execution", "--worker", str(path)]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    child = subprocess.Popen(command, cwd=plan["project_root"], env=env,
                             stdin=subprocess.DEVNULL, **options)
    launched = psutil.Process(child.pid)
    launched_identity = (launched.pid, launched.create_time())
    started_path = path / "child_started.json"
    while not started_path.is_file():
        if child.poll() is not None:
            raise RuntimeError("Stage wrapper stopped before its identity handshake; native work was not authorized")
        time.sleep(0.05)
    started = _read(started_path)
    if (not isinstance(started, dict) or set(started) != {"host", "pid", "process_started"}
            or started["host"] != socket.gethostname() or type(started["pid"]) is not int):
        raise ValueError("Malformed native worker identity handshake")
    worker_process = psutil.Process(started["pid"])
    if worker_process.create_time() != started["process_started"]:
        raise ValueError("Native worker identity was reused before authorization")
    ancestor = worker_process
    while ancestor is not None and ancestor.pid != launched_identity[0]:
        ancestor = ancestor.parent()
    if ancestor is None or ancestor.create_time() != launched_identity[1]:
        raise ValueError("Native worker is not the launched process or its verified descendant")
    # The worker cannot execute native work until this durable permit exists.
    # Windows venv launchers may create a second interpreter process. Bind the
    # verified worker identity, not an assumed equality with the launcher PID.
    publish_new_json(path / "permit.json", {"attempt_sha256": _sha(spec), "child_pid": started["pid"]})
    returncode = child.wait()
    terminal = verified_child_result(plan, row)
    if returncode != 0 or terminal["returncode"] != 0:
        raise subprocess.CalledProcessError(terminal["returncode"] or returncode, row["argv"])


def execute_command_stage(plan, journal, command, *, runner, env, previously_attempted=False):
    from tools import run_feedback_experiment as launch
    if previously_attempted:
        for previous in journal["stages"]:
            if previous["name"] == command["name"] and previous.get("execution_backend") == "native_receipted":
                path = _attempt_path(plan, previous)
                if (path / "permit.json").exists():
                    verified_child_result(plan, previous)
    argv = launch._resume_command(plan, command, previously_attempted=previously_attempted)
    row = {"name": command["name"], "status": "running", "argv": argv,
           "attempt_id": uuid.uuid4().hex, "owner": owner_identity(),
           "input_files": launch._resume_inputs(plan),
           "execution_backend": "native_receipted" if runner is subprocess.run else "injected_debug_runner"}
    journal["stages"].append(row)
    if "training_started" in journal and row["name"] not in {"split", "gnn-prepare", "environment", "install_private_trainers",
                                                            "basic_source_preflight", "basic_reuse"}:
        journal["training_started"] = True
    launch._save_journal(plan["run_root"], journal)
    try:
        print(f"[STAGE {row['name']}] {subprocess.list2cmdline(argv)}", flush=True)
        run_owned_command(plan, row, runner=runner, env=env)
        row["completion_evidence"] = launch._stage_evidence(plan, row["name"])
    except (Exception, KeyboardInterrupt) as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        launch._save_journal(plan["run_root"], journal)
        raise
    row["status"] = "completed"
    launch._save_journal(plan["run_root"], journal)
    return row


def reconcile_history(plan, journal, *, allow_debug=False):
    """In-memory reconciliation; caller publishes the journal only under lock.

    Historical unreceipted attempts remain ambiguous. New producer receipts
    prove native termination; completion additionally needs the native artifact
    verifier. This never restarts a command or edits an existing artifact.
    """
    from tools import run_feedback_experiment as launch
    for row in journal["stages"]:
        backend = row.get("execution_backend")
        if backend == "injected_debug_runner" and not allow_debug:
            raise ValueError("DEBUG native-runner receipts cannot authorize production resume")
        if row.get("status") == "running":
            if backend != "native_receipted":
                raise ValueError("Active or interrupted legacy attempt has no native child receipt")
            terminal = verified_child_result(plan, row)
            if terminal["returncode"]:
                row.update(status="failed", error=f"Native child exited with {terminal['returncode']}",
                           reconciliation="producer-written terminal receipt")
            else:
                row.update(status="completed", completion_evidence=launch._stage_evidence(plan, row["name"]),
                           reconciliation="producer exit and current artifact verification")
        elif row.get("status") == "completed" and backend == "native_receipted":
            if verified_child_result(plan, row)["returncode"] != 0:
                raise ValueError("Completed stage has a failed native child receipt")


def worker(path):
    path = Path(path).resolve()
    spec = _read(path / "attempt.json")
    if spec.get("format") != "feedback_child_attempt_v1" or spec.get("attempt_id") != path.name:
        raise ValueError("Invalid native stage attempt")
    child = owner_identity()
    publish_new_json(path / "child_started.json", child)
    permit_path = path / "permit.json"
    while not permit_path.is_file():
        parent = spec["parent"]
        if parent["host"] != socket.gethostname():
            raise ValueError("Stage parent host changed")
        try:
            live = psutil.Process(parent["pid"]).create_time() == parent["process_started"]
        except psutil.NoSuchProcess:
            live = False
        if not live:
            raise RuntimeError("Launcher disappeared before authorizing native work; no native command was started")
        time.sleep(0.05)
    if _read(permit_path) != {"attempt_sha256": _sha(spec), "child_pid": os.getpid()}:
        raise ValueError("Native stage permit does not identify this exact attempt")
    result = subprocess.run(spec["argv"], cwd=spec["cwd"], check=False)
    publish_new_json(path / "child_complete.json", {
        "format": "feedback_child_complete_v1", "attempt_sha256": _sha(spec),
        "child": child, "returncode": result.returncode})
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, spec["argv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    worker(parser.parse_args().worker)
