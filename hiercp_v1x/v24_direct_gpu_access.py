"""Explicit existing direct GPU4/5 availability guard; no PBS ownership claim.

The user requested the previous direct execution mode. Other PBS reservations
are known not to be owned by this account. This guard establishes only current
device availability, exact own process identity and an independent container
lifetime bound. It never reserves a GPU, starts/signals a process, writes a
receipt, reduces an experiment, or relabels GPU6's PBS allocation as GPU4/5's.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import psutil

FORMAT = "v24_existing_direct_GPU_matched_stage_availability_guard_v2"
STAGE_REQUEST_FORMAT = "v24_matched_direct_GPU_stage_request_v1"
DIRECT_MODE = "previous_explicit_direct_physical_GPU_number"
STAGE_ACTIONS = ("pin-current-gnn", "prepare-current-bank", "prepare-native", "calibrate-native", "train")
AFFINITIES = {4: [0, 1, 2, 3], 5: [42, 43, 44, 45]}
LIFETIME_ROLE = "host_container_lifetime_only_not_GPU45_PBS_allocation"


class DirectGPUAccessUnavailable(RuntimeError):
    """Fail cooperatively at the caller's safe checkpoint; never signal anyone."""


def _uid():
    if not hasattr(os, "getuid"):
        raise DirectGPUAccessUnavailable("Actual Linux process/file UID required for direct server execution")
    return os.getuid()


def _owned(path):
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError("Absolute non-traversing proof/request path required")
    if any(part.is_symlink() for part in (value, *value.parents)):
        raise ValueError("Symlink proof/request alias refused")
    info = value.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != _uid():
        raise ValueError("Own regular immutable proof/request/guard file required")
    return value


def _sha(path):
    return hashlib.sha256(_owned(path).read_bytes()).hexdigest()


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _validate_proof_document(proof, gpu, CPU_affinity):
    if type(gpu) is not int or gpu not in AFFINITIES or CPU_affinity != AFFINITIES[gpu]:
        raise ValueError("Original direct GPU4/5 CPU assignment required")
    keys = {"format", "GPU_arm", "GPU_UUID", "CPU_affinity", "UID", "PBS_owned_reservation",
            "other_PBS_reservations_known_not_owned", "explicit_existing_direct_execution_requested",
            "guard_sha256", "request", "host_container_lifetime", "poll_seconds",
            "minimum_initial_remaining_seconds", "native_minimum_initial_remaining_seconds", "expiry_safety_reserve_seconds"}
    if (not isinstance(proof, dict) or set(proof) != keys or proof.get("format") != FORMAT or proof.get("GPU_arm") != gpu
            or proof.get("CPU_affinity") != CPU_affinity or proof.get("UID") != 41833 or _uid() != 41833
            or proof.get("PBS_owned_reservation") is not False
            or proof.get("other_PBS_reservations_known_not_owned") is not True
            or proof.get("explicit_existing_direct_execution_requested") is not True
            or not isinstance(proof.get("GPU_UUID"), str) or not proof["GPU_UUID"].startswith("GPU-")
            or proof.get("poll_seconds") != 30
            or type(proof.get("minimum_initial_remaining_seconds")) not in (int, float)
            or not math.isfinite(proof["minimum_initial_remaining_seconds"])
            or proof["minimum_initial_remaining_seconds"] < 28800
            or type(proof.get("native_minimum_initial_remaining_seconds")) not in (int, float)
            or not math.isfinite(proof["native_minimum_initial_remaining_seconds"])
            or proof["native_minimum_initial_remaining_seconds"] < 25200
            or type(proof.get("expiry_safety_reserve_seconds")) not in (int, float)
            or not math.isfinite(proof["expiry_safety_reserve_seconds"])
            or proof["expiry_safety_reserve_seconds"] < 180):
        raise ValueError("Honest explicit direct GPU availability/container lifetime proof required; no PBS GPU4/5 claim")
    lifetime = proof["host_container_lifetime"]
    if (not isinstance(lifetime, dict) or set(lifetime) != {"role", "expires_at_unix", "PBS_job_id", "allocation_process"}
            or lifetime["role"] != LIFETIME_ROLE or lifetime["PBS_job_id"] != "129528.ECE-util1"
            or type(lifetime["expires_at_unix"]) not in (int, float) or not math.isfinite(lifetime["expires_at_unix"])
            or not _valid_identity(lifetime["allocation_process"])):
        raise ValueError("Exact separate host lifetime PID/birth/argv/UID witness required; never GPU4/5 PBS ownership")

def stage_request_document(request):
    """Seal the full declaration before adding its separate proof witness.

    The exclusion is one explicit witness field; all scientific and allocation
    declarations remain in this new stage request. No files are written here.
    """
    if not isinstance(request, dict):
        raise ValueError("Actual controller request declaration required")
    return dict(format=STAGE_REQUEST_FORMAT, controller_request_declaration={
        key: value for key, value in request.items() if key != "direct_GPU_access"})


def _stage_declaration(stage, path, gpu, CPU_affinity):
    if (not isinstance(stage, dict) or set(stage) != {"format", "controller_request_declaration"}
            or stage["format"] != STAGE_REQUEST_FORMAT):
        raise ValueError("New matched downstream stage declaration required; original GNN proof cannot be reused")
    request = stage["controller_request_declaration"]
    if isinstance(request, dict) and request.get("format") == "v24_scored_bank_to_native_chain_request_v1":
        from .v24_scored_native_chain import validate_declaration
        validate_declaration(request)
        if ("direct_GPU_access" in request or request.get("GPU_arm") != gpu
                or request.get("CPU_affinity") != CPU_affinity
                or Path(request["chain_root"]) != Path(path).parent
                or Path(path).name != "direct_GPU_stage_request.json"):
            raise ValueError("Exact new native-only direct stage declaration required")
        return request
    if (not isinstance(request, dict) or "direct_GPU_access" in request or "context_depth1_direct_allocation_proof" in request
            or request.get("direct_mode") != DIRECT_MODE or request.get("PBS_owned_reservation") is not False
            or request.get("context_depth1") is not True or request.get("scheduler") is not None
            or request.get("GPU_arm") != gpu or request.get("CPU_affinity") != CPU_affinity
            or request.get("source_job", {}).get("format") != "v24_interrupted_direct_GNN_source_job_v1"
            or request.get("storage_profile") != "reuse_historical_Basic642_scores_only_and_full131_Blosc2_readonly"
            or request.get("scoring_RAM_GiB") != 64 or request.get("native_RAM_GiB") != 48
            or request.get("minimum_runtime_free_disk_GiB", 10) < 10
            or not isinstance(request.get("GPU_UUID"), str) or not request["GPU_UUID"].startswith("GPU-")
            or not isinstance(request.get("source_output"), str) or not isinstance(request.get("source_code"), str)
            or not isinstance(request.get("stages"), list)
            or tuple(row.get("action") for row in request["stages"]) != STAGE_ACTIONS
            or Path(request.get("chain_root", "")) != Path(path).parent
            or Path(path).name != "direct_GPU_stage_request.json"):
        raise ValueError("Full current direct-source/GPU4/5 matched642 depth1 controller declaration required")
    return request


def _file_stat(path):
    value = _owned(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def validate_controller_request(request):
    """Read-only CPU admission of a distinct, SHA-pinned downstream proof.

    GPU idle checks and process ownership are established by the actual stage
    child when make_guard is called, rather than claimed by this metadata check.
    """
    witness = request.get("direct_GPU_access")
    if (not isinstance(witness, dict) or set(witness) != {"path", "sha256", "stat"}
            or not _checksum(witness["sha256"])):
        raise ValueError("Exact new downstream direct proof path/raw SHA/stat required")
    path = _owned(witness["path"])
    root = Path(request["chain_root"])
    if (path.parent != root or path.name != "direct_GPU_access_proof.json"
            or _file_stat(path) != witness["stat"] or _sha(path) != witness["sha256"]):
        raise ValueError("New downstream direct proof path/stat/SHA changed")
    proof = json.loads(path.read_text(encoding="utf8"))
    _validate_proof_document(proof, request.get("GPU_arm"), request.get("CPU_affinity"))
    if (proof.get("format") != FORMAT or proof.get("GPU_arm") != request.get("GPU_arm")
            or proof.get("GPU_UUID") != request.get("GPU_UUID")
            or proof.get("CPU_affinity") != request.get("CPU_affinity")
            or proof.get("PBS_owned_reservation") is not False
            or proof.get("other_PBS_reservations_known_not_owned") is not True
            or proof.get("explicit_existing_direct_execution_requested") is not True
            or proof.get("guard_sha256") != _sha(Path(__file__).resolve())):
        raise ValueError("Honest tracked downstream direct access proof/implementation differs")
    source = proof.get("request")
    if not isinstance(source, dict) or set(source) != {"path", "sha256"} or not _checksum(source["sha256"]):
        raise ValueError("New full stage request raw SHA binding required")
    declaration_path = _owned(source["path"])
    if _sha(declaration_path) != source["sha256"]:
        raise ValueError("New stage request SHA changed")
    stage = json.loads(declaration_path.read_text(encoding="utf8"))
    _stage_declaration(stage, declaration_path, request["GPU_arm"], request["CPU_affinity"])
    if stage != stage_request_document(request):
        raise ValueError("Every sealed controller declaration field must match the new direct stage request")
    return dict(proof=proof, proof_path=str(path), proof_sha256=witness["sha256"],
                stage_request_path=str(declaration_path), stage_request_sha256=source["sha256"],
                PBS_owned_reservation=False, actual_GPU_idle_check_performed=False)


def _command(command):
    return (isinstance(command, list) and bool(command)
            and all(isinstance(arg, str) and "\x00" not in arg for arg in command))


def _identity(process):
    if process.status() == psutil.STATUS_ZOMBIE:
        raise DirectGPUAccessUnavailable("Managed process is a zombie")
    return dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(), UID=process.uids().real)


def _valid_identity(identity):
    return not (not isinstance(identity, dict) or set(identity) != {"pid", "create_time", "command", "UID"}
            or type(identity["pid"]) is not int or identity["pid"] <= 0
            or type(identity["create_time"]) not in (int, float) or not math.isfinite(identity["create_time"])
            or identity["create_time"] <= 0 or identity["UID"] != 41833 or not _command(identity["command"]))


def _exact(identity):
    if not _valid_identity(identity):
        raise ValueError("Exact PID/birth/argv/UID identity required")
    try:
        process = psutil.Process(identity["pid"])
        actual = _identity(process)
    except psutil.NoSuchProcess as error:
        raise DirectGPUAccessUnavailable("Exact managed/container process disappeared") from error
    if actual != identity or actual["UID"] != _uid():
        raise DirectGPUAccessUnavailable("Exact own process identity changed")
    return process


def _gpu(gpu):
    row = subprocess.check_output(["nvidia-smi", "--id=" + str(gpu),
        "--query-gpu=index,uuid,memory.total,memory.free,mig.mode.current", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    fields = [value.strip() for value in row.split(",")]
    if len(fields) != 5 or fields[0] != str(gpu) or fields[4] not in ("N/A", "Disabled", "[N/A]"):
        raise DirectGPUAccessUnavailable("Actual full physical GPU identity/MIG query changed")
    return dict(index=gpu, UUID=fields[1], total_MiB=float(fields[2]), free_MiB=float(fields[3]))


def _apps(uuid):
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    result = []
    for line in rows.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if not line.strip():
            continue
        if len(fields) != 2 or not fields[0].isdigit():
            raise DirectGPUAccessUnavailable("Compute-process ownership query is not exact PID/UUID data")
        if fields[1] == uuid:
            result.append(int(fields[0]))
    return result


class DirectGPUAccessGuard:
    def __init__(self, proof_path, gpu, CPU_affinity, *, stage_action="prepare-current-bank"):
        if stage_action not in STAGE_ACTIONS:
            raise ValueError("Exact matched downstream stage action required")
        self.stage_action = stage_action
        if type(gpu) is not int or gpu not in AFFINITIES or CPU_affinity != AFFINITIES[gpu]:
            raise ValueError("Explicit original physical GPU4/5 and four CPU IDs required")
        self.proof_path = _owned(proof_path)
        self.proof_sha256 = _sha(self.proof_path)
        self.proof = json.loads(self.proof_path.read_text(encoding="utf8"))
        proof = self.proof
        _validate_proof_document(proof, gpu, CPU_affinity)
        self.guard_path = Path(__file__).resolve()
        if not _checksum(proof.get("guard_sha256")) or _sha(self.guard_path) != proof["guard_sha256"]:
            raise ValueError("Direct guard implementation SHA changed")
        request = proof["request"]
        if not isinstance(request, dict) or set(request) != {"path", "sha256"} or not _checksum(request["sha256"]):
            raise ValueError("Immutable full-scale request path/SHA required")
        self.request_path = _owned(request["path"])
        if _sha(self.request_path) != request["sha256"]:
            raise ValueError("Direct full-scale request SHA changed")
        stage = json.loads(self.request_path.read_text(encoding="utf8"))
        declaration = _stage_declaration(stage, self.request_path, gpu, CPU_affinity)
        if declaration["GPU_UUID"] != proof["GPU_UUID"]:
            raise ValueError("Stage declaration and physical GPU proof UUID differ")
        lifetime = proof["host_container_lifetime"]
        if (not isinstance(lifetime, dict) or set(lifetime) != {"role", "expires_at_unix", "PBS_job_id", "allocation_process"}
                or lifetime["role"] != LIFETIME_ROLE or lifetime["PBS_job_id"] != "129528.ECE-util1"
                or type(lifetime["expires_at_unix"]) not in (int, float) or not math.isfinite(lifetime["expires_at_unix"])):
            raise ValueError("Separate host-container expiry witness required; never a GPU4/5 PBS allocation")
        self.gpu, self.CPU_affinity = gpu, list(CPU_affinity)
        self.self_identity = _identity(psutil.Process())
        if self.self_identity["UID"] != 41833:
            raise DirectGPUAccessUnavailable("Direct managed process belongs to a different UID")
        self.managed_processes = {self.self_identity["pid"]: self.self_identity}
        self.selected = False
        self.last_gpu_check = None
        self.monotonic_expiry = time.monotonic() + lifetime["expires_at_unix"] - time.time()
        self.__call__(initial=True)

    def _lifetime(self, initial):
        lifetime = self.proof["host_container_lifetime"]
        process = _exact(lifetime["allocation_process"])
        if process.environ().get("PBS_JOBID") != lifetime["PBS_job_id"]:
            raise DirectGPUAccessUnavailable("Separate host-container PBS identity changed")
        remaining = min(lifetime["expires_at_unix"] - time.time(), self.monotonic_expiry - time.monotonic())
        initial_key = "native_minimum_initial_remaining_seconds" if self.stage_action == "train" else "minimum_initial_remaining_seconds"
        minimum = self.proof[initial_key] if initial else self.proof["expiry_safety_reserve_seconds"]
        if remaining < minimum:
            raise DirectGPUAccessUnavailable("Independent host-container lifetime is insufficient; save at the caller's safe boundary")
        return remaining

    def _managed(self):
        root = _exact(self.self_identity)
        # Only actual descendants of this exact own root become managed. Each
        # newly observed child is pinned to birth/argv/UID before acceptance.
        current = {root.pid: root}
        for child in root.children(recursive=True):
            try:
                identity = _identity(child)
            except psutil.NoSuchProcess:
                continue
            if identity["UID"] != 41833:
                raise DirectGPUAccessUnavailable("Foreign-UID descendant cannot be a managed GPU process")
            previous = self.managed_processes.get(child.pid)
            if previous is not None and previous != identity:
                raise DirectGPUAccessUnavailable("Managed child PID/birth/argv identity changed")
            self.managed_processes[child.pid] = identity
            current[child.pid] = child
        return current

    def __call__(self, *, initial=False):
        if (_sha(self.proof_path) != self.proof_sha256
                or _sha(self.guard_path) != self.proof["guard_sha256"]
                or _sha(self.request_path) != self.proof["request"]["sha256"]):
            raise DirectGPUAccessUnavailable("Immutable direct request/proof/guard changed")
        current = _exact(self.self_identity)
        if current.cpu_affinity() != self.CPU_affinity or _uid() != 41833:
            raise DirectGPUAccessUnavailable("Original assigned CPU/UID contract changed")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        accepted = (str(self.gpu), self.proof["GPU_UUID"])
        if visible in accepted:
            self.selected = True
        elif visible != "" or self.selected:
            raise DirectGPUAccessUnavailable("Direct CUDA visibility must identify exactly the original target GPU")
        else:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                raise DirectGPUAccessUnavailable("Select the exact target GPU before CUDA initialization")
        remaining = self._lifetime(initial)
        now = time.monotonic()
        if initial or self.last_gpu_check is None or now - self.last_gpu_check >= 30:
            row = _gpu(self.gpu)
            if (row["UUID"] != self.proof["GPU_UUID"] or not math.isfinite(row["total_MiB"])
                    or not math.isfinite(row["free_MiB"]) or row["total_MiB"] < 40 * 1024):
                raise DirectGPUAccessUnavailable("Original full physical GPU UUID/VRAM changed")
            applications = _apps(self.proof["GPU_UUID"])
            if initial and (applications or row["free_MiB"] < 40 * 1024):
                raise DirectGPUAccessUnavailable("Direct target GPU must initially be idle with at least40GiB free")
            managed = self._managed()
            for pid in applications:
                if pid not in managed:
                    raise DirectGPUAccessUnavailable("Foreign or unmanaged compute PID on direct target GPU: " + str(pid))
                _exact(self.managed_processes[pid])
            self.last_gpu_check = now
        return dict(format=FORMAT, GPU_arm=self.gpu, GPU_UUID=self.proof["GPU_UUID"],
            PBS_owned_reservation=False, other_PBS_reservations_known_not_owned=True,
            host_container_lifetime_remaining_seconds=remaining,
            stage_action=self.stage_action, managed_process_identity=dict(self.self_identity), request_sha256=self.proof["request"]["sha256"],
            proof_sha256=self.proof_sha256, guard_sha256=self.proof["guard_sha256"],
            exact_CPU_UID_CVD_checked=True, GPU_compute_poll_seconds=30, signals_sent=False)


def make_guard(proof_path, gpu, CPU_affinity, *, stage_action="prepare-current-bank"):
    return DirectGPUAccessGuard(proof_path, gpu, CPU_affinity, stage_action=stage_action)


_NATIVE_STEP_BINDINGS = {}


def install_native_step_guards(trainer_class, guard):
    """Check direct access before original train/validation CUDA methods.

    These wrappers must be installed before final-publication coordination
    captures train_step identity. Original methods, arguments, returned values,
    losses, gradients and optimizer updates are delegated unchanged.
    """
    previous = _NATIVE_STEP_BINDINGS.get(trainer_class)
    if previous is not None:
        if previous["guard"] is not guard:
            raise ValueError("Native trainer direct stage guard already differs")
        previous["check"]()
        return previous["proof"]
    names = ("train_step", "validation_step", "perform_actual_validation")
    original = {name: getattr(trainer_class, name) for name in names}
    codes = {name: method.__code__ for name, method in original.items()}
    wrappers = {}
    def check():
        if (any(getattr(trainer_class, name) is not wrappers[name] for name in names)
                or any(original[name].__code__ is not codes[name] for name in names)):
            raise ValueError("Original guarded native CUDA methods changed")
        return guard()
    def wrap(name):
        method = original[name]
        def guarded(self, *args, **kwargs):
            check()
            return method(self, *args, **kwargs)
        return guarded
    for name in names:
        wrappers[name] = wrap(name)
        setattr(trainer_class, name, wrappers[name])
    proof = dict(format="v24_matched_original_native_direct_pre_step_guard_v1",
                 wrapped_original_methods=list(names), original_method_code_objects_retained=True,
                 math_weights_optimizer_unchanged=True, direct_access=check(), signals_sent=False)
    _NATIVE_STEP_BINDINGS[trainer_class] = dict(guard=guard, check=check, proof=proof)
    return proof
