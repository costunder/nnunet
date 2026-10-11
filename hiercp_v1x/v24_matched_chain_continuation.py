"""CPU-only append-only context-profile/owned-allocation chain continuation.

Old requests, source claims, scheduler receipts, code and GNN results remain
immutable. Only an exactly stopped pre-bank controller may be superseded.
No launch, process signal, lease creation or old-file write occurs here.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import v24_matched_cli_recovery as old_cli

FORMAT = "v24_matched_Basic642_depth1_owned_allocation_continuation_proof_v1"
LEASE = "v24_matched_Basic642_depth1_owned_allocation_continuation_lease_v1"
HELPER = "hiercp_v1x/v24_matched_chain_continuation.py"
CONTEXT_HELPER = "hiercp_v1x/v24_context_depth1.py"
ALLOCATION_FORMAT = "v24_owned_PBS_physical_GPU_context_chain_binding_v1"
CHANGED = old_cli.CHANGED_REQUEST_FIELDS - {"matched_cli_handoff"} | {
    "context_depth1", "context_allocation_continuation", "scheduler",
}
GPU6_UUID = "GPU-e2ad7465-3573-504b-3913-4277bd99b2ed"


def lease_path(claim):
    claim = Path(claim)
    return claim.with_name(claim.stem + ".context_allocation_continuation_lease.json")


def _finished(chain, pid, birth, command, role):
    if not old_cli._positive_identity(pid, birth):
        raise ValueError("Exact old " + role + " PID/birth required")
    witness = chain.process_witness(pid, birth, command)
    if witness is not None and witness["status"] != "zombie":
        raise RuntimeError("Previous exact own " + role + " remains live; continuation refused")


def _allocation(request, previous, proof, chain):
    """Require a separately sealed physical PBS assignment if its lease changes."""
    old, new = previous.get("scheduler"), request.get("scheduler")
    binding = proof.get("allocation_binding")
    if new == old:
        if binding is not None:
            raise ValueError("Unchanged allocation must not carry a replacement allocation witness")
        return None
    if (request["GPU_arm"] != 6 or not isinstance(new, dict)
            or new.get("job_id") != "129528.ECE-util1" or new.get("state") != "R"
            or new.get("expected_owner") != "aicompetition06" or new.get("expected_node") != "ece-a6gpu6"
            or new.get("minimum_remaining_seconds", 0) < 28800
            or new.get("native_minimum_remaining_seconds", 0) < 25200
            or type(new.get("expires_at_unix")) not in (int, float)
            or not math.isfinite(new["expires_at_unix"])
            or request.get("GPU_UUID") != GPU6_UUID
            or not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not chain._checksum(binding["sha256"])):
        raise ValueError("Only the explicitly owned replacement GPU6 PBS reservation is supported; no GPU4/5 expansion")
    path = chain.owned(binding["path"])
    if chain.sha(path) != binding["sha256"]:
        raise ValueError("Owned physical PBS allocation witness changed")
    value = chain.read(path)
    expected = {"format", "scheduler", "GPU_arm", "GPU_UUID", "CPU_affinity", "assigned_CPU_ids",
                "ncpus", "ngpus", "RAM_GiB", "owner", "node", "physical_GPU_indices", "PBS_job_id", "allocation_process"}
    assigned = value.get("assigned_CPU_ids")
    if (set(value) != expected or value.get("format") != ALLOCATION_FORMAT
            or value.get("scheduler") != new or value.get("GPU_arm") != 6
            or value.get("GPU_UUID") != request["GPU_UUID"] or value.get("CPU_affinity") != request["CPU_affinity"]
            or not isinstance(assigned, list) or len(assigned) != 6
            or any(type(core) is not int or core < 0 for core in assigned)
            or assigned != sorted(set(assigned)) or not set(request["CPU_affinity"]).issubset(assigned)
            or value.get("ncpus") != 6 or value.get("ngpus") != 1 or value.get("RAM_GiB") != 128
            or value.get("owner") != "aicompetition06" or value.get("node") != "ece-a6gpu6"
            or value.get("physical_GPU_indices") != [6] or value.get("PBS_job_id") != new["job_id"]):
        raise ValueError("Physical GPU6/UUID/six-CPU/128GiB owned PBS binding is incomplete")
    process = value.get("allocation_process")
    if (not isinstance(process, dict) or set(process) != {"pid", "create_time", "command", "UID"}
            or not old_cli._positive_identity(process["pid"], process["create_time"])
            or type(process["UID"]) is not int or process["UID"] != 41833
            or not isinstance(process["command"], list) or not process["command"]
            or any(not isinstance(arg, str) or "\x00" in arg for arg in process["command"])):
        raise ValueError("Exact own live PBS allocation process PID/birth/argv/UID required")
    witness = chain.process_witness(process["pid"], process["create_time"], process["command"])
    if witness is None or witness["status"] == "zombie":
        raise ValueError("The explicitly owned PBS allocation process is no longer live")
    import psutil
    actual = psutil.Process(process["pid"])
    environment = actual.environ()
    if (hasattr(actual, "uids") and actual.uids().real != process["UID"]
            or actual.cpu_affinity() != assigned
            or environment.get("PBS_JOBID") != new["job_id"]
            or environment.get("CUDA_VISIBLE_DEVICES") != request["GPU_UUID"]):
        raise ValueError("Live PBS allocation CPU/UID/job/physical CUDA binding changed")
    return value


def validate_scheduler(request):
    """Structural opt-in check before the legacy scheduler default is examined."""
    if request.get("context_allocation_continuation") is None:
        return False
    from tools import watch_v24_gnn_to_nnunet as chain
    handoff = request["context_allocation_continuation"]
    if not isinstance(handoff, dict) or set(handoff) != {"proof", "proof_sha256", "lease"}:
        raise ValueError("Explicit context/allocation continuation proof and sole lease required")
    if not chain._checksum(handoff["proof_sha256"]) or chain.sha(handoff["proof"]) != handoff["proof_sha256"]:
        raise ValueError("Context/allocation continuation proof SHA changed")
    proof = chain.read(handoff["proof"])
    previous = chain.read(Path(proof["previous_chain_root"]) / "request.json")
    _allocation(request, previous, proof, chain)
    return request.get("scheduler") != previous.get("scheduler")


def validate(request):
    """Return the complete preserved predecessor proof and request checksum."""
    from tools import watch_v24_gnn_to_nnunet as chain
    handoff = request.get("context_allocation_continuation")
    if (request.get("context_depth1") is not True or request.get("GPU_arm") not in (4, 5, 6)
            or request.get("storage_profile") != chain.MATCHED_BASIC_PROFILE or request.get("recovery") is not None
            or request.get("cp_parity_handoff") is None or not isinstance(handoff, dict)
            or set(handoff) != {"proof", "proof_sha256", "lease"}
            or any(name not in request.get("code_files", {}) for name in (HELPER, CONTEXT_HELPER))):
        raise ValueError("Explicit SHA-pinned depth1 own-arm matched continuation required")
    claim = chain.owned(request["source_claim_path"])
    path = chain.owned(handoff["proof"])
    if (not chain._checksum(handoff["proof_sha256"]) or chain.sha(path) != handoff["proof_sha256"]
            or path.parent != claim.parent or not path.name.startswith(claim.stem + ".context_allocation_continuation_proof_")
            or path.suffix != ".json" or handoff["lease"] != str(lease_path(claim))):
        raise ValueError("Context continuation requires its SHA proof and globally derived exclusive lease")
    chain.owned(handoff["lease"], must_exist=False)
    proof = chain.read(path)
    failed = proof.get("mode") == "failed_first_pin"
    fields = old_cli.COMMON_PROOF | {"allocation_binding"} | ({"previous_child_pid", "previous_child_create_time", "previous_child_command"} if failed else set())
    expected_files = old_cli.COMMON_FILES | ({"pin_current_gnn.log"} if failed else set())
    previous_root = chain.owned(proof["previous_chain_root"], directory=True)
    root = Path(request["chain_root"])
    if (set(proof) != fields or proof.get("format") != FORMAT or proof.get("mode") not in ("closed_waiter", "failed_first_pin")
            or proof.get("chain_root") != str(root) or proof.get("source_output") != request["source_output"]
            or root == previous_root or root in previous_root.parents or previous_root in root.parents
            or set(proof.get("previous_files_sha256", {})) != expected_files
            or not chain._checksum(proof.get("source_claim_sha256"))
            or proof.get("old_waiter_closed") is not True or proof.get("old_waiter_children") != []
            or proof.get("old_waiter_cuda_visible_devices") != "" or proof.get("original_GNN_processes_signaled") is not False
            or proof.get("previous_results_written") is not False or proof.get("source_payloads_written") is not False
            or proof.get("evaluation_checkpoint") != "checkpoint_best.pth" or proof.get("full_training_preserved") is not True
            or proof.get("matched_historical_Basic_sources") != 642
            or proof.get("matched_historical_Basic_candidates_per_source") != 128):
        raise ValueError("Exact stopped pre-bank/full642/128/BEST continuation proof required")
    for name, checksum in proof["previous_files_sha256"].items():
        if not chain._checksum(checksum) or chain.sha(previous_root / name) != checksum:
            raise ValueError("Previous matched controller artifact changed: " + name)
    previous = chain.read(previous_root / "request.json")
    if (previous.get("chain_root") != str(previous_root) or previous.get("storage_profile") != chain.MATCHED_BASIC_PROFILE
            or previous.get("context_allocation_continuation") is not None or previous.get("context_depth1") is not None
            or previous.get("recovery") is not None or previous.get("cp_parity_handoff") is None):
        raise ValueError("Original matched predecessor required; context continuation replay refused")
    unchanged_old = {key: value for key, value in previous.items() if key not in CHANGED}
    unchanged_new = {key: value for key, value in request.items() if key not in CHANGED}
    if (unchanged_old != unchanged_new or previous.get("code") == request.get("code")
            or previous.get("commit") == request.get("commit")):
        raise ValueError("Context continuation changed original science/assets/claim/old lease lineage or replayed code")
    _allocation(request, previous, proof, chain)
    chain.validate_request(previous, fresh=False)
    chain.verify_code(previous)
    checksum = proof["previous_files_sha256"]["request.json"]
    if chain.verify_source_claim(previous, checksum) != proof["source_claim_sha256"]:
        raise ValueError("Original matched source claim or earlier lineage changed")
    status, registration, lease = (chain.read(previous_root / name) for name in ("status.json", "registration.json", "launch.lease.json"))
    pid, birth, command = (proof.get("previous_supervisor_" + name) for name in ("pid", "create_time", "command"))
    expected_command = [previous["python"], "-B", "-u", str(Path(previous["code"]) / "tools/watch_v24_gnn_to_nnunet.py"), "--request", str(previous_root / "request.json")]
    if (command != expected_command or registration.get("GPU_arm") != request["GPU_arm"]
            or registration.get("root") != str(previous_root) or registration.get("pid") != pid
            or registration.get("create_time") != birth or registration.get("command") != command
            or registration.get("request_sha256") != checksum or registration.get("CPU_affinity") != request["CPU_affinity"]
            or registration.get("CUDA_visible_devices_while_waiting") != ""
            or status.get("GPU_arm") != request["GPU_arm"] or status.get("request_sha256") != checksum
            or status.get("source_job") != request["source_job"] or status.get("supervisor_pid") != pid
            or status.get("supervisor_create_time") != birth or status.get("actual_CPU_affinity") != request["CPU_affinity"]
            or status.get("source_claim_path") != request["source_claim_path"] or status.get("source_claim_sha256") != proof["source_claim_sha256"]
            or status.get("signals_sent") is not False or status.get("stages_completed") != [] or status.get("stage_proofs") != []
            or lease.get("supervisor_pid") != pid or lease.get("request_sha256") != checksum):
        raise ValueError("Previous controller is not the exact own registered pre-bank supervisor")
    _finished(chain, pid, birth, command, "matched controller")
    if failed:
        child_pid, child_birth, child_command = (proof.get("previous_child_" + name) for name in ("pid", "create_time", "command"))
        log = previous_root / "pin_current_gnn.log"
        if (status.get("status") != "FAILED" or status.get("stage") != "pin_current_gnn" or status.get("error") != old_cli.CONTROLLER_ERROR
                or status.get("command") != previous["stages"][0]["command"] or child_command != previous["stages"][0]["command"]
                or status.get("child_pid") != child_pid or status.get("child_create_time") != child_birth or status.get("stage_log") != str(log)):
            raise ValueError("Only the exact preserved first-pin CLI failure may continue")
        _finished(chain, child_pid, child_birth, child_command, "first-pin child")
        text = chain.owned(log).read_text(encoding="utf8")
        if ("Traceback (most recent call last):" not in text or not text.rstrip().endswith(old_cli.PIN_ERROR)
                or str(Path(previous["code"]) / "hiercp_v1x/v24_nnunet_cp.py") not in text):
            raise ValueError("Exact original first-pin CLI failure traceback required")
        complete = chain.inspect_source(previous)
        if complete is None or status.get("gnn_completion") != complete:
            raise ValueError("Completed own full40/680 GNN must match the old failure witness")
    elif (status.get("status") != "WAITING_FOR_FULL_GNN" or status.get("stage") is not None
            or status.get("child_pid") is not None or status.get("child_create_time") is not None or status.get("gnn_completion") is not None):
        raise ValueError("Old closed waiter must have launched no downstream child")
    for item in previous_root.iterdir():
        if item.name not in expected_files:
            raise ValueError("Old controller contains a pin/score/native artifact; continuation refused")
        chain.owned(item)
    return proof, checksum


def lease_document(request, request_checksum, proof=None):
    if proof is None:
        proof, _ = validate(request)
    handoff = request["context_allocation_continuation"]
    return dict(format=LEASE, GPU_arm=request["GPU_arm"], source_output=request["source_output"],
        source_claim_path=request["source_claim_path"], source_claim_sha256=proof["source_claim_sha256"],
        previous_chain_root=proof["previous_chain_root"], chain_root=request["chain_root"],
        request_sha256=request_checksum, proof=handoff["proof"], proof_sha256=handoff["proof_sha256"])


def verify_source_claim(request, request_checksum):
    from tools import watch_v24_gnn_to_nnunet as chain
    proof, _ = validate(request)
    if chain.read(request["context_allocation_continuation"]["lease"]) != lease_document(request, request_checksum, proof):
        raise ValueError("Exclusive context/allocation lease belongs to another request/root; no duplicate launch")
    return proof["source_claim_sha256"]
