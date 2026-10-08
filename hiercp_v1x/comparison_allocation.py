"""Read-only PBS admission checks before importing PyTorch or initializing CUDA.

This checks one arm's declared contract against one running, single-host PBS
allocation. It neither reserves resources nor certifies the combined demand of
multiple arms. There is no production bypass or resource-reduction fallback.
"""
from __future__ import annotations

import getpass
import math
import os
import re
import socket
import subprocess


class AllocationError(RuntimeError):
    """The current scheduler allocation cannot admit the preserved arm."""


_JOB_ID = re.compile(r"[0-9]+(?:\[[0-9]+\])?(?:\.[A-Za-z0-9_.-]+)?\Z")
_FIELDS = frozenset({"Job_Owner", "job_state", "Resource_List.ncpus",
                     "Resource_List.mem", "Resource_List.ngpus", "exec_host"})


def scheduler_environment(environ):
    job_id = environ.get("PBS_JOBID", "")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise AllocationError("A valid PBS_JOBID is required; launch this arm inside its running PBS allocation")
    visible = environ.get("CUDA_VISIBLE_DEVICES", "")
    if not isinstance(visible, str) or not visible.strip():
        raise AllocationError("PBS CUDA_VISIBLE_DEVICES is missing; GPU ownership cannot be verified")
    tokens = tuple(part.strip() for part in visible.split(","))
    if (any(not re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9/-]+", token) for token in tokens)
            or len(set(tokens)) != len(tokens)):
        raise AllocationError("CUDA_VISIBLE_DEVICES must contain distinct scheduler-assigned GPU/MIG UUIDs; numeric ordinals are ambiguous")
    return job_id, tokens


def _positive_integer(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise AllocationError(f"PBS {field} must be an explicit positive integer")
    return int(value)


def _memory_bytes(value):
    if not isinstance(value, str):
        raise AllocationError("PBS Resource_List.mem is missing")
    match = re.fullmatch(r"([1-9][0-9]*)(b|kb|mb|gb|tb)", value.lower())
    if match is None:
        raise AllocationError("PBS Resource_List.mem must be an explicit byte quantity (b/kb/mb/gb/tb)")
    return int(match.group(1)) * 1024 ** {"b": 0, "kb": 1, "mb": 2, "gb": 3, "tb": 4}[match.group(2)]


def parse_qstat(text, *, job_id, user, hostname):
    """Keep only the required public fields; never return Variable_List or raw text."""
    if not isinstance(text, str):
        raise AllocationError("qstat did not return text")
    headers, fields, current = [], {}, None
    for line in text.splitlines():
        header = re.fullmatch(r"Job Id:\s*(\S+)\s*", line.strip())
        if header:
            headers.append(header.group(1))
            current = None
            continue
        assignment = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$", line)
        if assignment:
            key, value = assignment.groups()
            current = key if key in _FIELDS else None
            if current:
                if current in fields:
                    raise AllocationError(f"qstat returned duplicate {current}")
                fields[current] = value.strip()
        elif current and line[:1].isspace():
            fields[current] += line.strip()
        elif line.strip():
            current = None
    if (len(headers) != 1 or not _JOB_ID.fullmatch(headers[0])
            or headers[0].split(".", 1)[0] != job_id.split(".", 1)[0]
            or ("." in job_id and headers[0] != job_id)):
        raise AllocationError("qstat returned a different or malformed PBS job identity")
    if fields.get("job_state") != "R":
        raise AllocationError("The requested PBS job is not running")
    if fields.get("Job_Owner", "").split("@", 1)[0] != user:
        raise AllocationError("The PBS job is not owned by the current account")
    exec_host = fields.get("exec_host", "")
    host_slots = exec_host.split("+")
    if not exec_host or any(not re.fullmatch(r"[A-Za-z0-9_.-]+/[0-9]+(?:\*[1-9][0-9]*)?", item)
                            for item in host_slots):
        raise AllocationError("PBS exec_host is missing or malformed")
    hosts = {item.split("/", 1)[0].split(".", 1)[0] for item in host_slots}
    if hosts != {hostname.split(".", 1)[0]}:
        raise AllocationError("This admission check requires the running PBS job to allocate exactly the current host")
    return dict(job_id=headers[0], owner=user, host=hostname,
                ncpus=_positive_integer(fields.get("Resource_List.ncpus"), "Resource_List.ncpus"),
                memory_bytes=_memory_bytes(fields.get("Resource_List.mem")),
                ngpus=_positive_integer(fields.get("Resource_List.ngpus"), "Resource_List.ngpus"))


def declared_requirements(manifest):
    """Use preserved runtime values, never invented machine-size defaults."""
    if not isinstance(manifest, dict):
        raise AllocationError("The preserved experiment manifest must be a mapping")
    nested = manifest.get("runtime", {})
    if not isinstance(nested, dict):
        raise AllocationError("Manifest runtime must be a mapping when present")
    values = {}
    for key in ("workers", "rss_gib"):
        if key in manifest and key in nested and manifest[key] != nested[key]:
            raise AllocationError(f"Conflicting preserved {key} values in manifest and runtime")
        values[key] = manifest[key] if key in manifest else nested.get(key)
    workers, rss = values["workers"], values["rss_gib"]
    if type(workers) is not int or workers <= 0:
        raise AllocationError("Preserved manifest workers must be a positive integer")
    if isinstance(rss, bool) or not isinstance(rss, (int, float)) or not math.isfinite(rss) or rss <= 0:
        raise AllocationError("Preserved manifest rss_gib must be a positive finite number")
    return dict(workers=workers, rss_budget_bytes=math.ceil(rss * 2**30))


def validate_allocation(manifest, physical_gpu, *, environ, qstat_text,
                        inventory_reader, user, hostname):
    """Pure/injected UNIT boundary; production supplies qstat and current NVML."""
    job_id, tokens = scheduler_environment(environ)
    allocation = parse_qstat(qstat_text, job_id=job_id, user=user, hostname=hostname)
    required = declared_requirements(manifest)
    if required["workers"] > allocation["ncpus"]:
        raise AllocationError(f"Preserved workers={required['workers']} exceeds PBS ncpus={allocation['ncpus']}; obtain a sufficient allocation without reducing the contract")
    if required["rss_budget_bytes"] > allocation["memory_bytes"]:
        raise AllocationError(f"Preserved RSS budget={required['rss_budget_bytes']} bytes exceeds PBS memory={allocation['memory_bytes']} bytes; obtain a sufficient allocation without reducing the contract")
    if len(tokens) > allocation["ngpus"]:
        raise AllocationError("CUDA visibility contains more devices than PBS Resource_List.ngpus")
    if type(physical_gpu) is not int or physical_gpu < 0:
        raise AllocationError("An explicit nonnegative physical GPU index is required")
    observed = inventory_reader(physical_gpu)
    if (not isinstance(observed, dict) or type(observed.get("physical_index")) is not int
            or observed.get("physical_index") != physical_gpu
            or not isinstance(observed.get("physical_uuid"), str)
            or not observed["physical_uuid"].startswith("GPU-")
            or type(observed.get("mig_enabled")) is not bool):
        raise AllocationError("NVML did not verify the requested physical GPU identity")
    if observed.get("mig_enabled"):
        children = observed.get("mig")
        if not isinstance(children, list) or any(not isinstance(row, dict) or not isinstance(row.get("uuid"), str) for row in children):
            raise AllocationError("NVML MIG inventory is malformed")
        selected = [row["uuid"] for row in children if row["uuid"] in tokens]
        if len(selected) != 1:
            raise AllocationError("The requested physical GPU must have exactly one scheduler-visible MIG instance")
        selected_uuid, kind = selected[0], "MIG"
    else:
        selected_uuid, kind = observed["physical_uuid"], "GPU"
        if selected_uuid not in tokens:
            raise AllocationError(f"Physical GPU {physical_gpu} is outside scheduler CUDA_VISIBLE_DEVICES")
    return dict(format="comparison_pbs_allocation_admission_v1", allocation=allocation,
                preserved_requirements=required, requested_physical_gpu=physical_gpu,
                selected_uuid=selected_uuid, selected_type=kind,
                scheduler_visible_uuids=list(tokens), environment_changed=False,
                scope="one arm only; no shared-job aggregate reservation or admission claim",
                training_or_cuda_started=False)


def check_current_allocation(manifest, physical_gpu, *, environ=None,
                             qstat_runner=None, inventory_reader=None):
    """Read current scheduler state and NVML; stdout contains no raw environment."""
    environ = os.environ if environ is None else environ
    job_id, _ = scheduler_environment(environ)
    if qstat_runner is None:
        qstat_runner = subprocess.run
    try:
        result = qstat_runner(["qstat", "-f", job_id], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, check=False)
    except OSError as error:
        raise AllocationError("Cannot execute read-only qstat; scheduler allocation is unverified") from error
    if result.returncode != 0:
        raise AllocationError(f"qstat allocation query failed with status {result.returncode}; no training was started")
    if inventory_reader is None:
        from tools.current_gpu import inventory
        inventory_reader = inventory
    return validate_allocation(manifest, physical_gpu, environ=environ,
                               qstat_text=result.stdout, inventory_reader=inventory_reader,
                               user=getpass.getuser(), hostname=socket.gethostname())
