"""Run a preserved arm only after validating its current PBS allocation.

Use this wrapper with the same arguments as run_comparison_arm.py. The check
runs before the original supervisor starts its child. The child inherits the
validated environment; the frozen original entry point remains unchanged.
Directly calling the old entry point bypasses this wrapper and is not certified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from hiercp_v1x.comparison_allocation import AllocationError, check_current_allocation
from tools import run_comparison_arm


def _manifest(request):
    path = request["experiment"] / "experiment.json"
    if not path.is_file() and request["requires_clone"]:
        path = request["source_experiment"] / "experiment.json"
    with path.open(encoding="utf8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise AllocationError("Preserved experiment manifest is not a mapping")
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    checksum = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    if manifest.get("sha256") != checksum:
        raise AllocationError("Preserved experiment manifest digest changed")
    return manifest


def run(arguments):
    # These imports and resolve_request only inspect metadata. CUDA selection
    # remains owned by the original child, after the allocation is validated.
    request = run_comparison_arm.resolve_request(arguments)
    manifest = _manifest(request)
    receipt = check_current_allocation(manifest, arguments.gpu)
    print("PBS allocation | " + json.dumps(receipt, allow_nan=False), flush=True)
    return run_comparison_arm.run(arguments)


if __name__ == "__main__":
    run(run_comparison_arm.parse())
