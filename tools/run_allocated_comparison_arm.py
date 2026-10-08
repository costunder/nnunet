"""Run a preserved arm only after validating its current PBS allocation.

Use this wrapper with the same arguments as run_comparison_arm.py. The check
runs before the original supervisor starts its child. The child inherits the
validated environment; the frozen original entry point remains unchanged.
Directly calling the old entry point bypasses this wrapper and is not certified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import functools
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from hiercp_v1x.comparison_allocation import AllocationError, check_current_allocation
from tools import run_comparison_arm

CURRICULUM_POLICY = 'dual_validation_cumulative_v1'
_DISPATCH = threading.RLock()


def parse(argv=None):
    """Consume only the additive policy flag; retain the frozen arm CLI."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--curriculum-policy', choices=(CURRICULUM_POLICY,),
                       help='Explicit dual validation; cumulative U expansion for native/native_listwise')
    extra.add_argument('--preserved-entrypoint', type=Path, help=argparse.SUPPRESS)
    selected, original = extra.parse_known_args(supplied)
    if '--help' in original or '-h' in original:
        print('Additional allocated-launcher option: --curriculum-policy ' + CURRICULUM_POLICY)
    arguments = run_comparison_arm.parse(original)
    arguments.curriculum_policy = selected.curriculum_policy
    if selected.preserved_entrypoint is not None:
        expected = (run_comparison_arm.ROOT / 'tools/run_comparison_arm.py').resolve()
        if (not arguments.owned_child or selected.curriculum_policy is None
                or selected.preserved_entrypoint.resolve() != expected):
            extra.error('Preserved entrypoint must identify the exact opt-in owned child')
    elif arguments.owned_child and selected.curriculum_policy is not None:
        extra.error('An opt-in owned child requires its preserved entrypoint identity')
    return arguments


@contextmanager
def _curriculum_dispatch(arguments, *, child_entrypoint=None, child_options=()):
    """Carry an explicit policy through the unchanged owned-child supervisor.

    There is no environment-variable fallback. The parent launches this same
    allocated wrapper, and its verified child passes the selected policy only
    to its cached runner. Exact experiment/GPU/cache arguments remain intact.
    """
    policy = getattr(arguments, 'curriculum_policy', None)
    if policy not in (None, CURRICULUM_POLICY):
        raise ValueError('Unknown comparison curriculum policy')
    if policy is None:
        yield
        return
    with _DISPATCH:
        if arguments.owned_child:
            from tools import resume_comparison_cached as cached
            original = cached.run

            @functools.wraps(original)
            def selected(supplied):
                copied = copy.copy(supplied)
                existing = getattr(copied, 'curriculum_policy', None)
                if existing not in (None, policy):
                    raise ValueError('Owned child has a conflicting curriculum policy')
                copied.curriculum_policy = policy
                return original(copied)

            cached.run = selected
            try:
                yield
            finally:
                cached.run = original
        else:
            from hiercp_v1x import arm_process
            original = arm_process.run_owned

            @functools.wraps(original)
            def selected(command, **kwargs):
                expected = str(run_comparison_arm.ROOT / 'tools/run_comparison_arm.py')
                if (len(command) < 5 or command[1:4] != ['-B', '-u', expected]
                        or command[4] != '--owned-child' or '--curriculum-policy' in command):
                    raise ValueError('Curriculum dispatch requires the unchanged exact owned-arm command')
                copied = list(command)
                copied[3] = str(Path(child_entrypoint or __file__).resolve())
                copied.extend(('--curriculum-policy', policy))
                # The frozen ownership verifier recognizes the actual preserved
                # entrypoint by its argv path. Keep that identity visible while
                # this additive wrapper carries its explicitly selected policy.
                copied.extend(('--preserved-entrypoint', expected))
                copied.extend(child_options)
                return original(copied, **kwargs)

            arm_process.run_owned = selected
            try:
                yield
            finally:
                arm_process.run_owned = original


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
    with _curriculum_dispatch(arguments):
        return run_comparison_arm.run(arguments)


if __name__ == "__main__":
    run(parse())
