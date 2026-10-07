"""Install an execution-only loop while preserving sealed comparison sources.

The immutable v1.8 engine remains the authority for model, loss, checkpoint,
identity and RNG helpers. The v1.9 facade can still copy that namespace and
apply its original objective/checkpoint policy bindings. No sealed file,
experiment manifest, configuration or checkpoint identity is rewritten.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import inspect
from pathlib import Path
import threading
from types import FunctionType

FORMAT = "comparison_progress_prefetch_execution_v1"
ENGINE_SHA256 = "9bcb9d67fc2b69861028b5241110270a144ee0ea849c573ab757da7ba391a8cc"
HELPERS = ("_prefetch", "PhaseProgress", "RunningPatientMetrics", "closing", "ComparisonGpuRuntime")
_CONTEXT = threading.RLock()
_MISSING = object()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound_run_arm(original, replacement, namespace):
    """Use frozen globals, so facade-specific policy overrides remain active."""
    if not isinstance(original, FunctionType) or not isinstance(replacement, FunctionType):
        raise TypeError("Frozen and execution training loops must be Python functions")
    if original.__closure__ is not None or replacement.__closure__ is not None:
        raise ValueError("Training execution loops cannot acquire closure dependencies")
    if inspect.signature(original) != inspect.signature(replacement):
        raise ValueError("Execution loop must preserve the complete frozen run_arm signature")
    bound = FunctionType(replacement.__code__, namespace, original.__name__,
                         original.__defaults__, None)
    bound.__kwdefaults__ = copy.deepcopy(original.__kwdefaults__)
    bound.__annotations__ = copy.deepcopy(original.__annotations__)
    bound.__doc__ = replacement.__doc__
    bound.__module__ = original.__module__
    return bound


@contextmanager
def comparison_execution():
    """Temporarily bind progress/prefetch execution for both preserved families.

    The caller must finish selecting its physical GPU before entering. The
    existing controller still verifies sealed files, restores the exact saved
    identity and state, and applies the measured batch/worker configuration.
    All aliases are restored on normal return or failure. Independent arms
    run in separate processes; this process-wide adapter is serialized.
    """
    from . import u_bridge_training as engine
    from . import comparison_runtime as runtime

    with _CONTEXT:
        if _sha(engine.__file__) != ENGINE_SHA256:
            raise ValueError("Frozen v1.8 engine source changed; execution adapter cannot migrate it")
        helpers = runtime.EXECUTION_HELPERS
        if not isinstance(helpers, dict) or set(helpers) != set(HELPERS):
            raise ValueError("Execution helper inventory differs from the reviewed adapter")
        if any(not callable(value) for value in helpers.values()):
            raise TypeError("Execution helpers must be callable")
        namespace = vars(engine)
        previous = {name: namespace.get(name, _MISSING) for name in ("run_arm", *HELPERS)}
        bound = _bound_run_arm(engine.run_arm, runtime.run_arm, namespace)
        evidence = dict(format=FORMAT, frozen_engine_sha256=ENGINE_SHA256,
            execution_runtime_sha256=_sha(runtime.__file__),
            execution_adapter_sha256=_sha(__file__), execution_helpers=list(HELPERS),
            training_loop_implementation_changed=True,
            model_objective_optimizer_schedule_and_checkpoint_protocol_preserved=True,
            sealed_source_files_and_identity_unchanged=True,
            physical_batch_workers_and_candidate_coverage_unchanged=True)
        namespace.update(helpers)
        namespace["run_arm"] = bound
        try:
            yield evidence
        finally:
            for name, value in previous.items():
                if value is _MISSING:
                    namespace.pop(name, None)
                else:
                    namespace[name] = value
