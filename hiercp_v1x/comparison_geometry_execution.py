"""Scoped, explicitly recorded recipient-absence semantics for comparisons.

The byte-sealed v1 model/configuration stays unchanged. Existing nonempty
canonical caches and completed optimizer updates retain their original meaning.
New absent-recipient graphs carry the adapter's exact policy proof, and new
checkpoints record that policy separately from the preserved training identity.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import functools
import importlib
from pathlib import Path
import threading

_LOCK = threading.RLock()
_CURRENT = None


def current_policy():
    return copy.deepcopy(_CURRENT)


def validate_resume_policy(saved_policy, active_policy):
    """Allow an explicit first adoption; never ignore a later policy change."""
    if saved_policy is not None and saved_policy != active_policy:
        raise ValueError('Checkpoint recipient-context policy differs from the active execution policy')
    if active_policy is None:
        return None
    return dict(policy=copy.deepcopy(active_policy),
                transition='adopt_from_strict_nonempty_checkpoint' if saved_policy is None else 'same_policy_resume',
                model_optimizer_scheduler_rng_and_cursor_preserved=True,
                candidate_order_scope_and_existing_nonempty_geometry_preserved=True)


@contextmanager
def comparison_geometry_execution():
    """Activate after the controller verifies and installs its original scope."""
    global _CURRENT
    from . import bounded_scope
    from . import comparison_empty_context as adapter

    with _LOCK, ExitStack() as stack:
        if _CURRENT is not None:
            raise RuntimeError('Nested recipient-context comparison execution is unsupported')
        original_install = bounded_scope.install

        def activate(scope, snapshot):
            global _CURRENT
            runtime = {name: importlib.import_module('hiercp.' + name)
                       for name in ('spatial', 'local', 'sample', 'model', 'schema')}
            runtime.update(scope=scope, snapshot=Path(snapshot).resolve(strict=True))
            stack.enter_context(adapter.activated(runtime))
            _CURRENT = adapter.identity()

        @functools.wraps(original_install)
        def install(margin_mm, expected_snapshot_root):
            scope = original_install(margin_mm, expected_snapshot_root)
            activate(scope, expected_snapshot_root)
            return scope  # The controller still verifies the original scope digest.

        bounded_scope.install = install
        try:
            if bounded_scope._ACTIVE is not None:
                local = importlib.import_module('hiercp.local')
                activate(copy.deepcopy(bounded_scope._ACTIVE), Path(local.__file__).resolve().parents[1])
            yield
        finally:
            bounded_scope.install = original_install
            # Drain/restore function aliases before advertising inactive state.
            try:
                stack.close()
            finally:
                _CURRENT = None
