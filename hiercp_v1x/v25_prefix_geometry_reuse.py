"""V25 execution-only binding of the existing exact full128 geometry reuse.

The original full patient/prototype tensors are admitted through their existing
query, CT/organ, donor, source and tensor proofs. Every candidate-dependent
relation is recomputed by the original hierarchy functions on the requested
complete P+active-U set. No neural output, raw CT, EDT, or cold builder is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .v24_upper_reuse import UpperReuseAdapter, runtime_contract as original_runtime_contract


FORMAT = 'v25_exact_full128_prefix_geometry_reuse_execution_v1'
_SOURCE = Path(__file__).resolve()
_FROZEN = ('v25_factory.py', 'v25_preparation.py', 'v25_context.py',
    'v25_targets.py', 'v25_training.py', 'v24_geometry.py', 'v24_inputs.py',
    'v24_model.py', 'v24_provider.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_contract():
    """Pure provenance; this receipt never activates CUDA or original builders."""
    paths = [_SOURCE.parent/name for name in _FROZEN]
    paths.append(_SOURCE.parents[1]/'tools/run_v25.py')
    contract = original_runtime_contract()
    contract.update(format=FORMAT, prefix_runtime_source_sha256=_sha(_SOURCE),
        V25_frozen_source_sha256={str(path.relative_to(_SOURCE.parents[1])):_sha(path)
            for path in paths},
        original_exact_reuse_module='hiercp_v1x.v24_upper_reuse.UpperReuseAdapter',
        original_exact_reuse_source_sha256=_sha(_SOURCE.with_name('v24_upper_reuse.py')),
        instance_only_builder_override=True, existing_cache_hits_unchanged=True,
        CUDA_context_initialized_before_binding_allowed=False,
        CPU_geometry_derivation_after_CUDA_context_initialization_allowed=True,
        current_processes_or_global_cache_classes_modified=False,
        original_optimizer_model_graph_and_candidate_contract_unchanged=True)
    return contract


class PrefixGeometryReuseAdapter(UpperReuseAdapter):
    """Reuse the verified original transform; add V25 execution provenance."""
    def __init__(self, inputs):
        super().__init__(inputs)
        paths = [_SOURCE] + [_SOURCE.parent/name for name in _FROZEN]
        paths.append(_SOURCE.parents[1]/'tools/run_v25.py')
        for path in paths:
            path = path.resolve(strict=True)
            proof = self.cache._file_identity(path)
            self._source_guard[str(path)] = proof
            inputs._input_file_proofs[str(path)] = proof
            inputs._file_proofs[str(path)] = proof
        self.contract.update(runtime_contract())
        self._guard()


def install(inputs):
    """Install only on a newly created CPU preparation instance, before CUDA."""
    previous = getattr(inputs, '_v25_prefix_geometry_reuse_adapter', None)
    if previous is not None:
        if not isinstance(previous, PrefixGeometryReuseAdapter) or inputs.geometry.builder != previous.derive:
            raise ValueError('Foreign V25 prefix geometry reuse binding refused')
        previous._guard()
        return previous
    if getattr(inputs, '_v24_upper_reuse_adapter', None) is not None:
        raise ValueError('Existing foreign upper reuse binding must be preserved')
    adapter = PrefixGeometryReuseAdapter(inputs)
    inputs.geometry.builder = adapter.derive
    inputs._v25_prefix_geometry_reuse_adapter = adapter
    return adapter
