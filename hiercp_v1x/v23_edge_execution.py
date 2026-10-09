"""Explicitly bound, L0-only original edge-workspace execution overlay.

This changes only an instance's edge-hidden chunk length. The pinned archived
operator still computes complete relation logits, global softmax and one full
attention dropout. Parameters, buffers, COO topology and L1/L2 are untouched.
Floating accumulation/reduction order can change; CPU/CUDA DEBUG checks do not
replace the required full-native paired numerical and exact-next-RNG gate.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from types import MethodType


FORMAT = 'v23_pinned_original_L0_instance_edge_workspace_v1'
MODEL_SOURCE_SHA256 = '004544962db2ed2d39efcb7a2ef7ee2972ad31b5769d804b5e4c88d866c47049'
ARCHIVE_SHA256 = '5157bafe641e9189824826532a3b055ea560f5c5dfc299374842a3bd1d20a22e'
ORIGINAL_WORKSPACE_BYTES = 64 * 1024**2
EDGE_HIDDEN_WORKSPACE_TENSORS = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity(model):
    """Storage/ownership metadata only; no tensor copies, RNG or state mutation."""
    modules = tuple((name, id(value), type(value)) for name, value in model.named_modules())
    parameters = tuple((name, id(value), tuple(value.shape), value.dtype, value.requires_grad)
        for name, value in model.named_parameters())
    buffers = tuple((name, id(value), tuple(value.shape), value.dtype)
        for name, value in model.named_buffers())
    return modules, parameters, buffers


def _original_method(module):
    cls = module['CompatibilityGatedGATv2Conv']
    function = cls._edge_chunk_size
    _require(inspect.isfunction(function) and function.__globals__ is module,
        'Pinned original class edge-chunk function/global ownership required')
    _require(module.get('EDGE_ATTENTION_EXECUTION_VERSION') == 'hiercp_exact_edge_streaming_v1'
        and type(module.get('EDGE_ATTENTION_WORKSPACE_BYTES')) is int
        and module['EDGE_ATTENTION_WORKSPACE_BYTES'] == ORIGINAL_WORKSPACE_BYTES
        and type(module.get('_EDGE_HIDDEN_WORKSPACE_TENSORS')) is int
        and module['_EDGE_HIDDEN_WORKSPACE_TENSORS'] == EDGE_HIDDEN_WORKSPACE_TENSORS,
        'Original global64MiB/factor8/execution version must remain unchanged')
    return cls, function


def _source_and_convs(model, expected_model_source_sha256, expected_archive_sha256):
    _require(expected_model_source_sha256 == MODEL_SOURCE_SHA256
        and expected_archive_sha256 == ARCHIVE_SHA256,
        'This execution overlay admits only the explicitly pinned original source/archive')
    _require(type(model).__name__ == 'HierarchicalPyGPlacementModel'
        and type(model).__module__ == 'hiercp.model', 'Direct archived full HierCP model required')
    model_class = type(model)
    source = Path(inspect.getsourcefile(model_class)).resolve(strict=True)
    _require(source.is_file() and not source.is_symlink() and _file_sha(source) == MODEL_SOURCE_SHA256,
        'Loaded original model source byte SHA256 differs from sealed archive')
    import sys
    module = vars(sys.modules[model_class.__module__])
    _require(Path(module['__file__']).resolve(strict=True) == source
        and module.get('HierarchicalPyGPlacementModel') is model_class,
        'Loaded original model module/class ownership differs')
    cls, function = _original_method(module)
    _require(Path(inspect.getsourcefile(function)).resolve(strict=True) == source,
        'Original edge function is not from the admitted sealed model source')
    local = getattr(model, 'local_encoder', None)
    _require(type(local) is module.get('LocalTumorContextPyGEncoder'),
        'Original direct local_encoder is required; wrappers are not silently admitted')
    blocks = getattr(local, 'blocks', ())
    relations = tuple(module.get('LOCAL_EDGE_TYPES', ()))
    _require(len(blocks) == 3 and len(relations) == 16 and len(set(relations)) == 16,
        'All original three L0 blocks and sixteen relation types are required')
    names_by_id = {id(value): name for name, value in model.named_modules()}
    chosen = []
    for block_index, block in enumerate(blocks):
        _require(type(block) is module.get('HeteroGATv2Block')
            and tuple(block.edge_types) == relations, 'Exact original L0 block/relation order required')
        items = tuple(block.conv.convs.items())
        _require(tuple(relation for relation, _ in items) == relations,
            'All original L0 relation convolution instances/order are required')
        prefix = f'local_encoder.blocks.{block_index}.conv.convs.'
        for relation, conv in items:
            name = names_by_id.get(id(conv), '')
            _require(type(conv) is cls and name.startswith(prefix)
                and conv.heads == 4 and conv.out_channels == 32,
                'Exact named original L0 convolution/head4/channel32 ownership required')
            _require('_edge_chunk_size' not in vars(conv),
                'An existing instance execution override must be restored explicitly before installation')
            _require(conv._edge_chunk_size.__func__ is function,
                'Original class method changed before L0 overlay installation')
            chosen.append((name, relation, conv))
    _require(len(chosen) == len({id(conv) for _, _, conv in chosen}) == 48,
        'Duplicate/shared/missing L0 convolution ownership is forbidden')
    chosen_ids = {id(conv) for _, _, conv in chosen}
    all_convs = tuple((name, value) for name, value in model.named_modules() if type(value) is cls)
    outside = tuple((name, value) for name, value in all_convs if id(value) not in chosen_ids)
    _require(outside and all(name.startswith(('patient_encoder.', 'prototype_encoder.'))
        for name, _ in outside), 'Only explicit original L1/L2 convolution instances may remain outside L0')
    for name, conv in outside:
        _require('_edge_chunk_size' not in vars(conv) and conv._edge_chunk_size.__func__ is function,
            'Upper-level execution overrides are not admitted: ' + name)
    return source, module, function, tuple(chosen), outside


class L0EdgeWorkspaceHandle:
    """Own a reversible instance overlay; restore never modifies changed methods."""
    def __init__(self, model, workspace_bytes, source, module, original, chosen, outside):
        self._model, self._module, self._original = model, module, original
        self._chosen, self._outside = chosen, outside
        self._identity = _identity(model)
        self._workspace_bytes = workspace_bytes
        self._active = False
        self._installed = {}
        torch = module['torch']
        adapter_sha256 = _file_sha(__file__)

        def chunk_size(conv, dtype):
            _require(conv.heads == 4 and conv.out_channels == 32,
                'L0 dimensions changed after execution overlay installation')
            _original_method(module)
            scalar_bytes = 8 if dtype == torch.float64 else 4
            per_edge = EDGE_HIDDEN_WORKSPACE_TENSORS * conv.heads * conv.out_channels * scalar_bytes
            _require(type(workspace_bytes) is int and workspace_bytes >= per_edge,
                'Explicit L0 workspace must fit one complete edge-hidden calculation')
            return workspace_bytes // per_edge

        # Validate every instance before any mutation. Normal nn.Module attribute
        # assignment of a MethodType adds no module, parameter or buffer.
        for name, _, conv in chosen:
            self._installed[name] = MethodType(chunk_size, conv)
        for name, _, conv in chosen:
            conv._edge_chunk_size = self._installed[name]
        self._active = True
        self._receipt = dict(format=FORMAT, scope='L0_instances_only', workspace_bytes=workspace_bytes,
            original_global_workspace_bytes=ORIGINAL_WORKSPACE_BYTES,
            model_source_path=str(source), model_source_sha256=MODEL_SOURCE_SHA256,
            sealed_archive_sha256=ARCHIVE_SHA256, adapter_sha256=adapter_sha256,
            archive_verification_owner='caller_original_source_activation',
            whole_archive_bytes_verified_by_adapter=False,
            original_edge_hidden_workspace_factor=EDGE_HIDDEN_WORKSPACE_TENSORS,
            local_blocks=3, relation_convolutions=48, heads=4, out_channels=32,
            upper_convolutions=len(outside), global_workspace_changed=False,
            upper_execution_changed=False, module_parameter_buffer_identity_unchanged=True,
            full_relation_softmax_and_dropout_unchanged=True,
            independent_full_native_calibration_required=True, adapter_itself_approves_gate=False,
            convs=[dict(name=name, relation=list(relation),
                original_float32_chunk=original(conv, torch.float32),
                original_float64_chunk=original(conv, torch.float64),
                float32_chunk=conv._edge_chunk_size(torch.float32),
                float64_chunk=conv._edge_chunk_size(torch.float64)) for name, relation, conv in chosen])
        self.guard()

    def guard(self):
        _require(self._active, 'L0 edge-workspace handle is already restored')
        _original_method(self._module)
        _require(_identity(self._model) == self._identity,
            'Module/parameter/buffer identity or dimension changed while overlay is active')
        for name, _, conv in self._chosen:
            _require(vars(conv).get('_edge_chunk_size') is self._installed[name],
                'L0 overlay ownership changed: ' + name)
        for name, conv in self._outside:
            _require('_edge_chunk_size' not in vars(conv)
                and conv._edge_chunk_size.__func__ is self._original,
                'Upper-level execution changed while L0 overlay active: ' + name)
        return True

    def receipt(self):
        self.guard()
        return copy.deepcopy(self._receipt)

    def restore(self):
        if not self._active:
            return False
        # Check all ownership first, so a failed restore cannot partially modify
        # methods which another caller has changed.
        self.guard()
        for _, _, conv in self._chosen:
            del conv._edge_chunk_size
        self._active = False
        _require(_identity(self._model) == self._identity,
            'Unexpected module/parameter/buffer identity change during restore')
        return True

    def __enter__(self):
        self.guard()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.restore()
        return False


def install_l0_edge_workspace(model, workspace_bytes, *,
        expected_model_source_sha256=MODEL_SOURCE_SHA256,
        expected_archive_sha256=ARCHIVE_SHA256):
    """Install one explicit L0-only workspace on the exact activated original.

    Caller must activate/verify the original archive before constructing model.
    This overlay verifies the actual loaded model bytes; it does not extract or
    mutate an archive, choose model/data sizes, or approve a numerical gate.
    """
    _require(type(workspace_bytes) is int and workspace_bytes >= ORIGINAL_WORKSPACE_BYTES,
        'Explicit integer workspace at least original64MiB required; no smaller fallback')
    source, module, original, chosen, outside = _source_and_convs(model,
        expected_model_source_sha256, expected_archive_sha256)
    return L0EdgeWorkspaceHandle(model, workspace_bytes, source, module, original, chosen, outside)
