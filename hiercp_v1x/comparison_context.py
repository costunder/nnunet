"""Compatibility evidence for persisted execution measurements, not model state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def gpu_cache_context(net, *, binding_hash, arm, training, comparison_policy=None, curriculum_policy=None):
    """Bind decisions to the verified experiment and selected CUDA device.

    Trained weights are deliberately not a cache key: the original runtime
    already reuses an admitted execution policy across optimizer updates.
    Neither a workload inventory nor this context proves worst-case memory.
    """
    from .comparison_geometry_execution import current_policy
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, 'uuid', None)
    if device_uuid is None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        if ',' in visible or not visible.startswith(('GPU-', 'MIG-')):
            raise RuntimeError('A verified single-device CUDA UUID is required for policy reuse')
        device_uuid = visible
    structure = dict(
        modules=[(name, type(module).__module__, type(module).__qualname__)
                 for name, module in net.named_modules()],
        parameters=[(name, list(value.shape), str(value.dtype), value.requires_grad)
                    for name, value in net.named_parameters()],
        buffers=[(name, list(value.shape), str(value.dtype)) for name, value in net.named_buffers()])
    directory = Path(__file__).resolve().parent
    files = ('comparison_context.py', 'comparison_gpu_runtime.py', 'comparison_gpu_policy.py',
             'comparison_gpu_cache.py', 'comparison_runtime.py', 'comparison_execution.py',
             'comparison_checkpoint.py', 'comparison_empty_context.py',
             'comparison_geometry_execution.py', 'transition_v1_empty_context.py',
             'comparison_curriculum.py', 'comparison_curriculum_data.py', 'comparison_stage_validation.py')
    implementation = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                      for name in files}
    return dict(
        identity_sha256=binding_hash,
        recipient_context_policy=current_policy(),
        curriculum_policy=curriculum_policy,
        cuda_environment=dict(device_uuid=str(device_uuid), device_name=properties.name,
            total_memory_bytes=properties.total_memory,
            compute_capability=[properties.major, properties.minor],
            cuda_runtime=torch.version.cuda, cudnn_version=torch.backends.cudnn.version()),
        torch_version=str(torch.__version__),
        precision=dict(amp=training['amp'], autocast_dtype=str(torch.get_autocast_dtype('cuda')),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        implementation_sha256=_json_hash(implementation),
        model_structure_sha256=_json_hash(structure),
        loss_identity_sha256=_json_hash(dict(arm=arm, training=training,
                                            comparison_policy=comparison_policy)))
