"""Scope frozen GNN runtime settings without changing native nnU-Net RNG/state."""
from contextlib import contextmanager
import os
import torch
from .training import configure_runtime


@contextmanager
def frozen_scoring_runtime(base, seed):
    """Caller holds the existing shared GNN/segmentation GPU lock.

    Model construction consumes random numbers even when loaded weights replace
    its initialization. Preserve both RNG streams and backend policy, including
    restoration if loading or scoring raises an exception.
    """
    previous = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            configure_runtime(base,seed)
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous['matmul_tf32']
            torch.backends.cudnn.allow_tf32=previous['cudnn_tf32']
            torch.backends.cudnn.benchmark=previous['benchmark']
            torch.backends.cudnn.deterministic=previous['cudnn_deterministic']
            torch.use_deterministic_algorithms(previous['deterministic'],warn_only=previous['warn_only'])
            if previous['cublas'] is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
            else:os.environ['CUBLAS_WORKSPACE_CONFIG']=previous['cublas']
