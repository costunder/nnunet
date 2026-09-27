"""Scoped reviewed scorer execution; process CUDA workspace is admitted early."""
from contextlib import contextmanager
import os
import random
import numpy as np
import torch

WORKSPACE=':4096:8'


def admit_cuda_workspace():
    value=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if value!=WORKSPACE:
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA already initialized without reviewed CUBLAS_WORKSPACE_CONFIG=:4096:8; start a new correctly configured process')
        if value is not None:
            raise RuntimeError(f'Unexpected CUBLAS_WORKSPACE_CONFIG={value!r}; use {WORKSPACE} before CUDA initialization')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=WORKSPACE
    return WORKSPACE


def backend_state():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


@contextmanager
def scoring_runtime(base,seed):
    """Caller owns GPU lock. Restore Python/NumPy/Torch RNG and backend flags."""
    from hiercp_v222.runtime import frozen_scoring_runtime
    admit_cuda_workspace()
    python_state=random.getstate();numpy_state=np.random.get_state()
    try:
        with frozen_scoring_runtime(base,seed):
            yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
