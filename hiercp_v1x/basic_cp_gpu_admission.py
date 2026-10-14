"""Explicit queue protocol for exactly observed live, non-admitted CUDA actors."""
from __future__ import annotations

import copy
import json

GPU_WAIT_EXIT_CODE = 75
GPU_WAIT_FORMAT = 'basic_cp_exact_GPU_admission_wait_v1'
GPU_GUARD_FILE = 'hiercp_v1x/basic_cp_gpu_admission.py'


class GPUAdmissionUnavailable(RuntimeError):
    """A live outside actor blocks admission; other resource errors stay fatal."""

    def __init__(self, observation):
        self.observation = copy.deepcopy(observation)
        if 'pid' in observation and 'occupied_GPU_compute_actors' not in observation:
            self.observation['occupied_GPU_compute_actors'] = [copy.deepcopy(observation)]
        super().__init__('Target Basic GPU is occupied by a non-admitted CUDA actor: '
                         + json.dumps(self.observation, sort_keys=True))
