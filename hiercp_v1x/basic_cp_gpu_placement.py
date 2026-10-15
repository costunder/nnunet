"""Explicitly approved physical placement changes; scientific arms stay fixed.

The original seven-arm mapping remains the default.  A fresh cohort may move
the waiting Multi7 continuation from physical GPU1 to physical GPU6 only with
one identical SHA-pinned receipt in all seven immutable requests.  No live
registry, process, checkpoint, scientific setting, or CUDA visibility is edited.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

FILE = 'hiercp_v1x/basic_cp_gpu_placement.py'
FIELD = 'GPU_placement_receipt'
FORMAT = 'basic_cp_approved_physical_GPU_placement_v1'
OPERATION = 'resume_multi7_GPU1_on_idle_GPU6'
ORIGINAL = {0: 'multi5', 1: 'multi7', 2: 'multi10', 3: 'candidates512',
            4: 'candidates1024', 5: 'candidates2048', 6: 'blend50'}
APPROVED = dict(ORIGINAL)
APPROVED.update({1: 'blend50', 6: 'multi7'})
QUEUE_OPERATION = 'queue_multi5_after_candidates2048_and_candidates1024_after_multi7'


def queue_placement(*, multi5_released=False, candidates1024_released=False):
    """Only the two independently approved successor swaps are permitted."""
    if type(multi5_released) is not bool or type(candidates1024_released) is not bool:
        raise ValueError('Actual independent queue-release flags must be booleans')
    result = dict(APPROVED)
    if multi5_released:
        result[0], result[5] = result[5], result[0]
    if candidates1024_released:
        result[4], result[6] = result[6], result[4]
    return result


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def admit_placement(request, admit):
    """Admit the exact shared receipt through the lease's immutable file guard."""
    if FIELD not in request:
        return dict(ORIGINAL)
    reference = request[FIELD]
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference['path'], str)
            or not Path(reference['path']).is_absolute()
            or '..' in Path(reference['path']).parts
            or Path(reference['path']).parent != Path(request['root']).parent
            or not _checksum(reference['sha256'])
            or not _checksum(request.get('code_sha256', {}).get(FILE))):
        raise ValueError('Physical GPU placement requires its exact shared receipt and code SHA')
    admit(reference['path'], reference['sha256'])
    value = json.loads(Path(reference['path']).read_bytes())
    admitted = [dict(operation=OPERATION, placement=dict(APPROVED))]
    admitted.extend(dict(operation=QUEUE_OPERATION, placement=queue_placement(
        multi5_released=first, candidates1024_released=second))
        for first in (False, True) for second in (False, True))
    for candidate in admitted:
        expected = dict(format=FORMAT, operation=candidate['operation'],
            physical_GPU_arms={str(gpu): arm for gpu, arm in candidate['placement'].items()})
        if value == expected:
            return dict(candidate['placement'])
    raise ValueError('Only the approved whole seven-arm placement permutations are allowed')


def validate_peer_placement(request, peer, placement):
    """A migrated cohort cannot mix old placement or substitute another receipt."""
    if peer.get(FIELD) != request.get(FIELD):
        raise ValueError('All seven peers require one identical immutable physical GPU placement receipt')
    if FIELD in request and peer.get('code_sha256', {}).get(FILE) != request['code_sha256'][FILE]:
        raise ValueError('All seven placement peers require the same exact mapping guard SHA')
    gpu = peer.get('gpu')
    if type(gpu) is not int or gpu not in placement or peer.get('arm') != placement[gpu]:
        raise ValueError('Registered physical GPU and scientific arm differ from the admitted placement')

