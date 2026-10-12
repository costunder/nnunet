"""Read C's separate recovery monitor through the established UI schema2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v25_pipeline as original


def monitor_binding(group, seal_path, expected_sha256):
    group = original.input_path(group)
    seal_path = original.input_path(seal_path)
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('Exact C recovery seal checksum required')
    if seal_path.parent != group:
        raise ValueError('C recovery seal must belong to this exact group')
    body = seal_path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError('C recovery viewer seal checksum changed')
    value = json.loads(body)
    if (value.get('format') != 'v25_exact_C_standalone_continuation_v1'
            or value.get('selected_arms') != ['C']
            or value.get('group_root') != str(group)
            or value.get('scientific_configuration_unchanged') is not True
            or value.get('no_process_signaled') is not True):
        raise ValueError('Exact own C recovery group and execution contract required')
    slots = value.get('slots')
    if not isinstance(slots, dict) or set(slots) != {'C'}:
        raise ValueError('Only the exact recovered C slot is admitted')
    slot = slots['C']
    if not isinstance(slot, dict) or slot.get('arm') != 'C' or slot.get('gpu') != 1:
        raise ValueError('Exact recovered C/GPU1 slot required')
    monitor = original.input_path(slot.get('monitor_path', ''))
    if (monitor.name != 'status.json' or monitor.parent.parent != group/'C/monitor'
            or not re.fullmatch(r'C_recovery_[A-Za-z0-9_]+', monitor.parent.name)):
        raise ValueError('Separate owned C recovery monitor namespace required')
    return monitor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--recovery-seal', type=Path, required=True)
    parser.add_argument('--recovery-seal-sha256', required=True)
    parser.add_argument('--arm', choices=('C',), default='C')
    args, remaining = parser.parse_known_args(argv)
    monitor = monitor_binding(args.group, args.recovery_seal, args.recovery_seal_sha256)
    normal_read = original.read
    canonical = args.group/'C/monitor/status.json'

    def read(path, *, required=False):
        path = Path(path)
        if path == canonical:
            path = monitor
        elif path == canonical.parent/'request.json':
            path = monitor.parent/'request.json'
        return normal_read(path, required=required)

    # Only this read-only observer process changes its reader binding. No file,
    # controller, GPU policy, status record or display layout is modified.
    original.read = read
    try:
        return original.main(['--group', str(args.group), '--arm', 'C', *remaining])
    finally:
        original.read = normal_read


if __name__ == '__main__':
    main()
