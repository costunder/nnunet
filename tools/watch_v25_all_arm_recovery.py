"""Read separate recovery monitors using the unchanged v2.5 UI schema2."""
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

FORMAT = 'v25_exact_all_arm_terminal_recovery_v1'
GPUS = {'A':6, 'B':5, 'C':1, 'D':4}


def monitor_bindings(group, seal_path, expected_sha256):
    group = original.input_path(group)
    seal_path = original.input_path(seal_path)
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256) or seal_path.parent != group:
        raise ValueError('Exact owned all-arm recovery seal checksum required')
    body = seal_path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError('All-arm recovery viewer seal checksum changed')
    value = json.loads(body)
    if (value.get('format') != FORMAT or value.get('selected_arms') != list(GPUS)
            or value.get('group_root') != str(group)
            or value.get('scientific_configuration_unchanged') is not True
            or value.get('no_process_signaled') is not True):
        raise ValueError('Exact complete own ABCD recovery execution contract required')
    slots = value.get('slots')
    if not isinstance(slots, dict) or set(slots) != set(GPUS):
        raise ValueError('Every original A/B/C/D recovery monitor is required')
    result = {}
    for arm, gpu in GPUS.items():
        slot = slots[arm]
        if not isinstance(slot, dict) or slot.get('arm') != arm or slot.get('gpu') != gpu:
            raise ValueError('Original arm/GPU assignment changed')
        monitor = original.input_path(slot.get('monitor_path', ''))
        if (monitor.name != 'status.json' or monitor.parent.parent != group/arm/'monitor'
                or not re.fullmatch(r'ALL_recovery_[A-Za-z0-9_]+', monitor.parent.name)):
            raise ValueError('Separate owned all-arm recovery monitor namespace required')
        result[arm] = monitor
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--recovery-seal', type=Path, required=True)
    parser.add_argument('--recovery-seal-sha256', required=True)
    parser.add_argument('--arm', choices=tuple(GPUS), required=True)
    args, remaining = parser.parse_known_args(argv)
    monitors = monitor_bindings(args.group, args.recovery_seal, args.recovery_seal_sha256)
    normal_read = original.read
    routing = {args.group/arm/'monitor/status.json':path for arm,path in monitors.items()}
    routing.update({args.group/arm/'monitor/request.json':path.parent/'request.json'
        for arm,path in monitors.items()})

    def read(path, *, required=False):
        path = Path(path)
        return normal_read(routing.get(path, path), required=required)

    original.read = read
    try:
        return original.main(['--group', str(args.group), '--arm', args.arm, *remaining])
    finally:
        original.read = normal_read


if __name__ == '__main__':main()
