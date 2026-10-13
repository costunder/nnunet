"""Observe the fresh B recovery through the existing CP8/GNN9 and native console.

Only metadata and actual console bytes are read. Original science, releases,
paused GNN counters, CP score rows, loggers and process state are never written.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v25_B_only as legacy
from tools.watch_basic_cp_ablation import NativeConsole

original = legacy.original
RECOVERY_FORMAT = 'v25_B_own_CP529_fresh_host_recovery_v1'


class RecoveryTransition(legacy.Transition):
    def __init__(self, *args, recovery_proof):
        super().__init__(*args)
        self.recovery_proof = recovery_proof

    def early(self):
        old = super().early()
        if old is None:
            raise ValueError('Original own partial BEST route must precede recovery')
        recovery = self.proof(self.recovery_proof)
        if (recovery.get('format') != RECOVERY_FORMAT or recovery.get('gpu') != 5
                or recovery.get('gpu_uuid') != 'GPU-e41d3421-8819-0a49-4a97-6c4518a54f5f'
                or recovery.get('CPU_affinity') != list(range(42, 48))
                or recovery.get('previous_results_preserved') is not True
                or recovery.get('processes_signaled') is not False):
            raise ValueError('Exact own GPU5 unchanged CP529 recovery proof required')
        return self.bind_recovery(old, recovery)

    def bind_recovery(self, old, recovery):
        """Bind actual new execution; retain all already verified original science."""
        durable = recovery.get('durable') or {}
        root = original.input_path(durable.get('root', ''))
        if (root != old['cp_root'] or durable.get('completed_batches') != 133
                or durable.get('completed_sources') != 529
                or durable.get('completed_positions') != 67712
                or (durable.get('request') or {}).get('path') != str(root/'request.json')
                or original.sha(root/'request.json') != durable['request']['sha256']):
            raise ValueError('Actual original immutable 529-source CP namespace differs')
        prior = recovery.get('previous_failed_execution') or {}
        route = original.read(self.route, required=True)
        if (prior.get('controller_seal') != route.get('early_seal')
                or (prior.get('monitor') or {}).get('path') != str(old['monitor'])
                or prior.get('controller') != old['seal'].get('controller_identity')):
            raise ValueError('Actual failed original B route/controller binding differs')
        failed = self.proof(prior.get('monitor'))
        declared = original.declared_actor(failed)
        if (failed.get('status') != 'FAILED' or failed.get('arm') != 'B'
                or failed.get('controller_pid') != prior['controller']['pid']
                or declared != prior.get('child')):
            raise ValueError('Actual original failed B monitor/child differs')
        for actor in (prior['controller'], prior['child']):
            actual, _ = original.observe_actor(actor, commands=[actor['command']], path=old['monitor'])
            if actual is not None:
                raise ValueError('Failed original B actor has not naturally released')
        fresh = self.proof(recovery.get('B_controller_seal'))
        slot = fresh.get('slot') or {}
        monitor = original.input_path(slot.get('monitor_path', ''))
        if (fresh.get('format') != legacy.EARLY_FORMAT or fresh.get('selected_arms') != ['B']
                or fresh.get('group_root') != str(self.group)
                or fresh.get('authorization_request') != old['seal'].get('authorization_request')
                or fresh.get('scientific_request') != old['seal'].get('scientific_request')
                or fresh.get('release_receipt') != old['seal'].get('release_receipt')
                or fresh.get('GNN_early_stop_authorized') is not True
                or fresh.get('scientific_configuration_unchanged') is not True
                or fresh.get('processes_signaled_by_this_adapter') is not False
                or slot.get('arm') != 'B' or slot.get('gpu') != 5
                or slot.get('CPU_affinity') != list(range(42, 48))
                or not monitor.is_relative_to(self.group/'B/monitor')
                or not monitor.parent.name.startswith('B_CP529_RAM_recovery_')
                or monitor.name != 'status.json'):
            raise ValueError('Exact separate B recovery monitor and original science required')
        best = str(self.group/'B/training/checkpoint_best.pt')
        if (fresh.get('preserved_files') or {}).get(best) != (old['seal'].get('preserved_files') or {}).get(best):
            raise ValueError('Original own B BEST witness changed')
        state = original.read(monitor, required=True)
        controller = fresh.get('controller_identity') or {}
        if state.get('controller_pid') != controller.get('pid'):
            raise ValueError('Actual new B monitor is bound to another controller')
        actual, _ = original.observe_actor(controller, commands=[controller.get('command')], path=monitor)
        if actual is None and state.get('status') == 'RUNNING':
            raise ValueError('Fresh B recovery controller exited before completion')
        actor = original.declared_actor(state)
        if actual is not None and actor is not None:
            child, _ = original.observe_actor(actor, commands=[actor.get('command')], path=monitor)
            if child is not None and child['parent_pid'] != controller['pid']:
                raise ValueError('Actual new B stage child belongs to another controller')
        return dict(old, seal=fresh, monitor=monitor, cp_root=root)


def observe(viewer, *, once=False, as_json=False, interval=2):
    """Reuse task renderer and append native console bytes without a new design."""
    terminal = original.ui.TerminalView(sys.stdout, layout='task') if sys.stdout.isatty() and not (once or as_json) else None
    console = NativeConsole(sys.stdout)
    previous = None
    try:
        while True:
            value = viewer.snapshot()
            if as_json:
                value.update(ui_schema_version=2, experiment_version='v2.5')
                print(json.dumps(value, ensure_ascii=False, allow_nan=False)); return
            native = value.get('native') or {}
            if (value.get('snapshot_fresh') and value.get('active_stage') == 'native'
                    and native.get('stage_log')):
                if terminal:
                    terminal.close(); terminal = None
                console.render(native['stage_log'])
            elif terminal:
                terminal.render(original.task_model(value))
            else:
                shown = original.display(value)
                if shown != previous:
                    print(shown, flush=True); previous = shown
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습/scoring 작업은 계속됩니다.', flush=True)
    finally:
        if terminal:
            terminal.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--recovery-seal', type=Path, required=True)
    parser.add_argument('--recovery-seal-sha256', required=True)
    parser.add_argument('--authorization-request', type=Path, required=True)
    parser.add_argument('--authorization-request-sha256', required=True)
    parser.add_argument('--B-route', type=Path, required=True)
    parser.add_argument('--B-recovery-proof', type=Path, required=True)
    parser.add_argument('--B-recovery-proof-sha256', required=True)
    parser.add_argument('--arm', choices=tuple(legacy.previous.GPUS), required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('Positive finite viewer interval required')
    context = RecoveryTransition(args.group, args.recovery_seal, args.recovery_seal_sha256,
        args.authorization_request, args.authorization_request_sha256, args.B_route,
        recovery_proof=dict(path=str(args.B_recovery_proof), sha256=args.B_recovery_proof_sha256))
    normal_read = original.read
    routing = {args.group/arm/'monitor/status.json': path for arm, path in context.monitors.items()}
    routing.update({args.group/arm/'monitor/request.json': path.parent/'request.json'
        for arm, path in context.monitors.items()})
    original.read = lambda path, required=False: normal_read(routing.get(Path(path), Path(path)), required=required)
    try:
        return observe(legacy.Viewer(args.group, args.arm, context), once=args.once,
            as_json=args.json, interval=args.interval)
    finally:
        original.read = normal_read


if __name__ == '__main__':
    main()
