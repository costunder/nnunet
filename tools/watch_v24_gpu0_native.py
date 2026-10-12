"""Read-only GPU4 experiment / GPU0 execution viewer using the existing UI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from tools import watch_v24_training as shared


def snapshot(root):
    request = shared.read_json(root / 'request.json')
    state = shared.read_json(root / 'status.json')
    if not request or not state:
        raise ValueError('GPU0 native request/status not yet published')
    if request.get('source_arm') != 4 or request.get('execution_physical_gpu') != 0:
        raise ValueError('Expected the GPU4 experiment running on physical GPU0')
    stage = state.get('stage')
    phase = {'prepare-native': 'prepare_native', 'calibrate-native': 'calibrate_native',
             'train': 'nnunet_training', 'nnunet_complete': 'nnunet_training'}.get(stage, stage or 'prepare_native')
    log = Path(state['stage_log']) if state.get('stage_log') else root / 'controller_console.log'
    # The production child log contains the unchanged original Epoch/loss lines.
    native_logs = sorted((Path(request['execution_root']) / 'native').glob('train_*.log'), key=lambda p: p.stat().st_mtime)
    if phase == 'nnunet_training' and native_logs:
        log = native_logs[-1]
    metrics = shared.parse_nnunet_lines(shared.log_tail(log, 65536)) if phase == 'nnunet_training' else None
    rss, source = shared.Viewer._rss(state, None)
    status = state.get('status', 'UNKNOWN')
    if status == 'RUNNING':
        import psutil
        registration = shared.read_json(root / 'registration.json')
        if not registration:
            raise ValueError('Exact controller registration is not yet published')
        try:
            actor = psutil.Process(registration['pid'])
            if (abs(actor.create_time() - registration['create_time']) > .01
                    or actor.cmdline() != registration['command']
                    or actor.uids().real != registration['UID']
                    or actor.status() == psutil.STATUS_ZOMBIE):
                status = 'EXITED_WITHOUT_COMPLETION'
        except psutil.NoSuchProcess:
            status = 'EXITED_WITHOUT_COMPLETION'
    return dict(GPU='4 (실행 GPU0)', status=status, stage=phase,
        stage_label=shared.LABELS.get(phase, phase), epochs=250,
        epoch=metrics['epoch_index'] + 1 if metrics else None,
        completed_epochs=250 if status == 'COMPLETE' else metrics['epoch_index'] + int(metrics['epoch_finished']) if metrics else 0,
        native_metrics=metrics or {}, reports={}, active_U=None, progress=None,
        RSS_GiB=rss / 2**30 if rss is not None else None, RSS_source=source,
        RAM_limit_GiB=request.get('native_RAM_GiB'), error=state.get('error'),
        full_training_completed=status == 'COMPLETE', snapshot_fresh=True,
        full_native_evaluation=status == 'COMPLETE', stage_log=str(log))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('Positive polling interval required')
    terminal = shared.TerminalView(sys.stdout, experiment_version='v2.4') if sys.stdout.isatty() and not (args.once or args.json) else None
    try:
        while True:
            value = snapshot(args.root)
            if args.json:
                print(json.dumps(value, ensure_ascii=False, allow_nan=False))
            elif terminal:
                terminal.render(value)
            else:
                print(shared.text_snapshot(value, experiment_version='v2.4'), flush=True)
            if args.once or args.json:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습 작업은 계속됩니다.')
    finally:
        if terminal:
            terminal.close()


if __name__ == '__main__':
    main()
