"""Read the three independent Basic CP ablations through the existing UI2.

No training, process signaling, source writes or native console changes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from tools import watch_v24_training as shared
from tools import training_monitor_ui as ui


def read(path):
    return shared.read_json(Path(path))


def snapshot(root):
    root = Path(root)
    request, state = read(root / 'request.json'), read(root / 'status.json')
    if not request or not state:
        raise ValueError('Independent Basic request/status not yet published')
    identity = request['controller_identity']
    status = state['status']
    import psutil
    rss, actor_state = None, None
    try:
        actor = psutil.Process(identity['pid'])
        if (actor.create_time() != identity['create_time'] or actor.uids().real != identity['UID']
                or actor.cmdline() != identity['command'] or actor.status() == psutil.STATUS_ZOMBIE):
            if status != 'COMPLETE':
                status = 'EXITED_WITHOUT_COMPLETION' if status != 'FAILED' else status
        else:
            actor_state = actor.status()
            rss = actor.memory_info().rss
            for child in actor.children(recursive=True):
                try:
                    if child.status() != psutil.STATUS_ZOMBIE:
                        rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        if status not in ('COMPLETE', 'FAILED'):
            status = 'EXITED_WITHOUT_COMPLETION'
    phase = state['stage']
    native = read(root / 'native.json')
    log = root / 'train.log' if phase in ('train', 'historical_metrics', 'report', 'complete') else root / (phase + '.log')
    metrics = shared.parse_nnunet_lines(shared.log_tail(log, 65536)) if log.is_file() else {}
    native_phase = bool(native) and phase not in ('prepare', 'preflight', 'candidate_bank', 'waiting_candidate_bank', 'waiting_preflight')
    bank_root = Path(state.get('candidate_bank_root', str(root / 'bank256')))
    generation = read(bank_root / 'generation_progress.json') or {}
    return dict(GPU=str(request['gpu']) + ' / Basic CP ' + request['arm'], status=status,
        physical_GPU=request['gpu'], arm=request['arm'],
        stage='nnunet_training' if native_phase else 'scoring',
        stage_label={'prepare': '전처리 공유 연결', 'preflight': '전체 CP 조건 검증',
            'CUDA_DEBUG': '실제 CUDA B2/B4 검사', 'train': 'nnU-Net 학습',
            'historical_metrics': 'BEST26 병변·크기·통계 평가', 'report': 'Result schema1 저장',
            'complete': '학습·BEST26 평가 완료', 'candidate_bank': 'CP 위치 후보 256개 준비',
            'waiting_preflight': '기존 전체 CP 검사 연결·완료 후 자동 CUDA 검사',
            'waiting_candidate_bank': '기존 CP 위치 후보 생성 연결·완료 후 자동 학습'}.get(phase, phase),
        epochs=250, epoch=metrics.get('epoch_index', -1) + 1 if metrics else None,
        completed_epochs=250 if status == 'COMPLETE' else state.get('completed_epochs', 0),
        native_metrics=metrics, reports={}, active_U=None, progress=None,
        sources_done=generation.get('source_completed'), sources_total=642,
        candidates_done=generation.get('source_completed', 0)*256 if generation else None,
        candidates_total=642*(256 if request['arm'] == 'candidates256' else 128),
        cp_input='Basic uniform / 자기 환자 병변 / GNN 미사용',
        RSS_GiB=rss / 2**30 if rss is not None else None,
        RSS_source='현재 프로세스 트리' if rss is not None else None,
        RAM_limit_GiB=96, process_state=actor_state, error=state.get('error'),
        controller_status=status, controller_stage=phase,
        full_training_completed=status == 'COMPLETE', snapshot_fresh=True,
        full_native_evaluation=status == 'COMPLETE', stage_log=str(log),
        native_started=phase in ('train', 'historical_metrics', 'report', 'complete'))


def preparation_model(value):
    return dict(gpu=value['physical_GPU'], experiment_version='Basic CP ' + value['arm'],
        state=dict(stage='scoring', status=value['status'], fresh=True),
        cp=dict(status=value['stage_label'], source_done=value['sources_done'], source_total=642,
            positions_done=value['candidates_done'], positions_total=value['candidates_total'],
            input_basis=value['cp_input']),
        resources=dict(rss_gib=value['RSS_GiB'], ram_limit_gib=96,
            source=value['RSS_source'], process_status=value['process_state']),
        waiting_reason=None, controller=dict(status=value['controller_status'],
            stage=value['controller_stage']), error=value['error'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('Positive viewer interval required')
    terminal = None
    mode = None
    animated = sys.stdout.isatty() and not (args.once or args.json)
    try:
        while True:
            value = snapshot(args.root)
            if args.json:
                print(json.dumps(value, ensure_ascii=False, allow_nan=False))
            elif animated:
                next_mode = 'native' if value['stage'] == 'nnunet_training' else 'preparation'
                if next_mode != mode:
                    if terminal:
                        terminal.close()
                    terminal = (shared._LegacyNativeTerminalView(sys.stdout) if next_mode == 'native'
                        else ui.TerminalView(sys.stdout, layout='task'))
                    mode = next_mode
                terminal.render(value if mode == 'native' else preparation_model(value))
            else:
                print(shared.text_snapshot(value, experiment_version='Basic CP')
                    if value['stage'] == 'nnunet_training' else ui.render_task(preparation_model(value)), flush=True)
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
