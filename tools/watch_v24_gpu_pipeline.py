"""Read-only current GPU4/5 GNN -> full642 CP scoring -> native observer.

Every path is supplied explicitly. GNN completion comes from DirectSource's
registered process and real invocation; scoring comes from its sealed request
and complete console events. This viewer never starts, writes or signals work.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import psutil

from tools import watch_v24_direct_gnn_training as direct
from tools import watch_v24_matched_cp_scoring as scoring


DEFAULT_NATIVE_PENDING = '실제 nnU-Net 후속 controller가 등록되지 않았습니다. nnU-Net 시작 전입니다.'


def input_path(path):
    """Reject aliases/foreign inputs while admitting not-yet-published records."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Absolute canonical viewer path required: ' + str(path))
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Symlink viewer input differs: ' + str(item))
    if path.exists() and hasattr(os, 'getuid') and path.stat().st_uid != os.getuid():
        raise ValueError('Viewer input belongs to another UID: ' + str(path))
    return path


def read(path, required=False):
    path = input_path(path)
    value = scoring.read_record(path)
    if value is None and required:
        raise scoring.PendingJSON(path, 'Required controller record not yet published')
    return value


def checksum(path):
    return direct.sha(input_path(path))


def scoring_source(root, output, gpu, observed, source_code):
    """A missing CPU pin is a real starting state, never a fabricated score."""
    request = read(root / 'request.json')
    if request is None:
        return None, None
    if (request.get('source_output') != str(output) or request.get('gpu') != gpu
            or request.get('source_code') != source_code):
        raise ValueError(str(root / 'request.json') + ': scoring source/GPU differs from current GNN')
    pin_name = request.get('pin_path')
    if not isinstance(pin_name, str):
        raise ValueError(str(root / 'request.json') + ': explicit own BEST pin_path required')
    pin_path = input_path(pin_name)
    pin = read(pin_path)
    if pin is None:
        if observed.get('source_done') is not None or observed['status'].startswith('SCORING_COMPLETE'):
            raise ValueError(str(pin_path) + ': scoring progress exists without its own BEST pin')
        if observed.get('alive') and observed['status'] in ('STARTING', 'SCORING_PROCESS_ALIVE'):
            observed.update(status='CPU_PIN_PENDING', detail='현재 GNN의 자기 BEST CPU 검증·pin 기록 대기')
        return request, None
    if (pin.get('format') != 'v24_completed_single_arm_GT_blind_best_v1'
            or pin.get('physical_GPU') != gpu or pin.get('source_output') != str(output)
            or pin.get('source_code') != source_code
            or pin.get('full_training') is not True or pin.get('completed_epochs') != 40
            or pin.get('optimizer_updates') != 680):
        raise ValueError(str(pin_path) + ': actual own full40/680 BEST source/GPU differs')
    observed['own_BEST_pin_verified'] = True
    return request, pin


def native_snapshot(root, output, gpu, renderer, source_code):
    """Observe only a source-bound registered controller and its exact child."""
    request_path = root / 'request.json'
    request, registration, state = (read(root / name) for name in ('request.json', 'registration.json', 'status.json'))
    if request is None or registration is None or state is None:
        return dict(GPU=gpu, status='WAITING_NATIVE_METADATA', stage=None,
            native_pending_reason='지정한 nnU-Net 경로에 실제 request/registration/status 기록이 아직 없습니다.',
            full_native_training=False, full_native_evaluation=False, native_training_started=False)
    digest = checksum(request_path)
    if (request.get('chain_root') != str(root) or request.get('GPU_arm') != gpu
            or request.get('source_output') != str(output) or request.get('source_code') != source_code
            or state.get('format') != 'v24_completed_gnn_to_native_chain_status_v1'
            or state.get('request_sha256') != digest or state.get('GPU_arm') != gpu
            or state.get('source_output') != str(output)
            or registration.get('root') != str(root) or registration.get('request_sha256') != digest
            or registration.get('GPU_arm', gpu) != gpu
            or type(registration.get('pid')) is not int or registration['pid'] <= 0
            or type(registration.get('create_time')) not in (int, float)
            or not math.isfinite(registration['create_time'])
            or not isinstance(registration.get('command'), list)
            or not all(isinstance(item, str) for item in registration['command'])
            or registration['command'][-2:] != ['--request', str(request_path)]
            or state.get('supervisor_pid') != registration['pid']
            or state.get('supervisor_create_time') != registration['create_time']):
        raise ValueError(str(root) + ': native controller request/source/GPU/PID binding differs')
    actor = scoring.process_snapshot(registration)
    stages = request.get('stages')
    if not isinstance(stages, list) or not all(isinstance(row, dict) for row in stages):
        raise ValueError(str(request_path) + ': native declared stages required')
    selected = [row for row in stages if row.get('name') == state.get('stage')]
    if state.get('stage') is not None and len(selected) != 1:
        raise ValueError(str(root / 'status.json') + ': native stage differs from sealed request')
    stage = selected[0] if selected else {}
    child = dict(alive=False, process_status='not_registered')
    if state.get('child_pid') is not None:
        if (state.get('command') != stage.get('command') or type(state.get('child_pid')) is not int
                or state['child_pid'] <= 0 or type(state.get('child_create_time')) not in (int, float)
                or not math.isfinite(state['child_create_time']) or not isinstance(state.get('command'), list)):
            raise ValueError(str(root / 'status.json') + ': native child command/PID/birth differs')
        child = scoring.process_snapshot(dict(pid=state['child_pid'], create_time=state['child_create_time'],
            command=state['command']))
    label = state.get('status')
    if not isinstance(label, str):
        raise ValueError(str(root / 'status.json') + ': native status required')
    if not actor['alive'] and label not in ('COMPLETE', 'FAILED'):
        label = 'CONTROLLER_EXITED_CHILD_ALIVE' if child['alive'] else 'EXITED_WITHOUT_COMPLETION'
    training = stage.get('action') == 'train' and child['alive']
    value = dict(GPU=gpu, native_root=str(root), status=label, stage=state.get('stage'),
        action=stage.get('action'), controller_alive=actor['alive'], child_alive=child['alive'],
        native_training_started=training, full_native_training=False, full_native_evaluation=False,
        native_pending_reason=state.get('waiting_reason') or state.get('error'),
        stage_log=state.get('stage_log'), stages_completed=state.get('stages_completed'),
        snapshot_source='sealed native request/registration/status and current exact process identity')
    if training:
        log_path = input_path(root / (stage['name'] + '.log'))
        if state.get('stage_log') != str(log_path):
            raise ValueError(str(root / 'status.json') + ': native stage log differs')
        metrics = renderer.parse_nnunet_lines(renderer.log_tail(log_path, 65536))
        if metrics:
            value.update(epoch=metrics['epoch_index'] + 1, completed_epochs=metrics['epoch_index'] + int(metrics['epoch_finished']),
                epochs=250, native_metrics=metrics)
    if label == 'COMPLETE':
        value.update(native_controller_completion_reported=True,
            native_training_completion_reported=state.get('full_native_training_completed') is True,
            native_pending_reason='controller 완료 기록만으로 전체250 학습/BEST26 평가 완료를 검증하지 않습니다.',
            completion_evidence_source=str(root / 'status.json'))
    return value


class PipelineViewer:
    def __init__(self, renderer, arm_root, scoring_root, native_root=None, native_pending_reason=None):
        self.renderer = renderer
        self.source = direct.DirectSource(renderer, arm_root)
        self.source.install()
        self.gnn_viewer = renderer.Viewer()
        self.scoring_viewer = scoring.ScoringViewer()
        self.scoring_root = Path(scoring_root)
        self.native_root = Path(native_root) if native_root else None
        self.pending_reason = native_pending_reason or DEFAULT_NATIVE_PENDING
        self.last_valid = None
        self.problem = None

    def _snapshot(self):
        gnn = self.gnn_viewer.snapshot(self.source.output)
        gpu = gnn.get('GPU')
        value = dict(GPU=gpu, status=gnn['status'], active_stage='gnn', gnn=gnn,
            scoring=None, native=None, native_pending_reason=self.pending_reason,
            full_native_training=False, full_native_evaluation=False,
            snapshot_fresh=gnn.get('snapshot_fresh', False))
        if not gnn.get('snapshot_fresh') or gnn.get('full_training_completed') is not True:
            return value
        observed = self.scoring_viewer.snapshot(self.scoring_root)
        value.update(active_stage='scoring', status=observed['status'], scoring=observed,
            snapshot_fresh=observed.get('snapshot_fresh', False))
        if not observed.get('snapshot_fresh'):
            return value
        source_request = read(self.source.root / 'direct_training_request.json', required=True)
        source_code = source_request.get('source_code')
        request, pin = scoring_source(self.scoring_root, self.source.output, gpu, observed, source_code)
        if request:
            value['native_pending_reason'] = request.get('native_pending_reason') or self.pending_reason
        value['status'] = observed['status']
        if self.native_root is not None:
            native = native_snapshot(self.native_root, self.source.output, gpu, self.renderer, source_code)
            value['native'] = native
            # A path alone cannot claim that a controller or training started.
            if native['status'] != 'WAITING_NATIVE_METADATA':
                if observed['status'] != 'SCORING_COMPLETE_NATIVE_STORAGE_PENDING':
                    raise ValueError(str(self.native_root) + ': native controller present before confirmed full642 scoring completion')
                value.update(active_stage='native', status=native['status'])
                value['full_native_training'] = native['full_native_training']
                value['full_native_evaluation'] = native['full_native_evaluation']
            if native.get('native_pending_reason'):
                value['native_pending_reason'] = native['native_pending_reason']
        return value

    def snapshot(self):
        try:
            value = self._snapshot()
        except (scoring.PendingJSON, self.renderer.JsonRecordError) as error:
            signature = (error.path, error.identity, error.detail)
            now = time.monotonic()
            if self.problem is None or self.problem['signature'] != signature:
                self.problem = dict(signature=signature, since=now)
            seconds = now - self.problem['since']
            pending = isinstance(error, scoring.PendingJSON) or error.pending
            return self.failure('READ_PENDING' if pending and seconds < 10 else 'READ_ERROR', str(error), seconds)
        except (ValueError, OSError, psutil.AccessDenied) as error:
            return self.failure('READ_ERROR', type(error).__name__ + ': ' + str(error))
        if value['snapshot_fresh']:
            self.last_valid = copy.deepcopy(value)
            self.problem = None
        return value

    def failure(self, status, detail, seconds=None):
        value = copy.deepcopy(self.last_valid) if self.last_valid else dict(
            GPU=None, active_stage='metadata', gnn=None, scoring=None, native=None,
            full_native_training=False, full_native_evaluation=False, native_pending_reason=self.pending_reason)
        value.update(status=status, snapshot_fresh=False, last_confirmed_status=value.get('status'),
            last_valid_snapshot_preserved=self.last_valid is not None,
            read_problem=dict(detail=detail, unchanged_seconds=seconds, retries_continue=True))
        return value


def display(value, renderer):
    gnn = value.get('gnn')
    if value['active_stage'] == 'gnn' and gnn:
        lines = [renderer.text_snapshot(gnn)]
    else:
        lines = [f"GPU{value.get('GPU') or '?'} | 현재 파이프라인 {value['status']}"]
        if gnn:
            lines.append('GNN: ' + ('전체40epoch/680update 완료 확인' if gnn.get('full_training_completed') else gnn['status']))
        if value.get('scoring'):
            lines.append(scoring.display(value['scoring'], include_native=value['active_stage'] != 'native'))
        native = value.get('native')
        if native:
            lines.append('nnU-Net controller: ' + native['status'] + ' | 단계 ' + str(native.get('stage')))
            if native.get('native_training_started'):
                lines.append('현재 등록된 nnU-Net 학습 프로세스 실행 중')
            elif native.get('full_native_training'):
                lines.append('전체 nnU-Net 학습 완료 controller 기록 확인; 전체 BEST 평가는 별도 증명 필요')
            elif native.get('native_controller_completion_reported'):
                lines.append('nnU-Net controller 완료 기록 있음; 전체 학습/BEST 평가 증명 확인은 별도입니다.')
            else:
                lines.append('nnU-Net 본학습 실행 중으로 확인되지 않았습니다.')
            if native.get('epoch') is not None:
                lines.append(f"nnU-Net epoch {native['epoch']}/{native['epochs']} | 실제 console: {native['native_metrics']}")
    if value.get('last_valid_snapshot_preserved'):
        lines.append('최근 정상 표시 보존 (' + str(value.get('last_confirmed_status')) + '); 수치는 새 기록이 아님')
    if value.get('read_problem'):
        lines.append('표시 기록 읽기 재시도: ' + value['read_problem']['detail'])
    native = value.get('native')
    if not native or not native.get('native_training_started') and not native.get('full_native_training'):
        label = 'nnU-Net 증명 확인: ' if native and native.get('native_controller_completion_reported') else 'nnU-Net 시작 전/대기 이유: '
        lines.append(label + str(value['native_pending_reason']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-root', type=Path, required=True)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--renderer-sha', required=True)
    parser.add_argument('--scoring-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path)
    parser.add_argument('--native-pending-reason')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--no-clear', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    direct.owned(args.arm_root)
    input_path(args.scoring_root)
    if args.native_root:
        input_path(args.native_root)
    renderer = direct.load_renderer(args.renderer, args.renderer_sha)
    viewer = PipelineViewer(renderer, args.arm_root, args.scoring_root, args.native_root, args.native_pending_reason)
    try:
        while True:
            value = viewer.snapshot()
            if args.json:
                print(json.dumps(value, ensure_ascii=False, allow_nan=False)); return
            if sys.stdout.isatty() and not args.no_clear:
                print('\033[2J\033[H', end='')
            print(display(value, renderer), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습/scoring 작업은 계속됩니다.', flush=True)


if __name__ == '__main__':
    main()
