"""Read-only GPU5 early-stop adapter for the existing common GNN/CP UI."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import psutil

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v24_direct_gnn_training as direct
from tools import watch_v24_gpu_pipeline as pipeline
from tools import watch_v24_matched_cp_scoring as scoring
from tools import training_monitor_ui as ui

REQUEST = 'v24_user_authorized_early_stop_full642_full250_request_v1'
STATUS = 'v24_user_authorized_early_stop_full642_full250_status_v1'
PIN = 'v24_user_authorized_epoch_boundary_early_stop_best_v1'


def proof(record):
    path = pipeline.input_path(record['path'])
    if pipeline._file_stat(path) != record['stat'] or direct.sha(path) != record['sha256']:
        raise ValueError('Viewer immutable witness differs: ' + str(path))
    return path


def process(pid, birth, command):
    if pid is None:
        return dict(alive=False, process_status='absent')
    return scoring.process_snapshot(dict(pid=pid, create_time=birth, command=command))


class Viewer:
    def __init__(self, request_path, guardian_root, arm_root, renderer):
        self.path = Path(request_path)
        self.guardian = Path(guardian_root)
        self.renderer = renderer
        self.source = direct.DirectSource(renderer, arm_root)
        self.source.install()
        self.gnn = renderer.Viewer()
        self.last_valid = None

    def snapshot(self):
        try:
            result = self._snapshot()
        except (OSError, ValueError, KeyError, TypeError, psutil.AccessDenied) as error:
            result = copy.deepcopy(self.last_valid) if self.last_valid else dict(GPU=5, active_stage='gnn', gnn={}, native={}, scoring={},
                full_native_training=False, full_native_evaluation=False)
            result.update(status='READ_PENDING' if isinstance(error, scoring.PendingJSON) else 'READ_ERROR',
                snapshot_fresh=False, last_valid_snapshot_preserved=self.last_valid is not None,
                last_confirmed_status=result.get('status'), read_problem=dict(detail=type(error).__name__ + ': ' + str(error)))
            return result
        if result.get('snapshot_fresh'):
            self.last_valid = copy.deepcopy(result)
        return result

    def _snapshot(self):
        gnn = self.gnn.snapshot(self.source.output)
        value = dict(GPU=5, status=gnn['status'], active_stage='gnn', gnn=gnn, scoring=None, native=None,
            full_native_training=False, full_native_evaluation=False, snapshot_fresh=gnn.get('snapshot_fresh', False))
        if not value['snapshot_fresh']:
            return value
        request = pipeline.read(self.path, required=True)
        resource = request.get('resource_contract', {})
        if (request.get('format') != REQUEST or request.get('GPU_arm') != 5
                or request.get('full_GNN_training') is not False or request.get('debug') is not False
                or request.get('target_epoch') != 27 or request.get('original_planned_epochs') != 40
                or request.get('source_output') != str(self.source.output)
                or request.get('source_entries') != 642 or request.get('CP_positions') != 82176
                or request.get('native_epochs') != 250 or request.get('native_physical_batch') != 2
                or request.get('full_native_evaluation_cases') != 26 or request.get('controller_root') != str(self.path.parent)
                or resource.get('GPU_arm') != 5 or resource.get('source_output') != str(self.source.output)):
            raise ValueError('Actual authorized GPU5 e27/full642/native250/B2/BEST26 viewer binding differs')
        auth = pipeline.read(proof(request['authorization']), required=True)
        if auth.get('user_authorized') is not True or auth.get('target_epoch') != 27 or auth.get('source_output') != str(self.source.output):
            raise ValueError('Actual user-authorized stop metadata differs')
        state = pipeline.read(self.path.parent / 'status.json', required=True)
        lease = pipeline.read(self.path.parent / 'lease.json', required=True)
        digest = direct.sha(self.path)
        if (state.get('format') != STATUS or state.get('GPU_arm') != 5 or state.get('request_sha256') != digest
                or state.get('full_GNN_training') is not False or state.get('authorized_stop_epoch') != 27
                or state.get('source_output') != str(self.source.output) or lease.get('request_sha256') != digest
                or state.get('supervisor_pid') != lease.get('pid') or state.get('supervisor_create_time') != lease.get('create_time')):
            raise ValueError('Actual early-stop controller/lease identity differs')
        expected = [resource['python'], '-B', '-u', str(Path(request['code']) / 'tools/run_v24_early_stop_downstream.py'), 'watch', '--request', str(self.path)]
        if lease.get('command') != expected or lease.get('UID') != 41833:
            raise ValueError('Exact own early controller command/UID differs')
        controller = process(lease['pid'], lease['create_time'], lease['command'])
        label = state['status']
        if not controller['alive'] and label not in ('COMPLETE', 'FAILED'):
            label = 'EXITED_WITHOUT_COMPLETION'
        stage_name = state.get('stage')
        stages = {entry['name']: entry for entry in request['stages']}
        if stage_name is not None and stage_name not in stages:
            raise ValueError('Unknown actual early controller stage')
        child = process(state.get('child_pid'), state.get('child_create_time'), stages[stage_name]['command'] if stage_name else None)
        if child['alive'] and state.get('command') != stages[stage_name]['command']:
            raise ValueError('Exact early stage command differs')
        pending_reason = (state.get('error') or state.get('waiting_reason')) if label == 'FAILED' else (state.get('waiting_reason') or state.get('error'))
        native = dict(status=label, stage=stage_name, controller_alive=controller['alive'], child_alive=child['alive'],
            native_training_started=stage_name == 'train' and child['alive'], full_native_training=False,
            full_native_evaluation=False, native_pending_reason=pending_reason)
        value.update(native=native, native_pending_reason=native['native_pending_reason'])
        guardian = pipeline.read(self.guardian / 'receipt.json')
        if guardian:
            if (guardian.get('status') != 'VERIFIED_PAUSED_AT_EPOCH_BOUNDARY' or guardian.get('verified') is not True
                    or guardian.get('completed_epochs') != 27 or guardian.get('optimizer_updates') != 459
                    or guardian.get('source_output') != str(self.source.output)
                    or guardian.get('authorization') != request['authorization'] or gnn['status'] != 'PAUSED'):
                raise ValueError('Verified actual source pause and guardian receipt differ')
            gnn.update(status='EARLY_STOPPED', epoch=27, completed_epochs=27, full_training_completed=False,
                stage_label='사용자 승인 조기 종료: 27/40epoch', stage='early_stopped')
            value['status'] = 'EARLY_STOPPED'
        pin_path = Path(request['pin_path'])
        pin = pipeline.read(pin_path)
        if pin:
            if (pin.get('format') != PIN or pin.get('physical_GPU') != 5 or pin.get('full_training') is not False
                    or pin.get('completed_epochs') != 27 or pin.get('optimizer_updates') != 459
                    or pin.get('source_output') != str(self.source.output) or pin.get('source_code') != request['source_code']):
                raise ValueError('Actual own early BEST pin/source differs')
            pin_receipt = pipeline.read(self.path.parent / 'stage_pin_early_stopped.json', required=True)
            if pin_receipt.get('output') != str(pin_path) or pin_receipt.get('output_sha256') != direct.sha(pin_path):
                raise ValueError('CPU-admitted early BEST pin completion witness differs')
        if stage_name == 'score' or 'score' in state.get('stages_completed', []):
            console = self.path.parent / 'score.log'
            events, partial = scoring.complete_events(console, 5)
            last = events[-1] if events else None
            done = last['completed_sources'] if last else None
            try:
                console_stat = console.stat()
            except FileNotFoundError:
                console_age = None
            else:
                console_age = max(0, time.time() - console_stat.st_mtime) if console_stat.st_size else None
            cp_status, startup_reason = label, None
            if label == 'RUNNING' and stage_name == 'score' and child['alive']:
                cp_status = 'SCORING_PROCESS_ALIVE'
                if last is None:
                    if (Path(request['scoring_root']) / 'bank/request.json').is_file():
                        cp_status = 'FIRST_BATCH_PENDING'
                        startup_reason = 'CP 입력 request 파일 생성 확인; 첫 batch 완료 기록 대기(완료 수 미확인)'
                    else:
                        cp_status = 'SCORING_STARTUP'
                        startup_reason = '입력·BEST 검증 및 모델 준비 구간; CP batch 완료 기록이 아직 없어 채점 진행량 미확인'
                    native['native_pending_reason'] = startup_reason
                    value['native_pending_reason'] = startup_reason
            cp = dict(GPU=5, status=cp_status,
                source_done=done, source_total=642, CP_done=done * 128 if done is not None else None, CP_total=82176,
                own_BEST_pin_verified=pin is not None, last_batch_seconds=last['seconds'] if last else None,
                seconds_since_console_write=console_age, detail=startup_reason,
                partial_console_line_pending=partial, RAM_limit_GiB=64, **child)
            if 'score' in state.get('stages_completed', []):
                receipt = pipeline.read(self.path.parent / 'stage_score.json', required=True)
                overlay = pipeline.read(proof(dict(path=receipt['output'], sha256=receipt['output_sha256'],
                    stat=pipeline._file_stat(receipt['output']))), required=True)
                if overlay.get('complete_sources') != 642 or overlay.get('complete_positions') != 82176 or overlay.get('pin') != pin:
                    raise ValueError('Actual complete own642/82176 overlay required')
                cp.update(status='SCORING_COMPLETE_NATIVE_PENDING', source_done=642, CP_done=82176)
            value.update(active_stage='scoring', status=cp['status'], scoring=cp)
        if stage_name == 'train' and child['alive']:
            path = self.path.parent / 'train.log'
            if state.get('stage_log') != str(path):
                raise ValueError('Actual native console log binding differs')
            metrics = self.renderer.parse_nnunet_lines(self.renderer.log_tail(path, 65536))
            native.update(child)
            if state.get('child_process_tree_RSS_bytes') is not None:
                native['RSS_GiB'] = state['child_process_tree_RSS_bytes'] / 2**30
            native['RAM_limit_GiB'] = 48
            if metrics:
                native.update(epoch=metrics['epoch_index'] + 1, completed_epochs=metrics['epoch_index'] + int(metrics['epoch_finished']),
                    epochs=250, native_metrics=metrics)
            value.update(active_stage='native', status=label)
        if label == 'COMPLETE':
            if state.get('full_native_training_completed') is not True or state.get('full_native_evaluation_completed') is not True:
                raise ValueError('Actual full250/BEST26 completion flags required')
            complete = pipeline.native_completion(Path(resource['chain_root']), resource, state,
                self.source.output, 5, request['source_code'])
            native.update(complete, native_pending_reason=None)
            value.update(active_stage='native', status='COMPLETE', full_native_training=True, full_native_evaluation=True)
        if label in ('FAILED', 'EXITED_WITHOUT_COMPLETION'):
            value.update(status=label, read_problem=dict(detail=native.get('native_pending_reason') or label))
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'guardian-root', 'arm-root', 'renderer'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--renderer-sha', required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--no-clear', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    renderer = direct.load_renderer(args.renderer, args.renderer_sha)
    viewer = Viewer(args.request, args.guardian_root, args.arm_root, renderer)
    terminal = ui.TerminalView(sys.stdout, layout='task') if sys.stdout.isatty() and not (args.no_clear or args.once or args.json) else None
    previous = None
    try:
        while True:
            value = viewer.snapshot()
            if args.json:
                print(json.dumps(dict(value, ui_schema_version=2, experiment_version='v2.4'), ensure_ascii=False, allow_nan=False)); return
            if value.get('active_stage') == 'native' and value.get('native', {}).get('native_training_started'):
                if terminal:
                    terminal.close(); terminal = None
                shown = pipeline.display(value, experiment_version='v2.4')
            elif terminal:
                terminal.render(pipeline.task_model(value, experiment_version='v2.4'))
                shown = None
            else:
                shown = pipeline.display(value, experiment_version='v2.4')
            if shown is not None and shown != previous:
                print(shown, flush=True); previous = shown
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습/scoring 작업은 계속됩니다.', flush=True)
    finally:
        if terminal:
            terminal.close()


if __name__ == '__main__':
    main()
