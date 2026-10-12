"""Read-only adapter connecting the fixed common UI to an explicit startup retry."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v24_direct_gnn_training as direct
from tools import watch_v24_gpu_pipeline as pipeline
from tools import watch_v24_matched_cp_scoring as scoring
from tools import training_monitor_ui as ui

REQUEST = 'v24_zero_update_native_startup_retry_request_v1'
STATUS = 'v24_zero_update_native_startup_retry_status_v1'
COMPLETION_REQUEST = 'v24_post_training_existing_best26_completion_request_v1'
COMPLETION_STATUS = 'v24_post_training_existing_best26_completion_status_v1'


def proof(record):
    if (not isinstance(record, dict) or not {'path', 'sha256', 'stat'} <= set(record)
            or not isinstance(record['path'], str) or not pipeline._sha_text(record['sha256'])
            or not isinstance(record['stat'], list) or len(record['stat']) != 5):
        raise ValueError('Explicit startup retry immutable path/SHA/stat witness required')
    path = pipeline.input_path(record['path'])
    if pipeline._file_stat(path) != record['stat'] or direct.sha(path) != record['sha256']:
        raise ValueError('Startup retry immutable witness changed: ' + str(path))
    return path


def retry_snapshot(value, renderer, request_path):
    """Only verified own retry metadata may supersede the preserved old failure."""
    if not value.get('snapshot_fresh'):
        return value
    path = Path(request_path)
    request = pipeline.read(path, required=True)
    old_path = proof(request['original_request'])
    original = pipeline.read(old_path, required=True)
    digest = direct.sha(path)
    if (request.get('format') != REQUEST or request.get('GPU_arm') != 6
            or request.get('recovery_root') != str(path.parent)
            or original.get('chain_root') != str(old_path.parent)
            or original.get('GPU_arm') != 6 or value.get('GPU') != 6
            or value.get('gnn', {}).get('full_training_completed') is not True
            or value.get('gnn', {}).get('job_root') != original.get('source_output')
            or value.get('scoring', {}).get('source_done') != 642
            or value.get('scoring', {}).get('CP_done') != 82176
            or original.get('epochs') != 250 or original.get('physical_batch') != 2
            or original.get('train_cases') != 105 or original.get('validation_cases') != 26
            or original.get('evaluation_checkpoint') != 'checkpoint_best.pth'):
        raise ValueError('Actual own GPU6 full642/native250/B2/BEST26 retry binding differs')
    registration = pipeline.read(path.parent / 'registration.json', required=True)
    lease = pipeline.read(path.parent / 'lease.json', required=True)
    state = pipeline.read(path.parent / 'status.json', required=True)
    command = [request['python'], '-B', '-u', str(Path(request['code']) / 'tools/run_v24_native_startup_retry.py'), 'watch', '--request', str(path)]
    if (state.get('format') != STATUS or state.get('request_sha256') != digest
            or state.get('GPU_arm') != 6 or state.get('source_output') != original['source_output']
            or registration.get('request_sha256') != digest or registration.get('command') != command
            or registration.get('UID') != 41833 or lease.get('request_sha256') != digest
            or lease.get('command') != command or lease.get('UID') != 41833
            or state.get('supervisor_pid') != registration.get('pid')
            or state.get('supervisor_create_time') != registration.get('create_time')
            or lease.get('pid') != registration.get('pid') or lease.get('create_time') != registration.get('create_time')):
        raise ValueError('Exact own startup retry controller/lease identity differs')
    actor = scoring.process_snapshot(registration)
    entry = [request['python'], '-B', '-u', str(Path(request['code']) / 'tools/run_v24_native_startup_retry.py'),
        'entry', '--request', str(path), '--', '730', '3d_fullres', '0', '-tr',
        'nnUNetTrainer_250epochs_FrozenV23CP', '-p', 'nnUNetResEncUNetMPlans', '--val_best']
    child = dict(alive=False, process_status='absent')
    if state.get('child_pid') is not None:
        if state.get('command') != entry or state.get('stage') != 'train':
            raise ValueError('Exact original full native retry child command differs')
        child = scoring.process_snapshot(dict(pid=state['child_pid'], create_time=state['child_create_time'], command=entry))
    label = state['status']
    if not actor['alive'] and label not in ('COMPLETE', 'FAILED'):
        label = 'CONTROLLER_EXITED_CHILD_ALIVE' if child['alive'] else 'EXITED_WITHOUT_COMPLETION'
    training = label == 'RUNNING' and state.get('stage') == 'train' and child['alive']
    reason = (state.get('error') or '새 재시도 controller의 FAILED 원인 미기록') if label == 'FAILED' else state.get('waiting_reason')
    native = dict(GPU=6, status=label, stage=state.get('stage'), controller_alive=actor['alive'], child_alive=child['alive'],
        native_training_started=training, epochs=250, full_native_training=False, full_native_evaluation=False,
        native_pending_reason=reason, startup_retry_root=str(path.parent),
        snapshot_source='sealed zero-update startup retry and exact current process identity')
    if training or label == 'FAILED' and state.get('stage') == 'train':
        log = path.parent / 'train.log'
        if state.get('stage_log') != str(log):
            raise ValueError('Actual startup retry native log binding differs')
        metrics = renderer.parse_nnunet_lines(renderer.log_tail(log, 65536))
        native['native_log_observation_source'] = str(log)
        native['native_log_observation_is_completion_proof'] = False
        if training:
            native.update(child, RAM_limit_GiB=48)
        if training and state.get('child_process_tree_RSS_bytes') is not None:
            native['RSS_GiB'] = state['child_process_tree_RSS_bytes'] / 2**30
        if metrics:
            native.update(epoch=metrics['epoch_index'] + 1,
                completed_epochs=metrics['epoch_index'] + int(metrics['epoch_finished']), native_metrics=metrics)
            if label == 'FAILED':
                native['native_log_observation'] = ('nnU-Net 콘솔 기록: 완료 epoch '
                    + str(native['completed_epochs']) + '/250; 전체 학습/BEST26 완료 증거 미확인')
                reason = native['native_log_observation'] + ' | ' + reason
    if label == 'FAILED':
        reason = '실패 단계 ' + str(state.get('stage') or '미기록') + ' | ' + reason
    # Old native status is preserved on disk, but only this exactly bound retry
    # may supply the current error. Console epochs never certify completion.
    value.pop('read_problem', None)
    native['native_pending_reason'] = reason
    value.update(active_stage='native' if training or label in ('COMPLETE', 'FAILED') else 'scoring',
        status=label, native=native, native_pending_reason=reason,
        full_native_training=False, full_native_evaluation=False)
    if label == 'FAILED':
        value['read_problem'] = dict(path=str(path.parent / 'status.json'),
            detail=state.get('error') or '새 재시도 controller의 FAILED 원인 미기록', stage=state.get('stage'))
    if label == 'COMPLETE':
        if state.get('full_native_training_completed') is not True or state.get('full_native_evaluation_completed') is not True:
            raise ValueError('Full250/BEST26 completion required for startup retry')
        native.update(pipeline.native_completion(old_path.parent, original, state,
            Path(original['source_output']), 6, original['source_code']), native_pending_reason=None)
        value.update(full_native_training=True, full_native_evaluation=True)
    return value


def completion_snapshot(value, retry_path, completion_path):
    """Only an explicit CPU completion declaration may certify saved artifacts."""
    if not value.get('snapshot_fresh'):
        return value
    path = pipeline.input_path(completion_path)
    request = pipeline.read(path, required=True)
    if (request.get('format') != COMPLETION_REQUEST or not isinstance(request.get('failed_retry_declaration'), dict)
            or not isinstance(request.get('failed_retry_request'), dict) or not isinstance(request.get('proofs'), dict)):
        raise ValueError(str(path) + ': explicit sealed existing-artifact completion declaration required')
    retry_path = Path(retry_path)
    retry = pipeline.read(proof(request['failed_retry_request']), required=True)
    original_path = proof(retry['original_request'])
    original = pipeline.read(original_path, required=True)
    expected_proofs = {'failed_retry_request', 'status', 'registration', 'lease', 'global_retry_lease',
        'entry_lease', 'training_started', 'train_log', 'actual_configuration', 'best_receipt',
        'best_checkpoint', 'final_checkpoint'}
    records = request.get('proofs')
    if (request.get('format') != COMPLETION_REQUEST or request.get('GPU_arm') != 6
            or request.get('recovery_root') != str(path.parent) or path.name != 'request.json'
            or path.parent.parent != original_path.parent or path.parent == retry_path.parent
            or request.get('training_rerun') is not False or request.get('evaluation_rerun') is not False
            or request.get('CUDA_compute') is not False or request.get('epochs') != 250
            or request.get('physical_batch') != 2 or request.get('evaluation_checkpoint') != 'checkpoint_best.pth'
            or request.get('python') != retry.get('python') or request.get('failed_retry_declaration') != retry
            or Path(request['failed_retry_request']['path']) != retry_path
            or not isinstance(records, dict) or set(records) != expected_proofs
            or records.get('failed_retry_request') != request['failed_retry_request']
            or retry.get('original_declaration') != original
            or value.get('status') != 'FAILED' or (value.get('native') or {}).get('controller_alive')
            or (value.get('native') or {}).get('child_alive')):
        raise ValueError('Explicit original failed retry/full250/BEST26 CPU completion binding differs')
    # The producer verified checkpoint/prediction bytes. This observer verifies
    # small original records; it does not reload or rehash huge model weights.
    for key in expected_proofs - {'best_checkpoint', 'final_checkpoint'}:
        proof(records[key])
    for key in ('status', 'registration', 'lease', 'entry_lease', 'training_started', 'train_log'):
        name = 'entry.lease.json' if key == 'entry_lease' else key + '.json' if key != 'train_log' else 'train.log'
        if Path(records[key]['path']) != retry_path.parent / name:
            raise ValueError('Exact original retry completion witness path differs: ' + key)
    old_state = pipeline.read(retry_path.parent / 'status.json', required=True)
    if old_state.get('status') != 'FAILED' or old_state.get('stage') != 'train':
        raise ValueError('Preserved post-training failed retry state required')
    digest = direct.sha(path)
    registration = pipeline.read(path.parent / 'registration.json', required=True)
    lease = pipeline.read(path.parent / 'lease.json', required=True)
    state = pipeline.read(path.parent / 'status.json', required=True)
    command = registration.get('command')
    if (state.get('format') != COMPLETION_STATUS or state.get('request_sha256') != digest
            or state.get('GPU_arm') != 6 or state.get('UID') != 41833
            or state.get('source_output') != original['source_output'] or state.get('source_code') != original['source_code']
            or registration.get('request_sha256') != digest or registration.get('UID') != 41833
            or not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command)
            or lease.get('format') != COMPLETION_REQUEST or lease.get('command') != command
            or lease.get('UID') != 41833 or lease.get('request_sha256') != digest
            or lease.get('recovery_root') != str(path.parent)
            or state.get('supervisor_pid') != registration.get('pid')
            or state.get('supervisor_create_time') != registration.get('create_time')
            or lease.get('pid') != registration.get('pid') or lease.get('create_time') != registration.get('create_time')
            or state.get('child_pid') is not None or state.get('child_create_time') is not None
            or any(record.get(key) is not False for record in (state, lease)
                for key in ('training_rerun', 'evaluation_rerun', 'CUDA_compute'))):
        raise ValueError('Exact own existing-artifact completion registration/lease/status differs')
    actor = scoring.process_snapshot(registration)
    label = state.get('status')
    if label not in ('FINALIZING_EXISTING_ARTIFACTS', 'FAILED', 'COMPLETE'):
        raise ValueError('Unknown explicit existing-artifact completion state')
    native = dict(value['native'], status=label, stage=state.get('stage'),
        controller_alive=actor['alive'], child_alive=False, native_training_started=False,
        completion_root=str(path.parent), completion_request_sha256=digest)
    reason = '기존 학습/BEST26 평가 결과의 완료 증거를 CPU에서 검증 중; 재학습·재평가 없음'
    value.pop('read_problem', None)
    if label == 'COMPLETE':
        if (state.get('stage') is not None or state.get('completed_epochs') != 250 or state.get('evaluated_cases') != 26
                or state.get('full_native_training_completed') is not True
                or state.get('full_native_evaluation_completed') is not True):
            raise ValueError('Actual full250/BEST26 existing-artifact completion required')
        native.update(pipeline.native_completion(original_path.parent, original, state,
            Path(original['source_output']), 6, original['source_code']), completed_epochs=250, BEST_evaluated_cases=26)
        native['native_log_observation'] = ('nnU-Net 콘솔 기록: 완료 epoch 250/250; '
            '전체 학습/BEST26 완료는 봉인된 canonical 증거로 별도 검증됨')
        reason = '전체 250 epoch 학습·자기 BEST 26개 평가 완료 증거 검증됨; 재학습·재평가 없음'
    else:
        if state.get('full_native_training_completed') is not False or state.get('full_native_evaluation_completed') is not False:
            raise ValueError('Unfinished existing-artifact completion cannot claim full training/evaluation')
        if label == 'FAILED':
            reason = state.get('error') or '기존 결과 완료 증거 검증 실패; 원인 미기록'
            value['read_problem'] = dict(path=str(path.parent / 'status.json'), detail=reason, stage=state.get('stage'))
        elif not actor['alive']:
            label = 'EXITED_WITHOUT_COMPLETION'
            native['status'] = label
            reason = '기존 결과 완료 증거 검증 controller가 완료 기록 없이 종료됨'
            value['read_problem'] = dict(path=str(path.parent / 'status.json'), detail=reason)
    native['native_pending_reason'] = reason
    value.update(active_stage='native', status=label, native=native, native_pending_reason=reason,
        full_native_training=native['full_native_training'], full_native_evaluation=native['full_native_evaluation'])
    return value


class RetryViewer(pipeline.PipelineViewer):
    def __init__(self, renderer, request_path, completion_request=None):
        self.retry_path = Path(request_path)
        self.completion_path = Path(completion_request) if completion_request is not None else None
        request = pipeline.read(self.retry_path, required=True)
        original_path = proof(request['original_request'])
        original = pipeline.read(original_path, required=True)
        super().__init__(renderer, None, original['scoring_root'], None,
            gnn_root=original['source_output'], gpu=6)

    def _snapshot(self):
        value = retry_snapshot(super()._snapshot(), self.renderer, self.retry_path)
        completion_path = getattr(self, 'completion_path', None)
        return completion_snapshot(value, self.retry_path, completion_path) if completion_path is not None else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--completion-request', type=Path, help='Explicit sealed CPU completion of existing full250/BEST26 artifacts')
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--renderer-sha', required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--no-clear', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    renderer = direct.load_renderer(args.renderer, args.renderer_sha)
    viewer = RetryViewer(renderer, args.request, args.completion_request)
    terminal = ui.TerminalView(sys.stdout, layout='task') if sys.stdout.isatty() and not (args.no_clear or args.once or args.json) else None
    previous = None
    try:
        while True:
            value = viewer.snapshot()
            if args.json:
                print(json.dumps(dict(value, ui_schema_version=2, experiment_version='v2.4'), ensure_ascii=False, allow_nan=False))
                return
            if value.get('active_stage') == 'native' and value.get('native', {}).get('native_training_started'):
                if terminal:
                    terminal.close()
                    terminal = None
                shown = pipeline.display(value, experiment_version='v2.4')
            elif terminal:
                terminal.render(pipeline.task_model(value, experiment_version='v2.4'))
                shown = None
            else:
                shown = pipeline.display(value, experiment_version='v2.4')
            if shown is not None and shown != previous:
                print(shown, flush=True)
                previous = shown
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습 작업은 계속됩니다.', flush=True)
    finally:
        if terminal:
            terminal.close()


if __name__ == '__main__':
    main()
