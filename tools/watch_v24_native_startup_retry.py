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


def proof(record):
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
    reason = state.get('error') if label == 'FAILED' else state.get('waiting_reason')
    native = dict(GPU=6, status=label, stage=state.get('stage'), controller_alive=actor['alive'], child_alive=child['alive'],
        native_training_started=training, epochs=250, full_native_training=False, full_native_evaluation=False,
        native_pending_reason=reason, startup_retry_root=str(path.parent),
        snapshot_source='sealed zero-update startup retry and exact current process identity')
    if training:
        log = path.parent / 'train.log'
        if state.get('stage_log') != str(log):
            raise ValueError('Actual startup retry native log binding differs')
        metrics = renderer.parse_nnunet_lines(renderer.log_tail(log, 65536))
        native.update(child, RAM_limit_GiB=48)
        if state.get('child_process_tree_RSS_bytes') is not None:
            native['RSS_GiB'] = state['child_process_tree_RSS_bytes'] / 2**30
        if metrics:
            native.update(epoch=metrics['epoch_index'] + 1,
                completed_epochs=metrics['epoch_index'] + int(metrics['epoch_finished']), native_metrics=metrics)
    value.update(active_stage='native' if training or label in ('COMPLETE', 'FAILED') else 'scoring',
        status=label, native=native, native_pending_reason=reason,
        full_native_training=False, full_native_evaluation=False)
    if label == 'COMPLETE':
        if state.get('full_native_training_completed') is not True or state.get('full_native_evaluation_completed') is not True:
            raise ValueError('Full250/BEST26 completion required for startup retry')
        native.update(pipeline.native_completion(old_path.parent, original, state,
            Path(original['source_output']), 6, original['source_code']), native_pending_reason=None)
        value.update(full_native_training=True, full_native_evaluation=True)
    return value


class RetryViewer(pipeline.PipelineViewer):
    def __init__(self, renderer, request_path):
        self.retry_path = Path(request_path)
        request = pipeline.read(self.retry_path, required=True)
        original_path = proof(request['original_request'])
        original = pipeline.read(original_path, required=True)
        super().__init__(renderer, None, original['scoring_root'], original['chain_root'],
            gnn_root=original['source_output'], gpu=6)

    def _snapshot(self):
        return retry_snapshot(super()._snapshot(), self.renderer, self.retry_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
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
    viewer = RetryViewer(renderer, args.request)
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
