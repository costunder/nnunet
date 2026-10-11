"""Read-only current GPU4/5/6 GNN -> full642 CP scoring -> native observer.

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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v24_direct_gnn_training as direct
from tools import watch_v24_matched_cp_scoring as scoring
from tools import training_monitor_adapter as monitor_adapter
from tools import training_monitor_ui as monitor_ui


DEFAULT_NATIVE_PENDING = '실제 nnU-Net 후속 controller가 등록되지 않았습니다. nnU-Net 시작 전입니다.'
NATIVE_REQUEST = 'v24_scored_bank_to_native_chain_request_v1'
NATIVE_STATUS = 'v24_scored_bank_to_native_chain_status_v1'
NATIVE_REGISTRATION = 'v24_scored_bank_to_native_chain_registration_v1'
NATIVE_WAITING = {'WAITING_FOR_FULL_GNN', 'WAITING_FOR_FULL642_SCORING',
    'WAITING_FOR_NATIVE_STORAGE', 'WAITING_FOR_ASSIGNED_GPU', 'WAITING_ALLOCATION',
    'WAITING_METADATA_PUBLICATION'}


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


def scoring_source(root, output, gpu, observed, source_code, *, completed_source=False):
    """A missing CPU pin is a real starting state, never a fabricated score."""
    request = read(root / 'request.json')
    if request is None:
        return None, None
    # The original sealed GPU6 scorer carries its source in pin_path rather
    # than direct source fields. Admit that one explicit completed-source API.
    pin_only_source = completed_source and gpu == 6 and 'source_output' not in request and 'source_code' not in request
    if (request.get('gpu') != gpu or not pin_only_source and
            (request.get('source_output') != str(output) or request.get('source_code') != source_code)):
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


class CompletedGNNSource:
    """GPU6's immutable completed source, not an inferred live GNN process."""
    def __init__(self, renderer, root, scoring_root, gpu):
        if gpu != 6:
            raise ValueError('--gnn-root requires the explicit completed GPU6 source')
        self.renderer, self.output, self.scoring_root = renderer, Path(root), Path(scoring_root)
        self.source_code = None

    def snapshot(self, root):
        scoring.snapshot(self.scoring_root)  # Sealed full642 request/registration, including exact current PID.
        request = read(self.scoring_root / 'request.json', required=True)
        if request.get('gpu') != 6 or not isinstance(request.get('pin_path'), str):
            raise ValueError(str(self.scoring_root / 'request.json') + ': completed GPU6 own pin required')
        pin_path = input_path(request['pin_path'])
        pin = read(pin_path, required=True)
        if (pin.get('format') != 'v24_completed_single_arm_GT_blind_best_v1' or pin.get('physical_GPU') != 6
                or pin.get('source_output') != str(self.output) or pin.get('full_training') is not True
                or pin.get('debug') is not False or pin.get('completed_epochs') != 40
                or pin.get('optimizer_updates') != 680 or not isinstance(pin.get('source_code'), str)):
            raise ValueError(str(pin_path) + ': completed GPU6 own full40/680 source binding differs')
        invocation_path = self.output / 'training/invocations.jsonl'
        rows, _ = self.renderer.JsonlCursor().read(input_path(invocation_path))
        final = rows[-1] if rows else {}
        if (final.get('status') != 'COMPLETE' or final.get('full_training') is not True or final.get('debug') is not False
                or final.get('completed_epochs') != 40 or final.get('optimizer_updates') != 680
                or final.get('actual_CUDA') is not True or final.get('connected_parameter_tensors') != 537
                or final.get('expected_parameter_tensors') != 537
                or not isinstance(pin.get('source_artifacts_sha256'), dict)
                or pin['source_artifacts_sha256'].get('training/invocations.jsonl') != checksum(invocation_path)):
            raise ValueError(str(invocation_path) + ': actual sealed complete full40/680 GPU6 invocation differs')
        self.source_code = pin['source_code']
        return dict(GPU=6, status='TRAINING_COMPLETE', stage='complete', stage_label='전체 GNN 학습 완료',
            epoch=40, epochs=40, completed_epochs=40, optimizer_updates=680, active_U=128,
            progress=None, reports={}, job_root=str(self.output), full_training_completed=True,
            snapshot_fresh=True, status_source='own full40/680 pin and SHA-bound actual COMPLETE invocation')


def _sha_text(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def _file_stat(path):
    info = input_path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def native_storage_binding(root, request, state):
    """A preregistered controller seals paths; the later binding seals bytes."""
    binding_path = input_path(request.get('storage_binding', ''))
    admission_path = input_path(request.get('storage_admission', ''))
    arms = request.get('arm_roots')
    if (binding_path.parent != root.parent or admission_path.parent != root.parent
            or not isinstance(arms, dict) or set(arms) != {'gpu4', 'gpu5', 'gpu6'}
            or arms.get('gpu' + str(request['GPU_arm'])) != str(root)
            or any(not isinstance(path, str) for path in arms.values())
            or len(set(arms.values())) != 3
            or any(input_path(path).parent != root.parent for path in arms.values())):
        raise ValueError(str(root / 'request.json') + ': explicit fresh three-arm storage paths differ')
    binding = read(binding_path, required=True)
    proof, storage = binding.get('admission'), state.get('storage')
    if (binding.get('format') != 'v24_scored_native_storage_ready_v1'
            or binding.get('arm_roots') != arms or binding.get('model_data_scale_preserved') is not True
            or not isinstance(proof, dict) or set(proof) != {'path', 'sha256', 'stat'}
            or proof.get('path') != str(admission_path) or not _sha_text(proof.get('sha256'))
            or proof.get('stat') != _file_stat(admission_path)
            or proof['sha256'] != checksum(admission_path)
            or not isinstance(storage, dict)
            or any(storage.get(key) != proof[key] for key in ('path', 'sha256', 'stat'))
            or proof.get('stat') != _file_stat(admission_path)):
        raise ValueError(str(binding_path) + ': actual storage admission SHA/stat/controller binding differs')
    admission = read(admission_path, required=True)
    if admission.get('format') != 'v24_matched_Basic642_storage_admission_v1' or admission.get('arm_roots') != arms:
        raise ValueError(str(admission_path) + ': actual native three-arm admission differs')
    return str(admission_path), proof['sha256']


def native_completion(root, request, state, output, gpu, source_code):
    """Verify small actual completion receipts without rereading huge weights."""
    admission_path, admission_sha = native_storage_binding(root, request, state)
    proof = state.get('completion_proof')
    stage_path = root / 'matched_storage_train.json'
    if (not isinstance(proof, dict) or proof.get('action') != 'train' or proof.get('physical_GPU') != gpu
            or proof.get('matched_stage_receipt') != str(stage_path)
            or proof.get('matched_stage_receipt_sha256') != checksum(stage_path)):
        raise ValueError(str(root / 'status.json') + ': actual matched native train completion proof required')
    path = input_path(proof.get('path', ''))
    completions = list((root / 'native').glob('training_complete_*.json'))
    if len(completions) != 1 or path != completions[0] or proof.get('sha256') != checksum(path):
        raise ValueError(str(root / 'status.json') + ': one SHA-bound full250 native receipt required')
    stage, complete, native = read(stage_path, required=True), read(path, required=True), read(root / 'native/native.json', required=True)
    if (stage.get('format') != 'v24_matched_basic_native_stage_v1' or stage.get('status') != 'COMPLETE'
            or stage.get('action') != 'train' or stage.get('storage_admission_sha256') != admission_sha
            or stage.get('model_data_scale_preserved') is not True or stage.get('cached_inputs_written') is not False
            or native.get('format') != 'v24_matched_Basic642_native_v1' or native.get('physical_GPU') != gpu
            or native.get('root') != str(root / 'native') or native.get('epochs') != 250
            or native.get('train_cases') != 105 or native.get('validation_cases') != 26 or native.get('debug') is not False
            or native.get('physical_batch') != 2 or native.get('patch_size') != [128] * 3
            or native.get('cp_probability') != .5 or native.get('source_entries') != 642
            or native.get('candidates_per_source') != 128
            or native.get('storage_admission') != admission_path or native.get('storage_admission_sha256') != admission_sha
            or complete.get('format') != native['format'] or complete.get('epochs') != 250 or complete.get('GPU') != gpu
            or complete.get('checkpoint_role') != 'full250_training_completion_witness'
            or complete.get('bank_sha256') != native.get('bank_sha256')
            or complete.get('evaluation_checkpoint_name') != 'checkpoint_best.pth'
            or not _sha_text(complete.get('checkpoint_sha256')) or not _sha_text(complete.get('evaluation_checkpoint_sha256'))):
        raise ValueError(str(path) + ': actual full250 own-arm native metadata differs')
    bank_path = input_path(native.get('bank', ''))
    if checksum(bank_path) != native.get('bank_sha256'):
        raise ValueError(str(bank_path) + ': native scored bank SHA differs')
    overlay = read(bank_path, required=True)
    pin = overlay.get('pin')
    if (not isinstance(pin, dict) or pin.get('source_output') != str(output) or pin.get('source_code') != source_code
            or pin.get('physical_GPU') != gpu):
        raise ValueError(str(bank_path) + ': native own scored BEST source differs')
    receipt_path = input_path(complete.get('best_validation_receipt', ''))
    receipt = read(receipt_path, required=True)
    split = native.get('split')
    coordination = native.get('checkpoint_coordination')
    if (not isinstance(split, dict) or not isinstance(receipt.get('predictions_sha256'), dict)
            or not isinstance(coordination, dict) or not isinstance(coordination.get('folds'), dict)
            or not isinstance(coordination['folds'].get('gpu' + str(gpu)), str)):
        raise ValueError(str(receipt_path) + ': native split/prediction SHA and fold objects required')
    fold = input_path(coordination['folds']['gpu' + str(gpu)])
    if (receipt_path != fold / 'best_validation_complete.json'
            or complete.get('checkpoint') != str(fold / 'checkpoint_final.pth')
            or complete.get('evaluation_checkpoint_path') != str(fold / 'checkpoint_best.pth')
            or receipt.get('native_fold') != str(fold)):
        raise ValueError(str(receipt_path) + ': own native fold/final witness/BEST evaluation paths differ')
    cases = split.get('outer_val')
    if (not isinstance(cases, list) or len(cases) != 26 or any(not isinstance(case, str) or not case for case in cases)
            or len(set(cases)) != 26
            or complete.get('best_validation_receipt_sha256') != checksum(receipt_path)
            or receipt.get('format') != 'v24_native_best_loaded_full26_validation_v1'
            or receipt.get('checkpoint_name') != 'checkpoint_best.pth'
            or receipt.get('checkpoint_path') != complete.get('evaluation_checkpoint_path')
            or receipt.get('checkpoint_sha256') != complete.get('evaluation_checkpoint_sha256')
            or receipt.get('bank_sha256') != native['bank_sha256'] or receipt.get('validation_cases') != sorted(cases)
            or receipt.get('complete') is not True or receipt.get('native_validation_returned_successfully') is not True
            or receipt.get('original_checkpoint_load_called') is not True or receipt.get('original_full_validation_called') is not True
            or receipt.get('final_checkpoint_fallback') is not False or receipt.get('GT_passed_to_predictor') is not False
            or receipt.get('CP_at_inference') is not False or set(receipt.get('predictions_sha256', {})) != set(cases)
            or any(not _sha_text(item) for item in receipt['predictions_sha256'].values())
            or not _sha_text(receipt.get('summary_sha256'))):
        raise ValueError(str(receipt_path) + ': actual complete own BEST/full26 validation receipt differs')
    return dict(full_native_training=True, full_native_evaluation=True, completion_metadata_verified=True,
        completion_evidence_source=str(path), BEST_validation_evidence_source=str(receipt_path),
        completion_verification_scope='sealed actual full250/BEST26 metadata; checkpoint/prediction bytes not reread by viewer')


def native_snapshot(root, output, gpu, renderer, source_code):
    """Observe only a source-bound registered controller and its exact child."""
    request_path = root / 'request.json'
    request, registration, state = (read(root / name) for name in ('request.json', 'registration.json', 'status.json'))
    if request is None or registration is None or state is None:
        return dict(GPU=gpu, status='WAITING_NATIVE_METADATA', stage=None,
            native_pending_reason='지정한 nnU-Net 경로에 실제 request/registration/status 기록이 아직 없습니다.',
            full_native_training=False, full_native_evaluation=False, native_training_started=False)
    digest = checksum(request_path)
    new_controller = request.get('format') == NATIVE_REQUEST
    if (request.get('chain_root') != str(root) or request.get('GPU_arm') != gpu
            or request.get('source_output') != str(output) or request.get('source_code') != source_code
            or state.get('format') != (NATIVE_STATUS if new_controller else 'v24_completed_gnn_to_native_chain_status_v1')
            or new_controller and registration.get('format') != NATIVE_REGISTRATION
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
    if child['alive']:
        value.update(RSS_GiB=child.get('RSS_GiB'), process_status=child.get('process_status'))
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
        if new_controller:
            if state.get('full_native_training_completed') is not True or state.get('full_native_evaluation_completed') is not True:
                raise ValueError(str(root / 'status.json') + ': native full training and BEST evaluation completion required')
            value.update(native_completion(root, request, state, output, gpu, source_code), native_pending_reason=None)
    return value


class PipelineViewer:
    def __init__(self, renderer, arm_root, scoring_root, native_root=None, native_pending_reason=None, *, gnn_root=None, gpu=None):
        self.renderer = renderer
        self.completed_source = gnn_root is not None
        if self.completed_source:
            self.source = CompletedGNNSource(renderer, gnn_root, scoring_root, gpu)
            self.gnn_viewer = self.source
        else:
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
        if not gnn.get('snapshot_fresh'):
            return value
        source_code = self.source.source_code if self.completed_source else read(
            self.source.root / 'direct_training_request.json', required=True).get('source_code')
        if gnn.get('full_training_completed') is not True:
            if self.native_root is not None:
                native = native_snapshot(self.native_root, self.source.output, gpu, self.renderer, source_code)
                if native['status'] not in NATIVE_WAITING | {'WAITING_NATIVE_METADATA', 'FAILED', 'EXITED_WITHOUT_COMPLETION'} or native.get('child_alive'):
                    raise ValueError(str(self.native_root) + ': native stage started before confirmed full GNN completion')
                value['native'] = native
                value['native_pending_reason'] = native.get('native_pending_reason') or '전체 GNN 학습 완료를 기다리는 등록된 nnU-Net controller입니다.'
            return value
        observed = self.scoring_viewer.snapshot(self.scoring_root)
        value.update(active_stage='scoring', status=observed['status'], scoring=observed,
            snapshot_fresh=observed.get('snapshot_fresh', False))
        if not observed.get('snapshot_fresh'):
            return value
        request, pin = scoring_source(self.scoring_root, self.source.output, gpu, observed, source_code,
            completed_source=self.completed_source)
        if request:
            value['native_pending_reason'] = request.get('native_pending_reason') or self.pending_reason
        value['status'] = observed['status']
        if self.native_root is not None:
            native = native_snapshot(self.native_root, self.source.output, gpu, self.renderer, source_code)
            value['native'] = native
            # A path alone cannot claim that a controller or training started.
            if native['status'] != 'WAITING_NATIVE_METADATA':
                if observed['status'] != 'SCORING_COMPLETE_NATIVE_STORAGE_PENDING':
                    if native['status'] not in NATIVE_WAITING or native.get('stage') is not None or native.get('child_alive'):
                        raise ValueError(str(self.native_root) + ': native stage started before confirmed full642 scoring completion')
                else:
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


def display(value, renderer=None, *, experiment_version=None):
    # Source validation remains version-specific; the screen never is.
    return monitor_ui.render(monitor_adapter.from_pipeline(value, experiment_version=experiment_version))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--arm-root', type=Path)
    source.add_argument('--gnn-root', type=Path, help='Explicit completed GPU6 GNN source carried by the own scoring pin')
    parser.add_argument('--gpu', type=int, choices=(6,), help='Required with --gnn-root')
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--renderer-sha', required=True)
    parser.add_argument('--scoring-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path)
    parser.add_argument('--native-pending-reason')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--no-clear', action='store_true')
    parser.add_argument('--experiment-version', help='Scientific version label; does not change the UI schema')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    if args.gnn_root:
        if args.gpu != 6:
            parser.error('--gnn-root requires --gpu 6')
        direct.owned(args.gnn_root)
    else:
        if args.gpu is not None:
            parser.error('--gpu is only used with the completed --gnn-root')
        direct.owned(args.arm_root)
    input_path(args.scoring_root)
    if args.native_root:
        input_path(args.native_root)
    renderer = direct.load_renderer(args.renderer, args.renderer_sha)
    viewer = PipelineViewer(renderer, args.arm_root, args.scoring_root, args.native_root, args.native_pending_reason,
        gnn_root=args.gnn_root, gpu=args.gpu)
    terminal = monitor_ui.TerminalView(sys.stdout) if sys.stdout.isatty() and not (args.no_clear or args.once or args.json) else None
    previous = None
    try:
        while True:
            value = viewer.snapshot()
            if args.json:
                value.update(ui_schema_version=1, experiment_version=args.experiment_version)
                print(json.dumps(value, ensure_ascii=False, allow_nan=False)); return
            model = monitor_adapter.from_pipeline(value, experiment_version=args.experiment_version)
            if terminal:
                terminal.render(model)
            else:
                shown = monitor_ui.render(model)
                if shown != previous:
                    print(shown, flush=True)
                    previous = shown
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
