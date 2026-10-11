"""Read-only direct GPU4/5 GNN viewer using a SHA-pinned shared renderer.

Only the renderer's missing gnn/status.json is projected in memory from the
actual direct registration. No pipeline status, checkpoint, or file is written.
The prepared cursor and all evolving training JSON/JSONL remain real inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import stat

import psutil


def owned(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Absolute canonical owned path required: ' + str(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Symlink is not an admitted viewer input: ' + str(parent))
    info = path.stat()
    if hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise ValueError('Viewer input belongs to another UID: ' + str(path))
    return info


def sha(path):
    info = owned(path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Regular viewer input required: ' + str(path))
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    after = owned(path)
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(info, key) != getattr(after, key) for key in fields):
        raise ValueError('Viewer input changed while hashing: ' + str(path))
    return value


def load_renderer(path, checksum):
    path = Path(path)
    if (path.name != 'watch_v24_training.py' or len(checksum) != 64
            or set(checksum) - set('0123456789abcdef') or sha(path) != checksum):
        raise ValueError('Exact watch_v24_training.py renderer SHA256 required')
    spec = importlib.util.spec_from_file_location('_v24_direct_pinned_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectSource:
    def __init__(self, renderer, root):
        self.renderer = renderer
        self.root = Path(root)
        self.output = self.root / 'gnn'
        self.read_original = renderer.read_json
        self.log_original = renderer.log_tail
        self.current = None

    def required(self, path):
        try:
            owned(path)
        except FileNotFoundError as error:
            raise self.renderer.JsonRecordError(path, 'Required direct record not yet present', pending=True) from error
        value = self.read_original(path)
        if value is None:
            raise self.renderer.JsonRecordError(path, 'Required direct record not yet present', pending=True)
        return value

    def check(self, condition, path, detail):
        if not condition:
            raise self.renderer.JsonRecordError(path, detail)

    def actor(self, registration, path):
        try:
            process = psutil.Process(registration['pid'])
            if process.status() == psutil.STATUS_ZOMBIE:
                return False
            self.check(process.create_time() == registration['create_time']
                and process.cmdline() == registration['command']
                and (not hasattr(os, 'getuid') or process.uids().real == os.getuid()),
                path, 'Registered PID birth/full argv/UID differs; PID reuse is not training')
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied as error:
            raise self.renderer.JsonRecordError(path, 'Cannot verify registered process: ' + str(error)) from error

    def projection(self):
        reg_path = self.root / 'direct_training_registration.json'
        req_path = self.root / 'direct_training_request.json'
        receipt_path = self.output / 'interrupted_gnn_continuation.json'
        registered, request, receipt = (self.required(path) for path in (reg_path, req_path, receipt_path))
        gpu = request.get('GPU_arm')
        source = receipt.get('source', {})
        latest = source.get('latest', {})
        command = registered.get('command')
        self.check(gpu in (4, 5) and self.root.name == 'gpu' + str(gpu)
            and request.get('format') == 'v24_existing_direct_GPU_exact_continuation_request_v1'
            and registered.get('format') == 'v24_existing_direct_GPU_exact_continuation_registration_v1'
            and registered.get('GPU_arm') == gpu and registered.get('root') == str(self.root)
            and request.get('output') == str(self.output) and registered.get('output') == str(self.output)
            and registered.get('request_sha256') == sha(req_path)
            and request.get('original_epochs') == 40 and request.get('physical_patient_batch') == 4
            and request.get('physical_candidate_batch') == (64 if gpu == 4 else 32)
            and request.get('full_original_population') is True and request.get('debug') is False
            and request.get('PBS_owned_reservation') is False and registered.get('PBS_owned_reservation') is False
            and request.get('six_saved_states_unchanged') is True and request.get('BEST_unchanged') is True
            and registered.get('six_saved_states_unchanged') is True
            and registered.get('old_results_written') is False and registered.get('signals_sent') is False
            and receipt.get('format') == 'v24_interrupted_exact_saved_GNN_continuation_v1'
            and receipt.get('status') == 'PREPARED_EXACT_SAVED_STATE'
            and receipt.get('destination') == str(self.output) and receipt.get('execution_code') == request.get('code')
            and source.get('source_output') == request.get('source_output')
            and source.get('source_code') == request.get('source_code') and source.get('GPU_arm') == gpu
            and receipt.get('six_numerical_states_unchanged') is True
            and receipt.get('scientific_request_unchanged') is True
            and receipt.get('original_files_modified') is False
            and receipt.get('production_optimizer_updates_performed') == 0
            and latest.get('raw_sha256') == request.get('saved_checkpoint') == registered.get('source_checkpoint_sha256')
            and (latest.get('epoch'), latest.get('phase'), latest.get('updates')) ==
                (request.get('checkpoint_epoch'), request.get('checkpoint_phase'), request.get('checkpoint_updates')) ==
                (registered.get('resumed_epoch'), registered.get('resumed_phase'), registered.get('resumed_updates')),
            reg_path, 'Direct registration/request/prepared full GNN source binding differs')
        self.check(type(registered.get('pid')) is int and registered['pid'] > 0
            and type(registered.get('create_time')) in (int, float) and math.isfinite(registered['create_time'])
            and isinstance(command, list) and len(command) >= 5 and all(isinstance(item, str) for item in command)
            and registered.get('python') == command[0]
            and command[1:4] == ['-B', '-u', str(Path(request['code']) / 'tools/run_v24_interrupted_gnn_continuation.py')],
            reg_path, 'Exact direct training PID/birth/entry argv required')
        original = request.get('original_args')
        self.check(isinstance(original, list) and len(original) % 2 == 0, req_path, 'Original scientific argument pairs required')
        expected = ['--action', 'train', '--source-output', request['source_output'], '--source-code', request['source_code'],
            '--interruption-proof', str(self.root / 'original_interruption_witness.json'),
            '--allocation-guard-module', str(self.root.parent / 'direct_GPU45_access_guard.py'),
            '--allocation-guard-sha256', registered.get('guard_sha256'),
            '--allocation-proof', str(self.root / 'direct_GPU_access_proof.json'),
            '--allocation-proof-sha256', registered.get('proof_sha256'), *original, '--output', str(self.output)]
        self.check(command[4:] == expected, reg_path, 'Registered direct command differs from sealed source request')
        cursor_proof = receipt.get('viewer_cursor', {})
        self.check(cursor_proof.get('path') == str(self.output / 'training_continuation.json')
            and sha(cursor_proof['path']) == cursor_proof.get('sha256'), receipt_path, 'Prepared viewer cursor SHA differs')
        cursor = self.required(cursor_proof['path'])
        self.check(cursor.get('source') == source, receipt_path, 'Prepared viewer cursor/source boundary differs')
        alive = self.actor(registered, reg_path)
        state, detail = 'RUNNING', None
        if not alive:
            events, _ = self.renderer.JsonlCursor().read(self.output / 'training/invocations.jsonl')
            cutoff = cursor.get('copied_files', {}).get('training/invocations.jsonl', {}).get('bytes', 0)
            fresh = [row for row in events if row['_viewer_end_offset'] > cutoff]
            last = fresh[-1] if fresh else {}
            if (last.get('status') == 'COMPLETE' and last.get('full_training') is True
                    and last.get('debug') is False and last.get('completed_epochs') == 40
                    and last.get('optimizer_updates') == 680 and last.get('actual_CUDA') is True):
                state = 'COMPLETE'
            elif last.get('status') == 'PAUSED' and last.get('debug') is False:
                state = 'PAUSED'
            else:
                state, detail = 'EXITED_WITHOUT_COMPLETION', '등록된 학습 프로세스 종료; 전체 학습 완료 기록 없음'
        value = dict(status=state, stage='train', child_pid=registered['pid'],
            child_create_time=registered['create_time'], error=detail,
            request=dict(GPU=gpu, RAM_GiB=64, production_output=str(self.output)))
        self.current = value
        return value

    def read_json(self, path):
        if Path(path) != self.output / 'status.json':
            return self.read_original(path)
        try:
            return self.projection()
        except (OSError, ValueError, KeyError, TypeError) as error:
            if isinstance(error, self.renderer.JsonRecordError):
                raise
            raise self.renderer.JsonRecordError(path, type(error).__name__ + ': ' + str(error)) from error

    def log_tail(self, path, byte_count=16384):
        return self.log_original(self.root / 'train.log' if Path(path) == self.output / 'console.log' else path, byte_count)

    def install(self):
        renderer = self.renderer
        source = self
        original = renderer.Viewer
        class DirectViewer(original):
            def _snapshot(self, root):
                value = super()._snapshot(root)
                if source.current is not None and root == source.output:
                    state = source.current['status']
                    value['status'] = 'TRAINING_COMPLETE' if state == 'COMPLETE' else state
                    value['full_training_completed'] = state == 'COMPLETE'
                    value['direct_arm_root'] = str(source.root)
                    value['status_source'] = 'verified direct registration/PID; actual invocation if exited'
                    if state == 'COMPLETE':
                        value.update(epoch=40, epochs=40, completed_epochs=40, optimizer_updates=680,
                            stage='complete', stage_label='전체 GNN 학습 완료', progress=None)
                return value
        renderer.Viewer = DirectViewer
        renderer.read_json = self.read_json
        renderer.log_tail = self.log_tail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--renderer-sha', required=True)
    parser.add_argument('--arm-root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    owned(args.arm_root)
    renderer = load_renderer(args.renderer, args.renderer_sha)
    source = DirectSource(renderer, args.arm_root)
    source.install()
    forwarded = ['--job-root', str(source.output), '--interval', str(args.interval)]
    if args.once:
        forwarded.append('--once')
    if args.json:
        forwarded.append('--json')
    renderer.main(forwarded)


if __name__ == '__main__':
    main()
