"""Read-only full642 scoring viewer; no Torch, CUDA, checkpoint load or signals.

This viewer reports sealed metadata and current PID identity. It does not
revalidate the GNN's six numerical states or claim native training completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import psutil

PHASE = 'matched_Basic_current_BEST_scores'


class PendingJSON(ValueError):
    pass


def read_record(path):
    try:
        raw = path.read_text(encoding='utf8')
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as error:
        raise PendingJSON(str(path) + ': incomplete UTF8: ' + str(error)) from error
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PendingJSON(str(path) + ': ' + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError(str(path) + ': JSON object required')
    return value


def complete_events(path):
    """Ignore only an unfinished final line; malformed completed JSON is visible."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return [], False
    events = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            return events, bool(line)
        if not line.lstrip().startswith(b'{'):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('Console JSON object required: ' + str(path))
        if value.get('phase') == PHASE:
            if (value.get('GPU') != 6 or value.get('total_sources') != 642
                    or type(value.get('completed_sources')) is not int or not 0 <= value['completed_sources'] <= 642
                    or value.get('complete_CP_positions') != value['completed_sources'] * 128
                    or value.get('debug') is not False or value.get('production_optimizer_updates') != 0
                    or type(value.get('seconds')) not in (int, float) or not math.isfinite(value['seconds']) or value['seconds'] < 0):
                raise ValueError('Actual full642/128 frozen scoring event differs')
            events.append(value)
    return events, False


def process_snapshot(registration):
    """PID reuse/foreign argv is an error; a dead PID is never RUNNING."""
    try:
        process = psutil.Process(registration['pid'])
        if (process.create_time() != registration['create_time'] or process.cmdline() != registration['command']
                or process.uids().real != os.getuid()):
            raise ValueError('Registered scoring PID/birth/full argv/UID changed')
        status = process.status()
        if status == psutil.STATUS_ZOMBIE:
            return dict(alive=False, process_status=status)
        return dict(alive=True, process_status=status, RSS_GiB=process.memory_info().rss / 2**30)
    except psutil.NoSuchProcess:
        return dict(alive=False, process_status='absent')


def snapshot(root):
    request, registration = (read_record(root / name) for name in ('request.json', 'registration.json'))
    if request is None or registration is None:
        return dict(status='WAITING_METADATA', detail='request/registration 기록 대기', source_done=None, CP_done=None)
    if (request.get('format') != 'v24_full642_depth1_current_BEST_scoring_request_v1'
            or registration.get('format') != 'v24_full642_depth1_current_BEST_scoring_registration_v1'
            or request.get('gpu') != 6 or request.get('source_entries') != 642 or request.get('CP_positions') != 82176
            or request.get('debug') is not False or request.get('full642_scoring') is not True
            or registration.get('sources') != 642 or registration.get('CP_positions') != 82176
            or registration.get('root') != str(root) or request.get('output') != str(root / 'bank')
            or registration.get('request_sha256') != hashlib.sha256((root / 'request.json').read_bytes()).hexdigest()
            or registration.get('command', [])[-2:] != ['--request', str(root / 'request.json')]
            or type(registration.get('pid')) is not int or registration['pid'] <= 0
            or type(registration.get('create_time')) not in (int, float)):
        raise ValueError('Sealed full642/82176 GPU6 request/registration differs')
    process = process_snapshot(registration)
    events, partial = complete_events(root / 'console.log')
    last = events[-1] if events else None
    source_done = last['completed_sources'] if last else None
    started, failed, complete = (read_record(root / name) for name in ('started.json', 'failed.json', 'complete.json'))
    state = 'SCORING_PROCESS_ALIVE' if process['alive'] and started else 'STARTING' if process['alive'] else 'EXITED_WITHOUT_COMPLETION'
    detail = None
    if failed is not None:
        if failed.get('status') != 'FULL642_SCORING_FAILED':
            raise ValueError('Actual failed scoring record differs')
        state, detail = 'SCORING_FAILED', failed.get('error')
    elif complete is not None:
        if (complete.get('status') != 'FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING'
                or complete.get('completed_sources') != 642 or complete.get('completed_CP_positions') != 82176
                or complete.get('full_native_training') is not False or complete.get('full_native_evaluation') is not False):
            raise ValueError('Actual full scoring/native-pending completion record differs')
        state, source_done = 'SCORING_COMPLETE_NATIVE_STORAGE_PENDING', 642
        detail = complete.get('native_pending_reason')
    elif started is not None and (started.get('status') != 'FULL642_SCORING_RUNNING' or started.get('debug') is not False
            or started.get('full_native_training') is not False or started.get('full_native_evaluation') is not False):
        raise ValueError('Actual full scoring start record differs')
    latest_age = None if last is None else max(0, time.time() - (root / 'console.log').stat().st_mtime)
    return dict(status=state, source_done=source_done, source_total=642,
        CP_done=None if source_done is None else source_done * 128, CP_total=82176,
        last_batch_seconds=last['seconds'] if last else None, seconds_since_console_write=latest_age,
        partial_console_line_pending=partial, detail=detail, native_pending_reason=request.get('native_pending_reason'),
        GNN_completion_evidence_reverified=False, full_native_training=False, full_native_evaluation=False, **process)


def display(value):
    done = '?' if value.get('source_done') is None else str(value['source_done'])
    positions = '?' if value.get('CP_done') is None else str(value['CP_done'])
    lines = [f"GPU6 | CP 후보 채점 | {value['status']}",
        f"source {done}/642 | CP 후보 {positions}/82176 (source당 전체128)",
        '입력: GPU6 GNN BEST | 현재 단계: CP 후보 채점',
        '후속 nnU-Net: 아직 시작 전(저장공간 대기)']
    if value.get('last_batch_seconds') is not None:
        lines.append(f"최근 실제 batch {value['last_batch_seconds']:.1f}초 | console 갱신 {value['seconds_since_console_write']:.0f}초 전")
    if value.get('RSS_GiB') is not None:
        lines.append(f"프로세스 {value['process_status']} | RAM {value['RSS_GiB']:.2f} GiB")
    if value.get('partial_console_line_pending'):
        lines.append('console 마지막 줄 기록 중; 완료된 줄만 표시')
    if value.get('detail') and value['status'] != 'SCORING_COMPLETE_NATIVE_STORAGE_PENDING':
        lines.append(str(value['detail']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--no-clear', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    root = args.root.absolute()
    if (not root.is_dir() or any(path.is_symlink() for path in (root, *root.parents))
            or hasattr(os, 'getuid') and root.stat().st_uid != os.getuid()):
        parser.error('Owned scoring root without symlink aliases required')
    problem, since = None, None
    try:
        while True:
            try:
                value = snapshot(root)
            except PendingJSON as error:
                if problem != str(error):
                    problem, since = str(error), time.monotonic()
                value = dict(status='READ_PENDING' if time.monotonic() - since < 10 else 'READ_ERROR', detail=str(error))
            except (ValueError, OSError, psutil.AccessDenied) as error:
                value = dict(status='READ_ERROR', detail=type(error).__name__ + ': ' + str(error))
            else:
                problem, since = None, None
            if args.json:
                print(json.dumps(value, ensure_ascii=False)); return
            if sys.stdout.isatty() and not args.no_clear:
                print('\033[2J\033[H', end='')
            print(display(value), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습/scoring 작업은 계속됩니다.', flush=True)


if __name__ == '__main__':
    main()
