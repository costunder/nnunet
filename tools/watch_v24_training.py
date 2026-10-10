"""Read-only MobaXterm progress viewer; never imports or controls training.

Examples: --log /path/to/stable/gpu5.log, --job-root /path/to/run/gpu5,
or --pointer /path/to/gpu5.json. Ctrl-C stops this viewer alone.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

PHASES = ('initial_full_validation', 'train_probe', 'stage_validation', 'full_validation')
LABELS = dict(training='학습', initial_full_validation='초기 전체 validation',
    train_probe='train 범위 평가', stage_validation='단계 validation',
    full_validation='전체 validation', epoch_completion='epoch 정리',
    upper_prepare='U 후보 그래프 준비',
    prepare_inputs='CPU 전처리', shared_input_wait='공유 전처리 대기', calibrate='batch 실측')
LABELS.update(prepare_continuation='체크포인트 이어받기 준비', calibrate_native='nnUNet batch 실측',
    prepare_bank='CP bank 준비', materialize_bank='CP bank 저장', prepare_native='nnUNet 입력 준비',
    nnunet_training='nnUNet 학습')


def read_json(path):
    try:
        with Path(path).open(encoding='utf8') as stream:
            return json.load(stream)
    except FileNotFoundError:
        return None


def resolve_job_root(*, job_root=None, log=None, pointer=None):
    """Resolve again every poll, so a stable log symlink can follow a new run."""
    if pointer is not None:
        path = Path(pointer)
        try:
            text = path.read_text(encoding='utf8').strip()
        except FileNotFoundError:
            return None
        if text.startswith('{'):
            value = json.loads(text)
            if value.get('job_root') or value.get('root'):
                job_root = value.get('job_root') or value['root']
            elif value.get('log') or value.get('console_log'):
                log = value.get('log') or value['console_log']
            else:
                raise ValueError('Pointer requires job_root/root or log/console_log')
        else:
            job_root = text
        candidate = Path(job_root or log)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if job_root:
            job_root = candidate
        else:
            log = candidate
    if log is not None:
        resolved = Path(log).resolve()
        return resolved.parent if resolved.exists() else None
    return Path(job_root).resolve() if job_root else None


class JsonlCursor:
    """Consume only new complete lines; tolerate an in-progress final line."""
    def __init__(self):
        self.identity = None
        self.offset = 0

    def read(self, path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return [], False
        identity = (stat.st_dev, stat.st_ino)
        reset = identity != self.identity or stat.st_size < self.offset
        if reset:
            self.identity, self.offset = identity, 0
        rows = []
        with path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                if line.strip():
                    row = json.loads(line)
                    row['_viewer_end_offset'] = stream.tell()
                    rows.append(row)
                self.offset = stream.tell()
        return rows, reset


def log_tail(path, byte_count=16384):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-byte_count))
            return stream.read().decode(errors='replace').splitlines()
    except FileNotFoundError:
        return []


def production_output(request):
    if request.get('production_output'):
        return Path(request['production_output'])
    for stage in request.get('stages', []):
        command = stage.get('command', [])
        if '--output' in command:
            return Path(command[command.index('--output')+1])
    return None


class Viewer:
    def __init__(self):
        self.root = None

    def snapshot(self, root):
        if root != self.root:
            self.root = root
            self.cursors = {name: JsonlCursor() for name in ('update_timing', 'validation_timing', 'curve', 'invocations', 'failures')}
            self.events = {name: [] for name in self.cursors}
        if root is None:
            return dict(status='WAITING', stage='로그 경로 대기', job_root=None, progress=None)
        status = read_json(root/'status.json')
        if status is None:
            return dict(status='WAITING', stage='status 기록 대기', job_root=str(root), progress=None)
        request = status.get('request') or read_json(root/'request.json') or {}
        output = production_output(request)
        training = output/'training' if output else None
        contract = read_json(training/'execution_contract.json') if training else None
        contract = contract or {}
        ownership = read_json(training/'training_identity.json') if training else None
        binding = (ownership or {}).get('binding') or contract.get('binding', {})
        train_total = len(binding.get('train_cases', [])) or None
        val_total = len(binding.get('val_cases', [])) or None
        epoch_total = binding.get('epochs') or contract.get('epochs')
        continuation = read_json(output/'training_continuation.json') if output else None
        baseline = (continuation or {}).get('source', {}).get('latest', {})
        copied = (continuation or {}).get('copied_files', {})
        continuation_time = (continuation or {}).get('created_at', 0)
        if training:
            for name, cursor in self.cursors.items():
                rows, reset = cursor.read(training/(name+'.jsonl'))
                if reset:
                    self.events[name] = []
                cutoff = copied.get('training/'+name+'.jsonl', {}).get('bytes', 0)
                self.events[name].extend(self._event(name, row, row['_viewer_end_offset'] > cutoff) for row in rows)
        events = self.events
        updates = [row for row in events['update_timing'] if row.get('status') == 'OPTIMIZER_UPDATED']
        latest_update = updates[-1] if updates else None
        validations = events['validation_timing']
        last_invocation = events['invocations'][-1] if events['invocations'] else {}
        complete_epochs = max([*baseline.get('history_epochs', []), *(row['epoch'] for row in events['curve'])], default=0)
        reports = {}
        candidates = []
        if baseline:
            candidates.append((continuation_time, baseline['phase'], baseline['epoch'], 'sealed_continuation_cursor'))
        fresh_updates = [row for row in updates if row['post_continuation']]
        fresh_validations = [row for row in validations if row['post_continuation']]
        if training:
            for phase in PHASES:
                paths = sorted(training.glob(phase+'_epoch_*.json'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
                if not paths:
                    continue
                path = paths[-1]
                report = read_json(path)
                if report is not None:
                    reports[phase] = {key: report.get(key) for key in ('epoch', 'active_u', 'updates', 'metrics')}
                    next_phase = {'initial_full_validation': 'training', 'train_probe': 'stage_validation',
                        'stage_validation': 'full_validation', 'full_validation': 'epoch_completion'}[phase]
                    if path.stat().st_mtime > continuation_time:
                        candidates.append((path.stat().st_mtime, next_phase, max(1, report['epoch']), 'completed_report'))
            if fresh_updates:
                row = fresh_updates[-1]
                done = sum(r['patients'] for r in fresh_updates if r['epoch'] == row['epoch'])
                if row['epoch'] == baseline.get('epoch'):
                    done += baseline.get('train_position', 0)
                next_phase = 'train_probe' if train_total is not None and done == train_total else 'training'
                candidates.append(((training/'update_timing.jsonl').stat().st_mtime, next_phase, row['epoch'], 'update_timing'))
            if fresh_validations:
                row = fresh_validations[-1]
                candidates.append(((training/'validation_timing.jsonl').stat().st_mtime, row['phase'], row['epoch'], 'validation_timing'))
            if any(row['post_continuation'] for row in events['curve']):
                candidates.append(((training/'curve.jsonl').stat().st_mtime, 'training', complete_epochs+1, 'completed_curve'))
        pipeline_stage = status.get('stage')
        phase, epoch, phase_source = pipeline_stage, None, 'pipeline_status'
        if pipeline_stage == 'train':
            if candidates:
                # Epoch/phase progress is monotonic. File timestamp resolution
                # may tie across a completed curve and the previous update.
                order = dict(initial_full_validation=0, training=1, train_probe=2,
                    stage_validation=3, full_validation=4, epoch_completion=5, complete=6)
                _, phase, epoch, phase_source = max(candidates,
                    key=lambda item: (item[2], order[item[1]], item[0]))
            elif binding:
                phase, epoch, phase_source = 'initial_full_validation', 1, 'execution_contract_before_first_batch'
        active_u = latest_update.get('active_u') if latest_update else baseline.get('active_u')
        if phase == 'training' and events['curve']:
            curve = events['curve'][-1]
            if (epoch == curve['epoch']+1 and (latest_update is None or latest_update['epoch'] != epoch)
                    and curve.get('next_active_u') is not None):
                active_u = curve['next_active_u']
        progress = None
        if phase == 'training' and epoch is not None:
            rows = [r for r in fresh_updates if r['epoch'] == epoch]
            initial = baseline.get('train_position', 0) if baseline.get('epoch') == epoch else 0
            progress = dict(done=initial+sum(r['patients'] for r in rows), total=train_total, unit='환자')
        elif phase in PHASES and epoch is not None:
            rows = [r for r in fresh_validations if r['epoch'] == epoch and r['phase'] == phase]
            total = train_total if phase == 'train_probe' else val_total
            initial = baseline.get('evaluation_position', 0) if baseline.get('epoch') == epoch and baseline.get('phase') == phase else 0
            progress = dict(done=initial+sum(r['patients'] for r in rows), total=total, unit='환자')
            active_u = rows[-1]['active_u'] if rows else 128 if phase in ('initial_full_validation', 'full_validation') else active_u
        elif pipeline_stage == 'prepare_inputs':
            for line in reversed(log_tail(root/'console.log')):
                match = re.search(r'UPPER PREPARE completed=(\d+)/(\d+) active_U=(\d+)', line)
                canonical = re.search(r'CPU canonical .*: (\d+)/(\d+) records', line)
                if match:
                    done, total, active_u = map(int, match.groups())
                    progress = dict(done=done, total=total, unit='upper 환자'); break
                if canonical:
                    done, total = map(int, canonical.groups())
                    progress = dict(done=done, total=total, unit='canonical 기록'); break
        if pipeline_stage == 'train' and request.get('GPU') in (5, 6):
            for line in reversed(log_tail(root/'console.log')):
                match = re.search(r'UPPER PREPARE completed=(\d+)/(\d+) active_U=(\d+)', line)
                if match:
                    done, total, prepared_u = map(int, match.groups())
                    if done < total:
                        phase, phase_source, active_u = 'upper_prepare', 'unfinished console upper preparation', prepared_u
                        progress = dict(done=done, total=total, unit='upper 환자')
                    break
        effective_status = status.get('status', 'UNKNOWN')
        if effective_status not in ('RUNNING', 'FAILED', 'PREPARING', 'WAITING_FOR_SHARED_CPU_INPUT') and last_invocation.get('status') == 'PAUSED':
            effective_status = 'PAUSED'
        elif effective_status == 'COMPLETE':
            effective_status = 'TRAINING_COMPLETE' if last_invocation.get('full_training') is True else 'PIPELINE_FINISHED'
        rss, rss_source = self._rss(status, latest_update)
        calibration = read_json(output/'calibration.json') if output else None
        error = status.get('error')
        fresh_failures = [row for row in events['failures'] if row['post_continuation']]
        if effective_status == 'FAILED' and fresh_failures:
            error = fresh_failures[-1].get('error') or error
        snapshot = dict(job_root=str(root), production_output=str(output) if output else None,
            GPU=request.get('GPU'), status=effective_status, stage=phase, stage_label=LABELS.get(phase, phase),
            phase_source=phase_source, epoch=epoch, epochs=epoch_total, completed_epochs=complete_epochs,
            active_U=active_u, progress=progress, optimizer_updates=latest_update.get('update') if latest_update else baseline.get('updates'),
            loss=latest_update.get('loss') if latest_update else None, loss_source='last successful train update' if latest_update else None,
            RSS_GiB=rss/2**30 if rss is not None else None, RSS_source=rss_source,
            RAM_limit_GiB=request.get('RAM_GiB'), reports=reports, error=error,
            selected_batch=calibration.get('selected_physical_patient_batch') if calibration else None,
            selected_chunk=calibration.get('selected_physical_candidate_batch') if calibration else None,
            full_training_completed=effective_status == 'TRAINING_COMPLETE',
            calibration_progress=self._calibration(root) if pipeline_stage == 'calibrate' else None)
        if request.get('GPU') == 1:
            native = parse_nnunet_lines(log_tail(root/'console.log', 65536))
            if native:
                total = native_epoch_total(request)
                snapshot.update(stage='nnunet_training', stage_label=LABELS['nnunet_training'],
                    epoch=native['epoch_index']+1, epochs=total, phase_source='nnUNet console log',
                    completed_epochs=native['epoch_index']+int(native['epoch_finished']), progress=None,
                    native_metrics=native, active_U=None)
        return snapshot

    @staticmethod
    def _event(name, row, fresh=True):
        if name in ('update_timing', 'validation_timing'):
            result = {key: row.get(key) for key in ('status', 'epoch', 'update', 'active_u', 'phase', 'loss', 'RSS_bytes')}
            result['patients'] = len(row.get('case_ids', []))
            result['post_continuation'] = fresh
            return result
        if name == 'curve':
            return dict(epoch=row['epoch'], next_active_u=row.get('curriculum_transition', {}).get('next_active_u'),
                post_continuation=fresh)
        return dict({key: row.get(key) for key in ('status', 'full_training', 'completed_epochs', 'error')}, post_continuation=fresh)

    @staticmethod
    def _rss(status, latest):
        if status.get('status') == 'RUNNING' and status.get('child_pid'):
            try:
                import psutil
            except ImportError:
                pass
            else:
                try:
                    process = psutil.Process(status['child_pid'])
                    if (abs(process.create_time()-status['child_create_time']) < .1
                            and (not hasattr(os, 'getuid') or process.uids().real == os.getuid())):
                        total = process.memory_info().rss
                        for child in process.children(recursive=True):
                            try:
                                if not hasattr(os, 'getuid') or child.uids().real == os.getuid():
                                    total += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                continue
                        return total, 'live owned arm process tree'
                except psutil.NoSuchProcess:
                    pass
        if latest and latest.get('RSS_bytes') is not None:
            return latest['RSS_bytes'], 'last successful train update'
        return None, None

    @staticmethod
    def _calibration(root):
        state = None
        for line in log_tail(root/'console.log'):
            match = re.search(r'v24 CALIBRATION patients(\d+)/chunk(\d+)/allP\+128U', line)
            if match:
                state = dict(patients=int(match[1]), chunk=int(match[2]), repeats_completed=0)
            match = re.search(r'v24 CALIBRATION repeat(\d+) ([\d.]+)s', line)
            if match and state:
                state.update(repeats_completed=int(match[1]), last_seconds=float(match[2]))
        return state


def parse_nnunet_lines(lines):
    """Native textual metrics are labelled pseudo-Dice, never final test Dice."""
    current = None
    number = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    for line in lines:
        epoch = re.search(r'\bEpoch\s+(\d+)\b', line)
        if epoch:
            current = dict(epoch_index=int(epoch[1]), epoch_finished=False)
        if current is None:
            continue
        for key in ('train_loss', 'val_loss'):
            match = re.search(r'\b'+key+r'\s*[:=]?\s*'+number, line)
            if match:
                current[key] = float(match[1])
        dice = re.search(r'Pseudo\s+[Dd]ice\s*:?\s*(\[[^\]]+\])', line)
        if dice:
            # NumPy2 repr appears in the actual nnUNet console. Strip only
            # numeric scalar wrappers; no eval or training/NumPy import.
            cleaned = re.sub(r'(?:np\.)?float(?:32|64)\(\s*'+number+r'\s*\)', r'\1', dice[1])
            current['pseudo_dice'] = json.loads(cleaned)
        if 'Epoch time' in line:
            current['epoch_finished'] = True
    return current


def native_epoch_total(request):
    for stage in request.get('stages', []):
        command = stage.get('command', [])
        if '--native' not in command:
            continue
        native = read_json(Path(command[command.index('--native')+1]))
        if not native or not native.get('root'):
            continue
        started = sorted([*Path(native['root']).glob('training_started_*.json'),
            *Path(native['root']).glob('production_runtime_binding_*.json')], key=lambda p:p.stat().st_mtime)
        if started:
            return (read_json(started[-1]) or {}).get('epochs')
    return None


def text_snapshot(snapshot):
    s = snapshot
    header = f"GPU{s.get('GPU', '?')} | {s['status']} | {s.get('stage_label') or s['stage']}"
    if s.get('epoch') is not None:
        header += f" | epoch {s['epoch']}/{s.get('epochs') or '?'}"
    if s.get('active_U') is not None:
        header += f" | U{s['active_U']}"
    p = s.get('progress')
    if p:
        header += f" | {p['done']}/{p['total'] or '?'} {p['unit']} 완료 기록"
    details = []
    if s.get('loss') is not None:
        details.append(f"loss {s['loss']:.6f} (최근 train update {s['optimizer_updates']})")
    if s.get('RSS_GiB') is not None:
        details.append(f"RAM {s['RSS_GiB']:.2f}/{s.get('RAM_limit_GiB') or '?'}GiB ({s['RSS_source']})")
    for phase, report in s.get('reports', {}).items():
        metrics = report.get('metrics') or {}
        mrr, top1 = metrics.get('per_P_patient_mrr'), metrics.get('per_P_patient_top1')
        if mrr is not None and top1 is not None:
            details.append(f"{LABELS[phase]} e{report['epoch']} U{report['active_u']}: MRR {mrr:.4f} top1 {top1:.4f}")
    if s.get('calibration_progress'):
        c = s['calibration_progress']
        details.append(f"batch {c['patients']} chunk {c['chunk']}: repeat {c['repeats_completed']} 완료")
    if s.get('native_metrics'):
        native = s['native_metrics']
        details.append(' | '.join(f'{key} {native[key]}' for key in ('train_loss', 'val_loss', 'pseudo_dice') if key in native))
    if s.get('error'):
        details.append('오류: '+s['error'])
    if s['status'] == 'PIPELINE_FINISHED':
        details.append('파이프라인 종료; 전체 학습 완료는 확인되지 않음')
    return header+'\n'+'\n'.join(details)


class TerminalView:
    def __init__(self, stream):
        from tqdm import tqdm
        self.stream = stream
        self.bars = [tqdm(total=None, position=i, leave=False, dynamic_ncols=True, file=stream, disable=False,
            bar_format='{desc} |{bar}| {n_fmt}/{total_fmt}' if i < 2 else '{desc}') for i in range(4)]

    def render(self, snapshot):
        s = snapshot
        top, current, detail, metrics = self.bars
        top.set_description_str(f"GPU{s.get('GPU', '?')} {s['status']} epoch {s.get('epoch') or '?'}/{s.get('epochs') or '?'}", refresh=False)
        top.total, top.n = snapshot.get('epochs'), snapshot.get('completed_epochs', 0)
        p = snapshot.get('progress') or {}
        current.total, current.n = p.get('total'), p.get('done', 0)
        current.set_description_str((snapshot.get('stage_label') or snapshot['stage'])+(f" U{s['active_U']}" if s.get('active_U') is not None else ''), refresh=False)
        info = []
        if s.get('loss') is not None:
            info.append(f"last train loss {s['loss']:.5f} upd{s['optimizer_updates']}")
        if s.get('RSS_GiB') is not None:
            info.append(f"RAM {s['RSS_GiB']:.1f}/{s.get('RAM_limit_GiB') or '?'}GiB")
        if s.get('error'):
            info.append(s['error'][:100])
        if s.get('native_metrics'):
            n = s['native_metrics']
            info += [f'{key} {n[key]}' for key in ('train_loss', 'val_loss') if key in n]
        detail.set_description_str(' | '.join(info) or '현재 단계 완료 기록 대기', refresh=False)
        shown = []
        reports = s.get('reports', {})
        for name, label in (('stage_validation', 'stage'), ('full_validation' if 'full_validation' in reports else 'initial_full_validation', 'full128')):
            report = reports.get(name)
            if report and report.get('metrics'):
                m = report['metrics']
                if m.get('per_P_patient_mrr') is not None and m.get('per_P_patient_top1') is not None:
                    shown.append(f"{label} e{report['epoch']} U{report['active_u']} MRR {m['per_P_patient_mrr']:.4f}/T1 {m['per_P_patient_top1']:.4f}")
        if s.get('native_metrics', {}).get('pseudo_dice') is not None:
            shown.append('pseudo Dice '+str(s['native_metrics']['pseudo_dice']))
        metrics.set_description_str(' | '.join(shown) or '평가 metric 미기록', refresh=False)
        for bar in self.bars:
            bar.refresh()

    def close(self):
        for bar in reversed(self.bars):
            bar.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--job-root', type=Path)
    source.add_argument('--log', type=Path, help='Stable console.log symlink; follows new target each poll')
    source.add_argument('--pointer', type=Path, help='JSON job_root/root/log/console_log or plain job-root path')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--once', action='store_true', help='One readable snapshot; no terminal animation')
    parser.add_argument('--json', action='store_true', help='One JSON metadata snapshot')
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('--interval must be positive')
    viewer = Viewer()
    terminal = TerminalView(sys.stdout) if sys.stdout.isatty() and not (args.once or args.json) else None
    previous = None
    try:
        while True:
            root = resolve_job_root(job_root=args.job_root, log=args.log, pointer=args.pointer)
            snapshot = viewer.snapshot(root)
            if args.json:
                print(json.dumps(snapshot, ensure_ascii=False)); return
            if terminal:
                terminal.render(snapshot)
            else:
                signature = {key: value for key, value in snapshot.items() if key not in ('RSS_GiB',)}
                if signature != previous:
                    print(text_snapshot(snapshot), flush=True)
                    previous = signature
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if terminal:
            terminal.close(); terminal = None
        print('\n진행 뷰어만 종료했습니다. 학습 작업은 계속됩니다.', flush=True)
    finally:
        if terminal:
            terminal.close()


if __name__ == '__main__':
    main()
