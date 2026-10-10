"""Read-only compact CP monitor; native nnUNet logs pass through unchanged."""
import argparse
import json
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return '--'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'


def fit(text, width):
    used, chars = 0, []
    for char in text:
        used += 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used > width:
            break
        chars.append(char)
    return ''.join(chars)


class Monitor:
    def __init__(self, job):
        self.job = job
        self.request = json.loads((job / 'request.json').read_text())
        self.output = Path(self.request['production_output'])
        self.records = {}
        self.process = None
        self.cpu = None
        self.drawn = 0
        self.stage = None
        self.offset = None

    def rows(self, state):
        import psutil
        saving = False
        for path in (self.output / 'score_receipts').glob('*.json'):
            if path.name in self.records:
                continue
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                # Receipts are written after successful scoring; retry an active write.
                saving = True
                continue
            self.records[path.name] = (path.stat().st_mtime, row)
        ordered = sorted(self.records.values(), key=lambda item: item[0])
        total = int(self.request['source_payloads'])
        count = len(ordered)
        latest_time, latest = ordered[-1] if ordered else (None, {})
        reused_path = self.output / 'reused_scores.json'
        reused = json.loads(reused_path.read_text())['entries'] if reused_path.is_file() else 0
        # Previously completed sources must not inflate this run's speed/ETA.
        fresh = ordered[reused:]
        recent = fresh[-12:]
        mean = sum(row['seconds'] for _, row in recent) / len(recent) if recent else None
        remaining = mean * (total - count) if mean is not None else None
        elapsed = time.time() - state.get('stage_started', state['started'])
        filled = min(28, int(28 * count / total))
        bar = '█' * filled + '░' * (28 - filled)
        resources = '프로세스 상태 확인 중'
        pid = state.get('child_pid')
        if pid is not None:
            try:
                if self.process is None or self.process.pid != pid:
                    self.process = psutil.Process(pid)
                    if abs(self.process.create_time() - state['child_create_time']) > .001:
                        raise RuntimeError('Recorded process identity changed')
                    self.process.cpu_percent()
                    self.cpu = None
                else:
                    self.cpu = self.process.cpu_percent()
                cpu = '--' if self.cpu is None else f'{self.cpu:.0f}%'
                resources = f'CPU {cpu} / 4코어 | RAM {self.process.memory_info().rss / 2**30:.1f}/{self.request["RAM_GiB"]} GiB'
            except psutil.NoSuchProcess:
                resources = '채점 프로세스 종료 — 단계 상태 갱신 확인 중'
        result = subprocess.run(['nvidia-smi', '--id=' + str(self.request['GPU']),
            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=False)
        if result.returncode == 0:
            utilization, memory = result.stdout.strip().split(',')
            resources += f' | GPU {utilization.strip()}% · {float(memory) / 1024:.1f} GiB'
        else:
            resources += ' | GPU 조회 실패'
        recent_text = ('최근 완료: ' + latest['case_id'] + ' / source ' + str(latest['source_component'])
            + f' | {latest["seconds"]:.1f}초 | batch {latest["physical_candidate_batch"]}') if latest else '첫 source 준비 중'
        quiet = f'마지막 완료 {max(0, int(time.time() - latest_time))}초 전' if latest_time else '첫 완료 대기'
        if saving:
            quiet += ' · 결과 저장 중'
        return [f'GPU{self.request["GPU"]} | CP 후보 채점 | {state["status"]}',
            f'{bar} {count}/{total} ({count / total:.1%})',
            f'이번 실행 {duration(elapsed)} | 예상 남음 {duration(remaining)} · 최근 처리시간 기준',
            recent_text, resources, quiet + ' | 다음: nnUNet 250epoch → BEST 평가']

    def draw(self, rows, once=False):
        if once or not sys.stdout.isatty():
            print('\n'.join(rows), flush=True)
            return
        width = max(20, shutil.get_terminal_size((110, 24)).columns - 1)
        if self.drawn:
            sys.stdout.write(f'\033[{self.drawn}F')
        sys.stdout.write(''.join('\033[2K' + fit(line, width) + '\n' for line in rows))
        self.drawn = len(rows)
        sys.stdout.flush()

    def native_log(self, stage, once=False):
        path = self.job / (stage + '.log')
        if self.stage != stage:
            if self.drawn:
                self.draw([''] * self.drawn)
            self.drawn = 0
            print(f'GPU{self.request["GPU"]} | {stage} | 원본 로그', flush=True)
            self.stage, self.offset = stage, None
        if not path.is_file():
            return
        with path.open('rb') as stream:
            if self.offset is None:
                stream.seek(max(0, path.stat().st_size - 16000))
                lines = stream.read().decode(errors='replace').splitlines(keepends=True)
                text = ''.join(lines[-35:])
            else:
                stream.seek(self.offset)
                text = stream.read().decode(errors='replace')
            self.offset = stream.tell()
        print(text, end='', flush=True)

    def run(self, once=False):
        while True:
            state = json.loads((self.job / 'status.json').read_text())
            if state.get('stage') == 'score_full_historical_bank':
                self.draw(self.rows(state), once)
            elif state.get('stage'):
                self.native_log(state['stage'], once)
            if state['status'] in ('FAILED', 'COMPLETE', 'PAUSED_BETWEEN_STAGES'):
                print(state['status'] + ': ' + state.get('error', ''), flush=True)
                if state['status'] == 'FAILED':
                    self.native_log(state['stage'], True)
                return
            if once:
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        Monitor(args.job.resolve(strict=True)).run(args.once)
    except KeyboardInterrupt:
        print('\n보기 종료. 서버 작업은 계속 실행됩니다.', flush=True)


if __name__ == '__main__':
    main()
