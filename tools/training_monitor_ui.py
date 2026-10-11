"""Version-independent, read-only GNN and CP monitor presentation.

Adapters supply the normalized metadata dictionary; this module reads no job,
file, model or environment variable and starts no process. Missing measurements
remain missing. Plain text keeps complete single-line warnings and errors; only
an adequately sized interactive terminal truncates to its Unicode display width.
"""
from collections.abc import Mapping
import math
from numbers import Real
import os
import re
import unicodedata


ROW_KEYS = ('header', 'state', 'gnn_epoch', 'gnn_progress', 'gnn_loss',
    'stage_validation', 'full_validation', 'cp_sources', 'cp_positions',
    'native_epoch', 'native_loss', 'native_pseudo_dice', 'resources',
    'cp_timing', 'waiting', 'warning_error')
ROW_LABELS = ('실험', '상태', 'GNN 학습', 'GNN 진행', 'GNN loss',
    'GNN 단계검증', 'GNN 전체검증', 'CP source', 'CP 위치',
    'nnU-Net 학습', 'nnU-Net loss', 'nnU-Net pseudo Dice', 'RAM/프로세스',
    'CP 시간/로그', '대기/후속 controller', '경고/오류')
ROWS = 16
UI_VERSION = 1
TASK_UI_VERSION = 2
GNN_ROW_LABELS = ('실험/상태', 'GNN 학습', 'GNN 진행', 'GNN loss',
    'GNN 단계검증', 'GNN 전체검증', 'RAM/프로세스', '대기/후속 controller', '경고/오류')
CP_ROW_LABELS = ('실험/상태', 'CP source', 'CP 위치', 'CP 입력',
    'RAM/프로세스', 'CP 시간/로그', '대기/후속 controller', '경고/오류')
_ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[PX^_][^\x1b]*(?:\x1b\\)|[@-_])')
_SOURCE_NAMES = {'current_process_tree': '현재 프로세스 트리',
    'live_process_tree': '현재 프로세스 트리', 'actual_process_tree': '현재 프로세스 트리',
    'history': '저장된 학습 이력', 'training_history': '저장된 학습 이력',
    'last_history': '저장된 학습 이력', 'last_confirmed_history': '저장된 학습 이력',
    'last_valid_snapshot': '이전 정상 기록', 'stale_snapshot': '이전 정상 기록'}


def sanitize(value):
    """Remove terminal commands/control characters and flatten all newlines."""
    text = _ANSI.sub('', str(value))
    text = ''.join(' ' if character.isspace() else character
        for character in text if not unicodedata.category(character).startswith('C') or character.isspace())
    return ' '.join(text.split())


def display_width(text):
    """Conservative cell width, including Korean and combining characters."""
    return sum(0 if unicodedata.combining(character) or unicodedata.category(character) in ('Mn', 'Me')
        else 2 if unicodedata.east_asian_width(character) in ('W', 'F') else 1 for character in text)


def truncate(text, width):
    if type(width) is not int or width <= 0:
        raise ValueError('Terminal width must be a positive integer')
    if display_width(text) <= width:
        return text
    remaining = width - 1
    kept = []
    for character in text:
        cells = display_width(character)
        if cells > remaining:
            break
        kept.append(character); remaining -= cells
    return ''.join(kept) + '…'


def _text(value):
    return '-' if value is None or not sanitize(value) else sanitize(value)


def _section(model, name):
    value = model.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('Normalized monitor section must be an object: ' + name)
    return value


def _number(value, digits=4, *, count=False):
    if value is None:
        return '-'
    if isinstance(value, bool) or not isinstance(value, Real):
        return '미기록(형식 확인)'
    if not math.isfinite(value):
        return '미기록(비유한 값)'
    if count:
        if value < 0 or int(value) != value:
            return '미기록(계수 확인)'
        return str(int(value))
    return format(value, '.' + str(digits) + 'f')


def _count(value):
    return _number(value, count=True)


def _fraction(done, total, unit=None):
    label = _count(done) + '/' + _count(total)
    if unit is not None and sanitize(unit):
        label += ' ' + sanitize(unit)
    valid = (not isinstance(done, bool) and not isinstance(total, bool)
        and isinstance(done, Real) and isinstance(total, Real)
        and math.isfinite(done) and math.isfinite(total) and done >= 0 and total > 0
        and int(done) == done and int(total) == total)
    if valid and done <= total:
        label += ' (' + format(100 * done / total, '.1f') + '%)'
    elif valid:
        label += ' (계수 확인 필요)'
    return label


def _validation(value):
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError('Normalized validation record must be an object')
    return ('최근 완료 e' + _count(value.get('epoch')) + ' U' + _count(value.get('active_u'))
        + ' MRR ' + _number(value.get('mrr')) + ' top1 ' + _number(value.get('top1')))


def _dice(value):
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_number(item) for item in value) + ']' if value else '-'
    return _number(value)


def render_lines(model, width=None):
    """Always return the same sixteen rows from normalized, measured metadata."""
    if not isinstance(model, Mapping):
        raise ValueError('Normalized monitor model must be an object')
    state, gnn, cp, native, resource = (_section(model, name)
        for name in ('state', 'gnn', 'cp', 'native', 'resources'))
    fresh = state.get('fresh')
    reading = '읽기 성공' if fresh is True else '새 기록이 아님' if fresh is False else '읽기 상태 미기록'
    status = reading + ' | ' + _text(state.get('status')) + ' | 단계 ' + _text(state.get('stage'))
    if fresh is False:
        status += ' | 마지막 확인 ' + _text(state.get('last_confirmed_status'))
    progress = gnn.get('progress')
    if progress is None:
        progress = {}
    if not isinstance(progress, Mapping):
        raise ValueError('Normalized progress record must be an object')
    training = ('전체 완료 확인' if native.get('full_training') is True else
        '학습/BEST 평가 작업 실행 중' if native.get('started') is True else
        '현재 실행 미확인' if native.get('started') is False else '실행 상태 미기록')
    evaluation = '전체 완료 확인' if native.get('full_evaluation') is True else '전체 완료 미확인' if native.get('full_evaluation') is False else '미기록'
    warning = _text(model.get('warning'))
    if fresh is False and '새 기록이 아님' not in warning:
        warning = '새 기록이 아님' + (' | ' + warning if warning != '-' else '')
    source = resource.get('source')
    source = _SOURCE_NAMES.get(source, source) if isinstance(source, str) else source
    gnn_complete = '전체 완료 확인' if gnn.get('full_completed') is True else '전체 완료 미확인' if gnn.get('full_completed') is False else '완료 상태 미기록'
    values = [
        'GPU' + _count(model.get('gpu')) + ' | 버전 ' + _text(model.get('experiment_version')) + ' | UI ' + str(UI_VERSION),
        status,
        'epoch ' + _count(gnn.get('epoch')) + '/' + _count(gnn.get('epochs'))
            + ' | 완료 ' + _count(gnn.get('completed_epochs')) + ' | U ' + _count(gnn.get('active_u'))
            + ' | update ' + _count(gnn.get('optimizer_updates')) + ' | ' + gnn_complete,
        _text(gnn.get('phase_label')) + ' | ' + _fraction(progress.get('done'), progress.get('total'), progress.get('unit')),
        _number(gnn.get('loss'), 6) + ' | 측정 update ' + _count(gnn.get('loss_update')),
        _validation(gnn.get('stage_validation')),
        _validation(gnn.get('full_validation')),
        _text(cp.get('status')) + ' | ' + _fraction(cp.get('source_done'), cp.get('source_total')),
        _fraction(cp.get('positions_done'), cp.get('positions_total')) + ' | 입력 기준 ' + _text(cp.get('input_basis')),
        'epoch ' + _count(native.get('epoch')) + '/' + _count(native.get('epochs'))
            + ' | 완료 ' + _count(native.get('completed_epochs')) + ' | ' + training + ' | 평가 ' + evaluation,
        'train ' + _number(native.get('train_loss'), 6) + ' | val ' + _number(native.get('val_loss'), 6)
            + ' | 관측 epoch ' + _count(native.get('epoch')),
        _dice(native.get('pseudo_dice')) + ' | 관측 epoch ' + _count(native.get('epoch')),
        'RSS ' + _number(resource.get('rss_gib'), 2) + ' GiB / 한도 ' + _number(resource.get('ram_limit_gib'), 2)
            + ' GiB | 출처 ' + _text(source) + ' | ' + _text(resource.get('process_status')),
        '최근 batch ' + _number(cp.get('last_batch_seconds'), 2) + ' s | console 경과 ' + _number(cp.get('log_age_seconds'), 1) + ' s',
        '상태 ' + _text(native.get('status')) + ' | 단계 ' + _text(native.get('stage')) + ' | 사유 ' + _text(model.get('waiting_reason')),
        '오류 ' + _text(model.get('error')) + ' | 경고 ' + warning,
    ]
    lines = [label + ': ' + sanitize(value) for label, value in zip(ROW_LABELS, values)]
    return lines if width is None else [truncate(line, width) for line in lines]


def render(model, width=None):
    return '\n'.join(render_lines(model, width=width))


def _progress_bar(done, total, unit=None):
    """The bar and percentage use confirmed integer counts, never estimates."""
    valid = (not isinstance(done, bool) and not isinstance(total, bool)
        and isinstance(done, Real) and isinstance(total, Real)
        and math.isfinite(done) and math.isfinite(total)
        and 0 <= done <= total and total > 0
        and int(done) == done and int(total) == total)
    if valid:
        filled = int(20 * done / total)
        bar = '#' * filled + '-' * (20 - filled)
    else:
        bar = '?' * 20
    return _fraction(done, total, unit) + ' [' + bar + ']'


def _task_header(model, kind):
    state = _section(model, 'state')
    fresh = state.get('fresh')
    reading = '읽기 성공' if fresh is True else '새 기록이 아님' if fresh is False else '읽기 상태 미기록'
    value = ('GPU' + _count(model.get('gpu')) + ' | ' + kind + ' | '
        + _text(state.get('status')) + ' | ' + reading
        + ' | 버전 ' + _text(model.get('experiment_version')) + ' | UI ' + str(TASK_UI_VERSION))
    if fresh is False:
        value += ' | 마지막 확인 ' + _text(state.get('last_confirmed_status'))
    return value


def _typed_lines(labels, values, width):
    lines = [label + ': ' + sanitize(value) for label, value in zip(labels, values)]
    return lines if width is None else [truncate(line, width) for line in lines]


def _row_value(lines, index):
    return lines[index].partition(': ')[2]


def render_gnn_lines(model, width=None):
    """Nine fixed GNN rows, identical across experiment versions."""
    base = render_lines(model)
    gnn = _section(model, 'gnn')
    progress = gnn.get('progress') or {}
    values = [_task_header(model, 'GNN 학습'), _row_value(base, 2),
        _text(gnn.get('phase_label')) + ' | ' + _progress_bar(
            progress.get('done'), progress.get('total'), progress.get('unit')),
        *(_row_value(base, index) for index in (4, 5, 6, 12, 14, 15))]
    return _typed_lines(GNN_ROW_LABELS, values, width)


def render_cp_lines(model, width=None):
    """Eight fixed CP scoring rows; only actual downstream controller metadata."""
    base = render_lines(model)
    cp = _section(model, 'cp')
    values = [_task_header(model, 'CP 후보 채점'),
        _progress_bar(cp.get('source_done'), cp.get('source_total'))
            + ' | ' + _text(cp.get('status')),
        _progress_bar(cp.get('positions_done'), cp.get('positions_total')),
        _text(cp.get('input_basis')),
        *(_row_value(base, index) for index in (12, 13, 14, 15))]
    return _typed_lines(CP_ROW_LABELS, values, width)


def render_task_lines(model, width=None):
    """Choose the typed GNN/CP layout; nnU-Net keeps its own console formatter."""
    if not isinstance(model, Mapping):
        raise ValueError('Normalized monitor model must be an object')
    stage = _section(model, 'state').get('stage')
    if isinstance(stage, str):
        stage = sanitize(stage).lower()
    if stage == 'gnn':
        return render_gnn_lines(model, width=width)
    if stage in ('scoring', 'cp', 'cp 후보 채점'):
        return render_cp_lines(model, width=width)
    raise ValueError('Task monitor supports only GNN/CP stages; keep the nnU-Net console for native: ' + _text(stage))


def render_task(model, width=None):
    return '\n'.join(render_task_lines(model, width=width))


class TerminalView:
    """Redraw only this viewer's rows; close restores its cursor, not the stream.

    size_provider, when given, returns (columns, lines). Small terminals and
    non-TTY streams print complete plain-text frames without cursor animation.
    No tqdm/environment toggle can change the schema or hide its state.
    """
    def __init__(self, stream, size_provider=None, layout='full'):
        if layout not in ('full', 'task'):
            raise ValueError('Training monitor layout must be full or task')
        self.stream = stream
        self.size_provider = size_provider
        self.layout = layout
        self._drawn = False
        self._drawn_rows = 0
        self._cursor_hidden = False
        self._closed = False
        self._last_dimensions = None

    def _size(self):
        if self.size_provider is not None:
            dimensions = self.size_provider()
        else:
            try:
                dimensions = os.get_terminal_size(self.stream.fileno())
            except (AttributeError, OSError, ValueError):
                return None
        if (not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2
                or any(type(value) is not int or value <= 0 for value in dimensions)):
            return None
        return tuple(dimensions)

    def render(self, model):
        if self._closed:
            raise RuntimeError('Training monitor view is already closed')
        tty = bool(getattr(self.stream, 'isatty', lambda: False)())
        dimensions = self._size() if tty else None
        renderer = render_task_lines if self.layout == 'task' else render_lines
        lines = renderer(model)
        animated = dimensions is not None and dimensions[0] >= 80 and dimensions[1] >= len(lines) + 1
        if animated:
            lines = [truncate(line, dimensions[0] - 1) for line in lines]
        if dimensions != self._last_dimensions:
            self._drawn = False  # Resizing can reflow old rows; start a fresh region.
        if animated:
            if not self._cursor_hidden:
                self.stream.write('\x1b[?25l'); self._cursor_hidden = True
            if self._drawn:
                self.stream.write('\x1b[' + str(self._drawn_rows) + 'A\r')
            for line in lines:
                self.stream.write('\r\x1b[2K' + line + '\n')
            if self._drawn and self._drawn_rows > len(lines):
                surplus = self._drawn_rows - len(lines)
                for _ in range(surplus):
                    self.stream.write('\r\x1b[2K\n')
                self.stream.write('\x1b[' + str(surplus) + 'A\r')
            self._drawn = True
            self._drawn_rows = len(lines)
        else:
            if self._cursor_hidden:
                self.stream.write('\x1b[?25h'); self._cursor_hidden = False
            self._drawn = False
            self._drawn_rows = 0
            self.stream.write('\n'.join(lines) + '\n')
        self._last_dimensions = dimensions
        self.stream.flush()
        return lines

    def close(self):
        if not self._closed and self._cursor_hidden:
            self.stream.write('\x1b[?25h'); self.stream.flush()
        self._cursor_hidden = False
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
