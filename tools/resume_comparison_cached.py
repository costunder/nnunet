"""Resume a sealed v1.8/v1.9 arm; reuse completed exact caches from other arms."""
from __future__ import annotations
import argparse
import builtins
from contextlib import ExitStack, contextmanager
import importlib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class _CacheEventLogger:
    """Keep complete JSONL evidence without interrupting live phase progress."""
    def __init__(self, path, *, clock=time.monotonic, console=None):
        self.path = Path(path)
        self._clock = clock
        self._console = console or self._progress_safe_write
        self._lock = threading.Lock()
        self._last_summary = clock()
        self._reported_events = 0
        self._finished = False
        self._events = self._reused = self._misses = self._binding_misses = 0
        self._reused_bytes = self._pressure_events = 0
        self._field_reads = 0
        self._field_seconds = 0.

    @staticmethod
    def _progress_safe_write(line):
        from tqdm import tqdm
        tqdm.write(line, file=sys.stderr)

    @staticmethod
    def _gib(value):
        return f"{value / 2**30:.2f}" if type(value) in (int, float) else "unknown"

    def _summary(self, label):
        self._console(f"Cache log summary | {label} | events={self._events} "
            f"reused={self._reused} miss={self._misses} binding_miss={self._binding_misses} "
            f"reused_GiB={self._gib(self._reused_bytes)} pressure={self._pressure_events} "
            f"field_reads={self._field_reads} field_seconds={self._field_seconds:.2f} "
            f"| full details: {self.path}")
        self._reported_events = self._events
        self._last_summary = self._clock()

    def __call__(self, row):
        with self._lock:
            # Preserve the previous complete append/write/close durability and
            # exact record contents; console verbosity changes storage neither.
            with self.path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            self._events += 1
            status, stage = row.get('status'), row.get('stage')
            if stage == 'whole_case_fields_read':
                self._field_reads += 1
                if type(row.get('wall_seconds')) in (int, float):
                    self._field_seconds += row['wall_seconds']
            if status == 'reused':
                self._reused += 1
                if type(row.get('bytes')) is int:
                    self._reused_bytes += row['bytes']
            elif status == 'cache_miss':
                self._misses += 1
            elif status == 'binding_miss':
                self._binding_misses += 1
            if stage == 'host_cache_pressure':
                self._pressure_events += 1
                self._console(f"Host cache pressure | RSS_GiB="
                    f"{self._gib(row.get('rss_before_bytes'))}->{self._gib(row.get('rss_after_bytes'))} "
                    f"limit={self._gib(row.get('rss_limit_bytes'))} "
                    f"within_limit={row.get('measured_rss_within_hard_budget')} | details: {self.path}")
            elif stage == 'hard_budget_failed':
                self._console(f"Resource budget failed | resource={row.get('resource')} "
                    f"actual_GiB={self._gib(row.get('actual_bytes'))} "
                    f"limit_GiB={self._gib(row.get('limit_bytes'))} | details: {self.path}")
            elif status in ('failed', 'error', 'rejected') or row.get('error'):
                self._console(f"Cache execution error | kind={row.get('kind')} status={status} "
                    f"stage={stage} | full error: {self.path}")
            # The phase display owns live progress. A cache event is not a
            # phase transition; printing periodic 'preparation' lines here
            # obscured validation and split terminal progress bars.

    def finish(self):
        with self._lock:
            if self._finished:
                return
            self._finished = True
            if self._events == self._reported_events:
                return
            active = sys.exc_info()[1]
            try:
                self._summary('end')
            except OSError as error:
                if active is None:
                    raise
                # A nonessential final console line must not replace the
                # original training/cache failure. Keep the output failure
                # attached to that exception on Python3.10 as well as3.11+.
                detail = f"Final cache-summary console output failed: {error}"
                if hasattr(active, 'add_note'):
                    active.add_note(detail)
                else:
                    active.args = (*active.args, detail)


@contextmanager
def _field_read_console(event):
    """Capture the preserved provider's exact field-read report in JSONL.

    Only this module's print name is overridden. Other modules, builtins.print,
    arbitrary messages and custom print destinations/formatting are untouched.
    """
    from hiercp_v1x import u_bridge_data
    namespace = vars(u_bridge_data)
    had_print = 'print' in namespace
    original = namespace.get('print', builtins.print)
    prefix = 'v1.8 whole-case fields case='
    pattern = re.compile(r'^v1\.8 whole-case fields case=([A-Za-z0-9][A-Za-z0-9_.-]*) '
        r'status=(built|reopened|resident_mapping) array_bytes=(\d+) disk_bytes=(\d+) '
        r'wall_seconds=(\d+\.\d{3})$')
    def report(*args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], str) and args[0].startswith(prefix)
                and '\n' not in args[0] and '\r' not in args[0]
                and kwargs == {'flush': True}):
            row = dict(format='preserved_v18_field_read_console_v1', stage='whole_case_fields_read',
                       source_line=args[0])
            match = pattern.fullmatch(args[0])
            if match is not None:
                try:
                    array_bytes, disk_bytes = int(match[3]), int(match[4])
                    seconds = float(match[5])
                except (ValueError, OverflowError):
                    # Preserve every original character even when a printed
                    # numeric token is too large to trust as parsed telemetry.
                    match = None
                else:
                    if math.isfinite(seconds):
                        row.update(case_id=match[1], status=match[2], array_bytes=array_bytes,
                                   disk_bytes=disk_bytes, wall_seconds=seconds)
            event(row)
            return None
        return original(*args, **kwargs)
    namespace['print'] = report
    try:
        yield
    finally:
        if had_print:
            namespace['print'] = original
        else:
            del namespace['print']


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--arm', choices=('selected', 'native', 'native_fixed', 'native_listwise'), required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--cache-sources', type=Path, nargs='+', required=True,
                   help='Other arm data directories; read completed immutable publications only')
    p.add_argument('--inventory', type=Path, required=True)
    for name in ('fixture', 'config', 'source', 'bank'):
        p.add_argument('--debug-' + name, type=Path)
    return p.parse_args(argv)


def sealed_arguments(a):
    from hiercp_v1x.u_bridge_experiment import digest, read, sha
    root = a.experiment.resolve(strict=True)
    manifest = read(root / 'experiment.json')
    if manifest.get('sha256') != digest({k:v for k,v in manifest.items() if k != 'sha256'}):
        raise ValueError('Saved experiment contract changed')
    formats = {'v18_u_bridge_matched_experiment_v1': 'u_bridge',
               'v19_matched_comparison_controls_v1': 'comparison'}
    family = formats.get(manifest.get('format'))
    if family is None or (family == 'u_bridge' and a.arm not in ('selected', 'native')):
        raise ValueError('Arm does not belong to the preserved v1.8/v1.9 experiment')
    module = importlib.import_module('hiercp_v1x.' + family + '_experiment')
    if a.arm not in getattr(module, 'ARMS', ('selected', 'native')):
        raise ValueError('Unknown preserved experiment arm')
    if set(manifest['helpers']) != set(module.FILES):
        raise ValueError('Preserved helper inventory changed')
    changed = [name for name, checksum in manifest['helpers'].items() if sha(ROOT/name) != checksum]
    if changed:
        raise ValueError(f'Preserved input/model/training source changed: {changed}')
    if sha(a.inventory) != manifest['baseline']['inventory_sha256']:
        raise ValueError('Original128 candidate inventory changed')
    options = [getattr(a, 'debug_' + name) for name in ('fixture', 'config', 'source', 'bank')]
    if manifest['debug'] and any(x is None for x in options):
        raise ValueError('Explicit original actual-CT DEBUG inputs required')
    if not manifest['debug'] and any(x is not None for x in options):
        raise ValueError('DEBUG inputs cannot enter production')
    if not manifest['debug'] and manifest['epochs'] != 40:
        raise ValueError('Original40epoch production contract required')
    namespace = Path(manifest.get('prepared_data_root', root/'data')).resolve()
    args = SimpleNamespace(gpu=a.gpu, arm=a.arm, experiment=root, inventory=a.inventory,
        baseline=None if manifest['debug'] else Path(manifest['baseline']['baseline']),
        prepared_data=None if namespace == root/'data' else namespace,
        workers=manifest['workers'], batch_candidates=manifest['explicit_batch_candidates'],
        cuda_gib=manifest['cuda_gib'], rss_gib=manifest['rss_gib'], resident_gib=manifest['resident_gib'],
        validation_local_chunk=manifest['validation_local_chunk'], debug=manifest['debug'],
        debug_epochs=manifest['epochs'] if manifest['debug'] else None, debug_pause_updates=None,
        **{'debug_'+name:getattr(a, 'debug_'+name) for name in ('fixture','config','source','bank')})
    return manifest, family, args, namespace


def run(a):
    from tools.local_cnn_device import select
    select(a.gpu)
    from hiercp_v1x import host_memory
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x.comparison_execution import comparison_execution
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.u_bridge_experiment import lock, write_new, sha
    manifest, family, args, namespace = sealed_arguments(a)
    root = args.experiment
    experiment_module = importlib.import_module('hiercp_v1x.'+family+'_experiment')
    data_module = importlib.import_module('hiercp_v1x.'+family+'_data')
    class_name = 'UBridgeData' if family == 'u_bridge' else 'ComparisonData'
    controller = importlib.import_module('tools.run_v18_u_bridge' if family == 'u_bridge' else 'tools.run_v19_comparison')
    original_budget = experiment_module.Budget
    original_provider = getattr(data_module, class_name)
    events_path = root/'preparation_reuse.jsonl'
    event = _CacheEventLogger(events_path)
    def budget(cuda_bytes, rss_bytes):
        return PressureBudget(cuda_bytes, rss_bytes, resident_bytes=int(manifest['resident_gib']*2**30),
                              event_callback=event)
    def receipt(data_root):
        write_new(root/'execution_overrides'/('cache_'+uuid.uuid4().hex+'.json'),
            dict(format='comparison_completed_cache_reuse_execution_v1',
                contract_sha256=manifest['sha256'], arm=a.arm, own_data_namespace=str(data_root),
                read_only_sources=[str(p.resolve()) for p in a.cache_sources],
                original_checkpoint_protocol_and_frozen_sources_unchanged=True,
                execution_loop_changed=True,
                execution_policy='ordered memory-admitted input staging, original epoch views, completed hierarchy layout and existing local graph reuse, compact source/upper caches',
                helpers={name:sha(ROOT/name) for name in ('tools/resume_comparison_cached.py',
                    'hiercp_v1x/host_memory.py','hiercp_v1x/preparation_reuse.py',
                    'hiercp_v1x/comparison_execution.py','hiercp_v1x/comparison_runtime.py',
                    'hiercp_v1x/comparison_progress.py','hiercp_v1x/comparison_inputs.py',
                    'hiercp_v1x/comparison_views.py','hiercp_v1x/comparison_data_timing.py',
                    'hiercp_v1x/comparison_preparation.py','hiercp_v1x/comparison_source_cache.py',
                    'hiercp_v1x/comparison_upper_cache.py','hiercp_v1x/comparison_sample_cache.py')}))
    def execution_provider(provider):
        return prepared_provider(provider, root/'input_timing.jsonl')
    continuation_path = root/'continuation.json'
    if continuation_path.exists():
        if family != 'u_bridge':
            raise ValueError('Independent continuation belongs to v1.8 only')
        from hiercp_v1x.u_bridge_experiment import read
        from tools import run_v18_independent
        saved = read(continuation_path)
        request = saved['request']
        if request['arm'] != a.arm or request['contract_sha256'] != manifest['sha256']:
            raise ValueError('Independent continuation arm/contract differs')
        independent = SimpleNamespace(gpu=a.gpu, arm=a.arm,
            source_experiment=Path(request['source_root']), experiment=root,
            inventory=a.inventory, debug_bank=a.debug_bank)
        with preparation_reuse(pressure_aware_provider(original_provider), a.cache_sources, event) as provider:
            provider = execution_provider(provider)
            def registered(requested):
                if requested is not original_provider:
                    raise ValueError('Independent provider class differs')
                return provider
            host_memory.pressure_aware_provider = registered
            try:
                # Preserve the requested helper policy even if a busy lock or
                # preparation error prevents the owned runner from completing.
                receipt(Path(request['data_root']))
                # The preserved independent runner owns its locks, restores its
                # exact checkpoint cursor, and verifies the continuation receipt.
                with _field_read_console(event), comparison_execution():
                    result = run_v18_independent.run(independent)
                return result
            finally:
                host_memory.pressure_aware_provider = pressure_aware_provider
                event.finish()
    with ExitStack() as locks:
        locks.callback(event.finish)
        locks.enter_context(lock(root/'.pipeline.lock'))
        locks.enter_context(lock(namespace/'.data.lock'))
        with preparation_reuse(pressure_aware_provider(original_provider), a.cache_sources, event) as provider:
            provider = execution_provider(provider)
            experiment_module.Budget = budget
            setattr(data_module, class_name, provider)
            try:
                receipt(namespace)
                with _field_read_console(event), comparison_execution():
                    return controller.run(args)
            finally:
                experiment_module.Budget = original_budget
                setattr(data_module, class_name, original_provider)


if __name__ == '__main__':
    run(parse())
