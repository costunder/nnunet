"""Resume a sealed v1.8/v1.9 arm; reuse completed exact caches from other arms."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


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
    from hiercp_v1x.u_bridge_experiment import lock, write_new, sha
    manifest, family, args, namespace = sealed_arguments(a)
    root = args.experiment
    experiment_module = importlib.import_module('hiercp_v1x.'+family+'_experiment')
    data_module = importlib.import_module('hiercp_v1x.'+family+'_data')
    class_name = 'UBridgeData' if family == 'u_bridge' else 'ComparisonData'
    controller = importlib.import_module('tools.run_v18_u_bridge' if family == 'u_bridge' else 'tools.run_v19_comparison')
    original_budget = experiment_module.Budget
    original_provider = getattr(data_module, class_name)
    writer_lock = threading.Lock()
    events_path = root/'preparation_reuse.jsonl'
    def event(row):
        with writer_lock:
            with events_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print('Cache execution | '+json.dumps(row, allow_nan=False), flush=True)
    def budget(cuda_bytes, rss_bytes):
        return PressureBudget(cuda_bytes, rss_bytes, resident_bytes=int(manifest['resident_gib']*2**30),
                              event_callback=event)
    def receipt(data_root):
        write_new(root/'execution_overrides'/('cache_'+uuid.uuid4().hex+'.json'),
            dict(format='comparison_completed_cache_reuse_execution_v1',
                contract_sha256=manifest['sha256'], arm=a.arm, own_data_namespace=str(data_root),
                read_only_sources=[str(p.resolve()) for p in a.cache_sources],
                original_checkpoint_and_engine_unchanged=True,
                helpers={name:sha(ROOT/name) for name in ('tools/resume_comparison_cached.py',
                    'hiercp_v1x/host_memory.py','hiercp_v1x/preparation_reuse.py')}))
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
                result = run_v18_independent.run(independent)
                return result
            finally:
                host_memory.pressure_aware_provider = pressure_aware_provider
    with ExitStack() as locks:
        locks.enter_context(lock(root/'.pipeline.lock'))
        locks.enter_context(lock(namespace/'.data.lock'))
        with preparation_reuse(pressure_aware_provider(original_provider), a.cache_sources, event) as provider:
            experiment_module.Budget = budget
            setattr(data_module, class_name, provider)
            try:
                receipt(namespace)
                return controller.run(args)
            finally:
                experiment_module.Budget = original_budget
                setattr(data_module, class_name, original_provider)


if __name__ == '__main__':
    run(parse())
