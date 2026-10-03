"""Resume failed current v1 preparation in a fresh, explicitly bound suite.

Run only inside its verified frozen snapshot. Original geometry, candidate
generation and completion rules remain owned by the archived pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest(path):
    value = read(path)
    body = {key: item for key, item in value.items() if key != 'manifest_sha256'}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if actual != value.get('manifest_sha256'):
        raise ValueError('Recovery suite manifest changed')
    return value


def verify_binding(experiment):
    experiment = Path(experiment).resolve(strict=True)
    current = manifest(experiment/'manifest.json')
    binding = current.get('preparation_recovery')
    if not isinstance(binding, dict) or current.get('preparation_admission') is None:
        raise ValueError('Explicit resource admission and failed-cache recovery are required')
    source = Path(binding['path']).resolve(strict=True)
    if source == experiment or source in experiment.parents or experiment in source.parents:
        raise ValueError('Recovery experiment must be disjoint from its source')
    previous = manifest(source/'manifest.json')
    if (previous['manifest_sha256'] != binding['manifest_sha256']
            or previous['split'] != current['split']
            or previous['medical_root'] != current['medical_root']
            or previous.get('sampling_contract', {}).get('mode') != 'native'
            or current.get('sampling_contract', {}).get('mode') != 'native'):
        raise ValueError('Recovery source changed its native data/split binding')
    for name, key in (('config.json', 'cache_config_sha256'), ('manifest.csv', 'progress_sha256')):
        if digest(source/'shared/cache'/name) != binding[key]:
            raise ValueError('Failed cache metadata changed')
    if (source/'shared/prepare.lock').exists() or any((source/'results').glob('*/run.lock')):
        raise ValueError('Failed source is active; no concurrent source mutation permitted')
    snapshot = experiment/'source/v1.0'
    for name, expected in current['stages']['v1.0']['source_hashes'].items():
        if digest(snapshot/name) != expected:
            raise ValueError(f'Recovery snapshot changed: {name}')
    for name in ('hiercp/pipeline.py', 'hiercp/cache.py', 'hiercp/spatial.py'):
        if (snapshot/name).resolve() != Path(sys.modules['hiercp'].__file__).parent.parent.joinpath(name).resolve():
            raise ValueError('Recovery imported a different hiercp source tree')
    return source, current


def copy_prepared_assets(source, destination, *, workers):
    """Copy verified immutable bytes; never rebuild old regions/prototypes."""
    from hiercp.preparation_runtime import run_case_jobs
    origin, target = Path(source)/'shared', Path(destination)/'shared'
    files = [origin/'prototype_bank.pt', *sorted((origin/'regions').rglob('*'))]
    if any(path.is_symlink() for path in files):
        raise ValueError('Recovery refuses symlinked source assets')
    files = [path for path in files if path.is_file()]
    if not files or not (origin/'prototype_bank.pt').is_file():
        raise ValueError('Prepared source prototype/regions are missing')
    required = sum(path.stat().st_size for path in files)
    target.mkdir(parents=True, exist_ok=True)
    cache_bytes = sum(path.stat().st_size for path in (origin/'cache').glob('*.pt'))
    if shutil.disk_usage(target).free <= required + cache_bytes:
        raise OSError('Insufficient disk for non-destructive region/prototype and cache copies')

    def copy(path):
        expected = digest(path)
        output = target/path.relative_to(origin)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.is_symlink() or digest(output) != expected:
                raise ValueError(f'Existing recovery asset differs: {output}')
        else:
            with path.open('rb') as incoming, output.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
        if digest(path) != expected or digest(output) != expected:
            raise ValueError(f'Recovery asset changed during copy: {path}')
        return str(path.relative_to(origin))

    rows = []
    run_case_jobs(tasks=files, function=copy, commit=rows.append, workers=workers,
                  report_path=target/'roi_recovery_assets_resources.json')
    return rows


def install_recovery(pipeline, source, destination):
    from hiercp_v1x.cache_budget_recovery import (
        migrate_failed_current_cache, validate_failed_current_cache,
    )
    original = pipeline.prepare_hierarchical_cache

    def prepare(**kwargs):
        target = Path(kwargs['cache_dir']).resolve()
        if target != (Path(destination)/'shared/cache').resolve():
            raise ValueError('Recovery target differs from the explicit experiment')
        request = dict(source_cache_dir=Path(source)/'shared/cache',
                       destination_cache_dir=target, prepare_kwargs=kwargs)
        if target.exists():
            validate_failed_current_cache(**request)
        else:
            certificate = migrate_failed_current_cache(**request)
            print(f"Verified cache recovery | copied={certificate['copied_count']} "
                  '| geometry/tensors preserved | original cache unchanged', flush=True)
        result = original(**kwargs)
        validate_failed_current_cache(**request)
        return result

    pipeline.prepare_hierarchical_cache = prepare
    return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--experiment', required=True)
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    if not remaining or remaining[0] != 'prepare':
        parser.error('Recovery entry accepts preparation only; it never starts training')
    from hiercp import pipeline
    source, current = verify_binding(args.experiment)
    config = read(Path(args.experiment)/'configs/v1.0.json')
    print('ROI resource recovery | fresh destination | no model/sampler/loss change | no training', flush=True)
    copy_prepared_assets(source, args.experiment, workers=config['runtime']['prepare_workers'])
    install_recovery(pipeline, source, args.experiment)
    previous = sys.argv
    try:
        sys.argv = ['hiercp.pipeline', *remaining]
        pipeline.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
