"""Train one physically bounded v1 arm on the original full84/21 cohort.

Existing native/30mm results are references, never automatically rerun. Original
model, curriculum8/pool128, two views, GT, L1/L2 and40 epochs remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hiercp_v1x.contracts import canonical_hash, verify_archive
from hiercp_v1x.experiment import digest, read, write_new, load_suite, preparation_root, owned_lock

FORMAT = 'v1_bounded_roi_full_learning_v1'
HELPERS = ('hiercp_v1x/bounded_scope.py', 'hiercp_v1x/scope_training_entry.py',
    'hiercp_v1x/scope_learning_inputs.py', 'hiercp_v1x/scope_probe_support.py',
    'hiercp_v1x/epoch_telemetry.py', 'hiercp_v1x/snapshot_inventory.py',
    'hiercp_v1x/sampling_entry.py', 'tools/local_cnn_device.py',
    'hiercp_v1x/contracts.py', 'hiercp_v1x/experiment.py',
    'tools/run_v1_bounded_training.py')


def bounded_config(native, margin):
    if margin not in (10., 20., 30.): raise ValueError('One explicit10/20/30mm margin required')
    if (native['seed'] != 42 or native['training']['epochs'] != 40
            or native['cache']['total_candidates'] != 8 or native['cache']['candidate_pool_size'] != 128):
        raise ValueError('Original seed42/40epochs/8candidate/128pool contract required')
    result = copy.deepcopy(native)
    result['graph']['adaptive_roi_margin_mm'] = float(margin)
    result['graph']['context_outer_radius_mm'] = float(margin)
    changed = [key for key in native['graph'] if native['graph'][key] != result['graph'][key]]
    if not set(changed) <= {'adaptive_roi_margin_mm', 'context_outer_radius_mm'} or not changed:
        raise ValueError('Full scope learning may change only the two physical extent fields')
    return result


def copy_verified(source, destination):
    source = Path(source); destination = Path(destination)
    if source.is_symlink() or not source.is_file(): raise ValueError('Regular immutable reference file required')
    source = source.resolve(strict=True)
    expected = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or digest(destination) != expected:
            raise ValueError(f'Existing owned file differs from reference: {destination}')
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024**2)
    if digest(destination) != expected or digest(source) != expected:
        raise ValueError('Reference copy or source changed')
    return expected


def verify_bound_experiment(root, manifest=None):
    """Check actual owned bytes and reference evidence before prepare or resume."""
    from hiercp_v1x.snapshot_inventory import checked_snapshot_inventory
    root = Path(root).resolve(strict=True)
    manifest = read(root / 'manifest.json') if manifest is None else manifest
    content = {key: value for key, value in manifest.items() if key != 'contract_sha256'}
    if (manifest.get('format') != FORMAT or manifest.get('experiment') != str(root)
            or canonical_hash(content) != manifest.get('contract_sha256')):
        raise ValueError('Bounded experiment manifest identity/hash changed')
    for name, expected in manifest['helpers'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Bounded execution source changed: {name}')
    for name, expected in manifest['reference_files'].items():
        if Path(name).is_symlink() or digest(Path(name)) != expected:
            raise ValueError(f'Preserved reference changed: {name}')
    old = load_suite(manifest['source_experiment'])
    if (old['manifest_sha256'] != manifest['reference_manifest_sha256']
            or old['split'] != manifest['split']):
        raise ValueError('Preserved native experiment/cohort changed')
    if read(root / 'config.json') != manifest['config']:
        raise ValueError('Actual bounded config differs from its fixed contract')
    shared = preparation_root(Path(manifest['source_experiment']), old)
    for name in ('split.json', 'prototype_bank.pt', 'metadata.json', 'manifest.csv'):
        owned = root / 'shared' / name
        if owned.is_symlink() or digest(owned) != digest(shared / name):
            raise ValueError(f'Copied native population/cohort file changed: {name}')
    regions = root / 'shared/regions'
    if regions.is_symlink(): raise ValueError('Copied region directory cannot be a symlink')
    actual = {}
    for path in regions.rglob('*'):
        if path.is_symlink(): raise ValueError('Copied region cache has a symlink')
        if path.is_file(): actual[path.relative_to(regions).as_posix()] = digest(path)
    if actual != manifest['region_files']:
        raise ValueError('Copied native region inventory/bytes changed')
    source = root / 'source/v1.0'
    expected_source = read(ROOT / 'versions/v1/manifest.json')['files']
    checked_snapshot_inventory(source, expected_source)
    if any(digest(source / name) != expected for name, expected in expected_source.items()):
        raise ValueError('Actual preserved native source bytes changed')
    return manifest


def initialize(source_experiment, experiment, margin, gpu, cuda_gib, rss_gib):
    source_root = Path(source_experiment).resolve(strict=True)
    root = Path(experiment).resolve()
    if root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError('Bounded experiment must be disjoint from all native reference results')
    old = load_suite(source_root)
    if old.get('sampling_contract', {}).get('mode') != 'native':
        raise ValueError('Existing original native experiment required, no strict-nested input')
    if (len(old['split']['train']) != 84 or len(old['split']['val']) != 21
            or len(old['split']['outer_validation_excluded']) != 26):
        raise ValueError('Full frozen84/21 plus untouchedouter26 split required; no subset')
    shared = preparation_root(source_root, old)
    native_config = source_root / 'configs/v1.0.json'
    config = bounded_config(read(native_config), margin)
    proof = verify_archive(ROOT)
    helper_hashes = {name: digest(ROOT / name) for name in HELPERS}
    reference_files = [native_config, shared / 'split.json', shared / 'prototype_bank.pt',
        shared / 'metadata.json', shared / 'manifest.csv', shared / 'cache/config.json', shared / 'cache/manifest.csv']
    identities = {str(path): digest(path) for path in reference_files}
    regions = shared / 'regions'
    if not regions.is_dir() or regions.is_symlink(): raise ValueError('Native region cache directory required')
    region_files = {}
    for path in regions.rglob('*'):
        if path.is_symlink(): raise ValueError('Native region cache symlinks are not copied')
        if path.is_file(): region_files[path.relative_to(regions).as_posix()] = digest(path)
    if not region_files: raise ValueError('Full native region cache is absent')
    bank_meta = read(shared / 'metadata.json')
    if (bank_meta.get('state') != 'ready' or bank_meta.get('prototype_sha256') != digest(shared / 'prototype_bank.pt')
            or bank_meta.get('manifest_sha256') != digest(shared / 'manifest.csv')
            or set(bank_meta.get('training_cases', [])) != set(old['split']['train'])):
        raise ValueError('Native full training-only prototype publication must be intact')
    manifest = dict(format=FORMAT, source_experiment=str(source_root), experiment=str(root),
        reference_manifest_sha256=old['manifest_sha256'], margin_mm=float(margin),
        native_archive_sha256=proof['archive_sha256'], config=config, split=old['split'],
        reference_files=identities, region_files=region_files, helpers=helper_hashes, gpu=gpu,
        cuda_gib=float(cuda_gib), rss_gib=float(rss_gib),
        learning_target='original source anchor versus original curriculum candidates',
        original_model=True, candidate_pool=128, candidates_per_sample=8, views=2,
        original_curriculum=True, epochs=40, full_cohort=True, quality_verified=False,
        production_CP_started=False, nnunet_started=False, native_or_30mm_rerun=False,
        reference_cache_complete_required=False,
        unmatched_source_requests='Original failed/missing requests rebuilt with native curriculum and bounded L0; recorded individually, no false exact candidate-pair claim')
    manifest['contract_sha256'] = canonical_hash(manifest)
    if (root / 'manifest.json').exists():
        if read(root / 'manifest.json') != manifest:
            raise ValueError('Existing experiment configuration/source/resources changed; no silent resume')
        verify_bound_experiment(root, manifest)
        return root, manifest
    if root.exists(): raise FileExistsError('Unbound existing directory is preserved; choose a new experiment')
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source/v1.0'; source.mkdir(parents=True)
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError('Unsafe preserved archive member')
            target = source.joinpath(*path.parts); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(archive.read(info))
    owned = root / 'shared'; owned.mkdir()
    for name in ('split.json', 'prototype_bank.pt', 'metadata.json', 'manifest.csv'):
        copy_verified(shared / name, owned / name)
    for name, expected in region_files.items():
        if copy_verified(regions / name, owned / 'regions' / name) != expected:
            raise ValueError('Native region source changed during initialization')
    write_new(root / 'config.json', config)
    write_new(root / 'manifest.json', manifest)
    verify_bound_experiment(root, manifest)
    return root, manifest


def request_for(root, manifest, phase):
    if phase not in ('prepare', 'train'): raise ValueError('Explicit prepare/train phase required')
    root = Path(root).resolve(strict=True)
    verify_bound_experiment(root, manifest)
    source = Path(manifest['source_experiment'])
    old = load_suite(source); shared = preparation_root(source, old)
    return dict(source=str(root / 'source/v1.0'), config=str(root / 'config.json'),
        native_config=str(source / 'configs/v1.0.json'), experiment=str(root), phase=phase,
        source_experiment=str(source), native_cache=str(shared / 'cache'),
        prototype_bank=str(root / 'shared/prototype_bank.pt'), regions=str(root / 'shared/regions'),
        cache=str(root / 'shared/cache'), split=str(root / 'shared/split.json'), medical_root=old['medical_root'],
        margin_mm=manifest['margin_mm'], workers=1, cuda_gib=manifest['cuda_gib'], rss_gib=manifest['rss_gib'],
        gpu=manifest['gpu'], contract_sha256=manifest['contract_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--margin-mm', type=float, required=True)
    p.add_argument('--source-experiment', required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    p.add_argument('--plan-only', action='store_true', help='Print full learning contract without preparing/training')
    a = p.parse_args()
    if a.gpu < 0 or not all(math.isfinite(v) and v > 0 for v in (a.cuda_gib, a.rss_gib)):
        raise ValueError('Explicit positive finite resource budgets required')
    if a.plan_only:
        old = load_suite(a.source_experiment)
        print(json.dumps(dict(config=bounded_config(read(Path(a.source_experiment) / 'configs/v1.0.json'), a.margin_mm),
            train_cases=len(old['split']['train']), validation_cases=len(old['split']['val']),
            epochs=40, margin_mm=a.margin_mm, GPU=a.gpu, training_started=False, native_or_30mm_rerun=False), indent=2)); return
    root, manifest = initialize(a.source_experiment, a.experiment, a.margin_mm, a.gpu, a.cuda_gib, a.rss_gib)
    from hiercp_v1x.snapshot_inventory import isolated_snapshot_bytecode_env
    with owned_lock(root / 'run.lock'):
        print(f"Bounded v1 learning | margin={a.margin_mm:g}mm | full84train/21val | seed42 |40epochs", flush=True)
        print('Only this arm runs. Existing30mm/native results remain unchanged. Ctrl+C stops foreground work; training resumes from the last completed epoch.', flush=True)
        for phase in ('prepare', 'train'):
            invocation = root / 'invocations' / (phase + '_' + uuid.uuid4().hex)
            invocation.mkdir(parents=True)
            request = request_for(root, manifest, phase); path = invocation / 'request.json'; write_new(path, request)
            env = isolated_snapshot_bytecode_env(invocation / 'unused_bytecode_lookup')
            env.pop('PYTHONPATH', None); env.pop('HIERCP_V1X_SAMPLING_CONTRACT', None)
            command = [sys.executable, '-u', '-m', 'hiercp_v1x.scope_training_entry', '--request', str(path)]
            process = subprocess.Popen(command, cwd=ROOT, env=env)
            try: code = process.wait()
            except KeyboardInterrupt:
                print(f'Foreground interruption | directly launched PID {process.pid} | waiting for child; no shell/session termination', flush=True)
                process.wait(); raise
            if code: raise RuntimeError(f'{phase} failed({code}); original results and completed scoped work preserved; later phase not started')
    print(f"LEARNING RESULTS: {root / 'results/v1.0'}", flush=True)


if __name__ == '__main__': main()
