"""Read-only selection of explicitly small diagnostic inputs from native cache.

An incomplete native preparation is accepted ONLY as a DEBUG source of already
verified successful records. It is never promoted to a complete training cache.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def select_records(rows, split, physical_batch):
    if type(physical_batch) is not int or physical_batch < 2:
        raise ValueError('Explicit DEBUG physical batch must contain at least two samples')
    selected, counts, seen = [], {'train': 0, 'val': 0}, set()
    allowed = {name: set(split[name]) for name in counts}
    for row in sorted(rows, key=lambda x: (x.get('case_id', ''), x.get('sample_index', ''))):
        if row.get('status') != 'ok':
            continue
        name, case = row.get('split'), row.get('case_id')
        if name not in allowed or case not in allowed[name]:
            raise ValueError('Successful manifest record lies outside the frozen split')
        # The original CSV also contains terminal case-level status='ok' rows.
        # Their blank sample_index and artifact fields identify summaries, not
        # eight-candidate samples. Do not treat their blank candidates as data.
        index = row.get('sample_index')
        if index in ('', None):
            if any(row.get(key) not in ('', None) for key in ('candidates', 'path', 'artifact_sha256', 'file_size')):
                raise ValueError('Case-level summary cannot carry a sample artifact without sample_index')
            continue
        if (not isinstance(index, str) or not index.isascii() or not index.isdecimal()
                or str(int(index)) != index):
            raise ValueError('Successful sample requires a canonical nonnegative sample_index')
        if case in seen or counts[name] >= (physical_batch if name == 'train' else 1):
            continue
        if row.get('candidates') != '8':
            raise ValueError('DEBUG record must preserve the full eight-candidate curriculum')
        selected.append(row)
        seen.add(case)
        counts[name] += 1
    if counts != {'train': physical_batch, 'val': 1}:
        raise ValueError('Insufficient verified records for the explicit DEBUG batch and held-out case')
    return selected


def prepare_debug_inputs(experiment, output, physical_batch):
    from .experiment import load_suite, preparation_root
    from .contracts import verify_archive
    experiment = Path(experiment).resolve(strict=True)
    manifest = load_suite(experiment)
    if manifest.get('sampling_contract', {}).get('mode') != 'native':
        raise ValueError('Scope DEBUG input must be a native preparation owner')
    shared = preparation_root(experiment, manifest)
    cache = shared / 'cache'
    progress = cache / 'manifest.csv'
    cache_config = cache / 'config.json'
    config = json.loads(cache_config.read_text(encoding='utf-8'))
    with progress.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    records = select_records(rows, manifest['split'], physical_batch)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError('Existing diagnostic outputs are preserved; choose a new output')
    if destination.is_relative_to(experiment) or experiment.is_relative_to(destination):
        raise ValueError('Scope diagnostic output must be disjoint from the source experiment')
    proof = verify_archive(ROOT)
    watched = {str(p): sha(p) for p in (experiment / 'manifest.json', progress, cache_config)}
    checked = []
    raw_records = []
    for row in records:
        relative = PurePosixPath(row['path'])
        if (relative.is_absolute() or len(relative.parts) != 1 or '\\' in row['path']
                or ':' in row['path'] or '..' in relative.parts):
            raise ValueError('Manifest artifact must be a contained cache filename')
        path = cache / row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError('Selected cache artifact escaped its original cache directory')
        digest = sha(path)
        if digest != row['artifact_sha256'] or path.stat().st_size != int(row['file_size']):
            raise ValueError('Selected original cache record bytes/size changed')
        if row['config_fingerprint'] != config['config_fingerprint']:
            raise ValueError('Selected cache record has a different preparation config')
        watched[str(path)] = digest
        checked.append((row, path, digest))
        raw = {'case_id': row['case_id']}
        for name, folder, suffix in (('image', 'image', '_0000.nii.gz'), ('label', 'labels', '.nii.gz')):
            raw_path = Path(manifest['medical_root']) / 'Data' / folder / (row['case_id'] + suffix)
            raw_hash = sha(raw_path)
            if raw_hash != row[f'source_{name}_sha256']:
                raise ValueError('Selected raw CT/label no longer matches original cache provenance')
            raw[name], raw[f'{name}_sha256'] = str(raw_path), raw_hash
            watched[str(raw_path)] = raw_hash
        raw_records.append(raw)
    destination.mkdir(parents=True, exist_ok=False)
    source, samples = destination / 'source', destination / 'samples'
    source.mkdir()
    samples.mkdir()
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as z:
        for info in z.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError('Archive member path is unsafe')
            target = source.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(z.read(info))
    entries = []
    for row, path, digest in checked:
        target = samples / path.name
        shutil.copyfile(path, target)
        if sha(target) != digest:
            raise ValueError('Diagnostic input copy differs from original bytes')
        entries.append({'name': target.name, 'split': row['split'], 'sha256': digest,
                        'case_id': row['case_id'], 'sample_index': int(row['sample_index'])})
    identity = {'files': entries, 'source_records': raw_records,
                'debug': True, 'source_preparation_complete_claimed': False,
                'selection': 'first verified successful sample per distinct case; explicit DEBUG only',
                'source_experiment': str(experiment), 'source_split': manifest['split'],
                'source_archive_sha256': proof['archive_sha256'], 'source_signatures': watched,
                'original_failed_requests_replayed': False,
                'full_training': False, 'quality_verified': False, 'production_ready': False}
    with (samples / 'fixture_manifest.json').open('x', encoding='utf-8') as f:
        json.dump(identity, f, indent=2, allow_nan=False)
    for path, digest in watched.items():
        if sha(path) != digest:
            raise ValueError('Original inputs changed during diagnostic packaging')
    return source, samples
