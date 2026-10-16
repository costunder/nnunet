"""Run one complete TumorCP paper profile on a fixed, explicit CT cohort.

Source preparation uses the standard library. Imaging/model packages are imported
only by an explicitly requested preparation, smoke or CUDA training stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

from runtime import build_runtime


TASK = 'Task040_KiTS'
TRAINER = 'nnUNetTrainerV2_TumorCPPaper'
PLANS = 'nnUNetPlansv2.1'
CASE_IDS = tuple(f'case_{index:05d}' for index in range(210))
DATASET_PROFILE = None
RAW_LAYOUT = 'kits19'
HERE = Path(__file__).resolve().parent


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.new')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=_json_value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json_value(value):
    # NumPy is intentionally lazy-imported; plans contain its scalar types.
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Not a JSON value: {type(value).__name__}')


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def configure_profile(path: Path | None, root: Path):
    """Keep KiTS19 default, or bind the complete original liver 105/26 split."""
    global DATASET_PROFILE, CASE_IDS, TASK, RAW_LAYOUT
    saved = Path(root) / 'dataset_profile.json'
    if path is None and saved.is_file():
        path = saved
    if path is None:
        return
    profile = read_json(path)
    if (profile.get('format') != 'tumorcp_paper_dataset_profile_v1'
            or profile.get('dataset') != 'MSD Task03_Liver'
            or profile.get('task') != 'Task029_Liver'
            or profile.get('raw_layout') != 'msd'
            or profile.get('allow_empty_intra') is not True):
        raise ValueError('Only the complete fixed MSD Task03 Liver paper transfer is accepted')
    expected = {f'liver_{i}' for i in range(131)}
    all_ids, train, heldout = (profile.get(k, []) for k in ('all_ids', 'train_ids', 'heldout_ids'))
    if (len(all_ids) != 131 or set(all_ids) != expected
            or len(train) != 105 or len(set(train)) != 105
            or len(heldout) != 26 or len(set(heldout)) != 26
            or set(train) & set(heldout) or set(train) | set(heldout) != expected):
        raise ValueError('Preserve all 131 liver patients and the disjoint 105/26 split')
    source = Path(profile['source_split_file'])
    expected_sha = '7a993854c570b2cabe98dffc5df97550d74da3a688c3c3b75942ce4cbd15bc44'
    if profile.get('source_split_sha256') != expected_sha or fingerprint(source) != expected_sha:
        raise ValueError('Original liver split bytes changed; no new partition substituted')
    original = read_json(source)[0]
    source_tr = original.get('train', original.get('outer_train', []))
    source_va = original.get('val', original.get('outer_val', []))
    if set(source_tr) != set(train) or set(source_va) != set(heldout):
        raise ValueError('Liver profile does not match the historical source split')
    if saved.exists():
        if read_json(saved) != profile:
            raise ValueError('Saved dataset profile differs; existing output preserved')
    else:
        atomic_json(saved, profile)
    DATASET_PROFILE, CASE_IDS, TASK, RAW_LAYOUT = profile, tuple(sorted(expected)), profile['task'], 'msd'


def cohort_counts():
    return (len(CASE_IDS), len(DATASET_PROFILE['train_ids']), len(DATASET_PROFILE['heldout_ids'])) if DATASET_PROFILE else (210, 168, 42)


def raw_paths(data: Path, case: str):
    if RAW_LAYOUT == 'msd':
        return data / 'imagesTr' / (case + '.nii.gz'), data / 'labelsTr' / (case + '.nii.gz')
    return data / case / 'imaging.nii.gz', data / case / 'segmentation.nii.gz'


def final_report_path(root: Path):
    return root / f'final{cohort_counts()[2]}_report.json'


def runtime_env(root: Path, workers: int) -> dict:
    """Return the exact environment also used by the independent smoke command.

    ``workers`` is the upstream DA setting: actual train processes=2*workers,
    validation processes=workers. It also controls preprocessing concurrency.
    """
    root = Path(root).resolve()
    if workers < 1:
        raise ValueError('Explicit positive worker concurrency is required')
    env = os.environ.copy()
    env.update({
        'nnUNet_raw_data_base': str(root / 'data'),
        'nnUNet_preprocessed': str(root / 'preprocessed'),
        'RESULTS_FOLDER': str(root / 'results'),
        'PYTHONPATH': str(root / 'runtime'),
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1',
        'nnUNet_n_proc_DA': str(workers), 'nnUNet_def_n_proc': str(workers),
        'nnunet_use_progress_bar': '1',
        # Each Python loader process is parallel; avoid nested BLAS oversubscription.
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    })
    profile = root / 'dataset_profile.json'
    if profile.is_file():
        env['TUMORCP_PAPER_DATASET_JSON'] = str(profile)
    return env


def _dependencies(names):
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('The isolated reproduction interpreter lacks required packages: ' + ', '.join(missing))


def resource_snapshot(root: Path, *, cuda_required: bool) -> dict:
    _dependencies(('psutil', 'torch'))
    import psutil
    import torch
    process = psutil.Process()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    affinity = process.cpu_affinity() if hasattr(process, 'cpu_affinity') else None
    data = {
        'observed_at': utc_now(), 'host': socket.gethostname(), 'PID': os.getpid(),
        'CPU_logical': psutil.cpu_count(), 'CPU_affinity': affinity,
        'RAM_total_bytes': psutil.virtual_memory().total,
        'RAM_available_bytes': psutil.virtual_memory().available,
        'process_RSS_bytes': process.memory_info().rss,
        'disk_free_bytes': shutil.disk_usage(root).free,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__, 'torch_CUDA_version': torch.version.cuda,
        'CUDA_available': torch.cuda.is_available(),
        'visible_GPU_count': torch.cuda.device_count(), 'GPUs': [],
        'model': 'official TumorCP nnUNet Generic_UNet 3d_fullres',
        'epochs': 500, 'updates_per_epoch': 250, 'total_updates': 125000,
        'physical_batch_size': 2, 'gradient_accumulation': 1,
        'data_parallel_workers': 1, 'effective_batch_size': 2,
        'precision': 'official nnUNet autocast AMP',
        'DEBUG': False, 'subset': False, 'training_started': False,
    }
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        data['GPUs'].append({'logical_device': device, 'model': props.name,
                             'total_VRAM_bytes': total, 'free_VRAM_bytes': free})
    if cuda_required and (not data['CUDA_available'] or data['visible_GPU_count'] != 1):
        raise RuntimeError('Training requires exactly one allocated visible CUDA device; no CPU fallback')
    print('RESOURCE ' + json.dumps(data, ensure_ascii=False), flush=True)
    return data


@contextmanager
def owned_lock(root: Path):
    """Protect only this run, never another session or a shared experiment."""
    lock = root / '.pipeline.lock'
    token = uuid.uuid4().hex
    owner = {'host': socket.gethostname(), 'PID': os.getpid(), 'token': token,
             'command': sys.argv, 'created_at': utc_now()}
    try:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(owner, stream)
    except FileExistsError:
        previous = read_json(lock)
        _dependencies(('psutil',))
        import psutil
        if previous.get('host') != socket.gethostname() or psutil.pid_exists(int(previous['PID'])):
            raise RuntimeError(f'Owned run is already locked; no lock removed: {lock}; {previous}') from None
        archive = lock.with_name('.pipeline.lock.stale.' + uuid.uuid4().hex)
        lock.rename(archive)
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(owner, stream)
        print(f'Archived dead same-host owner receipt: {archive}', flush=True)
    try:
        yield
    finally:
        if lock.is_file() and read_json(lock).get('token') == token:
            lock.unlink()


def _state(root, stage, status='RUNNING', **extra):
    value = {'format': 'tumorcp_kits19_pipeline_state_v1', 'updated_at': utc_now(),
             'stage': stage, 'status': status, 'PID': os.getpid(),
             'host': socket.gethostname(), 'training_complete': False,
             'evaluation_complete': False, 'quality_verified': False}
    value.update(extra)
    atomic_json(Path(root) / 'state.json', value)


def run_command(command, *, root: Path, stage: str, env: dict):
    """Tee native output, with heartbeat while long native operations are quiet."""
    log_dir = root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f'{time.time_ns()}_{stage}.log'
    print(f'{stage}: native command starts; complete output: {log}', flush=True)
    with log.open('x', encoding='utf-8') as output:
        output.write(json.dumps({'command': [str(x) for x in command], 'stage': stage,
                                 'started_at': utc_now()}, ensure_ascii=False) + '\n')
        output.flush()
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env, cwd=root / 'runtime',
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        events = queue.Queue()

        def read_output():
            current = []
            try:
                while True:
                    char = process.stdout.read(1)
                    if not char:
                        if current:
                            events.put(''.join(current))
                        break
                    if char in '\r\n':
                        if current:
                            events.put(''.join(current))
                            current = []
                    else:
                        current.append(char)
            finally:
                events.put(None)

        reader = threading.Thread(target=read_output, name=f'{stage}_output', daemon=True)
        reader.start()
        started = time.monotonic()
        last_console = last_state = started
        latest = ''
        _state(root, stage, child_PID=process.pid, log=str(log), command=[str(x) for x in command])
        eof = False
        while not eof:
            try:
                line = events.get(timeout=1.)
            except queue.Empty:
                line = ''
            now = time.monotonic()
            if line is None:
                eof = True
            elif line:
                output.write(line + '\n')
                output.flush()
                latest = line
                progress_line = '%|' in line or ('/250' in line and 'Epoch' in line)
                if not progress_line or now - last_console >= 3.:
                    print(line, flush=True)
                    last_console = now
            if now - last_state >= 10.:
                _state(root, stage, child_PID=process.pid, log=str(log),
                       elapsed_seconds=now - started, latest_native_output=latest)
                if now - last_console >= 15.:
                    print(f'{stage} | running {now-started:.0f}s | {latest[-220:]}', flush=True)
                    last_console = now
                last_state = now
        code = process.wait()
        reader.join(timeout=1.)
    if code:
        raise RuntimeError(f'{stage} failed with code {code}; original inputs/results preserved; see {log}')
    return {'command': [str(x) for x in command], 'log': str(log),
            'returncode': code, 'wall_seconds': time.monotonic() - started}


def bind_contract(root: Path, data: Path, upstream: Path, workers: int):
    source = upstream / 'nnunet'
    if not source.is_dir():
        raise FileNotFoundError(source)
    total, training, heldout = cohort_counts()
    payload = {
        'format': 'tumorcp_kits19_paper_request_v1', 'data': str(data),
        'upstream': str(upstream),
        'trainer': TRAINER, 'dataset': TASK, 'cases': total, 'train': training, 'heldout': heldout,
        'fold': 0, 'split_seed': 12345, 'epochs': 500, 'updates_per_epoch': 250,
        'physical_batch': 2, 'accumulation': 1, 'effective_batch': 2,
        'checkpoint': 'model_final_checkpoint', 'evaluation': f'{heldout} full CTs, raw, fixed FINAL',
        'paper_exact_split_ids_published': False,
        'builder_sha256': fingerprint(HERE / 'runtime.py'),
        'trainer_sha256': fingerprint(HERE / 'paper_trainer.py'),
        'runner_sha256': fingerprint(Path(__file__)),
    }
    if DATASET_PROFILE:
        payload.update(format='tumorcp_liver_paper_request_v1', dataset_profile=DATASET_PROFILE,
                       dataset_profile_sha256=fingerprint(root / 'dataset_profile.json'),
                       split_seed=None, split_provenance=DATASET_PROFILE['split_provenance'],
                       comparison_scope='paper settings transferred to liver, not KiTS19 numerical reproduction')
    destination = root / 'contract.json'
    if destination.exists():
        if read_json(destination) != payload:
            raise ValueError(f'Run contract differs; use a new isolated root: {destination}')
    else:
        atomic_json(destination, payload)
    return payload


def validate_raw_case(case_id: str, data: Path):
    import nibabel as nib
    import numpy as np
    image_path, label_path = raw_paths(data, case_id)
    for path in (image_path, label_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Complete fixed CT cohort required: {path}')
    image, label = nib.load(image_path), nib.load(label_path)
    if len(image.shape) != 3 or image.shape != label.shape:
        raise ValueError(f'KiTS19 CT/GT shape mismatch: {case_id}')
    if not np.allclose(image.affine, label.affine, atol=1e-5):
        raise ValueError(f'KiTS19 CT/GT affine mismatch: {case_id}')
    if not np.isfinite(image.affine).all() or min(image.header.get_zooms()[:3]) <= 0:
        raise ValueError(f'Invalid CT physical geometry: {case_id}')
    labels = np.asanyarray(label.dataobj)
    classes, counts = np.unique(labels, return_counts=True)
    if not set(classes.tolist()) <= {0, 1, 2} or not np.any(labels == 1):
        raise ValueError(f'Unexpected KiTS19 labels: {case_id}: {classes}')
    if not np.isfinite(np.asanyarray(image.dataobj)).all():
        raise ValueError(f'Non-finite original CT intensities: {case_id}')
    return {'case_id': case_id, 'shape': list(image.shape), 'spacing': [float(x) for x in image.header.get_zooms()[:3]],
            'label_counts': {str(int(k)): int(v) for k, v in zip(classes, counts)},
            'image_bytes': image_path.stat().st_size, 'label_bytes': label_path.stat().st_size,
            'image_sha256': fingerprint(image_path), 'label_sha256': fingerprint(label_path)}


def validate_raw(data: Path, workers: int, root: Path):
    _dependencies(('numpy', 'nibabel'))
    expected = set(CASE_IDS)
    total = len(CASE_IDS)
    present = ({p.name[:-7] for p in (data / 'imagesTr').glob('liver_*.nii.gz')} if RAW_LAYOUT == 'msd' else
               {path.name for path in data.iterdir() if path.is_dir() and path.name.startswith('case_')})
    if not expected <= present:
        raise ValueError(f'All {total} labelled patients are required; missing {sorted(expected - present)}')
    records = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='kits19_verify') as pool:
        tasks = {pool.submit(validate_raw_case, case, data): case for case in CASE_IDS}
        for future in as_completed(tasks):
            records.append(future.result())
            elapsed = time.monotonic() - started
            print(f'{TASK} data verification {len(records)}/{total} {len(records)/total:.1%} | {tasks[future]} | {elapsed:.1f}s', flush=True)
            _state(root, 'verify_data', completed_cases=len(records), total_cases=total, elapsed_seconds=elapsed)
    records.sort(key=lambda row: row['case_id'])
    report = {'format': 'tumorcp_kits19_verified_data_v1', 'complete': True,
              'source_cases': total, 'data': str(data), 'workers': workers, 'records': records,
              'wall_seconds': time.monotonic() - started}
    if DATASET_PROFILE:
        report.update(format='tumorcp_liver_verified_data_v1', raw_layout=RAW_LAYOUT,
                      dataset_profile=DATASET_PROFILE,
                      dataset_profile_sha256=fingerprint(root / 'dataset_profile.json'))
    atomic_json(root / 'data_verified.json', report)
    return report


def _link_or_copy(source: Path, destination: Path):
    if destination.exists():
        if destination.is_symlink() or fingerprint(source) != fingerprint(destination):
            raise ValueError(f'Existing raw destination differs; preserved: {destination}')
        return 'verified_existing'
    try:
        os.link(source, destination)
        return 'hardlink'
    except OSError as error:
        # Cross-filesystem and restricted hardlinks are an explicitly recorded
        # storage choice, not a change to the copied CT/GT bytes.
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=4 * 2**20)
        if fingerprint(source) != fingerprint(destination):
            raise ValueError(f'Raw copy verification failed: {destination}')
        return f'byte_verified_copy; hardlink errno={error.errno}'


def materialize_task(root: Path, data: Path):
    task = root / 'data/nnUNet_raw_data' / TASK
    for name in ('imagesTr', 'labelsTr', 'imagesTs'):
        (task / name).mkdir(parents=True, exist_ok=True)
    storage = []
    for case in CASE_IDS:
        image_source, label_source = raw_paths(data, case)
        for name, source, target in (
                ('CT', image_source, task / 'imagesTr' / f'{case}_0000.nii.gz'),
                ('GT', label_source, task / 'labelsTr' / f'{case}.nii.gz')):
            storage.append({'case_id': case, 'kind': name, 'path': str(target),
                            'storage': _link_or_copy(source, target)})
    descriptor = {
        'name': 'KiTS', 'description': 'kidney and kidney tumor segmentation',
        'tensorImageSize': '4D', 'reference': 'KiTS19 official annotated cohort',
        'licence': 'KiTS19 dataset terms', 'release': '0.0', 'modality': {'0': 'CT'},
        'labels': {'0': 'background', '1': 'Kidney', '2': 'Tumor'},
        'numTraining': len(CASE_IDS), 'numTest': 0,
        'training': [{'image': f'./imagesTr/{case}.nii.gz', 'label': f'./labelsTr/{case}.nii.gz'} for case in CASE_IDS],
        'test': [],
    }
    if DATASET_PROFILE:
        descriptor.update(name='Liver', description='liver and liver tumor segmentation',
                          reference='MSD Task03_Liver, original project 105/26 split',
                          licence='MSD CC BY-SA 4.0',
                          labels={'0': 'background', '1': 'Liver', '2': 'Tumor'})
    path = task / 'dataset.json'
    if path.exists():
        if read_json(path) != descriptor:
            raise ValueError(f'Existing dataset descriptor differs: {path}')
    else:
        atomic_json(path, descriptor)
    atomic_json(root / 'raw_storage.json', {'files': storage, 'dataset_json_sha256': fingerprint(path)})
    return path


def _plans_information(root: Path):
    path = root / 'preprocessed' / TASK / f'{PLANS}_plans_3D.pkl'
    with path.open('rb') as stream:
        plans = pickle.load(stream)
    stages = plans['plans_per_stage']
    stage = max(stages)
    selected = stages[stage]
    if int(selected['batch_size']) != 2 or len(selected['patch_size']) != 3:
        raise ValueError(f'Original automatic fullres plans differ from paper B2/3D: {selected}; no plan was changed')
    files = root / 'preprocessed' / TASK / f"{plans['data_identifier']}_stage{stage}"
    npzs = {p.stem for p in files.glob('*.npz')}
    if npzs != set(CASE_IDS):
        raise ValueError(f'Preprocessing must contain exactly all {len(CASE_IDS)} cases, no subset: {files}')
    return plans, path, stage, files


def _write_split(root: Path):
    import numpy as np
    from sklearn.model_selection import KFold
    from collections import OrderedDict
    if DATASET_PROFILE:
        splits = [OrderedDict(train=np.asarray(sorted(DATASET_PROFILE['train_ids'])),
                              val=np.asarray(sorted(DATASET_PROFILE['heldout_ids'])))]
    else:
        keys = np.asarray(CASE_IDS)
        splits = [OrderedDict(train=keys[tr], val=keys[va])
                  for tr, va in KFold(5, shuffle=True, random_state=12345).split(keys)]
    path = root / 'preprocessed' / TASK / 'splits_final.pkl'
    if path.exists():
        with path.open('rb') as stream:
            previous = pickle.load(stream)
        if len(previous) != len(splits) or any(list(a[k]) != list(b[k]) for a, b in zip(previous, splits) for k in ('train', 'val')):
            raise ValueError('Existing fixed split differs; no split was overwritten')
    else:
        with path.open('xb') as stream:
            pickle.dump(splits, stream, protocol=4)
    return path, list(splits[0]['train']), list(splits[0]['val'])


def prepare(root: Path, data: Path, upstream: Path, workers: int):
    _dependencies(('numpy', 'nibabel', 'SimpleITK', 'scipy', 'skimage', 'sklearn',
                   'batchgenerators', 'medpy', 'torch', 'psutil'))
    resource = resource_snapshot(root, cuda_required=False)
    resource['preprocessing_workers'] = workers
    resource['actual_DA_training_processes'] = workers * 2
    resource['actual_DA_monitor_processes'] = workers
    atomic_json(root / 'resources_preparation.json', resource)
    if (root / 'prepared.json').is_file():
        receipt = read_json(root / 'prepared.json')
        if not receipt.get('complete'):
            raise ValueError('Malformed incomplete preparation receipt')
        build_runtime(upstream, root / 'runtime')
        _, plans_file, _, fullres = _plans_information(root)
        if fingerprint(plans_file) != receipt['plans_sha256'] or fingerprint(Path(receipt['split_file'])) != receipt['split_sha256']:
            raise ValueError('Completed plans/split bytes changed; no cache silently reused')
        expected_donors = {case for case, has_tumor in receipt['tumor_presence'].items() if has_tumor} if 'tumor_presence' in receipt else set(CASE_IDS)
        if {p.name[:-10] for p in fullres.glob('*_tumor.npy')} != expected_donors:
            raise ValueError('Completed tumor extraction inventory differs')
        print(f'Verified completed {len(CASE_IDS)}-case preparation reused: ' + str(root / 'prepared.json'), flush=True)
        return receipt
    verified = validate_raw(data, workers, root)
    tumor_presence = {r['case_id']: r['label_counts'].get('2', 0) > 0 for r in verified['records']}
    build_runtime(upstream, root / 'runtime')
    materialize_task(root, data)
    env = runtime_env(root, workers)
    command = [sys.executable, '-B', '-u', '-m', 'nnunet.experiment_planning.nnUNet_plan_and_preprocess',
               '-t', str(int(TASK[4:7])), '-pl2d', 'None', '-tf', str(workers), '-tl', str(workers), '--verify_dataset_integrity']
    preprocessing = run_command(command, root=root, stage='plan_and_preprocess', env=env)
    plans, plans_file, stage, fullres = _plans_information(root)
    # Run the preserved extractor implementation from the isolated runtime, but
    # discover its actual planned fullres stage rather than the author's stage1 path.
    script = '''import concurrent.futures,json,os,sys,time
from pathlib import Path
import numpy as np
from nnunet.extract_tumors import TumorExtractor
directory=Path(sys.argv[1]); workers=int(sys.argv[2])
verified=json.loads(Path(sys.argv[3]).read_text())
presence={r['case_id']:r['label_counts'].get('2',0)>0 for r in verified['records']}
extractor=TumorExtractor(str(directory),num_processes=workers)
cases=sorted(presence)
def one(case):
    output=directory/(case+"_tumor.npy")
    if not presence[case]:
        with np.load(directory/(case+'.npz')) as archive:
            if np.any(archive['data'][-1]==2): raise RuntimeError('Resampled tumor present despite raw-empty claim: '+case)
        if output.exists(): raise RuntimeError('Unexpected tumor file for actual tumor-free patient: '+case)
        return case,0,0
    if not output.exists(): extractor.extract(case+".npz")
    if not output.is_file(): raise RuntimeError("Tumor-bearing patient has no extracted tumor: "+case)
    tumors=np.load(output,allow_pickle=True)
    if len(tumors)==0: raise RuntimeError("Empty extracted tumor container: "+case)
    for data,mask in tumors:
        if data.shape != mask.shape or not np.any(mask==2): raise RuntimeError("Invalid extracted CT/tumor: "+case)
        if data.dtype.kind not in 'fiu' or mask.dtype.kind not in 'iu' or not np.isfinite(data).all(): raise RuntimeError("Extracted voxels must retain finite numeric ndarray dtypes: "+case)
    return case,len(tumors),output.stat().st_size
started=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures=[pool.submit(one,case) for case in cases]
    for number,future in enumerate(concurrent.futures.as_completed(futures),1):
        case,count,size=future.result()
        print(f"Tumor extraction {number}/{len(cases)} {number/len(cases):.1%} | {case} | components={count} bytes={size} elapsed={time.monotonic()-started:.1f}s",flush=True)
'''
    extraction = run_command([sys.executable, '-B', '-u', '-c', script, str(fullres), str(workers), str(root / 'data_verified.json')],
                             root=root, stage='tumor_extraction', env=env)
    split, training, heldout = _write_split(root)
    selected = plans['plans_per_stage'][stage]
    receipt = {
        'format': 'tumorcp_kits19_prepared_v1', 'complete': True, 'prepared_at': utc_now(),
        'source_cases': len(CASE_IDS), 'train_ids': training, 'heldout_ids': heldout, 'workers': workers,
        'runtime': str(root / 'runtime'),
        'data': str(data), 'data_verified_file': str(root / 'data_verified.json'),
        'raw_layout': RAW_LAYOUT, 'tumor_presence': tumor_presence,
        'runtime_receipt_sha256': fingerprint(root / 'runtime/runtime_receipt.json'),
        'data_verified_sha256': fingerprint(root / 'data_verified.json'),
        'plans_file': str(plans_file), 'plans_sha256': fingerprint(plans_file),
        'fullres_stage': int(stage), 'fullres_directory': str(fullres),
        'split_file': str(split), 'split_sha256': fingerprint(split),
        'physical_batch': 2, 'input_shape': [2, 1] + [int(x) for x in selected['patch_size']],
        'plans_pool_kernels': selected['pool_op_kernel_sizes'],
        'plans_conv_kernels': selected['conv_kernel_sizes'],
        'base_channels': int(plans['base_num_features']),
        'preprocessing': preprocessing, 'extraction': extraction,
        'paper_exact_split_ids_published': False,
        'training_started': False, 'full_training_complete': False,
        'full_evaluation_complete': False, 'quality_verified': False,
    }
    if DATASET_PROFILE:
        receipt.update(format='tumorcp_liver_prepared_v1', dataset_profile=DATASET_PROFILE,
                       dataset_profile_file=str(root / 'dataset_profile.json'),
                       dataset_profile_sha256=fingerprint(root / 'dataset_profile.json'))
    atomic_json(root / 'prepared.json', receipt)
    _state(root, 'preparation_complete', status='PREPARED', prepared=str(root / 'prepared.json'))
    print(f'PREPARED: all {len(CASE_IDS)} CTs, {len(training)}/{len(heldout)} fixed split; no training yet.', flush=True)
    return receipt


def check_smoke(root: Path, path: Path):
    receipt = read_json(path)
    expected = {'format': 'tumorcp_kits19_real_cuda_smoke_v1', 'completed': True,
                'actual_CT': True, 'CUDA': True, 'model': 'Generic_UNet',
                'physical_batch': 2, 'production_batch': 2}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f'Actual-CT CUDA smoke receipt missing/different {key}: {path}')
    if len(set(receipt.get('representative_cases', []))) < 3:
        raise ValueError('Actual-CT smoke requires at least three representative patients')
    if not {2, 4} <= set(receipt.get('batch_candidates', [])):
        raise ValueError('Actual B2/B4 resource measurement must be recorded before production')
    prepared = read_json(root / 'prepared.json')
    for key, value in (('prepared_sha256', fingerprint(root / 'prepared.json')),
                       ('plans_sha256', prepared['plans_sha256']),
                       ('runtime_receipt_sha256', prepared['runtime_receipt_sha256'])):
        if receipt.get(key) != value:
            raise ValueError(f'Smoke is not bound to these actual inputs/runtime: {key}')
    if DATASET_PROFILE and receipt.get('dataset_profile_sha256') != fingerprint(root / 'dataset_profile.json'):
        raise ValueError('Actual CT smoke is not bound to the fixed liver dataset profile')
    return receipt


def _training_folder(root: Path):
    return root / 'results/nnUNet/3d_fullres' / TASK / f'{TRAINER}__{PLANS}' / 'fold_0'


def _checkpoint_evidence(path: Path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    epochs = len(saved.get('plot_stuff', ([],))[0])
    return {'path': str(path), 'sha256': fingerprint(path), 'saved_epoch': int(saved.get('epoch', -1)),
            'completed_epochs': epochs, 'full_training_complete': epochs == 500 and int(saved.get('epoch', -1)) == 500}


def evaluate_report(root: Path, data: Path, heldout):
    import nibabel as nib
    import numpy as np
    count = len(heldout)
    organ = 'liver' if DATASET_PROFILE else 'kidney'
    folder = _training_folder(root) / f'validation_final_{count}_raw'
    predictions = {path.name for path in folder.glob('*.nii.gz')}
    expected = {case + '.nii.gz' for case in heldout}
    if predictions != expected:
        raise ValueError(f'All and only the held-out {count} predictions are required: missing {expected-predictions}, extra {predictions-expected}')
    rows = []
    for number, case in enumerate(heldout, 1):
        ref_img, pred_img = nib.load(raw_paths(data, case)[1]), nib.load(folder / (case + '.nii.gz'))
        if ref_img.shape != pred_img.shape or not np.allclose(ref_img.affine, pred_img.affine, atol=1e-4):
            raise ValueError(f'Original-resolution final prediction geometry differs: {case}')
        ref, pred = np.asanyarray(ref_img.dataobj), np.asanyarray(pred_img.dataobj)
        row = {'case_id': case}
        for name, truth, result in ((organ, ref > 0, pred > 0), ('tumor', ref == 2, pred == 2)):
            denominator = int(truth.sum()) + int(result.sum())
            row[name + '_Dice'] = (2. * int(np.count_nonzero(truth & result)) / denominator if denominator
                                    else (1.0 if DATASET_PROFILE else None))
            row[name + '_empty_both'] = denominator == 0
        rows.append(row)
        row['GT_tumor_positive'] = bool(np.any(ref == 2))
        print(f'FINAL full-CT Dice {number}/{count} {number/count:.1%} | {case} | tumor={row["tumor_Dice"]}', flush=True)
    summary = {}
    for name in (organ, 'tumor'):
        values = [row[name + '_Dice'] for row in rows if row[name + '_Dice'] is not None]
        if not values:
            raise ValueError(f'No evaluable real {name} cases; no zero metric substituted')
        summary[name] = {'mean_Dice': float(np.mean(values)), 'std_Dice_population': float(np.std(values)),
                         'evaluable_cases': len(values),
                         'empty_both_cases': sum(r[name + '_empty_both'] for r in rows)}
    report = {'format': 'tumorcp_kits19_final42_patient_Dice_v1', 'full_evaluation_complete': True,
              'cases': count, 'checkpoint': 'fixed epoch500 FINAL', 'postprocessing_selected_on_holdout': False,
              'kidney_definition': 'foreground union labels 1 and 2, official KiTS evaluation helper',
              'tumor_definition': 'label 2; per-patient Sorensen Dice',
              'metrics': summary, 'metric_per_case': rows, 'published_target_tumor_Dice': 0.8475,
              'paper_exact_split_ids_published': False,
              'quality_verified': False, 'target_reached_claim': False}
    if DATASET_PROFILE:
        positive = [r['tumor_Dice'] for r in rows if r['GT_tumor_positive'] and r['tumor_Dice'] is not None]
        report.pop('kidney_definition')
        report.update(format='tumorcp_liver_final26_patient_Dice_v1',
                      dataset_profile_sha256=fingerprint(root / 'dataset_profile.json'),
                      liver_definition='foreground union labels 1 and 2',
                      published_target_tumor_Dice=None,
                      paper_result_reproduction=False,
                      comparison_scope='TumorCP paper settings on fixed MSD liver split',
                      empty_both_Dice_policy=1.0,
                      GT_tumor_positive_cases=len(positive),
                      mean_tumor_Dice_GT_positive=float(np.mean(positive)) if positive else None)
    atomic_json(final_report_path(root), report)
    return report


def train(root: Path, data: Path, upstream: Path, workers: int, smoke_path: Path):
    prepared = prepare(root, data, upstream, workers)
    smoke = check_smoke(root, smoke_path)
    resource = resource_snapshot(root, cuda_required=True)
    resource['DA_setting_workers'] = workers
    resource['actual_DA_training_processes'] = 2 * workers
    resource['actual_DA_monitor_processes'] = workers
    atomic_json(root / 'resources_training.json', resource)
    folder = _training_folder(root)
    final, latest = folder / 'model_final_checkpoint.model', folder / 'model_latest.model'
    env = runtime_env(root, workers)
    command = [sys.executable, '-B', '-u', '-m', 'nnunet.run.run_training',
               '3d_fullres', TRAINER, str(int(TASK[4:7])), '0', '-epoch', '500', '--disable_postprocessing_on_folds']
    if (root / 'completion.json').is_file():
        completed = read_json(root / 'completion.json')
        if not completed.get('full_training_complete') or not completed.get('full_evaluation_complete'):
            raise ValueError('Existing completion receipt does not prove full training and full evaluation')
        if fingerprint(final) != completed['checkpoint']['sha256'] or fingerprint(final_report_path(root)) != completed['report_sha256']:
            raise ValueError('Existing completed checkpoint/report bytes changed; no completion silently reused')
        print('Verified prior full500 training and full heldout evaluation; no neural evaluation repeated.', flush=True)
        return completed
    if final.is_file():
        evidence = _checkpoint_evidence(final)
        if not evidence['full_training_complete']:
            raise ValueError('A FINAL-named checkpoint does not contain all 500 epochs; preserved for diagnosis')
        command.append('--validation_only')
        print('Full training already complete; fixed FINAL evaluation only.', flush=True)
    elif latest.is_file():
        evidence = _checkpoint_evidence(latest)
        if not 0 < evidence['completed_epochs'] < 500:
            raise ValueError('Latest checkpoint cursor is inconsistent with this 500epoch run')
        command.append('--continue_training')
        print(f'Resume own model/optimizer/AMP checkpoint after {evidence["completed_epochs"]}/500 epochs.', flush=True)
    heldout_count = cohort_counts()[2]
    invocation = run_command(command, root=root, stage=f'train500_and_final{heldout_count}', env=env)
    if not final.is_file():
        raise RuntimeError('Native command returned without a FINAL checkpoint; full training not claimed')
    evidence = _checkpoint_evidence(final)
    if not evidence['full_training_complete']:
        raise RuntimeError('Native command returned before all 500 epochs; full training not claimed')
    report = evaluate_report(root, data, prepared['heldout_ids'])
    result = {'format': 'tumorcp_kits19_reproduction_completion_v1', 'completed_at': utc_now(),
              'full_training_complete': True, 'completed_epochs': 500,
              'scheduled_update_iterations': 125000,
              'AMP_skipped_optimizer_steps': 'not instrumented by the preserved official training loop',
              'full_evaluation_complete': True, 'evaluation_cases': heldout_count, 'checkpoint': evidence,
              'metrics': report['metrics'], 'published_tumor_Dice': 0.8475,
              'report_sha256': fingerprint(final_report_path(root)),
              'paper_exact_split_ids_published': False, 'quality_verified': False,
              'smoke_receipt_sha256': fingerprint(smoke_path), 'native_invocation': invocation}
    if DATASET_PROFILE:
        result.update(format='tumorcp_liver_paper_transfer_completion_v1',
                      published_tumor_Dice=None, dataset_profile=DATASET_PROFILE)
    atomic_json(root / 'completion.json', result)
    _state(root, f'fixed_FINAL_full{heldout_count}_evaluation', status='COMPLETE', training_complete=True,
           evaluation_complete=True, result=str(root / 'completion.json'))
    print(f'COMPLETE: full500 training and fixed FINAL{heldout_count} evaluation; results in {final_report_path(root)}.', flush=True)
    print(json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--upstream', type=Path, default=HERE / 'upstream')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prepare-workers', type=int,
                        help='Independent preparation concurrency; defaults to --workers, may differ from GPU DA workers')
    parser.add_argument('--phase', choices=('prepare', 'train', 'status', 'all'), required=True)
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--dataset-profile', type=Path,
                        help='Explicit full liver 131 / original 105:26 contract; omitted keeps official KiTS19')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'status':
        for name in ('contract.json', 'state.json', 'prepared.json', 'smoke.json', 'completion.json'):
            path = root / name
            if path.is_file():
                value = read_json(path)
                if name in ('prepared.json', 'contract.json'):
                    value = {key: item for key, item in value.items() if key not in ('train_ids', 'heldout_ids', 'records')}
                print(name + ': ' + json.dumps(value, ensure_ascii=False), flush=True)
        return
    if args.data is None or args.workers is None or args.workers < 1:
        raise ValueError('--data and explicit positive --workers are required for prepare/train/all')
    data, upstream = args.data.resolve(), args.upstream.resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    if root == data or root == upstream or data in root.parents or upstream in root.parents:
        raise ValueError('Use a separate reproduction output root, outside source/raw data directories')
    root.mkdir(parents=True, exist_ok=True)
    configure_profile(args.dataset_profile, root)
    bind_contract(root, data, upstream, args.workers)
    prepare_workers = args.prepare_workers if args.prepare_workers is not None else args.workers
    if prepare_workers < 1:
        raise ValueError('--prepare-workers must be positive')
    with owned_lock(root):
        try:
            if args.phase == 'prepare':
                prepare(root, data, upstream, prepare_workers)
            else:
                smoke = args.smoke_receipt.resolve() if args.smoke_receipt else root / 'smoke.json'
                if args.phase == 'all':
                    prepare(root, data, upstream, prepare_workers)
                    if not smoke.is_file():
                        run_command([sys.executable, '-B', '-u', HERE / 'smoke.py', '--root', root,
                                     '--workers', str(args.workers), '--output', smoke],
                                    root=root, stage='real_CT_CUDA_smoke', env=runtime_env(root, args.workers))
                train(root, data, upstream, args.workers, smoke)
        except Exception as error:
            _state(root, 'failed', status='FAILED', error_type=type(error).__name__, error=str(error))
            print('Original data, existing liver experiments and session are preserved. Failure details follow.', flush=True)
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
