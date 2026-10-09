"""Immutable, full-cohort CPU geometry shared by native comparison checkpoints.

Only the exact historical ``upper_graphs`` is executed during preparation. No
neural model, L0 feature, score, training controller or optimizer is constructed.
The publication contains every held-out case, including cases with no observed
P. Each read creates a new private mmap, so PyG batching or caller mutations
cannot modify the sealed on-disk tensors or another checkpoint's inputs.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import copy
import hashlib
import json
import mmap
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from types import SimpleNamespace
from zipfile import ZipFile

from .contracts import canonical_hash, verify_archive
from .historical_evaluation import assert_new_destination, sha, write_new
from .transition_evaluation import QUERY_FIELDS, validate_cohort

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'comparison_native_complete_upper_geometry_cache_v1'
PAYLOAD_FORMAT = FORMAT + '_case'
CODE_FILES = (
    'hiercp_v1x/comparison_native_upper_cache.py',
    'hiercp_v1x/comparison_native_checkpoint.py',
    'hiercp_v1x/historical_evaluation.py',
    'hiercp_v1x/historical_patient_graph.py',
    'hiercp_v1x/historical_checkpoint.py',
    'hiercp_v1x/half_a_training.py',
    'hiercp_v1x/scope_probe_support.py',
    'hiercp_v1x/bounded_scope.py',
    'hiercp_v1x/transition_v1_local.py',
    'hiercp_v1x/transition_evaluation.py',
    'hiercp_v1x/native30_data_contract.py',
    'hiercp_v22/data.py',
)


def _regular(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError(f'Regular read-only geometry evidence required: {path}')
    return path.resolve(strict=True)


def _directory(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_dir():
        raise ValueError(f'Regular geometry directory required: {path}')
    return path.resolve(strict=True)


def _read(path):
    return json.loads(_regular(path).read_text(encoding='utf8'))


def _code():
    return {name: sha(_regular(ROOT / name)) for name in CODE_FILES}


def _sealed(value):
    result = copy.deepcopy(value)
    result['content_sha256'] = canonical_hash(value)
    return result


def _check_seal(value):
    if (not isinstance(value, dict) or value.get('content_sha256') != canonical_hash(
            {key: item for key, item in value.items() if key != 'content_sha256'})):
        raise ValueError('Upper geometry metadata content SHA256 differs')


def _relative(root, name):
    if not isinstance(name, str):
        raise ValueError('Exact relative geometry payload path required')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
            or path.as_posix() != name or not path.parts):
        raise ValueError('Unsafe relative geometry payload path')
    selected = _regular(root.joinpath(*path.parts))
    if not selected.is_relative_to(root):
        raise ValueError('Geometry payload escaped its sealed namespace')
    return selected


def _query(rows):
    """Match run_scoring's exact ordered, target-free query projection."""
    return [{key: list(row[key]) if key == 'center' else row[key]
             for key in QUERY_FIELDS} for row in rows]


def _source_proof(source):
    """Verify original source without importing or changing the active runtime."""
    source = _directory(source)
    archive_proof = verify_archive(ROOT)
    verified = {}
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
        for name in archive.namelist():
            if not ((name.startswith('hiercp/') and name.endswith('.py'))
                    or name == 'config/train.json'):
                continue
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if sha(_relative(source, name)) != expected:
                raise ValueError('Archived upper geometry source bytes differ: ' + name)
            verified[name] = expected
    actual = {path.relative_to(source).as_posix()
              for path in (source / 'hiercp').rglob('*.py')}
    if actual != {name for name in verified if name.startswith('hiercp/')}:
        raise ValueError('Archived upper geometry module inventory differs')
    return dict(source=str(source), archive_sha256=archive_proof['archive_sha256'],
                verified_files=verified)


def _raw_bindings(inventory, cohort):
    required = set(cohort['case_ids']) | {row['donor_case_id'] for row in cohort['rows']}
    raw = {row['case_id']: row for row in inventory['raw_records']}
    return {case: {key: copy.deepcopy(raw[case][key]) for key in
                  ('image', 'label', 'image_sha256', 'label_sha256')}
            for case in sorted(required)}


def _verify_raw(bindings):
    for case, row in bindings.items():
        for name in ('image', 'label'):
            if sha(_regular(row[name])) != row[name + '_sha256']:
                raise ValueError(f'Actual raw CT/annotation SHA256 differs: {case} {name}')


@dataclass(frozen=True)
class GeometryInputs:
    """The actual fields read by the historical builder; no neural placeholders."""
    baseline: Path
    source: Path
    config: dict
    prototype_bank: object
    receipt: dict
    scope: dict


def _signature(bundle, inventory_path, inventory, cohort):
    if bundle.receipt.get('debug') is not False or cohort['debug'] is not False:
        raise ValueError('Complete production upper geometry required; DEBUG cannot be promoted')
    bank_path = _regular(bundle.baseline / 'shared/prototype_bank.pt')
    bank = bundle.prototype_bank
    training = inventory['split']['inner_train']
    if (len(bank.training_case_ids) != len(set(bank.training_case_ids))
            or set(bank.training_case_ids) != set(training)):
        raise ValueError('Upper geometry requires the exact training-only prototype bank')
    proof = bundle.receipt.get('baseline_proof')
    if not isinstance(proof, dict):
        raise ValueError('Actual completed baseline proof required for upper geometry')
    baseline_manifest = _read(bundle.baseline / 'manifest.json')
    native_config_path = _regular(Path(baseline_manifest['source_experiment']) / 'configs/v1.0.json')
    native = _read(native_config_path)
    expected = copy.deepcopy(native['graph'])
    expected.update(adaptive_roi_margin_mm=10., context_outer_radius_mm=10.)
    if bundle.config['graph'] != expected:
        raise ValueError('Upper geometry physical-scope graph configuration differs')
    source_proof = _source_proof(bundle.source)
    recorded_source = getattr(bundle, 'checkpoint', {}).get('source_proof')
    if recorded_source and 'verified_files' in recorded_source and recorded_source != source_proof:
        raise ValueError('Loaded model archived source proof differs from actual geometry')
    if bundle.scope['contract_sha256'] != proof['neural_baseline']['scope_digest']:
        raise ValueError('Upper geometry trained scope differs from its baseline proof')
    return dict(format=FORMAT, native_inventory_path=str(_regular(inventory_path)),
        native_inventory_sha256=sha(inventory_path), cohort_sha256=cohort['cohort_sha256'],
        native_assignment=cohort['native_assignment'], case_ids=cohort['case_ids'],
        records=cohort['records'], observed_P=cohort['observed_P'], unobserved_U=cohort['unobserved_U'],
        query_fields=list(QUERY_FIELDS), query_rows_sha256=canonical_hash(_query(cohort['rows'])),
        baseline=str(_directory(bundle.baseline)), baseline_proof_sha256=canonical_hash(proof),
        graph=copy.deepcopy(bundle.config['graph']), ct_clip=copy.deepcopy(bundle.config['ct_clip']),
        source_pad=bundle.config['cache']['source_pad'],
        donor_max_diameter_mm=inventory['config']['donor_max_diameter_mm'],
        native_region_config_path=str(native_config_path), native_region_config_sha256=sha(native_config_path),
        source_proof=source_proof, scope=copy.deepcopy(bundle.scope),
        bank_path=str(bank_path), bank_sha256=sha(bank_path), bank_fingerprint=bank.fingerprint(),
        bank_training_case_ids=list(bank.training_case_ids), training_case_ids=list(training),
        raw_inputs=_raw_bindings(inventory, cohort),
        execution_code_sha256=_code(), target_in_graph_inputs=False,
        neural_model_constructed=False, L0_features_cached=False, optimizer_updates=0)


def _load_geometry_inputs(experiment, arm, inventory_path):
    """Reuse the comparison loader's admission and the real baseline proof."""
    from .comparison_native_checkpoint import ARMS, _manifest
    from .half_a_training import baseline_proof
    from .historical_checkpoint import _activate
    from .u_bridge_experiment import digest as comparison_digest
    if arm not in ARMS:
        raise ValueError('Exact preserved comparison arm required')
    experiment = _directory(experiment)
    manifest, _ = _manifest(experiment, arm)
    inventory_path = _regular(inventory_path)
    if sha(inventory_path) != manifest['baseline']['inventory_sha256']:
        raise ValueError('Native upper cache inventory differs from the sealed comparison')
    inventory = _read(inventory_path)
    cohort = validate_cohort(inventory, debug=False)
    baseline = _directory(manifest['baseline']['baseline'])
    native, proof = baseline_proof(baseline)
    config = copy.deepcopy(native['config'])
    config['graph'].update(adaptive_roi_margin_mm=10., context_outer_radius_mm=10.)
    source = _directory(baseline / 'source/v1.0')
    if (config != manifest['config'] or comparison_digest(proof) != manifest['baseline']['baseline_proof_sha256']
            or source != Path(manifest['original']['source']).resolve(strict=True)
            or set(native['split']['train']) != set(inventory['split']['inner_train'])
            or set(native['split']['val']) != set(cohort['case_ids'])):
        raise ValueError('Actual native upper geometry baseline/configuration/population differs')
    source_proof, scope = _activate(source, None)
    if source_proof != manifest['original'] or scope != manifest['scope']:
        raise ValueError('Activated original source/scope differs from the sealed comparison')
    from hiercp.prototype import PrototypeBank
    bank_path = _regular(baseline / 'shared/prototype_bank.pt')
    if sha(bank_path) != manifest['baseline']['bank_sha256']:
        raise ValueError('Actual original prototype bank SHA256 differs')
    bank = PrototypeBank.load(bank_path)
    if bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Actual original prototype fingerprint differs')
    bundle = GeometryInputs(baseline, source, config, bank,
                            dict(debug=False, baseline_proof=proof), scope)
    signature = _signature(bundle, inventory_path, inventory, cohort)
    preserved = {**proof['files'], str(experiment / 'experiment.json'): sha(experiment / 'experiment.json'),
                 str(inventory_path): signature['native_inventory_sha256'],
                 str(bank_path): signature['bank_sha256'],
                 str(signature['native_region_config_path']): signature['native_region_config_sha256']}
    preserved.update({str(ROOT / name): checksum for name, checksum in manifest['helpers'].items()})
    preserved.update({str(source / name): checksum for name, checksum in source_proof['verified_files'].items()})
    if (experiment / 'continuation.json').is_file():
        preserved[str(experiment / 'continuation.json')] = sha(experiment / 'continuation.json')
    return bundle, inventory, cohort, signature, preserved


def _region_files(root):
    root = _directory(root)
    paths = sorted(root.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('Original region cache contains a symlink')
    files = [path for path in paths if path.is_file()]
    if not files or not (root / 'metadata.json').is_file():
        raise ValueError('Complete original region publication required')
    return {path.relative_to(root).as_posix(): sha(_regular(path)) for path in files}


def _prepare_regions(bundle, signature, output, region_cache_source):
    """Precopy missing publications, then admit actual CT and native regions."""
    from hiercp.common import CasePaths, load_case, stable_case_seed
    from hiercp.region import load_or_build_patient_regions, REGION_CACHE_SEED_SALT
    from hiercp.schema import graph_config_from_dict
    config = graph_config_from_dict(_read(signature['native_region_config_path'])['graph'])
    fallback = output / 'regions'
    fallback.mkdir(exist_ok=False)
    source = _directory(region_cache_source) if region_cache_source is not None else None
    selected, preserved = {}, {}
    for case, raw in signature['raw_inputs'].items():
        old = bundle.baseline / 'shared/regions' / case
        if old.is_dir():
            directory = _directory(old)
        else:
            if source is None or not (source / case).is_dir():
                raise ValueError(f'Actual original region cache absent: {case}; no region replacement/rebuild')
            original = _directory(source / case)
            original_files = _region_files(original)
            preserved.update({str(original / name): checksum for name, checksum in original_files.items()})
            directory = fallback / case
            shutil.copytree(original, directory)
            if _region_files(directory) != original_files:
                raise ValueError('Copied actual original region bytes differ: ' + case)
        files = _region_files(directory)
        preserved.update({str(directory / name): checksum for name, checksum in files.items()})
        loaded = load_case(CasePaths(case, Path(raw['image']), Path(raw['label'])))
        regions = load_or_build_patient_regions(loaded, liver_label=1, tumor_label=2,
            config=config, ct_clip=tuple(bundle.config['ct_clip']),
            seed=stable_case_seed(42, case, REGION_CACHE_SEED_SALT), cache_dir=directory.parent,
            overwrite=False, mmap=True)
        selected[case] = dict(path=str(directory), files=files, metadata=_read(directory / 'metadata.json'))
        del regions, loaded
    return selected, preserved


def _verify_regions(bindings):
    for case, entry in bindings.items():
        if _region_files(entry['path']) != entry['files']:
            raise ValueError('Actual original region payload SHA256 differs: ' + case)
        if _read(Path(entry['path']) / 'metadata.json') != entry['metadata']:
            raise ValueError('Actual original region metadata differs: ' + case)


def _stat_identity(path):
    value = _regular(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _input_paths(signature, regions):
    paths = {signature['bank_path'], signature['native_region_config_path']}
    for item in signature['raw_inputs'].values():
        paths.update((item['image'], item['label']))
    for entry in regions.values():
        paths.update(str(Path(entry['path']) / name) for name in entry['files'])
    paths.update(str(Path(signature['source_proof']['source']) / name)
                 for name in signature['source_proof']['verified_files'])
    return sorted(paths)


def _private_mmap():
    import torch
    # torch.load always passes shared=False to UntypedStorage.from_file on
    # Windows, where Python exposes no MAP_PRIVATE/MAP_SHARED constants.
    return os.name == 'nt' or torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE


def _case_inputs(signature, region_bindings, rows):
    case_ids = list(dict.fromkeys([rows[0]['case_id'], rows[0]['donor_case_id']]))
    return dict(raw={case: signature['raw_inputs'][case] for case in case_ids},
                regions={case: region_bindings[case] for case in case_ids})


def _validate_graphs(graph, prototype, audit, rows, signature):
    import torch
    from torch_geometric.data import HeteroData
    if not isinstance(graph, HeteroData) or not isinstance(prototype, HeteroData):
        raise ValueError('Actual historical PyG heterogeneous graphs required')
    for value in (graph, prototype):
        value.validate(raise_on_error=True)
        for store in value.stores:
            for item in store.values():
                if isinstance(item, torch.Tensor):
                    if item.device.type != 'cpu' or item.requires_grad:
                        raise ValueError('CPU geometry tensors without neural gradients required')
                    if (item.is_floating_point() or item.is_complex()) and not bool(torch.isfinite(item).all()):
                        raise ValueError('Nonfinite original upper geometry')
    row = rows[0]
    if (graph['candidate'].raw_x.shape[0] != len(rows)
            or not isinstance(audit, dict) or audit.get('debug') is not False
            or audit.get('candidate_count') != len(rows)
            or audit.get('recipient_case_id') != row['case_id']
            or audit.get('donor_case_id') != row['donor_case_id']
            or audit.get('bank_fingerprint') != signature['bank_fingerprint']
            or audit.get('prototype_training_case_ids') != signature['bank_training_case_ids']
            or audit.get('P_U_labels_in_forward') is not False
            or audit.get('original_learned_operators_unchanged') is not True):
        raise ValueError('Actual historical whole-case graph/audit binding differs')
    modules = audit.get('original_source_module_sha256')
    expected = signature['source_proof']['verified_files']
    if not isinstance(modules, dict) or not modules or any(
            expected.get('hiercp/' + name + '.py') != checksum for name, checksum in modules.items()):
        raise ValueError('Historical upper graph audit source proof differs')


def _save_case(output, rows, graph, prototype, audit, signature, region_bindings, measurement):
    import torch
    _validate_graphs(graph, prototype, audit, rows, signature)
    case = rows[0]['case_id']
    name = 'cases/' + hashlib.sha256(case.encode()).hexdigest() + '.pt'
    query = _query(rows)
    inputs = _case_inputs(signature, region_bindings, query)
    receipt = dict(format=PAYLOAD_FORMAT, case_id=case, records=len(query),
        query_rows=query, query_rows_sha256=canonical_hash(query),
        signature_sha256=canonical_hash(signature), inputs_sha256=canonical_hash(inputs),
        audit_sha256=canonical_hash(audit))
    receipt = _sealed(receipt)
    payload = dict(format=PAYLOAD_FORMAT, graph=graph, prototype=prototype, audit=audit, receipt=receipt)
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (not _exact(graph.to_dict(), saved['graph'].to_dict())
            or not _exact(prototype.to_dict(), saved['prototype'].to_dict())
            or not _exact(audit, saved['audit']) or saved['receipt'] != receipt):
        raise ValueError('Original graph/prototype/audit changed during exact serialization roundtrip')
    del saved
    measurement = {**measurement, 'exact_serialization_roundtrip_verified': True}
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return dict(case_id=case, path=name, payload_sha256=sha(path), payload_bytes=path.stat().st_size,
                receipt=receipt, inputs=inputs, measurement=measurement)


def _exact(first, second):
    """Compare every persisted graph field, including tensor dtype and stride."""
    import numpy as np
    import torch
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return (first.dtype == second.dtype and first.shape == second.shape
                and first.stride() == second.stride() and torch.equal(first, second))
    if isinstance(first, np.ndarray):
        return (first.dtype == second.dtype and first.shape == second.shape
                and first.strides == second.strides and np.array_equal(first, second))
    if isinstance(first, dict):
        return list(first) == list(second) and all(_exact(value, second[key]) for key, value in first.items())
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(_exact(a, b) for a, b in zip(first, second))
    return first == second


_WORKER = None


def _initialize_worker(spec):
    global _WORKER
    import torch
    from .historical_checkpoint import _activate
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if _code() != spec['signature']['execution_code_sha256']:
        raise ValueError('Upper CPU builder code changed before worker initialization')
    proof, scope = _activate(Path(spec['source']), None)
    if proof != spec['signature']['source_proof'] or scope != spec['scope']:
        raise ValueError('Spawned CPU builder archived source/scope differs')
    from hiercp.prototype import PrototypeBank
    bank = PrototypeBank.load(spec['signature']['bank_path'])
    if (sha(spec['signature']['bank_path']) != spec['signature']['bank_sha256']
            or bank.fingerprint() != spec['signature']['bank_fingerprint']
            or list(bank.training_case_ids) != spec['signature']['bank_training_case_ids']):
        raise ValueError('Spawned CPU builder actual prototype differs')
    bundle = GeometryInputs(Path(spec['baseline']), Path(spec['source']), spec['config'], bank,
                            spec['receipt'], spec['scope'])
    _WORKER = dict(spec=spec, bundle=bundle,
                   provider=SimpleNamespace(ds=SimpleNamespace(meta=spec['inventory'])))


def _build_case(rows):
    import psutil
    from .historical_evaluation import upper_graphs
    if _WORKER is None:
        raise RuntimeError('Explicit CPU geometry worker initialization required')
    spec, bundle, provider = _WORKER['spec'], _WORKER['bundle'], _WORKER['provider']
    signature = spec['signature']
    inputs = _case_inputs(signature, spec['regions'], rows)
    _verify_raw(inputs['raw']); _verify_regions(inputs['regions'])
    process = psutil.Process()
    start, cpu_start = time.perf_counter(), sum(process.cpu_times()[:2])
    rss_before = process.memory_info().rss
    profiler = None
    if rows[0]['case_id'] == 'liver_109' or rows[0]['case_id'] == 'liver109':
        import cProfile
        profiler = cProfile.Profile(); profiler.enable()
    graph, prototype, audit = upper_graphs(bundle, provider, rows,
        region_output=Path(spec['output']) / 'regions')
    if profiler is not None:
        import io
        import pstats
        profiler.disable()
        report = io.StringIO()
        pstats.Stats(profiler, stream=report).sort_stats('cumulative').print_stats(80)
        with (Path(spec['output']) / 'liver109_full_input_cpu_profile.txt').open('x', encoding='utf8') as stream:
            stream.write(report.getvalue())
    rss_after = process.memory_info().rss
    peak_rss = max(rss_before, rss_after)
    if os.name != 'nt':
        import resource
        peak_rss = max(peak_rss, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    if peak_rss > spec['rss_bytes']:
        raise MemoryError('Full CPU builder RSS budget exceeded; no geometry/case reduction')
    measurement = dict(pid=os.getpid(), elapsed_seconds=time.perf_counter() - start,
        CPU_seconds=sum(process.cpu_times()[:2]) - cpu_start, peak_RSS_bytes=peak_rss,
        RSS_before_bytes=rss_before, RSS_after_bytes=rss_after, torch_threads=1,
        original_builder=True, complete_case=True,
        BLAS_environment={name: os.environ.get(name) for name in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS')})
    _verify_raw(inputs['raw']); _verify_regions(inputs['regions'])
    entry = _save_case(Path(spec['output']), rows, graph, prototype, audit, signature,
                       spec['regions'], measurement)
    print('NATIVE UPPER CACHE | case=' + rows[0]['case_id'] + ' | records=' + str(len(rows))
          + ' | seconds=' + format(measurement['elapsed_seconds'], '.2f')
          + ' | peak_RSS_GiB=' + format(peak_rss / 2**30, '.2f'), flush=True)
    return entry


def prepare_cache(experiment, arm, inventory, output, region_cache_source=None, workers=12, rss_gib=64):
    """Exclusively publish all 21 original case graphs with spawned CPU builders.

    ``rss_gib`` guards aggregate RSS for this parent and its exact pool children,
    not merely one worker. Failure leaves new partial evidence for inspection;
    no old file is replaced, no process is force-killed, and no case is skipped.
    """
    import psutil
    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, 'cpu_affinity') else list(range(os.cpu_count() or 1))
    available = psutil.virtual_memory().available
    if (type(workers) is not int or workers < 2 or workers > len(affinity)
            or type(rss_gib) not in (int, float) or not 0 < rss_gib * 2**30 < available):
        raise ValueError('Explicit parallel CPU count and aggregate RAM budget must fit actual availability')
    rss_bytes = int(rss_gib * 2**30)
    preserved_dirs = [experiment, inventory]
    if region_cache_source is not None:
        preserved_dirs.append(region_cache_source)
    assert_new_destination(output, preserved_dirs)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise FileExistsError('Existing upper cache destination is preserved; a fresh directory is required')
    start = time.perf_counter()
    bundle, metadata, cohort, signature, preserved = _load_geometry_inputs(experiment, arm, inventory)
    assert_new_destination(output, (bundle.baseline, bundle.source))
    _verify_raw(signature['raw_inputs'])
    output.mkdir(parents=True, exist_ok=False)
    request = _sealed(dict(format=FORMAT, signature=signature, workers=workers, rss_bytes=rss_bytes,
        actual_resources=dict(CPU_affinity=affinity, available_RAM_bytes=available,
            parent_RSS_bytes=process.memory_info().rss, storage_free_bytes=shutil.disk_usage(output).free),
        whole_inner_validation=True, debug=False, no_training=True))
    write_new(output / 'request.json', request)
    regions, region_preserved = _prepare_regions(bundle, signature, output, region_cache_source)
    preserved.update(region_preserved)
    spec = dict(baseline=str(bundle.baseline), source=str(bundle.source), config=bundle.config,
        receipt=bundle.receipt, scope=bundle.scope, inventory=metadata, signature=signature,
        output=str(output), regions=regions, rss_bytes=rss_bytes)
    completed = {}; aggregate_peak = process.memory_info().rss
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_worker, initargs=(spec,)) as pool:
        pending = {pool.submit(_build_case, _query(cohort['by_case'][case])): case for case in cohort['case_ids']}
        while pending:
            done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
            # These processes were created by this executor; unrelated workers
            # and the user's SSH/terminal session are never inspected or signaled.
            worker_processes = list(pool._processes.values())
            rss = process.memory_info().rss
            for worker in worker_processes:
                if worker.pid is not None and worker.is_alive():
                    try:
                        rss += psutil.Process(worker.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        # The executor reports any failed worker via its future.
                        pass
            aggregate_peak = max(aggregate_peak, rss)
            if rss > rss_bytes:
                for future in pending:
                    future.cancel()
                raise MemoryError('Aggregate full CPU geometry RSS budget exceeded; no case/graph reduction')
            for future in done:
                case = pending.pop(future)
                completed[case] = future.result()
    if set(completed) != set(cohort['case_ids']):
        raise ValueError('Full 21-case upper geometry incomplete; no publication')
    if _code() != signature['execution_code_sha256']:
        raise ValueError('Upper geometry execution code changed during preparation')
    _verify_raw(signature['raw_inputs']); _verify_regions(regions)
    if any(sha(_regular(path)) != checksum for path, checksum in preserved.items()):
        raise ValueError('Preserved actual baseline/source/region evidence changed during preparation')
    entries = [completed[case] for case in cohort['case_ids']]
    for entry in entries:
        if sha(_relative(output, entry['path'])) != entry['payload_sha256']:
            raise ValueError('Completed CPU geometry payload changed before publication')
    index = _sealed(dict(format=FORMAT, complete=True, signature=signature,
        request_sha256=sha(output / 'request.json'), cases=entries, region_bindings=regions,
        case_count=len(entries), records=cohort['records'], observed_P=cohort['observed_P'],
        unobserved_U=cohort['unobserved_U'], debug=False, actual_CT=True,
        zero_P_cases=sum(not any(row['target'] for row in rows) for rows in cohort['by_case'].values()),
        CPU_process_workers=workers, torch_threads_per_worker=1,
        aggregate_peak_RSS_bytes=aggregate_peak, aggregate_RSS_budget_bytes=rss_bytes,
        elapsed_seconds=time.perf_counter() - start, preserved_files=preserved,
        original_historical_upper_builder=True, neural_forward_executed=False,
        training_started=False, optimizer_updates=0, all_original_candidates_preserved=True))
    write_new(output / 'index.json', index)
    os.chmod(output / 'index.json', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return dict(index=str(output / 'index.json'), index_sha256=sha(output / 'index.json'),
        case_count=len(entries), records=cohort['records'], receipt=index)


class UpperGeometryCache:
    """Read-only exact common geometry for a separately loaded actual model."""
    def __init__(self, root, bundle, inventory_path):
        self.root = _directory(root)
        self.inventory_path = _regular(inventory_path)
        self.index_path = _regular(self.root / 'index.json')
        self.index_sha256 = sha(self.index_path)
        self.index = _read(self.index_path)
        _check_seal(self.index)
        inventory = _read(self.inventory_path)
        self.cohort = validate_cohort(inventory, debug=False)
        self.signature = _signature(bundle, self.inventory_path, inventory, self.cohort)
        index = self.index
        if (index.get('format') != FORMAT or index.get('complete') is not True
                or index.get('debug') is not False or index.get('actual_CT') is not True
                or index.get('signature') != self.signature
                or index.get('case_count') != 21 or len(index.get('cases', [])) != 21
                or index.get('records') != self.cohort['records']
                or index.get('observed_P') != self.cohort['observed_P']
                or index.get('unobserved_U') != self.cohort['unobserved_U']
                or [entry.get('case_id') for entry in index['cases']] != self.cohort['case_ids']
                or sha(_regular(self.root / 'request.json')) != index.get('request_sha256')):
            raise ValueError('Complete native upper cache signature/cohort/config/source/bank differs')
        request = _read(self.root / 'request.json'); _check_seal(request)
        if request.get('signature') != self.signature:
            raise ValueError('Native upper cache request signature differs')
        self.entries = {entry['case_id']: entry for entry in index['cases']}
        for case, entry in self.entries.items():
            rows = _query(self.cohort['by_case'][case])
            self._verify_entry(entry, rows)
        _verify_raw(self.signature['raw_inputs'])
        _verify_regions(index['region_bindings'])
        self._input_stats = {path: _stat_identity(path) for path in
                             _input_paths(self.signature, index['region_bindings'])}
        self.receipt = dict(format=FORMAT, index=str(self.index_path), index_sha256=self.index_sha256,
            signature_sha256=canonical_hash(self.signature), case_count=21,
            records=self.cohort['records'], read_only=True, mmap='MAP_PRIVATE copy on write',
            original_historical_upper_builder=True, full_case=True, model_tensors_cached=False,
            optimizer_updates=0, cache_hits=0, initial_full_input_SHA256_verified=True,
            per_case_input_identity_guard='device/inode/size/mtime_ns/ctime_ns after full SHA admission',
            final_full_input_SHA256_verified=False)

    def _guard_inputs(self):
        if any(_stat_identity(path) != identity for path, identity in self._input_stats.items()):
            raise ValueError('Upper cache admitted raw CT/annotation/region/source input changed')

    def finish(self):
        """Rehash all admitted original bytes before a result can be published."""
        self._guard_inputs()
        if (sha(self.index_path) != self.index_sha256
                or sha(self.inventory_path) != self.signature['native_inventory_sha256']
                or _code() != self.signature['execution_code_sha256']):
            raise ValueError('Upper geometry cache evidence changed before final verification')
        _verify_raw(self.signature['raw_inputs'])
        _verify_regions(self.index['region_bindings'])
        if (sha(self.signature['bank_path']) != self.signature['bank_sha256']
                or sha(self.signature['native_region_config_path']) != self.signature['native_region_config_sha256']
                or _source_proof(self.signature['source_proof']['source']) != self.signature['source_proof']
                or any(sha(_regular(path)) != checksum for path, checksum in self.index.get('preserved_files', {}).items())):
            raise ValueError('Upper cache original baseline/bank/source evidence changed')
        for entry in self.entries.values():
            self._verify_entry(entry, _query(self.cohort['by_case'][entry['case_id']]))
        self._guard_inputs()
        self.receipt['final_full_input_SHA256_verified'] = True
        return copy.deepcopy(self.receipt)

    def _verify_entry(self, entry, rows):
        receipt = entry.get('receipt'); _check_seal(receipt)
        if (receipt.get('format') != PAYLOAD_FORMAT or receipt.get('case_id') != rows[0]['case_id']
                or receipt.get('records') != len(rows) or receipt.get('query_rows') != rows
                or receipt.get('query_rows_sha256') != canonical_hash(rows)
                or receipt.get('signature_sha256') != canonical_hash(self.signature)
                or entry.get('inputs') != _case_inputs(self.signature, self.index['region_bindings'], rows)
                or receipt.get('inputs_sha256') != canonical_hash(entry['inputs'])):
            raise ValueError('Upper cache exact query order/input/bank/config receipt differs')
        path = _relative(self.root, entry['path'])
        expected_name = 'cases/' + hashlib.sha256(rows[0]['case_id'].encode()).hexdigest() + '.pt'
        if (entry['path'] != expected_name or path.stat().st_size != entry.get('payload_bytes')
                or sha(path) != entry.get('payload_sha256')):
            raise ValueError('Upper cache payload size/path/SHA256 differs')
        return path

    def upper_graphs(self, bundle, provider, rows, *, region_output, region_reuse=None):
        import torch
        if (sha(self.index_path) != self.index_sha256
                or sha(self.inventory_path) != self.signature['native_inventory_sha256']
                or _code() != self.signature['execution_code_sha256']):
            raise ValueError('Read-only upper cache inventory/code/manifest changed')
        self._guard_inputs()
        if region_reuse is not None:
            raise ValueError('Production sealed upper cache forbids DEBUG region replacement')
        if not rows or any(tuple(row) != QUERY_FIELDS for row in rows):
            raise ValueError('Exact target-free QUERY_FIELDS and complete ordered case required')
        case = rows[0]['case_id']
        if case not in self.entries:
            raise ValueError('Upper cache case absent; rebuilding/fallback is forbidden')
        query = _query(rows)
        expected = _query(self.cohort['by_case'][case])
        if query != expected:
            raise ValueError('Upper cache query order/donor/center/whole-case coverage differs')
        if (bundle.receipt.get('debug') is not False or bundle.config['graph'] != self.signature['graph']
                or bundle.config['ct_clip'] != self.signature['ct_clip']
                or bundle.config['cache']['source_pad'] != self.signature['source_pad']
                or bundle.scope != self.signature['scope']
                or Path(bundle.source).resolve() != Path(self.signature['source_proof']['source'])
                or Path(bundle.baseline).resolve() != Path(self.signature['baseline'])
                or bundle.prototype_bank.fingerprint() != self.signature['bank_fingerprint']
                or list(bundle.prototype_bank.training_case_ids) != self.signature['bank_training_case_ids']
                or sha(_regular(self.signature['bank_path'])) != self.signature['bank_sha256']
                or canonical_hash(bundle.receipt['baseline_proof']) != self.signature['baseline_proof_sha256']):
            raise ValueError('Upper cache loaded bundle bank/config/source/scope differs')
        if _source_proof(bundle.source) != self.signature['source_proof']:
            raise ValueError('Upper cache actual archived source proof changed')
        if (provider.ds.meta['config']['donor_max_diameter_mm'] != self.signature['donor_max_diameter_mm']
                or _raw_bindings(provider.ds.meta, self.cohort) != self.signature['raw_inputs']
                or provider.ds.meta['split']['inner_train'] != self.cohort_training):
            raise ValueError('Upper cache actual native provider input/population differs')
        entry = self.entries[case]
        path = self._verify_entry(entry, query)
        if not _private_mmap():
            raise ValueError('Upper cache requires private copy-on-write mmap tensors')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if (not isinstance(payload, dict) or set(payload) != {'format', 'graph', 'prototype', 'audit', 'receipt'}
                or payload['format'] != PAYLOAD_FORMAT or payload['receipt'] != entry['receipt']
                or canonical_hash(payload['audit']) != entry['receipt']['audit_sha256']):
            raise ValueError('Loaded immutable original upper graph receipt/audit differs')
        _validate_graphs(payload['graph'], payload['prototype'], payload['audit'], query, self.signature)
        if sha(path) != entry['payload_sha256'] or sha(self.index_path) != self.index_sha256:
            raise ValueError('Upper geometry payload changed while mmap loading')
        self._guard_inputs()
        self.receipt['cache_hits'] += 1
        return payload['graph'], payload['prototype'], payload['audit']

    @property
    def cohort_training(self):
        return self.signature['training_case_ids']
