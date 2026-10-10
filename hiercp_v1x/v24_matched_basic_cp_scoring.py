"""Own trained v24 BEST on every unchanged historical Basic CP placement.

The source bank remains immutable. Only a separately sealed 128-logit overlay
is published. Every source has its original patient's complete observed-P
coordinate context plus the original ordered 128 legal CP coordinates. No P/U
target, utility, recipient lesion node or recipient occupied-distance enters
the learned local/upper model. The explicitly allowed donor is that patient's
selected source lesion, matching the historical Basic source schedule.
"""
from __future__ import annotations

import ast
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
import time

FORMAT = 'v24_own_BEST_scores_on_unchanged_Basic_81case642source128positions_v1'
ADMISSION = 'historical_Basic_outer_train_same_patient_donor_only_v1'
EXPECTED_ENTRIES = 642
EXPECTED_CASES = 81
EXPECTED_CANDIDATES = 128
PAYLOAD_FIELDS = frozenset(('source_data', 'source_mask', 'anchor_offset',
    'candidate_centers', 'candidate_raw_centers', 'scores',
    'source_component', 'source_diameter_mm'))


def _sha(path):
    from .v24_nnunet_cp import sha
    return sha(path)


def _stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _sealed_json(path, value):
    from .v24_nnunet_cp import new_json
    new_json(path, value)


def _clone_admission(function, substitutions, extra_globals=None, input_absences=None):
    """Change only explicitly listed admission comparisons, never equations."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    patterns = {ast.dump(ast.parse(key, mode='eval').body): key for key in substitutions}
    counts = {key: 0 for key in substitutions}
    absent_patterns = {ast.dump(ast.parse(key, mode='eval').body): key for key in input_absences or {}}
    absence_counts = {key: 0 for key in input_absences or {}}
    class Substitute(ast.NodeTransformer):
        def visit_Compare(self, node):
            key = patterns.get(ast.dump(node))
            if key is not None:
                counts[key] += 1
                return ast.copy_location(ast.parse(substitutions[key], mode='eval').body, node)
            return self.generic_visit(node)
        def visit_BinOp(self, node):
            key = absent_patterns.get(ast.dump(node))
            if key is not None:
                absence_counts[key] += 1
                return ast.copy_location(ast.parse(input_absences[key], mode='eval').body, node)
            return self.generic_visit(node)
    tree = Substitute().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise ValueError('Actual original self-CP admission expressions changed: ' + str(counts))
    if any(count != 1 for count in absence_counts.values()):
        raise ValueError('Exactly one same-patient other-lesion occupancy input removal required')
    namespace = dict(function.__globals__)
    namespace.update(extra_globals or {})
    filename = inspect.getsourcefile(function)
    exec(compile(ast.fix_missing_locations(tree), filename + ':matched_Basic_self_CP', 'exec'), namespace)
    proof = dict(original_source_sha256=_sha(filename),
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        admission_comparisons_changed=counts, neural_or_geometry_expressions_changed=False,
        same_patient_recipient_other_lesion_input_removals=absence_counts,
        retained_original_source_raw_equations=True)
    return namespace[function.__name__], proof


def matched_builders(outer_train):
    """Explicit inference-only self donor; retain the train-only prototype bank."""
    from .v24_inputs import build_local_record
    from .v24_geometry import build_upper_graphs
    outer = frozenset(outer_train)
    if len(outer) != 105:
        raise ValueError('Historical full105 training patients required for self-CP admission')
    local, local_proof = _clone_admission(build_local_record, {
        'recipient.case_id == donor_case.paths.case_id':
        'recipient.case_id != donor_case.paths.case_id'})
    upper, upper_proof = _clone_admission(build_upper_graphs, {
        'donor_case.paths.case_id not in training':
        'donor_case.paths.case_id not in _matched_outer_train',
        'recipient.case_id == donor_case.paths.case_id':
        'recipient.case_id != donor_case.paths.case_id'}, {'_matched_outer_train': outer},
        input_absences={'(donor_case.label == 2) & ~donor_source.full_mask':
            'np.zeros_like(donor_source.full_mask, dtype=bool)'})
    def admit(recipient, donor_case):
        if recipient.case_id != donor_case.paths.case_id or recipient.case_id not in outer:
            raise ValueError('Only original outer-training same-patient Basic sources admitted')
    def local_same(recipient, donor_case, *args, **kwargs):
        admit(recipient, donor_case)
        return local(recipient, donor_case, *args, **kwargs)
    def upper_same(recipient, donor_case, *args, **kwargs):
        admit(recipient, donor_case)
        graph, prototype, audit = upper(recipient, donor_case, *args, **kwargs)
        audit.update(matched_Basic_admission=ADMISSION, donor_equals_recipient=True,
            prototype_bank_scope_unchanged='original inner_train84 only',
            same_patient_donor_other_lesion_occupancy_used=False,
            selected_source_annotation_available=True)
        return graph, prototype, audit
    return local_same, upper_same, dict(format=ADMISSION, local=local_proof, upper=upper_proof)


def validate_payload(payload):
    import numpy as np
    if set(payload) != PAYLOAD_FIELDS:
        raise ValueError('Exact historical preprocessed Basic entry schema required; raw-target entries refused')
    raw, mapped, scores = (np.asarray(payload[key]) for key in
        ('candidate_raw_centers', 'candidate_centers', 'scores'))
    mask, image, anchor = (np.asarray(payload[key]) for key in
        ('source_mask', 'source_data', 'anchor_offset'))
    component, diameter = (np.asarray(payload[key]) for key in
        ('source_component', 'source_diameter_mm'))
    if (raw.shape != (128, 3) or mapped.shape != raw.shape or raw.dtype.kind not in 'iu'
            or mapped.dtype.kind not in 'iu' or (raw < 0).any() or (mapped < 0).any()
            or len(np.unique(raw, axis=0)) != 128 or scores.shape != (128,)
            or scores.dtype.kind != 'f' or not np.isfinite(scores).all()
            or mask.ndim != 3 or mask.dtype.kind not in 'bu' or not np.isin(mask, (0, 1)).all()
            or not mask.any() or image.shape != (1, *mask.shape) or image.dtype.kind != 'f'
            or not np.isfinite(image).all() or anchor.shape != (3,) or anchor.dtype.kind not in 'iu'
            or (anchor < 0).any() or (anchor >= mask.shape).any()
            or component.shape != (1,) or component.dtype.kind not in 'iu' or component[0] < 1
            or diameter.shape != (1,) or not np.isfinite(diameter).all() or not 0 < diameter[0] <= 20.00001):
        raise ValueError('Complete real historical source and exact128 legal placement payload required')
    return int(component[0])


def admit_source_bank(source_bank, split):
    """Bind every original source byte, membership and entry order once."""
    from .v24_nnunet_cp import read, validate_split
    validate_split(split)
    if Path(source_bank).is_symlink():
        raise ValueError('Historical source bank must be a regular immutable directory')
    root = Path(source_bank).resolve(strict=True)
    index = root / 'index.json'
    metadata = read(index)
    entries = metadata.get('entries_by_case', {})
    if (len(entries) != EXPECTED_CASES or sum(map(len, entries.values())) != EXPECTED_ENTRIES
            or not set(entries) <= set(split['outer_train'])
            or any(not isinstance(names, list) or not names for names in entries.values())
            or metadata.get('candidate_count') != 128 or metadata.get('cp_probability') != .5
            or metadata.get('intensity_scale_range') != [.95, 1.05]
            or metadata.get('intensity_shift_range_hu') != [-5., 5.]
            or 'paste_contract' in metadata
            or (metadata.get('maximum_diameter_mm') is not None and metadata['maximum_diameter_mm'] != 20)):
        raise ValueError('Exact historical Basic81/642/128, probability0.5 and HU contract required')
    import numpy as np
    proofs = {}
    for case, names in entries.items():
        if len(names) != len(set(names)):
            raise ValueError('Historical source order contains duplicate entries')
        seen = set()
        for relative in names:
            path = root / relative
            if (Path(relative).is_absolute() or path.is_symlink()
                    or not path.resolve(strict=True).is_relative_to(root) or relative in proofs):
                raise ValueError('Historical source reference escapes bank or duplicates another patient')
            before = _stat(path)
            with np.load(path, allow_pickle=False) as loaded:
                payload = {key: loaded[key] for key in loaded.files}
                component = validate_payload(payload)
                from .v24_inputs import array_digest
                non_score = {key: array_digest(value) for key, value in payload.items() if key != 'scores'}
            if component in seen:
                raise ValueError('Historical patient repeats a source component')
            seen.add(component)
            checksum = _sha(path)
            if _stat(path) != before:
                raise ValueError('Original Basic source changed during admission')
            declared = metadata.get('entry_sha256', {}).get(relative)
            if declared is not None and declared != checksum:
                raise ValueError('Original Basic entry differs from its declared SHA')
            proofs[relative] = dict(case_id=case, source_component=component,
                sha256=checksum, stat_identity=list(before), non_score_arrays_sha256=non_score)
    return metadata, dict(root=str(root), index_sha256=_sha(index),
        index_stat_identity=list(_stat(index)), entries=proofs,
        complete_cases=81, complete_sources=642, complete_positions=82176)


def guard_source_bank(proof):
    root = Path(proof['root'])
    if _stat(root / 'index.json') != tuple(proof['index_stat_identity']):
        raise ValueError('Admitted historical Basic index changed')
    for relative, entry in proof['entries'].items():
        if _stat(root / relative) != tuple(entry['stat_identity']):
            raise ValueError('Admitted historical Basic source changed: ' + relative)


@dataclass(frozen=True)
class MatchedPlan:
    case_id: str
    partition: str
    entry: str
    donor_component: int
    record_ids: tuple
    query_rows: tuple
    observed_P: int


def matched_plan(case_id, entry, payload, population):
    """Keep all observed-P coordinates, strip every supervision field."""
    component = validate_payload(payload)
    native = population.case(case_id, 128)
    originals = native.query_rows
    positive = [originals[index] for index in native.positive_indices]
    rows = []
    for index, row in enumerate(positive):
        rows.append(dict(id=f'matched_Basic:{entry}:P:{row["id"]}', case_id=case_id,
            center=list(row['center']), donor_case_id=case_id, donor_component=component))
    for index, center in enumerate(payload['candidate_raw_centers']):
        rows.append(dict(id=f'matched_Basic:{entry}:CP:{index:03d}', case_id=case_id,
            center=[int(value) for value in center], donor_case_id=case_id, donor_component=component))
    from .v24_inputs import query_inputs
    queries = query_inputs(rows)
    return MatchedPlan(case_id, native.partition, entry, component,
        tuple(row['id'] for row in queries), queries, len(positive))


def source_batches(entries_by_case, partition_cases, physical=4):
    """Complete source coverage with disjoint patients and measured B4 capacity."""
    if physical != 4:
        raise ValueError('Matched scoring preserves the measured physical patient batch4')
    queues = {case: deque(entries_by_case[case]) for case in partition_cases if case in entries_by_case}
    result = []
    while queues:
        total = sum(map(len, queues.values()))
        count = min(4, len(queues), total if total != 5 else 3)
        selected = sorted(queues, key=lambda case: (-len(queues[case]), case))[:count]
        batch = [(case, queues[case].popleft()) for case in selected]
        for case in selected:
            if not queues[case]:
                del queues[case]
        result.append(batch)
    # Rebalance an actual source into a singleton tail without duplicating a
    # patient. Scheduling changes; original source lists and queries do not.
    for batch in result:
        if len(batch) == 1:
            candidates = [(other, index) for other in result if len(other) >= 3
                for index, (case, _) in enumerate(other) if case != batch[0][0]]
            if not candidates:
                raise ValueError('Complete source multiplicity cannot form parallel patient batches without duplication')
            other, index = candidates[-1]
            batch.append(other.pop(index))
    if any(not 2 <= len(batch) <= 4 or len({case for case, _ in batch}) != len(batch) for batch in result):
        raise ValueError('Complete disjoint source batching must retain parallel2+ patients and capacity4')
    return result


class MatchedInputProvider:
    """Persistent parallel CPU construction; full local graphs stream by chunk."""
    recipient_GT_used_in_forward = False

    def __init__(self, plans, builder, budget, *, workers=4):
        if workers != 4:
            raise ValueError('Original four CPU workers required')
        self.ds = SimpleNamespace(rows=[row for plan in plans for row in plan.query_rows])
        self.builder, self.budget, self.workers = builder, budget, workers
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='v24_matched_CPU')
        self.closed = False
        self.profile_rows = dict(chunks=0, observations=0, CPU_seconds=0., workers=workers)

    def get(self, ids, *, epoch):
        if self.closed or epoch != 29 or not ids or len(ids) != len(set(ids)):
            raise ValueError('Complete unique original observations and fixed evaluation view29 required')
        from .v24_input_runtime import _ChunkSourceReuse, _materialize_pair, collate
        began = time.perf_counter()
        def record(index):
            if type(index) is not int or not 0 <= index < len(self.ds.rows):
                raise ValueError('Matched local observation index outside complete source queries')
            return self.builder(self.ds.rows[index])
        records = list(self.executor.map(record, ids))
        reuse = _ChunkSourceReuse(records, RSS_guard=self.budget.check)
        try:
            items = list(self.executor.map(lambda item:
                (_materialize_pair(item[1], epoch=epoch, source_reuse=reuse), item[0]), zip(ids, records)))
        finally:
            reuse.close()
        batch = collate(items)
        del records, items
        self.budget.check()
        self.profile_rows['chunks'] += 1
        self.profile_rows['observations'] += len(ids)
        self.profile_rows['CPU_seconds'] += time.perf_counter() - began
        return batch

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True


def score_current_matched_basic_bank(*, pin_path, inventory_path, baseline_preprocessed,
        input_cache, source_bank, output, gpu, stunet_checkpoint=None):
    """Production: full642 sources/82176 CP queries; frozen own completed BEST."""
    from tools.local_cnn_device import select
    select(gpu)
    import numpy as np
    import torch
    from . import v24_factory, v24_memory_runtime, v24_prefetch_runtime, v24_input_runtime, v24_hash_runtime
    from .v24_nnunet_cp import (admit_pin, CURRENT_PIN_FORMAT, _admit_arm_gpu,
        require_project_budget, read, validate_baseline, pin_parameters,
        _inspect_completed_current, ROOT)
    from .u_bridge_training import digest, capture_rng, restore_rng
    from .historical_evaluation import ResourceBudget
    from .v24_training import V24Scorer
    from .v24_inputs import prepare_donor, immutable_array, array_digest
    from .v24_geometry import build_recipient_regions
    from hiercp_v22.data import sources
    from hiercp.common import stable_case_seed
    from hiercp.region import REGION_CACHE_SEED_SALT
    pin, request = admit_pin(pin_path, inventory_path)
    if pin.get('format') != CURRENT_PIN_FORMAT:
        raise ValueError('Only completed current full40 own-arm BEST is allowed')
    _admit_arm_gpu(pin, gpu)
    if (Path(input_cache).resolve(strict=True) != Path(pin['input_cache'])
            or (None if stunet_checkpoint is None else str(Path(stunet_checkpoint).resolve(strict=True))) != pin['stunet_checkpoint']):
        raise ValueError('Actual trained model input/STU provenance differs')
    require_project_budget()
    if torch.cuda.is_initialized():
        raise RuntimeError('Matched frozen scoring requires fresh pre-CUDA runtime installation')
    v24_memory_runtime.install_memory_runtime(v24_factory)
    v24_hash_runtime.install()
    v24_prefetch_runtime.install_runtime(v24_memory_runtime)
    v24_input_runtime.install_runtime(pin_final_outputs=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'A6000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Actual assigned singleton A6000 required; no CPU/model fallback')
    free, total = torch.cuda.mem_get_info()
    if min(free, total) <= 40 * 2**30:
        raise MemoryError('Original full trained-model40GiB CUDA budget unavailable')
    torch.cuda.set_per_process_memory_fraction(40 * 2**30 / total)
    torch.set_num_threads(1)
    meta = read(inventory_path)
    baseline = validate_baseline(baseline_preprocessed, meta['split'])
    metadata, source_proof = admit_source_bank(source_bank, meta['split'])
    root = Path(output).resolve()
    protected = [Path(pin['source_output']), Path(pin['source_code']), Path(input_cache),
        Path(baseline_preprocessed), Path(source_bank)]
    if root.exists() or any(root.is_relative_to(path.resolve(strict=True))
            or path.resolve(strict=True).is_relative_to(root) for path in protected):
        raise FileExistsError('Fresh matched scoring namespace must not overlap original inputs/results')
    root.mkdir(parents=True)
    args = SimpleNamespace(native_experiment=Path(request['native_experiment']), inventory=Path(inventory_path),
        input_cache=Path(input_cache), stunet_checkpoint=stunet_checkpoint)
    budget = ResourceBudget(40 * 2**30, 64 * 2**30)
    net, trained_scorer, population, config, contract, close = v24_factory.build_runtime(args, request['config'], budget)
    providers = {}
    try:
        if contract != pin['model_contract'] or population.manifest() != pin['population']:
            raise ValueError('Original full trained model/population differs')
        saved = torch.load(pin['selected']['best_path'], map_location='cpu', weights_only=False, mmap=True)
        if (saved['content_sha256'] != pin['selected']['best_content_sha256']
                or digest(saved['model']) != pin['selected']['best_model_sha256']):
            raise ValueError('Actual own-arm BEST tensor identity differs')
        net.load_state_dict(saved['model'], strict=True)
        del saved
        net.eval().requires_grad_(False)
        if sum(parameter.numel() for parameter in net.parameters()) != pin_parameters(pin):
            raise ValueError('Full trained own-arm model parameter count differs')
        calibration = read(Path(pin['source_output']) / 'calibration.json')
        chunk = calibration['selected_physical_candidate_batch']
        if chunk != (32 if gpu == 5 else 64):
            raise ValueError('Original measured full-model local candidate batch differs')
        inputs = trained_scorer.geometry.memory_guard.__self__
        local, upper, admission = matched_builders(meta['split']['outer_train'])
        # Read existing proven CT/organ/prototype inputs. Every matched local and
        # upper graph is freshly built; old ranking128 geometry is never reused.
        plans, payloads = {}, {}
        for case, names in metadata['entries_by_case'].items():
            for relative in names:
                with np.load(Path(source_proof['root']) / relative, allow_pickle=False) as loaded:
                    payload = {key: loaded[key] for key in loaded.files}
                plan = matched_plan(case, relative, payload, population)
                plans[relative] = plan
                # Only small original coordinate/scalar payloads remain resident.
                payloads[relative] = {key: payload[key] for key in
                    ('candidate_raw_centers', 'candidate_centers', 'source_diameter_mm')}
        active = {}
        def build_local(row):
            case, source, prepared, regions, binding = active[(row['case_id'], row['donor_component'])]
            return local(case['context'], case['donor_case'], source, prepared, row,
                config=inputs.graph_config, seed=config['seed'],
                ct_clip=inputs.ct_clip, scope_contract=inputs.bundle.scope['contract_sha256'],
                donor_mask_sha256=source.v24_mask_sha256)
        for partition in ('inner_train', 'inner_val'):
            part = [plan for plan in plans.values() if plan.partition == partition]
            providers[partition] = MatchedInputProvider(part, build_local, budget)
        class Geometry:
            recipient_GT_used_in_forward = False
            def __call__(self, plan, provider):
                return active_graphs[plan.entry]
        scorer = V24Scorer(net, providers, Geometry(), physical_candidate_batch=chunk,
            checkpoint_local_chunks=False, amp=bool(config['training']['amp']), budget=budget,
            prefetch_cpu_chunks=True, pin_cpu_batches=True)
        source_names = set(request['source']) | {
            'hiercp_v1x/v24_matched_basic_cp_scoring.py', 'hiercp_v1x/v24_nnunet_cp.py',
            'hiercp_v1x/v24_memory_runtime.py', 'hiercp_v1x/v24_prefetch_runtime.py',
            'hiercp_v1x/v24_input_runtime.py', 'hiercp_v1x/v24_hash_runtime.py',
            'hiercp_v1x/v23_training.py', 'hiercp_v1x/v24_geometry.py', 'hiercp_v1x/v24_inputs.py'}
        source_checksums = {name: _sha(ROOT / name) for name in source_names}
        cp_contract = dict(eligible_patients=81, total_sources=642, candidates_per_source=128,
            total_scored_candidates=82176, donor_policy='same_patient',
            source_selection='same_as_historical_Basic', candidate_payload_preserved=True,
            non_score_payload_preserved=True, paste_contract='historical_Basic_preprocessed_patch',
            cp_probability=.5, selection='gnn_argmax',
            source_policy='same_patient_uniform_original_sources_per_visit',
            paste_policy='historical_preprocessed_patch_hard_paste',
            selection_policy='argmax', train_cases=105, validation_cases=26,
            physical_batch=2, patch=[128, 128, 128], epochs=250)
        _sealed_json(root / 'request.json', dict(format=FORMAT, pin=pin, pin_sha256=_sha(pin_path),
            source_bank_proof=source_proof, baseline=baseline, admission=admission,
            split=copy.deepcopy(meta['split']), cp_contract=cp_contract,
            scoring_sources_sha256=source_checksums, own_BEST=True, model_frozen=True,
            recipient_GT_used_in_forward=False, full_P_coordinate_context=True,
            original128_CPvalid_positions=True, source_policy='original patient source uniform schedule',
            original_paste_contract='historical preprocessed source patch', physical_patient_batch=4,
            physical_candidate_batch=chunk, CPU_workers=4, RSS_bytes=64 * 2**30,
            CUDA_bytes=40 * 2**30, original_parameter_count=pin_parameters(pin), debug=False))
        initial_rng = capture_rng()
        rows = {}
        import pickle
        import blosc2
        mapping_plans = read(Path(baseline['preprocessed']) / (baseline['plans_name'] + '.json'))
        from hiercp_v22.bank import map_center
        directory = Path(baseline['preprocessed']) / baseline['data_identifier']
        from threading import RLock
        collections, collection_lock = OrderedDict(), RLock()
        collection_bytes = 0
        # This bounds only retained CPU arrays. All642 source components and all
        # complete local/upper graphs are still constructed and scored.
        collection_resident_bytes = 8 * 2**30
        def collection_for(case_id, raw):
            nonlocal collection_bytes
            with collection_lock:
                if case_id in collections:
                    collections.move_to_end(case_id)
                    return collections[case_id][0]
                collection = sources(raw['donor_case'], inputs.config['cache']['source_pad'], 20.)
                retained = (collection.components.nbytes + collection.sizes.nbytes
                    + raw['donor_case'].image.nbytes + raw['donor_case'].label.nbytes)
                while collections and collection_bytes + retained > collection_resident_bytes:
                    _, (_, released) = collections.popitem(last=False)
                    collection_bytes -= released
                if retained <= collection_resident_bytes:
                    collections[case_id] = collection, retained
                    collection_bytes += retained
                return collection
        def prepare_source(item):
            case_id, relative = item
            plan = plans[relative]
            raw = inputs._case(case_id)
            collection = collection_for(case_id, raw)
            found = [index for index, (component, _) in enumerate(collection.entries) if component == plan.donor_component]
            if len(found) != 1:
                raise ValueError('Original historical source component absent from actual raw donor')
            source, diameter = collection[found[0]]
            if not np.isclose(diameter, payloads[relative]['source_diameter_mm'][0], rtol=1e-6, atol=1e-6):
                raise ValueError('Historical source/raw component diameter differs')
            source.full_mask = immutable_array(source.full_mask)
            source.v24_mask_sha256 = array_digest(source.full_mask)
            prepared = prepare_donor(raw['donor_case'], source, config=inputs.graph_config,
                seed=inputs.config['seed'], ct_clip=inputs.ct_clip)
            # This batch contains distinct real patients. Build their independent
            # complete region volumes outside the raw-cache publication lock.
            # The original cache eviction policy and exact region seed stay.
            with inputs._lock:
                regions = inputs._regions.get(case_id)
            if regions is None:
                regions = build_recipient_regions(raw['context'], config=inputs.graph_config,
                    seed=stable_case_seed(42, case_id, REGION_CACHE_SEED_SALT), ct_clip=inputs.ct_clip)
                with inputs._lock:
                    if case_id in inputs._raw_cache:
                        regions = inputs._regions.setdefault(case_id, regions)
                inputs._rss()
            properties_path = directory / (case_id + '.pkl')
            properties_sha = _sha(properties_path)
            with properties_path.open('rb') as stream:
                properties = pickle.load(stream)
            data_path = directory / (case_id + '.b2nd')
            data = blosc2.open(str(data_path), mode='r')
            shape = tuple(data.shape[1:])
            mapped = np.stack([map_center(point, mapping_plans, properties, shape)
                for point in payloads[relative]['candidate_raw_centers']])
            del data
            if not np.array_equal(mapped, payloads[relative]['candidate_centers']):
                raise ValueError('Original CP candidate order/raw-to-preprocessed mapping differs')
            if any(any(value >= extent for value, extent in zip(row['center'], raw['context'].shape))
                    for row in plan.query_rows):
                raise ValueError('Complete matched P/128CP query leaves actual raw CT extent')
            graphs = upper(raw['context'], raw['donor_case'], source, regions, regions,
                plan.query_rows, inputs.bundle.prototype_bank, config=inputs.graph_config,
                ct_clip=inputs.ct_clip, training_case_ids=inputs.training_cases)
            binding = dict(recipient=raw['context'].binding(), donor_case_id=case_id,
                donor_component=source.component_id, donor_mask_sha256=source.v24_mask_sha256,
                donor_CT_sha256=inputs.raw[case_id]['image_sha256'],
                donor_label_sha256=inputs.raw[case_id]['label_sha256'],
                source_bank_entry_sha256=source_proof['entries'][relative]['sha256'],
                raw_centers_sha256=array_digest(payloads[relative]['candidate_raw_centers']),
                preprocessed_centers_sha256=array_digest(payloads[relative]['candidate_centers']),
                raw_source_patch_shape=list(source.patch_mask.shape),
                raw_source_anchor=list(source.anchor_center), raw_source_voxels=source.voxel_count,
                raw_source_patch_slices=[[part.start, part.stop] for part in source.patch_slices],
                raw_source_identity='exact same numbered 6-connected raw label2 component',
                original_non_score_arrays_sha256=source_proof['entries'][relative]['non_score_arrays_sha256'],
                properties_sha256=properties_sha, preprocessed_CT_stat_identity=list(_stat(data_path)),
                original_candidate_order_and_mapping_verified=True)
            return plan, (raw, source, prepared, regions, binding), graphs
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix='v24_matched_geometry') as executor:
            for partition in ('inner_train', 'inner_val'):
                schedule = source_batches(metadata['entries_by_case'], meta['split'][partition])
                for batch in schedule:
                    guard_source_bank(source_proof)
                    inputs.guard_source()
                    began = time.perf_counter()
                    prepared_batch = list(executor.map(prepare_source, batch))
                    active.clear()
                    active_graphs = {}
                    batch_plans = []
                    for plan, item, graphs in prepared_batch:
                        active[(plan.case_id, plan.donor_component)] = item
                        active_graphs[plan.entry] = graphs
                        batch_plans.append(plan)
                    rng = capture_rng()
                    try:
                        with torch.no_grad():
                            result = scorer(batch_plans, epoch=29, training=False)
                        torch.cuda.synchronize()
                        budget.check()
                        for score, plan in zip(result.scores, batch_plans):
                            values = score.detach().float().cpu().numpy()[plan.observed_P:]
                            if values.shape != (128,) or not np.isfinite(values).all():
                                raise ValueError('Actual frozen GNN must score every ordered128 original CP query')
                            binding = active[(plan.case_id, plan.donor_component)][4]
                            row = dict(case_id=plan.case_id, source_component=plan.donor_component,
                                entry=plan.entry,
                                original_sha256=source_proof['entries'][plan.entry]['sha256'],
                                scores=values.tolist(), selected_index=int(np.argmax(values)),
                                joint_observed_P=plan.observed_P, joint_candidates=len(plan.record_ids),
                                all_original_P_in_context=True, all_original128_CP_scored=True,
                                recipient_GT_used_in_forward=False, model_sha256=pin['selected']['best_model_sha256'],
                                input_binding=binding, workload=result.workload)
                            _sealed_json(root / 'scores' / (Path(plan.entry).stem + '.json'), row)
                            rows[plan.entry] = row
                        del result
                    finally:
                        restore_rng(rng)
                    active.clear()
                    active_graphs.clear()
                    prepared_batch.clear()
                    print(json.dumps(dict(phase='matched_Basic_current_BEST_scores', GPU=gpu,
                        completed_sources=len(rows), total_sources=642, complete_CP_positions=len(rows) * 128,
                        actual_patient_batch=len(batch), physical_capacity=4,
                        seconds=time.perf_counter() - began, production_optimizer_updates=0)), flush=True)
        guard_source_bank(source_proof)
        if len(rows) != 642 or set(rows) != set(source_proof['entries']):
            raise ValueError('Full historical642 source coverage required; no subset')
        if digest(capture_rng()) != digest(initial_rng) or digest(net.state_dict()) != pin['selected']['best_model_sha256']:
            raise ValueError('Frozen current BEST model or original RNG changed')
        if any(_sha(ROOT / name) != checksum for name, checksum in source_checksums.items()):
            raise ValueError('Actual matched scoring implementation changed')
        if _inspect_completed_current(pin['source_output'], pin['source_code'], inventory_path, gpu=gpu,
                input_cache=input_cache, stunet_checkpoint=stunet_checkpoint) != pin:
            raise ValueError('Actual completed own-arm BEST provenance changed')
        from .contracts import canonical_hash
        overlay = dict(format=FORMAT, complete=True, debug=False, source_bank=source_proof,
            split=copy.deepcopy(meta['split']), cp_contract=cp_contract,
            pin=pin, pin_sha256=_sha(pin_path), model_sha256=pin['selected']['best_model_sha256'],
            entries=rows, entries_by_case=copy.deepcopy(metadata['entries_by_case']),
            complete_cases=81, complete_sources=642, complete_positions=82176,
            only_CP_field_changed='scores', original_non_score_payloads_preserved=True,
            original_source_schedule_and_paste_contract_preserved=True,
            full_P_context=True, recipient_GT_used_in_forward=False, production_optimizer_updates=0,
            model_and_RNG_unchanged=True, scoring_sources_sha256=source_checksums)
        overlay['content_sha256'] = canonical_hash(overlay)
        _sealed_json(root / 'score_overlay.json', overlay)
        return root / 'score_overlay.json'
    finally:
        for provider in providers.values():
            provider.close()
        close()
