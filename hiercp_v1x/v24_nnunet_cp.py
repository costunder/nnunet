"""Frozen actual v2.3 GNN placement followed by native nnUNet training.

This historical annotation-conditioned arm keeps the original full GNN and
native P+128U assignments. It does not load a segmentation model into the GNN.
Only the 105 segmentation-training patients may enter placement construction.
The 26 segmentation-validation patients use the ordinary nnUNet loader.
"""
from __future__ import annotations

import copy
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

FORMAT = 'v24_frozen_v23_GNN_native_nnunet_CP_v1'
SCORE_FORMAT = 'v24_frozen_original_GNN_complete105_P_plus128U_scores_v1'
CP_SELECTION = 'stable_highest_GNN_score_among_original128_hard_CP_eligible_U_v1'
PIN_FORMAT = 'v24_selected_historical_v23_best_v1'
TRAINER = 'nnUNetTrainer_250epochs_FrozenV23CP'
PLANS = 'nnUNetResEncUNetMPlans'
PARAMETERS = 10434532
PROJECT_CPU_CORES = 4
PROJECT_RSS_BYTES = 48 * 2**30
INPUT_RESIDENT_BYTES = 16 * 2**30
GEOMETRY_RESIDENT_BYTES = 8 * 2**30
ROOT = Path(__file__).resolve().parents[1]
FILES = ('hiercp_v1x/v24_nnunet_cp.py', 'tools/run_v24_nnunet_cp.py',
         'custom_trainers/nnUNetTrainer_FrozenV23CP.py')


def resource_contract():
    return dict(logical_CPU_cores=PROJECT_CPU_CORES,process_and_children_RSS_bytes=PROJECT_RSS_BYTES,
        data_augmentation_workers=PROJECT_CPU_CORES,Torch_threads=1,
        input_resident_bytes=INPUT_RESIDENT_BYTES,geometry_resident_bytes=GEOMETRY_RESIDENT_BYTES,
        host_available_resources_are_not_project_allocation=True)


def process_tree_rss():
    import psutil
    process=psutil.Process()
    return process.memory_info().rss+sum(child.memory_info().rss for child in process.children(recursive=True)
        if child.is_running())


def require_project_budget():
    import psutil
    affinity=psutil.Process().cpu_affinity()
    if len(affinity)!=PROJECT_CPU_CORES:
        raise ValueError('Launcher must assign exactly four logical CPU cores; host capacity is not our allocation')
    rss=process_tree_rss()
    if rss>PROJECT_RSS_BYTES:
        raise MemoryError('GPU1 project process+DA children exceed48GiB; no host-RAM expansion or model reduction')
    return dict(**resource_contract(),actual_affinity=affinity,actual_process_and_children_RSS_bytes=rss)


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular immutable input required: ' + str(path))
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def new_json(path, document):
    from .transition_v1_data import _publish_new_json
    _publish_new_json(Path(path), document)


def validate_split(split):
    names = ('inner_train', 'inner_val', 'outer_train', 'outer_val')
    for name, count in zip(names, (84, 21, 105, 26)):
        values = split.get(name)
        if (not isinstance(values, list) or len(values) != count
                or len(set(values)) != count or any(not isinstance(x, str) or not x for x in values)):
            raise ValueError('Complete actual 84/21/105/26 split required: ' + name)
    sets = {name: set(split[name]) for name in names}
    if (sets['inner_train'] & sets['inner_val']
            or sets['inner_train'] | sets['inner_val'] != sets['outer_train']
            or sets['outer_train'] & sets['outer_val']):
        raise ValueError('Patient leakage or incomplete nested partition')
    return copy.deepcopy(split)


def validate_pin(pin):
    if (pin.get('format') != PIN_FORMAT or pin.get('all_three_paused') is not True
            or pin.get('recipient_annotation_exposed') is not True
            or pin.get('selection_metric') != 'fixed full128 per-P patient-macro MRR, top1, negativepairloss'):
        raise ValueError('Actual best of all three v2.3 full-validation checkpoints must be pinned')
    comparisons = pin.get('candidates')
    if (not isinstance(comparisons, list) or len(comparisons) != 3
            or {row.get('gpu') for row in comparisons} != {1, 5, 6}):
        raise ValueError('All three actual full-validation selections required')
    for row in comparisons:
        if (row['best_record']['selected_by'] != 'full_validation_only'
                or row['best_record']['candidate_universe'] != 'all native P + fixed128U'
                or len(row['best_record']['selection_key']) != 3
                or any(not isinstance(x,(float,int)) or not math.isfinite(x) for x in row['best_record']['selection_key'])):
            raise ValueError('Original whole128 full-validation best criterion required')
        for key in ('best_file_sha256','best_content_sha256','best_model_sha256','identity_sha256'):
            value=row.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Actual best checkpoint SHA evidence required: '+key)
    if pin.get('selected') != max(comparisons,key=lambda row:tuple(row['best_record']['selection_key'])):
        raise ValueError('Pinned selection is not the actual best of all three')
    return copy.deepcopy(pin)


def admit_pin(path, inventory):
    pin = validate_pin(read(path))
    selected=pin['selected'];training=Path(selected['best_path']).parent
    if sha(selected['best_path']) != selected['best_file_sha256']:
        raise ValueError('Pinned best checkpoint changed')
    request = read(training.parent / 'request.json')
    identity=read(training/'training_identity.json')
    from .u_bridge_training import digest
    from .contracts import canonical_hash
    if (identity['identity_sha256'] != digest(identity['binding'])
            or request['request_sha256'] != canonical_hash({k:v for k,v in request.items() if k!='request_sha256'})
            or identity['binding']['identity']['request_sha256'] != request['request_sha256']):
        raise ValueError('Pinned original model ownership/request checksum differs')
    if identity['identity_sha256']!=selected['identity_sha256'] or request['inventory_sha256'] != sha(inventory):
        raise ValueError('Best model and actual native assignment ownership differ')
    pin['admitted_source_experiment']=str(training.parent)
    pin['admitted_source_request_sha256']=request['request_sha256']
    return pin, request


def validate_baseline(preprocessed, split):
    """Reuse the actual comparison plans, while reporting their fitting scope."""
    split = validate_split(split)
    pre = Path(preprocessed).resolve(strict=True)
    if pre.name != 'Dataset730_LiverOnlineCP_OF0' or pre.is_symlink():
        raise ValueError('Actual historical Dataset730 preprocessing required')
    native_split = read(pre / 'splits_final.json')
    if native_split != [dict(train=split['outer_train'], val=split['outer_val'])]:
        raise ValueError('Historical Basic CP 105/26 patient split differs')
    plans = read(pre / (PLANS + '.json'))
    cfg = plans['configurations']['3d_fullres']
    if cfg['patch_size'] != [128, 128, 128] or type(cfg['batch_size']) is not int or cfg['batch_size'] < 2:
        raise ValueError('Actual full ResEncM 128-cube physical batch contract required')
    if 'ResidualEncoderUNet' not in str(cfg['architecture']['network_class_name']):
        raise ValueError('Historical ResEncM architecture differs')
    paths = {name: sha(pre / name) for name in
        ('splits_final.json', 'dataset.json', 'dataset_fingerprint.json', PLANS + '.json')}
    identifier = cfg['data_identifier']
    data = pre / identifier
    if not data.is_dir():
        raise ValueError('Complete historical preprocessed native arrays missing')
    return dict(preprocessed=str(pre), source_files_sha256=paths,
        dataset_name=pre.name, dataset_id=730, configuration='3d_fullres', plans_name=PLANS,
        data_identifier=identifier, patch_size=cfg['patch_size'], physical_batch=cfg['batch_size'],
        epochs=250, cp_probability=.5, split=split,
        normalization_fitting_scope='historical Dataset730 provenance; not retrospectively certified as train-only',
        independent_test_available=False, outer26_role='validation used during training/model selection')


def bank_u_values(score, plan):
    """All P remain joint context; only original 128 U can become CP locations."""
    import numpy as np
    values = np.asarray(score, dtype=np.float64)
    if values.shape != (len(plan.record_ids),) or not np.isfinite(values).all():
        raise ValueError('Complete finite joint P+128U score vector required')
    if len(plan.unobserved_indices) != 128 or set(plan.unobserved_bank_positions) != set(range(128)):
        raise ValueError('Exactly the complete original128 native U bank required')
    result = np.empty(128, dtype=np.float64)
    for index, position in zip(plan.unobserved_indices, plan.unobserved_bank_positions):
        result[position] = values[index]
    by_id = {row['id']: row for row in plan.rows}
    centers = np.asarray([by_id[name]['center'] for name in plan.bank_record_ids], dtype=np.int64)
    if centers.shape != (128, 3) or len(np.unique(centers, axis=0)) != 128:
        raise ValueError('Full original unique native center bank required')
    return result, centers


def verify_cp_centers(case, source, centers, settings):
    """Validate every frozen location using the actual complete paste footprint.

    This is the same bounds/coverage/occupied clearance/center-distance predicate
    used by original build_candidate_pool. No center, donor or mask is redrawn.
    U are ranking observations, not a promise that every donor footprint fits.
    Keep every128 score/center and expose a separate physical eligibility mask.
    """
    import numpy as np
    from scipy import ndimage as ndi
    from hiercp.common import slices_for_center, distance_to_mask_mm
    centers = np.asarray(centers)
    mask = np.asarray(source.patch_mask, dtype=bool)
    if centers.shape != (128, 3) or centers.dtype.kind not in 'iu' or not mask.any():
        raise ValueError('All128 centers and complete nonempty real donor footprint required')
    occupied = case.label == 2
    clearance = int(settings['occupied_clearance_vox'])
    forbidden = (ndi.binary_dilation(occupied, structure=ndi.generate_binary_structure(3, 1), iterations=clearance)
                 if clearance > 0 else occupied)
    distance_mm = distance_to_mask_mm(occupied, case.spacing)
    distance_vox = distance_to_mask_mm(occupied, (1., 1., 1.)) if settings['min_center_separation_vox'] > 0 else None
    failures = []
    coverages = []
    for index, center in enumerate(centers):
        center = tuple(map(int, center))
        box = slices_for_center(center, mask.shape, case.image.shape)
        reason = None
        coverage = 0.
        if box is None:
            reason = 'complete_footprint_outside_CT'
        elif np.any(mask & forbidden[box]):
            reason = 'existing_tumor_or_clearance_overlap'
        else:
            coverage = float(np.count_nonzero(mask & (case.label[box] == 1)) / np.count_nonzero(mask))
            if coverage < settings['min_liver_coverage']:
                reason = 'insufficient_liver_coverage'
            elif distance_mm[center] < settings['min_center_separation_mm']:
                reason = 'center_separation_mm'
            elif distance_vox is not None and distance_vox[center] < settings['min_center_separation_vox']:
                reason = 'center_separation_vox'
        coverages.append(coverage)
        if reason:
            failures.append(dict(index=index, center=list(center), reason=reason))
    eligible=[True]*128
    for failure in failures:eligible[failure['index']]=False
    audit=dict(all128_full_footprints_valid=not failures, source_occupied_voxels=int(mask.sum()),
        minimum_liver_coverage=min(coverages), invalid_candidates=len(failures), donor_redraw=False,
        eligible_count=sum(eligible),eligible_mask=eligible,invalid_candidate_reasons=failures,
        full128_ranking_universe_retained=True,physical_constraints=copy.deepcopy(settings),
        centers_relocated=False, candidates_scored=128, validation_annotations_used=False)
    if not any(eligible):
        raise ValueError('No physically eligible CP position for the fixed donor; no recipient skipped or donor/center redrawn: '+json.dumps(audit))
    return audit


def eligible_argmax(scores,eligible_mask):
    """Stable original-bank tie order; never alter/discard the128 score vector."""
    import numpy as np
    values=np.asarray(scores);eligible=np.asarray(eligible_mask)
    if (values.shape!=(128,) or not np.isfinite(values).all() or eligible.shape!=(128,)
            or eligible.dtype.kind not in 'bu' or not np.isin(eligible,[0,1]).all()):
        raise ValueError('Complete finite128 scores and literal128 physical eligibility flags required')
    indices=np.flatnonzero(eligible)
    if not len(indices):raise ValueError('No eligible original CP position; no synthetic/default candidate')
    return int(indices[int(np.argmax(values[indices]))])


def validate_score_rows(rows, population, pin):
    """Admit all105 immutable original scores against the native assignment."""
    import numpy as np
    expected=tuple(population.partition_cases('inner_train'))+tuple(population.partition_cases('inner_val'))
    by_case={row.get('case_id'):row for row in rows}
    if len(rows)!=105 or len(by_case)!=105 or set(by_case)!=set(expected):
        raise ValueError('Exactly all105 original recipient score files required; no skipped case')
    for case_id in expected:
        row=by_case[case_id];plan=population.case(case_id,128)
        _,centers=bank_u_values(np.zeros(len(plan.record_ids)),plan)
        values=np.asarray(row.get('scores'))
        if (values.shape!=(128,) or not np.isfinite(values).all()
                or row.get('centers')!=centers.tolist()
                or row.get('donor_case_id')!=plan.donor_case_id
                or row.get('donor_component')!=plan.donor_component
                or row.get('joint_observed_P')!=plan.observed_P
                or row.get('joint_candidates')!=len(plan.record_ids)
                or row.get('all128U_scored') is not True or row.get('all_P_in_joint_context') is not True
                or row.get('model_sha256')!=pin['selected']['best_model_sha256']):
            raise ValueError('Frozen score/input/donor/full-P ownership differs: '+case_id)
    return [copy.deepcopy(by_case[case]) for case in expected]


def _scoring_core_ast(text):
    """Only new post-inference publication is excluded from numerical identity."""
    tree=ast.parse(text);result={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('prepare_bank','bank_u_values','make_geometry'):
            if node.name=='prepare_bank':
                node.body=[statement for statement in node.body if not (
                    isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call)
                    and isinstance(statement.value.func,ast.Name) and statement.value.func.id=='seal_scores')]
            result[node.name]=ast.dump(node,include_attributes=False)
    if set(result)!= {'prepare_bank','bank_u_values','make_geometry'}:
        raise ValueError('Original scoring core definitions missing')
    return result


def seal_scores(root, rows, population, pin, *, scoring_provenance=None):
    """Publish the complete score proof before any fallible raw materialization."""
    from .contracts import canonical_hash
    root=Path(root);request=read(root/'request.json')
    rows=validate_score_rows(rows,population,pin)
    files={row['case_id']:dict(path='scores/'+row['case_id']+'.json',
        sha256=sha(root/'scores'/(row['case_id']+'.json'))) for row in rows}
    for row in rows:
        if read(root/files[row['case_id']]['path'])!=row:
            raise ValueError('Persisted complete score row differs from admitted result')
    manifest=dict(format=SCORE_FORMAT,complete=True,debug=False,recipients=105,candidates_per_recipient=128,
        all_P_retained=True,model_sha256=pin['selected']['best_model_sha256'],pin=pin,
        inventory_sha256=request['inventory_sha256'],baseline=request['baseline'],
        request_file_sha256=sha(root/'request.json'),population=population.manifest(),score_files=files,
        score_row_sha256=canonical_hash(rows),scoring_provenance=scoring_provenance or dict(
            source_files_sha256=request['source_files_sha256'],original_frozen_GNN_scoring=True),
        production_GNN_updates=0,raw_materialization_started=False)
    manifest['manifest_sha256']=canonical_hash(manifest)
    new_json(root/'scoring_manifest.json',manifest)
    return manifest


def materialize_bank(*, pin_path, inventory_path, baseline_preprocessed, scores, output,
                     score_source_code=None, gpu=1):
    """CPU-only recovery from sealed scores or the exact audited legacy source."""
    if gpu!=1:raise ValueError('This downstream arm is assigned GPU1; recovery itself is CPU-only')
    from .comparison_native_upper_cache import _load_geometry_inputs
    from .contracts import canonical_hash
    from .v23_data import V23Population
    pin,original_request=admit_pin(pin_path,inventory_path)
    # Preserve the same preactivation boundary as actual GNN inference.
    _load_geometry_inputs(original_request['native_experiment'],'native',inventory_path)
    meta=read(inventory_path);population=V23Population(meta,debug=False)
    baseline=validate_baseline(baseline_preprocessed,meta['split'])
    source=Path(scores).resolve(strict=True)
    if source.is_symlink() or not source.is_dir() or source.name!='scores':
        raise ValueError('Exact original scores directory required')
    source_root=source.parent;old_request=read(source_root/'request.json')
    if (old_request.get('format')!=FORMAT or old_request.get('pin')!=pin
            or old_request.get('inventory_sha256')!=sha(inventory_path)
            or old_request.get('baseline')!=baseline or old_request.get('debug') is not False
            or old_request.get('GPU')!=1 or old_request.get('cp_probability')!=.5
            or old_request.get('epochs')!=250 or old_request.get('resource_contract')!=resource_contract()):
        raise ValueError('Original completed scoring request differs from the pinned full experiment')
    expected=tuple(population.partition_cases('inner_train'))+tuple(population.partition_cases('inner_val'))
    actual=list(source.iterdir())
    if {path.name for path in actual}!={case+'.json' for case in expected} or any(path.is_symlink() or not path.is_file() for path in actual):
        raise ValueError('Exact105 regular original score files required')
    files={case:dict(path=str(source/(case+'.json')),sha256=sha(source/(case+'.json'))) for case in expected}
    rows=validate_score_rows([read(files[case]['path']) for case in expected],population,pin)
    sealed=source_root/'scoring_manifest.json'
    provenance=dict(source_request=str(source_root/'request.json'),source_request_file_sha256=sha(source_root/'request.json'),
        source_score_files=files,original_frozen_GNN_scoring=True,CPU_only_recovery=True,GNN_rescored=False)
    if sealed.exists():
        manifest=read(sealed)
        if (manifest.get('format')!=SCORE_FORMAT or manifest.get('complete') is not True
                or manifest.get('manifest_sha256')!=canonical_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})
                or manifest.get('request_file_sha256')!=provenance['source_request_file_sha256']
                or manifest.get('population')!=population.manifest() or manifest.get('pin')!=pin
                or manifest.get('inventory_sha256')!=sha(inventory_path)
                or manifest.get('score_row_sha256')!=canonical_hash(rows)
                or manifest.get('score_files')!={case:dict(path='scores/'+case+'.json',sha256=files[case]['sha256']) for case in expected}):
            raise ValueError('Sealed original complete score proof differs')
        provenance['source_scoring_manifest_sha256']=sha(sealed)
        provenance['original_scoring_provenance']=manifest['scoring_provenance']
    else:
        if score_source_code is None:raise ValueError('Legacy unsealed score recovery requires explicit audited --score-source-code')
        old_code=Path(score_source_code).resolve(strict=True)
        if any(sha(old_code/name)!=old_request['source_files_sha256'].get(name) for name in FILES):
            raise ValueError('Legacy actual scoring code differs from the original request SHA proof')
        if _scoring_core_ast((old_code/FILES[0]).read_text(encoding='utf8'))!=_scoring_core_ast((ROOT/FILES[0]).read_text(encoding='utf8')):
            raise ValueError('Original scoring equations/input enumeration changed; scores cannot be silently reused')
        from tools.run_v23_all_p import EDGE_FEEDING_SHA256,EDGE_ADAPTER_SHA256
        unchanged=(*EDGE_FEEDING_SHA256,'hiercp_v1x/v23_edge_execution.py','tools/run_v23_all_p.py',
            'hiercp_v1x/comparison_native_upper_cache.py','hiercp_v1x/transition_v1_local.py')
        hashes={name:sha(old_code/name) for name in unchanged}
        if any(sha(ROOT/name)!=checksum for name,checksum in hashes.items()):
            raise ValueError('Original GNN feeding/archive/execution source bytes changed')
        provenance.update(source_code=str(old_code),source_files_sha256=old_request['source_files_sha256'],
            unchanged_GNN_source_sha256=hashes,scoring_core_AST_identical=True,
            execution_adapter_sha256=EDGE_ADAPTER_SHA256,legacy_unsealed_complete105_recovery=True)
    root=Path(output).resolve()
    if root.exists():raise FileExistsError('New recovery bank required; original failed output preserved')
    for protected in (source_root,Path(pin['admitted_source_experiment']),Path(original_request['native_experiment']),Path(baseline_preprocessed)):
        protected=protected.resolve()
        if root.is_relative_to(protected) or protected.is_relative_to(root):raise ValueError('Recovery output overlaps protected original input/results')
    require_project_budget();root.mkdir(parents=True);(root/'scores').mkdir()
    request=copy.deepcopy(old_request);request['source_files_sha256']={name:sha(ROOT/name) for name in FILES}
    request['score_recovery']=provenance
    new_json(root/'request.json',request)
    for case in expected:
        old_file=Path(files[case]['path']);new_file=root/'scores'/(case+'.json')
        if sha(old_file)!=files[case]['sha256']:raise ValueError('Original score changed during recovery')
        with old_file.open('rb') as original,new_file.open('xb') as destination:shutil.copyfileobj(original,destination,8*2**20)
        if sha(old_file)!=files[case]['sha256'] or sha(new_file)!=files[case]['sha256']:
            raise ValueError('Lossless complete score recovery SHA mismatch')
    seal_scores(root,rows,population,pin,scoring_provenance=provenance)
    return _materialize_bank(root,meta,rows,baseline,pin)


def make_geometry(bundle, population, output, runtime, full_validation_cache):
    """Include the zero-P training patients in CP, without changing v23 training."""
    from .v23_geometry import V23UpperGeometryCache
    class CPGeometry(V23UpperGeometryCache):
        def _plans_for_stage(self, count, selection):
            if count != 128 or selection is not None:
                raise ValueError('CP requires full128 and no curriculum/target subset')
            cases = tuple(self.train_cases) + tuple(self.validation_cases)
            plans = [self.population.case(case, 128) for case in cases]
            for plan in plans:
                self._plan_stages[self._plan_key(plan)] = 128
            return cases, plans, 128, None
        def _prepare_regions(self, *, readonly=False):
            # Existing region equations and input SHA guards are reused exactly.
            # The original method's training-loss ranking filter is inapplicable
            # to augmentation recipients, so adapt only that enumeration call.
            import types
            fn = V23UpperGeometryCache._prepare_regions
            class PopulationView:
                def partition_cases(view, partition, *, ranking_only=False):
                    return population.partition_cases(partition, ranking_only=False)
                def case(view, *args, **kwargs): return population.case(*args, **kwargs)
            original = self.population
            self.population = PopulationView()
            try:
                return types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)(self, readonly=readonly)
            finally:
                self.population = original
    return CPGeometry(bundle, population, output, workers=runtime['workers'],
        rss_bytes=int(runtime['rss_gib_per_rank'] * 2**30),
        resident_bytes=int(runtime['geometry_resident_gib_per_rank'] * 2**30),
        full_validation_cache=full_validation_cache,
        memoize_cpu_geometry=runtime['memoize_cpu_geometry'], cache_stage_admission=runtime['cache_stage_admission'])


def prepare_bank(*, pin_path, inventory_path, baseline_preprocessed, output, gpu=1):
    """Score the actual original model once and publish immutable raw paste bank."""
    from tools.local_cnn_device import select
    select(gpu)  # Before Torch import/initialization.
    if gpu != 1:
        raise ValueError('This explicitly assigned downstream arm uses physical GPU1')
    import numpy as np
    import psutil
    import torch
    from .comparison_native_upper_cache import _load_geometry_inputs
    from .v23_data import V23Population
    from .v23_inputs import V23InputProvider, V23InputMemoryCoordinator
    from .transition_v1_data import NativeObservationDataset
    from .v23_training import V23Scorer
    from .u_bridge_training import digest
    from .historical_evaluation import ResourceBudget
    pin, request = admit_pin(pin_path, inventory_path)
    # Population admission invokes the native donor-assignment AST, which
    # imports hiercp.common. Activate the pinned source BEFORE constructing it.
    # This also precedes publishing any new output directory/request.
    bundle, _, _, _, preserved = _load_geometry_inputs(request['native_experiment'], 'native', inventory_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('New CP bank output required; existing results are preserved')
    for protected in (Path(pin['admitted_source_experiment']), Path(request['native_experiment']), Path(baseline_preprocessed)):
        if output.is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output):
            raise ValueError('New CP output must be separate from original inputs/results')
    meta = read(inventory_path)
    population = V23Population(meta, debug=False)
    baseline = validate_baseline(baseline_preprocessed, meta['split'])
    runtime = copy.deepcopy(request['config']['v23_runtime'])
    runtime.update(workers=PROJECT_CPU_CORES,rss_gib_per_rank=PROJECT_RSS_BYTES/2**30,
        resident_gib_per_rank=INPUT_RESIDENT_BYTES/2**30,
        geometry_resident_gib_per_rank=GEOMETRY_RESIDENT_BYTES/2**30)
    require_project_budget()
    if (runtime['workers'] < 2 or runtime['workers'] > len(psutil.Process().cpu_affinity())
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or 'A6000' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Measured parallel CPU and singleton actual A6000 required')
    free, total = torch.cuda.mem_get_info()
    if min(free, total) <= runtime['cuda_gib_per_gpu'] * 2**30:
        raise MemoryError('Actual GPU lacks the unchanged GNN resource budget')
    torch.cuda.set_per_process_memory_fraction(runtime['cuda_gib_per_gpu'] * 2**30 / total)
    torch.set_num_threads(1)
    budget = ResourceBudget(int(runtime['cuda_gib_per_gpu'] * 2**30), int(runtime['rss_gib_per_rank'] * 2**30))
    output.mkdir(parents=True)
    new_json(output / 'request.json', dict(format=FORMAT, pin=pin, pin_sha256=sha(pin_path),
        inventory=str(Path(inventory_path).resolve()), inventory_sha256=sha(inventory_path), baseline=baseline,
        source_files_sha256={name:sha(ROOT / name) for name in FILES},
        historical_training_annotation_contract=True, inference_GT=False, GPU=1,resource_contract=resource_contract(),
        donor_policy='unchanged_v23_independent_inner_train_fixed_per_recipient',
        donor_policy_matches_historical_Basic=False, cp_probability=.5, epochs=250, debug=False))
    geometry = make_geometry(bundle, population, output / 'upper_geometry', runtime, request['full_validation_upper_cache'])
    geometry.prepare(128)
    providers = {}
    coordinator = V23InputMemoryCoordinator(budget.rss_bytes)
    for partition in ('inner_train', 'inner_val'):
        providers[partition] = V23InputProvider(NativeObservationDataset(inventory_path, partition, False),
            workers=runtime['workers'], resident_bytes=int(runtime['resident_gib_per_rank'] * 2**30),
            rss_bytes=budget.rss_bytes, cache_index=request['prepared_cache'],
            persistent_cpu_workers=True, cache_sampled_views=True,
            fixed_validation_epoch=bundle.config['training']['fixed_validation_epoch'], memory_coordinator=coordinator)
    # _load_geometry_inputs activates the sealed archive before any hiercp
    # imports. Resolve its already verified runtime explicitly: state keys and
    # parameter counts alone cannot establish original forward equations.
    from .transition_v1_local import _runtime
    original_runtime = _runtime(expected_snapshot_root=bundle.source,
        scope_contract=bundle.scope['contract_sha256'])
    model_module = original_runtime['model']
    original_model_path = Path(model_module.__file__).resolve(strict=True)
    if original_model_path != Path(bundle.source).resolve(strict=True)/'hiercp/model.py':
        raise ValueError('Frozen GNN class is not the admitted original archive')
    if str(original_model_path) not in preserved or sha(original_model_path)!=preserved[str(original_model_path)]:
        raise ValueError('Original GNN module lacks its byte-exact source proof')
    from hiercp.tensor import configure_runtime
    configure_runtime(deterministic=bundle.config['runtime']['deterministic'],
        allow_tf32=bundle.config['runtime']['allow_tf32'], cudnn_benchmark=bundle.config['runtime']['cudnn_benchmark'])
    selected_pin=pin['selected']
    saved = torch.load(selected_pin['best_path'], map_location='cpu', weights_only=False, mmap=True)
    identity = read(Path(selected_pin['best_path']).parent/'training_identity.json')
    if saved.get('identity_sha256') != identity['identity_sha256'] or saved['content_sha256'] != digest({k:v for k,v in saved.items() if k != 'content_sha256'}):
        raise ValueError('Actual GNN checkpoint content/identity checksum differs')
    if (digest(saved['model']) != selected_pin['best_model_sha256']
            or saved['content_sha256']!=selected_pin['best_content_sha256']
            or identity['binding']['population'] != population.manifest()):
        raise ValueError('Pinned GNN weights or complete native population differ')
    model = model_module.HierarchicalPyGPlacementModel(**bundle.config['model'])
    model.load_state_dict(saved['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Original full 10,434,532 parameter GNN required')
    del saved
    model.eval().to('cuda')
    scorer = V23Scorer(model, providers, geometry, physical_candidate_batch=32,
        checkpoint_local_chunks=runtime['checkpoint_local_chunks'], amp=bundle.config['training']['amp'],
        budget=budget, prefetch_cpu_chunks=True, pin_cpu_batches=True)
    from tools.run_v23_all_p import install_v23_l0_execution
    config = copy.deepcopy(bundle.config); config['v23_runtime'] = runtime
    handle, hook = install_v23_l0_execution(model, scorer, config)
    try:
        rows = []
        for partition in ('inner_train', 'inner_val'):
            cases = population.partition_cases(partition)
            for start in range(0, len(cases), 4):
                plans = [population.case(case, 128) for case in cases[start:start+4]]
                began = time.perf_counter()
                with torch.no_grad(): result = scorer(plans, epoch=bundle.config['training']['fixed_validation_epoch'], training=False)
                require_project_budget()
                for score, plan in zip(result.scores, plans):
                    values, centers = bank_u_values(score.detach().float().cpu().numpy(), plan)
                    row = dict(case_id=plan.case_id, donor_case_id=plan.donor_case_id, donor_component=plan.donor_component,
                        scores=values.tolist(), centers=centers.tolist(), joint_observed_P=plan.observed_P,
                        joint_candidates=len(plan.record_ids), all128U_scored=True, all_P_in_joint_context=True,
                        model_sha256=selected_pin['best_model_sha256'], workload=result.workload)
                    new_json(output / 'scores' / (plan.case_id + '.json'), row)
                    rows.append(row)
                print(json.dumps(dict(phase='frozen_GNN_CP_scores', partition=partition,
                    completed=len(rows), total=105, seconds=time.perf_counter()-began)), flush=True)
        if {row['case_id'] for row in rows} != set(meta['split']['outer_train']):
            raise ValueError('Every105 CP recipient must be scored, including zero-P cases')
        if digest(model.state_dict()) != selected_pin['best_model_sha256']:
            raise ValueError('Frozen GNN weights changed during inference')
        if sha(selected_pin['best_path']) != selected_pin['best_file_sha256']:
            raise ValueError('Pinned original best changed during placement inference')
        for path, checksum in preserved.items():
            if sha(path) != checksum: raise ValueError('Historical source input changed')
    finally:
        hook.remove(); handle.restore()
        for provider in providers.values(): provider.close()
    del scorer,model,providers,geometry,result
    import gc
    gc.collect();torch.cuda.empty_cache()
    seal_scores(output,rows,population,pin)
    return _materialize_bank(output, meta, rows, baseline, pin)


def _materialize_bank(root, meta, rows, baseline, pin):
    import numpy as np
    import nibabel as nib
    from hiercp.common import CasePaths, load_case
    from hiercp_v22.data import sources, donor_in_target_spacing
    from hiercp_v22.bank import map_center
    from tools.online_raw_bank_preparation import prepare_raw_case, prepare_source_candidates
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    score_manifest=read(root/'scoring_manifest.json')
    if score_manifest.get('complete') is not True or score_manifest.get('pin')!=pin:
        raise ValueError('Completed frozen105 scores must be sealed before raw materialization')
    raw = {row['case_id']: row for row in meta['raw_records']}
    plans = read(Path(baseline['preprocessed']) / (PLANS + '.json'))
    directory = Path(baseline['preprocessed']) / baseline['data_identifier']
    def load(case_id):
        info = raw[case_id]
        for kind in ('image', 'label'):
            if sha(info[kind]) != info[kind + '_sha256']: raise ValueError('Original CT/annotation changed')
        return load_case(CasePaths(case_id, Path(info['image']), Path(info['label'])))
    entries, digests, audits = {}, {}, {}
    def one(row):
        case_id, donor_id = row['case_id'], row['donor_case_id']
        score_file=root/score_manifest['score_files'][case_id]['path']
        if sha(score_file)!=score_manifest['score_files'][case_id]['sha256'] or read(score_file)!=row:
            raise ValueError('Sealed original128 GNN scores changed before materialization: '+case_id)
        target, donor = load(case_id), load(donor_id)
        collection = sources(donor, meta['base']['cache']['source_pad'], meta['config']['donor_max_diameter_mm'])
        matches = [i for i,(component,_) in enumerate(collection.entries) if component == row['donor_component']]
        if len(matches) != 1: raise ValueError('Pinned actual donor component missing')
        original, diameter = collection[matches[0]]
        source, _ = donor_in_target_spacing(original, donor.spacing, target.spacing)
        centers = np.asarray(row['centers'], dtype=np.int64)
        try:eligibility = verify_cp_centers(target, source, centers, meta['base']['generation'])
        except ValueError as error:
            raise ValueError('Recipient '+case_id+', fixed donor '+donor_id+', component '+str(row['donor_component'])+': '+str(error)) from error
        ds = infer_dataset_class(str(directory))(str(directory), [case_id])
        pre, seg, _, props = ds.load_case(case_id)
        native, reference, reference_sha = prepare_raw_case(root, case_id, target.image, target.label,
            props, plans, pre, seg, configuration_name='3d_fullres', raw_spacing=target.spacing,
            raw_spatial_unit=nib.load(raw[case_id]['image']).header.get_xyzt_units()[0], minimum_free_bytes=8*2**30)
        selected = eligible_argmax(row['scores'],eligibility['eligible_mask'])
        anchor = np.asarray(source.anchor_center) - np.asarray([s.start for s in source.patch_slices])
        refs, hashes, transport = prepare_source_candidates(root, case_id, row['donor_component'], native,
            reference_sha, source.patch_image, source.patch_mask, anchor, centers[selected:selected+1],
            baseline['patch_size'])
        mapped = np.stack([map_center(center, plans, props, native['metadata']['preprocessed_shape']) for center in centers])
        relative = f'entries/{case_id}.npz'; file = root / relative; file.parent.mkdir(exist_ok=True)
        with file.open('xb') as stream:
            np.savez(stream, paste_contract=np.asarray(['onlinecp_raw_target_paste_v1']), case_id=np.asarray([case_id]),
                donor_case_id=np.asarray([donor_id]), donor_component_id=np.asarray([row['donor_component']]),
                source_component=np.asarray([row['donor_component']]), source_diameter_mm=np.asarray([diameter]),
                candidate_eligibility=np.asarray(eligibility['eligible_mask'],dtype=np.uint8),
                selection_policy=np.asarray([CP_SELECTION]),
                selected_candidate=np.asarray([selected]), selected_payload=refs, selected_payload_sha256=hashes,
                raw_case_reference=np.asarray([reference]), raw_case_reference_sha256=np.asarray([reference_sha]),
                candidate_centers=mapped, candidate_raw_centers=centers, scores=np.asarray(row['scores'], dtype=np.float32))
        return case_id, relative, sha(file), dict(eligibility=eligibility, transport=transport, selected_candidate=selected,
            selection_policy=CP_SELECTION,unconstrained_argmax=int(np.argmax(row['scores'])),
            selected_score=float(row['scores'][selected]),scores_file_sha256=score_manifest['score_files'][case_id]['sha256'],
            all_P_in_joint_context=True, scored128=True, donor_case_id=donor_id, donor_component=row['donor_component'])
    from hiercp.preparation_runtime import run_case_jobs, snapshot
    import types
    def bounded_snapshot():
        state=snapshot();rss=process_tree_rss()
        state.update(cpu_capacity=PROJECT_CPU_CORES,
            available_memory_bytes=min(state['available_memory_bytes'],max(0,PROJECT_RSS_BYTES-rss)),
            project_resource_contract=resource_contract(),actual_process_and_children_RSS_bytes=rss)
        return state
    bounded_run=types.FunctionType(run_case_jobs.__code__,dict(run_case_jobs.__globals__,snapshot=bounded_snapshot),
        run_case_jobs.__name__,run_case_jobs.__defaults__,run_case_jobs.__closure__)
    def commit(result):
        case_id,relative,checksum,audit=result
        entries[case_id] = [relative]; digests[relative] = checksum; audits[case_id] = audit
        print(json.dumps(dict(phase='complete_raw_CP', completed=len(entries), total=105, case_id=case_id)), flush=True)
    require_project_budget()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        bounded_run(tasks=rows,function=one,commit=commit,workers=PROJECT_CPU_CORES,report_path=root/'raw_CP_parallel_resources.json')
    require_project_budget()
    normalization = plans['foreground_intensity_properties_per_channel']['0']
    bank = dict(format='hiercp_online_bank_v2', pipeline_version=FORMAT, complete=True, debug=False,
        paste_contract='onlinecp_raw_target_paste_v1', entry_storage='v23_scores128_eligible_selected_raw_target_v1',
        candidate_count=128, hier_top_k=1, tumor_label=2, liver_label=1, cp_probability=.5,
        intensity_scale_range=[.95,1.05], intensity_shift_range_hu=[-5.,5.],
        normalization=dict(mean=normalization['mean'], std=normalization['std']),
        entries_by_case=entries, entry_sha256=digests, split=meta['split'], pin=pin, CP_audits=audits,
        no_placement_policy='error',
        donor_policy='unchanged_v23_independent_inner_train_fixed_per_recipient',
        donor_policy_matches_historical_Basic=False, source_identity={name:sha(ROOT/name) for name in FILES},
        baseline=baseline, inference_GT=False, outer_validation_CP=False,
        original_GNN_parameters=PARAMETERS, all105_recipients=True, all_joint_P_retained=True,
        selection_policy=CP_SELECTION,scoring_manifest='scoring_manifest.json',scoring_manifest_file_sha256=sha(root/'scoring_manifest.json'),
        resource_contract=resource_contract())
    validate_bank(bank)
    new_json(root / 'index.json', bank)
    return root / 'index.json'


def validate_bank(bank):
    split = validate_split(bank['split'])
    if (bank.get('pipeline_version') != FORMAT or bank.get('complete') is not True or bank.get('debug') is not False
            or bank.get('cp_probability') != .5 or bank.get('candidate_count') != 128
            or bank.get('original_GNN_parameters') != PARAMETERS or bank.get('all105_recipients') is not True
            or bank.get('all_joint_P_retained') is not True or bank.get('outer_validation_CP') is not False
            or bank.get('inference_GT') is not False or bank.get('scoring_manifest')!='scoring_manifest.json'
            or not isinstance(bank.get('scoring_manifest_file_sha256'),str)
            or len(bank['scoring_manifest_file_sha256'])!=64):
        raise ValueError('Complete frozen original-GNN CP training contract required')
    expected = set(split['outer_train'])
    if set(bank['entries_by_case']) != expected or set(bank['CP_audits']) != expected:
        raise ValueError('Full105 training-only recipient bank required')
    for case in expected:
        audit = bank['CP_audits'][case]
        eligibility=audit['eligibility'];mask=eligibility.get('eligible_mask',[])
        if (len(bank['entries_by_case'][case]) != 1
                or audit['donor_case_id'] not in split['inner_train'] or audit['donor_case_id'] == case
                or len(mask)!=128 or any(type(value) is not bool for value in mask)
                or eligibility.get('eligible_count')!=sum(mask) or sum(mask)==0
                or eligibility.get('invalid_candidates')!=128-sum(mask)
                or eligibility.get('full128_ranking_universe_retained') is not True
                or audit.get('selection_policy')!=CP_SELECTION or bank.get('selection_policy')!=CP_SELECTION
                or type(audit.get('selected_candidate')) is not int or not 0<=audit['selected_candidate']<128
                or not mask[audit['selected_candidate']]):
            raise ValueError('Invalid native donor/candidate eligibility: ' + case)
        name = bank['entries_by_case'][case][0]
        if name not in bank['entry_sha256']:
            raise ValueError('SHA-bound native entry missing')
    validate_pin(bank['pin'])
    return bank


def prepare_nnunet(bank_path, output):
    """Private runtime and immutable preprocessing references; no old writes."""
    import nnunetv2
    bank_path = Path(bank_path).resolve(); bank = validate_bank(read(bank_path))
    baseline = validate_baseline(bank['baseline']['preprocessed'], bank['split'])
    if baseline != bank['baseline']: raise ValueError('Historical native plans changed')
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh segmentation output required')
    if output.is_relative_to(bank_path.parent) or bank_path.parent.is_relative_to(output):
        raise ValueError('Segmentation output and immutable CP bank must be separate')
    output.mkdir(parents=True)
    package = Path(nnunetv2.__file__).resolve().parent
    private = output / 'runtime' / 'nnunetv2'
    custom_names=('nnUNetTrainer_OnlinePairedCP.py','onlinecp_raw_bank.py','onlinecp_raw_resampling.py','nnUNetTrainer_FrozenV23CP.py')
    shutil.copytree(package, private, ignore=shutil.ignore_patterns('__pycache__','*.pyc',*custom_names))
    dest = private / 'training' / 'nnUNetTrainer'
    for name in custom_names:
        path = dest / name; source = ROOT / 'custom_trainers' / name
        if path.exists():
            if sha(path) != sha(source): raise ValueError('Private runtime conflicting existing trainer: '+name)
        else: shutil.copy2(source,path)
    # Independent copies prevent nnUNet's unpack/cache preparation from writing
    # through a directory symlink or hardlink into any historical experiment.
    source = Path(baseline['preprocessed']); pre = output/'nnUNet_preprocessed'/source.name
    # Native end-of-training validation uses this directory for metrics. Copy
    # its real annotations as evaluator assets; the predictor still gets CT
    # images only, and validation has no copy-paste loader.
    ground_truth=source/'gt_segmentations'
    expected=set(bank['split']['outer_train'])|set(bank['split']['outer_val'])
    if (not ground_truth.is_dir() or ground_truth.is_symlink()
            or {p.name[:-7] for p in ground_truth.glob('*.nii.gz')}!=expected):
        raise ValueError('Complete historical native131 gt_segmentations required for end-of-training validation')
    directories=(baseline['data_identifier'],'gt_segmentations')
    data_files=[p for name in directories for p in (source/name).rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in data_files):raise ValueError('Regular original preprocessing arrays required')
    required=sum(p.stat().st_size for p in data_files)+8*2**30
    if shutil.disk_usage(output).free<=required:raise OSError('Full native preprocessing copy and checkpoint reserve do not fit; no subset or shared-source writes')
    pre.mkdir(parents=True)
    for file in source.iterdir():
        target = pre/file.name
        if file.is_file(): shutil.copy2(file,target)
        elif file.name in directories:
            shutil.copytree(file,target)
    copied_arrays={}
    for original in data_files:
        relative=original.relative_to(source).as_posix();checksum=sha(original)
        if sha(pre/relative)!=checksum:raise ValueError('Full native preprocessing copy differs: '+relative)
        copied_arrays[relative]=checksum
    if validate_baseline(source,bank['split'])!=baseline:raise ValueError('Native plans changed during reuse')
    document = dict(format=FORMAT, root=str(output), bank=str(bank_path), bank_sha256=sha(bank_path),
        baseline=baseline, private_runtime=str(private.parent), trainer=TRAINER,
        native_preprocessed_source_written=False, model_weights_fresh=True,
        preprocessed_storage='independent full original arrays; no directory symlink/hardlink',
        preprocessing_copy_bytes=sum(p.stat().st_size for p in data_files),
        copied_native_array_sha256=copied_arrays,
        private_trainer_sha256={name:sha(dest/name) for name in custom_names},
        pretrained_segmentation_used=False, prediction_inputs='outer26 CT images only',
        source_files_sha256={name:sha(ROOT/name) for name in FILES},resource_contract=resource_contract())
    new_json(output/'native.json',document)
    return output/'native.json'


def environment(native):
    env = os.environ.copy(); root = Path(native['root'])
    env.update(nnUNet_preprocessed=str(root/'nnUNet_preprocessed'),nnUNet_results=str(root/'nnUNet_results'),
        nnUNet_raw=str(root/'nnUNet_raw'), ONLINE_CP_BANK=native['bank'], ONLINE_CP_SEED='42',
        nnUNet_n_proc_DA=str(PROJECT_CPU_CORES), PYTHONDONTWRITEBYTECODE='1', nnUNet_compile='false',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join((native['private_runtime'], str(ROOT))))
    return env


def admit_native(native_path):
    """Guard the actual private runtime and complete source copy before use."""
    native=read(native_path)
    bank=validate_bank(read(native['bank']))
    if (native['format']!=FORMAT or sha(native['bank'])!=native['bank_sha256']
            or native['resource_contract']!=resource_contract()
            or any(sha(ROOT/name)!=checksum for name,checksum in native['source_files_sha256'].items())):
        raise ValueError('Completed immutable native runtime/source/bank changed')
    pre=Path(native['root'])/'nnUNet_preprocessed'/bank['baseline']['dataset_name']
    if validate_baseline(pre,bank['split'])['source_files_sha256']!=native['baseline']['source_files_sha256']:
        raise ValueError('Private native plans/split/normalization differ from the historical baseline')
    for name,checksum in native['copied_native_array_sha256'].items():
        if sha(pre/name)!=checksum:raise ValueError('Copied native array/validation annotation changed: '+name)
    trainers=Path(native['private_runtime'])/'nnunetv2/training/nnUNetTrainer'
    for name,checksum in native['private_trainer_sha256'].items():
        if sha(trainers/name)!=checksum:raise ValueError('Private native CP trainer changed: '+name)
    return native,bank


def _native_inactive_auxiliary_gradient_proof(trainer, parameters):
    """Map only actual zero-weight outputs to the original reversed decoder heads."""
    import ast
    import hashlib
    import inspect
    import math
    import textwrap
    weights = getattr(trainer.loss, 'weight_factors', None)
    if weights is None:
        if getattr(trainer, 'enable_deep_supervision', False):
            raise ValueError('Actual native deep-supervision loss weights are missing')
        return dict(actual_auxiliary_loss_weights=None, zero_weight_output_indices=[],
            inactive_decoder_layer_indices=[], official_inactive_parameter_names=[])
    weights = [float(value) for value in weights]
    if (not weights or not all(math.isfinite(value) and value >= 0 for value in weights)
            or not any(value > 0 for value in weights)):
        raise ValueError('Actual native deep-supervision weights must be finite/nonnegative/nonempty')
    zero = [index for index, value in enumerate(weights) if value == 0]
    decoder = getattr(trainer.network, 'decoder', None)
    heads = getattr(decoder, 'seg_layers', None)
    if heads is None or len(heads) != len(weights):
        raise ValueError('Actual native auxiliary output/head mapping is missing')
    source = textwrap.dedent(inspect.getsource(type(decoder).forward))
    tree = ast.parse(source)
    reversed_outputs = any(isinstance(node, ast.Return)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name) and node.value.value.id == 'seg_outputs'
        and isinstance(node.value.slice, ast.Slice)
        and node.value.slice.lower is None and node.value.slice.upper is None
        and isinstance(node.value.slice.step, ast.UnaryOp)
        and isinstance(node.value.slice.step.op, ast.USub)
        and isinstance(node.value.slice.step.operand, ast.Constant)
        and node.value.slice.step.operand.value == 1 for node in ast.walk(tree))
    if not reversed_outputs:
        raise ValueError('Original native seg_outputs[::-1] head ordering is unproved')
    layers = [len(heads) - 1 - index for index in zero]
    inactive = {id(parameter) for index in layers for parameter in heads[index].parameters()}
    names = [name for name, parameter in parameters.items() if id(parameter) in inactive]
    if len(names) != len(inactive):
        raise ValueError('Official inactive auxiliary heads differ from actual trainable parameters')
    return dict(actual_auxiliary_loss_weights=weights, zero_weight_output_indices=zero,
        inactive_decoder_layer_indices=layers, official_inactive_parameter_names=names,
        original_decoder_forward_sha256=hashlib.sha256(source.encode()).hexdigest(),
        output_order='seg_outputs[::-1]', inactive_parameters_are_not_disconnected_core=True)


def _native_clone_step_with_amp_retry(trainer, batch, parameters, before):
    """Retry the exact augmented batch only after a proved native AMP skip.

    Native train_step owns backward, clipping, scaler.step/update. We never
    modify loss weights, requires_grad, optimizer, precision or input tensors.
    CP counters and model RNG are restored before a retry, so this batch is
    counted once. Actual attempt/resource logs remain separate observations.
    """
    import copy
    import math
    import numpy as np
    import torch
    from hiercp_v1x.u_bridge_training import capture_rng, restore_rng
    proof = _native_inactive_auxiliary_gradient_proof(trainer, parameters)
    allowed_missing = set(proof['official_inactive_parameter_names'])
    if set(before) != set(parameters):
        raise ValueError('Every actual native parameter needs its pre-update snapshot')
    counter_names = ('_online_cp_events', '_online_cp_samples', '_online_schedule_hash',
                     '_online_native_transport')
    if any(not hasattr(trainer, name) for name in counter_names):
        raise ValueError('Actual native CP counter ownership is missing')
    counters = {name: copy.deepcopy(getattr(trainer, name)) for name in counter_names}
    rng = capture_rng()
    attempts = []
    scaler = trainer.grad_scaler
    while True:
        scale_before = None if scaler is None else float(scaler.get_scale())
        # Native train_step removes CP metadata. A shallow private dictionary
        # preserves the identical original data/target tensors and audit values.
        result = trainer.train_step(dict(batch))
        loss = float(np.asarray(result['loss']))
        if not math.isfinite(loss):
            raise ValueError('Actual native CUDA forward loss is nonfinite; no AMP retry')
        missing = [name for name, parameter in parameters.items() if parameter.grad is None]
        unexpected = sorted(set(missing) - allowed_missing)
        if unexpected:
            raise ValueError('Disconnected actual native active/core gradients: ' + repr(unexpected))
        nonfinite = [name for name, parameter in parameters.items()
                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
        changed = [name for name, parameter in parameters.items()
                   if not torch.equal(before[name], parameter.detach())]
        scale_after = None if scaler is None else float(scaler.get_scale())
        if not nonfinite:
            if not changed:
                raise ValueError('Finite native gradients did not produce an actual optimizer update')
            if scale_before is not None and (not math.isfinite(scale_after) or scale_after <= 0
                    or scale_after < scale_before):
                raise ValueError('Finite native gradients contradict the GradScaler update state')
            proof.update(missing_native_gradient_parameters=missing,
                present_gradient_tensors=len(parameters)-len(missing),
                trainable_parameter_tensors=len(parameters),
                changed_parameter_tensors=len(changed), AMP_overflow_attempts=attempts,
                AMP_scale_after_successful_update=scale_after)
            return result, proof
        if (scaler is None or not scaler.is_enabled() or scale_before is None
                or not math.isfinite(scale_before) or scale_before <= 0
                or not math.isfinite(scale_after) or not 0 < scale_after < scale_before):
            raise ValueError('Nonfinite native gradients require a strictly decreased positive AMP scale')
        if changed:
            raise ValueError('Native AMP overflow changed model weights; retry is unsafe')
        attempts.append(dict(attempt=len(attempts)+1, scale_before=scale_before, scale_after=scale_after,
            nonfinite_gradient_names=nonfinite, weights_unchanged=True, same_augmented_batch=True,
            model_RNG_restored_before_retry=True, CP_counters_restored_before_retry=True,
            production_optimizer_updates=0))
        print('Native CP calibration AMP skip ' + str(attempts[-1]), flush=True)
        for name, value in counters.items():
            setattr(trainer, name, copy.deepcopy(value))
        restore_rng(rng)


def _calibrate_native_worker(native_path,physical_batch,output):
    """Actual isolated native clone; no production weights/checkpoints used."""
    from tools.local_cnn_device import select
    select(1)
    import numpy as np
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP import nnUNetTrainer_250epochs_FrozenV23CP
    from hiercp.preparation_runtime import Measurement,snapshot
    native,bank=admit_native(native_path);require_project_budget()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Actual singleton native CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    output=Path(output)
    plans=read(Path(native['root'])/'nnUNet_preprocessed'/bank['baseline']['dataset_name']/(PLANS+'.json'))
    dataset=read(Path(native['root'])/'nnUNet_preprocessed'/bank['baseline']['dataset_name']/'dataset.json')
    trainer=nnUNetTrainer_250epochs_FrozenV23CP(plans,'3d_fullres',0,dataset,torch.device('cuda'))
    trainer.initialize()
    if trainer.configuration_manager.patch_size!=[128]*3 or trainer.num_epochs!=250:
        raise ValueError('Original full native architecture/250-epoch profile changed')
    baseline_batch=int(trainer.batch_size)
    # This affects this fresh calibration clone and its actual loader only.
    # Original plans, loss, network architecture and production batch persist.
    trainer.batch_size=int(physical_batch)
    train_loader,val_loader=trainer.get_dataloaders()
    trainer.network.train();torch.cuda.synchronize()
    parameters={name:p for name,p in trainer.network.named_parameters() if p.requires_grad}
    if not parameters or {id(p) for group in trainer.optimizer.param_groups for p in group['params']}!={id(p) for p in parameters.values()}:
        raise ValueError('Native trainable model parameters are absent from the optimizer')
    warmup=None;measurements=[]
    try:
        for iteration in range(4):
            load_start=time.perf_counter();batch=next(train_loader);load_seconds=time.perf_counter()-load_start
            shape=list(batch['data'].shape)
            if shape[0]!=physical_batch or shape[-3:]!=[128]*3:
                raise ValueError('Actual calibration must retain full physical batch and128-cube input')
            flags=np.asarray(batch['online_cp_applied']).reshape(-1).tolist()
            before={name:p.detach().clone() for name,p in parameters.items()}
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            started=time.perf_counter();measurement=Measurement()
            with measurement:
                result,native_gradient_admission=_native_clone_step_with_amp_retry(trainer,batch,parameters,before)
                torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            loss=float(np.asarray(result['loss']))
            if not math.isfinite(loss):raise ValueError('Actual native CUDA loss is nonfinite')
            missing=[name for name,p in parameters.items() if p.grad is None]
            nonfinite=[name for name,p in parameters.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
            changed=[name for name,p in parameters.items() if not torch.equal(before[name],p.detach())]
            if nonfinite or not changed:raise ValueError('Native loss/backward/update is nonfinite or disconnected')
            # nnUNet can disable a last deep-supervision loss weight. Preserve
            # and report that native behavior; require encoder/decoder updates.
            if not any('encoder' in name for name in changed) or not any('decoder' in name for name in changed):
                raise ValueError('Actual native encoder/decoder did not receive an optimizer update')
            row=dict(iteration=iteration,role='warmup' if iteration==0 else 'measured_repeat',
                input_shape=shape,physical_batch=physical_batch,input_wait_seconds=load_seconds,
                forward_backward_optimizer_seconds=elapsed,full_step_seconds=load_seconds+elapsed,
                loss=loss,CP_flags=flags,gradient_finite=True,present_gradient_tensors=len(parameters)-len(missing),
                trainable_parameter_tensors=len(parameters),missing_native_gradient_parameters=missing,
                model_parameter_count=sum(p.numel() for p in parameters.values()),
                changed_parameter_tensors=len(changed),encoder_and_decoder_updated=True,
                native_gradient_admission=native_gradient_admission,
                AMP_overflow_attempts=native_gradient_admission['AMP_overflow_attempts'],
                timing_includes_native_AMP_retries_and_gradient_admission=True,
                peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),
                measurement=measurement.report,project_resource_snapshot=require_project_budget())
            if iteration==0:warmup=row
            else:measurements.append(row)
            new_json(output/('update_'+str(iteration)+'.json'),row)
            print(json.dumps(dict(phase='native_CP_clone_calibration',physical_batch=physical_batch,
                iteration=iteration,full_step_seconds=row['full_step_seconds'],CP_flags=flags)),flush=True)
            del before,batch
        transport=getattr(trainer,'_online_native_transport',{})
        if transport.get('raw_events',0)<=0 or transport.get('native_support_voxels',0)<=0:
            raise ValueError('Actual CP transport was not observed; no native CP smoke admission')
        report=dict(format='v24_actual_native_CP_full128_clone_calibration_trial_v1',debug=True,
            production_epochs=250,production_updates=0,clone_optimizer_updates=4,
            cold_warmup_excluded=True,measurements=measurements,warmup=warmup,
            physical_batch=physical_batch,baseline_physical_batch=baseline_batch,
            original_plans_unchanged=True,patch_size=[128]*3,cp_probability=.5,
            full105_training_loader=True,ordinary26_validation_loader=True,
            native_CP_transport=transport,original_model_architecture_unchanged=True,
            native_trainable_parameters=sum(p.numel() for p in parameters.values()),
            GPU=torch.cuda.get_device_name(),resources=snapshot(),project_resource_contract=resource_contract(),
            accepted=True,bank_sha256=native['bank_sha256'])
        new_json(output/'trial.json',report)
    except torch.cuda.OutOfMemoryError as error:
        # A measured larger-batch rejection is distinct from an unexpected
        # failure. It never changes the original production batch or model.
        report=dict(format='v24_actual_native_CP_full128_clone_calibration_trial_v1',debug=True,
            accepted=False,rejection='actual_torch_CUDA_OutOfMemoryError',error=str(error),
            physical_batch=physical_batch,baseline_physical_batch=baseline_batch,
            patch_size=[128]*3,cp_probability=.5,original_model_architecture_unchanged=True,
            production_updates=0,completed_clone_optimizer_updates=len(measurements)+(warmup is not None),
            bank_sha256=native['bank_sha256'],project_resource_contract=resource_contract(),
            peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),resources=snapshot())
        new_json(output/'trial.json',report)
    finally:
        # These are only this isolated clone's own ordinary nnUNet augmenters.
        for loader in (train_loader,val_loader):
            finish=getattr(loader,'_finish',None)
            if finish is not None:finish()
    return output/'trial.json'


def calibrate_native(native_path,*,gpu=1):
    """Measure the fixed baseline batch and double-sized native clone batch."""
    from tools.local_cnn_device import select
    select(gpu)
    if gpu!=1:raise ValueError('This isolated native calibration is assigned GPU1')
    native,bank=admit_native(native_path);require_project_budget()
    root=Path(native['root']);final=root/'calibration.json'
    if final.exists():raise FileExistsError('Completed native calibration is immutable')
    output=root/('native_clone_calibration_'+str(time.time_ns()));output.mkdir()
    original=bank['baseline']['physical_batch'];reports=[]
    for batch in (original,original*2):
        trial=output/('batch_'+str(batch));trial.mkdir()
        env=environment(native);env['nnUNet_results']=str(trial/'nnUNet_results')
        command=[sys.executable,'-B','-c','from hiercp_v1x.v24_nnunet_cp import _calibrate_native_worker; import sys; _calibrate_native_worker(sys.argv[1],int(sys.argv[2]),sys.argv[3])',str(native_path),str(batch),str(trial)]
        with (trial/'clone.log').open('x',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            status=child.wait()
        if status:
            failure=dict(batch=batch,status=status,log=str(trial/'clone.log'),production_updates=0)
            new_json(trial/'failure.json',failure)
            raise RuntimeError('Actual native clone calibration failed; production was not started: '+str(trial/'clone.log'))
        report=read(trial/'trial.json')
        validate_native_trial(report,baseline_batch=original,bank_sha256=native['bank_sha256'])
        reports.append(dict(path=str(trial/'trial.json'),sha256=sha(trial/'trial.json'),report=report))
    new_json(final,dict(format='v24_actual_native_CP_full128_clone_calibration_v1',debug=True,
        source_files_sha256=native['source_files_sha256'],bank_sha256=native['bank_sha256'],trials=reports,
        production_updates=0,production_physical_batch=original,measured_physical_batches=[original,original*2],
        selection='historical baseline physical batch retained for comparison; larger batch measured separately',
        production_epochs=250,production_cp_probability=.5,project_resource_contract=resource_contract()))
    return final


def validate_native_trial(report,*,baseline_batch,bank_sha256):
    if (report['format']!='v24_actual_native_CP_full128_clone_calibration_trial_v1'
            or report['debug'] is not True or report['bank_sha256']!=bank_sha256
            or report['patch_size']!=[128]*3 or report['cp_probability']!=.5
            or report['original_model_architecture_unchanged'] is not True
            or report['production_updates']!=0 or report['baseline_physical_batch']!=baseline_batch
            or report['project_resource_contract']!=resource_contract()
            or report['physical_batch'] not in (baseline_batch,baseline_batch*2)):
        raise ValueError('Actual full native fixed-baseline clone trial contract required')
    if report['accepted'] is False:
        if (report.get('rejection')!='actual_torch_CUDA_OutOfMemoryError'
                or report['physical_batch']==baseline_batch):
            raise ValueError('The original production batch must pass; only an actual larger-batch CUDA OOM may be rejected')
        return report
    if (report['accepted'] is not True or len(report['measurements'])!=3
            or report['clone_optimizer_updates']!=4
            or report['native_CP_transport'].get('raw_events',0)<=0
            or report['native_CP_transport'].get('native_support_voxels',0)<=0
            or report['full105_training_loader'] is not True or report['ordinary26_validation_loader'] is not True):
        raise ValueError('Three complete actual native updates and nonempty real CP transport required')
    for row in [report['warmup'],*report['measurements']]:
        if (row['input_shape'][0]!=report['physical_batch'] or row['input_shape'][-3:]!=[128]*3
                or not math.isfinite(row['loss']) or row['gradient_finite'] is not True
                or row['present_gradient_tensors']<=0 or row['changed_parameter_tensors']<=0
                or row['encoder_and_decoder_updated'] is not True
                or row['project_resource_snapshot']['actual_process_and_children_RSS_bytes']>PROJECT_RSS_BYTES):
            raise ValueError('Native full-shape loss/gradient/update/resource evidence failed')
    return report


def admit_native_calibration(native):
    path=Path(native['root'])/'calibration.json';proof=read(path)
    expected=native['baseline']['physical_batch']
    if (proof['format']!='v24_actual_native_CP_full128_clone_calibration_v1'
            or proof['bank_sha256']!=native['bank_sha256'] or proof['source_files_sha256']!=native['source_files_sha256']
            or proof['production_physical_batch']!=expected or proof['production_epochs']!=250
            or proof['production_cp_probability']!=.5 or proof['project_resource_contract']!=resource_contract()):
        raise ValueError('Actual fixed-baseline native CP calibration proof required')
    for row in proof['trials']:
        if sha(row['path'])!=row['sha256'] or read(row['path'])!=row['report']:
            raise ValueError('Actual native trial evidence changed')
        validate_native_trial(row['report'],baseline_batch=expected,bank_sha256=native['bank_sha256'])
    if {row['report']['physical_batch'] for row in proof['trials']}!={expected,expected*2}:
        raise ValueError('Both native physical batch candidates must be measured')
    return path


def train(native_path, *, gpu=1, resume=False):
    from tools.local_cnn_device import select
    select(gpu)
    if gpu != 1: raise ValueError('GPU1 assigned to actual frozen-best GNN segmentation arm')
    require_project_budget()
    native,bank = admit_native(native_path)
    calibration=admit_native_calibration(native)
    root=Path(native['root']); result=root/'nnUNet_results'/bank['baseline']['dataset_name']/f'{TRAINER}__{PLANS}__3d_fullres'/'fold_0'
    if result.exists() and not resume: raise FileExistsError('Existing segmentation run requires explicit native resume')
    if resume and not (result/'checkpoint_latest.pth').is_file(): raise ValueError('Native resume checkpoint missing; no fresh fallback')
    command=[sys.executable,'-B','-m','nnunetv2.run.run_training','730','3d_fullres','0','-tr',TRAINER,'-p',PLANS]
    if resume: command.append('--c')
    stamp=str(time.time_ns()); log=root/('train_'+stamp+'.log')
    new_json(root/('training_started_'+stamp+'.json'),dict(format=FORMAT,command=command,epochs=250,
        cp_probability=.5,split=bank['split'],bank_sha256=native['bank_sha256'],resume=resume,debug=False,
        native_calibration_sha256=sha(calibration)))
    with log.open('x',encoding='utf8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=environment(native),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True)
        status=child.wait()
    if status: raise RuntimeError(f'Native segmentation failed ({status}); preserved log {log}')
    checkpoint=result/'checkpoint_final.pth'
    new_json(root/('training_complete_'+stamp+'.json'),dict(format=FORMAT,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),bank_sha256=native['bank_sha256'],epochs=250,log=str(log)))
    return checkpoint


def predict(native_path, *, inventory_path, output, gpu=1):
    from tools.local_cnn_device import select
    select(gpu)
    if gpu != 1: raise ValueError('GPU1 assigned downstream evaluation')
    native,bank=admit_native(native_path);meta=read(inventory_path)
    if bank['split'] != meta['split'] or sha(native['bank']) != native['bank_sha256']:
        raise ValueError('Frozen bank/evaluation cohort changed')
    target=Path(output).resolve()
    if target.exists():raise FileExistsError('Fresh prediction output required')
    target.mkdir(parents=True); inputs=target/'inputs';inputs.mkdir()
    raw={row['case_id']:row for row in meta['raw_records']}
    hashes={}
    for case in bank['split']['outer_val']:
        source=Path(raw[case]['image'])
        if sha(source)!=raw[case]['image_sha256']:raise ValueError('Original outer26 CT changed')
        dest=inputs/(case+'_0000.nii.gz');shutil.copy2(source,dest);hashes[case]=sha(dest)
    checkpoint=Path(native['root'])/'nnUNet_results'/bank['baseline']['dataset_name']/f'{TRAINER}__{PLANS}__3d_fullres'/'fold_0'/'checkpoint_final.pth'
    checkpoint_sha=sha(checkpoint)
    command=[sys.executable,'-B','-c','from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()','-i',str(inputs),
        '-o',str(target/'predictions'),'-d','730','-c','3d_fullres','-f','0','-tr',TRAINER,'-p',PLANS,'-chk','checkpoint_final.pth']
    with (target/'predict.log').open('x',encoding='utf8') as stream:
        status=subprocess.run(command,cwd=ROOT,env=environment(native),stdout=stream,stderr=subprocess.STDOUT).returncode
    if status:raise RuntimeError('Prediction failed; original artifacts and new log preserved')
    new_json(target/'prediction_complete.json',dict(format=FORMAT,checkpoint_sha256=checkpoint_sha,
        CT_inputs_sha256=hashes,GT_passed_to_predictor=False,CP_at_inference=False,
        outer26_role='segmentation validation, not independent test',
        predictions_sha256={case:sha(target/'predictions'/(case+'.nii.gz')) for case in bank['split']['outer_val']}))
    return target/'predictions'


def evaluate(native_path, *, inventory_path, predictions, output, basic_predictions=None):
    """Use the established paired full26 evaluator and exact historical baseline."""
    from tools.online_eval_v2 import parser, evaluate as evaluate_full
    native=read(native_path);bank=validate_bank(read(native['bank']));meta=read(inventory_path)
    if bank['split']!=meta['split']:raise ValueError('Evaluation cohort changed')
    predictions=Path(predictions).resolve(strict=True)
    proof=read(predictions.parent/'prediction_complete.json')
    expected=bank['split']['outer_val']
    if (proof['GT_passed_to_predictor'] is not False or proof['CP_at_inference'] is not False
            or set(proof['predictions_sha256'])!=set(expected)
            or any(sha(predictions/(case+'.nii.gz'))!=proof['predictions_sha256'][case] for case in expected)):
        raise ValueError('Complete CT-only full26 native predictions must be proved')
    pre=Path(bank['baseline']['preprocessed']);legacy_project=pre.parents[4]
    basic=(Path(basic_predictions) if basic_predictions else pre.parent.parent/'nnUNet_results'/pre.name/
        f'nnUNetTrainer_250epochs_OnlineBasicCP__{PLANS}__3d_fullres'/'fold_0'/'validation')
    args=parser().parse_args(['--project',str(legacy_project),'--basic-validation',str(basic),
        '--hier-validation',str(predictions),'--hier-trainer',TRAINER,'--output',str(output)])
    result=evaluate_full(args)
    new_json(Path(output)/'v24_comparison_scope.json',dict(format=FORMAT,
        frozen_GNN_pin=bank['pin'],bank_sha256=native['bank_sha256'],prediction_receipt_sha256=sha(predictions.parent/'prediction_complete.json'),
        cohort=expected,validation_patients=26,independent_test_available=False,
        cp_probability_equal=.5,segmentation_epochs_equal=250,native_patch_equal=[128,128,128],
        donor_policy_equal=False,donor_policy_difference='historical Basic source policy versus independent inner-train v23 fixed donor per recipient',
        historical_prediction_checkpoint_provenance='retrospective existing-byte evaluation; old inference loaded tensor provenance not established here'))
    return result
