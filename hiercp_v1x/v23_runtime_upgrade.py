"""Verified same-policy singleton continuation across an input-runtime upgrade.

No source files or processes are changed. Publication requires an independently
verified paused source, numerical equivalence and real full128 engine evidence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import canonical_hash
from .u_bridge_training import atomic_save, cpu_copy, digest
from .v23_training import (FORMAT, EXPECTED_PARAMETERS, _file_sha256, _validate_progress,
    _write_new, execution_identity, training_binding, remaining_training_updates)

PERFORMANCE_KEYS = frozenset(('batch_calibration', 'resume_checkpoint', 'pause_file',
    'prefetch_cpu_chunks', 'pin_cpu_batches', 'memoize_cpu_geometry',
    'cache_stage_admission', 'geometry_resident_gib_per_rank',
    'memoize_fixed_views', 'persistent_cpu_workers',
    'l0_edge_workspace_mib', 'l0_edge_adapter_sha256'))
NUMERICAL_KEYS = ('model', 'optimizer', 'scheduler', 'scaler', 'shuffle_generator', 'rank_rng')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _request_check(request):
    _require(canonical_hash({k: v for k, v in request.items() if k != 'request_sha256'})
        == request['request_sha256'], 'Exact request SHA required')


def _load_checkpoint(path, identity_sha):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'Regular source checkpoint required')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    _require(saved['format'] == FORMAT and saved['identity_sha256'] == identity_sha
        and saved['content_sha256'] == digest({k: v for k, v in saved.items() if k != 'content_sha256'}),
        'Source checkpoint ownership/content differs')
    return saved


def upgrade_single_gpu_checkpoint(source_checkpoint, source_identity, destination_request,
        config, calibration, output, *, source_proof, population, source_best_checkpoint=None,
        debug=False):
    """Keep all numerical state, candidate membership and partial-epoch cursors."""
    required = ('source_commit', 'source_request', 'source_files_sha256',
        'source_checkpoint_file_sha256', 'source_identity_file_sha256',
        'geometry_receipt_sha256', 'calibration_receipt_sha256',
        'core_equations_preserved', 'original_model_parameters', 'clean_pause_verified',
        'new_runtime_full128_backward_verified', 'numerical_equivalence_receipt_sha256')
    _require(all(key in source_proof for key in required), 'Complete explicit upgrade proof required')
    _require(source_proof['clean_pause_verified'] is True
        and source_proof['core_equations_preserved'] is True
        and source_proof['new_runtime_full128_backward_verified'] is True
        and (debug or source_proof['original_model_parameters'] == EXPECTED_PARAMETERS),
        'Paused original-model numerical/full128 evidence required')
    paths = dict(source_checkpoint_file_sha256=Path(source_checkpoint),
        source_identity_file_sha256=Path(source_identity))
    for key, path in paths.items():
        _require(path.is_file() and not path.is_symlink() and _file_sha256(path) == source_proof[key],
            'Immutable source file SHA differs: ' + key)
    for key in ('geometry_receipt_sha256', 'calibration_receipt_sha256',
                'numerical_equivalence_receipt_sha256'):
        value = source_proof[key]
        _require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Exact upgrade SHA required: ' + key)
    ownership = json.loads(paths['source_identity_file_sha256'].read_text(encoding='utf8'))
    old = ownership['binding']
    _require(ownership['identity_sha256'] == digest(old), 'Original ownership binding differs')
    saved = _load_checkpoint(source_checkpoint, ownership['identity_sha256'])
    state = saved['state']
    engineering_document = None
    edge_keys = {'l0_edge_workspace_mib', 'l0_edge_adapter_sha256'}
    if edge_keys.intersection(config['v23_runtime']):
        _require(edge_keys.issubset(config['v23_runtime']), 'Both explicit edge execution keys are required')
        _require(source_proof.get('engineering_contract_verified') is True
            and source_proof.get('native_strict_equivalence_passed') is False
            and source_proof.get('scientific_equivalence_established') is False,
            'Execution engineering proof must preserve original strict/scientific FAIL')
        _require(source_proof['engineering_contract_receipt_sha256']
            == calibration['engineering_contract']['receipt_sha256']
            and source_proof['engineering_integration_smoke_receipt_sha256']
            == calibration['engineering_integration_smoke']['receipt_sha256'],
            'Actual engineering certificate and integrated smoke raw SHA differ')
        engineering_document = require_engineering_calibration(calibration, config['v23_runtime'],
            destination_request['gpus'][0], Path(__file__).resolve().parents[1],
            source_checkpoint_sha256=source_proof['source_checkpoint_file_sha256'],
            source_numerical_state_sha256={key: digest(saved[key]) for key in NUMERICAL_KEYS},
            execution_request=destination_request)
    _require(state['status'] == 'PAUSED' and state['phase'] != 'complete' and state['epoch'] <= 40,
        'Only the explicitly paused unfinished experiment can be upgraded')
    source_request = source_proof['source_request']
    _request_check(source_request); _request_check(destination_request)
    _require(old['identity']['request_sha256'] == source_request['request_sha256']
        and old['identity']['source'] == source_request['source']
        and source_proof['source_files_sha256'] == source_request['source'], 'Original source proof differs')
    _require(source_request['gpus'] == destination_request['gpus']
        and len(destination_request['gpus']) == 1 and destination_request['gpus'][0] in (1, 5, 6)
        and old['data_parallel_world_size'] == 1 and len(saved['rank_rng']) == 1
        and old['physical_patient_batch'] == 4 and old['physical_candidate_batch'] == 32,
        'Same GPU/world1/physical4/chunk32 upgrade required')
    _require(old['epochs'] == 40 and old['debug'] is debug
        and old['population'] == population.manifest(), 'Complete population/forty epochs must remain identical')
    for key in ('model', 'graph', 'training', 'ct_clip', 'runtime', 'labels', 'cache', 'generation'):
        _require(config.get(key) == old['config'].get(key), 'Numerical configuration changed: ' + key)
    old_runtime = {k: v for k, v in old['config']['v23_runtime'].items() if k not in PERFORMANCE_KEYS}
    new_runtime = {k: v for k, v in config['v23_runtime'].items() if k not in PERFORMANCE_KEYS}
    _require(old_runtime == new_runtime, 'Target policy, training gates or resource/model contract changed')
    _require(calibration['world_size'] == 1 and calibration['request_sha256'] == destination_request['request_sha256']
        and calibration['physical_GPUs'] == destination_request['gpus']
        and calibration['selected_physical_patient_batch'] == 4
        and calibration['selected_physical_candidate_batch'] == 32
        and calibration['initial_state_sha256'] == old['initial_state_sha256']
        and calibration['measured_full_P_U128_backward'] is True
        and calibration['original_model_and_RNG_preserved'] is True,
        'New runtime same-shape actual full128 calibration evidence required')
    _validate_progress(state, old['train_cases'], old['val_cases'], 4, 40,
        SimpleNamespace(last_epoch=saved['scheduler']['last_epoch']), 1)
    from .v23_training import validate_target_state
    validate_target_state(population, state['targets'],
        policy=config['v23_runtime']['target_selection_policy'], active_u=state['curriculum']['active_u'])
    for slot in saved['optimizer']['state'].values():
        step = slot['step']
        _require(int(step.item() if torch.is_tensor(step) else step) == state['updates'],
            'Source optimizer history differs from successful updates')
    continuation = dict(source_commit=source_proof['source_commit'],
        source_checkpoint=str(Path(source_checkpoint).resolve()), source_identity=str(Path(source_identity).resolve()),
        source_checkpoint_file_sha256=source_proof['source_checkpoint_file_sha256'],
        source_checkpoint_content_sha256=saved['content_sha256'],
        source_identity_sha256=ownership['identity_sha256'], source_request_sha256=source_request['request_sha256'],
        source_world_size=1, source_rank_rng_index=0, destination_world_size=1,
        destination_physical_GPU=destination_request['gpus'][0], inherited_epoch=state['epoch'],
        inherited_phase=state['phase'], inherited_updates=state['updates'],
        geometry_receipt_sha256=source_proof['geometry_receipt_sha256'],
        calibration_receipt_sha256=source_proof['calibration_receipt_sha256'],
        runtime_upgrade=True, numerical_equivalence_receipt_sha256=source_proof['numerical_equivalence_receipt_sha256'])
    if engineering_document is not None:
        _require(source_proof['numerical_equivalence_receipt_sha256']
            == engineering_document['evidence_refs']['original_strict_failure']['raw_sha256'],
            'Original immutable strictFAIL raw reference differs')
        continuation.update(engineering_contract_verified=True,
            engineering_contract_receipt_sha256=source_proof['engineering_contract_receipt_sha256'],
            engineering_integration_smoke_receipt_sha256=source_proof['engineering_integration_smoke_receipt_sha256'],
            root_drift_review_receipt_sha256=engineering_document['actual_drift_review_ref']['raw_sha256'],
            reused_native_performance_receipt_sha256=engineering_document['evidence_refs']['native_performance']['raw_sha256'],
            native_strict_equivalence_passed=False,scientific_equivalence_established=False,
            original_strict_FAIL_unchanged=True,
            observed_numeric_envelope_status=engineering_document['observed_numeric_envelope_status'])
    identity = execution_identity(destination_request, continuation)
    config = copy.deepcopy(config); config['v23_runtime']['batch_calibration'] = cpu_copy(calibration)
    binding = training_binding(config, population, identity=identity, physical_patient_batch=4,
        physical_candidate_batch=32, world_size=1, workers=config['v23_runtime']['workers'],
        initial_hash=old['initial_state_sha256'], debug=debug)
    fork = cpu_copy(saved); fork['identity_sha256'] = digest(binding)
    fork['state']['handoff'] = continuation
    _require(digest({k: v for k, v in fork['state'].items() if k != 'handoff'})
        == digest({k: v for k, v in state.items() if k != 'handoff'}), 'Partial epoch or candidate state changed')
    for key in NUMERICAL_KEYS:
        _require(digest(fork[key]) == digest(saved[key]), 'Numerical state changed: ' + key)
    fork.pop('content_sha256'); fork['content_sha256'] = digest(fork)
    best_fork = None; best_proof = None
    if state['best'] is not None:
        _require(source_best_checkpoint is not None and 'source_best_checkpoint_file_sha256' in source_proof,
            'Actual source BEST required; latest cannot substitute')
        path = Path(source_best_checkpoint)
        _require(_file_sha256(path) == source_proof['source_best_checkpoint_file_sha256'], 'Source BEST file SHA differs')
        best = _load_checkpoint(path, ownership['identity_sha256'])
        _require(best['state']['best'] == state['best'] and len(best['rank_rng']) == 1,
            'Actual source BEST record differs')
        best_fork = cpu_copy(best); best_fork['identity_sha256'] = digest(binding)
        best_fork['state']['handoff'] = continuation
        best_fork.pop('content_sha256'); best_fork['content_sha256'] = digest(best_fork)
        _require(all(digest(best_fork[key]) == digest(best[key]) for key in NUMERICAL_KEYS), 'Actual BEST state changed')
        paths['source_best_checkpoint_file_sha256'] = path
        best_proof = dict(source_checkpoint=str(path.resolve()),
            source_file_sha256=source_proof['source_best_checkpoint_file_sha256'], model_sha256=digest(best['model']),
            best_record=cpu_copy(best['state']['best']), actual_best_weights_preserved=True, latest_weights_substituted=False)
    for key, path in paths.items():
        _require(_file_sha256(path) == source_proof[key], 'Source changed during upgrade: ' + key)
    output = Path(output)
    _require(not output.is_symlink(), 'Fresh regular output required')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing results preserved; fresh upgrade output required')
    receipt = dict(format='v23_verified_same_policy_singleton_runtime_upgrade_v1',
        destination_request_sha256=destination_request['request_sha256'], destination_identity=identity,
        destination_binding_sha256=digest(binding), continuation=continuation, source_proof=cpu_copy(source_proof),
        preserved_state_sha256={key: digest(saved[key]) for key in NUMERICAL_KEYS},
        preserved_progress_sha256=digest({k: v for k, v in state.items() if k != 'handoff'}),
        inherited_best_checkpoint=best_proof, source_best_record=cpu_copy(state['best']),
        destination_checkpoint_content_sha256=fork['content_sha256'],
        total_target_epochs=40, completed_epochs=len(state['history']),
        remaining_planned_updates=remaining_training_updates(state, old['train_cases'], 4, 1),
        no_new_optimizer_update=True, original_files_written=False, debug=debug)
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output/'training_identity.json', dict(identity_sha256=digest(binding), binding=cpu_copy(binding)))
    if best_fork is not None: atomic_save(output/'checkpoint_best.pt', best_fork)
    atomic_save(output/'checkpoint_latest.pt', fork)
    _write_new(output/'handoff_receipt.json', receipt)
    return receipt


import hashlib
import json
import math
from pathlib import Path

OLD_COMMIT = '142bedf68ec58bd56be4e8ec147477dca435df9c'
FEEDING_COMMIT = 'd3f558a77d96c3ffb98e645615126bf156035031'
ADAPTER_SHA256 = 'ead4ec316d19cef4cf6388476cb792f275f0e73f174f15b58bea70e5ae09b6fb'
MODEL_SHA256 = '004544962db2ed2d39efcb7a2ef7ee2972ad31b5769d804b5e4c88d866c47049'
ARCHIVE_SHA256 = '5157bafe641e9189824826532a3b055ea560f5c5dfc299374842a3bd1d20a22e'
NATIVE_WORKER_SHA256 = 'ee876629592dadb8b559c8c38cb56c4edaddb99b53af99764af5129ea6197cfe'
CPU_INPUT_WORKER_SHA256 = '1715913ee98616477e640c4e7846f97f779a235ceebf6095f5517cc7f5619354'
NOISE_PROTOCOL_SHA256 = '1f71dfcb168629a43010db4f623a5b32e74cb3f1929904390e8a4423f37b0220'
NOISE_PROTOCOL_RAW_SHA256 = '281754bc81e7d11a98fbfed2831b85e6e2ae317d058c589ecea30f42b9e7410d'
FEEDING_FILES = ('hiercp_v1x/v23_training.py', 'hiercp_v1x/v23_inputs.py',
    'hiercp_v1x/v23_data.py', 'hiercp_v1x/v23_geometry.py', 'hiercp_v1x/v23_targets.py')
EXECUTION_RUNTIME_FIELDS = ('l0_edge_workspace_mib', 'l0_edge_adapter_sha256')
PROOF_NAMES = ('original_strict_failure', 'CPU_input', 'core_equations', 'operator_DEBUG',
    'native_performance', 'noise_assessment', 'source_pause', 'source_request')
STATE_KEYS = ('model', 'optimizer', 'scheduler', 'scaler', 'shuffle_generator', 'rank_rng')
ENGINEERING_RELEASE_FILES = frozenset((
    'config/v23_all_p_native.json', 'hiercp_v1x/v23_data.py', 'hiercp_v1x/v23_geometry.py',
    'hiercp_v1x/v23_training.py', 'tools/run_v23_all_p.py', 'tests/test_v23_data.py',
    'tests/test_v23_geometry.py', 'tests/test_v23_training.py', 'docs/v23_all_p_native_20261009.txt',
    'config/v23_gpu1_prefix.json', 'config/v23_gpu5_hard_score.json', 'config/v23_gpu6_score_mix.json',
    'hiercp_v1x/v23_targets.py', 'tests/test_v23_targets.py', 'tests/test_v23_single_gpu.py',
    'docs/v23_independent_targets_20261009.txt', 'hiercp_v1x/v23_inputs.py',
    'hiercp_v1x/v23_runtime_upgrade.py', 'tests/test_v23_inputs.py',
    'tests/test_v23_runtime_upgrade.py', 'docs/v23_resource_optimization_20261009.txt',
    'hiercp_v1x/v23_edge_execution.py'))
CERTIFICATE_KEYS = frozenset(('format', 'status', 'engineering_contract_verified',
    'engineering_continuation_authorized', 'native_strict_equivalence_passed',
    'scientific_equivalence_established', 'original_strict_FAIL_unchanged',
    'observed_numeric_envelope_status', 'chosen_workspace_MiB', 'feeding_commit',
    'old_commit', 'adapter_source_sha256', 'immutable_feeding_files_sha256',
    'source_pause_proofs', 'evidence_refs', 'engineering_basis_sha256',
    'actual_drift_review', 'actual_drift_review_ref', 'certificate_sha256',
    'production_launch_performed', 'total_training_epochs', 'certificate_scope'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def valid_sha(value, size=64):
    return isinstance(value, str) and len(value) == size and all(c in '0123456789abcdef' for c in value)


def read_proof(reference):
    require(set(reference) == {'path', 'raw_sha256'} and valid_sha(reference['raw_sha256']),
        'Exact closed-schema raw file reference required')
    path = Path(reference['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Regular absolute evidence file required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference['raw_sha256'], 'Evidence raw bytes changed: ' + str(path))
    return json.loads(raw)


def _clean_comparison(row):
    require(row['RNG_exact'] is True and row['experimental_invariants_valid'] is True
        and row['comparison']['nonfinite_leaves'] == 0, 'Actual native RNG/finite/structure violation')
    require(not [v for v in row['comparison']['structural_mismatches'] if not v['path'].startswith('root/ranked_metrics')],
        'Non-ranking native structural discrepancy is an engineering hard failure')
    if row['kind'].startswith('train'):
        require(row['all_1085_gradients_verified'] is True and row['all_10434532_gradient_elements_verified'] is True,
            'All actual original parameter gradients required')


def _population(proof):
    p = proof['population']
    require(p['DEBUG'] is False and p['hidden_subset'] is False and p['data_fraction'] == 1.0
        and p['retained_observations'] == 14102 and p['all_observed_P_active_from_start'] is True
        and p['P_as_negative'] is False and p['U_bank_size'] == 128
        and p['independent_train_only_donor_fixed_per_case'] is True and p['donor_redraw'] is False,
        'Complete original P/U/donor contract required')
    require(p['counts']['inner_train']['observed_P'] == 527 and p['counts']['inner_train']['cases'] == 84
        and p['counts']['inner_val']['observed_P'] == 135 and p['counts']['inner_val']['cases'] == 21
        and len(proof['train_cases']) == 65 and len(proof['val_cases']) == 21,
        'All native train/validation patients and P observations required')


def engineering_basis(evidence_refs, chosen_workspace_MiB, *, proof_reader=None):
    """Verify factual engineering prerequisites; this function approves nothing."""
    require(set(evidence_refs) == set(PROOF_NAMES), 'Exact eight closed-world evidence references required')
    reader = proof_reader or read_proof
    for reference in evidence_refs.values():
        require(set(reference) == {'path', 'raw_sha256'} and valid_sha(reference['raw_sha256']), 'Invalid immutable evidence reference')
    d = {name: reader(ref) for name, ref in evidence_refs.items()}
    require(type(chosen_workspace_MiB) is int and chosen_workspace_MiB in (64, 256, 1024), 'Explicit actually measured workspace required')
    fail, cpu, core, operator, native, noise, pause, request = (d[name] for name in PROOF_NAMES)
    require(fail['format'] == 'v23_paired_full128_explicit_numerical_failure_v1'
        and fail['production_launch_allowed'] is False and len(fail['failures']) == 24
        and fail['rtol'] == 1e-5 and fail['atol'] == 1e-6, 'Original24 strict FAIL must remain immutable')
    require(cpu['status'] == 'CPU_INPUT_BIT_EXACT_PASS' and cpu['all_original_graph_patch_index_values_bit_exact'] is True
        and cpu['worker_source_sha256'] == CPU_INPUT_WORKER_SHA256
        and cpu['all_full_P_U128_two_view_workload_preserved'] is True
        and cpu['differences'] == [] and cpu['layout_differences'] == [], 'Full CPU graph/patch/index/stride identity required')
    require(len(cpu['GPU_reports']) == 3 and {v['GPU_source'] for v in cpu['GPU_reports']} == {1, 5, 6}
        and sum(v['input_chunks_compared'] for v in cpu['GPU_reports']) == 267
        and sum(v['upper_graphs_compared'] for v in cpu['GPU_reports']) == 24
        and all(v['all_chunks_and_upper_bit_exact'] for v in cpu['GPU_reports']), 'Actual267 local chunks/24upper graphs required')
    require(core['source_commit'] == OLD_COMMIT and core['destination_commit'] == FEEDING_COMMIT
        and core['numerical_source_preserved'] is True and len(core['checks']) == 22
        and all(valid_sha(v) for v in core['checks'].values()), 'Exact22 source/equation preservation checks required')
    require(operator['format'] == 'v23_original_L0_instance_workspace_local_DEBUG_UNIT_receipt_v1'
        and operator['status'] == 'PASS' and operator['tests_run'] == 8
        and operator['failures'] == operator['errors'] == operator['skipped'] == 0
        and operator['native_CT_loaded'] is False and operator['native_full_model_optimizer_gate_claimed'] is False
        and operator['source_model_sha256'] == MODEL_SHA256 and operator['sealed_archive_sha256'] == ARCHIVE_SHA256
        and operator['files']['v23_edge_execution.py']['sha256'] == ADAPTER_SHA256, 'Original exact adapter/operator DEBUG proof required')
    coverage = operator['coverage']
    require(coverage['original_L0_convolutions_named_gradients_CPU'] == 48
        and coverage['dropout_mask_exact'] is True and coverage['next_RNG_exact'] is True
        and coverage['module_parameter_buffer_state_bytes_preserved'] is True and coverage['upper64MiB_unchanged'] is True,
        'Exact original ideal operator/global softmax/dropout/state contract required')
    require(native['format'] == 'v23_L0_workspace_performance_only_native_noise_experiment_v1'
        and native['status'] == 'EXPERIMENTAL_PERFORMANCE_MEASUREMENT_COMPLETE_SCIENCE_REVIEW_REQUIRED'
        and native['worker_source_sha256'] == NATIVE_WORKER_SHA256
        and native['complete_performance_workspace_MiB'] == [64, 256, 1024]
        and native['old_commit'] == OLD_COMMIT and native['new_commit'] == FEEDING_COMMIT
        and native['adapter_source_sha256'] == ADAPTER_SHA256 and native['original_strict_FAIL_unchanged'] is True
        and native['native_strict_equivalence_passed'] is False and native['production_admission'] is False
        and native['production_optimizer_updates'] == native['source_checkpoint_updates'] == 0
        and native['source_checkpoints_unchanged'] is True and native['source_files_written'] is False,
        'Actual performance evidence may not be relabelled numerical/scientific PASS')
    require(native['physical_patient_batch'] == 4 and native['candidate_chunk'] == 32
        and native['original_model_parameters'] == 10434532 and native['full_allP_U128_two_views'] is True
        and native['all_steady_training_repeats_same_actual_source_epoch'] is True
        and native['all_repeats_restore_identical_model_optimizer_scaler_RNG'] is True,
        'Complete actual native shape/epoch/source-state performance contract required')
    require(native['original_paired_strict_FAIL']['failure_sha256'] == evidence_refs['original_strict_failure']['raw_sha256']
        and native['CPU_input_bit_equivalence_proof']['receipt_sha256'] == evidence_refs['CPU_input']['raw_sha256']
        and native['CPU_input_bit_equivalence_proof']['document'] == cpu, 'Performance body must bind actual strict FAIL/CPU raw bytes')
    require(noise['format'] == 'v23_observed_native_noise_assessment_v1' and noise['status'] in ('INCONCLUSIVE', 'OBSERVED_ENVELOPE_COMPATIBLE')
        and noise['protocol_sha256'] == NOISE_PROTOCOL_SHA256
        and noise['protocol_raw_sha256'] == NOISE_PROTOCOL_RAW_SHA256
        and noise['server_only_assessor'] is True and noise['CUDA_context_initialized'] is False
        and noise['model_instantiated'] is False and noise['optimizer_updates'] == 0
        and valid_sha(noise['original_parameter_schema_sha256'])
        and noise['original_strict_FAIL_unchanged'] is True and noise['native_strict_equivalence_passed'] is False
        and noise['scientific_equivalence_passed'] is False and noise['production_admission'] is False
        and noise['hard_failures'] == [] and noise['performance_receipt_raw_sha256'] == evidence_refs['native_performance']['raw_sha256']
        and noise['final_CPU_input_proof_raw_sha256'] == evidence_refs['CPU_input']['raw_sha256']
        and noise['source_pause_receipt_raw_sha256'] == evidence_refs['source_pause']['raw_sha256']
        and noise['source_request_raw_sha256'] == evidence_refs['source_request']['raw_sha256'],
        'Actual formal assessment/hard-invariant status and all raw bindings required')
    require(len(noise['groups']) == 6
        and {(row['physical_GPU'], row['partition']) for row in noise['groups']}
            == {(gpu, partition) for gpu in (1, 5, 6) for partition in ('training', 'validation')},
        'Full frozen-assessor original1085 train/validation coverage required')
    require(pause['commit'] == OLD_COMMIT and pause['all_three_workers_naturally_ended'] is True
        and pause['all_original_tmux_shells_preserved'] is True, 'Exact safe original paused sources required')
    proofs = {str(row['gpu']): row['checkpoint_proof'] for row in pause['variants']}
    require(set(proofs) == {'1', '5', '6'} and {str(row['gpu']): row['checkpoint_proof'] for row in request['variants']} == proofs,
        'Each independent source must bind its distinct actual paused state')
    feeding = request['new_deployment']['request']['hashes']
    require(request['new_commit'] == FEEDING_COMMIT and request['old_commit'] == OLD_COMMIT
        and request['worker_sha256'] == NATIVE_WORKER_SHA256
        and request['adapter_source_sha256'] == ADAPTER_SHA256, 'Measured feeding and exact adapter source required')
    measured = {(row['physical_GPU'], row['workspace_MiB']): row for row in native['GPU_reports']}
    require(len(native['GPU_reports']) == len(measured) == 9
        and set(measured) == {(gpu, mode) for gpu in (1, 5, 6) for mode in (64, 256, 1024)},
        'Exact complete nine immutable native workspace/GPU measurements required')
    selected = {}
    for gpu in (1, 5, 6):
        proof = proofs[str(gpu)]
        _population(proof)
        require(proof['status'] == 'PAUSED' and proof['world_size'] == proof['rank_rng_count'] == 1
            and proof['physical_patient_batch'] == 4 and proof['physical_candidate_batch'] == 32
            and proof['total_target_epochs'] == 40 and proof['connected_parameter_tensors'] == 1085
            and set(proof['numerical_state_sha256']) == set(STATE_KEYS)
            and proof['epoch'] <= 40 and proof['phase'] != 'complete', 'Exact unfinished singleton/physical4/chunk32/40epoch source required')
        for workspace in {64, chosen_workspace_MiB}:
            row = measured.get((gpu, workspace))
            require(row is not None and row['performance_measurement_complete'] is True and row['all_RNG_exact'] is True
                and row['all_1085_gradients_finite_present'] is True and row['steady_train_clone_updates'] == 3
                and row['warmup_clone_updates'] == 1 and row['validation_forward_repeats'] == 3
                and row['source_checkpoint_file_sha256'] == proof['latest_file_sha256']
                and row['source_model_optimizer_RNG_hashes'] == proof['numerical_state_sha256'],
                'All three GPUs need actual four same-source native controls plus real updates')
            require(row['peak_allocated_cuda_bytes'] <= 40 * 2**30
                and len(row['train_measurements']) == len(row['validation_measurements']) == 3,
                'Actual native chosen/reference measurements must fit40GiB without OOM')
            for sample in row['train_measurements']:
                require(sample['actual_view_epoch'] == proof['epoch'] and sample['physical_patients'] == 4
                    and sample['all_U_per_patient'] == 128 and sample['two_views'] is True
                    and sample['optimizer_updates_on_clone'] == 1 and sample['gradient']['finite'] is True
                    and sample['gradient']['missing'] == [] and sample['gradient']['gradient_present'] == 1085
                    and sample['gradient']['trainable_parameter_tensors'] == 1085
                    and sample['peak_allocated_cuda_bytes'] <= 40 * 2**30, 'Actual complete full-native train measurements required')
            for sample in row['validation_measurements']:
                require(sample['actual_view_epoch'] == 29 and sample['no_grad'] is True and sample['eval_mode'] is True
                    and sample['all_U_per_patient'] == 128 and sample['two_views'] is True
                    and sample['peak_allocated_cuda_bytes'] <= 40 * 2**30, 'Actual complete fixed29 native validation required')
            controls = row['same_mode_same_epoch_pairwise_controls']
            require(len(controls) == 12, 'Four same-source train/validation controls require12 unordered comparisons')
            for comparison in controls:
                _clean_comparison(comparison)
            admission = row['workspace_admission']
            require(admission['relation_convolutions'] == 48 and admission['workspace_bytes'] == workspace * 2**20
                and admission['adapter_sha256'] == ADAPTER_SHA256 and admission['model_source_sha256'] == MODEL_SHA256
                and admission['sealed_archive_sha256'] == ARCHIVE_SHA256
                and admission['global_workspace_changed'] is False and admission['upper_execution_changed'] is False
                and admission['full_relation_softmax_and_dropout_unchanged'] is True
                and admission['module_parameter_buffer_identity_unchanged'] is True, 'Only original48 L0 workspace execution may change')
        selected[str(gpu)] = dict(physical_GPU=gpu, physical_GPU_uuid=measured[(gpu, chosen_workspace_MiB)]['physical_GPU_uuid'],
            peak_allocated_cuda_bytes=measured[(gpu, chosen_workspace_MiB)]['peak_allocated_cuda_bytes'])
    basis = dict(format='v23_closed_world_execution_engineering_basis_v1', chosen_workspace_MiB=chosen_workspace_MiB,
        old_commit=OLD_COMMIT, feeding_commit=FEEDING_COMMIT, adapter_source_sha256=ADAPTER_SHA256,
        immutable_feeding_files_sha256={path: feeding[path] for path in FEEDING_FILES},
        source_pause_proofs=proofs, evidence_refs=evidence_refs, observed_numeric_envelope_status=noise['status'],
        original_strict_FAIL_unchanged=True, native_strict_equivalence_passed=False,
        scientific_equivalence_established=False, production_launch_performed=False,
        measured_selected_GPU_summary=selected, allowed_execution_runtime_fields=list(EXECUTION_RUNTIME_FIELDS),
        total_training_epochs=40, physical_patient_batch=4, candidate_chunk=32, world_size_per_experiment=1,
        original_model_parameters=10434532, all_trainable_parameter_tensors=1085)
    basis['engineering_basis_sha256'] = canonical_sha256(basis)
    return basis


def engineering_review_template(basis):
    """PENDING template. Actual root must inspect drift and publish a new file."""
    return dict(format='v23_root_actual_native_drift_review_v1', reviewer='/root', decision='PENDING',
        reason=None, engineering_basis_sha256=basis['engineering_basis_sha256'],
        reviewed_raw_evidence_sha256={name: ref['raw_sha256'] for name, ref in basis['evidence_refs'].items()},
        chosen_workspace_MiB=basis['chosen_workspace_MiB'], original_strict_FAIL_unchanged=True,
        observed_numeric_envelope_status=basis['observed_numeric_envelope_status'],
        scientific_equivalence_established=False, no_closed_world_contract_violation_found=None,
        per_GPU_reviews=[dict(physical_GPU=gpu, reviewed_same_source_reference64_controls=False,
            reviewed_all1085_leaf_gradients_and_actual_updates=False, reviewed_clear_gap_and_near_tie_drift=False,
            reason=None) for gpu in (1, 5, 6)])


def create_engineering_certificate(evidence_refs, chosen_workspace_MiB, actual_drift_review_ref, *, proof_reader=None):
    """Require a completed actual root review; never generate its approved value."""
    reader = proof_reader or read_proof
    basis = engineering_basis(evidence_refs, chosen_workspace_MiB, proof_reader=reader)
    review = reader(actual_drift_review_ref)
    expected = engineering_review_template(basis)
    require(set(review) == set(expected) and review['format'] == expected['format'] and review['reviewer'] == '/root'
        and review['decision'] == 'APPROVE_EXACT_EXECUTION_CONTINUATION'
        and isinstance(review['reason'], str) and review['reason'].strip()
        and review['engineering_basis_sha256'] == basis['engineering_basis_sha256']
        and review['reviewed_raw_evidence_sha256'] == expected['reviewed_raw_evidence_sha256']
        and review['chosen_workspace_MiB'] == chosen_workspace_MiB
        and review['observed_numeric_envelope_status'] == basis['observed_numeric_envelope_status']
        and review['original_strict_FAIL_unchanged'] is True and review['scientific_equivalence_established'] is False
        and review['no_closed_world_contract_violation_found'] is True, 'Actual SHA-bound root drift review is pending, incomplete or rejected')
    rows = review['per_GPU_reviews']
    require(len(rows) == 3 and {row['physical_GPU'] for row in rows} == {1, 5, 6}, 'Each actual physical GPU requires root drift review')
    for row in rows:
        require(set(row) == set(expected['per_GPU_reviews'][0])
            and row['reviewed_same_source_reference64_controls'] is True
            and row['reviewed_all1085_leaf_gradients_and_actual_updates'] is True
            and row['reviewed_clear_gap_and_near_tie_drift'] is True
            and isinstance(row['reason'], str) and row['reason'].strip(), 'Full actual per-GPU numeric/rank drift review required')
    document = {key: basis[key] for key in ('chosen_workspace_MiB', 'feeding_commit', 'old_commit',
        'adapter_source_sha256', 'immutable_feeding_files_sha256', 'source_pause_proofs', 'evidence_refs',
        'engineering_basis_sha256', 'observed_numeric_envelope_status')}
    document.update(format='v23_closed_world_execution_engineering_certificate_v1',
        status='ENGINEERING_CONTRACT_VERIFIED_FOR_EXACT_CONTINUATION', engineering_contract_verified=True,
        engineering_continuation_authorized=True, native_strict_equivalence_passed=False,
        scientific_equivalence_established=False, original_strict_FAIL_unchanged=True,
        actual_drift_review=review, actual_drift_review_ref=actual_drift_review_ref,
        production_launch_performed=False, total_training_epochs=40,
        certificate_scope='execution only; exact future deployment, actual3GPU integration smoke, partial-state/BEST-preserving handoff also required')
    document['certificate_sha256'] = canonical_sha256(document)
    return document


def validate_engineering_certificate(document, *, proof_reader=None):
    """Return exact validated document; malformed or pending evidence raises."""
    require(set(document) == CERTIFICATE_KEYS, 'Closed engineering certificate schema required')
    actual = create_engineering_certificate(document['evidence_refs'], document['chosen_workspace_MiB'],
        document['actual_drift_review_ref'], proof_reader=proof_reader)
    require(document == actual, 'Certificate fields/canonical digest differ from actual SHA-bound engineering review')
    return document


def validate_engineering_integration_smoke(document, certificate, deployment_request, *,
        certificate_raw_sha256, gpu=None, source_checkpoint_sha256=None,
        source_numerical_state_sha256=None):
    """Bind actual three-GPU full-native smoke to an independently proved release.

    Caller first validates the certificate and rehashes both raw attached files.
    ``deployment_request`` comes from the source-bound deployment request file;
    the smoke's own manifest is never used as its independent trust reference.
    This is one actual clone update per GPU, not a new three-repeat calibration.
    """
    hashes = deployment_request['hashes']
    require(set(hashes) == ENGINEERING_RELEASE_FILES and len(hashes) == 22
        and all(valid_sha(value) for value in hashes.values())
        and valid_sha(deployment_request['commit'], 40)
        and deployment_request['commit'] not in (OLD_COMMIT, FEEDING_COMMIT)
        and deployment_request['local_HEAD_and_GitHub_same'] is True,
        'Exact independently source-bound pushed22-file integration release required')
    require(document['format'] == 'v23_exact_engineering_integration_native_smoke_v1'
        and document['destination_commit'] == deployment_request['commit']
        and document['source_files_sha256'] == hashes
        and document['feeding_commit'] == certificate['feeding_commit'] == FEEDING_COMMIT
        and valid_sha(certificate_raw_sha256)
        and document['engineering_certificate_raw_sha256'] == certificate_raw_sha256
        and document['reused_native_performance_raw_sha256'] == certificate['evidence_refs']['native_performance']['raw_sha256']
        and document['adapter_source_sha256'] == certificate['adapter_source_sha256'] == ADAPTER_SHA256
        and document['chosen_workspace_MiB'] == certificate['chosen_workspace_MiB'],
        'Exact new-source smoke/certificate/performance/workspace binding required')
    require(document['all5_feeding_files_byte_identical_to_measured_source'] is True
        and document['source_checkpoints_unchanged'] is True and document['production_optimizer_updates'] == 0
        and document['all_three_smoke_workers_naturally_ended'] is True
        and document['native_strict_equivalence_passed'] is False
        and document['scientific_equivalence_established'] is False
        and document['new_three_repeat_calibration_performed'] is False,
        'Integration smoke cannot relabel strictFAIL or reused native measurements')
    require(certificate['engineering_contract_verified'] is True
        and certificate['engineering_continuation_authorized'] is True
        and certificate['native_strict_equivalence_passed'] is False
        and certificate['scientific_equivalence_established'] is False
        and certificate['original_strict_FAIL_unchanged'] is True,
        'An actual validated engineering review preserving strictFAIL is required')
    require(certificate['immutable_feeding_files_sha256'] == {path: hashes[path] for path in FEEDING_FILES}
        and hashes['hiercp_v1x/v23_edge_execution.py'] == ADAPTER_SHA256,
        'Exact measured five feeding files and ead4 adapter bytes required')
    rows = document['GPU_reports']
    require(len(rows) == 3 and {row['physical_GPU'] for row in rows} == {1, 5, 6},
        'Actual original-model integration updates on all three GPUs required')
    for row in rows:
        proof = certificate['source_pause_proofs'][str(row['physical_GPU'])]
        require(row['physical_patient_batch'] == 4 and row['candidate_chunk'] == 32
            and row['all_P_and_U128_two_views'] is True
            and row['original_model_parameters'] == 10434532
            and row['finite_present_gradient_tensors'] == 1085
            and row['finite_present_gradient_elements'] == 10434532
            and row['clone_optimizer_updates'] == 1
            and row['source_numerical_state_sha256'] == proof['numerical_state_sha256']
            and row['source_checkpoint_file_sha256'] == proof['latest_file_sha256']
            and row['RNG_exact'] is True and row['source_input_workspace_contract_verified'] is True
            and row['actual_runner_pre_hook_calls'] == 1 and row['runner_installer_used'] is True
            and row['pre_hook_is_original_runner_callback'] is True
            and row['own_overlay_restored_and_hook_removed'] is True
            and 0 <= row['peak_allocated_cuda_bytes'] <= 40 * 2**30,
            'Actual full-native source/gradient/update/RNG integration proof required')
    if gpu is not None:
        require(type(gpu) is int and gpu in (1, 5, 6), 'Exact independent physical GPU required')
        proof = certificate['source_pause_proofs'][str(gpu)]
        if source_checkpoint_sha256 is not None:
            require(proof['latest_file_sha256'] == source_checkpoint_sha256,
                'Actual upgrade source differs from integration checkpoint')
        if source_numerical_state_sha256 is not None:
            require(proof['numerical_state_sha256'] == source_numerical_state_sha256,
                'Actual upgrade numerical/RNG bytes differ from integrated source')
    return document


from pathlib import Path
import hashlib
import json
import subprocess

EDGE_ADAPTER_SHA256 = 'ead4ec316d19cef4cf6388476cb792f275f0e73f174f15b58bea70e5ae09b6fb'
EDGE_MEASURED_FEEDING_COMMIT = 'd3f558a77d96c3ffb98e645615126bf156035031'
EDGE_FEEDING_SHA256 = {
    'hiercp_v1x/v23_training.py': 'e8d488a95f0950382c464aab755bab7be5f2f1b0c33b86029df779a6b347af34',
    'hiercp_v1x/v23_data.py': 'cbaf9523183f2d76bdcf58a22d06dcb04c7e32f098c20b865278948ac114890a',
    'hiercp_v1x/v23_inputs.py': 'b5f42a5e49d4fce935489d82ce433d41633d868df68ae57611b0c7c60f557d65',
    'hiercp_v1x/v23_geometry.py': 'f537054cf1c68d5f5e44fa3545cca5df5c60b5bb36d78d845a75e6783f540198',
    'hiercp_v1x/v23_targets.py': '5e0c08637ce3c69aeee1b32cc42b6af0e86434cc4431738513b3ec5e2206a2a7',
}


def _edge_require(condition, message):
    if not condition:
        raise ValueError(message)


def _edge_read_json(path, checksum):
    path = Path(path)
    _edge_require(path.is_file() and not path.is_symlink(), 'Regular engineering evidence required')
    raw = path.read_bytes()
    _edge_require(hashlib.sha256(raw).hexdigest() == checksum,
        'Engineering evidence raw SHA differs: ' + str(path))
    return json.loads(raw)


def edge_runtime_contract(runtime, adapter_path, checkout_root):
    """Explicit execution-only settings and all five unchanged feeding sources."""
    mode = runtime.get('l0_edge_workspace_mib')
    _edge_require(type(mode) is int and mode in (64, 256, 1024),
        'Explicit measured integer L0 workspace64/256/1024 MiB required')
    path = Path(adapter_path)
    _edge_require(path.is_file() and not path.is_symlink(), 'Regular measured adapter required')
    _edge_require(hashlib.sha256(path.read_bytes()).hexdigest()
        == runtime.get('l0_edge_adapter_sha256') == EDGE_ADAPTER_SHA256,
        'The exact native-measured adapter bytes are required')
    root = Path(checkout_root).resolve(strict=True)
    for relative, expected in EDGE_FEEDING_SHA256.items():
        source = root / relative
        _edge_require(source.is_file() and not source.is_symlink()
            and hashlib.sha256(source.read_bytes()).hexdigest() == expected,
            'Scientific feeding source changed: ' + relative)
    return mode


def require_engineering_calibration(calibration, runtime, gpu, checkout_root, *,
        source_checkpoint_sha256=None, source_numerical_state_sha256=None,
        execution_request=None, certificate_validator=None,
        smoke_validator=None):
    """Admit only an actual SHA-bound root decision and its full native evidence.

    ``certificate_validator`` is an injection seam for DEBUG metadata tests.
    Production uses the closed-world validator in this existing module, not a
    certificate-supplied callable, key, boolean or tolerance. Invalid raises.
    """
    mode = edge_runtime_contract(runtime,
        Path(checkout_root) / 'hiercp_v1x/v23_edge_execution.py', checkout_root)
    _edge_require(type(gpu) is int and gpu in (1, 5, 6), 'Exact independent physical GPU required')
    attached = calibration['engineering_contract']
    _edge_require(set(attached) == {'receipt_path', 'receipt_sha256'},
        'Closed engineering receipt attachment required')
    document = _edge_read_json(attached['receipt_path'], attached['receipt_sha256'])
    if certificate_validator is None:
        certificate_validator = validate_engineering_certificate
    validated = certificate_validator(document)
    _edge_require(validated == document, 'Validated exact engineering document required')
    _edge_require(document['format'] == 'v23_closed_world_execution_engineering_certificate_v1'
        and document['status'] == 'ENGINEERING_CONTRACT_VERIFIED_FOR_EXACT_CONTINUATION'
        and document['engineering_contract_verified'] is True
        and document['engineering_continuation_authorized'] is True
        and document['native_strict_equivalence_passed'] is False
        and document['scientific_equivalence_established'] is False
        and document['original_strict_FAIL_unchanged'] is True,
        'Engineering authorization must preserve original failed strict/scientific status')
    _edge_require(document['chosen_workspace_MiB'] == mode
        and document['feeding_commit'] == EDGE_MEASURED_FEEDING_COMMIT
        and document['adapter_source_sha256'] == EDGE_ADAPTER_SHA256
        and document['immutable_feeding_files_sha256'] == EDGE_FEEDING_SHA256,
        'Actual measured workspace/adapter/unchanged feeding bytes differ')
    deployment_ref = calibration['engineering_deployment_request']
    _edge_require(set(deployment_ref) == {'receipt_path', 'receipt_sha256'},
        'Closed actual deployment request attachment required')
    deployment = _edge_read_json(deployment_ref['receipt_path'], deployment_ref['receipt_sha256'])
    checkout = Path(checkout_root).resolve(strict=True)
    commit = deployment['commit']
    _edge_require(isinstance(commit, str) and len(commit) == 40
        and all(character in '0123456789abcdef' for character in commit)
        and len(deployment['hashes']) == 22, 'Exact full commit/22-file release required')
    _edge_require(subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
        text=True).strip() == commit, 'Actual checkout HEAD differs from engineering deployment')
    for relative, checksum in deployment['hashes'].items():
        relative_path = Path(relative)
        _edge_require(not relative_path.is_absolute() and '..' not in relative_path.parts
            and '\\' not in relative and ':' not in relative, 'Safe canonical deployment path required')
        path = checkout / relative_path
        _edge_require(path.is_file() and not path.is_symlink()
            and hashlib.sha256(path.read_bytes()).hexdigest() == checksum
            and hashlib.sha256(subprocess.check_output(['git', '-C', str(checkout),
                'show', commit + ':' + relative])).hexdigest() == checksum,
            'Exact committed deployment file differs: ' + relative)
    if execution_request is not None:
        _edge_require(calibration['request_sha256'] == execution_request['request_sha256']
            and execution_request['gpus'] == [gpu]
            and execution_request['config']['v23_runtime']['l0_edge_workspace_mib'] == mode
            and execution_request['config']['v23_runtime']['l0_edge_adapter_sha256'] == EDGE_ADAPTER_SHA256
            and all(deployment['hashes'].get(name) == checksum
                for name, checksum in execution_request['source'].items()),
            'Bound native execution request differs from actual deployment')
    smoke_ref = calibration['engineering_integration_smoke']
    _edge_require(set(smoke_ref) == {'receipt_path', 'receipt_sha256'},
        'Closed actual integrated smoke attachment required')
    smoke = _edge_read_json(smoke_ref['receipt_path'], smoke_ref['receipt_sha256'])
    if smoke_validator is None:
        smoke_validator = validate_engineering_integration_smoke
    _edge_require(smoke_validator(smoke, document, deployment,
        certificate_raw_sha256=attached['receipt_sha256'], gpu=gpu,
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_numerical_state_sha256=source_numerical_state_sha256) == smoke,
        'Actual same-release integrated native smoke required')
    sources = document['source_pause_proofs']
    _edge_require(set(sources) == {'1', '5', '6'}, 'All three distinct paused sources required')
    source = sources[str(gpu)]
    if source_checkpoint_sha256 is not None:
        _edge_require(source['latest_file_sha256'] == source_checkpoint_sha256,
            'Engineering review used a different actual paused checkpoint')
    if source_numerical_state_sha256 is not None:
        _edge_require(source['numerical_state_sha256'] == source_numerical_state_sha256,
            'Engineering review used different actual six-state numerical/RNG bytes')
    reference = document['evidence_refs']['native_performance']
    performance = _edge_read_json(reference['path'], reference['raw_sha256'])
    _edge_require(performance['format'] == 'v23_L0_workspace_performance_only_native_noise_experiment_v1'
        and performance['status'] == 'EXPERIMENTAL_PERFORMANCE_MEASUREMENT_COMPLETE_SCIENCE_REVIEW_REQUIRED'
        and performance['new_commit'] == EDGE_MEASURED_FEEDING_COMMIT
        and performance['adapter_source_sha256'] == EDGE_ADAPTER_SHA256
        and performance['physical_patient_batch'] == 4 and performance['candidate_chunk'] == 32
        and performance['original_model_parameters'] == 10434532
        and performance['full_allP_U128_two_views'] is True
        and performance['all_steady_training_repeats_same_actual_source_epoch'] is True
        and performance['all_repeats_restore_identical_model_optimizer_scaler_RNG'] is True
        and performance['production_optimizer_updates'] == 0
        and performance['source_checkpoint_updates'] == 0
        and performance['source_checkpoints_unchanged'] is True
        and performance['source_files_written'] is False
        and performance['native_strict_equivalence_passed'] is False
        and performance['production_admission'] is False
        and performance['original_strict_FAIL_unchanged'] is True,
        'Actual complete unchanged full native performance experiment required')
    reports = [row for row in performance['GPU_reports'] if row['workspace_MiB'] == mode]
    _edge_require(len(reports) == 3 and {row['physical_GPU'] for row in reports} == {1, 5, 6},
        'Requested workspace requires three actual native GPU measurements')
    for row in reports:
        admission = row['workspace_admission']
        _edge_require(admission['workspace_bytes'] == mode * 2**20
            and admission['original_global_workspace_bytes'] == 64 * 2**20
            and admission['original_edge_hidden_workspace_factor'] == 8
            and admission['local_blocks'] == 3 and admission['relation_convolutions'] == 48
            and admission['heads'] == 4 and admission['out_channels'] == 32
            and admission['global_workspace_changed'] is False
            and admission['upper_execution_changed'] is False
            and admission['full_relation_softmax_and_dropout_unchanged'] is True,
            'Only original48 L0 instance execution chunks may change')
        _edge_require(row['all_1085_gradients_finite_present'] is True
            and row['all_RNG_exact'] is True and row['steady_train_clone_updates'] == 3
            and row['validation_forward_repeats'] == 3 and row['warmup_clone_updates'] == 1,
            'Complete native finite1085/RNG and three-repetition evidence required')
        paused = sources[str(row['physical_GPU'])]
        _edge_require(row['source_checkpoint_file_sha256'] == paused['latest_file_sha256']
            and row['source_model_optimizer_RNG_hashes'] == paused['numerical_state_sha256'],
            'Actual native measurements used different paused state')
        measurements = row['train_measurements'] + row['validation_measurements']
        _edge_require(len(measurements) == 6, 'Three actual train and validation measurements required')
        for item in measurements:
            _edge_require(0 <= item['peak_allocated_cuda_bytes'] <= 40 * 2**30
                and item['physical_patients'] == 4 and item['candidate_chunk'] == 32
                and item['all_U_per_patient'] == 128 and item['two_views'] is True
                and item['source_state_reset_verified'] is True,
                'Actual full4/U128/view2 work or measured40GiB budget differs')
    return document
