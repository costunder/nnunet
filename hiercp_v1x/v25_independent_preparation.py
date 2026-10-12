"""Execution-only common admission and independent per-arm prefix preparation."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid


FORMAT = 'v25_shared_complete_canonical_full128_preparation_v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_runtime_receipt(frozen, args, inputs, failure):
    """Persist actual admission/reclamation counters for successful or failed work."""
    profile = inputs.preparation_runtime_receipt()
    execution = args.output/'execution';execution.mkdir(exist_ok=True)
    frozen.publish(execution/('shared_preparation_runtime_'+str(time.time_ns())+'_'+uuid.uuid4().hex+'.json'),
        dict(format='v25_actual_shared_preparation_execution_runtime_v1', actual_PID=os.getpid(),
            completed=(args.output.parent/'shared_preparation.json').is_file(),
            failure=None if failure is None else f'{type(failure).__name__}: {failure}',
            profile=profile, scientific_configuration_unchanged=True, debug=False))


def validate_shared_preparation(group, input_cache):
    """Require source-bound complete inputs without depending on any arm prefix."""
    group, input_cache = Path(group).resolve(strict=True), Path(input_cache).resolve(strict=True)
    proof = json.loads((group/'shared_preparation.json').read_text())
    request = json.loads((group/'A/request.json').read_text())
    index_path = input_cache/'local/index.json'
    bindings_path = input_cache/'input_bindings.json'
    index = json.loads(index_path.read_text())
    bindings = json.loads(bindings_path.read_text())
    records = index.get('records', [])
    record_ids = [row.get('id') for row in records if isinstance(row, dict)]
    case_ids = bindings.get('case_ids', [])
    fields = ('query_rows_sha256', 'config_sha256', 'source_sha256',
        'prototype_bank_sha256', 'context_contract_sha256')
    population = proof.get('population', {})
    if (proof.get('format') != FORMAT or proof.get('complete') is not True
            or proof.get('debug') is not False or proof.get('input_cache') != str(input_cache)
            or proof.get('group_root') != str(group)
            or proof.get('content_sha256') != _hash({k:v for k,v in proof.items() if k != 'content_sha256'})
            or request.get('request_sha256') != _hash({k:v for k,v in request.items() if k != 'request_sha256'})
            or proof.get('request_sha256') != request.get('request_sha256')
            or request.get('input_cache') != str(input_cache)
            or proof.get('local_index_sha256') != _sha(index_path)
            or proof.get('input_bindings_file_sha256') != _sha(bindings_path)
            or index.get('format') != 'v24_complete_recipient_GT_free_native_canonical_v1'
            or index.get('complete') is not True or index.get('debug') is not False
            or index.get('recipient_GT_used_in_forward') is not False
            or len(records) != 14102 or len(record_ids) != 14102
            or any(not isinstance(identifier, str) for identifier in record_ids)
            or len(set(record_ids)) != 14102
            or bindings.get('complete') is not True
            or bindings.get('format') != 'v25_immutable_CT_organ_fixed_donor_shared_context_bindings_v1'
            or len(case_ids) != 105 or any(not isinstance(case, str) for case in case_ids)
            or len(set(case_ids)) != 105
            or set(bindings.get('bindings', {})) != set(case_ids)
            or any(not isinstance(index.get(key), str) or len(index[key]) != 64
                or index[key] != bindings.get(key) for key in fields)
            or proof.get('context', {}).get('contract_sha256') != index.get('context_contract_sha256')
            or bindings.get('content_sha256') != _hash({k:v for k,v in bindings.items() if k != 'content_sha256'})
            or index.get('input_bindings_sha256') != bindings.get('content_sha256')
            or proof.get('canonical_records') != 14102
            or proof.get('full_upper', {}).get('format') != 'v24_recipient_GT_blind_upper_graphs_v1'
            or proof.get('full_upper', {}).get('active_u') != 128
            or proof.get('full_upper', {}).get('cases') != 86
            or proof.get('full_upper', {}).get('recipient_GT_used_in_forward') is not False
            or proof.get('production_optimizer_updates') != 0
            or proof.get('neural_model_created') is not False
            or population.get('format') != 'v23_all_observed_P_cumulative_native_U_v1'
            or population.get('DEBUG') is not False or population.get('hidden_subset') is not False
            or population.get('retained_observations') != 14102 or population.get('data_fraction') != 1.
            or population.get('sha256') != _hash({k:v for k,v in population.items() if k!='sha256'})):
        raise ValueError('Complete source-bound shared14102/full128 admission required before independent prefixes')
    return proof


def validate_arm_preparation(group, input_cache, arm):
    """Each arm must admit its own real prefix before calibration or training."""
    common = validate_shared_preparation(group, input_cache)
    group = Path(group).resolve(strict=True)
    request = json.loads((group/arm/'request.json').read_text())
    common_request = json.loads((group/'A/request.json').read_text())
    prepared = json.loads((group/arm/'preparation.json').read_text())
    initial = {'A':8, 'B':16, 'C':32, 'D':128}[arm]
    expected = copy.deepcopy(common_request)
    expected['config']['arm'] = arm
    expected['config']['runtime']['curriculum'] = copy.deepcopy(expected['config']['arms'][arm])
    expected['request_sha256'] = _hash({k:v for k,v in expected.items() if k!='request_sha256'})
    from hiercp_v1x.v25_targets import validate_policy
    policy_receipt = validate_policy(expected['config']['runtime']['curriculum'])
    if (request.get('request_sha256') != _hash({k:v for k,v in request.items() if k != 'request_sha256'})
            or request != expected
            or prepared.get('request_sha256') != request.get('request_sha256')
            or prepared.get('local_index_sha256') != common['local_index_sha256']
            or prepared.get('full_upper', {}).get('active_u') != 128
            or prepared.get('full_upper', {}).get('format') != 'v24_recipient_GT_blind_upper_graphs_v1'
            or prepared.get('full_upper', {}).get('cases') != 86
            or prepared.get('full_upper', {}).get('recipient_GT_used_in_forward') is not False
            or prepared.get('initial_upper', {}).get('active_u') != initial
            or prepared.get('initial_upper', {}).get('format') != 'v24_recipient_GT_blind_upper_graphs_v1'
            or prepared.get('initial_upper', {}).get('cases') != 86
            or prepared.get('initial_upper', {}).get('recipient_GT_used_in_forward') is not False
            or prepared.get('context') != common.get('context')
            or prepared.get('curriculum') != policy_receipt
            or prepared.get('population') != common.get('population')
            or prepared.get('production_optimizer_updates') != 0
            or prepared.get('neural_model_created') is not False):
        raise ValueError('Exact own complete U prefix admission required before arm training: '+arm)
    return prepared


def prepare_shared(frozen, args, lease):
    """Run original complete canonical/full128 preparation, without any U prefix."""
    if args.mode != 'prepare' or args.arm != 'A':
        raise ValueError('Common complete preparation must use arm A in prepare mode')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import psutil
    import torch
    from hiercp_v1x import v25_factory
    from hiercp_v1x.v25_preparation import install_runtime
    from hiercp_v1x.v25_targets import validate_policy
    from hiercp_v1x.u_bridge_experiment import lock
    sys.path.insert(0, str(frozen.ROOT))
    config = frozen.configuration(args.config, args.arm)
    runtime = config['runtime']
    validate_policy(runtime['curriculum'])
    if args.cpu_affinity is not None:
        if len(args.cpu_affinity) < runtime['workers'] or not set(args.cpu_affinity) <= set(psutil.Process().cpu_affinity()):
            raise ValueError('Actual allowed production CPU affinity is required')
        psutil.Process().cpu_affinity(args.cpu_affinity)
    torch.set_num_threads(runtime['torch_threads'])
    if len(psutil.Process().cpu_affinity()) < runtime['workers']:
        raise RuntimeError('Allocated CPUs below four-worker full production contract')
    if psutil.virtual_memory().available < runtime['rss_gib']*2**30:
        raise MemoryError('Available RAM below full64GiB RSS contract')
    ancestor = args.input_cache.absolute()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < runtime['minimum_free_disk_gib']*2**30:
        raise OSError('Cache filesystem lacks reserve; prior data preserved')
    frozen.validate_output_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.input_cache.mkdir(parents=True, exist_ok=True)
    if args.output.name != 'A':
        raise ValueError('Shared preparation must retain the exact group/A namespace')
    group = args.output.resolve(strict=True).parent
    lease.verify(force=True)
    if lease.request['gpu'] != args.gpu:
        raise ValueError('Common preparation physical GPU differs from resource proof')
    install_runtime(v25_factory)
    lease.install_preparation_guard(v25_factory)
    with lock(args.output/'.v25.lock'):
        bound = frozen.request(args, config)
        frozen.publish(args.output/'request.json', bound)
        if (group/'shared_preparation.json').exists():
            return validate_shared_preparation(group, args.input_cache)
        with lock(args.input_cache/'.prepare.lock'):
            inputs = v25_factory.V25NativeInputs(args.native_experiment, args.inventory, args.input_cache, runtime)
            try:
                index = inputs.prepare_local()
                full = inputs.geometry.prepare(128)
                if full.get('active_u') != 128 or full.get('cases') != 86:
                    raise ValueError('All86 complete full128 geometry admissions required')
                proof = dict(format=FORMAT, complete=True, debug=False, group_root=str(group),
                    input_cache=str(args.input_cache.resolve()), request_sha256=bound['request_sha256'],
                    local_index_sha256=_sha(index), input_bindings_file_sha256=_sha(args.input_cache/'input_bindings.json'),
                    canonical_records=14102, full_upper=full, context=inputs.context_contract,
                    population=inputs.population.manifest(),
                    production_optimizer_updates=0, neural_model_created=False,
                    scientific_configuration_unchanged=True, initial_U_prefixes_prepared=False)
                proof['content_sha256'] = _hash(proof)
                frozen.publish(group/'shared_preparation.json', proof)
            finally:
                failure = sys.exc_info()[1]
                try:
                    publish_runtime_receipt(frozen, args, inputs, failure)
                finally:
                    inputs.close()
    return validate_shared_preparation(group, args.input_cache)


def install_prefix_lock(args, frozen=None, *, original_lock=None):
    """Replace only the frozen prepare cache lock after complete shared admission."""
    if args.mode != 'prepare' or args.arm not in ('A', 'B', 'C', 'D'):
        raise ValueError('Independent prefix lock is allowed only for complete production prepare mode')
    cache = args.input_cache.resolve(strict=True)
    output = args.output.resolve(strict=True)
    if output.name != args.arm:
        raise ValueError('Independent prefix must use the exact owned arm namespace')
    group = output.parent
    proof = validate_shared_preparation(group, cache)
    import hiercp_v1x.u_bridge_experiment as experiment
    if original_lock is None:
        original_lock = experiment.lock
    expected = cache/'.prepare.lock'
    replacement = output/'.prepare-prefix.lock'
    @contextmanager
    def lock(path):
        target = Path(path).resolve()
        with original_lock(replacement if target == expected else path):
            if target == expected:
                current = validate_shared_preparation(group, cache)
                if current['content_sha256'] != proof['content_sha256']:
                    raise ValueError('Shared preparation proof changed before prefix execution')
            yield
    experiment.lock = lock
    # frozen.main imports lock from the module at invocation time. Assigning an
    # optional pre-imported binding does not change any other lock operation.
    if frozen is not None and hasattr(frozen, 'lock'):
        frozen.lock = lock
    return dict(format='v25_independent_prefix_lock_execution_v1', arm=args.arm,
        original_prepare_lock=str(expected), independent_prepare_lock=str(replacement),
        shared_preparation_content_sha256=proof['content_sha256'],
        all_other_locks_unchanged=True, scientific_configuration_unchanged=True)
