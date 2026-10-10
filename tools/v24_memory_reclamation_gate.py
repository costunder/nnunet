"""Actual GPU6 memory-only source, checkpoint and numerical admission.

Preserved reports and tensor artifacts are owned immutable DEBUG evidence.
The old allocator implementation is imported from its real source file; no
input-only receipts or source hashes are projected onto memory-only evidence.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT=Path(__file__).resolve().parents[1]
PROVENANCE_FORMAT='v24_preserved_memory_only_runtime_provenance_v1'
PROBE_FORMAT='v24_actual_full128_memory_reclamation_probe_DEBUG_v1'
SCIENCE_NAMES=('v24_factory.py','v24_geometry.py','v24_inputs.py','v24_model.py','v24_provider.py')
UNCHANGED_NAMES=(*SCIENCE_NAMES,'v24_hash_runtime.py','v24_prefetch_runtime.py',
                 'v24_input_runtime.py','v24_preparation_runtime.py')


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):value.update(block)
    return value.hexdigest()


def _witness(path, *, directory=False):
    path=Path(path).absolute();value=path.lstat()
    if (path.resolve(strict=True)!=path or not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
            or (hasattr(os,'getuid') and value.st_uid!=os.getuid())):
        raise ValueError('Owned regular nonsymlink memory evidence required: '+str(path))
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def _guard(path, expected=None):
    path=Path(path).absolute();before=_witness(path);digest=_sha(path)
    if before!=_witness(path) or (expected is not None and digest!=expected):
        raise ValueError('Memory evidence source/SHA/stat differs: '+str(path))
    return dict(path=str(path),raw_sha256=digest,stat=list(before))


def _recheck(proof):
    if list(_witness(proof['path']))!=proof['stat'] or _sha(proof['path'])!=proof['raw_sha256']:
        raise ValueError('Memory evidence changed during actual comparison: '+proof['path'])


def _pressure_function_sha(path):
    source=Path(path).read_text(encoding='utf8');lines=source.splitlines(keepends=True)
    classes=[node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='MemorySafeCoordinator']
    functions=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name=='_trim_locked'] if len(classes)==1 else []
    if len(functions)!=1:raise ValueError('Actual memory pressure function source required')
    node=functions[0]
    return hashlib.sha256(''.join(lines[node.lineno-1:node.end_lineno]).encode()).hexdigest()


def memory_runtime_provenance(source_code):
    source=Path(source_code).absolute();_witness(source,directory=True)
    files=[];unchanged={}
    for name in UNCHANGED_NAMES:
        old=_guard(source/'hiercp_v1x'/name);current=_guard(ROOT/'hiercp_v1x'/name)
        if old['raw_sha256']!=current['raw_sha256']:
            raise ValueError('Only memory reclamation may change; scientific/input/hash/prefetch bytes differ: '+name)
        files.extend((old,current));unchanged[name]=old['raw_sha256']
    old=_guard(source/'hiercp_v1x/v24_memory_runtime.py');current=_guard(ROOT/'hiercp_v1x/v24_memory_runtime.py')
    if old['raw_sha256']==current['raw_sha256']:
        raise ValueError('Explicit changed memory runtime required for a memory-only comparison')
    files.extend((old,current))
    for proof in files:_recheck(proof)
    return dict(format=PROVENANCE_FORMAT,source_code=str(source),
        control_memory_source_preserved=True,only_memory_reclamation_source_changed=True,
        preserved_memory_path=old['path'],preserved_memory_sha256=old['raw_sha256'],
        candidate_memory_path=current['path'],candidate_memory_sha256=current['raw_sha256'],
        preserved_pressure_function_sha256=_pressure_function_sha(old['path']),
        candidate_pressure_function_sha256=_pressure_function_sha(current['path']),
        input_adapter_path=str(source/'hiercp_v1x/v24_input_runtime.py'),
        input_adapter_sha256=unchanged['v24_input_runtime.py'],
        unchanged_source_sha256=unchanged,source_file_proofs=files,
        comparison_scope='actual preserved allocator release versus actual coalesced allocator release; identical scientific/hash/prefetch/input bytes')


def route_memory_runtime(source_code, variant):
    """Cold-process routing before either dependent adapter captures its classes."""
    if variant not in ('current-control','optimized'):
        raise ValueError('Explicit preserved memory control or optimized candidate required')
    if any(name in sys.modules for name in ('hiercp_v1x.v24_prefetch_runtime','hiercp_v1x.v24_input_runtime')):
        raise RuntimeError('Memory routing must precede all prefetch/input adapter imports')
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('Fresh pre-CUDA memory diagnostic required')
    proof=memory_runtime_provenance(source_code)
    prefix='preserved' if variant=='current-control' else 'candidate'
    path=Path(proof[prefix+'_memory_path']);expected=proof[prefix+'_memory_sha256']
    import hiercp_v1x
    name='hiercp_v1x.v24_memory_runtime'
    previous=sys.modules.get(name);previous_attr=getattr(hiercp_v1x,'v24_memory_runtime',None)
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ValueError('Actual memory source cannot be imported')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;setattr(hiercp_v1x,'v24_memory_runtime',module)
    try:
        spec.loader.exec_module(module)
        if _sha(module.__file__)!=expected or module.memory_runtime_contract()['runtime_source_sha256']!=expected:
            raise ValueError('Actual selected memory implementation differs from its admitted source')
        hash_binding=_bind_selected_memory_hash_reference(module,proof,expected)
    except BaseException:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous
        if previous_attr is None:delattr(hiercp_v1x,'v24_memory_runtime')
        else:setattr(hiercp_v1x,'v24_memory_runtime',previous_attr)
        raise
    return dict(variant=variant,actual_module_name=module.__name__,actual_module_path=str(path),
        actual_module_sha256=expected,actual_contract=module.memory_runtime_contract(),
        imported_before_prefetch_input_factory=True,source_provenance=proof,
        actual_hash_memory_reference_binding=hash_binding)


def _bind_selected_memory_hash_reference(module,provenance,expected):
    """Bind the unchanged hash guard to the real admitted cold memory source.

    The hash implementation, private-get AST checks, input source and every
    reference allowlist remain unchanged. Retain the candidate file witness
    as well as the selected preserved file witness. No hash is memoized and
    no file identity is projected onto a different implementation.
    """
    from hiercp_v1x import v24_hash_runtime as hashes
    if hashes.torch.cuda.is_initialized():raise RuntimeError('Hash source binding must precede CUDA')
    initial=(ROOT/'hiercp_v1x/v24_hash_runtime.py',ROOT/'hiercp_v1x/v24_inputs.py',
             ROOT/'hiercp_v1x/v24_memory_runtime.py')
    if tuple(hashes._SOURCE_PATHS)!=initial or hashes.inputs.tensor_digest is not hashes._ORIGINAL_DIGEST:
        raise ValueError('Fresh unchanged hash reference paths and original input digest required')
    if Path(hashes.__file__).resolve()!=initial[0] or sys.modules.get(module.__name__) is not module:
        raise ValueError('Actual candidate hash and canonical admitted memory modules required')
    for item in provenance['source_file_proofs']:_recheck(item)
    selected=Path(module.__file__).resolve(strict=True)
    admitted=[item for item in provenance['source_file_proofs']
              if item['path']==str(selected) and item['raw_sha256']==expected]
    if len(admitted)!=1 or module.memory_runtime_contract()['runtime_source_sha256']!=expected:
        raise ValueError('Selected memory hash reference must match its actual owned SHA/stat proof')
    hashes._prove_original()
    for item in initial:
        witness=hashes._SOURCE_PROOFS.get(str(item))
        if witness is None or witness!=(hashes._stat(item),_sha(item)):
            raise ValueError('Original hash guard source proof changed before routing')
    original_paths=hashes._SOURCE_PATHS;original_proofs=dict(hashes._SOURCE_PROOFS)
    try:
        hashes._SOURCE_PROOFS[str(selected)]=(tuple(admitted[0]['stat']),expected)
        hashes._SOURCE_PATHS=(initial[0],initial[1],selected)
        # Execute the original private guard now, before installing any digest.
        actual=hashes._private_memory_reference()
        if actual is None or actual[1]['unchanged_memory_source_sha256']!=expected:
            raise ValueError('Actual original private-get hash guard failed selected source admission')
        for item in provenance['source_file_proofs']:_recheck(item)
    except BaseException:
        hashes._SOURCE_PATHS=original_paths;hashes._SOURCE_PROOFS=original_proofs
        raise
    return dict(actual_memory_path=str(selected),actual_memory_sha256=expected,
        actual_hash_source_sha256=_sha(initial[0]),original_hash_function_code_unchanged=True,
        original_private_get_AST_provider_input_checks_preserved=True,
        original_candidate_memory_witness_retained=True,bound_before_hash_install=True)


def _hash_pair_contract(baseline,optimized,provenance):
    """Only two actual selected-memory identities may differ in hash receipts."""
    import copy
    normalized=[]
    for row,prefix in ((baseline,'preserved'),(optimized,'candidate')):
        receipt=copy.deepcopy(row.get('hash_runtime',{}));expected=provenance[prefix+'_memory_sha256']
        contract=receipt.get('contract',{})
        private=receipt.get('installed_private_global_references')
        if (contract.get('unchanged_memory_source_sha256')!=expected
                or contract.get('runtime_source_sha256')!=provenance['unchanged_source_sha256']['v24_hash_runtime.py']
                or not isinstance(private,list) or len(private)!=1
                or private[0].get('name')!='hiercp_v1x.v24_memory_runtime._GET.__globals__[tensor_digest]'
                or private[0].get('unchanged_memory_source_sha256')!=expected
                or private[0].get('private_get_code_reconstructed_from_original') is not True
                or receipt.get('all_loaded_references_preflighted_before_install') is not True):
            raise ValueError('Actual unchanged hash code and selected-memory private reference proof required')
        contract['unchanged_memory_source_sha256']='ACTUAL_ADMITTED_MEMORY_ONLY'
        private[0]['unchanged_memory_source_sha256']='ACTUAL_ADMITTED_MEMORY_ONLY'
        normalized.append(receipt)
    if normalized[0]!=normalized[1]:
        raise ValueError('Hash receipts differ beyond actual bound selected-memory source identity')


def _memory_pair_contract(baseline,optimized,provenance):
    if provenance!=memory_runtime_provenance(provenance['source_code']):
        raise ValueError('Actual unchanged science and both memory sources no longer match')
    for row,prefix in ((baseline,'preserved'),(optimized,'candidate')):
        expected=provenance[prefix+'_memory_sha256'];path=provenance[prefix+'_memory_path']
        receipt=row.get('memory_runtime',{});contract=receipt.get('contract',{});profile=receipt.get('profile',{})
        selection=row.get('actual_memory_module_selection',{})
        if (row.get('format')!=PROBE_FORMAT or row.get('memory_only_execution_change') is not True
                or row.get('memory_runtime_source_sha256')!=expected or row.get('memory_runtime_module_path')!=path
                or contract.get('runtime_source_sha256')!=expected
                or contract.get('unchanged_scientific_files_sha256')!={name:provenance['unchanged_source_sha256'][name] for name in SCIENCE_NAMES}
                or receipt.get('RSS_limit_bytes')!=64*2**30 or not 0<receipt.get('actual_RSS_bytes',0)<=64*2**30
                or selection.get('actual_module_sha256')!=expected or selection.get('actual_module_path')!=path
                or selection.get('actual_contract')!=contract or selection.get('source_provenance')!=provenance
                or selection.get('imported_before_prefetch_input_factory') is not True
                or any(contract.get(name) is not False for name in ('limits_increased','graphs_or_data_reduced','model_or_loss_changed',
                    'candidate_chunk_and_patient_batch_changed','prefetch_chunks_changed'))
                or profile.get('strict_failures')!=0
                or any(type(profile.get(name)) is not int or profile[name]<0 for name in ('pressure_events','allocator_releases','trim_calls'))):
            raise ValueError('Actual bound old/candidate memory source, strict64GiB limit and measured profile required')
        prefetch=row.get('prefetch_runtime',{})
        if (prefetch.get('memory_runtime_source_sha256')!=expected
                or prefetch.get('original_locked_pressure_function_sha256')!=provenance[prefix+'_pressure_function_sha256']):
            raise ValueError('Actual prefetch must capture the selected real memory classes/functions')
        if set(row.get('trainable_parameter_schema',{}))==set() or len(row['trainable_parameter_schema'])!=537:
            raise ValueError('Actual complete named537 gradient schema required')
    _hash_pair_contract(baseline,optimized,provenance)
    if (baseline.get('input_runtime',{}).get('contract')!=optimized.get('input_runtime',{}).get('contract')
            or baseline['trainable_parameter_schema']!=optimized['trainable_parameter_schema']):
        raise ValueError('Hash/input contracts and all named native parameters must remain exact')
    first,second=baseline['prefetch_runtime'],optimized['prefetch_runtime']
    if set(first)!=set(second) or any(first[name]!=second[name] for name in first
            if name not in ('memory_runtime_source_sha256','original_locked_pressure_function_sha256')):
        raise ValueError('Only actual memory-source-dependent prefetch identity fields may differ')


def compare_numerical_tensor_files(baseline,optimized,*,expected_gradients):
    from tools.run_v24_performance_continuation import compare_numerical_tensor_files as original
    result=original(baseline,optimized,expected_gradients=expected_gradients)
    import torch
    for row in (baseline,optimized):
        actual=torch.load(row['numerical_tensor_file']['path'],map_location='cpu',weights_only=True)
        schema=row['trainable_parameter_schema']
        if set(actual['gradients'])!=set(schema):raise ValueError('Actual537 gradient parameter identities differ')
        for name,value in actual['gradients'].items():
            expected=schema[name]
            if list(value.shape)!=expected['shape'] or str(value.dtype)!=expected['dtype'] or value.numel()!=expected['numel']:
                raise ValueError('Actual named gradient schema differs: '+name)
    return result


def compare_probe_files(baseline_path,optimized_path,checkpoint_sha256):
    paths=[Path(baseline_path).absolute(),Path(optimized_path).absolute()]
    if paths[0]==paths[1]:raise ValueError('Distinct fresh memory probes required')
    proofs=[_guard(path) for path in paths]
    rows=[json.loads(path.read_text(encoding='utf8')) for path in paths]
    for path,row in zip(paths,rows):
        item=row.get('numerical_tensor_file',{});artifact=Path(item.get('path','')).absolute()
        if artifact!=path.parent/'numerical_tensors.pt':raise ValueError('Actual artifact must belong to its fresh memory probe directory')
        proof=_guard(artifact,item.get('raw_sha256'));proofs.append(proof)
        if item.get('size')!=proof['stat'][2]:raise ValueError('Actual memory numerical artifact size differs')
    result=compare_probes(*rows,checkpoint_sha256)
    for proof in proofs:_recheck(proof)
    result['owned_report_and_artifact_proofs']=proofs
    return result


# The dedicated compare_probes body below retains the full strict scientific
# and actual537 tensor gate. Its provenance checks explicitly describe memory.


def compare_probes(baseline,optimized,checkpoint_sha256):
    if (baseline.get('variant')!='current-control' or optimized.get('variant')!='optimized'
            or any(row.get('debug') is not True or row.get('production_optimizer_updates')!=0
                   or row.get('active_U')!=128 or row.get('all_P_included') is not True
                   or row.get('physical_patient_batch')!=4 or row.get('diagnostic_batches')!=1
                   for row in (baseline,optimized))):
        raise ValueError('Actual complete full128 diagnostic proof required')
    keys=('original_training_identity_sha256','initial_model_sha256','initial_RNG_sha256',
          'output_sha256','loss','gradient_sha256','after_forward_model_sha256','after_RNG_sha256',
          'case_ids','physical_candidate_chunk','debug_numerics','ordered_CPU_input_sha256')
    if any(baseline.get(key)!=optimized.get(key) for key in keys):
        raise ValueError('Actual native output/loss/gradient/model/RNG parity differs')
    chunk=baseline.get('physical_candidate_chunk')
    if chunk not in (32,64) or baseline.get('case_ids')!=['liver_6','liver_129','liver_123','liver_69']:
        raise ValueError('Original complete stress patients and physical candidate chunk required')
    expected_chunks=(524+chunk-1)//chunk
    for row in (baseline,optimized):
        policy=row.get('debug_numerics',{})
        flags=policy.get('actual_forward_flags',{})
        proof=row.get('input_tensor_proof',{})
        if (policy.get('debug_deterministic_numerics') is not True
                or flags.get('deterministic_algorithms') is not True
                or flags.get('deterministic_warn_only') is not False
                or flags.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'
                or policy.get('production_config_modified') is not False
                or policy.get('unsupported_operator_fallback') is not False
                or proof.get('complete') is not True or proof.get('every_actual_value_hashed') is not True
                or proof.get('both_sampled_views') is not True or proof.get('ordered_records')!=524
                or proof.get('observed_native_chunks')!=expected_chunks or proof.get('expected_native_chunks')!=expected_chunks
                or proof.get('observed_upper_graphs')!=4 or proof.get('expected_upper_graphs')!=4
                or proof.get('physical_candidate_chunk')!=chunk or proof.get('epoch')!=1
                or len(proof.get('content',{}).get('local_chunks',[]))!=expected_chunks
                or len(proof.get('content',{}).get('upper_graphs',[]))!=4
                or row.get('ordered_CPU_input_sha256')!=proof.get('content_sha256')
                or not isinstance(proof.get('content_sha256'),str)
                or len(proof['content_sha256'])!=64):
            raise ValueError('Strict DEBUG kernels and complete actual CPU tensor proof required')
    if any(row['source_checkpoint']['raw_sha256']!=checkpoint_sha256 for row in (baseline,optimized)):
        raise ValueError('Native diagnostic belongs to a different exact checkpoint')
    if any(row['gradient']['missing'] or not row['gradient']['finite'] for row in (baseline,optimized)):
        raise ValueError('Native gradients must be connected and finite')
    incremental=baseline['variant']=='current-control';tensor_comparison=None
    if incremental:
        if chunk!=64:raise ValueError('Original GPU6 complete candidate chunk64 required')
        provenance=baseline.get('comparison_memory_runtime_provenance')
        if (not isinstance(provenance,dict)
                or provenance.get('format')!=PROVENANCE_FORMAT
                or provenance!=optimized.get('comparison_memory_runtime_provenance')
                or provenance.get('control_memory_source_preserved') is not True
                or baseline.get('input_runtime',{}).get('contract',{}).get('runtime_source_sha256')!=provenance.get('input_adapter_sha256')):
            raise ValueError('Explicit preserved current-runtime provenance required for incremental comparison')
        root=Path(__file__).resolve().parents[1]
        if (optimized.get('input_runtime',{}).get('contract',{}).get('runtime_source_sha256')
                !=_sha(root/'hiercp_v1x/v24_input_runtime.py')
                or any(row.get('probe_source_sha256')!=_sha(root/'tools/run_v24_memory_reclamation_probe.py')
                    for row in (baseline,optimized))
                or baseline.get('source_checkpoint')!=optimized.get('source_checkpoint')
                or not baseline.get('model_contract')
                or baseline['model_contract']!=optimized.get('model_contract')):
            raise ValueError('Exact deployed candidate input/probe source and model/checkpoint proof required')
        for key in ('initial_optimizer_sha256','after_optimizer_sha256','initial_scaler_sha256','after_scaler_sha256'):
            if not isinstance(baseline.get(key),str) or len(baseline[key])!=64 or baseline[key]!=optimized.get(key):
                raise ValueError('Actual optimizer/scaler state parity differs')
        for row in (baseline,optimized):
            expected=row.get('source_checkpoint',{}).get('numerical_state_sha256',{})
            if (set(expected)!={'model','optimizer','scheduler','scaler','rank_rng','shuffle_generator'}
                    or row.get('source_checkpoint',{}).get('optimizer_parameter_tensors')!=537
                    or row.get('original_named_trainable_parameter_count')!=537
                    or row.get('optimizer_contains_all_trainable_parameters_exactly_once') is not True
                    or row.get('gradient',{}).get('trainable_parameter_tensors')!=537
                    or row.get('gradient',{}).get('gradient_present')!=537
                    or row['initial_model_sha256']!=expected['model']):
                raise ValueError('Actual complete GPU6 model/optimizer/all537 gradient checkpoint proof required')
            for name in ('optimizer','scaler'):
                if (row['initial_'+name+'_sha256']!=row['after_'+name+'_sha256']
                        or row['initial_'+name+'_sha256']!=expected[name]):
                    raise ValueError('DEBUG persisted '+name+' state changed or differs from checkpoint')
        _memory_pair_contract(baseline,optimized,provenance)
        profile=optimized.get('input_runtime_profile',{})
        if any(profile.get(name)!=0 for name in ('source_live_chunks','source_cache_live_entries',
                'source_cache_live_key_bytes','source_cache_live_result_bytes',
                'source_cache_live_tree_visible_array_bytes')):
            raise ValueError('Pure source chunk cache was not completely released')
        tensor_comparison=compare_numerical_tensor_files(baseline,optimized,expected_gradients=537)
    return dict(status='ACTUAL_FULL128_MEMORY_ONLY_OUTPUT_LOSS_ALL537_GRADIENT_MODEL_RNG_EXACT_PASS',
        checkpoint_sha256=checkpoint_sha256,baseline_seconds=baseline['seconds'],
        optimized_seconds=optimized['seconds'],debug_single_batch_timing=True,
        baseline_variant=baseline['variant'],incremental_previous_deployed_runtime=incremental,
        numerical_tensor_comparison=tensor_comparison,
        ordered_CPU_input_sha256=baseline['ordered_CPU_input_sha256'],
        explicit_DEBUG_deterministic_kernels=True,production_numerical_policy_changed=False,
        memory_only_source_provenance=baseline['comparison_memory_runtime_provenance'],
        actual_memory_pressure_profiles=[row['memory_runtime']['profile'] for row in (baseline,optimized)],
        observed_pressure_in_both=all(row['memory_runtime']['profile'].get('pressure_events',0)>0 for row in (baseline,optimized)),
        memory_reclamation_speedup_claimed=False,
        full_training_completion_claimed=False)


