"""DEBUG admission using real full native tensors and immutable source files.

This compares two isolated GPU1 numerical probes. It performs no training,
optimizer update, production stop or whole-epoch speed measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
FORMAT = 'v24_native_execution_actual_tensor_admission_DEBUG_v1'
PROBE_FORMAT = 'v24_native_batch2_full128_CUDA_execution_admission_DEBUG_v1'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _witness(path):
    path = Path(path).absolute()
    value = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISREG(value.st_mode)
            or (hasattr(os, 'getuid') and value.st_uid != os.getuid())):
        raise ValueError('Owned regular nonsymlink artifact required: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _guard(path, expected=None):
    path = Path(path).absolute()
    before = _witness(path)
    digest = _sha(path)
    if before != _witness(path) or (expected is not None and digest != expected):
        raise ValueError('Artifact SHA/stat changed or differs: ' + str(path))
    return dict(path=str(path), sha256=digest, stat=before)


def _recheck(proof):
    if _witness(proof['path']) != proof['stat'] or _sha(proof['path']) != proof['sha256']:
        raise ValueError('Admitted artifact changed while inspecting actual tensors: ' + proof['path'])


def _equal_leaves(left, right, path='root', counts=None):
    """Compare dtype, shape and actual bytes, including scalar and RNG leaves."""
    import numpy as np
    import torch
    counts = counts if counts is not None else dict(torch_tensors=0, numpy_arrays=0, tensor_bytes=0)
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError('Tensor leaf type differs: ' + path)
        if left.device.type != 'cpu' or right.device.type != 'cpu':
            raise ValueError('Saved actual CPU tensor leaves required: ' + path)
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError('Tensor dtype/shape differs: ' + path)
        if ((left.is_floating_point() or left.is_complex())
                and (not torch.isfinite(left).all() or not torch.isfinite(right).all())):
            raise ValueError('Nonfinite tensor leaf: ' + path)
        first = left.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        second = right.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if first != second:
            raise ValueError('Actual tensor bytes differ: ' + path)
        counts['torch_tensors'] += 1
        counts['tensor_bytes'] += len(first)
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ValueError('NumPy leaf type differs: ' + path)
        if left.dtype.hasobject or right.dtype.hasobject or left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError('NumPy dtype/shape differs or object data present: ' + path)
        if np.issubdtype(left.dtype, np.inexact) and (not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError('Nonfinite NumPy leaf: ' + path)
        if left.tobytes(order='C') != right.tobytes(order='C'):
            raise ValueError('Actual NumPy bytes differ: ' + path)
        counts['numpy_arrays'] += 1
        counts['tensor_bytes'] += left.nbytes
    elif isinstance(left, dict) or isinstance(right, dict):
        if type(left) is not type(right) or left.keys() != right.keys():
            raise ValueError('Dictionary schema differs: ' + path)
        for name in left:
            _equal_leaves(left[name], right[name], path+'/'+str(name), counts)
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('Sequence schema differs: ' + path)
        for index, (first, second) in enumerate(zip(left, right)):
            _equal_leaves(first, second, path+'/'+str(index), counts)
    elif type(left) is not type(right) or left != right:
        raise ValueError('Scalar/metadata leaf differs: ' + path)
    return counts


def _gradient_contract(report, numerical):
    import torch
    names = report['trainable_parameter_names']
    schema = report['trainable_parameter_schema']
    if len(names) != len(set(names)) or set(names) != set(schema):
        raise ValueError('Exactly one schema entry per trainable parameter required')
    groups = [name for group in report['optimizer_parameter_names'] for name in group]
    if len(groups) != len(names) or len(groups) != len(set(groups)) or set(groups) != set(names):
        raise ValueError('Optimizer trainable parameter membership must be exactly once')
    gradients = numerical['gradients']
    missing = set(names)-set(gradients)
    official = report['inactive_head_proof']['official_inactive_parameter_names']
    if (not gradients or len(official) != len(set(official)) or not set(official) <= set(names)
            or set(gradients)-set(names) or missing != set(official)
            or sorted(missing) != report['missing_parameter_names']
            or len(gradients) != report['gradient_tensors']):
        raise ValueError('All actual active gradients and exact official inactive head proof required')
    for name in names:
        parameter = numerical['initial_model'][name]
        expected = schema[name]
        if (not torch.is_tensor(parameter) or list(parameter.shape) != expected['shape']
                or str(parameter.dtype) != expected['dtype'] or parameter.numel() != expected['numel']):
            raise ValueError('Actual parameter tensor differs from declared schema: '+name)
        if name in gradients:
            value = gradients[name]
            if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError('Actual active gradient shape/dtype/finite proof differs: '+name)
    if sum(row['numel'] for row in schema.values()) != report['trainable_parameter_count']:
        raise ValueError('Trainable parameter total differs from actual tensor schema')


def _report_contract(report, mode):
    fixed = dict(format=PROBE_FORMAT, debug=True, mode=mode, physical_batch=2, epochs=250,
        actual_input_shape=[2,1,128,128,128], full105_train=True, ordinary26_validation=True,
        original_production_augmentation_workers=4, original_validation_workers=2,
        trainable_parameter_count=102350575, optimizer_trainable_parameters_exactly_once=True,
        optimizer_state_unchanged=True, AMP_scaler_state_unchanged=True, optimizer_updates=0,
        production_training_performed=False, CUDA_deterministic_DEBUG_only=True,
        production_numerics_changed=False, explicit_DEBUG_seed=42, original_ONLINE_CP_SEED=42,
        exact_future_original_production_RNG_recovery_claimed=False)
    if any(report.get(name) != expected or type(report.get(name)) is not type(expected)
           for name, expected in fixed.items()):
        raise ValueError('Complete original full native DEBUG contract required')
    if (not isinstance(report.get('batch_position'),int) or isinstance(report['batch_position'],bool)
            or report['batch_position'] < 0 or len(report.get('CP_flags',[])) != 2
            or not any(report['CP_flags']) or len(report.get('input_keys',[])) != 2
            or type(report.get('source_epoch')) is not int or not 0<=report['source_epoch']<=250
            or report['initial_optimizer_sha256'] != report['final_optimizer_sha256']
            or report['initial_scaler'] != report['final_scaler']):
        raise ValueError('Actual CP batch, unchanged optimizer/scaler and explicit batch position required')
    binding = report['actual_loaded_checkpoint_binding']
    if (binding.get('all_actual_loaded_states_match_source') is not True
            or binding['network_weights_sha256'] != report['initial_model_sha256']
            or binding['optimizer_state_sha256'] != report['initial_optimizer_sha256']
            or binding['grad_scaler_state'] != report['initial_scaler']
            or binding['current_epoch'] != report['source_epoch']
            or binding['trainer_name'] != 'nnUNetTrainer_250epochs_FrozenV23CP'):
        raise ValueError('Actual loaded checkpoint model/optimizer/scaler/epoch binding required')


def _actual_batch_contract(report, numerical):
    import numpy as np
    import torch
    inputs=numerical['inputs']
    if (not torch.is_tensor(inputs['data']) or list(inputs['data'].shape)!=[2,1,128,128,128]
            or np.asarray(inputs['keys']).tolist()!=report['input_keys']
            or np.asarray(inputs['online_cp_applied']).tolist()!=report['CP_flags']):
        raise ValueError('Actual complete full native input/CP/key identity required')
    targets=inputs['target'] if isinstance(inputs['target'],list) else [inputs['target']]
    outputs=numerical['outputs'] if isinstance(numerical['outputs'],list) else [numerical['outputs']]
    if (not targets or len(outputs)<2 or len(targets)!=len(outputs)
            or any(not torch.is_tensor(value) or value.ndim!=5 or value.shape[0]!=2 for value in (*targets,*outputs))
            or any(output.shape[1]!=3 or target.shape[1]!=1 or output.shape[2:]!=target.shape[2:]
                   for target,output in zip(targets,outputs))
            or list(targets[0].shape)!=[2,1,128,128,128]
            or list(outputs[0].shape)!=[2,3,128,128,128]
            or not torch.is_tensor(numerical['loss']) or numerical['loss'].numel()!=1):
        raise ValueError('Actual full native segmentation outputs/targets and scalar loss required')
    proof=report['inactive_head_proof']
    weights=proof['actual_auxiliary_loss_weights']
    expected=[1/(2**index) for index in range(len(outputs))]
    expected[-1]=0.
    expected=[weight/sum(expected) for weight in expected]
    official={name for name in report['trainable_parameter_names'] if name.startswith('decoder.seg_layers.0.')}
    if (weights!=expected or proof['zero_weight_output_indices']!=[len(outputs)-1]
            or proof['inactive_decoder_layer_indices']!=[0]
            or not official or set(proof['official_inactive_parameter_names'])!=official
            or proof['output_order']!='seg_outputs[::-1]'
            or proof['inactive_parameters_are_not_disconnected_core'] is not True):
        raise ValueError('Original native deep supervision weights and exact zero-weight head mapping required')


def admit(baseline_path, optimized_path, output):
    import torch
    from tools.probe_v24_native_execution_cuda_debug import _hash_cpu_tensors
    from hiercp_v1x import v24_native_execution_runtime as runtime
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Fresh DEBUG admission output required; no overwrite')
    paths = [Path(baseline_path).absolute(),Path(optimized_path).absolute()]
    if paths[0] == paths[1]:
        raise ValueError('Distinct baseline and optimized reports required')
    proofs = [_guard(path) for path in paths]
    reports = [json.loads(path.read_text(encoding='utf8')) for path in paths]
    numericals = []
    for mode, path, report in zip(('baseline','optimized'),paths,reports):
        _report_contract(report,mode)
        if (report['probe_source_sha256'] != _sha(ROOT/'tools/probe_v24_native_execution_cuda_debug.py')
                or report['production_entry_source_sha256'] != _sha(ROOT/'tools/run_v24_native_execution.py')
                or report['execution_runtime'] != runtime.runtime_contract()):
            raise ValueError('Stale execution runtime/probe/production entry source proof')
        artifact = Path(report['numerical_tensors_path']).absolute()
        if artifact.parent != path.parent or artifact.name != 'numerical_tensors.pt':
            raise ValueError('Numerical tensors must belong to the same fresh DEBUG report directory')
        proof = _guard(artifact,report['numerical_tensors_sha256']);proofs.append(proof)
        numerical = torch.load(artifact,map_location='cpu',weights_only=False)
        _recheck(proof)
        if set(numerical) != {'inputs','outputs','loss','gradients','initial_model','final_model','rng_before','rng_after'}:
            raise ValueError('Complete actual native input/output/loss/gradient/model/RNG tensors required')
        _gradient_contract(report,numerical)
        for field, key in (('inputs','input_tensor_and_CP_schedule_sha256'),('initial_model','initial_model_sha256'),
                           ('final_model','final_model_sha256'),('rng_before','RNG_before_sha256'),('rng_after','RNG_after_sha256')):
            if _hash_cpu_tensors(numerical[field]) != report[key]:
                raise ValueError('Actual numerical tensor values differ from report digest: '+field)
        _actual_batch_contract(report,numerical)
        _equal_leaves(numerical['initial_model'],numerical['final_model'],'unchanged_model')
        numericals.append(numerical)
    left,right = reports
    for name in ('original_checkpoint_sha256','source_checkpoint_path','source_checkpoint_stat','source_epoch',
                 'original_source_proof','probe_source_sha256','production_entry_source_sha256','original_native_train_step_sha256',
                 'original_OnlineCP_initialize_sha256','execution_runtime','batch_position','input_keys','CP_flags','native_transport',
                 'actual_loaded_checkpoint_binding','initial_optimizer_sha256','initial_scaler','inactive_head_proof',
                 'trainable_parameter_names','trainable_parameter_schema','optimizer_parameter_names'):
        if left[name] != right[name]:
            raise ValueError('Paired source/checkpoint/native execution proof differs: '+name)
    checkpoint_proof = _guard(left['source_checkpoint_path'],left['original_checkpoint_sha256']);proofs.append(checkpoint_proof)
    if list(checkpoint_proof['stat']) != [left['source_checkpoint_stat'][key] for key in ('device','inode','size','mtime_ns','ctime_ns')]:
        raise ValueError('Actual source checkpoint stat differs from loaded probe source')
    saved = torch.load(checkpoint_proof['path'],map_location='cpu',weights_only=False)
    _recheck(checkpoint_proof)
    _equal_leaves(saved['network_weights'],numericals[0]['initial_model'],'actual_source_checkpoint_weights')
    if (saved['current_epoch'] != left['source_epoch'] or saved['trainer_name'] != left['actual_loaded_checkpoint_binding']['trainer_name']
            or _hash_cpu_tensors(saved['optimizer_state']) != left['initial_optimizer_sha256']
            or saved['grad_scaler_state'] != left['initial_scaler']):
        raise ValueError('Actual source checkpoint optimizer/scaler/epoch proof differs')
    del saved
    source = left['original_source_proof']
    for name,digest in source['source_files_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Only repository-relative scientific source paths admitted')
        proofs.append(_guard(ROOT/relative,digest))
    for prefix in ('native','bank','plans','private_trainer','private_CLI','wrapper_module','crop_runtime_module','gradient_runtime_module','continuation_CLI'):
        proofs.append(_guard(source[prefix+'_path'],source[prefix+'_sha256']))
    decoder=left['inactive_head_proof']['gradient_runtime']
    if (decoder['complete_ascending_head_order'] is not True or decoder['reversal_count']!=1
            or decoder['deep_supervision_active'] is not True
            or decoder['head_count']!=len(numericals[0]['outputs'])
            or decoder['decoder_forward_source_sha256']!=left['inactive_head_proof']['original_decoder_forward_sha256']
            or decoder['source_contract']!=source['gradient_runtime_contract']):
        raise ValueError('Actual complete native decoder forward/gradient runtime source proof required')
    proofs.append(_guard(decoder['decoder_source_file'],decoder['decoder_source_file_sha256']))
    for name in ('tools/probe_v24_native_execution_cuda_debug.py','tools/run_v24_native_execution.py',
                 'hiercp_v1x/v24_native_execution_runtime.py','tools/run_v24_native_execution_admission.py'):
        proofs.append(_guard(ROOT/name))
    counts = _equal_leaves(numericals[0],numericals[1],'actual_native_pair')
    for proof in proofs:
        _recheck(proof)
    result = dict(format=FORMAT,status='PASS',debug=True,actual_numerical_values_compared=True,
        zero_tolerance_byte_parity=True,compared=counts,baseline_report=str(paths[0]),optimized_report=str(paths[1]),
        source_epoch=left['source_epoch'],batch_position=left['batch_position'],CP_flags=left['CP_flags'],
        full_native_physical_batch=2,full_native_patch=[128]*3,full_native_epochs=250,
        full_train_population=105,full_validation_population=26,trainable_parameters=102350575,
        source_and_artifact_proofs=proofs,optimizer_updates=0,production_training_performed=False,
        whole_epoch_speedup_claimed=False,exact_future_original_production_RNG_recovery_claimed=False)
    with output.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--optimized',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=admit(args.baseline,args.optimized,args.output)
    print(json.dumps(dict(output=str(args.output),status=result['status'],compared=result['compared'],optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
