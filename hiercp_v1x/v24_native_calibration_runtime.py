"""Fresh native CLI flag and crop transport for the existing GPU1 native arm.

The pinned CP pipeline, private nnU-Net package, plans, arrays and model code
are unchanged. A private clone plans reader adds the fresh CLI's
``continue_training=False`` to a deep copy of the exact plans document. The
separately admitted crop runtime connects the lossless baseline segmentation
protocol to the original paste method in calibration and full training.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
from types import FunctionType, SimpleNamespace

FORMAT = 'v24_native_calibration_fresh_CLI_runtime_v1'
WORKER_FORMAT = 'v24_native_calibration_fresh_CLI_worker_v1'
RECEIPT = 'calibration_runtime.json'
WORKER_RECEIPT = 'fresh_CLI_runtime.json'
ORIGINAL_WORKER_COMMAND = ('from hiercp_v1x.v24_nnunet_cp import _calibrate_native_worker; '
    'import sys; _calibrate_native_worker(sys.argv[1],int(sys.argv[2]),sys.argv[3])')
WORKER_COMMAND = ('from hiercp_v1x.v24_native_calibration_runtime import _calibrate_native_worker; '
    'import sys; _calibrate_native_worker(sys.argv[1],int(sys.argv[2]),sys.argv[3])')
TRAIN_COMMAND = ('from hiercp_v1x.v24_native_crop_runtime import run_training_entry; '
    'run_training_entry()')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _publish(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _function(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    return result


def _function_sha(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()


def _prove_fresh_cli(trainer_source, cli_source):
    """Admit the actual private constructor/CLI flag semantics, not a default guess."""
    trainer = ast.parse(trainer_source)
    classes = [node for node in trainer.body if isinstance(node, ast.ClassDef)
        and node.name == 'nnUNetTrainer']
    constructors = [] if len(classes) != 1 else [node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    if len(constructors) != 1:
        raise ValueError('Exactly one actual private nnUNetTrainer constructor required')
    constructor = constructors[0]
    assignments = [node for node in ast.walk(constructor) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'continue_training'
            for target in node.targets)]
    expected = ast.parse('continue_training = plans.pop("continue_training")').body[0]
    if len(assignments) != 1 or ast.dump(assignments[0], include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Actual private constructor must pop the exact CLI continue_training flag')
    calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'MetaLogger'
        and len(node.args) == 2 and isinstance(node.args[1], ast.Name)
        and node.args[1].id == 'continue_training']
    uses = [node for node in ast.walk(constructor) if isinstance(node, ast.Name)
        and node.id == 'continue_training' and isinstance(node.ctx, ast.Load)]
    if len(calls) != 1 or len(uses) != 1 or uses[0] is not calls[0].args[1]:
        raise ValueError('Private constructor continue_training must affect only actual MetaLogger')
    cli = ast.parse(cli_source)
    functions = [node for node in cli.body if isinstance(node, ast.FunctionDef)
        and node.name == 'get_trainer_from_args']
    if len(functions) != 1:
        raise ValueError('Actual private get_trainer_from_args required')
    function = functions[0]
    args = function.args.posonlyargs + function.args.args
    defaults = dict(zip([arg.arg for arg in args[-len(function.args.defaults):]], function.args.defaults))
    default = defaults.get('continue_training')
    if not isinstance(default, ast.Constant) or default.value is not False:
        raise ValueError('Actual fresh native CLI continue_training default must be False')
    expected = ast.parse('plans["continue_training"] = continue_training').body[0]
    injections = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
        and ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    if len(injections) != 1:
        raise ValueError('Actual native CLI must inject exactly one in-memory plans flag')
    return dict(fresh_CLI_default=False, constructor_flag_use='MetaLogger_only',
        private_CLI_in_memory_assignment='plans[continue_training] = continue_training',
        fresh_CLI_equivalent=True)


def _metadata(pipeline, native_path):
    from . import v24_native_crop_runtime as crop_runtime
    from . import v24_native_gradient_runtime as gradient_runtime
    native_path = Path(native_path).resolve(strict=True)
    native = pipeline.read(native_path)
    bank = pipeline.read(native['bank'])
    if (native['format'] != pipeline.FORMAT or native['model_weights_fresh'] is not True
            or _sha(native['bank']) != native['bank_sha256']
            or native['source_files_sha256'].keys() != set(pipeline.FILES)
            or any(_sha(pipeline.ROOT / name) != checksum
                for name, checksum in native['source_files_sha256'].items())):
        raise ValueError('Pinned completed native/bank/source ownership differs')
    root = Path(native['root']).resolve(strict=True)
    plans = root / 'nnUNet_preprocessed' / bank['baseline']['dataset_name'] / (pipeline.PLANS + '.json')
    flags = _read(plans)
    if any(key in flags for key in ('continue_training', 'only_run_validation')):
        raise ValueError('Immutable native plans must not contain fresh/resume CLI runtime flags')
    private = Path(native['private_runtime']).resolve(strict=True) / 'nnunetv2'
    trainer = private / 'training/nnUNetTrainer/nnUNetTrainer.py'
    cli = private / 'run/run_training.py'
    proof = _prove_fresh_cli(trainer.read_text(encoding='utf8'), cli.read_text(encoding='utf8'))
    proof.update(native_path=str(native_path), native_sha256=_sha(native_path),
        bank_path=native['bank'], bank_sha256=native['bank_sha256'],
        source_files_sha256=copy.deepcopy(native['source_files_sha256']),
        plans_path=str(plans), plans_sha256=_sha(plans),
        private_trainer_path=str(trainer), private_trainer_sha256=_sha(trainer),
        private_CLI_path=str(cli), private_CLI_sha256=_sha(cli),
        wrapper_module_path=str(Path(__file__).resolve()), wrapper_module_sha256=_sha(__file__),
        crop_runtime_module_path=str(Path(crop_runtime.__file__).resolve()),
        crop_runtime_module_sha256=_sha(crop_runtime.__file__),
        crop_runtime_contract=crop_runtime.crop_protocol_contract(),
        gradient_runtime_module_path=str(Path(gradient_runtime.__file__).resolve()),
        gradient_runtime_module_sha256=_sha(gradient_runtime.__file__),
        gradient_runtime_contract=gradient_runtime.gradient_runtime_contract(),
        continuation_CLI_path=str(pipeline.ROOT / 'tools/run_v24_native_continuation.py'),
        continuation_CLI_sha256=_sha(pipeline.ROOT / 'tools/run_v24_native_continuation.py'),
        original_calibration_function_sha256=_function_sha(pipeline.calibrate_native),
        original_worker_function_sha256=_function_sha(pipeline._calibrate_native_worker),
        original_training_function_sha256=_function_sha(pipeline.train))
    return native, bank, proof


def _plans_reader(original_read, plans_path):
    plans_path = Path(plans_path).resolve(strict=True)
    count = [0]
    def read(path):
        value = original_read(path)
        if Path(path).resolve(strict=True) != plans_path:
            return value
        if not isinstance(value, dict) or any(key in value for key in ('continue_training', 'only_run_validation')):
            raise ValueError('Unexpected existing runtime flag in pinned native plans')
        result = copy.deepcopy(value)
        result['continue_training'] = False
        count[0] += 1
        return result
    return read, count


def _replace_worker_command(command, native_path, executable):
    if (not isinstance(command, list) or len(command) != 7
            or any(not isinstance(item, str) for item in command) or command[0] != executable
            or command[1:4] != ['-B', '-c', ORIGINAL_WORKER_COMMAND]
            or Path(command[4]).resolve(strict=True) != Path(native_path).resolve(strict=True)
            or not command[5].isdigit() or int(command[5]) <= 0):
        raise ValueError('Only the exact original native calibration worker command may be redirected')
    result = command.copy()
    result[3] = WORKER_COMMAND
    return result


def _replace_training_command(command, executable, *, trainer, plans):
    expected = [executable, '-B', '-m', 'nnunetv2.run.run_training',
        '730', '3d_fullres', '0', '-tr', trainer, '-p', plans]
    if not isinstance(command, list) or command != expected:
        raise ValueError('Only the exact fresh original native training CLI command may be redirected')
    return [*command[:2], '-c', TRAIN_COMMAND, *command[4:]]


def _calibrate_native_worker(native_path, physical_batch, output):
    """Called only by the isolated clone command; model/step code stays original."""
    from . import v24_nnunet_cp as pipeline
    from . import v24_native_crop_runtime as crop_runtime
    from . import v24_native_gradient_runtime as gradient_runtime
    native, bank, proof = _metadata(pipeline, native_path)
    output = Path(output).resolve(strict=True)
    if (not output.is_relative_to(Path(native['root']).resolve())
            or output.name != 'batch_' + str(physical_batch)
            or physical_batch not in (bank['baseline']['physical_batch'], bank['baseline']['physical_batch'] * 2)):
        raise ValueError('Exact fresh native clone trial ownership/batch required')
    runtime_receipt = output / WORKER_RECEIPT
    if runtime_receipt.exists():
        raise FileExistsError('Existing native runtime clone proof is immutable')
    if crop_runtime.install_crop_protocol() != proof['crop_runtime_contract']:
        raise ValueError('Installed crop protocol differs from the admitted exact parent source')
    reader, count = _plans_reader(pipeline.read, proof['plans_path'])
    original = pipeline._calibrate_native_worker
    step = gradient_runtime.clone_step(pipeline._native_clone_step_with_amp_retry)
    if step.__code__ is not pipeline._native_clone_step_with_amp_retry.__code__:
        raise ValueError('Original complete native AMP retry step code must be preserved')
    worker = _function(original, {'read': reader, '_native_clone_step_with_amp_retry': step})
    if worker.__code__ is not original.__code__:
        raise ValueError('Original complete native worker code must be preserved')
    result = worker(native_path, physical_batch, output)
    if count != [1] or _sha(proof['plans_path']) != proof['plans_sha256']:
        raise ValueError('Exactly one in-memory fresh plans read and unchanged disk plans required')
    trial = output / 'trial.json'
    value = dict(format=WORKER_FORMAT, complete=True, physical_batch=physical_batch,
        proof=proof, trial_path=str(trial), trial_sha256=_sha(trial),
        continue_training_value=False, in_memory_plans_only=True,
        original_worker_code_preserved=True,
        globals_replaced=['read', '_native_clone_step_with_amp_retry'],
        original_step_code_preserved=True, gradient_proof_adapter_connected=True,
        plans_file_unchanged=True, crop_protocol_installed=True, production_optimizer_updates=0)
    _publish(runtime_receipt, value)
    return result


def calibrate_native(native_path, *, gpu=1):
    """Reuse the pinned pipeline, redirecting only its two fresh clone workers."""
    if gpu != 1:
        raise ValueError('This existing native downstream arm is assigned GPU1')
    from . import v24_nnunet_cp as pipeline
    native, bank, proof = _metadata(pipeline, native_path)
    root = Path(native['root']).resolve(strict=True)
    receipt = root / RECEIPT
    if receipt.exists():
        raise FileExistsError('Completed native calibration runtime proof is immutable')
    original = pipeline.calibrate_native
    launches = []
    def popen(command, *args, **kwargs):
        actual = _replace_worker_command(command, native_path, pipeline.sys.executable)
        launches.append(dict(original_command=command.copy(), actual_command=actual.copy()))
        return pipeline.subprocess.Popen(actual, *args, **kwargs)
    proxy = SimpleNamespace(Popen=popen, PIPE=pipeline.subprocess.PIPE, STDOUT=pipeline.subprocess.STDOUT)
    calibration = _function(original, {'subprocess': proxy})(native_path, gpu=gpu)
    if len(launches) != 2 or _sha(proof['plans_path']) != proof['plans_sha256']:
        raise ValueError('Both original baseline/doubled clone trials and unchanged plans required')
    calibration = Path(calibration).resolve(strict=True)
    pipeline.admit_native_calibration(native)
    workers = []
    for launch in launches:
        path = Path(launch['actual_command'][6]) / WORKER_RECEIPT
        value = _read(path)
        if value.get('proof') != proof or value.get('complete') is not True:
            raise ValueError('Actual isolated worker source/fresh CLI proof differs')
        workers.append(dict(path=str(path), sha256=_sha(path), physical_batch=value['physical_batch']))
    value = dict(format=FORMAT, status='COMPLETE', debug=True, proof=proof,
        calibration_path=str(calibration), calibration_sha256=_sha(calibration),
        worker_runtime_receipts=workers, launches=launches, GPU=1,
        continue_training_value=False, fresh_CLI_equivalent=True, in_memory_plans_only=True,
        plans_file_unchanged=True, original_calibration_code_preserved=True,
        original_worker_code_preserved=True, model_or_data_equations_changed=False,
        existing_native_source_files_changed=False, production_optimizer_updates=0,
        completed_at=time.time())
    _publish(receipt, value)
    admit_runtime_receipt(native_path)
    return calibration


def admit_runtime_receipt(native_path):
    """Bind actual calibration/trials to the exact fresh-CLI runtime adapter."""
    from . import v24_nnunet_cp as pipeline
    native, bank, proof = _metadata(pipeline, native_path)
    path = Path(native['root']) / RECEIPT
    value = _read(path)
    if (value.get('format') != FORMAT or value.get('status') != 'COMPLETE'
            or value.get('debug') is not True or value.get('proof') != proof or value.get('GPU') != 1
            or value.get('continue_training_value') is not False
            or value.get('fresh_CLI_equivalent') is not True or value.get('in_memory_plans_only') is not True
            or value.get('plans_file_unchanged') is not True
            or value.get('original_calibration_code_preserved') is not True
            or value.get('original_worker_code_preserved') is not True
            or value.get('model_or_data_equations_changed') is not False
            or value.get('existing_native_source_files_changed') is not False
            or value.get('production_optimizer_updates') != 0):
        raise ValueError('Actual complete fresh-CLI native runtime proof required')
    calibration_path = Path(pipeline.admit_native_calibration(native)).resolve(strict=True)
    if str(calibration_path) != value['calibration_path'] or _sha(calibration_path) != value['calibration_sha256']:
        raise ValueError('Actual original calibration evidence changed')
    calibration = pipeline.read(calibration_path)
    reports = {str(Path(row['path']).resolve(strict=True)): row for row in calibration['trials']}
    workers = value['worker_runtime_receipts']
    baseline = bank['baseline']['physical_batch']
    if len(workers) != 2 or {row['physical_batch'] for row in workers} != {baseline, baseline * 2}:
        raise ValueError('Both actual original clone physical batches need runtime proofs')
    for row in workers:
        worker_path = Path(row['path']).resolve(strict=True)
        if not worker_path.is_relative_to(Path(native['root']).resolve()) or _sha(worker_path) != row['sha256']:
            raise ValueError('Native clone runtime receipt ownership/hash differs')
        worker = _read(worker_path)
        trial = str(Path(worker['trial_path']).resolve(strict=True))
        if (worker.get('format') != WORKER_FORMAT or worker.get('complete') is not True
                or worker.get('proof') != proof or worker.get('physical_batch') != row['physical_batch']
                or worker.get('continue_training_value') is not False
                or worker.get('in_memory_plans_only') is not True or worker.get('plans_file_unchanged') is not True
                or worker.get('crop_protocol_installed') is not True
                or worker.get('original_worker_code_preserved') is not True
                or worker.get('globals_replaced') != ['read', '_native_clone_step_with_amp_retry']
                or worker.get('original_step_code_preserved') is not True
                or worker.get('gradient_proof_adapter_connected') is not True
                or worker.get('production_optimizer_updates') != 0
                or trial not in reports or worker['trial_sha256'] != reports[trial]['sha256']
                or _sha(trial) != worker['trial_sha256']):
            raise ValueError('Actual native trial is not bound to the fresh in-memory flag proof')
    return path


def train_native(native_path, *, gpu=1):
    """Run the unchanged full native trainer through the admitted crop entry point.

    This is the original fresh training path. Existing outputs are still refused
    by the original pipeline; resume is not silently selected by this wrapper.
    """
    if gpu != 1:
        raise ValueError('This existing native downstream arm is assigned GPU1')
    from . import v24_nnunet_cp as pipeline
    runtime_receipt = Path(admit_runtime_receipt(native_path)).resolve(strict=True)
    native, bank, proof = _metadata(pipeline, native_path)
    root = Path(native['root']).resolve(strict=True)
    original = pipeline.train
    launches = []
    def popen(command, *args, **kwargs):
        actual = _replace_training_command(command, pipeline.sys.executable,
            trainer=pipeline.TRAINER, plans=pipeline.PLANS)
        stamp = str(time.time_ns())
        binding_path = root / ('production_runtime_binding_' + stamp + '.json')
        binding = dict(format=FORMAT, status='ADMITTED_BEFORE_CHILD_LAUNCH', debug=False,
            proof=proof, native_calibration_runtime_path=str(runtime_receipt),
            native_calibration_runtime_sha256=_sha(runtime_receipt),
            calibration_path=str(root / 'calibration.json'),
            calibration_sha256=_sha(root / 'calibration.json'), GPU=1,
            original_command=command.copy(), actual_command=actual.copy(),
            original_training_code_preserved=True, globals_replaced=['subprocess'],
            original_CLI_argument_tail_preserved=actual[4:] == command[4:],
            fresh_CLI_default_continue_training=False, plans_file_unchanged=True,
            crop_protocol_child_entrypoint=True, model_or_data_equations_changed=False,
            existing_native_source_files_changed=False, epochs=250,
            physical_batch=bank['baseline']['physical_batch'], cp_probability=.5,
            created_at=time.time())
        _publish(binding_path, binding)
        child = pipeline.subprocess.Popen(actual, *args, **kwargs)
        launches.append(dict(binding_path=str(binding_path), binding_sha256=_sha(binding_path),
            original_command=command.copy(), actual_command=actual.copy(), child_PID=child.pid))
        return child
    proxy = SimpleNamespace(Popen=popen, PIPE=pipeline.subprocess.PIPE, STDOUT=pipeline.subprocess.STDOUT)
    checkpoint = _function(original, {'subprocess': proxy})(native_path, gpu=gpu, resume=False)
    if len(launches) != 1 or _metadata(pipeline, native_path)[2] != proof:
        raise ValueError('One completed original fresh training child and unchanged source/inputs required')
    checkpoint = Path(checkpoint).resolve(strict=True)
    complete = root / ('production_runtime_complete_' + str(time.time_ns()) + '.json')
    _publish(complete, dict(format=FORMAT, status='COMPLETE', debug=False, proof=proof,
        launches=launches, checkpoint_path=str(checkpoint), checkpoint_sha256=_sha(checkpoint),
        native_calibration_runtime_path=str(runtime_receipt),
        native_calibration_runtime_sha256=_sha(runtime_receipt), epochs=250, GPU=1,
        original_training_code_preserved=True, model_or_data_equations_changed=False,
        existing_native_source_files_changed=False, completed_at=time.time()))
    return checkpoint
