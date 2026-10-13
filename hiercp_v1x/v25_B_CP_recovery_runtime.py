"""Explicit fresh-process routing for B's preserved CP/native computation.

The original scientific files and prior scores remain unchanged. Every new
entry admits the recovery seal, then uses the original authorized B functions.
Only subprocess entry paths are routed through this same admission.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
_INSTALLED = False
_CHILD_INSTALLED = False
_ORIGINAL_CHILD = None
EXECUTION_ENTRY = 'tools/run_v25_B_CP_recovery_execution.py'
NATIVE_ENTRY = 'tools/run_v25_B_CP_recovery_native.py'
METRICS_ENTRY = 'tools/run_v25_B_CP_recovery_metrics.py'
ORIGINAL_METRICS_ENTRY = 'tools/evaluate_v24_historical_metrics.py'
TRAIN_PROGRAM = ('from hiercp_v1x.v25_authorized_early_downstream '
                 'import run_training_entry; run_training_entry()')
RECOVERY_TRAIN_PROGRAM = ('from hiercp_v1x.v25_B_CP_recovery_runtime '
                          'import run_training_entry; run_training_entry()')


def wait_registration(seal_path, *, timeout=120):
    from . import v25_B_early_downstream_resources as old
    from . import v25_B_CP_recovery_resources as recovery
    path = Path(seal_path)
    proof = Path(os.environ[recovery.ENV_PATH])
    deadline = time.monotonic() + timeout
    while not (path.is_file() and proof.is_file()):
        if time.monotonic() >= deadline:
            raise RuntimeError('Fresh B controller/recovery registration not published within120seconds')
        time.sleep(.2)
    os.environ[old.ENV_PATH] = str(path)
    os.environ[old.ENV_SHA] = old.sha(path)
    os.environ[recovery.ENV_SHA] = old.sha(proof)


def route_native_command(command):
    """Keep native CLI arguments intact; change only its admitted entry."""
    command = list(command)
    if len(command) > 4 and command[:3] == [sys.executable, '-B', '-u']:
        if command[3] == '-c':
            if command[4] != TRAIN_PROGRAM:
                raise ValueError('Unknown native Python program; recovery cannot silently reroute it')
            command[4] = RECOVERY_TRAIN_PROGRAM
        elif command[3] == str(ROOT/'tools/run_v25_authorized_early_downstream.py'):
            command[3] = str(ROOT/NATIVE_ENTRY)
        elif command[3] == str(ROOT/ORIGINAL_METRICS_ENTRY):
            if command.count('--checkpoint') != 1 or command[command.index('--checkpoint')+1] != 'BEST':
                raise ValueError('Historical metrics must evaluate the original own BEST predictions')
            command[3] = str(ROOT/METRICS_ENTRY)
        else:
            raise ValueError('Unknown native child entry; exact preserved CLI required')
    else:
        raise ValueError('Exact original native Python executable/flags required')
    return command


def _child(command, env, log_path):
    from . import v25_B_CP_recovery_resources as resources
    if (env.get(resources.ENV_PATH) != os.environ.get(resources.ENV_PATH)
            or env.get(resources.ENV_SHA) != os.environ.get(resources.ENV_SHA)):
        raise ValueError('Native child must inherit the same immutable recovery registration')
    # The native runtime prepends its editable private nnU-Net package. Retain
    # that package order and make the sealed admission package explicit too.
    forwarded = dict(env)
    paths = forwarded.get('PYTHONPATH', '').split(os.pathsep)
    if str(ROOT) not in paths:
        forwarded['PYTHONPATH'] = os.pathsep.join([*filter(None, paths), str(ROOT)])
    return _ORIGINAL_CHILD(route_native_command(command), forwarded, log_path)


def install():
    global _INSTALLED, _CHILD_INSTALLED, _ORIGINAL_CHILD
    from . import v25_B_CP_recovery_resources as resources
    contract = resources.install_from_environment()
    from . import v25_authorized_early_downstream as adapter
    adapter.install_from_environment()
    modules = adapter.private_modules()
    from . import v25_cp_memory_runtime as memory
    from . import v25_B_early_downstream_resources as previous
    sealed = previous.load_seal()['source_scientific23']
    recovery, _ = resources.load_recovery_proof()
    declaration = adapter.read(recovery['durable']['request']['path'])
    memory_sources = {'v25_factory.py': sealed[str(ROOT/'hiercp_v1x/v25_factory.py')],
                      'v25_cp.py': declaration['implementation_sha256']['hiercp_v1x/v25_cp.py']}
    memory_contract = memory.install_runtime(cp_module=modules['CP'],
        expected_frozen_source_sha256=memory_sources)
    if not _CHILD_INSTALLED:
        native = modules['NATIVE']
        _ORIGINAL_CHILD = native._child
        native._child = _child
        # A caller may already have created the private native modules. Rebind
        # only the original subprocess hook, never the training functions.
        for module in native._MODULES.values():
            if module.__dict__.get('_child') is _ORIGINAL_CHILD:
                module._child = _child
        _CHILD_INSTALLED = True
    _INSTALLED = True
    return dict(resources=contract, memory=memory_contract,
                frozen_scientific_file_bytes_changed=False,
                native_CLI_arguments_and_logger_unchanged=True)


def run_training_entry():
    install()
    from . import v25_authorized_early_downstream as adapter
    return adapter.run_training_entry()


def verify_stage_completion(stage, output, cp_output):
    install()
    from . import v25_authorized_early_downstream as adapter
    return adapter.verify_stage_completion(stage, output, cp_output)
