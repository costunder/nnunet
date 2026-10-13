"""Resume only B CP529->642, then original native250 and own BEST26."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def replace_execution_entry(command):
    command = list(command)
    expected = str(ROOT/'tools/run_v25_B_early_downstream_execution.py')
    if command[:4] != [sys.executable, '-B', '-u', expected]:
        raise ValueError('Exact original B stage executable and entry required')
    command[3] = str(ROOT/'tools/run_v25_B_CP_recovery_execution.py')
    return command


def verify_stage_completion(stage, output, cp_output):
    import json
    program = ('import json,sys;from hiercp_v1x.v25_B_CP_recovery_runtime '
        'import verify_stage_completion;'
        'verify_stage_completion(sys.argv[1],sys.argv[2],sys.argv[3]);'
        'print(json.dumps(dict(actual_saved_completion_verified=True,stage=sys.argv[1])))')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.check_output([sys.executable, '-B', '-c', program,
        stage, str(output), str(cp_output)], cwd=ROOT, env=env, text=True, timeout=600)
    value = json.loads(result)
    if value != dict(actual_saved_completion_verified=True, stage=stage):
        raise RuntimeError('Actual saved B-only full downstream completion was not verified')
    return value


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments.count('--execution-manifest') != 1:
        raise ValueError('Exactly one fresh B controller seal is required')
    seal_path = arguments[arguments.index('--execution-manifest') + 1]
    from hiercp_v1x.v25_B_CP_recovery_runtime import wait_registration, install
    wait_registration(seal_path)
    install()
    from tools import run_v25_B_early_downstream_controller as original
    old_stage, old_native = original.stage_command, original.native_command
    original.stage_command = lambda args, seal: replace_execution_entry(old_stage(args, seal))
    original.native_command = lambda args, seal: replace_execution_entry(old_native(args, seal))
    original.verify_stage_completion = verify_stage_completion
    return original.main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
