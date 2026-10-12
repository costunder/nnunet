"""Sealed C-only bootstrap around unchanged complete scientific stage CLI."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NATIVE_ENTRY = 'from hiercp_v1x.v25_native_cp import run_training_entry; run_training_entry()'
NATIVE_BOOTSTRAP = ('from hiercp_v1x.v25_standalone_c_resources import install_from_environment; '
    'install_from_environment(); ')


def adapt_native_child(command):
    command = list(command)
    if command[:3] != [sys.executable, '-B', '-u']:
        raise ValueError('Exact original nnunet Python native child prefix required')
    if len(command) >= 5 and command[3:5] == ['-c', NATIVE_ENTRY]:
        command[4] = NATIVE_BOOTSTRAP+command[4]
    elif len(command) >= 5 and command[3:5] == [str(ROOT/'tools/run_v25_downstream.py'), 'clone-debug-worker']:
        command[3] = str(ROOT/'tools/run_v25_c_execution.py')
        command.insert(4, '--native')
    else:
        raise ValueError('Unknown native child entry; no unguarded execution fallback')
    return command


def install_native_children(native):
    original = native._child
    def child(command, env, log_path):
        return original(adapt_native_child(command), env, log_path)
    native._child = child
    if 'NATIVE_PIPELINE' in native._MODULES:
        native._MODULES['NATIVE_PIPELINE']._child = child
    return original


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    from hiercp_v1x import v25_standalone_c_resources as resources
    native_mode = bool(arguments and arguments[0] == '--native')
    if native_mode:
        arguments.pop(0)
    if '--continuation-seal' in arguments:
        offset = arguments.index('--continuation-seal')
        path = Path(arguments[offset+1]).resolve(strict=True)
        del arguments[offset:offset+2]
        if str(path) != os.environ.get(resources.ENV_PATH):
            raise ValueError('C child seal differs from inherited controller admission')
    contract = resources.install_from_environment()
    seal = resources.load_seal()
    files = (ROOT/'tools/run_v25_c_execution.py', ROOT/'hiercp_v1x/v25_standalone_c_resources.py',
        ROOT/'hiercp_v1x/v25_standalone_ad_resources.py', ROOT/'hiercp_v1x/v25_provider_memory_runtime.py',
        ROOT/'hiercp_v1x/host_memory.py', ROOT/'hiercp_v1x/v25_u128_validation_reuse.py',
        ROOT/'hiercp_v1x/v25_u128_reuse_resources.py', ROOT/'hiercp_v1x/v25_terminal_descendant_resources.py')
    if native_mode:
        from hiercp_v1x import v25_native_cp as native
        install_native_children(native)
        from tools.run_v25_downstream import main as downstream
        return downstream(arguments)
    from tools import run_v25_execution as execution, run_v25 as frozen
    args = frozen.parse(arguments)
    if args.arm != 'C':
        raise ValueError('This bootstrap resumes failed C only; ABD is preserved')
    slot = seal['slots']['C']
    if (args.gpu != 1 or args.mode not in ('train', 'score-cp')
            or str(args.output) != str(Path(seal['group_root'])/'C')
            or str(args.allocation_request) != slot['direct_proof']['path']):
        raise ValueError('Exact own C train/BEST scoring stage binding required')
    if args.mode == 'train' and str(args.resume) != slot['checkpoint']['path']:
        raise ValueError('Explicit exact own latest checkpoint is required')
    from hiercp_v1x.v25_upper_memory_runtime import install_upper_memory_runtime
    install_upper_memory_runtime(runtime_files=files)
    from hiercp_v1x.v25_provider_memory_runtime import install_runtime
    memory = install_runtime()
    reuse_contract = None
    if args.mode == 'train':
        from hiercp_v1x.v25_u128_validation_reuse import install_runtime as install_validation_reuse
        reuse_contract = install_validation_reuse(expected_source_sha256=
            seal['preserved_files'][str(ROOT/'hiercp_v1x/v25_training.py')])
    folder = args.output/'execution'; folder.mkdir(exist_ok=True)
    frozen.publish(folder/('C_provider_runtime_'+str(time.time_ns())+'.json'),
        dict(format='v25_exact_C_provider_execution_v1', mode=args.mode, resources=contract,
            provider_memory=memory, validation_reuse=reuse_contract,
            runtime_files_sha256={str(p): resources.sha(p) for p in files},
            actual_PID=os.getpid(), scientific_configuration_unchanged=True, debug=False))
    return execution.main(arguments)


if __name__ == '__main__':
    main()
