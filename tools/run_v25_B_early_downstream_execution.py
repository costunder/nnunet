"""Sealed B-only complete CP/native execution with truthful early GNN admission."""
from __future__ import annotations
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))


def take(arguments,flag,*,required=False):
    if arguments.count(flag)>1:raise ValueError('Repeated immutable binding flag: '+flag)
    if flag not in arguments:
        if required:raise ValueError('Exact immutable argument required: '+flag)
        return None
    offset=arguments.index(flag)
    if offset+1>=len(arguments):raise ValueError('Argument value required: '+flag)
    value=arguments[offset+1];del arguments[offset:offset+2]
    return value


def main(argv=None):
    arguments=list(sys.argv[1:]if argv is None else argv)
    native=bool(arguments and arguments[0]=='--native')
    if native:arguments.pop(0)
    from hiercp_v1x import v25_B_early_downstream_resources as resources
    seal_path=Path(take(arguments,'--controller-seal',required=True)).resolve(strict=True)
    scientific_path=Path(take(arguments,'--early-request',required=True)).resolve(strict=True)
    if (str(seal_path)!=os.environ.get(resources.ENV_PATH)
            or resources.sha(seal_path)!=os.environ.get(resources.ENV_SHA)):
        raise ValueError('Exact inherited fresh B-only controller seal required')
    from hiercp_v1x import v25_authorized_early_downstream as adapter
    adapter.install_from_environment();seal=resources.load_seal()
    if (str(scientific_path)!=seal['scientific_request']['path']
            or adapter.sha(scientific_path)!=seal['scientific_request']['sha256']):
        raise ValueError('Exact inherited scientific request required')
    declared=adapter.read(scientific_path)
    if native:
        from tools.run_v25_authorized_early_downstream import main as downstream
        return downstream(arguments)
    from tools import run_v25 as original
    explicit_cp=take(arguments,'--cp-output')
    if explicit_cp is not None and Path(explicit_cp).absolute()!=Path(declared['cp_output']).absolute():
        raise ValueError('Declared immutable fresh CP output differs')
    args=original.parse(arguments)
    if (args.arm!='B' or args.gpu!=5 or args.mode!='score-cp'
            or str(args.output)!=declared['source_gnn_output']
            or str(args.output)!=str(Path(seal['group_root'])/'B')
            or str(args.allocation_request)!=seal['slot']['direct_proof']['path']):
        raise ValueError('Only exact own B/GPU5/source/request/full642x128 scoring admitted')
    runtime_files=tuple(ROOT/name for name in (*adapter.FILES,
        'hiercp_v1x/v25_B_early_downstream_resources.py',
        'hiercp_v1x/v25_provider_memory_runtime.py','hiercp_v1x/host_memory.py',
        'hiercp_v1x/v25_terminal_descendant_resources.py'))
    from hiercp_v1x.v25_upper_memory_runtime import install_upper_memory_runtime
    install_upper_memory_runtime(runtime_files=runtime_files)
    from hiercp_v1x.v25_provider_memory_runtime import install_runtime
    install_runtime()
    return adapter.run_score(arguments,authorization_path=os.environ[adapter.ENV_AUTHORIZATION],
        cp_output=declared['cp_output'])


if __name__=='__main__':raise SystemExit(main())
