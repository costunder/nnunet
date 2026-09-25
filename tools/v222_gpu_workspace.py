"""Explicit GPU execution workspace; graph/model/cache sources stay unchanged.

Use a separate process and record the numerical execution policy. Changing
chunk sizes can change floating-point gradient summation; it is not bit-exact
with the old policy. Resume requires the original workspace policy.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def output_path(entry,arguments):
    if entry=='run_v222_v1_l0.py':
        if not arguments or arguments[0] not in ('train','verify'):
            raise ValueError('Only GPU train/verify commands supported')
        if arguments.count('--output')!=1:raise ValueError('Exactly one --output required')
        return Path(arguments[arguments.index('--output')+1]).resolve()
    if entry=='tools/profile_v1_execution.py':
        if len(arguments)<3:raise ValueError('Profile needs cache/output/policy')
        return Path(arguments[1]).resolve()
    raise ValueError('Unrecognized GPU entry point')


def policy_path(output):
    output=Path(output)
    return output.with_name(output.name+'.workspace.json')


def check_resume(arguments,workspace_mib):
    if '--resume' not in arguments:return
    checkpoint=Path(arguments[arguments.index('--resume')+1]).resolve()
    receipt=policy_path(checkpoint.parent)
    # Published pre-wrapper execution used the preserved 64 MiB constant.
    previous=json.loads(receipt.read_text())['workspace_mib'] if receipt.exists() else 64
    if previous!=workspace_mib:
        raise ValueError(f'Resume workspace mismatch: saved={previous}, requested={workspace_mib}; '
                         'refuse to silently change numerical policy')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-mib',type=int,choices=(64,256,512),required=True)
    p.add_argument('--optimized-runtime',action='store_true',help='Use optimized input loader for DEBUG profiling')
    p.add_argument('entry',choices=('run_v222_v1_l0.py','tools/profile_v1_execution.py'))
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();output=output_path(a.entry,a.arguments)
    if output.exists():raise FileExistsError(f'New output required: {output}')
    check_resume(a.arguments,a.workspace_mib)
    import hiercp.model as implementation
    if implementation.EDGE_ATTENTION_WORKSPACE_BYTES!=64*1024**2:
        raise RuntimeError('Unexpected preserved workspace default')
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=a.workspace_mib*1024**2
    receipt=policy_path(output);receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:
        json.dump(dict(workspace_mib=a.workspace_mib,entry=a.entry,arguments=a.arguments,
            model_and_graph_scale_changed=False,cache_rebuild_required=False,
            numerical_policy='unchanged operator; changed edge chunk accumulation order; not bit-exact across workspace sizes',
            validation_scope='local real full-size DEBUG batches; not A100 MIG throughput or full-training validation'),f,indent=2)
    print(json.dumps(dict(stage='edge_execution_workspace',workspace_mib=a.workspace_mib,receipt=str(receipt))),flush=True)
    sys.argv=[str(ROOT/a.entry),*a.arguments]
    if a.optimized_runtime:
        if a.entry!='tools/profile_v1_execution.py':
            raise ValueError('Optimized production training uses run_v222_optimized.py')
        from tools.v222_runtime_execution import installed
        from tools.run_v222_optimized import runtime_identity
        with installed(dict(debug=True,runtime_sha256=runtime_identity())):
            runpy.run_path(str(ROOT/a.entry),run_name='__main__')
    else:
        runpy.run_path(str(ROOT/a.entry),run_name='__main__')


if __name__=='__main__':main()
