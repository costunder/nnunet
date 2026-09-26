"""Explicit loader migration; retain weights, optimizer, RNG and all cursors.

This entry point selects the measured process producer instead of the original
thread producer. It accepts only the current, unmodified prior runtime hashes
or its own exact hashes. Unknown runtime changes still fail. The saved tensor
state is never edited or materialized as a replacement checkpoint.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools import run_v222_optimized as base

prior_identity=base.runtime_identity
prior_resume_policy=base.resume_policy

# Exact published 683a9f7 process backend; only execution scheduling/snapshot
# reuse changes. The three original numerical/runtime modules stay unchanged.
PUBLISHED_PROCESS_RUNTIME={
    'tools/run_v222_optimized.py':'a15430575a7675cc6d574ecf90f9bbffbda2d973b9e94fea72e8b2f867a246f5',
    'tools/v222_runtime_cache.py':'aeeca2d5cc738a6f98107861eade686dd7c0e5decee52aaa41a004ea340749dd',
    'tools/v222_runtime_execution.py':'2da04cfe566e646c729eeb9c896a92d2bff205239eb70dd10b4114378be0129f',
    'tools/run_v222_process_runtime.py':'f189da47b1dc78fcda16c82988cd341a0113f9a597a7db6bd06ba086d54f36c7',
    'tools/v222_process_loader.py':'ce193811b8ad140b91b8a0ff7f3ba91f42647798bae4da968d0ed40705d53231',
}


def runtime_identity():
    paths=('tools/run_v222_process_runtime.py','tools/v222_process_loader.py','tools/v222_support_snapshot.py')
    return {**prior_identity(), **{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}}


def validate_resume(saved, checkpoint, *args, **kwargs):
    policy=saved.get('execution_policy')
    if policy is not None and policy.get('runtime_sha256') in (prior_identity(),PUBLISHED_PROCESS_RUNTIME):
        # This copy is only for the inherited validator's exact-runtime gate.
        # The original checkpoint goes unchanged to execution.train.
        validated=dict(saved,execution_policy=dict(policy,runtime_sha256=runtime_identity()))
    else:
        validated=saved
    with patch.object(base,'runtime_identity',runtime_identity):
        return prior_resume_policy(validated,checkpoint,*args,**kwargs)


def main():
    from tools.v222_resume_guard import assert_source_runs_idle
    guard=argparse.ArgumentParser(add_help=False)
    guard.add_argument('--cache',type=Path)
    guard.add_argument('--resume',type=Path)
    known,_=guard.parse_known_args()
    assert_source_runs_idle(known.cache,known.resume)
    from tools import v222_runtime_execution as execution
    from tools import v222_process_loader as producer
    from tools import v222_support_snapshot as support_snapshot
    from hiercp_v222 import v1_execution
    original_install=execution.installed
    original_write=v1_execution.write_new

    def report(path, value):
        if Path(path).name=='execution_contract.json':
            value=dict(value,worker_kind='CPU producer processes; decode threads split across producers',
                cpu_producer_processes=producer.ProcessPairLoader.producer_count,
                prefetch_batches=producer.ProcessPairLoader.producer_count,
                decode_threads_per_producer=value['workers']//producer.ProcessPairLoader.producer_count,
                pinning='one parent-process prefetch thread',
                support_snapshot='one immutable CPU model/Adam copy per support pass; complete checkpoint each batch',
                loader_cache_policy='persistent across phases; combined cache budget 20% of available RAM')
        return original_write(path,value)

    @contextmanager
    def install(policy):
        policy.update(loader_backend='process_producer_v2',
            cpu_producer_processes=producer.ProcessPairLoader.producer_count,
            prefetch_batches=producer.ProcessPairLoader.producer_count,
            async_pin_memory=True,fixed_support_snapshot=True,
            loader_migration='explicit entry point; original saved tensor state and numerical policies preserved')
        print(json.dumps(dict(stage='process_loader_execution',**policy)),flush=True)
        with original_install(policy),producer.installed(),support_snapshot.installed(),patch.object(v1_execution,'write_new',report):
            yield

    with patch.object(base,'runtime_identity',runtime_identity), \
         patch.object(base,'resume_policy',validate_resume), \
         patch.object(execution,'installed',install):
        base.main()


if __name__=='__main__':main()
