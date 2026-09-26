"""Explicit loader migration; retain weights, optimizer, RNG and all cursors.

This entry point selects the measured process producer instead of the original
thread producer. It accepts only the current, unmodified prior runtime hashes
or its own exact hashes. Unknown runtime changes still fail. The saved tensor
state is never edited or materialized as a replacement checkpoint.
"""
from contextlib import contextmanager, nullcontext
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
PUBLISHED_SNAPSHOT_RUNTIME={**PUBLISHED_PROCESS_RUNTIME,
    'tools/run_v222_process_runtime.py':'235577b0d3b7d1d36f52723f1b3814a8eab339df7eac513bb9ab09d7963a83d9',
    'tools/v222_process_loader.py':'9bec2453ed68c2dfb3f6235440a8603f1cbd2963bb8990f0a1f6d53c8c456387',
    'tools/v222_support_snapshot.py':'579a9661b39dd910e1ba6bb44a5157408ca7afff6c52d050df271be2f23d1c77',
}


# Exact locally committed 6480715 reviewed runtime; its saved objective remains CE.
REVIEWED_RUNTIME={'tools/run_v222_optimized.py': 'a15430575a7675cc6d574ecf90f9bbffbda2d973b9e94fea72e8b2f867a246f5', 'tools/v222_runtime_cache.py': 'aeeca2d5cc738a6f98107861eade686dd7c0e5decee52aaa41a004ea340749dd', 'tools/v222_runtime_execution.py': '2da04cfe566e646c729eeb9c896a92d2bff205239eb70dd10b4114378be0129f', 'tools/run_v222_process_runtime.py': '4b81a5b8142d08966da7921ab245ccb59960ae90c03e5f8439707c41f00923f5', 'tools/v222_process_loader.py': 'f2facb94e36ffa6c36aef48fe59ae8c967adbec21693791eb55f5c619537d36a', 'tools/v222_support_snapshot.py': '579a9661b39dd910e1ba6bb44a5157408ca7afff6c52d050df271be2f23d1c77', 'tools/v222_resume_guard.py': 'ef069f09591475ac435a43ffea8e14098ac9c0df7b25b0de42e5c7def3ef2a3b', 'tools/v222_review_contracts.py': '7d5d37e96474e8d45bac749601b4c6497bc2e4a3d9b80d488e53ba5fe9e63b44'}

def runtime_identity():
    paths=('tools/run_v222_process_runtime.py','tools/v222_process_loader.py','tools/v222_support_snapshot.py',
           'tools/v222_resume_guard.py','tools/v222_review_contracts.py',
           'tools/v22_rank_objective.py','tools/v22_ranking_steps.py','tools/v22_ranking_training.py',
           'tools/v22_rank_recommendation.py','config/v22_observed_ranking.json')
    return {**prior_identity(), **{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}}


def validate_resume(saved, checkpoint, *args, **kwargs):
    policy=saved.get('execution_policy')
    if policy is not None and policy.get('runtime_sha256') in (prior_identity(),PUBLISHED_PROCESS_RUNTIME,PUBLISHED_SNAPSHOT_RUNTIME,REVIEWED_RUNTIME):
        # This copy is only for the inherited validator's exact-runtime gate.
        # The original checkpoint goes unchanged to execution.train.
        validated=dict(saved,execution_policy=dict(policy,runtime_sha256=runtime_identity()))
    else:
        validated=saved
    with patch.object(base,'runtime_identity',runtime_identity):
        return prior_resume_policy(validated,checkpoint,*args,**kwargs)


def main():
    from tools.v222_resume_guard import assert_source_runs_idle
    from tools import v222_review_contracts as reviewed
    from tools import v22_rank_objective as rank_objective
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--feature-coordinates',choices=reviewed.FEATURE_CONTRACTS,
        help='New runs default stride4; resume must retain saved coordinates, legacy when absent')
    options.add_argument('--training-objective',choices=(rank_objective.OBJECTIVE,rank_objective.LEGACY))
    review_options,remaining=options.parse_known_args()
    guard=argparse.ArgumentParser(add_help=False)
    guard.add_argument('--cache',type=Path)
    guard.add_argument('--resume',type=Path)
    guard.add_argument('--output',type=Path)
    known,_=guard.parse_known_args(remaining)
    assert_source_runs_idle(known.cache,known.resume,output=known.output)
    saved=None
    if known.resume:
        import torch
        saved=torch.load(known.resume,map_location='cpu',weights_only=False)
        reviewed.validate_group_resume(saved,json.loads(known.cache.read_text(encoding='utf-8')))
    feature_coordinates=reviewed.resolve_feature_contract(saved,review_options.feature_coordinates)
    objective=rank_objective.resolve_objective(saved,review_options.training_objective)
    if objective==rank_objective.OBJECTIVE and feature_coordinates!='stride4':
        raise ValueError('New ranking contract requires corrected stride4 feature coordinates')
    del saved
    from tools import v222_runtime_execution as execution
    from tools import v222_process_loader as producer
    from tools import v222_support_snapshot as support_snapshot
    from hiercp_v222 import v1_execution
    original_install=execution.installed
    original_write=v1_execution.write_new
    original_save=v1_execution.save_torch_new

    semantics=dict(feature_coordinates=feature_coordinates,support_task_contract=reviewed.TASK_CONTRACT,training_objective=objective,
        feature_sampling_semantics='CNN centers at input 0,4,...44' if feature_coordinates=='stride4' else 'preserved legacy endpoint correspondence')

    def save_model(path,value):
        return original_save(path,dict(value,**semantics,execution_policy_runtime_sha256=runtime_identity()))

    def report(path, value):
        value=dict(value,**semantics)
        if Path(path).name=='execution_contract.json':
            depth,width=producer.producer_layout(value['workers'])
            value=dict(value,worker_kind='CPU producer processes; decode threads split across producers',
                cpu_producer_processes=depth,
                prefetch_batches=depth,
                decode_threads_per_producer=width,
                pinning='one parent-process prefetch thread',
                support_snapshot='one immutable CPU model/Adam copy per support pass; complete checkpoint each batch',
                loader_cache_policy='persistent across phases; combined cache budget 20% of available RAM')
        return original_write(path,value)

    @contextmanager
    def install(policy):
        from tools.v22_ranking_training import train as ranking_train
        policy.update(loader_backend='process_producer_v2',
            cpu_producer_processes=producer.ProcessPairLoader.producer_count,
            prefetch_batches=producer.ProcessPairLoader.producer_count,
            async_pin_memory=True,fixed_support_snapshot=True,
            loader_migration='explicit entry point; original saved tensor state and numerical policies preserved')
        policy.update(semantics)
        print(json.dumps(dict(stage='process_loader_execution',**policy)),flush=True)
        with original_install(policy),producer.installed(),support_snapshot.installed(),reviewed.installed(feature_coordinates), \
             patch.object(v1_execution,'write_new',report),patch.object(v1_execution,'save_torch_new',save_model), \
             (patch.object(v1_execution,'train',ranking_train) if objective==rank_objective.OBJECTIVE else nullcontext()):
            yield

    with patch.object(base,'runtime_identity',runtime_identity), \
         patch.object(base,'resume_policy',validate_resume), \
         patch.object(execution,'installed',install),patch.object(sys,'argv',[sys.argv[0],*remaining]):
        base.main()


if __name__=='__main__':main()
