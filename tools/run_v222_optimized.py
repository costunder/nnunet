"""Optimized execution of the unchanged paired v2.22 research model.

Existing graph caches remain valid: this tool changes neither their source
identity nor their bytes. Execution code hashes are recorded separately in
each rolling checkpoint. New runs default to the measured 256MiB workspace;
resumes inherit the saved numerical workspace instead of changing it silently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def runtime_identity():
    names = ('tools/run_v222_optimized.py', 'tools/v222_runtime_cache.py',
             'tools/v222_runtime_execution.py')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def resume_workspace(checkpoint, saved_policy=None):
    from tools.v222_gpu_workspace import policy_path
    receipt = policy_path(Path(checkpoint).parent)
    external = json.loads(receipt.read_text())['workspace_mib'] if receipt.exists() else None
    embedded = saved_policy.get('workspace_mib') if saved_policy else None
    if embedded is not None and external is not None and embedded != external:
        raise ValueError('Checkpoint and sibling workspace receipt disagree')
    return embedded if embedded is not None else (external if external is not None else 64)


def resume_policy(saved, checkpoint, workspace_override=None, release_override=None):
    policy = saved.get('execution_policy')
    if policy is not None and policy.get('runtime_sha256') != runtime_identity():
        raise ValueError('Optimized runtime changed since checkpoint; review a migration instead of silent resume')
    workspace = resume_workspace(checkpoint, policy)
    if workspace_override is not None and workspace_override != workspace:
        raise ValueError('Resume must retain numerical workspace policy')
    release = bool(saved['state']['release_unused'])
    if release_override is not None and release_override != release:
        raise ValueError('Resume must retain allocator policy')
    return workspace, release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--workspace-mib', type=int, choices=(64, 256, 512))
    # Kept explicit for resuming a prior allocation policy.
    parser.add_argument('--release-unused', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('A new output directory is required; existing run is preserved')
    import torch
    from hiercp_v222.v1_cache import configuration, provenance, PairDataset
    from hiercp_v222 import v1_execution as execution
    from hiercp_v222.contracts import sha
    from tools.v222_gpu_workspace import policy_path
    from tools.v222_runtime_execution import installed
    import hiercp.model as local_model
    cfg, base = configuration()
    # Validate before publishing a new execution policy.
    PairDataset(args.cache, 'inner_train')
    old_policy = None
    if args.resume:
        saved = torch.load(args.resume, weights_only=False, map_location='cpu')
        if (saved.get('format') != execution.RESUME_FORMAT or saved.get('source_identity') != provenance()
                or saved.get('config') != cfg or saved.get('base') != base
                or saved.get('cache_sha256') != sha(args.cache) or saved.get('debug') is not False):
            raise ValueError('Resume requires an exact production model/cache/config identity')
        old_policy = saved.get('execution_policy')
        workspace, release = resume_policy(saved,args.resume,args.workspace_mib,args.release_unused)
        del saved
    else:
        workspace = 256 if args.workspace_mib is None else args.workspace_mib
        release = False if args.release_unused is None else args.release_unused
    local_model.EDGE_ATTENTION_WORKSPACE_BYTES = workspace * 1024**2
    policy = dict(format='v222_optimized_execution_v1', runtime_sha256=runtime_identity(),
        workspace_mib=workspace, release_unused=release, cache_rebuild_required=False,
        original_runtime_resume=args.resume is not None and old_policy is None,
        checkpoint_policy='one bounded asynchronous writer; immutable CPU snapshot every batch; durable status separate',
        graph_view_cache='RAM bounded; exact record identity and epoch; no learned embedding reuse across weight updates')
    receipt = policy_path(args.output)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(policy, stream, indent=2)
    print(json.dumps(dict(stage='optimized_execution', **policy)), flush=True)
    with installed(policy):
        result = execution.train(args.cache, args.output, resume=args.resume, release_unused=release)
    print(str(result), flush=True)


if __name__ == '__main__':
    main()
