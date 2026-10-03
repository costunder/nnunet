"""Actual CT/CUDA smoke for full-training scope workers and checkpoint binding.

Uses already verified native DEBUG inputs. No optimizer step, full preparation,
long training, production checkpoint or readiness flag is created.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
# Spawn inherits the parent's verified snapshot-first search path. Do not move
# the development checkout ahead of it when this main module is reconstructed.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class FixedCanonicalDataset:
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        import torch
        from hiercp.sample import materialize_sample_views
        sample = dict(self.samples[index])
        sample['source_patch'] = sample['source_patch'].to(torch.float16)
        sample['target_patches'] = sample['target_patches'].to(torch.float16)
        return materialize_sample_views(sample, training=False, epoch=29, global_seed=42)


def assert_equal_batches(left, right):
    import torch
    if left.counts != right.counts or left.case_ids != right.case_ids:
        raise AssertionError('Native batch identities differ across scope worker')
    for field in ('source_patches', 'target_patches', 'difficulties'):
        if not torch.equal(getattr(left, field), getattr(right, field)):
            raise AssertionError(f'Dense data changed across scope worker: {field}')
    for field in ('local_batch', 'local_batch_view2', 'patient_batch', 'prototype_batch'):
        first, second = getattr(left, field), getattr(right, field)
        if first.node_types != second.node_types or first.edge_types != second.edge_types:
            raise AssertionError('Graph relations changed across scope worker')
        other = second.to_dict()
        for key, store in first.to_dict().items():
            for name, value in store.items():
                if isinstance(value, torch.Tensor) and not torch.equal(value, other[key][name]):
                    raise AssertionError(f'Graph tensor changed across scope worker: {field}/{key}/{name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'fixture', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--margin-mm', type=float, required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    a = p.parse_args()
    output = Path(a.output).resolve()
    if output.exists(): raise FileExistsError('Existing smoke evidence is preserved')
    if a.workers < 2: raise ValueError('Explicit parallel worker smoke required')
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.scope_probe_support import activate_original, _sha
    from hiercp_v1x.scope_learning_inputs import load_samples, rebuild_scope
    from hiercp_v1x.scope_learning_loop import independent_transfer
    from hiercp_v1x.scope_training_entry import ScopeWorkerInitializer, install_checkpoint_binding, ResourceBudget
    from hiercp_v1x.contracts import verify_archive
    from tools.local_cnn_device import select
    select(a.gpu)
    # Match the full controller: a fresh exact archive snapshot, without older
    # stage helpers that the legacy DEBUG experiment stored beside hiercp.
    verify_archive(ROOT)
    source = output.with_suffix('.source')
    source.mkdir(parents=True, exist_ok=False)
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
        import hashlib
        for name in archive.namelist():
            if _sha(Path(a.source) / name) != hashlib.sha256(archive.read(name)).hexdigest():
                raise ValueError('Reference original snapshot bytes changed: ' + name)
        archive.extractall(source)  # Every member/path was checked above.
    proof = activate_original(source)
    receipt = bounded_scope.install(a.margin_mm, source)
    install_checkpoint_binding(receipt)
    import torch
    from torch.utils.data import DataLoader
    from hiercp import contracts
    from hiercp.data import collate_samples
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import set_seed
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA required')
    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < a.cuda_gib * 2**30 < total: raise ValueError('Explicit GPU headroom required')
    torch.cuda.set_per_process_memory_fraction(a.cuda_gib * 2**30 / total)
    torch.set_num_threads(a.workers)
    budget = ResourceBudget(a.cuda_gib, a.rss_gib)
    config = json.loads((source / 'config/train.json').read_text())
    samples, manifest = load_samples(a.fixture, 2, 1, budget)
    samples, preparation = rebuild_scope(samples, manifest, config, bounded_scope,
        a.margin_mm, a.workers, budget)
    dataset = FixedCanonicalDataset([sample for sample in samples if sample['split'] == 'train'])
    direct = collate_samples([dataset[index] for index in range(len(dataset))])
    loader = DataLoader(dataset, batch_size=2, num_workers=a.workers,
        multiprocessing_context='spawn', collate_fn=collate_samples,
        worker_init_fn=ScopeWorkerInitializer(str(source), a.margin_mm,
            receipt['contract_sha256']), pin_memory=True)
    iterator = iter(loader)
    spawned = next(iterator)
    try: next(iterator)
    except StopIteration: pass
    else: raise AssertionError('Unexpected additional DEBUG batch')
    del iterator, loader
    assert_equal_batches(direct, spawned)
    set_seed(42, deterministic=config['runtime']['deterministic'])
    model = HierarchicalPyGPlacementModel(**config['model']).cuda().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 10434532
    # This checks transfer/worker parity in FP32. The separate learning smoke
    # uses the native AMP setting. Half-precision atomic reductions can differ
    # across repeated forwards even when every input tensor is identical.
    with torch.no_grad():
        direct_result = model(independent_transfer(direct))
        spawned_result = model(independent_transfer(spawned))
    for first, second in zip(direct_result.scores, spawned_result.scores):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
    graph = bounded_scope.configure(config['graph'], a.margin_mm).to_dict()
    checkpoint = dict(architecture_version=model.architecture_version,
        geometry_contract=contracts.GEOMETRY_CONTRACT, graph_config=graph,
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()})
    contracts.require_current_checkpoint(checkpoint)
    rejected = []
    for variant in ('native_architecture', 'missing_marker', 'wrong_marker'):
        invalid = dict(checkpoint, state_dict=dict(checkpoint['state_dict']))
        if variant == 'native_architecture': invalid['architecture_version'] = contracts.ARCHITECTURE_VERSION
        elif variant == 'missing_marker': invalid['state_dict'].pop('v1x_bounded_scope_digest')
        else: invalid['state_dict']['v1x_bounded_scope_digest'] = torch.zeros(32, dtype=torch.uint8)
        try: contracts.require_current_checkpoint(invalid)
        except ValueError: rejected.append(variant)
        else: raise AssertionError('Unbound actual neural state was accepted')
    report = dict(status='PASS', debug=True, actual_CT=True, actual_CUDA=True,
        full_training=False, full_evaluation=False, quality_verified=False, production_ready=False,
        optimizer_steps=0, checkpoint_written=False, model_parameters=10434532,
        GPU=torch.cuda.get_device_name(0), device_total_bytes=total,
        physical_sample_batch=2, candidate_graph_batch=16, workers=a.workers,
        worker_start_method='spawn', workers_apply_exact_scope=True,
        worker_graphs_equal=True, worker_CUDA_scores_equal=True,
        worker_score_comparison_precision='FP32 diagnostic; training AMP unchanged',
        worker_score_max_absolute_difference=max(float((first - second).abs().max())
            for first, second in zip(direct_result.scores, spawned_result.scores)),
        actual_neural_scope_state_bound=True, rejected_checkpoints=rejected,
        original_source=proof, scope_contract=receipt, preparation=preparation,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(f'ACTUAL CT/CUDA scope worker smoke PASS | {output}', flush=True)


if __name__ == '__main__': main()
