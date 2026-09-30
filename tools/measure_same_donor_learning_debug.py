"""Short actual-CT learnability curve; no production checkpoint or admission."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from l0_regions.donor_data import DonorDataset, DonorLoader
from l0_regions.training_data import Budget, write_new, sha, source_identity
from l0_regions.training import (make_model, calibrate, initial_memory, evaluate,
    groups, legacy_groups, ranking_context, rank_config, forward_loss)
from l0_regions.support_episodes import PatientEpisodes
from l0_regions.execution_pipeline import gradient_check_batched
from l0_regions.sparse import workspace
from hiercp_v222.v1_execution import rng_state, restore_rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'fine-cache', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--debug-passes', type=int, required=True)
    a = p.parse_args()
    if a.debug_passes <= 0:
        raise ValueError('Positive explicit DEBUG pass count required')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ds = DonorDataset(a.cache, 'inner_train', True, 'research-report', a.fine_cache)
    val = DonorDataset(a.cache, 'inner_val', True, 'research-report', a.fine_cache)
    budget = Budget(8 * 2**30, 24 * 2**30)
    loader = DonorLoader(ds, 4, 4 * 2**30, rss_limit=budget.rss_bytes)
    vl = DonorLoader(val, 4, 4 * 2**30, store=loader.store, rss_limit=budget.rss_bytes)
    net = make_model(ds, budget, True, 'retained')
    base = ds.meta['base']; seed = ds.meta['config']['seed']
    report = dict(debug=True, full_training=False, production_ready=False,
        scope='Actual CT fixed small-cohort learnability only; not 128-candidate generalization or CP Dice',
        gpu=torch.cuda.get_device_name(), train_records=len(ds), validation_records=len(val),
        train_cases=len({r['case_id'] for r in ds.rows}), validation_cases=len({r['case_id'] for r in val.rows}),
        physical_batch=2, support_patients=2, workers=4, precision='FP32', seed=seed,
        parameter_count=sum(p.numel() for p in net.parameters()),
        cache_sha256=sha(a.cache), fine_cache_sha256=sha(a.fine_cache), source=source_identity(),
        snapshots=[], updates=[])
    started = time.perf_counter()
    with workspace(64 * 2**20):
        batch, calibration, memory = calibrate(net, ds, loader, [2], base, budget, support_patients=2)
        report['calibration'] = calibration
        optimizer = torch.optim.AdamW(net.parameters(), lr=base['training']['lr'],
            weight_decay=base['training']['weight_decay'], fused=base['training']['fused_optimizer'])
        counts = torch.bincount(memory['classes'], minlength=2).float()
        weights = counts.sum() / (2 * counts)
        def snapshot(step):
            saved = rng_state()
            train_metrics, train_cases = evaluate(net, ds, loader, memory, batch)
            val_metrics, val_cases = evaluate(net, val, vl, memory, batch)
            restore_rng(saved)
            row = dict(step=step, train=train_metrics, validation=val_metrics,
                train_details=train_cases, validation_details=val_cases)
            report['snapshots'].append(row)
            write_new(a.output / f'step_{step:04d}.json', row)
            print(json.dumps(dict(step=step, train=train_metrics, validation=val_metrics)), flush=True)
        snapshot(0)
        step = 0
        for epoch in range(a.debug_passes):
            order = list(groups(ds, batch, seed, epoch))
            episodes = PatientEpisodes(ds.rows, list(legacy_groups(ds, batch, seed, epoch)),
                2, seed, epoch).bind(memory)
            context = ranking_context(ds, memory, batch)
            last = None
            net.train()
            for cpu in loader.batches(order):
                torch.cuda.synchronize(); tick = time.perf_counter()
                ids = cpu.indices.tolist(); group = ds.rows[ids[0]]['patient_group']
                support = episodes.support(group)
                if group != last:
                    net.eval()
                    with torch.no_grad():
                        plan = net.fit_support_clusters(*support)
                    net.train(); last = group
                optimizer.zero_grad(set_to_none=True)
                query = cpu.to('cuda')
                loss, terms = forward_loss(net, query, support, plan, memory['classes'][ids],
                    weights, context, rank_config(ds), indices=ids)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite diagnostic loss')
                loss.backward(); gradient_check_batched(net)
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), base['training']['grad_clip'], error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize(); step += 1
                report['updates'].append(dict(step=step, epoch=epoch+1, loss=float(loss.detach()),
                    grad_norm=float(norm), seconds=time.perf_counter()-tick))
                budget.check()
                del loss, terms, query, support
            memory = initial_memory(net, ds, loader, batch)
            snapshot(step)
    report.update(elapsed_seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=step)
    write_new(a.output / 'report.json', report)
    print('DEBUG REPORT:', a.output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
