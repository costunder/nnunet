"""Read-only actual-bank diagnostic; independent fullN feasibility certificate.

Never changes any bank, trainer, controller, checkpoint or experiment result.
The output must be a NEW file. CPU-only process, no CUDA initialization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def independent_clique(adjacency, needed):
    """Exact bitset k-clique decision with greedy-color upper bound, no cap.

    Independent from the production catalog/DFS/index arithmetic. A proper
    coloring bounds clique size; all remaining finite branches are explored.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.shape[0] != adjacency.shape[1] or np.any(np.diag(adjacency)) or not np.array_equal(adjacency, adjacency.T):
        raise ValueError('Independent graph must be square, symmetric and loop-free')
    size = len(adjacency)
    neighbors = [sum(1 << int(v) for v in np.flatnonzero(row)) for row in adjacency]
    selected = []; explored = 0; witness = None

    def coloring(pool):
        remaining = pool; order = []; bounds = []; color = 0
        while remaining:
            color += 1; independent = remaining
            while independent:
                bit = independent & -independent; vertex = bit.bit_length() - 1
                order.append(vertex); bounds.append(color)
                remaining &= ~bit
                independent &= ~bit & ~neighbors[vertex]
        return order, bounds

    def search(pool):
        nonlocal explored, witness
        explored += 1
        if len(selected) >= needed:
            witness = selected.copy(); return True
        if pool.bit_count() + len(selected) < needed:
            return False
        order, bounds = coloring(pool)
        for i in range(len(order) - 1, -1, -1):
            if bounds[i] + len(selected) < needed:
                return False
            vertex = order[i]; bit = 1 << vertex
            selected.append(vertex)
            if search(pool & neighbors[vertex]):
                return True
            selected.pop(); pool &= ~bit
        return False

    feasible = search((1 << size) - 1)
    return dict(feasible=feasible, needed_extra_copies=needed, available_primary_compatible_placements=size,
                explored_exact_clique_nodes=explored, selected_graph_indices=witness,
                proof='Exhaustive independent bitset clique search with valid greedy-color upper bounds; no search caps')


def exact_overlap(first, second):
    """Independent translated mask-intersection implementation."""
    low = np.maximum(first['start'], second['start'])
    high = np.minimum(first['stop'], second['stop'])
    if np.any(low >= high):
        return False
    first_slice = tuple(slice(int(v), int(w)) for v, w in zip(low - first['start'], high - first['start']))
    second_slice = tuple(slice(int(v), int(w)) for v, w in zip(low - second['start'], high - second['start']))
    return bool(np.any(first['entry']['source_mask'][first_slice].astype(bool, copy=False)
                       & second['entry']['source_mask'][second_slice].astype(bool, copy=False)))


def independent_geometry(entries, primary_source_index, primary_center, shape, count):
    """Build a separate exact compatibility graph for an unchanged primary."""
    maximum = np.asarray([128] * 3); shape = np.asarray(shape)
    entry = entries[primary_source_index]
    center = np.asarray(primary_center, dtype=np.int64)
    start = center - entry['anchor_offset']; stop = start + np.asarray(entry['source_mask'].shape)
    primary = dict(entry=entry, center=center, start=start, stop=stop)
    if np.any(start < 0) or np.any(stop > shape) or np.any(stop - start > maximum):
        raise ValueError('Actual primary source patch itself is outside the declared patient/network bounds')
    points = []
    for source_index, source in enumerate(entries):
        for candidate_index, point in enumerate(source['candidate_centers']):
            point = np.asarray(point, dtype=np.int64)
            low = point - source['anchor_offset']; high = low + np.asarray(source['source_mask'].shape)
            if np.any(low < 0) or np.any(high > shape) or np.all(point == center):
                continue
            if np.any(np.maximum(stop, high) - np.minimum(start, low) > maximum):
                continue
            item = dict(entry=source, center=point, start=low, stop=high,
                        source_index=source_index, candidate_index=candidate_index)
            if not exact_overlap(primary, item):
                points.append(item)
    size = len(points); matrix = np.zeros((size, size), dtype=bool)
    if size:
        all_starts = np.stack([p['start'] for p in points]); all_stops = np.stack([p['stop'] for p in points])
        all_centers = np.stack([p['center'] for p in points])
        for i, first in enumerate(points):
            possible = (np.arange(size) > i) & np.any(all_centers != first['center'], axis=1)
            possible &= np.all(np.maximum(all_stops, first['stop']) - np.minimum(all_starts, first['start']) <= maximum, axis=1)
            for j in np.flatnonzero(possible):
                if not exact_overlap(first, points[j]):
                    matrix[i, j] = matrix[j, i] = True
    result = independent_clique(matrix, count - 1)
    result['compatible_pair_count'] = int(np.count_nonzero(matrix) // 2)
    result['primary_patch_start'] = start.tolist(); result['primary_patch_stop'] = stop.tolist()
    result['available_placements'] = [dict(source_index=p['source_index'], candidate_index=p['candidate_index'],
                                         center=p['center'].tolist(), patch_start=p['start'].tolist(), patch_stop=p['stop'].tolist())
                                     for p in points]
    return result


_WORKER = None


def initialize(native):
    global _WORKER
    from tools import run_basic_cp_expanded as runner
    os.environ.update(runner.environment(native))
    module, adapter = runner.initialize_native(native)
    if module.historical.torch.cuda.is_initialized():
        raise RuntimeError('Read-only geometry diagnostic must run before CUDA initialization')
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    dataset = nnUNetDatasetBlosc2(native['private_data_folder'], list(native['split']['outer_train']))
    bank = module.ExpandedBank(native['bank'], arm=native['arm'])
    if module.sha(native['bank']) != native['bank_sha256']:
        raise RuntimeError('Actual requested bank SHA differs')
    _WORKER = (native, module, dataset, bank)


def inspect_case(case_id):
    native, module, dataset, bank = _WORKER
    started = time.perf_counter(); entries = bank.case_entries(case_id); names = bank.entry_names(case_id)
    _, segmentation, _, _ = dataset.load_case(case_id)
    shape = tuple(int(v) for v in segmentation.shape[-3:]); geometry = module.CandidateGeometry(entries)
    count = module.copies_per_event(native['arm']); checked = 0
    try:
        for source_index, entry in enumerate(entries):
            for candidate_index, center in enumerate(entry['candidate_centers']):
                primary = dict(entry=entry, center=tuple(map(int, center)))
                try:
                    extras = geometry.choose(primary, np.zeros(3, dtype=np.int64), np.asarray(shape), count)
                    module.joint_network_crop(shape, [primary, *extras])
                except module.OnlineCPError as error:
                    independent = independent_geometry(entries, source_index, center, shape, count)
                    result = dict(case_id=case_id, arm=native['arm'], source_name=names[source_index],
                        source_index=source_index, candidate_index=candidate_index, center=list(map(int, center)),
                        actual_case_shape=list(shape), actual_source_patch_shape=list(entry['source_mask'].shape),
                        actual_source_mask_voxels=int(np.count_nonzero(entry['source_mask'])),
                        original_error=str(error), preceding_primary_positions_checked=checked,
                        independent_proof=independent, algorithm_disagreement=independent['feasible'],
                        bank_sha256=module.sha(native['bank']), source_sha256=module.sha(bank.root / names[source_index]),
                        seconds=time.perf_counter() - started)
                    print('READONLY_MULTIN_GEOMETRY_FAILURE ' + json.dumps(result), flush=True)
                    return result
                checked += 1
        return dict(case_id=case_id, arm=native['arm'], full_primary_positions_checked=checked,
                    no_failure=True, seconds=time.perf_counter() - started)
    finally:
        bank._cache.clear(); bank._validated_entries.clear(); bank._patient_geometry.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', action='append')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Read-only diagnostic output must be a new file')
    native = json.loads(args.native.read_text(encoding='utf8'))
    if native['arm'] not in ('multi5', 'multi7', 'multi10'):
        raise ValueError('Exact multi5/multi7/multi10 native request required')
    if args.workers < 1:
        raise ValueError('Positive explicit CPU diagnostic worker count required')
    metadata = json.loads(Path(native['bank']).read_text(encoding='utf8'))
    cases = args.case or sorted(metadata['entries_by_case'])
    if any(case not in metadata['entries_by_case'] for case in cases):
        raise ValueError('Requested diagnostic case is outside the original81 source patients')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork'),
                             initializer=initialize, initargs=(native,)) as pool:
        results = list(pool.map(inspect_case, cases))
    result = dict(format='basic_cp_multin_independent_geometry_readonly_v1', native_path=str(args.native),
                  arm=native['arm'], cases=results, debug_diagnostic=True, training_executed=False,
                  candidate_sources_modified=False, live_code_modified=False,
                  algorithm_disagreements=sum(bool(row.get('algorithm_disagreement')) for row in results))
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print('READONLY_MULTIN_DIAGNOSTIC_COMPLETE ' + json.dumps(dict(output=str(args.output),
          arm=native['arm'], cases=len(results), algorithm_disagreements=result['algorithm_disagreements'])), flush=True)


if __name__ == '__main__': main()
