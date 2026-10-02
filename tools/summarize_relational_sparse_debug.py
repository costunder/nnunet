"""Compact numeric receipt of the complete local relational L0 DEBUG report.

The full report stays in its original task-owned directory. This export retains
all 133 record-level counts and all measured updates; it exports no CT pixels,
annotation contours or learned checkpoints. No model is executed here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    r = json.loads(path.read_text(encoding='utf-8'))
    if (not r['debug'] or r['production_ready'] or r['checkpoint_written']
            or r['full_training'] or r['full_evaluation']
            or r['complete_case_observations'] != 133
            or r['physical_batch'] != 32
            or not r['implementation_unchanged_during_run']
            or not all(r['assertions'].values())):
        raise ValueError('The complete, immutable, explicitly scoped CUDA DEBUG receipt is required')
    out = {k: v for k, v in r.items() if k not in ('budgets', 'visualization')}
    out['raw_report'] = dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size,
        preserved=True, model_executed_by_this_reader=False)
    out['budgets'] = {}
    for quota, branches in r['budgets'].items():
        out['budgets'][quota] = {}
        for variant, b in branches.items():
            rows = b['scenes']
            if len(rows) != 133 or len({x['id'] for x in rows}) != 133:
                raise ValueError('Incomplete or duplicate original case observations')
            audits = b['graph_audits']
            indices = [i for a in audits for i in a['original_indices']]
            if indices != list(range(133)):
                raise ValueError('Natural physical32 batches did not preserve the complete record order')
            isolates = [n for a in audits for n in a['statistics'].get('pair_isolated_node_counts', [])]
            if variant == 'baseline':
                # Every baseline node connects to its own query. This is a
                # verified property of the exported adjacency, not a repair.
                isolates = [0] * len(rows)
            if len(isolates) != len(rows):
                raise ValueError('Actual per-record isolation counts are incomplete')
            records = []
            for i, row in enumerate(rows):
                count = sum(row['relation_counts'].values()) if variant == 'relational' else row['joint_directed_edges']
                records.append(dict(id=row['id'], kind=row['kind'], nodes=row['pair_node_count'],
                    directed_edges=count, cross_edges=row['cross_edges'],
                    weak_components=row['components'], isolated_nodes=isolates[i],
                    relation_counts=row.get('relation_counts'),
                    unmatched_target_counts=row.get('unmatched_target_counts'),
                    unmatched_source_counts=row.get('unmatched_source_counts')))
            compact = {k: v for k, v in b.items() if k not in ('graph_audits', 'scenes')}
            compact['records'] = records
            compact['graph_audits'] = [dict(original_indices=a['original_indices'],
                load_transfer=a['load_transfer'], forward_seconds=a['forward_seconds'],
                forward_phases_seconds=a['forward_phases_seconds'], coverage_seconds=a['coverage_seconds'],
                actual_scene_node_counts=a['actual_scene_node_counts'],
                actual_scene_directed_edge_counts=a['actual_scene_directed_edge_counts'],
                relation_edge_audit=a['relation_edge_audit']) for a in audits]
            compact['count_summary'] = dict(
                records=len(rows), P=sum(x['kind'] == 'P' for x in rows), U=sum(x['kind'] == 'U' for x in rows),
                nodes=sorted({x['nodes'] for x in records}),
                directed_edges_range=[min(x['directed_edges'] for x in records), max(x['directed_edges'] for x in records)],
                weak_components_range=[min(x['weak_components'] for x in records), max(x['weak_components'] for x in records)],
                isolated_nodes_range=[min(isolates), max(isolates)],
                zero_cross_pairs=sum(x['cross_edges'] == 0 for x in records),
                cross_edges_range=[min(x['cross_edges'] for x in records), max(x['cross_edges'] for x in records)])
            out['budgets'][quota][variant] = compact
    out['memory_interpretation'] = 'Peak allocated bytes are whole-process CUDA counts with six comparative models resident, not isolated single-model VRAM admission.'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    out = summarize(a.report)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(out, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(a.output)


if __name__ == '__main__':
    main()
