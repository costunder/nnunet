"""Numeric receipt of complete CUDA relay sampling evidence; no model runs.

Both branches are typed relational graphs. The previous seed-only branch is
not the older independent-query-hub baseline. No CT pixels, annotation contours
or learned weights are copied into this summary.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    initial_hash = sha(path)
    report = json.loads(path.read_text(encoding='utf8'))
    if (not report['debug'] or not report['actual_CT'] or not report['actual_CUDA']
            or report['production_ready'] or report['checkpoint_written']
            or report['full_training'] or report['full_evaluation']
            or report['physical_batch']!=32 or report['actual_case_observations_used']!=133
            or not report['implementation_unchanged_during_run']
            or not all(report['assertions'].values())):
        raise ValueError('Complete immutable actual-CT CUDA DEBUG scope required')
    output={key:value for key,value in report.items() if key not in ('budgets','visualization')}
    output['raw_report']={'path':str(path.resolve()),'sha256':initial_hash,'bytes':path.stat().st_size}
    output['branch_semantics']={'baseline':'Previous relational seed-only sampler',
        'relational':'Same seeds/weights; native path relays + mandatory context edges + exact scene reuse'}
    output['budgets']={}
    for quota,branches in report['budgets'].items():
        output['budgets'][quota]={}
        for name,branch in branches.items():
            rows=branch['scenes']
            if (len(rows)!=133 or len({row['id'] for row in rows})!=133
                    or [i for a in branch['graph_audits'] for i in a['original_indices']]!=list(range(133))):
                raise ValueError('Original candidate order/coverage is incomplete')
            records=[{key:row[key] for key in ('id','kind','donor_node_count','recipient_node_count',
                'pair_node_count','components','isolated_nodes','relation_counts','cross_edges',
                'seed_context_counts','relay_counts','native6_substrate_unreachable_counts')}
                for row in rows]
            compact={key:value for key,value in branch.items() if key not in ('scenes','graph_audits')}
            compact['records']=records
            compact['graph_audits']=branch['graph_audits']
            edges=[sum(row['relation_counts'].values()) for row in records]
            compact['count_summary']={
                'records':133,'P':sum(row['kind']=='P' for row in records),'U':sum(row['kind']=='U' for row in records),
                'nodes_range':[min(row['pair_node_count'] for row in records),max(row['pair_node_count'] for row in records)],
                'directed_edges_range':[min(edges),max(edges)],
                'weak_components_range':[min(row['components'] for row in records),max(row['components'] for row in records)],
                'isolated_nodes_range':[min(row['isolated_nodes'] for row in records),max(row['isolated_nodes'] for row in records)],
                'relay_nodes_range':[min(sum(row['relay_counts']) for row in records),max(sum(row['relay_counts']) for row in records)],
                'cross_edges_range':[min(row['cross_edges'] for row in records),max(row['cross_edges'] for row in records)],
                'native6_substrate_unreachable_seed_occurrences':sum(sum(row['native6_substrate_unreachable_counts']) for row in records)
                    if name=='relational' else None}
            output['budgets'][quota][name]=compact
    if sha(path)!=initial_hash:
        raise ValueError('Raw evidence changed while reading')
    output['memory_interpretation']='Whole-process CUDA allocation with six comparative models resident; not production/worst-case admission.'
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.report)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(args.output)


if __name__=='__main__':
    main()
