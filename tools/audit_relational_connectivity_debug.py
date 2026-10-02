"""Independent CPU graph-statistics audit of saved relational DEBUG exports.

Read actual exported source/target edges, not a CPU model implementation. BFS
checks weak components against the CUDA report and CPU export, including exact
CUDA component labels after model-slot remapping. Directed BFS records both
incoming ancestors and outgoing influence within the model's three layers.
This reader never imports torch, executes a model, changes a graph, repairs a
connection, or writes into an existing evidence file.
"""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROLE_NAMES = tuple(f'{side}_{role}' for side in ('donor', 'recipient')
                   for role in ('query', 'near', 'mid', 'wide'))
RELATION_NAMES = (
    'source_context_neighbor', 'target_context_neighbor',
    'source_query_to_context', 'source_context_to_query',
    'target_query_to_context', 'target_context_to_query',
    'source_context_to_target_context', 'source_query_to_target_context',
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    if not values:
        return dict(count=0, min=None, max=None, mean=None)
    return dict(count=len(values), min=min(values), max=max(values),
                mean=statistics.fmean(values))


def role_counts(nodes, indices):
    counts = Counter(nodes[index]['side'] + '_' + nodes[index]['role'] for index in indices)
    return {role: counts[role] for role in ROLE_NAMES}


def weak_components(outgoing, incoming):
    """Independent Python set/BFS over actual edges, including isolates."""
    unseen = set(range(len(outgoing)))
    components = []
    while unseen:
        start = min(unseen)
        reached, pending = {start}, deque([start])
        while pending:
            node = pending.popleft()
            for neighbor in outgoing[node] | incoming[node]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        unseen.difference_update(reached)
        components.append(reached)
    return components


def reachable(start, neighbors, limit=None):
    """Shortest directed paths; limit includes the start node at zero hops."""
    distance = {start: 0}
    pending = deque([start])
    while pending:
        node = pending.popleft()
        if limit is not None and distance[node] >= limit:
            continue
        for neighbor in neighbors[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                pending.append(neighbor)
    return set(distance)


def report_binding(arm, record_id):
    """Find the original case record and its CUDA graph statistics row."""
    matches = [index for index, scene in enumerate(arm['scenes']) if scene['id'] == record_id]
    if len(matches) != 1:
        raise ValueError('Export record must match exactly one original report scene: ' + record_id)
    original_index = matches[0]
    bound = [(audit, audit['original_indices'].index(original_index))
             for audit in arm['graph_audits'] if original_index in audit['original_indices']]
    if len(bound) != 1:
        raise ValueError('Original record must occur in exactly one physical graph audit: ' + record_id)
    return arm['scenes'][original_index], bound[0][0]['statistics'], bound[0][1], original_index


def audit_record(candidate, arm, quota):
    graph, nodes = candidate['joint'], candidate['joint']['nodes']
    if graph['edge_orientation'] != 'source,target,relation_name' or not nodes:
        raise ValueError('Explicit nonempty source/target relational graph export required')
    width = 1 + 3 * quota
    node_ids = []
    for node in nodes:
        role = node['side'] + '_' + node['role']
        if role not in ROLE_NAMES or type(node['model_slot']) is not int or not 0 <= node['model_slot'] < width:
            raise ValueError('Unknown role or invalid original model slot')
        if (len(node['relative_xyz_mm']) != 3
                or not all(math.isfinite(value) for value in node['relative_xyz_mm'])):
            raise ValueError('Finite original anchor-relative physical coordinates required')
        node_ids.append(node['model_slot'] + (width if node['side'] == 'recipient' else 0))
    if len(set(node_ids)) != len(nodes):
        raise ValueError('Duplicate original pair model node index')
    queries = {side: [index for index, node in enumerate(nodes)
                      if node['side'] == side and node['role'] == 'query']
               for side in ('donor', 'recipient')}
    if any(len(indices) != 1 for indices in queries.values()):
        raise ValueError('Exactly one unchanged abstract query per branch required')
    query_ids = {indices[0] for indices in queries.values()}
    outgoing, incoming = [set() for _ in nodes], [set() for _ in nodes]
    typed_edges = set()
    for source, target, relation in graph['edges']:
        if (type(source) is not int or type(target) is not int
                or not 0 <= source < len(nodes) or not 0 <= target < len(nodes)
                or source == target or relation not in RELATION_NAMES):
            raise ValueError('Invalid actual typed edge endpoint or relation')
        if (source, target, relation) in typed_edges:
            raise ValueError('Duplicate typed export edge')
        typed_edges.add((source, target, relation))
        outgoing[source].add(target)
        incoming[target].add(source)
    components = weak_components(outgoing, incoming)
    isolate_ids = {index for index in range(len(nodes)) if not outgoing[index] and not incoming[index]}
    no_query_ids = set().union(*(part for part in components if part.isdisjoint(query_ids)))
    scene, gpu, row, original_index = report_binding(arm, candidate['id'])
    actual_components = len(components)
    expected = (scene['components'], gpu['pair_components'][row], graph['stats']['components'])
    if any(value != actual_components for value in expected):
        raise ValueError(f'Independent weak component count differs for {candidate["id"]}: {actual_components} vs {expected}')
    if gpu['pair_isolated_node_counts'][row] != len(isolate_ids):
        raise ValueError('Independent isolate count differs from CUDA report: ' + candidate['id'])
    expected_labels = gpu['pair_component_labels'][row]
    components_as_model_ids = []
    for part in components:
        model_ids = sorted(node_ids[index] for index in part)
        label = min(model_ids)
        if any(expected_labels[model_id] != label for model_id in model_ids):
            raise ValueError('Independent component partition differs from CUDA labels: ' + candidate['id'])
        components_as_model_ids.append(model_ids)
    query_paths = {}
    all_ids = set(range(len(nodes)))
    for side, indices in queries.items():
        query = indices[0]
        ancestors = reachable(query, incoming)
        ancestors3 = reachable(query, incoming, 3)
        influence = reachable(query, outgoing)
        influence3 = reachable(query, outgoing, 3)
        query_paths[side + '_query'] = dict(
            model_node_id=node_ids[query], ancestors_all_count=len(ancestors),
            ancestors_3_layers_count=len(ancestors3),
            ancestors_3_layers_model_ids=sorted(node_ids[index] for index in ancestors3),
            cannot_send_to_query_all_by_role=role_counts(nodes, all_ids - ancestors),
            cannot_send_to_query_3_layers_by_role=role_counts(nodes, all_ids - ancestors3),
            influence_all_count=len(influence), influence_3_layers_count=len(influence3),
            influence_3_layers_model_ids=sorted(node_ids[index] for index in influence3),
            reached_from_query_3_layers_by_role=role_counts(nodes, influence3))
    nearest_context = []
    for index in sorted(isolate_ids):
        node = nodes[index]
        candidates = [other for other_index, other in enumerate(nodes)
                      if other_index != index and other['side'] == node['side'] and other['role'] != 'query']
        nearest = (min(math.dist(node['relative_xyz_mm'], other['relative_xyz_mm']) for other in candidates)
                   if candidates else None)
        nearest_context.append(dict(model_node_id=node_ids[index],
            role=node['side'] + '_' + node['role'], nearest_same_branch_context_mm=nearest))
    return dict(id=candidate['id'], kind=candidate['kind'], original_report_index=original_index,
        node_count=len(nodes), directed_typed_edges=len(typed_edges),
        weak_components=actual_components,
        weak_component_sizes=sorted(map(len, components), reverse=True),
        weak_components_model_ids=components_as_model_ids,
        isolated_count=len(isolate_ids), isolated_by_role=role_counts(nodes, isolate_ids),
        isolated_model_ids=sorted(node_ids[index] for index in isolate_ids),
        isolated_nearest_context=nearest_context,
        outside_both_query_components_count=len(no_query_ids),
        outside_both_query_components_by_role=role_counts(nodes, no_query_ids),
        query_paths=query_paths,
        verified=dict(CUDA_component_count=True, CUDA_component_partition=True,
                      CUDA_isolate_count=True, CPU_export_component_count=True))


def budget_summary(records, arm):
    fields = ('weak_components', 'isolated_count', 'outside_both_query_components_count')
    result = dict(displayed_records=len(records),
        displayed_kind_counts=dict(Counter(record['kind'] for record in records)),
        independent_displayed={field: distribution([record[field] for record in records]) for field in fields},
        displayed_isolated_by_role_total={role: sum(record['isolated_by_role'][role] for record in records) for role in ROLE_NAMES},
        displayed_outside_query_components_by_role_total={role: sum(record['outside_both_query_components_by_role'][role] for record in records) for role in ROLE_NAMES})
    result['directed_query_paths'] = {}
    for query in ('donor_query', 'recipient_query'):
        names = ('ancestors_all_count', 'ancestors_3_layers_count', 'influence_all_count', 'influence_3_layers_count')
        summary = {name: distribution([record['query_paths'][query][name] for record in records]) for name in names}
        for name in ('cannot_send_to_query_3_layers_by_role', 'reached_from_query_3_layers_by_role'):
            summary[name] = {role: distribution([record['query_paths'][query][name][role] for record in records]) for role in ROLE_NAMES}
        result['directed_query_paths'][query] = summary
    result['reported_all_case_only'] = dict(records=len(arm['scenes']),
        note='All-case ranges are read from the CUDA report; independent edge BFS covers the displayed exports only',
        weak_components=distribution([scene['components'] for scene in arm['scenes']]),
        isolates=distribution([count for audit in arm['graph_audits'] for count in audit['statistics']['pair_isolated_node_counts']]))
    nearest = [entry['nearest_same_branch_context_mm'] for record in records for entry in record['isolated_nearest_context']
               if entry['nearest_same_branch_context_mm'] is not None]
    result['isolated_nearest_same_branch_context_mm'] = distribution(nearest)
    return result


def run(payload_path, report_path, output_path):
    if output_path.exists():
        raise FileExistsError('Prior audit evidence is preserved; select a new --output path')
    payload_path, report_path = payload_path.resolve(), report_path.resolve()
    source_hashes = dict(payload=sha256(payload_path), report=sha256(report_path))
    with payload_path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    with report_path.open(encoding='utf-8') as stream:
        report = json.load(stream)
    if not report['debug'] or not report['actual_CT'] or not report['actual_CUDA'] or payload['case'] != report['case']:
        raise ValueError('Matching actual-CT/CUDA DEBUG report and export required')
    budgets = {}
    for budget in payload['budgets']:
        context, quota = budget['context_nodes'], budget['quota']
        if context != 3 * quota:
            raise ValueError('Explicit context quota differs from exported budget')
        arm = report['budgets'][str(context)]['relational']
        candidates = budget['variants']['relational']['candidates']
        if len({candidate['id'] for candidate in candidates}) != len(candidates):
            raise ValueError('Duplicate displayed original candidate')
        records = [audit_record(candidate, arm, quota) for candidate in candidates]
        expected_count = report['visualization']['displayed_P'] + report['visualization']['displayed_U']
        if len(records) != expected_count:
            raise ValueError('Not every displayed candidate was independently audited')
        budgets[str(context)] = dict(quota=quota, summary=budget_summary(records, arm), records=records)
    if source_hashes != dict(payload=sha256(payload_path), report=sha256(report_path)):
        raise RuntimeError('Saved numerical/export evidence changed during the read-only audit')
    evidence = dict(format='independent_relational_connectivity_DEBUG_v1', debug=True,
        created_utc=datetime.now(timezone.utc).isoformat(), case=report['case'],
        scope='Independent Python CPU BFS on saved actual edges; no CPU model or GPU execution',
        executed_model=False, executed_GPU=False, repaired_graph=False,
        sources=dict(payload=dict(path=str(payload_path), sha256=source_hashes['payload']),
                     report=dict(path=str(report_path), sha256=source_hashes['report'])),
        implementation=dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__).resolve())),
        evidence_unchanged_during_read=True, independent_records=sum(len(budget['records']) for budget in budgets.values()),
        all_component_counts_partitions_isolates_match=True,
        directed_hops=3, budgets=budgets,
        interpretation=['Weak connectivity does not establish directed influence into a query.',
            'Three-hop reachability is structural; it is not a numerical attribution or quality metric.',
            'Role/shell pooling reads every valid node hidden state, including isolates; query reachability is not readout inclusion.',
            'This DEBUG graph does not add v1 interface/hop relay closure or repair radius gaps.'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output_path.resolve()), independent_records=evidence['independent_records'],
        all_component_counts_partitions_isolates_match=True,
        budgets={context: budget['summary']['independent_displayed'] for context, budget in budgets.items()}), separators=(',', ':')))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.payload, args.report, args.output)


if __name__ == '__main__':
    main()
