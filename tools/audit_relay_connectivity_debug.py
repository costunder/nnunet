"""Independent JSON-only audit of actual retained-relay DEBUG graph exports.

This reader performs CPU set/BFS and double-precision coordinate checks on
saved CUDA exports. It never imports torch, executes a CPU model, creates an
edge, changes a partition, or accesses CT/mask pixels. Independent BFS covers
the 48 displayed graphs; the 133-record cohort summaries remain CUDA-reported
metrics. Parent-path support is not independently rechecked against mask pixels.
"""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_MM = 1e-5
SIDES = ('donor', 'recipient')
ROLES = ('query', 'near', 'mid', 'wide')
GLOBAL_ROLES = tuple(side + '_' + role for side in SIDES for role in ROLES)
RELATIONS = (
    'source_context_neighbor', 'target_context_neighbor',
    'source_query_to_context', 'source_context_to_query',
    'target_query_to_context', 'target_context_to_query',
    'source_context_to_target_context', 'source_query_to_target_context',
)
RADII = (6., 6., 8., 8., 8., 8., 5., 8.)
RULES = (
    ('donor', False, 'donor', False),
    ('recipient', False, 'recipient', False),
    ('donor', True, 'donor', False),
    ('donor', False, 'donor', True),
    ('recipient', True, 'recipient', False),
    ('recipient', False, 'recipient', True),
    ('donor', False, 'recipient', False),
    ('donor', True, 'recipient', False),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def vector(values):
    require(isinstance(values, list) and len(values) == 3
            and all(type(value) in (int, float) and math.isfinite(value)
                    for value in values), 'A finite three-dimensional numeric export is required')
    return [float(value) for value in values]


def same_vector(first, second):
    return all(abs(a - b) <= TOLERANCE_MM for a, b in zip(vector(first), vector(second)))


def fp32(value):
    """IEEE-754 serialization, not a CPU model or an enlarged tolerance."""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def native_mm(node, spacing):
    return [value * scale for value, scale in zip(vector(node['native_voxel']), spacing)]


def distribution(values):
    return dict(count=len(values), min=min(values) if values else None,
                max=max(values) if values else None,
                mean=statistics.fmean(values) if values else None,
                sum=sum(values))


def role_counts(nodes, indices):
    counts = Counter(nodes[index]['side'] + '_' + nodes[index]['role'] for index in indices)
    return {role: counts[role] for role in GLOBAL_ROLES}


def weak_components(outgoing, incoming):
    unseen, components = set(range(len(outgoing))), []
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
    depth, pending = {start: 0}, deque([start])
    while pending:
        node = pending.popleft()
        if limit is not None and depth[node] == limit:
            continue
        for neighbor in neighbors[node]:
            if neighbor not in depth:
                depth[neighbor] = depth[node] + 1
                pending.append(neighbor)
    return set(depth)


def scene_geometry(scene):
    """Infer native spacing from exported native/mm pairs, without CT access."""
    nodes = scene['nodes']
    require(bool(nodes), 'Each exported native branch must contain its query')
    low, high = map(vector, scene['box'])
    require(all(a < b for a, b in zip(low, high)), 'Positive unchanged crop box required')
    ratios = [[] for _ in range(3)]
    for node in nodes:
        native, physical = vector(node['native_voxel']), vector(node['xyz'])
        require(all(value == round(value) and value >= 0 for value in native),
                'Exported native anchors/nodes must remain nonnegative integers')
        for axis, (value, mm) in enumerate(zip(native, physical)):
            if value != 0:
                ratios[axis].append(mm / value)
            else:
                require(abs(mm) <= TOLERANCE_MM, 'Native zero disagrees with exported absolute mm')
    require(all(values for values in ratios),
            'Export does not contain enough native/mm information to verify all three spacing axes')
    coarse_spacing = [statistics.median(values) for values in ratios]
    require(all(math.isfinite(value) and value > 0 for value in coarse_spacing), 'Positive native spacing required')
    # Node mm exports are FP32 whereas native-header crop bounds are FP64.
    # First identify the integer extent, then recover the header spacing from
    # those unchanged bounds. Never widen the 1e-5 mm geometric tolerance.
    shape = [round((b - a) / scale) for a, b, scale in zip(low, high, coarse_spacing)]
    require(all(size > 0 for size in shape), 'Positive native crop shape required')
    spacing = [(b - a) / size for a, b, size in zip(low, high, shape)]
    origin = [round(a / scale + .5) for a, scale in zip(low, spacing)]
    require(same_vector(low, [(value - .5) * scale for value, scale in zip(origin, spacing)])
            and same_vector(high, [(a + size - .5) * scale for a, size, scale in zip(origin, shape, spacing)]),
            'Crop bounds cannot be reconciled with original integer native origin/shape')
    quantization = []
    for node in nodes:
        native, physical = vector(node['native_voxel']), vector(node['xyz'])
        original = native_mm(node, spacing)
        require(same_vector(physical, [fp32(value) for value in original]),
                'Native/mm node coordinates disagree with their explicit FP32 export')
        quantization.extend(abs(a - b) for a, b in zip(physical, original))
        require(all(a - TOLERANCE_MM <= value <= b + TOLERANCE_MM
                    for a, value, b in zip(low, physical, high)), 'Node outside unchanged native crop bounds')
    return dict(spacing=spacing, low=low, high=high, origin=origin, shape=shape,
        model_coordinate_quantization_max_mm=max(quantization),
        model_mm_contract='FP32 serialization of original integer native coordinate times restored header spacing',
        path_mm_contract='FP64 original native coordinate times restored header spacing; fixed1e-5mm comparisons')


def validate_polyline(path, source, target, geometry):
    require(isinstance(path, list) and len(path) >= 2, 'Mandatory path must expose its real native parent polyline')
    points = [vector(point) for point in path]
    spacing = geometry['spacing']
    source_original, target_original = native_mm(source, spacing), native_mm(target, spacing)
    require(same_vector(points[0], source_original) and same_vector(points[-1], target_original)
            and same_vector(source['xyz'], [fp32(value) for value in points[0]])
            and same_vector(target['xyz'], [fp32(value) for value in points[-1]]),
            'Mandatory path source/target direction disagrees with actual exported nodes')
    for point in points:
        require(all(a - TOLERANCE_MM <= value <= b + TOLERANCE_MM
                    for a, value, b in zip(geometry['low'], point, geometry['high'])),
                'Mandatory path leaves the original native crop bounds')
        require(all(abs(value - round(value / scale) * scale) <= TOLERANCE_MM
                    for value, scale in zip(point, spacing)), 'Mandatory parent point is not an original native voxel')
    lengths = []
    for first, last in zip(points, points[1:]):
        delta = [b - a for a, b in zip(first, last)]
        possible = [axis for axis in range(3)
                    if abs(abs(delta[axis]) - spacing[axis]) <= TOLERANCE_MM
                    and all(abs(delta[other]) <= TOLERANCE_MM for other in range(3) if other != axis)]
        require(len(possible) == 1, 'Mandatory parent path skips a voxel or uses a diagonal/nonaxis step')
        length = math.sqrt(sum(value * value for value in delta))
        require(0 < length <= 6. + TOLERANCE_MM, 'Native parent hop violates the unchanged physical radius')
        lengths.append(length)
    length = math.fsum(lengths)
    require(length <= 6. + TOLERANCE_MM, 'Mandatory path exceeds cumulative 6 mm despite a nearby endpoint')
    require(math.dist(points[0], points[-1]) <= length + TOLERANCE_MM,
            'Mandatory path length is shorter than the physical endpoint distance')
    return dict(native_steps=len(lengths), cumulative_mm=length)


def report_scene(arm, record_id):
    matches = [scene for scene in arm['scenes'] if scene['id'] == record_id]
    require(len(matches) == 1, 'Actual export must bind exactly one CUDA cohort record: ' + record_id)
    return matches[0]


def audit_record(candidate, arm, *, relayed):
    graph, nodes = candidate['joint'], candidate['joint']['nodes']
    require(graph['edge_orientation'] == 'source,target,relation_name' and nodes,
            'Actual nonempty directed typed joint graph required')
    scene = report_scene(arm, candidate['id'])
    require(candidate['kind'] == scene['kind'], 'Original P/U observation changed')
    branch = dict(donor=candidate['donor'], recipient=candidate['scene'])
    geometry = {side: scene_geometry(value) for side, value in branch.items()}
    node_keys = []
    for index, node in enumerate(nodes):
        require(node['side'] in SIDES and node['role'] in ROLES, 'Unknown actual side/role')
        require(type(node['model_slot']) is int and node['model_slot'] >= 0, 'Original model slot required')
        require(type(node['local_index']) is int and 0 <= node['local_index'] < len(branch[node['side']]['nodes']),
                'Joint export local index outside native branch')
        original = branch[node['side']]['nodes'][node['local_index']]
        require(original['model_slot'] == node['model_slot'] and original['role'] == node['role']
                and original['retention'] == node['retention'], 'Joint/branch node role or retention disagrees')
        for name in ('xyz', 'native_voxel', 'relative_xyz_mm'):
            require(same_vector(node[name], original[name]), 'Joint/branch coordinate mismatch: ' + name)
        require(node['retention'] in ('query', 'seed', 'relay') and
                ((node['role'] == 'query') == (node['retention'] == 'query')),
                'Explicit query/seed/relay interpretation required')
        node_keys.append((node['side'], node['model_slot']))
    require(len(set(node_keys)) == len(nodes), 'Duplicate original physical branch/model slot')
    queries = {side: [index for index, node in enumerate(nodes)
                     if node['side'] == side and node['role'] == 'query'] for side in SIDES}
    require(all(len(indices) == 1 for indices in queries.values()), 'Exactly one original abstract query per branch required')
    query_ids = {indices[0] for indices in queries.values()}
    for side, indices in queries.items():
        query = nodes[indices[0]]
        require(query['abstract'] is True and same_vector(query['relative_xyz_mm'], [0., 0., 0.])
                and same_vector(query['xyz'], branch[side]['center']), 'Original abstract anchor shifted')
        for node in (node for node in nodes if node['side'] == side):
            require(same_vector(node['relative_xyz_mm'], [fp32((a - b) * scale)
                for a, b, scale in zip(vector(node['native_voxel']), vector(query['native_voxel']), geometry[side]['spacing'])]),
                    'Anchor-relative coordinate changed')
            radius = math.sqrt(sum(value * value for value in vector(node['relative_xyz_mm'])))
            shell_ok = (node['role'] == 'query' or
                        node['role'] == 'near' and radius <= 5. + TOLERANCE_MM or
                        node['role'] == 'mid' and 5. - TOLERANCE_MM <= radius <= 10. + TOLERANCE_MM or
                        node['role'] == 'wide' and radius >= 10. - TOLERANCE_MM)
            require(shell_ok, 'Actual relay/context shell disagrees with original anchor-relative radius')
    outgoing, incoming = [set() for _ in nodes], [set() for _ in nodes]
    edges, paths = graph['edges'], graph['edge_path_mm']
    require(len(edges) == len(paths) == len(graph['edge_distance_mm']), 'Edge/path/physical length arrays are misaligned')
    typed, counts, witnesses = set(), Counter(), []
    for index, edge in enumerate(edges):
        require(isinstance(edge, list) and len(edge) == 3, 'Three-field actual directed typed edge required')
        source, target, relation = edge
        require(type(source) is int and type(target) is int and 0 <= source < len(nodes)
                and 0 <= target < len(nodes) and source != target and relation in RELATIONS,
                'Invalid actual edge endpoint or typed relation')
        require(tuple(edge) not in typed, 'Duplicate actual typed edge')
        typed.add(tuple(edge))
        sn, tn = nodes[source], nodes[target]
        rule = RULES[RELATIONS.index(relation)]
        require((sn['side'], sn['role'] == 'query', tn['side'], tn['role'] == 'query') == rule,
                'Typed relation connects the wrong physical branch/node role')
        length = math.dist(vector(sn['relative_xyz_mm']), vector(tn['relative_xyz_mm']))
        require(length <= RADII[RELATIONS.index(relation)] + TOLERANCE_MM
                and abs(length - graph['edge_distance_mm'][index]) <= TOLERANCE_MM,
                'Actual typed edge physical radius or recorded distance disagrees')
        outgoing[source].add(target)
        incoming[target].add(source)
        counts[relation] += 1
        if paths[index] is not None:
            require(relayed and relation in RELATIONS[:2], 'Only retained context-parent edges may claim a native path')
            witnesses.append(validate_polyline(paths[index], sn, tn, geometry[sn['side']]))
    # Every witnessed directed parent edge must have an exact reverse witness.
    lookup = {tuple(edge): index for index, edge in enumerate(edges)}
    for index, path in enumerate(paths):
        if path is None:
            continue
        source, target, relation = edges[index]
        reverse = lookup.get((target, source, relation))
        require(reverse is not None and paths[reverse] is not None
                and len(paths[reverse]) == len(path)
                and all(same_vector(a, b) for a, b in zip(reversed(path), paths[reverse])),
                'Mandatory native parent edge lost its reciprocal path witness')
    components = weak_components(outgoing, incoming)
    isolates = {index for index in range(len(nodes)) if not outgoing[index] and not incoming[index]}
    outside = set().union(*(part for part in components if part.isdisjoint(query_ids)))
    require(len(components) == scene['components'] == graph['stats']['components'],
            'Independent weak component count differs from saved CUDA count')
    require(len(isolates) == scene['isolated_nodes'], 'Independent isolate count differs from saved CUDA count')
    require(len(nodes) == scene['pair_node_count'] == graph['stats']['node_count'], 'Actual node count disagrees with CUDA cohort record')
    require(len(edges) == scene['donor_directed_edges'] + scene['recipient_directed_edges'] + scene['cross_edges']
            == graph['stats']['directed_typed_edges'], 'Actual typed edge count disagrees with CUDA record')
    require(all(counts[name] == scene['relation_counts'][name] for name in RELATIONS), 'Actual relation counts disagree with CUDA cohort record')
    retained = {}
    for side_index, side in enumerate(SIDES):
        side_nodes = [node for node in nodes if node['side'] == side]
        values = Counter(node['retention'] for node in side_nodes)
        require(values['query'] == 1 and values['seed'] == scene['seed_context_counts'][side_index]
                and values['relay'] == scene['relay_counts'][side_index], 'Seed/relay counts disagree with saved CUDA record')
        require(len(side_nodes) == scene[side + '_node_count'], 'Actual branch node count changed')
        retained[side] = dict(query=1, seeds=values['seed'], relays=values['relay'],
                              total_nodes=len(side_nodes), roles=dict(Counter(node['role'] for node in side_nodes)),
                              native6_substrate_unreachable_seeds_GPU_reported=scene['native6_substrate_unreachable_counts'][side_index],
                              original_native_shape=geometry[side]['shape'],
                              model_coordinate_FP32_quantization_max_mm=geometry[side]['model_coordinate_quantization_max_mm'])
    paths_summary = {}
    all_ids = set(range(len(nodes)))
    for side, indices in queries.items():
        query = indices[0]
        ancestors, ancestors3 = reachable(query, incoming), reachable(query, incoming, 3)
        influence, influence3 = reachable(query, outgoing), reachable(query, outgoing, 3)
        paths_summary[side + '_query'] = dict(ancestors_all_count=len(ancestors), ancestors_3_layers_count=len(ancestors3),
            ancestors_3_layers_by_role=role_counts(nodes, ancestors3),
            cannot_send_to_query_3_layers_by_role=role_counts(nodes, all_ids - ancestors3),
            influence_all_count=len(influence), influence_3_layers_count=len(influence3),
            reached_from_query_3_layers_by_role=role_counts(nodes, influence3))
    return dict(id=candidate['id'], kind=candidate['kind'], node_count=len(nodes), directed_typed_edges=len(edges),
        weak_components=len(components), weak_component_sizes=sorted(map(len, components), reverse=True),
        isolated_count=len(isolates), isolated_by_role=role_counts(nodes, isolates),
        outside_both_query_components_count=len(outside), outside_both_query_components_by_role=role_counts(nodes, outside),
        retained_by_branch=retained, query_paths=paths_summary, mandatory_native_paths=len(witnesses),
        mandatory_native_steps=distribution([value['native_steps'] for value in witnesses]),
        mandatory_cumulative_mm=distribution([value['cumulative_mm'] for value in witnesses]),
        verified=dict(CUDA_component_count=True, CUDA_isolate_count=True, CUDA_component_partition='NOT_EXPORTED',
            exact_branch_joint_mapping=True, native_parent_coordinates_steps_radius_bbox=True,
            organ_support_independently_rechecked=False))


def validate_seed_preservation(old, new):
    require(old['id'] == new['id'] and old['kind'] == new['kind'], 'Original candidate/observation identity changed')
    old_nodes, new_nodes = old['joint']['nodes'], new['joint']['nodes']
    def key(node):
        return node['side'], node['role'], tuple(vector(node['native_voxel']))
    lookup = {}
    for node in new_nodes:
        require(key(node) not in lookup, 'Duplicate native context/query representation')
        lookup[key(node)] = node
    for node in old_nodes:
        require(node['retention'] in ('query', 'seed') and key(node) in lookup, 'An original selected seed/query was lost')
        other = lookup[key(node)]
        require(other['retention'] == node['retention'], 'An original seed/query was relabeled as a relay')
        for name in ('xyz', 'relative_xyz_mm'):
            require(same_vector(node[name], other[name]), 'Original seed/query physical coordinate changed')
    require(sum(node['retention'] == 'seed' for node in new_nodes) ==
            sum(node['retention'] == 'seed' for node in old_nodes), 'A new relay was falsely counted as an original seed')
    for label in ('donor', 'scene'):
        require(old[label]['case'] == new[label]['case']
                and same_vector(old[label]['center'], new[label]['center'])
                and all(same_vector(a, b) for a, b in zip(old[label]['box'], new[label]['box'])),
                'Original branch case, anchor or crop shape changed')
    return dict(original_context_seeds_preserved=sum(node['retention'] == 'seed' for node in old_nodes),
                original_abstract_queries_preserved=2, coordinates_roles_native_shape_preserved=True)


def summarize(records, arm):
    cohort = arm['scenes']
    require(len(cohort) == 133 and len({scene['id'] for scene in cohort}) == 133
            and Counter(scene['kind'] for scene in cohort) == Counter(P=5, U=128), 'Original all133 CUDA cohort coverage changed')
    result = dict(independently_audited_displayed=len(records), displayed_kind_counts=dict(Counter(record['kind'] for record in records)),
        independent_displayed={name: distribution([record[name] for record in records]) for name in
            ('node_count', 'directed_typed_edges', 'weak_components', 'isolated_count', 'outside_both_query_components_count', 'mandatory_native_paths')},
        all133_GPU_reported_only=dict(records=133, independent_edge_BFS=False,
            weak_components=distribution([scene['components'] for scene in cohort]),
            isolates=distribution([scene['isolated_nodes'] for scene in cohort]),
            node_counts=distribution([scene['pair_node_count'] for scene in cohort]),
            seed_counts=distribution([sum(scene['seed_context_counts']) for scene in cohort]),
            relay_counts=distribution([sum(scene['relay_counts']) for scene in cohort]),
            native6_substrate_unreachable_seed_counts=(distribution([sum(scene['native6_substrate_unreachable_counts']) for scene in cohort])
                if all(all(value is not None for value in scene['native6_substrate_unreachable_counts']) for scene in cohort) else 'NOT_APPLICABLE_PREVIOUS_SAMPLER')))
    result['directed_query_paths'] = {}
    for query in ('donor_query', 'recipient_query'):
        result['directed_query_paths'][query] = {name: distribution([record['query_paths'][query][name] for record in records])
            for name in ('ancestors_all_count', 'ancestors_3_layers_count', 'influence_all_count', 'influence_3_layers_count')}
        for name in ('ancestors_3_layers_by_role', 'cannot_send_to_query_3_layers_by_role', 'reached_from_query_3_layers_by_role'):
            result['directed_query_paths'][query][name] = {role: distribution([record['query_paths'][query][name][role] for record in records]) for role in GLOBAL_ROLES}
    return result


def run(payload_path, report_path, output_path):
    require(not output_path.exists(), 'Existing audit evidence is preserved; select an explicitly new --output path')
    payload_path, report_path = payload_path.resolve(), report_path.resolve()
    inputs = dict(payload=sha256(payload_path), report=sha256(report_path))
    auditor_hash = sha256(Path(__file__).resolve())
    payload, report = (json.loads(path.read_text(encoding='utf-8')) for path in (payload_path, report_path))
    require(report['debug'] is True and report['actual_CT'] is True and report['actual_CUDA'] is True
            and report['production_training_started'] is False and report['checkpoint_written'] is False
            and payload['case'] == report['case'], 'Matched actual-CT/CUDA DEBUG report/export required')
    require(report['physical_batch'] == 32 and report['actual_case_observations_used'] == 133
            and report['original_P'] == 5 and report['original_U'] == 128, 'Original paired DEBUG scope changed')
    source_hashes = {}
    for name, expected in report['implementation_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Implementation path outside the repository')
        source_hashes[name] = sha256(ROOT / relative)
        require(source_hashes[name] == expected, 'Current numerical implementation differs from the executed source: ' + name)
    require([budget['context_nodes'] for budget in payload['budgets']] == [48, 96, 192], 'All three explicit seed quotas required')
    budgets = {}
    for budget in payload['budgets']:
        context, quota = budget['context_nodes'], budget['quota']
        require(context == 3 * quota, 'Explicit seed quota differs')
        arms, baseline = {}, {}
        for variant in ('baseline', 'relational'):
            arm = report['budgets'][str(context)][variant]
            candidates = budget['variants'][variant]['candidates']
            require(len(candidates) == len({candidate['id'] for candidate in candidates}) == 8
                    and Counter(candidate['kind'] for candidate in candidates) == Counter(P=5, U=3), 'All displayed original P5/U3 graphs required')
            records = []
            for candidate in candidates:
                record = audit_record(candidate, arm, relayed=variant == 'relational')
                if variant == 'baseline':
                    baseline[candidate['id']] = candidate
                else:
                    require(candidate['id'] in baseline, 'Previous sampler control record missing')
                    record['seed_preservation'] = validate_seed_preservation(baseline[candidate['id']], candidate)
                records.append(record)
            arms[variant] = dict(summary=summarize(records, arm), records=records)
        require([scene['id'] for scene in report['budgets'][str(context)]['baseline']['scenes']] ==
                [scene['id'] for scene in report['budgets'][str(context)]['relational']['scenes']], 'Original cohort IDs/order changed across samplers')
        budgets[str(context)] = dict(quota=quota, variants=arms)
    require(inputs == dict(payload=sha256(payload_path), report=sha256(report_path)), 'Saved CUDA/JSON evidence changed during independent audit')
    require(auditor_hash == sha256(Path(__file__).resolve()) and
            source_hashes == {name: sha256(ROOT / name) for name in source_hashes}, 'Audit/numerical sources changed during read')
    evidence = dict(format='independent_native_relay_connectivity_DEBUG_v1', debug=True,
        created_utc=datetime.now(timezone.utc).isoformat(), case=report['case'], directed_hops=3,
        numerical_tolerance_mm=TOLERANCE_MM, scope='JSON-only CPU graph BFS and f64 parent-polyline audit; no CPU model or CT pixels',
        coordinate_precision=dict(node_model_mm='FP32 original native/header-spacing product; serialized FP32 value checked',
            native_parent_path_mm='FP64 native/header-spacing product; 1e-5mm geometry tolerance unchanged',
            spacing_reconstruction='FP32 node/native median locates integer native extent; exact FP64 crop extent/shape restores header spacing',
            quantization_difference_included_in_branch_records=True, tolerance_automatically_changed=False),
        executed_model=False, executed_GPU=False, repaired_graph=False, CT_or_mask_pixels_accessed=False,
        production_ready=False, independent_displayed_graphs=48, independent_all133_graph_BFS=False,
        all_displayed_component_isolate_counts_match=True, all_displayed_original_seeds_queries_preserved=True,
        native_parent_paths_numeric_verified=True, mask_support_rechecked_independently=False,
        sources=dict(payload=dict(path=str(payload_path), sha256=inputs['payload']), report=dict(path=str(report_path), sha256=inputs['report'])),
        implementation=dict(path=str(Path(__file__).resolve()), sha256=auditor_hash, executed_numerical_source_sha256=source_hashes),
        evidence_and_sources_unchanged_during_read=True, budgets=budgets,
        interpretation=['Weak connectivity does not establish three-layer directed influence into a query.',
            'Native-axis nonreachability is not proof of full 6mm radius-graph disconnection.',
            'Role/shell readout includes each valid hidden node, independently of directed query reach.',
            'Only nonnull mandatory parent polylines claim an organ-supported native route; other typed edges are message relations.',
            'Parent mask support was checked by the CUDA path validator/exporter; this independent JSON reader verifies coordinates, steps, bounds and length only.',
            'All133 summaries are reported CUDA metrics; independent BFS covers the exported 48 displayed graphs.'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output_path.resolve()), independent_displayed_graphs=48,
        component_isolate_counts_match=True, original_seeds_queries_preserved=True,
        mandatory_parent_polylines_verified=True,
        new_all133_GPU_metrics={name: budget['variants']['relational']['summary']['all133_GPU_reported_only']
                               for name, budget in budgets.items()}), separators=(',', ':')))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('payload', 'report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.payload, args.report, args.output)


if __name__ == '__main__':
    main()
