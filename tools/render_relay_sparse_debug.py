"""Build a fresh actual-CT view of original seeds versus retained path relays.

Only completed DEBUG exports are read. Original files/results are preserved,
every old seed and native coordinate is checked, and no model runs here.
"""
import argparse
import base64
import copy
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.render_relational_sparse_debug import (
    BUDGETS, VARIANTS, RELATIONS, require, vector, near, connectivity,
)


def validate_scene(scene, quota, label, *, relayed):
    require(scene.get('original_native_coordinates') is True
            and scene.get('coordinate_rounding_applied') is False,
            label + ': original native precision required')
    center = vector(scene['center'], label + '.center')
    low, high = [vector(v, label + '.box') for v in scene['box']]
    require(all(a < b for a, b in zip(low, high)), label + ': invalid original crop')
    nodes = scene['nodes']
    require(len(nodes) >= 1 + 3 * quota, label + ': original seeds are missing')
    require(nodes[0]['role'] == 'query' and nodes[0]['abstract'] is True
            and near(nodes[0]['xyz'], center), label + ': original abstract query moved')
    seeds = [node for node in nodes if node['role'] != 'query'
             and node.get('retention', 'seed') == 'seed']
    require(Counter(node['role'] for node in seeds) == {'near': quota, 'mid': quota, 'wide': quota},
            label + ': original FPS seed role/quota changed')
    require(relayed or len(nodes) == 1 + 3 * quota,
            label + ': baseline has fabricated relay nodes')
    spacing = [None] * 3
    identities = set()
    for index, node in enumerate(nodes):
        xyz = vector(node['xyz'], label + f'.node{index}.xyz')
        native = vector(node['native_voxel'], label + f'.node{index}.native')
        relative = vector(node['relative_xyz_mm'], label + f'.node{index}.relative')
        retention = node.get('retention', 'query' if node['abstract'] else 'seed')
        require(retention in ('query', 'seed', 'relay'), label + ': unknown retention type')
        require(node['role'] in ('query', 'near', 'mid', 'wide')
                and node['abstract'] == (node['role'] == 'query')
                and (retention == 'query') == node['abstract'], label + ': role/query mismatch')
        require(all(value == round(value) for value in native), label + ': noninteger native voxel')
        require(near(relative, [x - c for x, c in zip(xyz, center)]),
                label + ': actual and relative mm disagree')
        radius = math.sqrt(sum(value * value for value in relative))
        if node['role'] == 'near':
            require(radius <= 5.0001, label + ': near role outside original5mm band')
        elif node['role'] == 'mid':
            require(4.9999 < radius <= 10.0001, label + ': mid role outside original5–10mm band')
        elif node['role'] == 'wide':
            require(radius > 9.9999, label + ': wide role inside original10mm band')
        require(all(a - .001 <= p <= b + .001 for a, p, b in zip(low, xyz, high)),
                label + ': actual node outside original crop')
        identity = tuple(native) + (node['role'],)
        require(identity not in identities, label + ': duplicated native context node')
        identities.add(identity)
        for axis in range(3):
            if native[axis]:
                value = xyz[axis] / native[axis]
                require(value > 0, label + ': nonpositive spacing')
                if spacing[axis] is None:
                    spacing[axis] = value
                else:
                    require(abs(xyz[axis] - native[axis] * spacing[axis]) <= .002,
                            label + ': native mm spacing changed')
    require(all(value is not None for value in spacing), label + ': missing native spacing')
    require(scene['stats']['node_count'] == len(nodes)
            and scene['stats']['directed_edges'] == len(scene['edges']), label + ': wrong scene counts')
    return spacing


def validate_joint(candidate, variant, quota, label, report_scene):
    donor, recipient, joint = candidate['donor'], candidate['scene'], candidate['joint']
    spacings = {side: validate_scene(scene, quota, label + '.' + side, relayed=variant == 'relational')
                for side, scene in (('donor', donor), ('recipient', recipient))}
    require(joint['edge_orientation'] == 'source,target,relation_name'
            and joint['display_geometry_generated'] is False
            and joint['anatomical_path_claim'] is False, label + ': explicit message geometry required')
    expected = [dict(node, side=side, case=scene['case'], local_index=index)
                for side, scene in (('donor', donor), ('recipient', recipient))
                for index, node in enumerate(scene['nodes'])]
    require(joint['nodes'] == expected, label + ': joint nodes do not match actual scene export')
    nodes, edges = joint['nodes'], joint['edges']
    require(len(edges) == len(joint['edge_distance_mm']), label + ': edge distance coverage incomplete')
    paths = joint.get('edge_path_mm', [None] * len(edges))
    require(len(paths) == len(edges), label + ': path witness coverage incomplete')
    if any(path is not None for path in paths):
        require(joint.get('native_parent_paths_verified') is True,
                label + ': probe has not verified every native parent-path voxel support')
    scene_edges = {'donor': [], 'recipient': []}
    seen, relation_counts, cross = set(), Counter(), 0
    for index, ((source, target, relation), distance, path) in enumerate(zip(edges, joint['edge_distance_mm'], paths)):
        require(type(source) is int and type(target) is int
                and 0 <= source < len(nodes) and 0 <= target < len(nodes) and source != target,
                label + ': invalid real edge index')
        require((source, target, relation) not in seen, label + ': duplicated typed edge')
        seen.add((source, target, relation))
        require(relation in RELATIONS, label + ': unknown actual relation')
        a, b = nodes[source], nodes[target]
        role_gate = (a['side'], a['role'] == 'query', b['side'], b['role'] == 'query')
        require(role_gate == RELATIONS[relation][:4], label + ': typed endpoint gate violated')
        actual = math.dist(a['relative_xyz_mm'], b['relative_xyz_mm'])
        require(type(distance) in (int, float) and math.isfinite(distance)
                and abs(distance - actual) <= .001 and distance <= RELATIONS[relation][4] + .00001,
                label + ': physical relation radius/distance violated')
        if path is not None:
            require(variant == 'relational' and relation in ('source_context_neighbor', 'target_context_neighbor')
                    and a['side'] == b['side'], label + ': parent witness on a non-context/cross edge')
            require(isinstance(path, list) and len(path) >= 2, label + ': empty actual path witness')
            points = [vector(point, label + f'.edge{index}.path') for point in path]
            require(near(points[0], a['xyz']) and near(points[-1], b['xyz']), label + ': parent path endpoints changed')
            scene = donor if a['side'] == 'donor' else recipient
            spacing = spacings[a['side']]
            total = 0
            for point in points:
                require(all(lo - .002 <= p <= hi + .002 for lo, p, hi in zip(scene['box'][0], point, scene['box'][1])),
                        label + ': parent-path witness outside bound original crop')
            for left, right in zip(points, points[1:]):
                delta = [(v - u) / scale for u, v, scale in zip(left, right, spacing)]
                require(sum(abs(value) > .002 for value in delta) == 1
                        and abs(sum(abs(value) for value in delta) - 1) <= .003,
                        label + ': parent path did not retain actual native-axis unit steps')
                total += math.dist(left, right)
            require(total <= 6.0001, label + ': native cumulative path exceeded existing6mm radius')
        relation_counts[relation] += 1
        if a['side'] != b['side']:
            cross += 1
        else:
            scene_edges[a['side']].append([a['local_index'], b['local_index'], relation])
    for side, scene in (('donor', donor), ('recipient', recipient)):
        require(scene['edges'] == scene_edges[side], label + ': scene edges differ from joint typed edges')
    components, isolated = connectivity(len(nodes), edges)
    require(joint['stats']['node_count'] == len(nodes)
            and joint['stats']['directed_typed_edges'] == len(edges)
            and joint['stats']['cross_edges'] == cross
            and joint['stats']['relation_counts'] == dict(relation_counts)
            and joint['stats']['components'] == components == report_scene['components'],
            label + ': exact displayed connectivity/statistics disagree')
    joint['stats']['isolated_nodes'] = isolated
    require(report_scene['pair_node_count'] == len(nodes) and report_scene['cross_edges'] == cross
            and report_scene['relation_counts'] == dict(relation_counts), label + ': all-case report binding mismatch')
    return isolated


def merge_evidence(report, payload):
    require(all(report.get(key) is True for key in ('debug', 'actual_CT', 'actual_CUDA', 'actual_data_used')),
            'Completed actual CT/CUDA DEBUG evidence required')
    require(all(report.get(key) is False for key in ('full_training', 'full_evaluation', 'production_default_changed', 'production_ready')),
            'Actual diagnostic-only scope required')
    require(report['case'] == payload['case']
            and (report['original_P'], report['original_U'], report['actual_case_observations_used']) == (5, 128, 133),
            'Original P5/U128 whole-case identity changed')
    require([b['context_nodes'] for b in payload['budgets']] == list(BUDGETS), 'Original three seed quotas required')
    merged = copy.deepcopy(payload)
    weight = merged['weight_source']
    merged['weight_source'] = weight['label'] if isinstance(weight, dict) else weight
    require(isinstance(merged['weight_source'], str) and merged['weight_source'], 'Readable weight provenance required')
    merged['evidence'] = {key: report[key] for key in ('original_P', 'original_U', 'actual_case_observations_used',
        'physical_batch', 'effective_batch', 'measured_continuous_updates_per_branch', 'full_training', 'full_evaluation')}
    displayed_identity = None
    for budget in merged['budgets']:
        context, quota = budget['context_nodes'], budget['quota']
        require(context == 3 * quota, 'Original context seed quota changed')
        for variant in VARIANTS:
            evidence = report['budgets'][str(context)][variant]
            scenes = evidence['scenes']
            require(len(scenes) == 133 and Counter(s['kind'] for s in scenes) == {'P': 5, 'U': 128}, 'Whole-case record coverage missing')
            ids = [s['id'] for s in scenes]
            require(len(set(ids)) == 133, 'Duplicate all-case record')
            isolated_by_index = {}
            for audit in evidence['graph_audits']:
                indices, stats = audit['original_indices'], audit['statistics']
                require(len(indices) == len(stats['pair_isolated_node_counts']) == len(stats['pair_components']), 'Connectivity audit incomplete')
                for index, isolated, components in zip(indices, stats['pair_isolated_node_counts'], stats['pair_components']):
                    require(type(index) is int and 0 <= index < 133 and index not in isolated_by_index,
                            'Duplicated or invalid connectivity audit index')
                    require(components == scenes[index]['components'] and 0 <= isolated < scenes[index]['pair_node_count'],
                            'Connectivity audit/report disagreement')
                    isolated_by_index[index] = isolated
            require(len(isolated_by_index) == 133, 'Every original pair requires an actual isolate audit')
            totals = [sum(s['relation_counts'].values()) for s in scenes]
            ns = [s['pair_node_count'] for s in scenes]
            require(all(type(n) is int and n >= 2 * (context + 1) for n in ns), 'Original seeds missing from neural graph')
            budget['metrics'][variant].update(pair_nodes_min=min(ns), pair_nodes_max=max(ns),
                pair_seed_nodes=2 * (context + 1), relays_min=min(ns) - 2 * (context + 1),
                relays_max=max(ns) - 2 * (context + 1), edges_min=min(totals), edges_max=max(totals),
                zero_cross_pairs=sum(s['cross_edges'] == 0 for s in scenes),
                components_min=min(s['components'] for s in scenes), components_max=max(s['components'] for s in scenes),
                isolated_min=min(isolated_by_index.values()), isolated_max=max(isolated_by_index.values()),
                update_seconds=evidence['update_summary']['full_update_seconds']['median'],
                peak_GiB=evidence['update_summary']['peak_allocated_bytes']['max'] / 2**30)
            candidates = budget['variants'][variant]['candidates']
            require(len(candidates) == 8 and Counter(c['kind'] for c in candidates) == {'P': 5, 'U': 3}, 'Explicit displayedP5/U3 required')
            identity = [(c['id'], c['kind']) for c in candidates]
            if displayed_identity is None:
                displayed_identity = identity
            require(identity == displayed_identity, 'Display candidate changed between quota/model branches')
            for index, candidate in enumerate(candidates):
                original_index = ids.index(candidate['id'])
                require(candidate['kind'] == scenes[original_index]['kind'], 'Original observation kind changed')
                isolated = validate_joint(candidate, variant, quota, f'{context}/{variant}/{index}', scenes[original_index])
                require(isolated == isolated_by_index[original_index], 'Displayed isolates disagree with complete GPU audit')
                require(all(scene['case'] in merged['anatomy'] for scene in (candidate['donor'], candidate['scene'])), 'Actual GT geometry missing')
        for index in range(8):
            for side in ('donor', 'scene'):
                before = budget['variants']['baseline']['candidates'][index][side]
                after = budget['variants']['relational']['candidates'][index][side]
                require(all(before[field] == after[field] for field in ('case', 'center', 'box')), 'Original branch crop/anchor changed')
                fields = ('xyz', 'native_voxel', 'relative_xyz_mm', 'role', 'abstract')
                old = {tuple(tuple(n[field]) if isinstance(n[field], list) else n[field] for field in fields)
                       for n in before['nodes']}
                original = {tuple(tuple(n[field]) if isinstance(n[field], list) else n[field] for field in fields)
                            for n in after['nodes'] if n['abstract'] or n.get('retention') == 'seed'}
                require(old == original, 'Every original FPS seed/query must survive unchanged; no substituted center/role/coordinate')
    for annotation in merged['anatomy'].values():
        for field in ('organ_surface', 'tumor_surface'):
            for contour in annotation[field]:
                for point in contour:
                    vector(point, 'Original annotation mm')
    return merged


def compact_payload(merged):
    compact = {key: value for key, value in merged.items() if key != 'budgets'}
    compact.update(format='vrs-relay-debug-compact-v2', node_pool=[], relation_names=[], budgets=[],
                   path_point_pool=[], path_pool=[])
    node_lookup, relation_lookup, point_lookup, path_lookup = {}, {}, {}, {}
    fields = ('xyz', 'native_voxel', 'relative_xyz_mm', 'role', 'abstract', 'case')
    for budget in merged['budgets']:
        packed = dict(context_nodes=budget['context_nodes'], quota=budget['quota'], metrics=budget['metrics'], variants={})
        compact['budgets'].append(packed)
        for variant in VARIANTS:
            candidates = []
            packed['variants'][variant] = dict(candidates=candidates)
            for candidate in budget['variants'][variant]['candidates']:
                joint, indices, edges = candidate['joint'], [], []
                for node in joint['nodes']:
                    geometry = {key: node[key] for key in fields}
                    geometry['retention'] = node.get('retention', 'query' if node['abstract'] else 'seed')
                    key = json.dumps(geometry, separators=(',', ':'), allow_nan=False)
                    if key not in node_lookup:
                        node_lookup[key] = len(compact['node_pool'])
                        compact['node_pool'].append(geometry)
                    indices.append(node_lookup[key])
                for source, target, relation in joint['edges']:
                    if relation not in relation_lookup:
                        relation_lookup[relation] = len(compact['relation_names'])
                        compact['relation_names'].append(relation)
                    edges.append([source, target, relation_lookup[relation]])
                path_refs = []
                for path in joint.get('edge_path_mm', [None] * len(edges)):
                    if path is None:
                        path_refs.append(None)
                        continue
                    point_ids = []
                    for point in path:
                        point_key = json.dumps(point, separators=(',', ':'), allow_nan=False)
                        if point_key not in point_lookup:
                            point_lookup[point_key] = len(compact['path_point_pool'])
                            compact['path_point_pool'].append(point)
                        point_ids.append(point_lookup[point_key])
                    forward, reverse = tuple(point_ids), tuple(reversed(point_ids))
                    reversed_order = reverse < forward
                    canonical = reverse if reversed_order else forward
                    if canonical not in path_lookup:
                        path_lookup[canonical] = len(compact['path_pool'])
                        compact['path_pool'].append(list(canonical))
                    path_id = path_lookup[canonical]
                    reference = [path_id, reversed_order]
                    restored_ids = list(reversed(compact['path_pool'][path_id])) if reversed_order else compact['path_pool'][path_id]
                    restored = [compact['path_point_pool'][point_id] for point_id in restored_ids]
                    require(json.dumps(restored, separators=(',', ':'), allow_nan=False)
                            == json.dumps(path, separators=(',', ':'), allow_nan=False),
                            'Lossless actual parent-path compact roundtrip failed')
                    path_refs.append(reference)
                candidates.append(dict(id=candidate['id'], kind=candidate['kind'],
                    donor={key: candidate['donor'][key] for key in ('case', 'center', 'box')},
                    scene={key: candidate['scene'][key] for key in ('case', 'center', 'box')},
                    joint=dict(node_indices=indices, donor_node_count=len(candidate['donor']['nodes']), edges=edges,
                        edge_path_refs=path_refs, stats=joint['stats'],
                        edge_orientation=joint['edge_orientation'])))
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing result')
    report = json.loads((args.run / 'report.json').read_text(encoding='utf8'))
    payload = json.loads((args.run / 'visual_payload.json').read_text(encoding='utf8'))
    merged = merge_evidence(report, payload)
    compact = compact_payload(merged)
    source = json.dumps(compact, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')
    compressed = base64.b64encode(gzip.compress(source, compresslevel=9, mtime=0)).decode('ascii')
    template = (ROOT / 'tools/relay-sparse.template.html').read_text(encoding='utf8')
    require(template.count('__DATA__') == 1, 'One embedded payload placeholder required')
    fragment = template.replace('__DATA__', compressed)
    require(len(fragment.encode('utf8')) < 1_000_000, 'Exact actual graph exceeds inline1MB limit')
    require(not any(tag in fragment.lower() for tag in ('<!doctype', '<html', '<head', '<body')), 'Fragment contract violated')
    require(args.output.parent.is_dir(), 'Existing task-owned durable output directory required')
    with args.output.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(fragment)
    require(args.output.read_text(encoding='utf8') == fragment, 'Readback differs from actual written fragment')
    print(json.dumps(dict(output=str(args.output.resolve()), html_bytes=len(fragment.encode('utf8')),
        payload_bytes=len(source), all133=True, displayed_P=5, displayed_U=3, exact_seed_retention=True,
        actual_parent_paths=True, path_points=len(compact['path_point_pool']), unique_paths=len(compact['path_pool']),
        lossless_path_roundtrip=True, coordinate_rounding=False, full_training=False, full_evaluation=False)))


if __name__ == '__main__':
    main()
