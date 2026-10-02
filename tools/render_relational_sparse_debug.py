"""Build a fresh inline view from completed actual-CT relational DEBUG evidence.

Only the final JSON exports are read. This command runs no model, rounds no
native node coordinates, synthesizes no edges, and refuses to overwrite output.
"""

import argparse
import base64
import copy
import gzip
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (48, 96, 192)
VARIANTS = ("baseline", "relational")
ROLES = ("query", "near", "mid", "wide")
RELATIONS = {
    "source_context_neighbor": ("donor", False, "donor", False, 6.0),
    "target_context_neighbor": ("recipient", False, "recipient", False, 6.0),
    "source_query_to_context": ("donor", True, "donor", False, 8.0),
    "source_context_to_query": ("donor", False, "donor", True, 8.0),
    "target_query_to_context": ("recipient", True, "recipient", False, 8.0),
    "target_context_to_query": ("recipient", False, "recipient", True, 8.0),
    "source_context_to_target_context": ("donor", False, "recipient", False, 5.0),
    "source_query_to_target_context": ("donor", True, "recipient", False, 8.0),
}
BASELINE_RELATIONS = {
    "spatial", "feature", "spatial + feature", "connectivity",
    "spatial + connectivity", "feature + connectivity",
    "spatial + feature + connectivity",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, label):
    require(isinstance(value, list) and len(value) == 3, label + ": expected xyz vector")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value),
            label + ": nonfinite coordinate")
    return value


def near(a, b, tolerance=0.001):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def connectivity(count, edges):
    neighbors = [set() for _ in range(count)]
    for source, target, _ in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    unseen, components = set(range(count)), 0
    while unseen:
        pending = [unseen.pop()]
        components += 1
        while pending:
            found = neighbors[pending.pop()] & unseen
            unseen -= found
            pending.extend(found)
    return components, sum(not adjacent for adjacent in neighbors)


def validate_scene(scene, quota, label):
    require(scene.get("original_native_coordinates") is True
            and scene.get("coordinate_rounding_applied") is False,
            label + ": original full-precision coordinates required")
    center = vector(scene["center"], label + ".center")
    low, high = [vector(v, label + ".box") for v in scene["box"]]
    require(all(a < b for a, b in zip(low, high)), label + ": invalid crop box")
    nodes = scene["nodes"]
    require(len(nodes) == 1 + 3 * quota, label + ": missing original context nodes")
    counts = Counter(node["role"] for node in nodes)
    require(counts == {"query": 1, "near": quota, "mid": quota, "wide": quota},
            label + ": role/shell quota changed")
    require(nodes[0]["role"] == "query" and nodes[0]["abstract"] is True,
            label + ": query must be the abstract CNN mean")
    require(near(nodes[0]["xyz"], center), label + ": query anchor mismatch")
    spacing = [None, None, None]
    for index, node in enumerate(nodes):
        xyz = vector(node["xyz"], label + f".nodes[{index}].xyz")
        native = vector(node["native_voxel"], label + f".nodes[{index}].native_voxel")
        relative = vector(node["relative_xyz_mm"], label + f".nodes[{index}].relative_xyz_mm")
        require(node["abstract"] == (node["role"] == "query"),
                label + ": abstract query flag mismatch")
        require(near(relative, [x - c for x, c in zip(xyz, center)]),
                label + ": native mm and anchor-relative mm mismatch")
        require(all(a - 0.001 <= p <= b + 0.001 for a, p, b in zip(low, xyz, high)),
                label + ": actual node lies outside its original crop")
        for axis in range(3):
            if abs(native[axis]) > 1e-8:
                value = xyz[axis] / native[axis]
                require(value > 0, label + ": nonpositive native voxel/mm spacing")
                if spacing[axis] is None:
                    spacing[axis] = value
                else:
                    require(abs(xyz[axis] - native[axis] * spacing[axis]) <= 0.002,
                            label + ": inconsistent native voxel/mm coordinates")
    require(scene["stats"]["node_count"] == len(nodes)
            and scene["stats"]["directed_edges"] == len(scene["edges"]),
            label + ": scene statistics disagree with exact export")


def validate_joint(candidate, variant, quota, label, report_scene):
    donor, recipient, joint = candidate["donor"], candidate["scene"], candidate["joint"]
    for side, scene in (("donor", donor), ("recipient", recipient)):
        validate_scene(scene, quota, label + "." + side)
    require(joint["edge_orientation"] == "source,target,relation_name",
            label + ": source/target orientation is not explicit")
    require(joint["display_geometry_generated"] is False
            and joint["anatomical_path_claim"] is False,
            label + ": display geometry or anatomical path claim in model export")
    expected_nodes = [dict(node, side=side, case=scene["case"], local_index=index)
                      for side, scene in (("donor", donor), ("recipient", recipient))
                      for index, node in enumerate(scene["nodes"])]
    require(joint["nodes"] == expected_nodes, label + ": joint nodes differ from actual branch scenes")
    nodes, edges = joint["nodes"], joint["edges"]
    require(len(edges) == len(joint["edge_distance_mm"]), label + ": edge distances missing")
    cross, types = 0, Counter()
    scene_edges = {"donor": [], "recipient": []}
    seen = set()
    for index, (edge, distance) in enumerate(zip(edges, joint["edge_distance_mm"])):
        require(isinstance(edge, list) and len(edge) == 3, label + ": malformed typed edge")
        source, target, relation = edge
        require(type(source) is int and type(target) is int
                and 0 <= source < len(nodes) and 0 <= target < len(nodes) and source != target,
                label + f": invalid edge index at {index}")
        require(tuple(edge) not in seen, label + ": duplicated typed edge")
        seen.add(tuple(edge))
        a, b = nodes[source], nodes[target]
        actual_distance = math.dist(a["relative_xyz_mm"], b["relative_xyz_mm"])
        require(type(distance) in (int, float) and math.isfinite(distance)
                and abs(distance - actual_distance) <= 0.001,
                label + ": edge distance used display offset or wrong coordinates")
        if variant == "relational":
            require(relation in RELATIONS, label + ": unknown relation type")
            source_side, source_query, target_side, target_query, radius = RELATIONS[relation]
            require((a["side"], a["role"] == "query", b["side"], b["role"] == "query")
                    == (source_side, source_query, target_side, target_query),
                    label + ": typed source/target role gate violated")
            require(distance <= radius + 0.00001, label + ": physical relation radius violated")
        else:
            require(relation in BASELINE_RELATIONS and a["side"] == b["side"],
                    label + ": synthetic baseline cross edge or unknown type")
        types[relation] += 1
        if a["side"] != b["side"]:
            cross += 1
        else:
            scene_edges[a["side"]].append([a["local_index"], b["local_index"], relation])
    for side, scene in (("donor", donor), ("recipient", recipient)):
        require(scene["edges"] == scene_edges[side], label + ": branch edges disagree with joint source/target edges")
    require(joint["stats"]["node_count"] == len(nodes)
            and joint["stats"]["directed_typed_edges"] == len(edges)
            and joint["stats"]["cross_edges"] == cross
            and joint["stats"]["relation_counts"] == dict(types),
            label + ": exact joint statistics disagree")
    components, isolated = connectivity(len(nodes), edges)
    require(joint["stats"]["components"] == components == report_scene["components"],
            label + ": actual weak components disagree with final report")
    joint["stats"]["isolated_nodes"] = isolated
    require(report_scene["pair_node_count"] == len(nodes)
            and report_scene["cross_edges"] == cross,
            label + ": displayed graph differs from final all-case report")
    reported_edges = (sum(report_scene["relation_counts"].values()) if variant == "relational"
                      else report_scene["joint_directed_edges"])
    require(reported_edges == len(edges), label + ": displayed edge total differs from final report")
    if variant == "relational":
        require(all(report_scene["relation_counts"].get(name, 0) == types[name] for name in RELATIONS),
                label + ": typed edge counts differ from final report")


def merge_evidence(report, payload):
    require(report.get("debug") is True and report.get("actual_CT") is True
            and report.get("actual_CUDA") is True and report.get("actual_data_used") is True,
            "Completed actual CT/CUDA DEBUG report required")
    require(report.get("full_training") is False and report.get("full_evaluation") is False
            and report.get("production_default_changed") is False,
            "This view requires the isolated DEBUG scope")
    require(report["case"] == payload["case"], "Report/payload CT case mismatch")
    require((report["original_P"], report["original_U"], report["actual_case_observations_used"])
            == (5, 128, 133), "Original P5/U128 all-case contract changed")
    require([b["context_nodes"] for b in payload["budgets"]] == list(BUDGETS),
            "All three approved density branches are required")
    merged = copy.deepcopy(payload)
    weight = merged["weight_source"]
    if isinstance(weight, dict):
        require(isinstance(weight.get("label"), str) and weight["label"], "Explicit weight provenance label required")
        merged["weight_source"] = weight["label"]
    require(isinstance(merged["weight_source"], str), "Weight source must be a readable string")
    merged["evidence"] = {key: report[key] for key in (
        "original_P", "original_U", "actual_case_observations_used", "physical_batch",
        "effective_batch", "measured_continuous_updates_per_branch", "full_training", "full_evaluation")}
    candidate_identity = None
    for budget in merged["budgets"]:
        context, quota = budget["context_nodes"], budget["quota"]
        require(context == quota * 3, "Context quota mismatch")
        for variant in VARIANTS:
            evidence = report["budgets"][str(context)][variant]
            scenes = evidence["scenes"]
            require(len(scenes) == 133 and Counter(s["kind"] for s in scenes) == {"P": 5, "U": 128},
                    f"{context}/{variant}: incomplete all133 scene evidence")
            ids = [s["id"] for s in scenes]
            require(len(ids) == len(set(ids)), "Duplicate all-case candidate id")
            pair_nodes = {s["pair_node_count"] for s in scenes}
            require(pair_nodes == {2 * (context + 1)}, "Actual pair nodes changed across candidates")
            totals = [(sum(s["relation_counts"].values()) if variant == "relational"
                       else s["joint_directed_edges"]) for s in scenes]
            require(all(type(value) is int and value >= 0 for value in totals), "Invalid all-case edge counts")
            metrics = budget["metrics"][variant]
            metrics.update(pair_nodes=next(iter(pair_nodes)), edges_min=min(totals), edges_max=max(totals),
                           zero_cross_pairs=sum(s["cross_edges"] == 0 for s in scenes),
                           components_min=min(s["components"] for s in scenes),
                           components_max=max(s["components"] for s in scenes),
                           update_seconds=evidence["update_summary"]["full_update_seconds"]["median"],
                           peak_GiB=evidence["update_summary"]["peak_allocated_bytes"]["max"] / 2**30)
            isolated_by_id = {}
            if variant == "relational":
                for audit in evidence["graph_audits"]:
                    indices, stats = audit["original_indices"], audit["statistics"]
                    isolated_values, components_values = stats["pair_isolated_node_counts"], stats["pair_components"]
                    require(len(indices) == len(isolated_values) == len(components_values),
                            "Incomplete all-case connectivity audit")
                    for index, isolated, components in zip(indices, isolated_values, components_values):
                        require(type(index) is int and 0 <= index < len(scenes) and index not in isolated_by_id,
                                "Duplicate or invalid all-case connectivity audit index")
                        require(components == scenes[index]["components"], "Weak-component report/audit mismatch")
                        require(type(isolated) is int and 0 <= isolated < scenes[index]["pair_node_count"],
                                "Invalid isolated-node audit count")
                        isolated_by_id[index] = isolated
                require(len(isolated_by_id) == 133, "Isolated-node audit must cover all133 original pairs")
                metrics.update(isolated_min=min(isolated_by_id.values()), isolated_max=max(isolated_by_id.values()))
            else:
                metrics.update(isolated_min=None, isolated_max=None)
            candidates = budget["variants"][variant]["candidates"]
            require(len(candidates) == 8 and Counter(c["kind"] for c in candidates) == {"P": 5, "U": 3},
                    "Display must contain original P5 and explicit U3")
            identity = [(c["id"], c["kind"]) for c in candidates]
            if candidate_identity is None:
                candidate_identity = identity
            require(identity == candidate_identity, "Candidate identity changed between density/variant branches")
            lookup = {s["id"]: s for s in scenes}
            for index, candidate in enumerate(candidates):
                require(candidate["id"] in lookup and lookup[candidate["id"]]["kind"] == candidate["kind"],
                        "Display candidate missing from original all-case report")
                validate_joint(candidate, variant, quota, f"{context}/{variant}/{index}", lookup[candidate["id"]])
                if variant == "relational":
                    original_index = ids.index(candidate["id"])
                    require(candidate["joint"]["stats"]["isolated_nodes"] == isolated_by_id[original_index],
                            "Actual displayed isolated nodes disagree with original all-case audit")
                for scene in (candidate["donor"], candidate["scene"]):
                    require(scene["case"] in merged["anatomy"], "Original annotation geometry missing")
        for index in range(8):
            for side in ("donor", "scene"):
                baseline = budget["variants"]["baseline"]["candidates"][index][side]
                relational = budget["variants"]["relational"]["candidates"][index][side]
                require(baseline["case"] == relational["case"] and baseline["center"] == relational["center"]
                        and baseline["box"] == relational["box"], "Comparison crop geometry changed")
                for field in ("xyz", "native_voxel", "relative_xyz_mm", "role", "abstract", "model_slot"):
                    require([n[field] for n in baseline["nodes"]] == [n[field] for n in relational["nodes"]],
                            "Comparison must use exactly the same selected actual CT nodes")
    for annotation in merged["anatomy"].values():
        for field in ("organ_surface", "tumor_surface"):
            for contour in annotation[field]:
                for point in contour:
                    vector(point, "original annotation mm")
    return merged


def compact_payload(merged):
    """Deduplicate coordinates losslessly; omit unused scalar CNN diagnostics.

    All original anatomy polylines, displayed native/mm coordinates and exact
    typed edges survive. Only serialization is changed; model geometry is not.
    """
    compact = {key: value for key, value in merged.items() if key != "budgets"}
    compact.update(format="vrs-debug-compact-v1", node_pool=[], relation_names=[], budgets=[])
    node_lookup, relation_lookup = {}, {}
    node_fields = ("xyz", "native_voxel", "relative_xyz_mm", "role", "abstract", "case")
    for budget in merged["budgets"]:
        packed_budget = dict(context_nodes=budget["context_nodes"], quota=budget["quota"],
                             metrics=budget["metrics"], variants={})
        compact["budgets"].append(packed_budget)
        for variant in VARIANTS:
            packed_candidates = []
            packed_budget["variants"][variant] = dict(candidates=packed_candidates)
            for candidate in budget["variants"][variant]["candidates"]:
                joint = candidate["joint"]
                node_indices = []
                for node in joint["nodes"]:
                    geometry = {key: node[key] for key in node_fields}
                    key = json.dumps(geometry, separators=(",", ":"), allow_nan=False)
                    if key not in node_lookup:
                        node_lookup[key] = len(compact["node_pool"])
                        compact["node_pool"].append(geometry)
                    node_indices.append(node_lookup[key])
                packed_edges = []
                for source, target, relation in joint["edges"]:
                    if relation not in relation_lookup:
                        relation_lookup[relation] = len(compact["relation_names"])
                        compact["relation_names"].append(relation)
                    packed_edges.append([source, target, relation_lookup[relation]])
                packed_candidates.append(dict(id=candidate["id"], kind=candidate["kind"],
                    donor={key: candidate["donor"][key] for key in ("case", "center", "box")},
                    scene={key: candidate["scene"][key] for key in ("case", "center", "box")},
                    joint=dict(node_indices=node_indices, donor_node_count=len(candidate["donor"]["nodes"]),
                               edges=packed_edges, stats=joint["stats"], edge_orientation=joint["edge_orientation"])))
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed final report/payload directory")
    parser.add_argument("--output", type=Path, required=True, help="New durable inline fragment path")
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing fragment: " + str(args.output))
    report_path, payload_path = args.run / "report.json", args.run / "visual_payload.json"
    require(report_path.is_file() and payload_path.is_file(), "Final report.json and visual_payload.json are not both available")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    merged = merge_evidence(report, payload)
    source = json.dumps(compact_payload(merged), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    compressed = base64.b64encode(gzip.compress(source, compresslevel=9, mtime=0)).decode("ascii")
    template = (ROOT / "tools" / "relational-sparse.template.html").read_text(encoding="utf-8")
    require(template.count("__DATA__") == 1, "Exactly one template payload placeholder required")
    fragment = template.replace("__DATA__", compressed)
    size = len(fragment.encode("utf-8"))
    require(size < 1_000_000, "Full-precision visualization exceeds the 1 MB inline limit")
    require(not any(tag in fragment.lower() for tag in ("<!doctype", "<html", "<head", "<body")),
            "Inline output must remain an HTML fragment")
    require(args.output.parent.is_dir(), "Choose an existing task-owned output directory")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(fragment)
    require(args.output.read_text(encoding="utf-8") == fragment, "Fragment readback mismatch")
    print(json.dumps(dict(output=str(args.output.resolve()), html_bytes=size, payload_bytes=len(source),
        case=merged["case"], budgets=list(BUDGETS), displayed_P=5, displayed_U=3,
        actual_case_observations_used=133, full_precision_node_coordinates=True,
        exact_directed_typed_edges=True, full_training=False, full_evaluation=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
