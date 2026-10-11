"""Export native, mask-derived vascular centerlines and branch adjacency.

This performs no training, nearest-neighbor gap bridging, component pruning,
resolution change, or graph-size limiting. NIfTI affine coordinates are RAS mm.
Audited Lee/simple-point thinning and EDT radii are voxel-mask estimates, not
VMTK lumen geometry. Invalid Lee components are retried on native axes, then
with sequential topology-preserving deletion; successful components stay intact.
The complete voxel skeleton is exported alongside its branch representation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.measure import euler_number
from skimage.morphology import skeletonize


def _matrix_and_spacing(affine):
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("A finite 4x4 voxel-to-RAS affine is required")
    if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-10):
        raise ValueError("Affine must have homogeneous last row [0,0,0,1]")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Affine voxel spacing must be positive")
    axes = affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("Sheared affine is unsupported by spacing-only EDT; do not silently approximate radii")
    return affine, spacing


def native_grid_alignment(reference, image, *, tolerance_mm=0.001):
    """Verify identical voxel indices allowing only NIfTI header roundoff.

    Every corresponding volume corner must differ by at most one micrometre.
    This does not resample either volume or alter its stored affine/voxels.
    Comparing only affine coefficients would miss accumulated scale errors.
    """
    if image.shape != reference.shape or len(reference.shape) != 3:
        raise ValueError("Volume shapes differ from the native CT grid")
    _matrix_and_spacing(reference.affine)
    _matrix_and_spacing(image.affine)
    corners = np.asarray(list(itertools.product(*[(0, size - 1) for size in reference.shape])), dtype=float)
    displacement = np.linalg.norm(nib.affines.apply_affine(reference.affine, corners)
                                  - nib.affines.apply_affine(image.affine, corners), axis=1)
    maximum = float(displacement.max())
    if maximum > tolerance_mm:
        raise ValueError(f"Volume differs from native CT grid by {maximum:.8g} mm; "
                         f"header roundoff tolerance is {tolerance_mm} mm")
    return dict(maximum_corner_displacement_mm=maximum,
                header_roundoff_tolerance_mm=tolerance_mm, resampled=False,
                matching_native_shape=True)


def _neighbors(voxel_xyz):
    """All actual native 26-neighbor edges; no distance-based gap connections."""
    if len(voxel_xyz) < 2:
        return np.empty((0, 2), dtype=np.int64)
    edges = cKDTree(voxel_xyz).query_pairs(1, p=np.inf, output_type="ndarray")
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


def _flatten_groups(groups):
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([len(g) for g in groups])
        points = np.concatenate(groups).astype(np.int64, copy=False)
    else:
        points = np.empty(0, dtype=np.int64)
    return points, offsets


def graph_from_skeleton(voxel_xyz, affine, radius_edt_mm, input_component_ids=None):
    """Compress degree-2 paths; keep all raw edges and junction-region voxels.

    Branch nodes are adjacent only when they share an actual junction region.
    Endpoints, isolated points, and cycle anchors have explicit terminal IDs.
    Junction interiors remain in the raw skeleton rather than becoming invented
    straight lines. Pure degree-2 cycles are split at an existing skeleton point.
    """
    affine, _ = _matrix_and_spacing(affine)
    coords = np.asarray(voxel_xyz)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("voxel_xyz must have shape [N,3]")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("Skeleton voxel coordinates must be integer native indices")
    coords = coords.astype(np.int32, copy=False)
    n = len(coords)
    if len(np.unique(coords, axis=0)) != n:
        raise ValueError("Skeleton voxel coordinates must be unique")
    radii = np.asarray(radius_edt_mm, dtype=np.float32)
    if radii.shape != (n,) or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Every skeleton point requires a finite positive EDT radius")
    raw_components = (np.ones(n, dtype=np.int32) if input_component_ids is None
                      else np.asarray(input_component_ids, dtype=np.int32))
    if raw_components.shape != (n,) or np.any(raw_components <= 0):
        raise ValueError("Every skeleton point requires an input mask component ID")
    world = nib.affines.apply_affine(affine, coords)
    edges = _neighbors(coords)
    if len(edges) and np.any(raw_components[edges[:, 0]] != raw_components[edges[:, 1]]):
        raise ValueError("An edge crossed input components; connectivity conventions disagree")
    edge_ids = np.arange(len(edges), dtype=np.int64)
    # Edge ID plus one avoids sparse zero elimination changing adjacency.
    adjacency = sparse.csr_matrix(
        (np.concatenate((edge_ids + 1, edge_ids + 1)),
         (np.concatenate((edges[:, 0], edges[:, 1])),
          np.concatenate((edges[:, 1], edges[:, 0])))), shape=(n, n))
    degree = np.diff(adjacency.indptr).astype(np.int32)
    skel_count, skel_components = connected_components(adjacency, directed=False)
    terminal_of_point = np.full(n, -1, dtype=np.int64)
    groups, kinds = [], []

    def add_group(points, kind):
        group_id = len(groups)
        members = np.asarray(points, dtype=np.int64)
        groups.append(members)
        kinds.append(kind)
        terminal_of_point[members] = group_id

    junction_points = np.flatnonzero(degree > 2)
    if len(junction_points):
        junction_count, junction_labels = connected_components(
            adjacency[junction_points][:, junction_points], directed=False)
        for component in range(junction_count):
            add_group(junction_points[junction_labels == component], "junction")
    for point in np.flatnonzero(degree == 1):
        add_group([point], "endpoint")
    for point in np.flatnonzero(degree == 0):
        add_group([point], "isolated_point")
    for component in range(skel_count):
        members = np.flatnonzero(skel_components == component)
        if len(members) and np.all(degree[members] == 2):
            add_group([members[0]], "cycle_anchor")

    same_terminal = ((terminal_of_point[edges[:, 0]] >= 0)
                     & (terminal_of_point[edges[:, 0]] == terminal_of_point[edges[:, 1]]))
    visited = same_terminal.copy()
    paths, terminal_pairs = [], []
    for terminal_id, members in enumerate(groups):
        for start in members:
            row_lo, row_hi = adjacency.indptr[start:start + 2]
            for offset in range(row_lo, row_hi):
                first_edge = adjacency.data[offset] - 1
                if visited[first_edge]:
                    continue
                nxt = int(adjacency.indices[offset])
                visited[first_edge] = True
                path = [int(start), nxt]
                previous, current = int(start), nxt
                while terminal_of_point[current] < 0:
                    if degree[current] != 2:
                        raise RuntimeError("Nonterminal point does not have degree 2")
                    lo, hi = adjacency.indptr[current:current + 2]
                    choices = [i for i in range(lo, hi) if adjacency.indices[i] != previous]
                    if len(choices) != 1:
                        raise RuntimeError("Ambiguous degree-2 path")
                    step = choices[0]
                    eid = adjacency.data[step] - 1
                    if visited[eid]:
                        raise RuntimeError("A skeleton edge was traversed twice")
                    visited[eid] = True
                    previous, current = current, int(adjacency.indices[step])
                    path.append(current)
                paths.append(np.asarray(path, dtype=np.int64))
                terminal_pairs.append((terminal_id, int(terminal_of_point[current])))
    if not visited.all():
        raise RuntimeError("Some actual skeleton edges were omitted from branch decomposition")

    # Retain components containing just a point or a junction region as explicit
    # degenerate nodes; their complete geometry is in the raw skeleton arrays.
    terminal_incident = [[] for _ in groups]
    for branch_id, pair in enumerate(terminal_pairs):
        for terminal in set(pair):
            terminal_incident[terminal].append(branch_id)
    degenerate_kinds = {}
    for terminal, incident in enumerate(terminal_incident):
        if not incident:
            members = groups[terminal]
            centroid = world[members].mean(axis=0)
            representative = int(members[np.argmin(np.linalg.norm(world[members] - centroid, axis=1))])
            branch_id = len(paths)
            paths.append(np.asarray([representative], dtype=np.int64))
            terminal_pairs.append((terminal, terminal))
            terminal_incident[terminal].append(branch_id)
            degenerate_kinds[branch_id] = ("isolated_point" if kinds[terminal] == "isolated_point"
                                          else "junction_region")

    nodes = []
    for branch_id, (path, terminals) in enumerate(zip(paths, terminal_pairs)):
        xyz = world[path]
        segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        length = float(cumulative[-1])
        # Pick an existing centerline voxel nearest the arclength midpoint. This
        # keeps displayed node locations within the actual input vascular mask.
        midpoint_index = int(np.argmin(np.abs(cumulative - length / 2)))
        radius = radii[path]
        loop = terminals[0] == terminals[1] and len(path) > 1
        kind = degenerate_kinds.get(branch_id, "closed_cycle" if loop else "branch")
        nodes.append({
            "id": branch_id, "kind": kind,
            "position_ras_mm": xyz[midpoint_index].tolist(),
            "position_point_index": int(path[midpoint_index]),
            "source_junction_id": int(terminals[0]), "target_junction_id": int(terminals[1]),
            "input_component_id": int(raw_components[path[0]]),
            "skeleton_component_id": int(skel_components[path[0]]) + 1,
            "closed_loop": loop, "point_count": len(path), "length_mm": length,
            "radius_edt_mean_mm": float(radius.mean()),
            "radius_edt_min_mm": float(radius.min()), "radius_edt_max_mm": float(radius.max()),
            "polyline_point_indices": path.tolist(),
            "geometry_note": "native traced path; junction interior excluded from branch length",
        })

    # Shared terminals become graph adjacency only for an actual branch junction.
    graph_edges = {}
    junctions = []
    for terminal, (members, kind) in enumerate(zip(groups, kinds)):
        centroid = world[members].mean(axis=0)
        representative = int(members[np.argmin(np.linalg.norm(world[members] - centroid, axis=1))])
        junctions.append({
            "id": terminal, "kind": kind, "position_ras_mm": world[representative].tolist(),
            "representative_point_index": representative,
            "member_point_indices": members.tolist(),
            "incident_branch_ids": terminal_incident[terminal],
            "input_component_id": int(raw_components[representative]),
        })
        if kind == "junction":
            for a, b in itertools.combinations(terminal_incident[terminal], 2):
                graph_edges.setdefault((min(a, b), max(a, b)), []).append(terminal)
    graph_edge_array = np.asarray(list(graph_edges), dtype=np.int64).reshape(-1, 2)
    path_points, path_offsets = _flatten_groups(paths)
    junction_points_flat, junction_offsets = _flatten_groups(groups)
    graph = {
        "schema_version": 1,
        "coordinate_system": "RAS", "units": "mm",
        "method": "native Lee thinning; full 26-neighbor skeleton; degree-2 path decomposition",
        "radius_method": "spacing-aware Euclidean distance to nearest background voxel center",
        "nodes": nodes,
        "edges": [{"source": a, "target": b, "via_junction_ids": ids}
                  for (a, b), ids in graph_edges.items()],
        "junctions": junctions,
        "counts": {"skeleton_points": n, "skeleton_edges": len(edges),
                   "skeleton_components": int(skel_count), "branch_nodes": len(nodes),
                   "branch_graph_edges": len(graph_edges),
                   "junction_regions": kinds.count("junction"),
                   "isolated_points": kinds.count("isolated_point"),
                   "closed_loop_nodes": sum(node["closed_loop"] for node in nodes)},
        "skeleton_edge_coverage": {"internal_junction_edges": int(same_terminal.sum()),
                                   "traced_branch_edges": int((~same_terminal).sum()),
                                   "unaccounted_edges": 0},
        "bounds_ras_mm": {"min": world.min(axis=0).tolist(), "max": world.max(axis=0).tolist()} if n else None,
        "notes": ["Graph adjacency follows shared actual junctions, with no gap bridging.",
                  "Mask-derived topology and EDT radii require inspection against CT.",
                  "All junction-region voxels and edges are retained in the NPZ raw skeleton.",
                  "Graph midpoint-to-midpoint edges are abstract adjacency, not lumen paths."],
    }
    arrays = {
        "voxel_xyz": coords, "point_ras_mm": world, "radius_edt_mm": radii,
        "input_component_id": raw_components, "skeleton_component_id": skel_components.astype(np.int32) + 1,
        "degree_26": degree, "skeleton_edge_index": edges,
        "branch_point_index": path_points, "branch_point_offsets": path_offsets,
        "junction_point_index": junction_points_flat, "junction_point_offsets": junction_offsets,
        "branch_junction_ids": np.asarray(terminal_pairs, dtype=np.int64).reshape(-1, 2),
        "branch_graph_edge_index": graph_edge_array,
    }
    return graph, arrays


def _component_topology(mask):
    """Exact finite-volume 26-object/6-background audit, including exterior."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    _, foreground = ndimage.label(padded, structure=np.ones((3, 3, 3), bool))
    _, background = ndimage.label(~padded, structure=ndimage.generate_binary_structure(3, 1))
    return {"foreground_26_components": int(foreground),
            "background_6_components": int(background),
            "euler_26": int(euler_number(padded, connectivity=3))}


def _component_candidate_audit(piece, candidate, original_topology):
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.shape != piece.shape:
        raise ValueError("Component candidate shape changed during thinning")
    topology = _component_topology(candidate)
    inside = not bool(np.any(candidate & ~piece))
    accepted = bool(candidate.any() and inside and topology == original_topology)
    return {"skeleton_voxel_count": int(candidate.sum()), **topology,
            "all_points_inside_input_component": inside,
            "accepted": accepted}


def _simple_point_26_6(neighborhood):
    """Local simple-point predicate, with complementary 26/6 connectivity.

    T26 counts object components in N26 without the point. T6 counts only
    background components in N18 that touch a face neighbor of the point.
    See Bertrand's local characterization and its geodesic neighborhood in
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2084352/ (Section 2.1), with object
    and background interchanged to obtain this project's 26/6 convention.
    """
    neighborhood = np.asarray(neighborhood, dtype=bool)
    if neighborhood.shape != (3, 3, 3) or not neighborhood[1, 1, 1]:
        raise ValueError("Simple-point test requires an occupied padded 3x3x3 neighborhood")
    foreground = neighborhood.copy()
    foreground[1, 1, 1] = False
    _, foreground_count = ndimage.label(foreground, structure=np.ones((3, 3, 3), bool))
    if foreground_count != 1:
        return False
    coordinates = np.indices((3, 3, 3)) - 1
    manhattan = np.abs(coordinates).sum(axis=0)
    background = ~neighborhood & (manhattan > 0) & (manhattan <= 2)
    background_labels, _ = ndimage.label(background, structure=ndimage.generate_binary_structure(3, 1))
    touching = np.unique(background_labels[manhattan == 1])
    return len(touching[touching > 0]) == 1


def _sequential_component_thinning(piece, spacing):
    """Repair Lee erasure with native simple-point deletion, never a mask cap.

    Deletions depend on the preceding state and are deliberately sequential.
    This path operates only on components rejected by every native Lee order;
    independent cases remain suitable for process-level parallel graph export.
    Every accepted deletion is also gated by the complete component's original
    foreground count, background count and Euler number. No endpoint is erased.
    """
    original_topology = _component_topology(piece)
    working = np.pad(np.asarray(piece, dtype=bool), 1)
    distance = ndimage.distance_transform_edt(working, sampling=spacing)
    coordinates = np.argwhere(working)
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0],
                        distance[tuple(coordinates.T)]))
    coordinates = coordinates[order]
    deleted = []
    passes = 0
    simple_point_rejections = topology_rejections = endpoint_visits = 0
    changed = True
    while changed:
        passes += 1
        changed = False
        for voxel in coordinates:
            point = tuple(voxel)
            if not working[point]:
                continue
            neighborhood = working[tuple(slice(int(value) - 1, int(value) + 2) for value in voxel)]
            degree = int(neighborhood.sum()) - 1
            if degree <= 1:
                endpoint_visits += 1
                continue
            if not _simple_point_26_6(neighborhood):
                simple_point_rejections += 1
                continue
            working[point] = False
            after = _component_topology(working[1:-1, 1:-1, 1:-1])
            if after != original_topology:
                working[point] = True
                topology_rejections += 1
                continue
            changed = True
            deleted.append({"component_local_voxel_xyz": (voxel - 1).tolist(),
                            "original_radius_edt_mm": float(distance[point]),
                            "local_simple_point_26_6": True,
                            "topology_after_deletion": after})
    result = working[1:-1, 1:-1, 1:-1].copy()
    return result, {
        "method": "sequential_native_simple_point_26_6_thinning",
        "ordering": "ascending original spacing-aware EDT; stable native X/Y/Z tie order",
        "endpoint_policy": "protect every remaining voxel with 26-neighbor degree <= 1",
        "termination": "a complete pass with no accepted topology-preserving deletion",
        "passes": passes, "deleted_voxel_count": len(deleted), "deletions": deleted,
        "endpoint_visits": endpoint_visits, "simple_point_rejections": simple_point_rejections,
        "global_topology_rejections": topology_rejections,
        "original_topology": original_topology, "final_topology": _component_topology(result),
        "native_resolution_preserved": True, "input_mask_modified": False,
        "simple_point_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2084352/",
    }


def _recover_component(piece, initial_candidate, thinning_axes, spacing, original_topology=None):
    """Use the first fully audited native candidate; retain failure if none pass."""
    if original_topology is None:
        original_topology = _component_topology(piece)
    initial_audit = _component_candidate_audit(piece, initial_candidate, original_topology)
    if initial_audit["accepted"]:
        return initial_candidate, None
    failures = []
    if not initial_candidate.any():
        failures.append("component_erased")
    elif initial_audit["foreground_26_components"] != original_topology["foreground_26_components"]:
        failures.append("component_split")
    if initial_audit["euler_26"] != original_topology["euler_26"]:
        failures.append("euler_changed")
    if initial_audit["background_6_components"] != original_topology["background_6_components"]:
        failures.append("background_topology_changed")
    if not initial_audit["all_points_inside_input_component"]:
        failures.append("point_outside_component")
    canonical = tuple(map(int, thinning_axes))
    orders = [canonical] + [order for order in itertools.permutations(range(3)) if order != canonical]
    record = {"initial_failure_reasons": failures, "initial_audit": initial_audit,
              "original_topology": original_topology, "axis_candidate_audits": [],
              "recovery_method": None, "recovered": False}
    padded = np.pad(piece, 1)
    for order in orders:
        inverse = np.argsort(order)
        candidate = np.asarray(skeletonize(padded.transpose(order), method="lee"), dtype=bool)
        candidate = candidate.transpose(tuple(inverse))[1:-1, 1:-1, 1:-1]
        audit = _component_candidate_audit(piece, candidate, original_topology)
        record["axis_candidate_audits"].append({"axis_permutation": list(order), **audit})
        if audit["accepted"]:
            record.update(recovery_method="native_component_lee_axis_retry", recovered=True,
                          selected_axis_permutation=list(order), final_audit=audit,
                          untried_axis_permutations=[list(value) for value in orders[len(record["axis_candidate_audits"]):]])
            return candidate, record
    candidate, sequential = _sequential_component_thinning(piece, spacing)
    audit = _component_candidate_audit(piece, candidate, original_topology)
    record["sequential_thinning"] = sequential
    record["sequential_candidate_audit"] = audit
    record["untried_axis_permutations"] = []
    if audit["accepted"]:
        record.update(recovery_method=sequential["method"], recovered=True, final_audit=audit)
        return candidate, record
    record["final_audit"] = initial_audit
    record["recovery_rejected"] = "No candidate retained the exact native component topology"
    return initial_candidate, record


def centerline_from_mask(mask, affine):
    """Thin every native foreground component and explicitly audit its survival."""
    affine, spacing = _matrix_and_spacing(affine)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("A nonempty three-dimensional vascular mask is required")
    started = time.perf_counter()
    foreground = np.argwhere(mask)
    lower = foreground.min(axis=0)
    upper = foreground.max(axis=0) + 1
    crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    # Empty exterior crop and padding retain every voxel and physical spacing.
    cropped = np.pad(mask[crop_slices], 1, mode="constant", constant_values=False)
    component_labels, component_count = ndimage.label(cropped, structure=np.ones((3, 3, 3), bool))
    # Lee's directional deletion schedule depends on array-axis order. Use a
    # documented spacing-derived canonical order, with stable native X/Y/Z tie
    # order, rather than silently erasing single-slice fragments. Transposition
    # and its exact inverse change neither input voxels nor physical resolution.
    thinning_axes = np.argsort(-spacing, kind="stable")
    inverse_axes = np.argsort(thinning_axes)
    oriented = cropped.transpose(tuple(thinning_axes))
    skeleton = np.asarray(skeletonize(oriented, method="lee"), dtype=bool).transpose(tuple(inverse_axes))
    thinning_seconds = time.perf_counter() - started
    if np.any(skeleton & ~cropped):
        raise RuntimeError("Thinning generated a point outside the native vessel mask")
    edt_started = time.perf_counter()
    distance = ndimage.distance_transform_edt(cropped, sampling=spacing)
    edt_seconds = time.perf_counter() - edt_started
    recovery_started = time.perf_counter()
    component_slices = ndimage.find_objects(component_labels)
    original_topologies = {}
    recovery_records = []
    initial_global_euler = int(euler_number(skeleton, connectivity=3))
    for component, slices in enumerate(component_slices, 1):
        piece = component_labels[slices] == component
        initial_candidate = skeleton[slices] & piece
        original_topologies[component] = _component_topology(piece)
        candidate, record = _recover_component(piece, initial_candidate, thinning_axes, spacing,
                                                original_topologies[component])
        if record is None:
            continue
        record["input_component_id"] = component
        native_lower = [int(s.start + offset - 1) for s, offset in zip(slices, lower)]
        record["bbox_lower_native_voxel_xyz"] = native_lower
        record["bbox_upper_exclusive_native_voxel_xyz"] = [
            int(s.stop + offset - 1) for s, offset in zip(slices, lower)]
        for deletion in record.get("sequential_thinning", {}).get("deletions", []):
            deletion["native_voxel_xyz"] = [
                int(value + offset) for value, offset in zip(deletion["component_local_voxel_xyz"], native_lower)]
        if record["recovered"]:
            # Write only this component's original native indices. Successful
            # global Lee components remain byte-for-byte identical in skeleton.
            skeleton[slices][piece] = candidate[piece]
        recovery_records.append(record)
    recovery_seconds = time.perf_counter() - recovery_started
    if np.any(skeleton & ~cropped):
        raise RuntimeError("Component recovery generated a point outside the native vessel mask")
    local_voxels = np.argwhere(skeleton)
    native_voxels = local_voxels + lower - 1
    covered_ids = component_labels[tuple(local_voxels.T)] if len(local_voxels) else np.empty(0, dtype=np.int32)
    covered = set(map(int, np.unique(covered_ids)))
    counts = np.bincount(component_labels.ravel(), minlength=component_count + 1)
    missing = [{"input_component_id": i, "voxel_count": int(counts[i])}
               for i in range(1, component_count + 1) if i not in covered]
    radii = distance[tuple(local_voxels.T)].astype(np.float32)
    topology_started = time.perf_counter()
    component_audits = []
    euler_mismatches = []
    background_mismatches = []
    for component, slices in enumerate(component_slices, 1):
        piece = component_labels[slices] == component
        piece_skeleton = skeleton[slices] & piece
        original_topology = original_topologies[component]
        skeleton_topology = _component_topology(piece_skeleton)
        original_euler = original_topology["euler_26"]
        skeleton_euler = skeleton_topology["euler_26"]
        skeleton_piece_count = skeleton_topology["foreground_26_components"]
        audit = {"input_component_id": component, "input_voxel_count": int(piece.sum()),
                 "skeleton_voxel_count": int(piece_skeleton.sum()),
                 "skeleton_component_count": int(skeleton_piece_count),
                 "input_euler_26": original_euler, "skeleton_euler_26": skeleton_euler,
                 "euler_preserved": original_euler == skeleton_euler,
                 "input_background_6_components": original_topology["background_6_components"],
                 "skeleton_background_6_components": skeleton_topology["background_6_components"],
                 "background_preserved": original_topology["background_6_components"] == skeleton_topology["background_6_components"],
                 "bbox_lower_native_voxel_xyz": [int(s.start + offset - 1) for s, offset in zip(slices, lower)],
                 "bbox_upper_exclusive_native_voxel_xyz": [int(s.stop + offset - 1) for s, offset in zip(slices, lower)]}
        component_audits.append(audit)
        if original_euler != skeleton_euler:
            euler_mismatches.append(audit)
        if not audit["background_preserved"]:
            background_mismatches.append(audit)
    original_global_euler = int(euler_number(cropped, connectivity=3))
    skeleton_global_euler = int(euler_number(skeleton, connectivity=3))
    topology_seconds = time.perf_counter() - topology_started
    graph, arrays = graph_from_skeleton(native_voxels.astype(np.int32), affine, radii, covered_ids)
    graph.update({
        "complete_centerline_coverage": not missing,
        "status": "complete_mask_component_coverage" if not missing else "incomplete_component_coverage",
        "input_mask_components": int(component_count), "missing_components": missing,
        "input_mask_voxels": int(mask.sum()), "native_shape_xyz": list(mask.shape),
        "voxel_spacing_xyz_mm": spacing.tolist(), "affine_voxel_to_ras_mm": affine.tolist(),
        "processing_crop_lower_voxel_xyz": lower.tolist(),
        "processing_crop_upper_exclusive_voxel_xyz": upper.tolist(),
        "thinning_axis_order_policy": "descending physical voxel spacing, stable native X/Y/Z tie order",
        "thinning_axis_permutation": thinning_axes.tolist(),
        "thinning_array_native_axes": ["XYZ"[int(axis)] for axis in thinning_axes],
        "thinning_inverse_axis_permutation": inverse_axes.tolist(),
        "thinning_reorientation_resampled": False,
        "native_resolution_preserved": True, "all_components_requested": True,
        "thinning_recovery": {
            "policy": "retain valid global canonical Lee components; retry invalid components on canonical then lexicographic native axis orders; if all fail, sequential audited 26/6 simple-point thinning",
            "initial_global_skeleton_euler_26": initial_global_euler,
            "initial_invalid_component_count": len(recovery_records),
            "recovered_component_count": sum(record["recovered"] for record in recovery_records),
            "unrecovered_component_count": sum(not record["recovered"] for record in recovery_records),
            "valid_initial_components_unchanged": True, "input_mask_modified": False,
            "native_resolution_preserved": True, "component_audits": recovery_records,
        },
        "timing_seconds": {"thinning": thinning_seconds, "edt": edt_seconds,
                           "component_recovery": recovery_seconds, "topology_audit": topology_seconds,
                           "total_centerline_and_graph": time.perf_counter() - started},
        "topology_audit": {"foreground_connectivity": 26, "background_connectivity": 6,
                           "input_euler_26": original_global_euler,
                           "skeleton_euler_26": skeleton_global_euler,
                           "euler_mismatches": euler_mismatches, "background_mismatches": background_mismatches,
                           "components": component_audits,
                           "all_skeleton_points_inside_input_mask": True,
                           "meaning": "digital mask component, connectivity and Euler audit; not anatomy validation"},
        "anatomy_validated": False,
    })
    if recovery_records:
        graph["method"] = ("native canonical Lee thinning with audited component axis retries and "
                           "sequential 26/6 simple-point recovery; full 26-neighbor skeleton; degree-2 path decomposition")
    # Detect input components that split after thinning; retain all resulting parts.
    split_components = []
    for component in sorted(covered):
        pieces = np.unique(arrays["skeleton_component_id"][covered_ids == component])
        if len(pieces) > 1:
            split_components.append({"input_component_id": component,
                                     "skeleton_component_ids": pieces.tolist()})
    graph["split_components_after_thinning"] = split_components
    graph["topology_audit_passed"] = (not missing and not split_components
                                     and not euler_mismatches and not background_mismatches)
    if split_components and not missing:
        graph["status"] = "incomplete_topology_split_after_thinning"
    if euler_mismatches and not missing and not split_components:
        graph["status"] = "incomplete_topology_euler_mismatch"
    if background_mismatches and not missing and not split_components and not euler_mismatches:
        graph["status"] = "incomplete_topology_background_mismatch"
    native_skeleton = np.zeros(mask.shape, dtype=np.uint8)
    if len(native_voxels):
        native_skeleton[tuple(native_voxels.T)] = 1
    return graph, arrays, native_skeleton


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary_volume(image, name):
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError(f"{name} must contain finite nonnegative mask values")
    if not np.equal(data, np.floor(data)).all():
        raise ValueError(f"{name} must contain integer segmentation labels")
    return data


def _save_native_mask(data, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), reference.affine, header)
    result.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    result.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(result, path)


def export_vessel_graph(ct_path, labels_path, vessels_path, out_dir, *, case_id=None,
                        run_label="native_case_preprocessing", assume_mm=False):
    """Create a new output directory; source CT, GT and raw TS mask stay intact."""
    started = time.perf_counter()
    paths = {"ct": Path(ct_path).resolve(), "labels": Path(labels_path).resolve(),
             "vessels": Path(vessels_path).resolve()}
    output = Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists; refusing overwrite: {output}")
    images = {name: nib.load(path) for name, path in paths.items()}
    ct = images["ct"]
    _matrix_and_spacing(ct.affine)
    if len(ct.shape) != 3:
        raise ValueError("CT must be a three-dimensional native volume")
    spatial_unit = ct.header.get_xyzt_units()[0]
    if spatial_unit != "mm" and not (spatial_unit == "unknown" and assume_mm):
        raise ValueError(f"CT spatial units are {spatial_unit!r}; explicit --assume-mm required for unknown units")
    grid_audit = {}
    for name in ("labels", "vessels"):
        image = images[name]
        try:
            grid_audit[name] = native_grid_alignment(ct, image)
        except ValueError as error:
            raise ValueError(f"{name} is not on the original CT grid: {error}") from error
        unit = image.header.get_xyzt_units()[0]
        if unit not in ("unknown", spatial_unit):
            raise ValueError(f"{name} spatial units disagree with the CT")
    labels = _binary_volume(images["labels"], "Ground-truth liver/tumor labels")
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError("LiTS ground-truth labels must use background=0, liver=1, tumor=2")
    raw_vessels = _binary_volume(images["vessels"], "Raw vessel mask") > 0
    liver = labels > 0
    tumor = labels == 2
    if not liver.any():
        raise ValueError("Ground-truth liver extent is empty")
    clipped = raw_vessels & liver
    graph, arrays, native_skeleton = centerline_from_mask(clipped, ct.affine)
    graph.update({"case_id": case_id or paths["ct"].name, "run_label": run_label,
                  "liver_constraint": "GT labels > 0, including tumor anatomy; exact native intersection",
                  "tumor_overlay": "GT labels == 2, separate from vessel mask",
                  "vessel_tumor_overlap_voxels": int((clipped & tumor).sum()),
                  "raw_vessel_voxels": int(raw_vessels.sum()),
                  "outside_liver_vessel_voxels": int((raw_vessels & ~liver).sum()),
                  "ct_spatial_units_header": spatial_unit, "unknown_units_assumed_mm": spatial_unit == "unknown",
                  "native_grid_audit": grid_audit,
                  "source_files": {name: {"path": str(path), "sha256": _sha256(path)}
                                   for name, path in paths.items()},
                  "training_started": False, "full_dataset_processed": False,
                  "sampling_ratio": 1.0, "graph_node_cap": None, "graph_edge_cap": None,
                  "component_pruning": False, "gap_bridging": False,
                  "radius_boundary_note": "Liver clipping can affect EDT radius near the liver boundary"})
    graph["npz_schema"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for name, value in arrays.items()}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "centerline_graph.npz", **arrays)
    _save_native_mask(clipped, ct, output / "vessels_liver_native.nii.gz")
    _save_native_mask(native_skeleton, ct, output / "skeleton_native.nii.gz")
    graph["export_total_seconds"] = time.perf_counter() - started
    with (output / "graph.json").open("x", encoding="utf-8") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    receipt = {"case_id": graph["case_id"], "output_directory": str(output),
               "status": graph["status"], "complete_centerline_coverage": graph["complete_centerline_coverage"],
               "topology_audit_passed": graph["topology_audit_passed"],
               "counts": graph["counts"], "missing_components": graph["missing_components"],
               "split_components_after_thinning": graph["split_components_after_thinning"],
               "euler_mismatch_components": [a["input_component_id"] for a in graph["topology_audit"]["euler_mismatches"]],
               "background_mismatch_components": [a["input_component_id"] for a in graph["topology_audit"]["background_mismatches"]],
               "thinning_recovery": {
                   "initial_invalid_component_count": graph["thinning_recovery"]["initial_invalid_component_count"],
                   "recovered_component_count": graph["thinning_recovery"]["recovered_component_count"],
                   "unrecovered_component_count": graph["thinning_recovery"]["unrecovered_component_count"],
                   "components": [{"input_component_id": record["input_component_id"],
                                   "initial_failure_reasons": record["initial_failure_reasons"],
                                   "recovery_method": record["recovery_method"],
                                   "recovered": record["recovered"]}
                                  for record in graph["thinning_recovery"]["component_audits"]],
               },
               "input_euler_26": graph["topology_audit"]["input_euler_26"],
               "skeleton_euler_26": graph["topology_audit"]["skeleton_euler_26"],
               "thinning_axis_permutation": graph["thinning_axis_permutation"],
               "anatomy_validated": False,
               "actual_native_data_used": True, "training_started": False,
               "full_dataset_processed": False, "total_seconds": graph["export_total_seconds"]}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--vessels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--run-label", default="native_case_preprocessing")
    parser.add_argument("--assume-mm", action="store_true",
                        help="Explicitly interpret unknown CT NIfTI spatial units as mm")
    args = parser.parse_args()
    receipt = export_vessel_graph(args.ct, args.labels, args.vessels, args.out,
                                 case_id=args.case_id, run_label=args.run_label, assume_mm=args.assume_mm)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
