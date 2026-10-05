"""Actual CT DEBUG replay of the exact failed D row, never a training run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import psutil
    import torch
    torch.set_num_threads(2)
    from tools.run_v17_crossed_training import prepare_source
    source, scope = prepare_source(args.source)
    from hiercp_v1x.transition_v1_data import NativeObservationDataset, OriginalInputProvider
    from hiercp_v1x import transition_v1_local as local
    dataset = NativeObservationDataset(args.inventory, 'outer_train', True, ['liver_106'])
    row = next(row for row in dataset.rows if row['id'] == 'liver_106:2')
    if (row['target'] != 0 or row['donor_case_id'] != 'liver_117'
            or row['donor_component'] != 14 or row['center'] != [164, 302, 444]):
        raise ValueError('Actual failed native row identity differs')
    provider = OriginalInputProvider(dataset, workers=2, resident_bytes=24*2**30, rss_bytes=32*2**30)
    provider._initialize_raw()
    provider._raw.preload([row])
    runtime = local._runtime()
    observations = []
    previous = runtime['spatial'].validate_canonical_coordinates

    def observed(fields, points, config):
        footprint = np.asarray(fields.footprint, dtype=bool)
        organ = np.asarray(fields.organ, dtype=bool)
        outside = np.asarray(fields.outside_tumor_mm)
        depth = np.asarray(fields.liver_depth_mm)
        annulus = organ & ~footprint & (outside >= config.context_inner_radius_mm) & (outside <= config.context_outer_radius_mm)
        eligible = annulus & (depth > config.boundary_depth_mm + config.context_liver_surface_separation_mm)
        surface = organ & ~footprint & (depth > 0) & (depth <= config.boundary_depth_mm)
        observations.append(dict(
            native_shape=list(organ.shape), organ_voxels=int(organ.sum()),
            full_footprint_voxels=int(footprint.sum()), annulus_liver_voxels=int(annulus.sum()),
            eligible_deep_context_voxels=int(eligible.sum()), actual_liver_surface_voxels=int(surface.sum()),
            maximum_annulus_depth_mm=float(depth[annulus].max()) if annulus.any() else None,
            canonical_nodes={key: len(value) for key, value in points.items()},
            semantic_mask_sha256=hashlib.sha256(np.ascontiguousarray(eligible).tobytes()).hexdigest(),
            organ_sha256=local._hash_array(organ), footprint_sha256=local._hash_array(footprint),
            boundary_depth_mm=config.boundary_depth_mm,
            surface_separation_mm=config.context_liver_surface_separation_mm,
            context_inner_mm=config.context_inner_radius_mm, context_outer_mm=config.context_outer_radius_mm))
        return previous(fields, points, config)

    runtime['spatial'].validate_canonical_coordinates = observed
    started = time.perf_counter()
    error = None
    record = None
    try:
        record = provider._records_for([row])[0]
    except RuntimeError as failure:
        error = str(failure)
        if 'context coordinates are invalid: (0, 3)' not in error:
            raise
    finally:
        runtime['spatial'].validate_canonical_coordinates = previous
    report = dict(format='v17_actual_failed_recipient_context_DEBUG_v1', debug=True,
        observation=row, source=source, scope=scope, local_identity=local.source_identity(),
        geometry_measurements=observations, error=error, constructed=record is not None,
        wall_seconds=time.perf_counter()-started, rss_bytes=psutil.Process().memory_info().rss,
        training_started=False, production_ready=False, original_files_modified=False,
        GT_center_donor_margin_changed=False)
    if record is not None:
        views, _, _ = local.materialize_pair(record, epoch=0)
        report['record_audit'] = record['audit']
        report['view_node_counts'] = [{key: graph[key].num_nodes for key in local._ROLE_NAMES} for graph in views]
        torch.save(record, args.output/'actual_record_DEBUG.pt')
    with (args.output/'report.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key:report[key] for key in ('constructed', 'error', 'geometry_measurements', 'wall_seconds', 'rss_bytes')}), flush=True)


if __name__ == '__main__':
    main()
