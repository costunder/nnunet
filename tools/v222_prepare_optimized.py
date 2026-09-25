"""Prepare the unchanged full paired cohort, retaining calibration work.

Delegates geometry/transport/serialization to the existing implementations.
Concurrency pilots construct real pending graphs exactly once and publish them;
no discarded repeated graph-building microbenchmarks are performed.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_pair(row, loaded, donor, base, writer):
    """One production pair task, also exercised by real-CT regression tests."""
    from hiercp_v222.v1_cache import pair_record
    case, organ, depth = loaded
    if row['target']==0 and case.label[tuple(row['center'])]!=1:
        raise ValueError('Comparison annotation changed')
    record = pair_record(case,donor['source'],donor['spacing'],donor['prepared'],row['center'],
                         organ,depth,base,donor_id=row['donor_case_id'])
    relative=f"graphs/{row['case_id']}/{row['id'].split(':')[-1]}.pt.gz"
    return row | writer.write(relative,record,f"{row['donor_case_id']}:{row['donor_component']}")


def prepare(index, output, reuse=None):
    import numpy as np
    import torch
    from hiercp_v222.v1_cache import (configuration, read_json, write_new, sha, provenance, assign_pairs,
        load_verified, sources, prepare_donor, pair_record, save_torch_new, CACHE_FORMAT, emit)
    from hiercp_v222.v1_recovery import transport_preflight, recover_files
    from hiercp_v22.storage import GraphWriter
    from hiercp_v22.volumes import volume_memory_bound
    from hiercp.preparation_runtime import run_case_jobs, snapshot
    from tools.v222_runtime_cache import TensorCache
    cfg, base = configuration()
    meta = read_json(index)
    if meta['source_identity'] != provenance():
        raise ValueError('Observation input belongs to a different model/construction source')
    rows = assign_pairs(meta, cfg)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = {r['case_id']: r for r in meta['raw_records']}
    write_new(root/'started.json', dict(format=CACHE_FORMAT, config=cfg, base=base,
        source_index_sha256=sha(index), source_identity=provenance(), debug=False, subset=False,
        execution='retained-work preparation', total_observations=len(rows), donors=len(meta['donor_pool'])))
    torch.set_num_threads(1)
    previous = Path(reuse).resolve() if reuse else None
    donors = {}
    donor_cases = sorted({d['case_id'] for d in meta['donor_pool']})
    if previous:
        import os
        prior = read_json(previous/'started.json')
        if prior['source_identity'] != provenance() or prior['config'] != cfg or prior['base'] != base or prior['source_index_sha256'] != sha(index):
            raise ValueError('Reuse source/config/observations differ')
        donors = read_json(previous/'donors.json')
        for entry in donors.values():
            source, target = (previous/entry['path']).resolve(), (root/entry['path']).resolve()
            if not source.is_relative_to(previous) or not target.is_relative_to(root) or sha(source) != entry['sha256']:
                raise ValueError('Unsafe donor reuse or hash mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
    else:
        bounds = {c: volume_memory_bound(raw[c]['image']) for c in donor_cases}
        if max(bounds.values()) > snapshot()['available_memory_bytes'] * .4:
            raise MemoryError('Full CT case exceeds reserved RAM; no data reduction')
        def build_donors(c):
            case, organ, depth = load_verified(raw[c])
            entries = {}
            for source, diameter in sources(case, base['cache']['source_pad'], cfg['donor_max_diameter_mm']):
                prepared = prepare_donor(case, source, organ, depth, base)
                origin = np.array([s.start for s in source.patch_slices])
                transport = replace(source, full_mask=source.patch_mask,
                    anchor_center=tuple(np.array(source.anchor_center)-origin),
                    centroid=tuple(np.array(source.centroid)-origin),
                    patch_slices=tuple(slice(0,n) for n in source.patch_mask.shape))
                path = root/'donors'/f'{c}_{source.component_id}.pt'
                save_torch_new(path, dict(source=transport, prepared=prepared, spacing=case.spacing, diameter=diameter))
                entries[f'{c}:{source.component_id}'] = dict(path=path.relative_to(root).as_posix(), sha256=sha(path))
            emit(stage='paired_donors', case=c, donors=len(entries))
            return entries
        run_case_jobs(tasks=sorted(donor_cases, key=lambda c: -bounds[c]), function=build_donors,
            commit=donors.update, workers='auto', report_path=root/'donor_workers.json')
    if set(donors) != {f"{d['case_id']}:{d['component_id']}" for d in meta['donor_pool']}:
        raise ValueError('Donor coverage differs')
    write_new(root/'donors.json', donors)
    donor_cache = TensorCache(int(snapshot()['available_memory_bytes'] * .15))
    def get_donor(row):
        key = f"{row['donor_case_id']}:{row['donor_component']}"
        def read():
            path = root/donors[key]['path']
            if sha(path) != donors[key]['sha256']:
                raise ValueError('Donor hash changed')
            return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        return donor_cache.get(key, read)
    rows, audit = transport_preflight(rows, meta, cfg, get_donor)
    write_new(root/'transport_preflight.json', audit)
    write_new(root/'pair_assignment.json', rows)
    writer = GraphWriter(root, minimum_free_bytes=cfg['minimum_free_gb']*1024**3)
    records = recover_files(previous, root, rows, read_json(previous/'pair_assignment.json')) if previous else []
    for row in records:
        writer.sources[f"{row['donor_case_id']}:{row['donor_component']}"] = row['shared_source']
    done = {r['id'] for r in records}
    grouped = {c:[r for r in rows if r['case_id']==c and r['id'] not in done] for c in meta['split']['outer_train']}
    cases = [c for c in grouped if grouped[c]]
    began = time.perf_counter()
    attempts = set()
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        future = prefetch.submit(load_verified, raw[cases[0]]) if cases else None
        for position, c in enumerate(cases):
            case, organ, depth = future.result()
            if position+1 < len(cases):
                future = prefetch.submit(load_verified, raw[cases[position+1]])
            case_rows = []
            def build(row):
                return build_pair(row,(case,organ,depth),get_donor(row),base,writer)
            def commit(row):
                if row['id'] in attempts or row['id'] in done:
                    raise RuntimeError('Duplicate construction of an observation')
                attempts.add(row['id'])
                case_rows.append(row)
                emit(stage='paired_cache', completed=len(done)+len(attempts), total=len(rows), case=c,
                     graphs_per_second=len(attempts)/(time.perf_counter()-began))
            run_case_jobs(tasks=grouped[c], function=build, commit=commit, workers='auto',
                          report_path=root/'pair_workers'/f'{c}.json')
            write_new(root/'cases'/f'{c}.json', [r for r in records if r['case_id']==c]+case_rows)
            records.extend(case_rows)
            del case, organ, depth
    if {r['id'] for r in records} != {r['id'] for r in rows} or len(records) != len(rows):
        raise ValueError('Incomplete or duplicate paired cohort')
    write_new(root/'index.json', dict(format=CACHE_FORMAT, complete=True, debug=False,
        config=cfg, base=base, split=meta['split'], identities=meta['identities'],
        records=sorted(records,key=lambda r:r['id']), donor_pool=meta['donor_pool'], raw_records=meta['raw_records'],
        source_identity=provenance(), source_index_sha256=sha(index), donor_files=donors,
        transport_preflight=audit, reused_graphs=len(done),
        preparation_workers='measured per-case waves, retaining every result'))
    write_new(root/'execution_audit.json', dict(observations=len(rows), reused=len(done),
        unique_new_graphs=len(attempts), calibration_graphs_discarded=0,
        shared_donor_cache=donor_cache.report(), runtime_sha256={
            'tools/v222_prepare_optimized.py':sha(Path(__file__)),
            'tools/v222_runtime_cache.py':sha(ROOT/'tools/v222_runtime_cache.py')}))
    emit(stage='paired_cache_complete', observations=len(rows), index=str(root/'index.json'))
    return root/'index.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse')
    args = parser.parse_args()
    prepare(args.index, args.output, args.reuse)
