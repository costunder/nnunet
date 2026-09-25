"""DEBUG controlled L0 cost comparison on matched real CT sites.

This is NOT a recreation of the historical trained model or a full-pipeline
benchmark. Each encoder retains its own native features/topology. A fused-output
squared-norm probe measures backward cost; it is not either training objective.
No production configuration or existing cache is modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from torch_geometric.data import Batch


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def prepare(root):
    from hiercp_v222.v1_cache import configuration, load_verified, read_json
    from hiercp_v222.v1_local import prepare_donor, pair_record, materialize
    from hiercp.local import prepare_local_source, build_local_graph
    from hiercp.sample import build_local_view
    from hiercp.schema import graph_config_from_dict
    from hiercp_v22.data import sources, candidate_spec
    cfg, base = configuration()
    meta = read_json(ROOT / 'work/v222_raw_ct_r3_training_20260923/context/index_vram_affine.json')
    torch.set_num_threads(1)
    graph_cfg = graph_config_from_dict(base['graph'])
    records = []
    timings = []
    # Explicit DEBUG cohort; all nodes/edges in each native graph are retained.
    # Same-case pairs isolate encoder cost, not current cross-patient training.
    for case_id in ('liver_5', 'liver_6'):
        raw = next(r for r in meta['raw_records'] if r['case_id'] == case_id)
        case, organ, depth = load_verified(raw)
        source, diameter = sources(case, base['cache']['source_pad'], cfg['donor_max_diameter_mm'])[0]
        centers = [r['center'] for r in meta['records'] if r['case_id'] == case_id and r['target'] == 0][:8]
        if len(centers) != 8:
            raise ValueError('The declared DEBUG cohort needs eight observed sites per case')
        common = dict(full_organ_mask=organ, organ_depth=depth, config=graph_cfg,
                      rng=np.random.default_rng(42), ct_clip=tuple(base['ct_clip']))
        start = time.perf_counter()
        old_source = prepare_local_source(case, source, **common)
        old_source_seconds = time.perf_counter() - start
        start = time.perf_counter()
        new_source = prepare_donor(case, source, organ, depth, base)
        new_source_seconds = time.perf_counter() - start

        def one(index_center):
            index, center = index_center
            began = time.perf_counter()
            built = build_local_graph(case, source, candidate_spec(source, center),
                                      prepared_source=old_source, **common)
            old_graph = build_local_view(built.source_local, built.target_local, graph_cfg, seed=42 + index)
            old_seconds = time.perf_counter() - began
            began = time.perf_counter()
            current = pair_record(case, source, case.spacing, new_source, center, organ, depth,
                                  base, donor_id=case_id)
            new_graph, new_patch, new_target = materialize(current)
            new_seconds = time.perf_counter() - began
            value = dict(case_id=case_id, component=source.component_id, center=list(center),
                old_graph=old_graph, old_source=torch.from_numpy(built.source_patch).half(),
                old_target=torch.from_numpy(built.target_patch).half(),
                new_graph=new_graph, new_source=new_patch.half(), new_target=new_target.half())
            print(json.dumps(dict(stage='pair_prepared', case=case_id, index=index,
                old_nodes=old_graph.num_nodes, new_nodes=new_graph.num_nodes)), flush=True)
            return value, dict(case_id=case_id, index=index, old_seconds=old_seconds, new_seconds=new_seconds)

        with ThreadPoolExecutor(max_workers=4) as pool:
            completed = list(pool.map(one, enumerate(centers)))
        records.extend(r for r, _ in completed)
        timings.append(dict(case_id=case_id, component=source.component_id, diameter_mm=diameter,
            image_sha256=raw['image_sha256'], label_sha256=raw['label_sha256'],
            old_source_seconds=old_source_seconds, new_source_seconds=new_source_seconds,
            pairs=[r for _, r in completed]))
        del case, organ, depth, old_source, new_source, source
        gc.collect()
    torch.save(records, root / 'matched_actual_CT_DEBUG.pt')
    write(root / 'preparation.json', dict(debug=True, cases=timings, base=base, cfg=cfg,
        physical_pair_batches=[8, 16], cpu_workers=4, source='same actual donor and recipient CT sites',
        scope='Native L0 inputs, same-case controlled probe; not the native full training datasets'))


def measure(root, variant, batch_size):
    from hiercp_v222.v1_cache import configuration
    from hiercp_v222.v1_local import V1LocalEncoder, LocalBatch
    from hiercp.model import LocalTumorContextPyGEncoder, HierarchicalPyGPlacementModel
    cfg, base = configuration()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(9e9 / torch.cuda.get_device_properties(0).total_memory)
    records = torch.load(root / 'matched_actual_CT_DEBUG.pt', map_location='cpu', weights_only=False)[:batch_size]
    if len(records) != batch_size:
        raise ValueError('Incomplete physical batch')
    m = base['model']
    full_old = HierarchicalPyGPlacementModel(**m)
    old_total = sum(p.numel() for p in full_old.parameters())
    del full_old
    keys = ('hidden_dim', 'heads', 'dropout', 'dense_base_channels', 'dense_feature_dim',
            'dense_batch_size', 'channels_last_3d', 'checkpoint_local_blocks', 'checkpoint_dense_encoder')
    net = (LocalTumorContextPyGEncoder(**{k: m[k] for k in keys}, layers=m['local_layers'])
           if variant == 'old' else V1LocalEncoder(base)).cuda().train()
    # Identical execution chunking; not a production physical-batch change.
    net.dense_batch_size = batch_size
    lookup, unique, indices = {}, [], []
    for row in records:
        identity = (row['case_id'], row['component'])
        if identity not in lookup:
            lookup[identity] = len(unique)
            unique.append(row[f'{variant}_source'])
        indices.append(lookup[identity])
    cpu_graph = Batch.from_data_list([r[f'{variant}_graph'] for r in records])
    info = dict(debug=True, full_training=False, historical_checkpoint_reproduced=False,
        scope='L0 only, single graph view, untrained weights, squared fused-output probe backward',
        variant=variant, physical_pair_batch=batch_size, unique_donors=len(unique),
        nodes=cpu_graph.num_nodes, edges=cpu_graph.num_edges,
        nodes_by_type={k: cpu_graph[k].num_nodes for k in cpu_graph.node_types},
        parameters=sum(p.numel() for p in net.parameters()), preserved_v1_full_parameters=old_total,
        native_patch_shape=list(records[0][f'{variant}_source'].shape),
        gpu=torch.cuda.get_device_name(), allocator_cap_bytes=9000000000,
        precision='bfloat16 autocast', edge_workspace_mib=64, dense_batch_size=batch_size,
        cpu_threads=8, available_ram=psutil.virtual_memory().available,
        identity=[dict(case_id=r['case_id'], component=r['component'], center=r['center']) for r in records])
    graph = cpu_graph.cuda()
    patches = torch.stack(unique).cuda()
    targets = torch.stack([r[f'{variant}_target'] for r in records]).cuda()
    source_index = torch.tensor(indices, device='cuda')
    batch = LocalBatch(graph, patches, targets, source_index, torch.arange(batch_size, device='cuda'))
    optimizer = torch.optim.AdamW(net.parameters(), lr=m.get('lr', base['training']['lr']), fused=True)
    rows = []
    for step in range(3):
        net.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = (net(graph, patches, source_index, targets)['fused']
                      if variant == 'old' else net(batch))
            loss = result.float().square().mean()
        events[1].record()
        loss.backward()
        events[2].record()
        optimizer.step()
        events[3].record()
        events[3].synchronize()
        row = dict(step=step, warmup=step==0, probe_value=float(loss.detach()),
            forward_seconds=events[0].elapsed_time(events[1])/1000,
            backward_seconds=events[1].elapsed_time(events[2])/1000,
            optimizer_seconds=events[2].elapsed_time(events[3])/1000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Invalid L0 probe gradient')
        row['gradient_tensors'] = len(grads)
        row['parameter_tensors'] = sum(1 for p in net.parameters() if p.requires_grad)
        rows.append(row)
        print(json.dumps(dict(variant=variant, batch=batch_size, **row)), flush=True)
    write(root / f'{variant}_batch{batch_size}.json', dict(**info, steps=rows))


def report(root):
    from hiercp_v222.v1_cache import configuration, provenance, PairDataset
    from hiercp_v222.v1_local import model
    cfg, base = configuration()  # Also verifies preserved source/archive hashes.
    rows = []
    for size in (8, 16):
        pair = [json.loads((root / f'{v}_batch{size}.json').read_text()) for v in ('old', 'new')]
        if pair[0]['identity'] != pair[1]['identity']:
            raise AssertionError('The two encoders did not receive matching CT sites')
        for value in pair:
            steps = value['steps'][1:]
            if any(s['gradient_tensors'] != s['parameter_tensors'] for s in steps):
                raise AssertionError('Missing probe gradients')
            rows.append(dict(variant=value['variant'], physical_pair_batch=size,
                forward_seconds=float(np.mean([s['forward_seconds'] for s in steps])),
                backward_seconds=float(np.mean([s['backward_seconds'] for s in steps])),
                optimizer_seconds=float(np.mean([s['optimizer_seconds'] for s in steps])),
                peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in steps),
                nodes=value['nodes'], edges=value['edges'], parameters=value['parameters']))
    cache = ROOT / 'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json'
    train, val = PairDataset(cache, 'inner_train'), PairDataset(cache, 'inner_val')
    with torch.device('meta'):
        current = model(cfg, base)
    manifest = json.loads((ROOT / 'versions/v1/manifest.json').read_text())
    result = dict(debug=True, matched_CT_sites=True, comparison=rows,
        scope='Preserved v1 vs current L0 native-input cost; no historical trained checkpoint and no full-pipeline timing',
        baseline_revision=manifest['revision'], source_sha256=provenance(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preserved_v1_full_parameters=pair[0]['preserved_v1_full_parameters'],
        current_full_parameters=sum(p.numel() for p in current.parameters()),
        preserved_v1=dict(samples_per_case=base['cache']['samples_per_case'],
            candidates_per_sample=base['cache']['total_candidates'], graph_views_per_candidate=2,
            dense_maps_shared_across_views=True, epoch_support_reencoding=False),
        current=dict(train_pairs=len(train), validation_pairs=len(val),
            train_cases=len(train.meta['split']['inner_train']), validation_cases=len(val.meta['split']['inner_val']),
            comparisons_per_case=cfg['comparison_centers_per_patient'], train_positive_pairs=sum(r['target']==1 for r in train.rows),
            validation_positive_pairs=sum(r['target']==1 for r in val.rows),
            l0_forward_pairs_per_regular_epoch=2*len(train)+len(val),
            l0_backward_pairs_per_epoch=len(train),
            extra_initial_and_final_memory_pairs=2*len(train)),
        limits=['Two cases and sixteen matched same-case sites; native inputs, no graph-size reduction',
            'Three L0 probe updates per process; last two CUDA-event times averaged',
            'Squared-norm probe is neither native full-model loss; no accuracy conclusion',
            'Single graph view cost; original v1 full pipeline shares CNN maps across two graph views',
            'RTX5070Ti with allocator cap is not A100 MIG speed emulation',
            'No original server cache, checkpoint or epoch timing artifact available locally',
            'Recovered older code dump uses stock GATv2, so preserved snapshot cannot establish historic wall time',
            'No server process or production configuration changed'])
    write(root / 'comparison.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'measure', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['old', 'new'])
    parser.add_argument('--batch', type=int, choices=[8, 16])
    args = parser.parse_args()
    if args.mode == 'prepare':
        args.output.mkdir(parents=True, exist_ok=False)
        prepare(args.output)
    elif args.mode == 'measure':
        if args.variant is None or args.batch is None:
            parser.error('measure requires --variant and --batch')
        measure(args.output, args.variant, args.batch)
    else:
        report(args.output)


if __name__ == '__main__':
    main()
