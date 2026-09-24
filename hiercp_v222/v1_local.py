"""V1 local context comparison with the later explicit CT-only exclusions.

Reuses the preserved radius-graph construction/sampling and CTOnlyEncoder.
No target erasure, handcrafted node/edge inputs, tumor-interior nodes, PPR,
A*, giant ResEnc encoder, or change to the L1/L2 objective is introduced.
Graph geometry still uses the known donor footprint and liver annotation;
CT-only describes the learned features, not an annotation-free graph builder.
"""
from dataclasses import dataclass, replace
from copy import copy
import numpy as np
import torch
from torch_geometric.data import Batch
from hiercp.common import stable_case_seed
from hiercp.schema import graph_config_from_dict
from hiercp_v22 import geometry as geometry
from hiercp_v22.data import candidate_spec, donor_in_target_spacing
from hiercp_v22.local import CTOnlyEncoder
from hiercp_v22.sample import build_local_view
from hiercp_v22.schema import LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES
from hiercp_v22.spatial import target_node_specifications, LEVEL0_GEOMETRY_CONTRACT

ENCODER_ID = 'v1_ct_only_context_pair'
RECORD_FORMAT = 'v222_v1_local_raw_ct_pair_v1'


class V1LocalEncoder(CTOnlyEncoder):
    """Return the existing fused 128-D pair representation to unchanged L1/L2."""
    def __init__(self, base):
        m = base['model']
        keys = ('hidden_dim', 'heads', 'dropout', 'checkpoint_local_blocks',
                'dense_base_channels', 'dense_feature_dim', 'dense_batch_size',
                'channels_last_3d', 'checkpoint_dense_encoder')
        if (m['hidden_dim'], m['heads'], m['local_layers'],
                m['dense_base_channels'], m['dense_feature_dim']) != (128, 4, 3, 12, 32):
            raise ValueError('V1 local depth/width must remain 3 / 128 / 4 and CNN 12/24/32')
        super().__init__(**{k: m[k] for k in keys}, layers=m['local_layers'])

    def forward_fields(self, batch):
        return super().forward(batch)

    @staticmethod
    def _sample_dense_features(feature_map, grid, node_batch):
        from .deterministic_sampling import sample_nodes
        return sample_nodes(feature_map, grid, node_batch)

    def forward(self, batch):
        if not isinstance(batch, LocalBatch):
            raise TypeError('V1 L0 requires paired CT and sampled physical graphs, not a CT tensor alone')
        return self.forward_fields(batch)['fused']


@dataclass
class LocalBatch:
    graph: Batch
    source_patches: torch.Tensor
    target_patches: torch.Tensor
    source_index: torch.Tensor
    indices: torch.Tensor

    def __len__(self):
        return int(self.target_patches.shape[0])

    def to(self, device, non_blocking=True):
        # Copy stores, not tensor storage: Tensor.clone() loses host pinning.
        # PyG's shallow copy owns separate stores, so .to replaces only its refs.
        return LocalBatch(copy(self.graph).to(device, non_blocking=non_blocking),
            *[getattr(self, k).to(device, non_blocking=non_blocking)
              for k in ('source_patches', 'target_patches', 'source_index', 'indices')])

    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)

    def pin_memory(self):
        return LocalBatch(self.graph.pin_memory(),
            *[getattr(self, k).pin_memory()
              for k in ('source_patches', 'target_patches', 'source_index', 'indices')])


def prepare_donor(case, source, organ, depth, base):
    return geometry.prepare_local_source(case, source, full_organ_mask=organ,
        organ_depth=depth, config=graph_config_from_dict(base['graph']),
        rng=np.random.default_rng(base['seed']), ct_clip=tuple(base['ct_clip']))


def pair_record(target, source, donor_spacing, prepared, center, organ, depth, base, *, donor_id):
    """Build from actual donor + recipient; query tumor annotation is not a shape input.

    Source geometry is computed in donor mm. Only its footprint is regridded
    for the recipient, matching the CP transport convention. The target CT
    remains unmodified, including the candidate center.
    """
    gc = graph_config_from_dict(base['graph'])
    geometry._require_full_graph(gc)
    target_source, _ = donor_in_target_spacing(source, donor_spacing, target.spacing)
    prepared = replace(prepared, source_footprint=geometry.exact_source_footprint(target_source))
    spec = candidate_spec(target_source, center)
    footprint, transform = geometry._transform_footprint(
        prepared.source_footprint, spec, spacing=target.spacing, config=gc)
    fields = geometry._patch_fields(target, spec.center, footprint, organ, depth,
        config=gc, erase_target=False, ct_clip=tuple(base['ct_clip']))
    # Genuine absence of deep parenchyma is not absence of the entire graph:
    # real liver-surface CT nodes remain mandatory. Existing empty-shell tokens
    # handle that observed absence; no node, mask or feature is fabricated.
    coordinates = geometry.canonical_coordinate_sets(fields, gc, target.spacing,allow_empty_context=True)
    nodes = geometry._pack_nodes(fields, target_node_specifications(coordinates),
                                 target, gc, branch_flag=-1.)
    edges = geometry._canonical_edges(
        {**prepared.canonical_nodes, **nodes},
        [e for e in LOCAL_EDGE_TYPES if e not in prepared.canonical_edges], gc, transform=transform)
    branch = lambda n, e: dict(format='canonical-full-v22', geometry_contract=LEVEL0_GEOMETRY_CONTRACT,
        nodes=n, edges=e, counts={k: len(v['grid']) for k, v in n.items()},
        edge_counts={k: v.shape[1] for k, v in e.items()})
    source_branch = branch(prepared.canonical_nodes, prepared.canonical_edges)
    source_branch['footprint_voxels'] = int(source.voxel_count)
    target_branch = branch(nodes, edges)
    target_branch['transform'] = torch.from_numpy(transform)
    return dict(format=RECORD_FORMAT, case_id=target.paths.case_id,
        donor_case_id=donor_id, component_id=int(source.component_id), center=list(map(int, center)),
        seed=base['seed'], graph_config=gc.to_dict(), center_masking=False,
        target_context_observed_absent=(len(nodes['target_context']['grid'])==0),
        source_patch=torch.from_numpy(prepared.source_patch),
        target_patch=torch.from_numpy(fields.model_input),
        source_local=source_branch, target_local=target_branch)


def materialize(record, *, epoch=0, view=0):
    if record.get('format') != RECORD_FORMAT or record.get('center_masking') is not False:
        raise ValueError('Explicit raw-CT paired record required; old erased caches cannot be relabeled')
    identity = f"{record['donor_case_id']}:{record['component_id']}:{record['center']}:{view}:{epoch}"
    seed = stable_case_seed(record['seed'], record['case_id'], identity)
    graph = build_local_view(record['source_local'], record['target_local'],
                            graph_config_from_dict(record['graph_config']), seed=seed,
                            allow_empty_target_context=record.get('target_context_observed_absent',False))
    if set(graph.node_types) != set(LOCAL_NODE_TYPES):
        raise ValueError('Retired/unknown graph node type')
    for kind in graph.node_types:
        del graph[kind].pos_mm
    for key in ('source_patch', 'target_patch'):
        patch = record[key]
        if patch.shape != (1, 48, 48, 48) or not torch.isfinite(patch).all():
            raise ValueError('Finite CT-only 48-cubed patches required')
    return graph, record['source_patch'], record['target_patch']


def collate(items):
    payloads, indices = zip(*items)
    graphs, sources, targets = zip(*payloads)
    # Shared immutable source tensors from prepare_donor are encoded once.
    unique, lookup, ids = [], {}, []
    for source in sources:
        key = (source.data_ptr(), tuple(source.shape), source.dtype)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(source)
        ids.append(lookup[key])
    return LocalBatch(Batch.from_data_list(list(graphs)), torch.stack(unique), torch.stack(targets),
                      torch.tensor(ids), torch.tensor(indices))


def model(cfg, base):
    from .model import PromptGraphModel
    if cfg.get('local_encoder') != ENCODER_ID:
        raise ValueError('Explicit v1 local encoder configuration required')
    return PromptGraphModel(cfg, base, {})


def support_for_recipient(memory, query_group):
    """Exclude query identity on BOTH sides of a paired context observation."""
    groups = memory['patient_groups']
    donor_groups = memory.get('donor_groups')
    if donor_groups is None or len(donor_groups) != len(memory['embeddings']):
        raise ValueError('Paired support requires a donor patient group for every record')
    owners = memory['owners']
    keep = torch.tensor([group != query_group for group in donor_groups],
                        device=owners.device, dtype=torch.bool)
    allowed = torch.tensor([g != query_group for g in groups], device=owners.device)
    keep = keep & allowed[owners]
    used = torch.unique(owners[keep], sorted=True)
    if len(used) < 2:
        raise ValueError('Two other recipient groups required after recipient/donor exclusion')
    mapping = torch.full((len(groups),), -1, device=owners.device, dtype=torch.long)
    mapping[used] = torch.arange(len(used), device=owners.device)
    return memory['embeddings'][keep], mapping[owners[keep]], memory['classes'][keep]


def observation_loss(network, batch, targets, memory, query_group, *, class_weights=None):
    """Actual unchanged v2.22 objective with paired-input leakage exclusion."""
    from .model import supervised_loss
    support = support_for_recipient(memory, query_group)
    output = network(batch, *support)
    return supervised_loss(output, targets, class_weights), output


def score_candidates(network, records, memory, *, query_group, batch_size):
    """All 128 admitted candidates use this L0 and the unchanged L1/L2 scorer.

    The caller supplies training-only support with per-record donor provenance.
    This function does not label unobserved candidates or update model weights.
    """
    if network.training or len(records) != 128 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Eval model, all 128 candidates and measured positive batch size required')
    if len({r['case_id'] for r in records}) != 1 or len({tuple(r['center']) for r in records}) != 128:
        raise ValueError('All 128 distinct candidate sites must belong to the same recipient')
    support = support_for_recipient(memory, query_group)
    device = next(network.parameters()).device
    with torch.no_grad():
        state = network.prepare_support(*support)
        values = []
        for start in range(0, len(records), batch_size):
            batch = collate([(materialize(r), i) for i, r in enumerate(records[start:start+batch_size], start)]).to(device)
            values.append(network.predict_embeddings(network.local(batch), state)['ranking_score'])
        scores = torch.cat(values)
    if scores.shape != (128,) or not torch.isfinite(scores).all():
        raise FloatingPointError('Invalid complete CP ranking')
    return scores, scores.argmax()
