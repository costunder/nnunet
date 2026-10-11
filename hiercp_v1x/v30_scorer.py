"""Batched v3.0 scoring with complete patient vessel context.

Only static, verified CPU vessel tensors are cached. Learned embeddings are
recomputed in the current forward so every genuine optimizer update reaches
the vascular branch. Local chunks still use the original checkpoint/prefetch.
"""
from __future__ import annotations

import time
import torch

from .v24_training import V24Scorer
from .v30_model import vessel_score_context
from .v30_vessels import RecipientBinding, voxel_to_ras_mm


class V30Scorer(V24Scorer):
    def __init__(self, net, providers, geometry, *, vessel_store, recipient_bindings, **kwargs):
        super().__init__(net, providers, geometry, **kwargs)
        self.vessel_store = vessel_store
        self.recipient_bindings = {case_id: RecipientBinding.from_mapping(binding)
                                   for case_id, binding in recipient_bindings.items()}
        if any(case_id != binding.case_id for case_id, binding in self.recipient_bindings.items()):
            raise ValueError('Vessel recipient binding key and actual CT case identity differ')

    def forward(self, plans, *, epoch, training):
        if not plans or len({plan.case_id for plan in plans}) != len(plans):
            raise ValueError('Unique real patients required for disjoint v3.0 scoring')
        started = time.perf_counter()
        case_ids = tuple(plan.case_id for plan in plans)
        bindings = {case: self.recipient_bindings[case] for case in case_ids}
        cpu = self.vessel_store.batch(case_ids, recipient_bindings=bindings)
        device = next(self.net.parameters()).device
        if device.type != 'cuda':
            raise RuntimeError('Actual v3.0 scorer requires CUDA; no CPU fallback')
        if self.pin_cpu_batches:
            cpu = cpu.pin_memory()
        batch = cpu.to(device, non_blocking=True)
        positions, owners = [], []
        for patient, plan in enumerate(plans):
            xyz = torch.tensor([row['center'] for row in plan.query_rows], dtype=torch.float32, device=device)
            positions.append(voxel_to_ras_mm(xyz, bindings[plan.case_id].affine_voxel_to_ras_mm))
            owners.append(torch.full((len(plan.query_rows),), patient, dtype=torch.long, device=device))
        positions = torch.cat(positions)
        owners = torch.cat(owners)
        prepared = time.perf_counter() - started
        with vessel_score_context(self.net, batch, positions, owners):
            result = super().forward(plans, epoch=epoch, training=training)
        result.workload.update(version='v3.0', vessel_nodes=int(batch.node_features.shape[0]),
            vessel_directed_edges=int(batch.edge_index.shape[1]),
            vessel_components=int(batch.component_patient_index.numel()),
            vessel_patient_encoder_invocations=1,
            vessel_input_prepare_enqueue_seconds=prepared,
            vessel_context='complete same-patient nodes/components; source-conditioned physical-coordinate attention',
            vascular_subsampling=False, vascular_flow_or_type_labels=False,
            context_policy='0<tumor distance<=10mm; liver depth>1mm; same train/validation/CP')
        return result
