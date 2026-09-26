"""DEBUG candidate: CSR message sums for evaluation, never installed in training.

All edges and per-head learned attention weights are retained. Duplicate edges
are summed explicitly. Topology is cached only within one forward, and scalar
attention is rebuilt for every layer. FP32 summation order differs from COO.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import hiercp.model as implementation


class SparseEvaluation:
    def __init__(self):
        self.topologies = {}

    def apply(self, features, edges, attention, destination_count, chunk_size):
        if torch.is_grad_enabled():
            raise RuntimeError('DEBUG sparse candidate is evaluation-only')
        count, heads, channels = features.shape
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        key = (id(edges), edges._version, count, destination_count)
        if key not in self.topologies:
            # A sorted unique coordinate per CSR value also covers multigraphs.
            coordinate = edges[1] * count + edges[0]
            ordered, permutation = torch.sort(coordinate, stable=True)
            unique, inverse = torch.unique_consecutive(ordered, return_inverse=True)
            rows = torch.div(unique, count, rounding_mode='floor')
            columns = unique.remainder(count)
            pointers = torch.cat((torch.zeros(1, device=edges.device, dtype=torch.long),
                torch.bincount(rows, minlength=destination_count).cumsum(0)))
            self.topologies[key] = (edges, permutation, inverse, pointers, columns)
        _, permutation, inverse, pointers, columns = self.topologies[key]
        weights = torch.zeros((columns.numel(), heads), dtype=dtype, device=features.device)
        weights.index_add_(0, inverse, attention[permutation].to(dtype))
        # sparse.mm must remain FP32/FP64 even inside the caller's BF16 context.
        with torch.autocast(features.device.type, enabled=False):
            outputs = []
            for head in range(heads):
                matrix = torch.sparse_csr_tensor(pointers, columns, weights[:, head].contiguous(),
                    size=(destination_count, count), device=features.device)
                outputs.append(torch.sparse.mm(matrix, features[:, head].to(dtype).contiguous()))
            return torch.stack(outputs, dim=1).to(features.dtype)


@contextmanager
def installed():
    """Caller scopes one complete no-grad L0 forward; no cross-batch retention."""
    operator = SparseEvaluation()
    with patch.object(implementation, '_StreamedEdgeAggregation', operator):
        yield operator
