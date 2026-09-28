"""L0 execution candidate: direct streamed derivative of attribute-free GAT logits.

Same edge chunks, FP32 forward, BF16 cast boundaries and per-chunk node-gradient
accumulation as the existing operator. Only checkpoint bookkeeping/replay for
the deterministic logit expression is replaced; outer L0/CNN checkpoints stay.
No tensor of all edge-by-hidden activations is retained.
"""
from contextlib import contextmanager, ExitStack
from types import MethodType, FunctionType
from unittest.mock import patch
from functools import lru_cache
import torch
from torch.nn import functional as F


@lru_cache(maxsize=1)
def kernels():
    from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
    forward = _create_jit_fn('''template <typename T>
    T l0_logit_product(T x, T y, T a, T slope) {
        T joint = y + x;
        T active = joint > T(0) ? joint : joint * slope;
        return active * a;
    }''', slope=0.2)
    backward = _create_multi_output_jit_fn('''template <typename T>
    void l0_logit_derivative(T x, T y, T a, T g, T slope, T& dx, T& da) {
        T joint = y + x;
        T active = joint > T(0) ? joint : joint * slope;
        T product = g * a;
        dx = joint > T(0) ? product : product * slope;
        da = g * active;
    }''', num_outputs=2, slope=0.2)
    return forward, backward


class EdgeLogit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, edges, attention, slope):
        dtype = torch.float64 if left.dtype == torch.float64 else torch.float32
        joint = right[edges[1]].to(dtype) + left[edges[0]].to(dtype)
        ctx.save_for_backward(left, right, edges, attention)
        ctx.slope = slope
        ctx.fused = False
        return (F.leaky_relu(joint, slope) * attention.to(dtype)).sum(-1)

    @staticmethod
    def backward(ctx, grad_output):
        left, right, edges, attention = ctx.saved_tensors
        dtype = torch.float64 if left.dtype == torch.float64 else torch.float32
        source, destination = edges
        g = grad_output.unsqueeze(-1)
        if ctx.fused:
            grad_joint, partial_attention = kernels()[1](
                left[source], right[destination], attention.to(dtype), g, slope=ctx.slope)
            grad_attention = partial_attention.sum(0, keepdim=True).to(attention.dtype) if ctx.needs_input_grad[3] else None
        else:
            joint = right[destination].to(dtype) + left[source].to(dtype)
            grad_attention = ((g * F.leaky_relu(joint, ctx.slope)).sum(0, keepdim=True).to(attention.dtype)
                              if ctx.needs_input_grad[3] else None)
            grad_joint = torch.ops.aten.leaky_relu_backward(g * attention.to(dtype), joint, ctx.slope, False)
        grad_left = grad_right = None
        if ctx.needs_input_grad[0]:
            grad_left = torch.zeros_like(left)
            grad_left.index_put_((source,), grad_joint.to(left.dtype), accumulate=True)
        if ctx.needs_input_grad[1]:
            grad_right = torch.zeros_like(right)
            grad_right.index_put_((destination,), grad_joint.to(right.dtype), accumulate=True)
        return grad_left, grad_right, None, grad_attention, None


class FusedEdgeLogit(EdgeLogit):
    @staticmethod
    def forward(ctx, left, right, edges, attention, slope):
        if left.device.type != 'cuda':
            raise RuntimeError('Fused L0 requires CUDA/NVRTC; no fallback')
        dtype = torch.float64 if left.dtype == torch.float64 else torch.float32
        ctx.save_for_backward(left, right, edges, attention)
        ctx.slope, ctx.fused = slope, True
        return kernels()[0](left[edges[0]], right[edges[1]], attention.to(dtype), slope=slope).sum(-1)


class NodeCastAggregation(torch.autograd.Function):
    """Cast each node once, rather than its repeated copy on every incident edge."""
    @staticmethod
    def forward(ctx, features, edges, attention, destination_count, chunk_size):
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        nodes = features.to(dtype)
        output = torch.zeros((destination_count, *features.shape[1:]), dtype=dtype, device=features.device)
        for start in range(0, int(edges.shape[1]), chunk_size):
            stop = start + chunk_size
            messages = nodes[edges[0, start:stop]] * attention[start:stop].to(dtype).unsqueeze(-1)
            output.index_add_(0, edges[1, start:stop], messages)
        ctx.save_for_backward(features, edges, attention)
        ctx.chunk_size = chunk_size
        return output.to(features.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        features, edges, attention = ctx.saved_tensors
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        nodes, gradients = features.to(dtype), grad_output.to(dtype)
        grad_features = torch.zeros_like(nodes) if ctx.needs_input_grad[0] else None
        grad_attention = torch.empty_like(attention) if ctx.needs_input_grad[2] else None
        for start in range(0, int(edges.shape[1]), ctx.chunk_size):
            stop = start + ctx.chunk_size
            source, destination = edges[:, start:stop]
            selected = gradients[destination]
            if grad_features is not None:
                grad_features.index_add_(0, source, selected * attention[start:stop].to(dtype).unsqueeze(-1))
            if grad_attention is not None:
                grad_attention[start:stop] = (selected * nodes[source]).sum(-1).to(attention.dtype)
        return (grad_features.to(features.dtype) if grad_features is not None else None,
                None, grad_attention, None, None)


@contextmanager
def installed(local, *, fused=False, node_cast=False):
    from hiercp.model import CompatibilityGatedGATv2Conv
    modules = [m for m in local.modules() if isinstance(m, CompatibilityGatedGATv2Conv)]
    if not modules or any(m.lin_edge is not None for m in modules):
        raise ValueError('Explicit CT-only L0 with no edge attributes required')
    def run(self, function, left, right, edges, attributes):
        if attributes is not None:
            raise ValueError('Unexpected L0 edge attributes')
        operator = FusedEdgeLogit if fused else EdgeLogit
        return operator.apply(left, right, edges, self.att, self.negative_slope)
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch.object(module, '_recompute_edge_chunk', MethodType(run, module)))
            if node_cast:
                # Bind the unchanged forward code to this L0 instance with its
                # own operator namespace. Never replace the module-global class:
                # support L1/L2 use that class in the same model invocation.
                original = module.forward.__func__
                namespace = dict(original.__globals__, _StreamedEdgeAggregation=NodeCastAggregation)
                forward = FunctionType(original.__code__, namespace, original.__name__,
                                       original.__defaults__, original.__closure__)
                forward.__kwdefaults__ = original.__kwdefaults__
                stack.enter_context(patch.object(module, 'forward', MethodType(forward, module)))
        yield
