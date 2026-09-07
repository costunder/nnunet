# Full-edge training execution repair

This change repairs execution memory handling. It does not change the scientific
graph: canonical nodes, context selection, required interface nodes, hop closure,
relation radii, candidates, model layers/widths, and physical graph batching remain
unchanged. Existing canonical cache files are not rewritten.

## Why preparation could finish but training could fail

Canonical preparation and training-view materialization are different operations.
The legacy `sample_relation_edge_limit=500000` check rejected an induced relation
before its first model forward, although its canonical cache was valid. The
reported source-context relation contained 1,499,368 legitimate edges. This was
not an actual CUDA OOM measurement, a missing CP candidate, or evidence that those
edges were invalid.

The cached limit is retained as historical metadata. Runtime materialization
records exceedance diagnostically and constructs all induced edges/attributes in
bounded scratch chunks. It does not clip the relation, substitute nodes, redraw
the selected tumor, or discard the sample. Malformed geometry, topology and
artifacts still fail explicitly.

## Exact attention with bounded edge-hidden work

`CompatibilityGatedGATv2Conv.forward` is the shared production path for every
active hierarchy relation, including quality training. The execution identity is
`hiercp_exact_edge_streaming_v1`; no trainable parameter, state-dictionary key or
model-constructor configuration was added.

For every relation, the implementation:

1. Reuses the existing source/destination node projections and edge projection.
2. Computes every edge's GATv2 logit in edge-workspace chunks.
3. Concatenates the scalar `[edges, heads]` logits and performs one global PyG
   destination softmax. Neighbors in different chunks compete in the same
   denominator. The sigmoid compatibility gate is unchanged.
4. Applies dropout once to the complete scalar attention tensor in original
   edge order, then adds every edge message into one destination accumulator.
5. Allocates that full destination output only once, not once per edge chunk,
   before the existing bias, residual, head merge and downstream hierarchy.

The execution workspace is 64 MiB divided by eight FP32 edge-hidden temporaries
per edge. With 128 hidden channels this gives 16,384 edges per workspace chunk.
It is a scratch-allocation policy, **not an edge-count or graph-size cap**: the
final chunk and every preceding chunk are processed. The workspace calculation
uses eight-byte scalars for explicit FP64 numerical DEBUG checks.

Node/edge linear projections retain their existing autocast behavior. Production
logit arithmetic, destination softmax and compatibility gating use FP32;
attention is cast back to the message dtype as before. FP64 is supported for
numerical derivative testing. Changing the reduction grouping may cause ordinary
floating-point rounding differences; bitwise GPU equivalence is not claimed.

Non-reentrant checkpointing wraps edge-logit chunks when gradients are enabled.
The exact streamed message-sum autograd operator saves only node projections,
COO indices and scalar attention. Its backward computes source-feature and
attention gradients in chunks, with one source-gradient accumulator. No message
chunk is saved for backward, and no list of per-chunk full destination-node
outputs is created. Accumulators use FP32 (FP64 for numerical DEBUG checks),
then return the original message/gradient dtype. Existing local-block and
dense-CNN checkpointing can remain enabled. RNG preservation supports nested
replay; dropout is outside the edge chunk computations.

This is **not constant total model memory**. Complete COO indices, ten-channel
edge attributes, scalar `[edges, heads]` attention, node activations, CNN work,
parameters and optimizer buffers still consume memory. A 64 MiB scratch policy
is not a promise of a 64 MiB CUDA peak or proof that an arbitrary graph fits.

## Training resource calibration and lifecycle

The training execution identity records the attention execution version and
workspace policy. Calibration uses full-cohort canonical input upper bounds
instead of assuming serialized file size is the induced-view memory cost.
RAM-aware checks cover complete input materialization, physical batching and
worker/prefetch pressure; they do not change graph or model scale.

Physical-batch trials measure end-to-end steps, use different deterministic view
epochs, retain Adam buffers between measured steps, and take the maximum memory
peak across all repeats rather than only the final repeat. CPU-side preparation
and transfer are included in the timing. A batch is accepted only under the
configured measured memory-headroom rule; an OOM remains an explicit failed
trial, not permission to shrink nodes, edges, model depth or data.

The epoch loop releases output/loss/batch references after their metrics are
consumed, so the previous output does not retain two old graph views during the
next model forward. CUDA prefetch remains a separately reported input-memory
choice. Runtime calibration is still required on the actual assigned server.

## Verification and remaining limits

The focused CPU-only DEBUG operator suite covers:

- Dense-reference forward and every input/parameter gradient across chunks,
  including destinations whose edges span multiple chunks.
- Dropout mask/gradient equivalence, empty relations and zero-gradient parameter
  connectivity, shared projections, self-loops and mean-head output.
- FP64 numerical gradient checking and finite CPU-autocast backward.
- Forward/backward workspace bounds, total saved edge-hidden tensor inspection,
  and exactly one full destination-output allocation across message chunks.
- A complete 3/2/2-level, physical two-patient DEBUG hierarchy with nested
  checkpoints: repeatable dropout outputs/gradients and optimizer updates in
  local, patient, population and score modules.

The existing hierarchy DEBUG suite also completed: seven passed and one explicit
production-sized opt-in test was skipped. These synthetic fixtures are not
medical-data performance results. They do not certify CUDA peak memory,
production throughput, complete quality-model training or nnU-Net training.
Those require the real server allocation and the preserved full-data run.
