# Level 0/1/2 and online placement curriculum — new experiment

This revision changes the learned ranking model and geometry. It is **not** a
patch that can be applied to old fold-0 scores without retraining. Keep existing
Dataset730/v2 banks, checkpoints, predictions and evaluations unchanged. Use new
work/result roots and a separately chosen, unused nnU-Net dataset ID. Do not
pool old and new results as identical-method cross-validation.

## Patient boundaries and labels

Training-patient tumor masks legitimately supervise a placement pretext task.
Outer held-out masks must never participate in GNN fitting, GNN checkpoint
selection, prototype fitting, CP donor/recipient pools, or fitted preprocessing
statistics. Inner GNN validation remains inside the outer training patients.
The new bank publisher verifies these nested cohorts and binds the actual
artifacts with SHA-256. The installed trainer additionally checks the live
nnU-Net split and actual dataset keys. Patient IDs are the repository's case
IDs; a dataset with multiple IDs per person needs an explicit patient grouping
before constructing splits.

Online fingerprint extraction and planning now see a retained, train-only raw
view. The full raw dataset is used only for preprocessing with those fixed
plans. Existing preprocessing without this contract is rejected. The old
globally fitted offline CP/five-fold training route cannot demonstrate this
boundary and its augmented-training entry point now fails explicitly.

No healthy location receives an invented positive label merely because its
descriptor looks similar. This is still a two-stage placement-ranking then
segmentation-training pipeline, not an end-to-end differentiable Dice objective.

## Geometry and model

- L0's 384-node setting is a **seed budget**, not a truncation after each hop.
  Required interface nodes and the complete configured-hop closure are retained.
  Node/edge/ROI resource guards fail explicitly rather than shrinking graphs.
- Physical-mm transforms, integer paste anchors and enlarged odd bounding boxes
  agree on anisotropic scans. Empty transformed masks fail instead of silently
  returning identity. Liver anchors must come from the actual surface band.
- L1's final region/lesion/liver representations feed candidate-conditioned
  readout; they are no longer unused final-layer outputs. Source-tumor positional
  and same-region edge shortcuts are masked by `shortcut_safe_upper_v2`.
- L2 retains train-only population **liver-context** clusters and now queries
  their learned representations conditional on the source tumor and available
  patient/local context. It is not a cancer-type clustering/classification model.
- GATv2 neighborhood attention is multiplied by a learned sigmoid compatibility
  gate, so single-neighbor relations can learn source/target/edge compatibility.
  Actual ablations omit their unused modules and input columns.

Versions: `level0_physical_closure_v2` and
`hiercp_conditioned_readout_v3`. Old graph caches and model states are rejected.
The configured scale remains hidden 128, 4 heads, 3/2/2 graph blocks, 48³ patches,
CNN channels 12/24/32, 24 patient regions, 16 prototypes, 8 training candidates,
128 placement candidates, 40 GNN epochs and 250 nnU-Net epochs.

## Physical minibatches and changed optimization contract

Variable-sized graphs use disjoint-union PyG batches; losses use masked B×C
tensors for CE, pairwise, ordinal and percentile mining. Physical-batch and
candidate-chunk equivalence are checked independently.

The old effective-batch-2 constraint is **not retained in the new experiment**.
The new explicit policy measures powers of two through the full training cohort,
uses the actual complete ranking loss/backward/AdamW memory (zero learning rate
only during calibration), and selects measured throughput with VRAM headroom.
Gradient accumulation is 1; effective batch equals the selected physical batch
on the supported single-device path. This changes optimization steps per epoch,
so the measured batch and step count belong to the new experiment identity.
No final medical-data batch/worker optimum has been measured on this Windows host.
Scoring probes explicitly include 1/2/4/8/16 source samples. GPU allocations and
full-graph memory still require target-server measurement. Multi-GPU training is
not implemented; unsupported multi-device paths fail explicitly.

## Online CP selection

`config/online_cp_curriculum.json` is an explicit, initially unvalidated policy:
five 50-epoch stages expand rank bands 1 → 4 → 8 → 16 → 32. Uniform selection
within the band is bounded by a maximum score drop of 2.0; this is a learned-score
gate, **not** a guarantee of anatomical correctness or a calibrated probability.
Every entry must satisfy each stage's minimum-choice requirement before publication.
Anatomical candidate gates, 128 candidates, CP probability 0.5 and the original
event/source/appearance RNG draws remain unchanged. Basic control samples the
whole valid pool; event and actual-choice digests are distinct.

See [curriculum integration](online_cp_curriculum.md) for trainer names, resume
contracts and limitations. The legacy `onlinecp all/train` command still names
legacy trainers; it does **not** implicitly opt into the new curriculum classes.
Use the new classes explicitly and never label a legacy launch as curriculum.

## Verification and remaining target checks

Local tests use clearly marked analytic or synthetic debug fixtures. They do not
produce medical training checkpoints or performance estimates. A separate CPU
smoke used the full configured model and 48³ tensors with physical batch 2 and
variable candidate counts 4/3: 10,444,516 trainable parameters, all active gradients
finite/present, and actual optimizer updates in every level/readout/head. Its
small graph fixture is not a real full-organ graph or a clinical validation.

Remaining: real-data whole-graph resource calibration, complete 40/250-epoch
training, installed-server checkpoint replay, full downstream evaluation, and
multi-fold efficacy. In particular, passing these tests does not establish that
L2 or curriculum improves Dice/recall. Inspect the final turn report for the
latest test totals and installed-runtime checks.
