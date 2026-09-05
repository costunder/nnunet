# Train-only nnU-Net difficulty feedback: separate experiment

This implementation is **not** the earlier expanding GNN rank band. Lower
compatibility rank is not segmentation difficulty. Existing exact-argmax and
rank-band results/checkpoints retain their original meaning and are not upgraded.

## Connected training path

1. The immutable bank retains its full 128 geometry-valid candidate centers and
   original GNN compatibility scores. A fixed, explicit score gate restricts the
   Full arm's support. This learned-score gate is not a calibrated anatomical
   safety probability. Basic still samples uniformly over all 128 candidates.
2. A frozen epoch snapshot selects a placement using measured difficulty, a
   separately trained difficulty GNN's estimates, and explicit exploration.
   Unobserved difficulty is missing information, never a zero or random label.
3. The normal nnU-Net minibatch is pasted and augmented. Pasted-support and valid
   image support follow the same spatial transformation. All ordinary
   deep-supervision targets and the segmentation objective are preserved.
4. An observer on the actual segmentation loss reads the highest-resolution
   predictions **before** the ordinary optimizer update. It does not perform
   another segmentation forward or replace the normal loss/backward.
5. Available training observations update an EMA table at the epoch boundary.
   A separate, full-sized GNN learns to predict those observed errors using the
   complete L0/L1/L2 graph path and a difficulty output. The original compatibility
   model and stored quality scores are not updated by this objective.
6. Complete candidate pools are rescored by the trained difficulty GNN. The
   following epoch's workers receive an immutable new snapshot. Network progress,
   observations, GNN state, table and snapshot identities are audited and saved.

`predictions.mode="optional"` allows missing predictions before the difficulty
GNN has learned from usable observations; it does not disable the GNN in Full.
The Full trainer rejects `disabled` rather than silently running a different arm.
If a completed epoch has no usable observations, nnU-Net still keeps its normal
update, while the GNN retains its last real training lineage and supplies no new
prediction. The current student checkpoint identity is saved separately.

## What is measured

The attributed lesion is the transformed pasted support intersected with the
authoritative transformed tumor target. Segmentation interpolation can resolve
binary support and three-class labels differently. Removed support is counted;
the code does not relabel the target or silently assume original voxel identity.
Padding validity follows the same transform with zero outside-image support.

Each component has its own voxel-count normalization: bounded foreground tumor
cross-entropy (`1-exp(-mean CE)`), inner-boundary tumor probability error, and
tumor probability in adjacent valid non-tumor tissue. The initial component
weights are explicit in the policy JSON. Neighborhood width is one preprocessed
voxel under measurement version `onlinecp_surviving_lesion_feedback_v1`.
Unrelated tumors are not counted as the pasted lesion or as background errors.

No-CP, erased support, possible crop/padding truncation and empty valid adjacent
regions produce explicit unavailable statuses, not fake easy observations. The
nnU-Net batch is still trained normally and the exclusions are reported. This
conservative eligibility can bias which placements provide feedback; inspect the
status/visibility counts before interpreting downstream performance.

The GNN predicts placement-conditional observed difficulty across stochastic
appearance/crop/augmentation, not the exact error of every future augmented
image. Both the student and difficulty estimates change over time. High error is
not proof of useful augmentation: real-data validation is still necessary.

## Data and resource boundaries

Only the verified outer-training patients may supply CP inputs, measurements or
difficulty-GNN targets. No downstream validation loss, Dice, predictions or test
labels enter the feedback state. The initial prototype/quality model retains its
stricter inner split. Difficulty fitting on the outer-training cohort does not
turn its original inner validation into an independent final evaluation set.

The native 3-D non-cascaded 0/1/2 liver path is explicit. Unsupported regional
labels, ignore-label/cascade paths or dummy-2D augmentation are rejected, not
silently converted to another pipeline. The implementation remains single-device;
it does not claim verified DDP replay. CPU debug runs do not select a server GPU
batch or justify reducing the model.

Full graph/model scale and the ordinary nnU-Net physical batch are preserved.
Difficulty-GNN batching is measured with nnU-Net already resident on the actual
device. This adds real graph construction, training and full-pool scoring cost;
the implementation must log its measured memory and throughput, not claim the
feedback is free. GNN randomness is isolated from segmentation augmentation and
network RNG. Graph caches live in the new trainer's own result directory and are
bound to the actual bank/raw/checkpoint identity.

## Publication and launch

Run from this repository in the intended nnU-Net environment, with the `hiercp`
package importable. First inspect `python custom_trainers/install_onlinecp_custom_trainers.py check`;
the installer audits all eight trainer/helper hashes. `apply` installs the audited
modules and runs import/paste checks. It refuses replacement of differing existing
files unless `--overwrite` is explicitly authorized; that option creates unique
backups. Do not update a trainer installation being used by a running experiment.

Use `tools.online_cp_curriculum` with `config/online_cp_feedback.json` to publish
`feedback_contract.json` beside a verified **current-version** bank. Its separate
sidecar preserves `curriculum_contract.json`; neither old results nor bank entries
are rewritten. Missing/current-version provenance requires rebuilding the proper
new experiment, not fabricating metadata.

The strict launcher is `python -m tools.train_online_feedback`. It requires
`--bank`, `--feedback-config`, `--arm basic|full`, and `--seed`. Full additionally
requires `--feedback-gnn-config` and `--feedback-raw-root`. The raw root is the
verified nnU-Net raw `Dataset...` directory containing `imagesTr`, `labelsTr` and
`online_cp_dataset.json`, **not** the parent Medical directory. The graph cache
is explicitly placed in this new trainer's `feedback_graph_cache` subdirectory.
Run `--help`/`--dry-run` to inspect paths before launching. All 250 nnU-Net epochs,
CP probability 0.5, and 128 candidate centers are retained.

Fresh launches refuse nonempty result folders. `--resume` requires a feedback
checkpoint; legacy checkpoints and missing-checkpoint fresh-start fallbacks are
rejected. A complete epoch checkpoint includes nnU-Net, the difficulty GNN and
optimizer, measured batch policy, table, predictions/provenance, RNG and the
last observed events. CUDA bitwise equivalence is not asserted.

## Verification scope

Tests named `debug` use synthetic analytic tensors/graphs to exercise the actual
implementation. They are not medical training, full-organ graph/resource
validation, or evidence that curriculum/L2 improves segmentation. See the final
task report for exact test totals. Real-data calibration, full training,
server checkpoint replay and downstream evaluation remain separate steps.

Comparing Basic and Full alone does not isolate the feedback contribution from
quality gating. A research claim needs matched fixed-quality/uniform and
feedback-policy comparisons and predeclared inner-training model selection.
