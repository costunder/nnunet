# Train-only nnU-Net difficulty feedback: separate experiment

This implementation is **not** the earlier expanding GNN rank band. Lower
compatibility rank is not segmentation difficulty. Existing exact-argmax and
rank-band results/checkpoints retain their original meaning and are not upgraded.

## Population-metric v5 and historical comparisons

The current upper architecture is `hiercp_source_content_population_metric_v5`.
Region descriptors retain all 16 dimensions but use observed CT/whole-organ union
instead of label-conditioned tumor filling. This removes the demonstrated fill
footprint, not all potentially visible source-pathology information. K16 and the
30-iteration prototype fit budget remain unchanged; a final nearest-center
assignment makes support/dispersion consistent, without claiming full convergence.
The v2 bank stores full descriptor/membership evidence for numerical revalidation.

Only prototype-involving edges gain two fields: standardized descriptor distance
and signed distance minus mean cluster dispersion (the mean of both dispersions
for prototype pairs). Candidate-region edges remain 6D. Relative assignment
weights and these distances are not calibrated safety/difficulty probabilities.
The original-anchor/corruption ranking target remains a proxy; same-bank negative
construction means its ranking scores alone do not establish independent L2 CP
utility. `no_population` removes learned L2, not all prototype-based preprocessing.

Keep old checkpoints, predictions and metrics under their original identities.
Compare predictions using the same evaluator definition in a new output folder.
The combined revision is a version comparison, not an isolated L2 ablation.
`--reuse-basic-from` requires a completed original Basic, identical full native
training data and ordered raw-CP inputs, and matched training/runtime conditions.
It is separate from preprocessing reuse and never relabels old GNN/Full weights.

## Connected training path

The current production bank uses `onlinecp_raw_target_paste_v1`: paste the raw CT
and source mask at the selected raw target, then apply the native nnU-Net CT
normalization and resampling. It is not the historical donor-only resample followed
by translation. The bank stores full native baselines and exact separable operators
once per case, shared raw donor arrays once per source, and all candidate-specific
label/support changes. Selected-candidate crop evaluation preserves global cubic
spline tails and native clipping; it is not a finite-halo approximation.

All 128 raw candidates remain distinct even when mapped native centers coincide.
A target with zero native pasted support still performs the raw CP event and
ordinary segmentation update; its lesion difficulty is unavailable, not zero.
Large lesions may be partially intersected by the unchanged native training crop.
Full-candidate and observed-crop support counts are recorded separately in
`[OnlineCPNativeTransport]`; the paired event/source/appearance schedule is unchanged.

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

The current v5 upper graph retains the source-content connection to all patient
regions uniformly instead of revealing its original host region. Recipient
anatomy is allowed; this is not a claim that all spatial/context information has
been removed. Explicit source-address, raw-column, region/prototype permutation,
and permitted-context-response tests are separate contracts. Old v3/v4 graphs,
quality/difficulty weights and score banks are not relabelled as v5 artifacts.

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
bound to the actual bank/raw/checkpoint identity. Patient preparation is shared
across its source entries. Complete graph generations publish payload and receipt
together, with mmap and stat/SHA witnesses on warm reads. Interrupted incomplete
generations are preserved rather than approved as complete graph pairs.

Resource admission uses the full entry inventory, not a small prefix. Update
trials use real observed targets only; prediction trials cover the full bank.
Largest and mixed graph batches, host/cgroup limits, worker/prefetch/pin buffers,
optimizer/workspace bytes and resident nnU-Net memory are accounted separately.
No invented BCE targets are used for unobserved entries. Measured quality worker
counts are re-admitted against the feedback workload; this does not claim a new
exhaustive feedback-worker search or measured server speedup.

## Initial online bank preparation

The initial immutable quality bank is distinct from the difficulty-GNN graph
cache used during nnU-Net epochs. Its standard builder removes duplicate full
target-graph construction; both standard and multi-pool argmax builders use
ordered, measured CPU graph preparation and disk-backed pending scoring inputs.
These bank-builder optimizations preserve the 128 candidate centers, graph scale,
CP rules and the already-fitted compatibility model of the same architecture
version. This statement is not v3/v4-to-v5 compatibility: population-metric v5 trains
a new quality GNN and creates a new score bank, as described below.

See [online bank preparation and scoring](online_bank_performance.md) for phase
logs, resource-report paths, calibration and storage costs, and the distinction
between progress and a verified completed bank. The update is not a claim that
feedback-GNN training during nnU-Net epochs has become free or has been separately
optimized. Do not update the checkout used by an active experiment.

## Publication and launch

For a new isolated experiment, run from a fresh checkout:

```bash
python tools/run_feedback_experiment.py --medical-root /home/aicompetition06/Medical
```

This runs the complete configured quality-GNN training, bank publication, then
Full and Basic feedback training for outer fold 0. It creates only
`work/feedback_experiment` in that checkout and installs trainers in a private
nnU-Net package copy. Original site-packages and prior results are not replaced.
It preserves the existing allocated GPU visibility and requires one visible GPU.
`--dry-run` prints commands without launching children or creating outputs.
Without an explicit resume option, an existing experiment directory is refused.
Failed-stage outputs and `launch_plan.json` remain for inspection. For an existing
recovery experiment, use the [verified continuation guide](feedback_recovery.md):
`--resume-preparation` and `--resume-experiment` have different eligibility checks
and must retain the original launch arguments and Python environment. Neither
option authorizes a second process on a running experiment.
Preparation recovery requires an experiment originally created with `--recover-from`.
The separate `--upgrade-bank-from` workflow also supports its own verified
`--resume-experiment` continuation, not `--resume-preparation`. These flags do not
retroactively make an old unjournaled launch resumable. Newly created fresh runs
now have their own durable stage journal. Producer-written native exit receipts
and current artifact verification allow completed stages to be skipped without
guessing that an orphan child stopped. No checkpoint means no fresh-training fallback.
The default ends after training; optional `--evaluate` adds checkpoint-bound native
prediction and paired evaluation/statistics in separate generation/receipt files.
`--outer-fold`, `--dataset-id`
and the nnU-Net `--seed` are explicit options; GNN/bank seeds still follow their
checked-in fold-specific configuration.

Run from this repository in the intended nnU-Net environment, with the `hiercp`
package importable. First inspect `python custom_trainers/install_onlinecp_custom_trainers.py check`;
the installer audits all ten trainer/helper hashes. `apply` installs the audited
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

For population-metric v5, use the preprocessing-only reuse command in
[README](../README.md). Its versioned dependency projection verifies unchanged
labels, full nnU-Net configuration, raw cohort/split, native implementation and
every preprocessing output. Known non-preprocessing configuration changes are
recorded; unknown keys are not silently ignored. Only verified preprocessing is
shared, while the full v5 quality model and a new score bank are built. Original
GNNs, runtimes, journals and results remain intact. Historical exact-argmax, rank-only curriculum and
legacy downstream-ablation launchers reject this typed bank before starting work;
their old checkpoints are not converted to the new CP semantics.

The source must be the original durably completed native preprocessing root
(`work/feedback_medical_aug` in the recorded experiment), not the later bank-upgrade
root. Its plan proof is the native producer's exact four-file record: preprocessing
completion marker, splits, dataset JSON and configured plans JSON. The producer
and reuse reader share this file list; two-marker substitutes and arbitrary
supersets are rejected. The raw marker and complete cohort/output hashes remain
independently checked through the native preprocessing contract.

A first-launch rejection before the new root/journal exists is not a resumable
training attempt: resolve the cause and repeat the original launch without
`--resume-experiment`. Existing journal-backed attempts are reverified on resume;
partial roots are preserved for inspection. An explicitly requested resume whose
root is missing must never silently become fresh training.

Packed preprocessing payloads are hard-linked; metadata and new unpacked
arrays are separate. The supported workflow only reads shared payloads, but they
are not filesystem-enforced immutable copies. Manual edits to a shared inode
would affect both roots and are outside the supported workflow.

The following paragraph describes the historical compatible-version disk retry,
not permission to load old GNN artifacts in population-metric v5. For an already-created
`work/feedback_rawcp` upgrade stopped at the raw-case
disk-reserve preflight, repeat its original upgrade arguments with
`--resume-experiment` after the failed process has stopped. Completed setup,
GNN/preprocessing and audited successful bank entries are retained. An error row
is retryable only when its exact native pre-write disk failure, current bank
contract, unique progress/resource evidence and absence of source/case partial
artifacts agree. The builder rechecks free space against the case baseline plus
the unchanged reserve, and archives the original CSV/config/failure evidence in
`bank/disk_retry_history/<uuid>.json` before normal processing. Generic errors,
OOM, ENOSPC during writes and ambiguous artifacts are not admitted. Do not erase
manifest rows, journals, payloads or checkpoints, or use `--overwrite` to resume.
Shared NFS free space can change; passing this per-case lower-bound check does
not reserve capacity or guarantee enough storage for the whole remaining bank.

Fresh launches refuse nonempty result folders. `--resume` requires a feedback
checkpoint; legacy checkpoints and missing-checkpoint fresh-start fallbacks are
rejected. A complete epoch checkpoint includes nnU-Net, the difficulty GNN and
optimizer, measured batch policy, table, predictions/provenance, RNG and the
last observed events. CUDA bitwise equivalence is not asserted.

Complete GNN model/optimizer/RNG state is validated before native segmentation
state is changed. If a later native restore step fails, that trainer instance is
unusable and must not train or save a partially restored state.

Evaluation v5 maximizes valid TP cardinality, then quality over valid edges only.
Below-threshold edges cannot choose which GT lesion is recorded as detected.
Old predictions may be independently re-evaluated in a new output folder, but
old reports and definitions are never overwritten. The optional feedback producer
also binds checkpoint files, the actually loaded network weights, native runtime,
input images and output case receipts. Unreceipted partial predictions are not
reused as certified completed cases.

## Verification scope

Tests named `debug` use synthetic analytic tensors/graphs to exercise the actual
implementation. They are not medical training, full-organ graph/resource
validation, or evidence that curriculum/L2 improves segmentation. See the final
task report for exact test totals. Real-data calibration, full training,
server checkpoint replay and downstream evaluation remain separate steps.

Comparing Basic and Full alone does not isolate the feedback contribution from
quality gating. A research claim needs matched fixed-quality/uniform and
feedback-policy comparisons and predeclared inner-training model selection.
