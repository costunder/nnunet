# Full-size feedback preparation recovery

## Current failed-bank upgrade

For the `liver_101` donor-only resampling failure in `work/feedback_medical_aug`,
the current action is **not** to rerun the historical continuation command below.
Use `tools/run_feedback_experiment.py --upgrade-bank-from work/feedback_medical_aug
--experiment-name feedback_rawcp` with the original medical root, fold, dataset ID,
seed and nnU-Net Python environment. The complete copyable command is in
[README](../README.md).

The upgrade verifies original launch/journal checksums, GNN/checkpoint/prototype
and shared preprocessing identities. It preserves the original root and creates
a separate raw-target bank, private runtime, preprocessing directory view and
Full/Basic result directories. Immutable dataset files may be linked; mutable
metadata and later native unpack outputs belong to the new view. It does not
retrain the completed quality GNN, rewrite an old bank, or resume an old
segmentation checkpoint under different CP semantics.

The source experiment must be stopped and must not have started segmentation
training. Failed/incomplete or ambiguous provenance is refused, not repaired by
editing checksums. Only validated resource-policy differences may be recorded;
the saved training configuration is retained byte-for-byte, and model/graph/data
contracts remain strict. Storage preflight includes native baseline payload and
new-view unpack lower bounds; it is not a peak-memory guarantee.

For an eligible interruption of this **new upgrade experiment**, use the same
original upgrade arguments plus `--resume-experiment`. Existing bank error rows,
incompatible payloads, ambiguous running stages, partial runtime setup or missing
training checkpoints can still prevent continuation. `--resume-preparation` is
not an upgrade option. Do not delete journals/manifests or use `--overwrite`.

## Historical preparation-recovery workflow

This workflow addresses the failed `work/feedback_experiment` preparation without
replacing any original cache, nnU-Net installation, model or experiment result.
It uses the existing Git checkout and writes a separate experiment, by default
`work/feedback_experiment_recovered`. It is not a migration of trained weights.

## Server command

Before either update/launch command in this document, confirm that no bank,
GNN or nnU-Net process is still using this checkout or the same experiment.
Do not run `git pull`, replace trainers, or launch a second continuation while
that work is active. Updating source files does not update code already loaded
by a running Python process and can mix versions in later child stages.

Run in the existing `(nnunet)` environment with the scheduler/user-assigned single
GPU visibility already set. The command preserves `CUDA_VISIBLE_DEVICES`; it does
not guess an available GPU from another experiment's example.

```bash
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
python -B tools/run_feedback_experiment.py \
  --recover-from work/feedback_experiment \
  --medical-root /home/aicompetition06/Medical \
  --outer-fold 0 \
  --dataset-id 760 \
  --seed 42
```

`--dry-run` validates the original launch identity and prints the plan without
creating outputs, copying packages, querying GPUs or starting child commands.
For an interrupted **preparation** attempt, use the same command with
`--resume-preparation`. The launcher verifies its private runtime and recovery
evidence; it does not trust a stage name in a journal as proof of completion.
Once quality-GNN training or a later stage has started, this option refuses to
automatically restart it. The explicit continuation below is a separate operation.
An interrupted, incompletely certified cache migration requires a new explicit
`--experiment-name`; the incomplete destination is preserved for diagnosis.

Do not use `--overwrite` to recover the old cache. In the legacy preparation
command that flag removes the old graph artifacts and their metadata.

## Historical same-contract continuation of Medical Data Aug

This section documents the older, unchanged-CP-contract continuation. It does not
upgrade donor-only banks to raw-target transport; use the current workflow above
for that change.

For the stopped `work/feedback_medical_aug` experiment whose preparation has
completed, retain the original launch arguments and add `--resume-experiment`.
The launcher still checks whether the failed stage is eligible to continue;
this is not a blanket restart of an arbitrary interrupted process.

The saved plan includes the Python executable. For this experiment it is
`/home/aicompetition06/.conda/envs/nnunet/bin/python`, not the `(base)` Python.
Changing it causes a plan-identity mismatch even when the journal checksum and
source identity are correct. Do not edit the saved plan/journal to disguise it.

```bash
conda activate /home/aicompetition06/.conda/envs/nnunet &&
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
/home/aicompetition06/.conda/envs/nnunet/bin/python -B tools/run_feedback_experiment.py \
  --recover-from work/feedback_experiment \
  --experiment-name feedback_medical_aug \
  --medical-root /home/aicompetition06/Medical \
  --outer-fold 0 \
  --dataset-id 760 \
  --seed 42 \
  --resume-experiment
```

This verifies the saved launch plan, unchanged source identity, preparation
receipt, full SHA-bound graph publication and private runtime. It does **not**
rerun the 187-cache preparation, copy nnU-Net again, change the saved training
configuration, or overwrite the previous experiments. It verifies completion
evidence for stages already finished, skips accepted completed stages, and
continues eligible pending/failed stages in the original order: quality GNN,
shared nnU-Net preprocessing, bank publication, then Full and Basic training.
Downstream statistical evaluation remains a separate task.

Reaching `[Bank] case 1/...` does not mean the original GNN/cache or nnU-Net
preprocessing has been erased or retrained. It is the separate online-bank
stage, which inspects source entries and reuses verified compatible entries.
The `ebe58ff` bank optimization keeps the existing data/configuration contracts;
do not delete a compatible bank to obtain the new implementation. New work uses
the new graph/scoring path after a safe update and an accepted continuation.
See [bank progress, resources and reuse](online_bank_performance.md) for the
actual logs and completion checks. A `running` journal row or existing lock
still requires diagnosis, not a forced retry or manual deletion.

A failed bank stage is not automatically reusable just because the launcher
accepts the retry. An existing `error` manifest row is refused by the bank;
unresolved `insufficient_candidates`, `unrepresentable_source` or
`source_patch_too_large` rows prevent final publication. Diagnose the recorded
failure without rewriting the manifest or removing its evidence.

The previous journal is archived byte-for-byte in `recovery/journal_history`.
Failed rows remain in the current journal and each new attempt is appended.
Completed continuation stages are skipped only after their recorded artifact
hashes and input configurations are checked. Older completed training-stage rows
without this completion evidence are refused, not retroactively certified.

A failed automatic GNN calibration may retry only when neither the published
preflight nor a training checkpoint exists. Otherwise a full native
`model.last.pt` is required, and the training loader must accept its exact
configuration, execution version, calibration, optimizer and RNG identity.
Older incompatible checkpoints are not silently migrated. Interrupted Full or
Basic training uses the trainer's strict `--resume` path and requires complete
feedback/optimizer/RNG state; an output folder without a checkpoint is preserved
and refused. Existing locks and still-running journal rows are ambiguous and are
not automatically cleared. Do not manually edit journals or delete preflight
files to bypass these checks. `--dry-run` validates and prints without writes.

## Patient cohort versus source eligibility

The entire inner train/validation split, outer validation exclusion, and source
image/label hashes remain bound to the experiment. Labels are checked before an
integer conversion for finite, integral values from the configured label set;
image/label geometry is checked as well.

### Numerical header equivalence and complete-cohort preflight

An elementwise affine tolerance alone is not an adequate grid comparison: it
can reject harmless float32 header roundoff yet admit small linear errors that
accumulate across a large image. The reported `liver_85` headers have matching
512 x 512 x 630 shape, mm units and active sforms. Reconstructing the printed
float32 srows gives approximately 0.0000572 mm / 0.0000858 voxel maximum
voxel-center-corner displacement. This reconstruction is header evidence, not
an inspection of the actual medical voxels or proof of anatomical alignment.

The previous two-float32-ULP restriction was not a suitable physical criterion.
The supplied `liver_97` diagnostic failed it at 32 ULP despite only 0.0000432 mm /
0.0000598 voxel maximum displacement, less than the `liver_85` displacement.
ULP size depends on coordinate magnitude. Version `donor_grid_physical_extent_v2`
therefore records ULP counts for diagnosis only, without increasing an ULP cap.

The numerical limits remain unchanged: over the **whole voxel-cell box**, both
reciprocal voxel-grid displacements must be at most 0.0001 voxel and physical
displacement at most 0.0001 mm. Known equal spatial units are converted for that
comparison, not rewritten in the data. Additional acceptance beyond the old
elementwise test requires matching mm units and compatible nonzero effective
coordinate-frame codes. Storage format (NIfTI-1 versus NIfTI-2) and whether the
selected transform is named qform or sform do not change this physical test.
Unknown units retain the old strict requirement plus both voxel bounds, and
are explicitly not reported as a verified mm displacement.

Nonfinite/singular transforms, unit/frame incompatibility, flips and changes
exceeding either whole-box limit are rejected. Selected affines are never
replaced by alternate transforms; headers and voxel arrays are not rewritten,
reoriented or resampled. Numerical equivalence is printed explicitly and its
evidence stored in the SHA-bound donor contract. This tests pairwise grid
agreement, not anatomy, pixdim-versus-own-affine consistency, or the separate
spacing-only graph assumption about shear. It is not a clinical tolerance.

For an unfinished recovery, `header_geometry.<unique-id>.json` now collects all
selected cases before full source hashing, cache copies, raw-label scans or
resource pilots. Expected invalid/corrupt headers are recorded for every case,
then the whole phase fails if any case is invalid. No failed case is dropped or
treated as eligible. Unexpected execution errors still propagate. Direct donor
preparation also performs this complete-cohort header phase before reading any
label voxels. Subsequent SHA checks and a repeated geometry check bind the actual
loaded source to the audited header; header-only reports do not claim full-file
SHA verification. Each report has a new name so previous evidence is retained.

Existing donor contracts and completed receipts remain readable using their
recorded policy and source hashes; a legacy receipt is not relabeled as a new
policy run. New or interrupted recovery attempts perform the new header phase.
After this pre-eligibility failure, update the same checkout and repeat the
recovery command with `--resume-preparation`; retain the original run root,
fold, dataset ID and seed so the preparation journal can be reverified.

Cases without the configured tumor label cannot supply a tumor source. This is
recorded in a checksummed `donor_eligibility` contract, with label histograms,
source hashes and explicit reasons. It is **not** a claim that those patients
are medically normal. They remain in the patient cohort, prototype support when
in inner training, and downstream nnU-Net training/validation as originally
defined. Only the required tumor-source graph keys use the eligible-donor set.

In the supplied fold-0 diagnostic, the whole cohort has 105 patients: inner train
84 and inner validation 21. Eleven have no configured tumor label (8 train,
3 validation). If direct source verification confirms that evidence, there are
94 eligible sources and 188 required graph samples (152 train, 36 validation).
The original 26 outer-validation patients remain excluded from preparation.
The old manifest records 182 successful samples, which must be verified against
their actual files before any reuse is accepted.

## Geometry, resources and candidate search

The model, graph geometry, voxel spacing, context/search radii, all lesion
components, candidate counts and training epochs are not reduced. The quality
GNN remains configured for 40 epochs; each nnU-Net arm remains at 250 epochs.

Recovery derives a conservative ROI enclosure from **every** real lesion
bounding box, source padding, the existing maximum physical corruption scale
1.60, rotation enclosure and the unchanged physical search radius. This is an
analytic allocation ceiling, not a larger cropped graph and not a measured
worst-case RAM claim. The candidate's actual requested ROI is unchanged.

Before publishing the recovered training configuration, isolated CPU processes
measure the original failed samples at full size, recording wall time, sampled
peak RSS and failure details. The initial dense-allocation estimate is labeled
as an estimate; it does not replace these measurements. A failure never creates
a successful pilot record or a synthetic training result. Extremely large
resource demands still stop explicitly; no context, resolution or graph is
silently reduced. Successful pilot graphs are not checkpoints or final results.

CPU preparation uses measured concurrency waves for real pending cases, within
CPU and memory constraints. Completed wave outputs are retained and all pending
cases are processed. Different cases have different costs: these wave timings
are not identical-input microbenchmarks or proof of globally optimal workers.
Per-wave and failure resource reports record the actual measurements and limits.

The legacy successful candidate path and RNG stream are preserved. Only an
insufficient/failed path continues with deterministic exhaustive legal-center
search, using the same source and anatomical constraints. Scratch-memory tiling
does not remove candidate centers. Pool exhaustion, a missing positive graph
node type, and rejected negative geometry have distinct diagnostics. A physically
unrepresentable positive remains an error; no fake context is substituted.

## Integrity and remaining verification boundaries

### Historical failed attempts during bank upgrade

The recovery writer before `301482c` stored failed attempts as only
`name/status/error`, without `input_files`. The bank-upgrade reader formerly
misreported these missing hashes as changed configuration. It now accepts only
that exact legacy failed-row shape, before any hash-bound attempt, and only if
a later completion of the **same stage** passes the existing input SHA and native
completion checks. The preparation receipt, runtime inventory and source files
must still verify. Failed-attempt outputs are not adopted and the source journal
is never rewritten to invent historical hashes.

Missing hashes on a completed or later modern row, malformed mappings, actual
hash differences, running attempts and missing completion evidence still fail.
Real differences report the stage, file path and recorded/current SHA instead
of a generic configuration error. Accepted legacy failures are printed as
`[SOURCE LEGACY HISTORY]` only after source validation succeeds.

This source check runs before creating the new upgrade root or launching any
GPU, bank or training work. For a failure at this check, rerun the original
upgrade command after updating the code, without `--resume-experiment`; do not
delete an existing root if another invocation has since created it.

Migration validates the original split, prototype, source data, manifest,
actual graph payloads and byte hashes. The allowed changes are explicit donor
eligibility and an increased resource-only ROI ceiling; incompatible geometry
or other graph settings are rejected. Original graph bytes are never rewritten.
Copied graphs receive the new metadata binding and an explicit migration
certificate. Old artifacts have no historical producer-code SHA, so migration
does not claim to authenticate code history that was never recorded.

The same recovered training-config path is passed through GNN training, online
bank preparation and the feedback-contract publisher. The recovery launcher then
continues quality-GNN training, Full feedback nnU-Net, and Basic feedback nnU-Net.
It does **not** perform downstream prediction, evaluation or statistical tests.

Local unit tests use explicitly labeled DEBUG fixtures and runner doubles. They
do not establish real-medical graph feasibility, GPU throughput, completed
40/250-epoch training, or downstream performance. Those remain server checks.
