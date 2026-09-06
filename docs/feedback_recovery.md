# Full-size feedback preparation recovery

This workflow addresses the failed `work/feedback_experiment` preparation without
replacing any original cache, nnU-Net installation, model or experiment result.
It uses the existing Git checkout and writes a separate experiment, by default
`work/feedback_experiment_recovered`. It is not a migration of trained weights.

## Server command

Run in the existing `(nnunet)` environment. GPU 3 below is the visibility used in
the supplied diagnostic log; use it only while that GPU remains assigned to you.

```bash
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 \
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
automatically restart it. Training-checkpoint resume is a separate operation.
An interrupted, incompletely certified cache migration requires a new explicit
`--experiment-name`; the incomplete destination is preserved for diagnosis.

Do not use `--overwrite` to recover the old cache. In the legacy preparation
command that flag removes the old graph artifacts and their metadata.

## Patient cohort versus source eligibility

The entire inner train/validation split, outer validation exclusion, and source
image/label hashes remain bound to the experiment. Labels are checked before an
integer conversion for finite, integral values from the configured label set;
image/label geometry is checked as well.

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
