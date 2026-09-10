# HierCP

HierCP 2.2.0 is a three-level PyTorch Geometric model for selecting 3D
liver-tumor Copy-Paste placements. The active graph contract is
`graph_schema_version=full_v22`: it preserves the full tumor footprint, builds
adaptive spatial context, and batches variable-node heterogeneous graphs.

The core HierCP implementation and configuration are present in this repository. Medical
images, graph caches, checkpoints, generated volumes, and nnU-Net results are
not included.

OnlineCP requires the versioned trainer bundle in
[`custom_trainers/`](custom_trainers/). It contains historical paired/argmax
trainers and the current Full/Basic Feedback trainers, including their policy,
contract and raw-target resampling/storage helpers. Every installed module is
bound by the installer's SHA-256 inventory.

After cloning, activate the target nnU-Net environment and run from this
repository root:

```bash
python custom_trainers/install_onlinecp_custom_trainers.py check
python custom_trainers/install_onlinecp_custom_trainers.py apply
```

The installer verifies the checked-in source hashes and runs policy/paste/import
smoke checks after installation. Different existing trainers require explicit
`--overwrite` and are backed up before replacement. Training contracts
fingerprint the installed trainer and base-class source files; missing sources
fail explicitly and changed sources invalidate completed-run reuse. The current
Feedback runner installs into a new private runtime; do not apply an updated
bundle over a running or preserved experiment's installation.

The historical source-anchored single-pool bank, current raw-target bank and
multi-pool argmax-v3 bank are different contracts. The first two share the old
`hiercp_online_bank_v2` envelope, so the envelope or dataset number alone is not
a compatibility check. Current raw banks additionally require
`paste_contract=onlinecp_raw_target_paste_v1`,
source-mapping format `online_cp_raw_target_resampling_v2` and
`entry_storage=npz_candidate_refs_raw_target_v1`.
Only Full/Basic Feedback trainers consume the new contract. Historical
train/evaluate and downstream exact-argmax ablation workflows remain available
for their original banks; they do not become raw-feedback experiments by
changing a dataset ID. The obsolete `online_cp_benchmark all` route is rejected
before preparation because it would join the new builder to legacy trainers.

Local verification covers Python/JSON syntax and isolated regression fixtures
for cache integrity, validation publication/rollback, causality contracts and
scoring batch order. These checks do not establish CUDA numerical equivalence
or completed full-data training/evaluation. Run the environment, smoke and
regression checks below on the target server before the full experiment.

## Active repository layout

- [`run.py`](run.py): supported top-level runner.
- [`tools/run_feedback_experiment.py`](tools/run_feedback_experiment.py): current
  paired online Full/Basic Feedback runner with private runtimes and verified
  reuse/upgrade boundaries.
- [`gpt_handoff.md`](gpt_handoff.md) and [`code.txt`](code.txt): current project
  handoff and complete tracked-text source export; no patient images are included.
- [`config/train.json`](config/train.json): graph, model, cache, training, and
  generation contract.
- [`config/nnunet.json`](config/nnunet.json): downstream nnU-Net contract.
- [`hiercp/`](hiercp/): active graph, cache, batching, model, loss, training,
  and generation implementation.
- [`tools/validate.py`](tools/validate.py): cohort-exact, fingerprinted output
  validation.
- [`tools/assemble.py`](tools/assemble.py): preflighted dataset assembly.
- [`docs/design.md`](docs/design.md): design and safety details.
- [`docs/online_cp_feedback.md`](docs/online_cp_feedback.md): train-only
  nnU-Net difficulty-feedback experiment and launch boundaries.
- [`docs/feedback_recovery.md`](docs/feedback_recovery.md): verified continuation
  of the existing Medical Data Aug experiment.
- [`docs/online_bank_performance.md`](docs/online_bank_performance.md): current
  online-bank preparation, scoring, progress reports and reuse limits.

`run.py` expects a medical root containing this external layout:

```text
<medical-root>/
└── Data/
    ├── image/<case_id>_0000.nii.gz
    └── labels/<case_id>.nii.gz
```

Pass the root with `--medical-root`, or set `MEDICAL_ROOT`. If neither is set,
the runner checks the repository parent and the current directory. The default
artifact directory is `<repository>/work`; use `--work` to select another safe
location.

## Environment and non-final checks

Run commands from the repository root. Replace the example medical root with
the path on the target host.

```bash
python run.py env --medical-root /home/aicompetition06/Medical
python -m tools.audit
python -m tools.smoke --device cuda:0
python -m tools.regress --device cuda:0
```

`env` checks the paired NIfTI layout and required imports, then reports live
CPU affinity, RAM, storage, scheduler/cgroup limits, visible CUDA devices,
VRAM, and detectable MIG state. `audit`, `smoke`, and `regress` are static or
synthetic checks; passing them is not evidence that full training or evaluation
has completed.

To inspect an existing work directory or one label without running training:

```bash
python run.py status --medical-root /home/aicompetition06/Medical
python run.py case --medical-root /home/aicompetition06/Medical \
  --label /home/aicompetition06/Medical/Data/labels/liver_70.nii.gz \
  --components 3
```

## Current online raw-CP Feedback workflow

Raw donor CT/HU jitter and its mask are pasted at each selected raw location
before the native CTNormalization/resampling transform. The donor's mask is not
resampled once at its original position and translated afterward. All 128 raw
candidates are retained even when native centers coincide or a tiny donor has
zero new native label support. A real zero-support CP still trains segmentation;
only its difficulty-feedback observation is unavailable. Full and Basic share
the same original preprocessing and event/source/appearance schedule.

For the stopped `feedback_medical_aug` experiment with verified completed GNN
and preprocessing, start a NEW bank/runtime/results directory:

```bash
conda activate /home/aicompetition06/.conda/envs/nnunet &&
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
read -r -p "Allocated GPU: " CP_GPU &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$CP_GPU" \
/home/aicompetition06/.conda/envs/nnunet/bin/python -B tools/run_feedback_experiment.py \
  --upgrade-bank-from work/feedback_medical_aug \
  --experiment-name feedback_rawcp \
  --medical-root /home/aicompetition06/Medical \
  --outer-fold 0 --dataset-id 760 --seed 42
```

The old GNN stays at its verified path, immutable preprocessing payloads get a
new directory view, and native unpacking writes only to the new view. Old bank,
runtime, journals and results are not overwritten. This launch trains Full then
Basic; downstream prediction/evaluation is a separate task. See
[`gpt_handoff.md`](gpt_handoff.md) for scope, test evidence and safe retry limits.

## Standalone/offline production workflow

This separate `run.py` workflow generates offline augmented volumes; it is not
the online Feedback runner above. `production` is its default run mode. The
individual stages are:

```bash
python run.py prepare --run-mode production \
  --medical-root /home/aicompetition06/Medical
python run.py train --run-mode production \
  --medical-root /home/aicompetition06/Medical --device cuda:0
python run.py generate --run-mode production \
  --medical-root /home/aicompetition06/Medical --device cuda:0
```

The aggregate targets are:

```bash
python run.py all --run-mode production \
  --medical-root /home/aicompetition06/Medical --device cuda:0
python run.py full --run-mode production \
  --medical-root /home/aicompetition06/Medical --device cuda:0
```

`all` performs preparation, HierCP training, generation, exact validation, and
dataset assembly. `full` performs those stages and then runs the configured
nnU-Net preparation, five-fold training, and evaluation. Only a successfully
completed `full` run is the complete standalone/offline production experiment.

Production rejects `--max-cases`, `--case-id`, `--epochs`, `--batch-size`, and
`--num-workers` command-line overrides. It also rejects skipped validation or
assembly. These checks keep a reduced or unvalidated artifact from being
reported as final.

## Explicit non-production modes

The runner also defines `debug`, `benchmark`, and `ablation`. Debug and
benchmark may run individual prepare, train, or generate stages. Ablation is
train-only and pairs `--run-mode ablation` with an explicit `--ablation-mode`
of `no_local`, `no_patient`, or `no_population`. None of these modes may run
`all`, `full`, or final nnU-Net targets. Every non-production pipeline stage
must use an explicit `--work` directory different from the production default,
so reduced artifacts remain isolated and labelled. Dedicated multi-mode
ablation orchestration and reporting live in
[`tools/ablation.py`](tools/ablation.py).

Validation and assembly reached by debug or benchmark generation use their
own explicit `--run-mode nonproduction` contract. Such generation may skip
assembly after validation; if validation is skipped, assembly must also be
skipped. These outputs remain non-final.

## Resource preflight and measured calibration

Before preparation, the runner computes the full selected case-by-sample cache
count, measures free space on the target filesystem, and checks it against a
conservative cache-size estimate. Existing graph-cache sizes supply the p90
estimate when available; otherwise the estimate records its conservative
fallback. This storage estimate is not a substitute for measuring the target
dataset.

Training records live CPU, RAM, storage, scheduler/cgroup, GPU, VRAM, and MIG
information. In the checked-in training configuration, both physical batch
size and DataLoader worker count are `"auto"`. Training therefore probes the
configured batch candidates for throughput and peak VRAM, records candidates
that cannot be safely measured, measures the configured worker candidates for
loader throughput, selects a passing configuration, and records the trials in
`work/model.pt.preflight.json` and the checkpoint. A saved calibration is
reused only when its training identity and hardware/resource fingerprint
match.

The reported effective batch size is:

```text
physical batch size × gradient accumulation steps × data-parallel workers
```

The current implementation has validated one-device semantics only, so the
data-parallel worker count is 1. For production CUDA training and generation,
the process fails if more than one CUDA device is visible. Request one GPU from
the scheduler and expose it as logical `cuda:0`; there is no validated DDP
fallback that silently leaves extra assigned GPUs idle.

## Online bank preparation and scoring

Current raw-target preparation stores one full-grid unclipped baseline per case,
shared content-addressed donor arrays, and candidate-specific label/CT inputs.
Training computes the exact native output within the unchanged nnU-Net crop;
larger lesions may be intersected by that crop without deleting their remaining
support from the bank. Full native support and observed crop support are audited
separately. No full-volume resampling is performed for every training event.
Preparation measures RSS/CPU/time and checks disk reserve. The baseline costs
approximately 10 bytes per native voxel plus operators/candidates; this is not
a total peak-RAM estimate. Only the verified standard CT/native kernels are
supported; an unproved crop/nonzero-mask change is reported, not hidden by
discarding a donor or tightening the original 0.85 liver-coverage condition.

The bank optimization in `ebe58ff` preserves the model, full graph topology,
candidate pools, CP rules and training settings. The standard bank no longer
builds complete target features/edges just to discard them during geometry
validation. Both bank builders share the prepared source and use measured,
ordered CPU candidate-graph preparation. Pending scoring inputs use an owned
disk spool and physical-batch mmap loading rather than accumulating complete
graphs from many sources in RAM. This adds storage I/O; it is not a measured
server-wide speedup claim.

Bank builds now emit phase/heartbeat logs, CPU preparation resource reports and
scoring calibration/I/O reports. Their locations, first-calibration delay,
remaining memory requirements and verification scope are documented in
[`docs/online_bank_performance.md`](docs/online_bank_performance.md).
Do not update a checkout or trainer installation used by a running experiment.
An existing compatible bank is verified and reused; the performance-only change
in `ebe58ff` does not require deleting it or restarting the GNN. The new raw
transport contract DOES require a separate bank/runtime as described above,
while verified GNN/shared preprocessing remain reusable. Follow the
[recovery guide](docs/feedback_recovery.md) for a stopped, eligible continuation.

## Exact validation and assembly

Production generation is accepted only when candidate and reference case-ID
sets are identical and every case passes shape, affine, label-domain, finite
intensity, and tumor-change checks. `tools.validate` fingerprints the candidate
image and label, reference image and label, validation contract, and complete
cohort with SHA-256. `--resume` reuses a row only when every fingerprint still
matches. Reports are written atomically to:

```text
work/output/data/validation.csv
work/output/data/validation_summary.json
```

Assembly requires that report, requires its accepted cohort to equal the
validated augmentation directory, and recomputes candidate hashes before any
write. In production, the augmentation cohort must also equal the complete
original cohort and contain zero rejected or non-production rows. A full
source/destination/collision preflight runs before materialization; existing
files are reusable only when they are the same underlying source file, and
unplanned stale outputs are rejected. The final manifest is written atomically
to `work/dataset/manifest.csv`.

## What remains to be run

This source checkout does not establish a scientific result by itself. On the
target MobaXterm/server environment, supply the real dataset and compatible
PyTorch, PyTorch Geometric, CUDA, and nnU-Net installation; run the environment
check; review the measured preflight; and execute the requested production
training and evaluation. No README statement or smoke test replaces those
actual-data runs.
