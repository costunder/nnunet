# HierCP

HierCP 2.2.0 is a three-level PyTorch Geometric model for selecting 3D
liver-tumor Copy-Paste placements. The active graph contract is
`graph_schema_version=full_v22`: it preserves the full tumor footprint, builds
adaptive spatial context, and batches variable-node heterogeneous graphs.

The core HierCP implementation and configuration are present in this repository. Medical
images, graph caches, checkpoints, generated volumes, and nnU-Net results are
not included.

The optional OnlineCP experiments also require the original custom trainer
sources on the target host. They are referenced by `code.txt` but their
implementations are not included in that snapshot:

- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py`
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCPArgmaxV3.py`

Copying this repository alone therefore does not reproduce the complete
OnlineCP training environment. Training contracts fingerprint the installed
trainer and base-class source files; missing sources fail explicitly and
changed sources invalidate completed-run reuse. No replacement experimental
trainer has been invented.

The standard bank (`hiercp_online_bank_v2`, Dataset730) and multi-pool argmax
bank (`hiercp_online_bank_argmax_v3`, Dataset740) are different contracts.
`tools/downstream_level_ablation.py` retains its existing single-pool v2
contract; changing only its dataset number does not migrate it to v3.

Local verification covers Python/JSON syntax and isolated regression fixtures
for cache integrity, validation publication/rollback, causality contracts and
scoring batch order. These checks do not establish CUDA numerical equivalence
or completed full-data training/evaluation. Run the environment, smoke and
regression checks below on the target server before the full experiment.

## Active repository layout

- [`run.py`](run.py): supported top-level runner.
- [`config/train.json`](config/train.json): graph, model, cache, training, and
  generation contract.
- [`config/nnunet.json`](config/nnunet.json): downstream nnU-Net contract.
- [`hiercp/`](hiercp/): active graph, cache, batching, model, loss, training,
  and generation implementation.
- [`tools/validate.py`](tools/validate.py): cohort-exact, fingerprinted output
  validation.
- [`tools/assemble.py`](tools/assemble.py): preflighted dataset assembly.
- [`docs/design.md`](docs/design.md): design and safety details.

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

## Production workflow

`production` is the default run mode. The individual stages are:

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
completed `full` run is the complete production experiment.

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
