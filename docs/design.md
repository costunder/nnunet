# HierCP 2.2 / `full_v22` design

## Scope and current status

The active package version is 2.2.0 and the only accepted graph schema is
`full_v22`. Historical V21 names still identify the adaptive geometry
algorithm in a few symbols, but the active implementation uses the unversioned
package paths listed below. The supported entry point is
[`run.py`](../run.py).

This checkout contains implementation and configuration only. It does not
contain the medical dataset, graph caches, trained checkpoints, generated
NIfTI volumes, or nnU-Net evaluation results. Static and synthetic checks do
not establish that the actual-data experiment has been run.

## Active implementation map

- [`config/train.json`](../config/train.json) is the authoritative HierCP
  graph, cache, model, training, and generation contract.
- [`config/nnunet.json`](../config/nnunet.json) defines downstream nnU-Net
  planning, five folds, and evaluation settings.
- [`hiercp/geometry.py`](../hiercp/geometry.py),
  [`hiercp/spatial.py`](../hiercp/spatial.py), and
  [`hiercp/local.py`](../hiercp/local.py) construct the full-shape local
  geometry.
- [`hiercp/hierarchy.py`](../hiercp/hierarchy.py) constructs the patient and
  population-prototype levels.
- [`hiercp/cache.py`](../hiercp/cache.py) owns region/prototype/graph cache
  creation and exact cache publication.
- [`hiercp/data.py`](../hiercp/data.py) owns variable-node PyG collation and
  DataLoader-facing datasets.
- [`hiercp/model.py`](../hiercp/model.py),
  [`hiercp/loss.py`](../hiercp/loss.py), and
  [`hiercp/pipeline.py`](../hiercp/pipeline.py) implement forward, ranking
  losses, optimization, generation, reporting, and resume.
- [`tools/validate.py`](../tools/validate.py) validates generated cases against
  an exact cohort and content contract.
- [`tools/assemble.py`](../tools/assemble.py) assembles original and validated
  cases only after a complete collision/staleness preflight.
- [`tools/audit.py`](../tools/audit.py),
  [`tools/smoke.py`](../tools/smoke.py), and
  [`tools/regress.py`](../tools/regress.py) are non-final checks.
- [`tools/ablation.py`](../tools/ablation.py) orchestrates, inspects, and
  summarizes the independent leave-one-level-out training runs.

## End-to-end production path

The top-level data flow is:

```text
external Data/image + Data/labels
  -> exact train/validation split
  -> patient regions + population prototype bank
  -> exact case-by-sample hierarchical graph cache
  -> variable-node PyG batches
  -> three-level model -> ranking/consistency loss -> optimizer -> checkpoint
  -> Copy-Paste generation
  -> exact SHA-256 validation -> accepted materialization
  -> preflighted dataset assembly
  -> nnU-Net preparation, training, and evaluation (`full` target only)
```

`python run.py all` stops after validated dataset assembly. Only
`python run.py full` also invokes the configured nnU-Net `all` target. A run is
not the complete production experiment merely because preparation, a smoke
test, or HierCP training finished.

## Hierarchical graph and model contract

Level 0 is a heterogeneous local graph with full tumor-surface and
tumor-interior nodes, separate source and target parenchymal-context nodes,
and separate source and target liver-surface anchors. Physical-radius and
interface relations preserve the configured receptive field. Adaptive ROI
budgets are fail-fast resource guards: an oversized requested context raises a
detailed error instead of silently reducing the margin or resolution.

Level 1 represents source tumor, placement candidates, liver regions, other
lesions, and the whole liver. A patient with no other lesions has zero lesion
nodes and empty lesion relations; it does not receive a fabricated placeholder
node. A positive `max_lesions` value is also a fail-fast guard, not a slicing
limit. The checked-in configuration uses `null`, meaning unlimited lesions.

Level 2 connects candidate/region representations to the population prototype
bank. The three levels use variable-node `HeteroData` batching rather than
padding every graph to a small fixed node count.

The ablation train modes disable exactly one of the local, patient, or
population encoders. A disabled encoder is frozen, and the ablation score head
contains only active feature blocks, so disabled zero-valued columns are not
left as trainable parameters. The full-mode score-head shape remains unchanged
for full-checkpoint compatibility.

Important checked-in values include:

| Contract | Value |
| --- | --- |
| Hidden dimension / attention heads | 128 / 4 |
| Local / patient / prototype layers | 3 / 2 / 2 |
| Liver regions / population prototypes | 24 / 16 |
| Sampled context nodes / graph hops | 384 / 2 |
| Cache samples per case / candidates per sample | 2 / 8 |
| Training epochs | 40 |
| Target effective batch size | 2 |
| Generation candidates / copies per case | 128 / 1 |

These values come from `config/train.json`; changing that file changes the
experiment contract. Node, relation-edge, and adaptive-ROI limits do not
silently truncate a graph. Exceeding a configured resource ceiling fails with
diagnostics so the limit can be reconsidered from measured hardware evidence.

Cache readiness is exact. For the selected split, every configured
`case_id × sample_index` must have a schema-valid artifact before the index and
ready marker are published. `failed`, `no_tumor`, insufficient-candidate,
unrepresentable-geometry, and resource-budget rows remain diagnostic failures
and never count as complete. Resource-budget failures retain their distinct
error type after the diagnostic manifest is saved.

## Run-mode separation

[`run.py`](../run.py) defines four modes:

| Mode | Intended use | Runner contract |
| --- | --- | --- |
| `production` | final experiment | default; exact full workload; aggregate and nnU-Net targets allowed |
| `debug` | labelled debugging | individual prepare/train/generate stage and separate work directory only |
| `benchmark` | labelled performance study | individual stage and separate work directory only |
| `ablation` | labelled ablation study | train target only, explicit non-full ablation mode, and separate work directory |

Production rejects command-line `--max-cases`, `--case-id`, `--epochs`,
`--batch-size`, and `--num-workers` overrides, and it rejects skipped
validation or assembly. Debug/benchmark prepare, train, or generate runs and
ablation train runs require an explicit `--work` path different from the
production default. A non-full `--ablation-mode` requires
`--run-mode ablation`, and that run mode rejects every target except `train`.
`all`, `full`, and nnU-Net final targets are production-only. The dedicated
multi-mode entry point is [`tools/ablation.py`](../tools/ablation.py).

The validator and assembler expose only `production` and `nonproduction`.
Debug or benchmark generation maps to `nonproduction` for those tools. A
validation subset can therefore exist only as an explicitly non-final
artifact. If non-production generation skips validation, assembly must also
be skipped.

## Runtime preflight and calibration

For prepare, train, generate, `all`, and `full`, the runner first invokes
[`tools/env.py`](../tools/env.py). It verifies imports and the paired-data
layout and reports the live host/allocation state: logical and affinity CPU
counts, total/available RAM, filesystem capacity, scheduler variables, cgroup
limits, CUDA visibility, GPU model/VRAM, and detectable MIG state. Fields the
runtime cannot obtain are reported as unavailable rather than invented.

Preparation also performs a storage preflight over the complete selected
cohort. Required graph-file count is exact. The size estimate uses the p90 of
existing graph artifacts when available, otherwise a recorded conservative
fallback, plus region-cache and headroom estimates; it compares that estimate
with actual free space before cache creation. It is an estimate and must still
be reviewed on the target dataset and filesystem.

Training calls the runtime collector again for its execution report. The
checked-in profile sets both `training.batch_size` and
`training.num_workers` to `"auto"` and supplies an explicit measurement plan:

- physical batch candidates: 1 and 2, each with three calibration repeats;
- maximum accepted peak-VRAM fraction: 0.9;
- DataLoader worker candidates: 0, 2, 4, and 8;
- loader calibration window: eight batches.

The physical-batch probe performs forward, backward, and an optimizer step at
zero learning rate on the largest cache artifacts. It records throughput and
peak VRAM; a CUDA-OOM candidate is recorded, and larger candidates may be
marked unmeasured rather than retried. Among memory-safe measured candidates,
the highest-throughput candidate is selected. The loader probe measures each
worker candidate and selects the highest measured throughput. If no configured
candidate succeeds, preflight fails; it does not shrink the model, graph,
dataset, or resolution.

The selected values and all trials are stored in
`work/model.pt.preflight.json` and in checkpoint state. Resume reuses an
automatic selection only when the saved training identity and a hardware/
allocation resource fingerprint match. The final loader applies the configured
pin-memory, prefetch, persistent-worker, and optional CUDA-prefetch settings.

Effective batch size is reported as:

```text
physical batch size × gradient accumulation steps × data-parallel workers
```

The current validated data-parallel worker count is 1. Production CUDA train
and generate call a single-device guard: if the process sees more than one CUDA
device, it fails with allocation diagnostics because no validated DDP or
case-distribution path exists. A production job must request one GPU and expose
it as logical `cuda:0`. This prevents an apparently successful run that leaves
other assigned GPUs silently unused.

## Validation and assembly integrity

Production validation requires the generated candidate case-ID set to equal
the complete reference case-ID set. Every case must pass shape, image and label
affine, allowed-label, finite-intensity, and expected tumor-change checks; any
rejection makes production validation fail.

Each validation row records SHA-256 for all four case inputs:

- generated candidate image and label;
- reference image and label.

It also records a canonical validation-contract fingerprint and an exact
reference/candidate cohort fingerprint. `--resume` reuses a prior `ok` row only
when its schema and all input, contract, and cohort fingerprints match. The CSV
and JSON summary are atomically replaced. A partial production cohort is never
materialized as accepted output.

Assembly requires `validation.csv` whenever an augmentation directory is
supplied. It requires the directory to equal the report's accepted cohort,
checks one supported schema/contract/cohort, and recomputes the augmented image
and label SHA-256 values to detect changes after validation. Production further
requires augmentation IDs to equal all original IDs and rejects any rejected
or non-production validation row.

Before materializing any file, assembly checks the complete source and
destination plan, duplicate IDs, destination collisions, hardlink filesystem
compatibility, stale unplanned outputs, and any existing manifest. Existing
destinations are reusable without overwrite only when `Path.samefile` confirms
they are the requested source. Stale outputs are not implicitly deleted, even
with `--overwrite`. The final manifest is written atomically and records the
validation contract and report fingerprints.

## Migration and required real execution

Do not reuse fixed-node V17 caches or checkpoints with this design. Rebuild
artifacts through the current `run.py` and `full_v22` configuration in a clean
work directory. Historical versioned-subpackage, repository-local wrapper, and
snapshot-helper paths are obsolete; use the active files linked above.

The real dataset must be provided outside the repository as
`<medical-root>/Data/image` and `<medical-root>/Data/labels`. Final claims
require running the environment check and measured calibration on the target
MobaXterm/server allocation, then completing production training and the
requested nnU-Net evaluation on actual data. No such run is implied by this
design document.
