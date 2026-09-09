# Online bank preparation and scoring

This guide describes the implementation introduced by
[`ebe58ff`](https://github.com/costunder/nnunet/commit/ebe58ff5b2227e10480f5c066076e5f766059fa3).
It covers the standard bank in [online_cp_benchmark.py](../tools/online_cp_benchmark.py)
and the multi-pool bank in [online_cp_argmax_benchmark.py](../tools/online_cp_argmax_benchmark.py).
They remain different bank contracts; this change does not convert one into the other.

## What changed, and what did not

- The standard bank previously built full target graphs during candidate
  validation, discarded them, then built them again for scoring. Validation now
  runs the same physical transform, adaptive ROI and required canonical
  coordinate checks without packing disposable node features or radius edges.
  Full target features/edges are built once for actual inference.
- The argmax builder already retained its full graphs; it did **not** have that
  same double-edge-build problem. It now separates geometry admission from
  measured parallel construction of the accepted full candidate graphs.
- Both builders reuse the prepared source and use `BankLocalGraphMapper` for
  complete target graphs. Actual pending work is measured in CPU concurrency
  waves; every result is retained in its original candidate slot even if worker
  completion order differs. No candidate, node or edge is dropped to save time.
- `PendingBankScorer` writes canonical samples to a unique, owned disk spool.
  Only the current physical batch is mmap-loaded for scoring. Fresh sample
  containers prevent collation from turning the pending canonical samples into
  permanently expanded views during calibration. After calibration, automatic
  flushes use the selected batch size rather than always waiting for the largest
  configured candidate batch.
- Checkpoints are initially loaded on CPU; model weights are copied to the
  requested device and the unused training-checkpoint state is released. Previous
  full-volume case inputs are released before loading the next CT.

Model depth/width, complete graph topology, voxel geometry, train/validation
cohorts, candidate ordering and seeds, 128 candidates per pool, CP constraints,
and GNN/nnU-Net training settings are unchanged. This is not a new L0/L1/L2
design, a different Medical Data Aug preprocessing policy, or a reduction of
the existing experiment. Geometry and resource failures still produce explicit
errors/statuses; they are not replaced with fabricated graphs or scores.

## Calibration, memory and remaining costs

The scoring physical batch counts **source/pool samples**, each with its full
128-candidate pool; it is not the number of candidates in a pool. The checked-in
scoring batch candidates remain `[1, 2, 4, 8, 16]`, with three repetitions.
The first automatic flush normally waits for 16 queued samples. An explicit
flush with fewer samples records larger batch candidates as unavailable instead
of inventing measurements. Thus the first GPU scoring call need not happen
immediately after the first patient's graph work.

Calibration compares the same queued cohort across batch sizes. It records
throughput, peak allocated VRAM and host measurements, rejecting unsafe trials
with evidence. Each executed batch also checks fresh host/cgroup input-memory
availability with headroom. Graph preparation similarly records CPU allocation,
sampled RSS and real-work wave timings. Waves process different candidates, so
their timing is not proof of a globally optimal worker count.

The spool bounds the lifetime of queued graph tensors across sources, not every
allocation in the pipeline. Assembling one complete 128-candidate sample still
requires memory; full-volume inputs, expanded inference views and model work
also consume RAM/VRAM. Input-budget estimates and sampled RSS are not hard total
memory guarantees. Native library threads are not independently tuned by the
outer graph-worker scheduler. Case loading and geometry admission remain
ordered; this change is not whole-pipeline CPU/GPU overlap or multi-GPU execution.

Disk capacity and I/O remain real costs. The spool adds writes and mmap loading;
OS caching can affect subsequent trials. `spool_io.read_seconds` measures mmap
deserialization/setup, while page-fault costs are included in scoring time.
Do not infer a server-wide speedup or an ETA from a small DEBUG measurement.

## Where to inspect progress

Paths below are relative to the actual bank directory used by the launcher
(`online/folds/fold_<n>/bank` within the feedback experiment).

| Output | Meaning |
| --- | --- |
| `preparation_progress.<uuid>.jsonl` | Per-invocation phase events and a default 30-second heartbeat, with elapsed time, case/source fields and resource snapshots. |
| `preparation_resources/candidate_graphs.<case>.<component>.<uuid>.resources.json` | CPU graph-preparation wave measurements and failure/resource evidence. |
| `preparation_resources/candidate_graphs.<case>.<component>.<uuid>.summary.json` | Successful ordered preparation, all target node/edge counts, shared source counts and resource report. |
| `index.json` → `scoring_execution` | Published scoring trials, selected physical batch, actual batch counts, host/VRAM evidence and spool I/O counters. |
| `manifest.csv`, `entries/`, `index.json`, `complete.json` | Bank entry publication and completion evidence, not interchangeable with progress logs. |

Console prefixes include `[BankProgress]`, `[BankGraphPreparation]`,
`[BankGraphPreparationComplete]`, `[BankScoringResources]`,
`[BankScoringProgress]` and `[BankScoringCalibration]`.
Progress phases distinguish `load_case`, `patient_regions`, `source_prepare`,
`candidate_search`, `candidate_validation`, `local_graphs`, `score_queue`,
and `score_remaining`. The scoring prefix provides finer-grained spool,
calibration and batch events while the enclosing phase is still active.

`[Bank] case 1/...` identifies the case being inspected, not a finished bank or
a restart of all preprocessing. Likewise, a progress `preparation_finished`
event only ends that monitored work scope. Final error-row, count, scoring,
entry/hash and completion checks still have to pass. A heartbeat is not proof
that the candidate count advanced or that the job will fit in memory.

The scratch `.bank-scoring-*` directory is unique to its invocation. Normal
completion and exception cleanup remove only this owned spool, not existing
bank entries or experiment outputs. Abrupt process/host termination can leave
scratch files. They are not reusable checkpoints; do not broadly delete files,
locks or journals to force continuation. Legacy completed banks need not contain
these new progress files.

## Existing results and safe continuation

Do not update the checkout/trainer package or start another process while an
experiment using them is active. A running process does not acquire this change
through `git pull`; later subprocesses could instead load a mixed code version.

A completed bank is reused only after its configuration/provenance, entry data,
counts/hashes and completion evidence pass verification. The scoring format
remains `hiercp_online_scoring_v1`; new reports additionally carry
`host_contract_format=hiercp_bank_scoring_host_v1`. Existing compatible reports
are not rewritten to pretend they contain newly measured telemetry.
Partial bank reuse still requires valid per-entry status, dimensions and hashes;
it is not a guarantee that every interrupted bank can be automatically resumed.
Existing `error` rows are refused; unresolved `insufficient_candidates`,
`unrepresentable_source` or `source_patch_too_large` rows prevent final
publication. These are distinct from a verified standard-bank `no_placement`
row accepted under its existing `retain_original` policy.

The experiment launcher adds its own saved-plan, source-identity, runtime and
stage-evidence checks. Its two resume options require an experiment originally
created with `--recover-from`; they do not resume ordinary fresh, non-recovery
launches. `--resume-preparation` is for eligible unfinished
preparation; after preparation has completed, use the original arguments with
`--resume-experiment` only when that continuation is accepted. Existing locks,
ambiguous running rows or missing/incompatible required checkpoints are refused.
Retain the saved Python executable as well as the original configurations.
Reusing a completed bank does not bypass the launcher's real-data, disk,
private-runtime or single-visible-GPU checks.
See [the recovery guide](feedback_recovery.md) for the server command and
the [feedback experiment guide](online_cp_feedback.md) for experiment scope.
Do not delete an existing bank, GNN cache, journal or preflight to install this
optimization, and do not use `--overwrite` as a generic recovery command.

## Verification status

For code commit `ebe58ff`, local Python 3.10 syntax checks and the complete
unit-test discovery run passed: **594 tests, 592 passed, 2 skipped**.
Synthetic DEBUG tests cover exact canonical tensor/edge/patch parity,
geometry rejection, source reuse, ordered parallel preparation, real bank
caller wiring with all 128 candidates, disk-spool lifecycle, scoring resource
guards, publication/reuse and failure cleanup. These are not patient-data runs.

An additional small CPU DEBUG comparison used the same three candidate graphs
in three repetitions: full target-edge construction calls fell from six to
three per repetition, with exact output parity. That observation applies to
the standard bank's removed duplicate work, not an argmax double-build claim
or a measured server/GPU speedup.

Real medical-data bank completion, server GPU throughput, full-size resource
feasibility, complete training/evaluation and downstream performance of this
updated execution path remain unverified here. This documentation update does
not launch or certify any of those experiments.
