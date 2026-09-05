# Opt-in OnlineCP curriculum: a separate experiment

The original `nnUNetTrainer_OnlinePairedCP.py` and `nnUNetTrainer_OnlinePairedCPArgmaxV3.py`
are unchanged. Do not reinterpret their checkpoints, schedules, or fold-0 results
as results of this new policy. New trainer names isolate nnU-Net result paths.

## What changes, and what does not

The new full-policy trainer samples from an explicitly scheduled, expanding
top-ranked prefix of the **same 128 anatomical candidates**. Each stage specifies
its boundaries, rank support, minimum number of score-eligible choices, and
temperature. `null` temperature means uniform sampling within that support;
a positive temperature means score-softmax sampling. A caller supplies the
existing candidate RNG draw: selection never consumes extra random numbers.

All 250 epochs, CP probability 0.5, source selection, intensity-jitter draws,
and original v2 training epoch/worker RNG draw order are retained. No source or
candidate is removed from the bank; the stage's selection support is explicitly
narrower. No model layer, graph node, or graph edge is dropped by this module.
The original anatomical/overlap checks remain the bank's prerequisite, not a
replacement for score eligibility. Insufficient eligible choices raise an error:
there is no silent argmax fallback, score-gate relaxation, or CP-event skip.

`score_floor` and `max_score_drop` constrain **learned scores**, not calibrated
medical probability. Lower-ranked does not mean a valid “hard positive,” and
this trainer does not create such graph-training labels. The top-rank expansion
must be evaluated using training-only model-selection data. Never choose these
settings by inspecting held-out downstream results.

The initial separate configuration proposed for this experiment uses five
50-epoch stages with top ranks 1, 4, 8, 16, 32; minimum choices 1, 2, 2, 4, 4;
uniform stage sampling; no absolute score floor; and maximum score drop 2.0.
These are explicit, **unvalidated initial design choices**, not measured optimal
settings. In particular, drop 2.0 can be permissive depending on score scale.
Bank publication must preflight every stage of every source. No code in this
module silently rewrites these settings or calls low-scoring placements safe.

## Required configuration and cohort verification

Set `ONLINE_CP_CURRICULUM_CONFIG` to the new explicit JSON path, alongside the
existing `ONLINE_CP_BANK` and `ONLINE_CP_SEED`. There is no embedded production
curriculum default. The schema is `onlinecp_rank_curriculum_v1`; the policy
validator rejects unknown fields, partial stages, a shortened horizon, fewer
candidates, reduced CP probability, and a curriculum that remains argmax-only.

The separately installed `onlinecp_curriculum_contract.py` must provide:

```python
verify_curriculum_bank_contract(
    bank_path, *, curriculum_sha256, expected_candidate_count,
    dataset_name, nnunet_fold,
) -> dict
```

It must verify actual bank/cohort/dataset/source hashes and return stable JSON
identity with `train_case_ids` and `validation_case_ids`. Missing provenance is
an error. The trainer also compares those lists against the actual nnU-Net
training/validation dataset keys before constructing its loaders. Merely adding
matching patient names to a JSON file is not an independently verified contract.

For publication-time score eligibility checks, use
`eligible_candidate_indices(scores, validated_config, epoch)` at each stage start
for every source. This does not evaluate downstream held-out patients.

## Paired control and audit

Available new classes are:

- `nnUNetTrainer_250epochs_OnlineBasicCPCurriculumControl`: uniform over the whole
  geometry-valid pool, not narrowed by learned scores.
- `nnUNetTrainer_250epochs_OnlineHierCPCurriculum`.
- `nnUNetTrainer_250epochs_OnlineHierCPNoPatientCurriculum`.
- `nnUNetTrainer_250epochs_OnlineHierCPNoPopulationCurriculum`.

Use the same configuration/seed/cohort/plans/physical batch/worker settings for
the paired arms. Ablation classes require the corresponding score-bank metadata.
The current publisher and strict launch wrapper support **Basic and Full only**.
The two ablation classes are implemented, but publishing independently verified
derived ablation-bank contracts is not yet integrated; they are not advertised
as completed or verified ablation launch paths.
The new `[OnlineCPCurriculum]` record contains epoch, stage, CP counts, config
SHA, event digest, and candidate-choice digest. Basic and curriculum should have
the same event digest but generally different choice digests. Do not feed these
records into a legacy `[OnlineCP]` schedule parser or call a matching event hash
proof that network weights are identical.

## Resume and resource scope

Only complete train-plus-validation epoch boundaries are saved atomically. The
checkpoint includes network, optimizer, scaler, scheduler, logger, Python/NumPy/
CPU/CUDA RNG state, exact curriculum/bank identity, installed trainer/helper source
hashes, plans, physical batch, iterations, workers, and the last epoch audit.
The last-epoch record must include valid event/choice digests, the matching stage,
and exactly physical batch times training iterations sampled items.
Legacy or incomplete checkpoints fail;
existing checkpoints without explicit resume cannot be silently overwritten by
a fresh training start. Validation augmentation is also restarted by epoch, so
its stream does not depend on how many earlier validation batches a prior process
consumed. Bitwise CUDA equivalence is not claimed.

Use `tools/train_online_curriculum.py` for the supported Basic/Full launches.
Its `--resume` path rejects a missing final/latest/best checkpoint before spawning
nnU-Net, and its fresh-start path refuses an already populated result directory.
Do not substitute raw `nnUNetv2_train --c` as a strict-resume guarantee: the base
CLI can warn and start fresh when a requested checkpoint is absent.

This implementation supports the existing single-device nnU-Net path. It refuses
DDP rather than pretending to restore all ranks from rank-0 RNG state. It does
not shrink the physical batch selected by nnU-Net. Hardware measurement and batch/
worker calibration must come from the experiment's resource preflight; the policy
module is not itself a GPU throughput benchmark. A resource-limited or unsupported
allocation must be reported rather than silently reduced.

## Verification status

`tests/test_online_cp_curriculum.py` uses explicitly debug numeric vectors and
framework scaffolding to exercise policy boundaries and the **actual inherited
five-draw sampling method with the new selection hook**. It also checks actual
loader wiring and missing-resume failure. These tests do not construct a medical
crop or execute nnU-Net forward/backward. Full installed-trainer loading, real
checkpoint replay, GPU throughput, training, and downstream efficacy require
separate target-environment validation. No dataset or learned checkpoint is
generated by the policy tests.
