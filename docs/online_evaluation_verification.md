# OnlineCP evaluation verification (no retraining)

This workflow reads existing full-cohort predictions. It does not change models,
training epochs, batch sizes, candidate banks, or the original trainer sources.
Dataset730/v2 ExactArgmax and Dataset740/ArgmaxV3 are separate experiments; do not
replace one dataset ID with the other to make an old result pass validation.

## Preserve the original result

Use a new output path. Existing directories, including empty/partial ones, are
refused. An interrupted run keeps its new partial directory for inspection;
choose another new path after fixing the reported problem. Do not delete or
overwrite old predictions, checkpoints, training logs, or result folders.

From the updated project checkout in the server's existing nnU-Net environment:

```bash
python tools/downstream_level_ablation.py evaluate \
  --project /home/aicompetition06/Medical/HierCP \
  --outer-fold 0 \
  --dataset-id 730 \
  --evaluation-output /home/aicompetition06/Medical/HierCP/work/online_basic_vs_hiercp/folds/fold_0/downstream_level_ablation_verified_01
```

This command evaluates existing predictions only. Do not substitute `all`,
`prepare`, or `train`: those commands have a different scope. If the target
already exists, choose a new explicit suffix; this command has no overwrite
option for evaluation results.

## What the audit establishes

- Exactly the expected validation cases, GT and prediction file SHA-256 hashes,
  declared trainer names, and the evaluation definition are recorded.
- Source files are checked again before completion. Full predictions, GT, and
  patient identities must match across the three pairwise comparisons.
- All four methods must have exactly the 250 unique epoch indices 0..249,
  positive sample counts, valid CP event counts and identical recorded schedules.
  The default duplicate policy is `error`: every repeated epoch fails with
  file/line evidence. The explicit policy below allows only identical recorded
  tuples, with a complete occurrence audit. Original logs remain untouched and
  their hashes are recorded.
- A legacy summary without input provenance is explicitly unverified, not a
  regression pass. Different experiment identities are not directly comparable.
  A metric mismatch on identical verified inputs fails even if the deprecated
  `--allow-regression-mismatch` flag is supplied.
- Recording existing prediction bytes does not retrospectively prove which
  checkpoint generated them. Old training/inference provenance remains a
  separate verification requirement.

The report's regression gate compares the complete recorded evaluation contract,
including code hashes, dependency versions and resampling settings. Changing
those fields yields `not_comparable`, even for identical GT/prediction bytes;
it is not a pass. This gate verifies replay under the same recorded contract,
not metric correctness across code revisions. The focused regression/statistical
unit tests are a separate check; real-data cross-version validation is still
required before claiming that an updated evaluator reproduces a past analysis.

An `evaluation_started.json` or report file alone is not completion. Check the
final `completion.json` and its bound summary/output hashes. A failed evaluation
must not be represented as a completed experiment.

## Repeated epoch logs and resume limitations

If logs repeat an epoch, inspect both reported source lines first. To explicitly
coalesce identical `(applied, samples, schedule)` tuples, select
`--schedule-duplicate-policy coalesce-identical` and a **new** output directory:

```bash
python tools/downstream_level_ablation.py evaluate \
  --project /home/aicompetition06/Medical/HierCP \
  --outer-fold 0 \
  --dataset-id 730 \
  --bootstrap-iterations 20000 \
  --permutation-iterations 50000 \
  --schedule-duplicate-policy coalesce-identical \
  --evaluation-output /home/aicompetition06/Medical/HierCP/work/online_basic_vs_hiercp/folds/fold_0/downstream_level_ablation_verified_resume_01
```

The policy applies to all four methods. It validates every occurrence and still
rejects conflicting duplicates, missing epochs, malformed counts, changed log
files and cross-method schedule mismatches. A third conflicting occurrence also
fails; no last-record-wins replacement is permitted. Schedule hex letter case is
normalized for comparison, but every original line is retained unchanged.

`schedule_audit.json` v3 records the selected policy, all source paths, line
numbers, full line text, parsed values and original file SHA-256 hashes. Both
`evaluation_started.json` and `completion.json` identify the policy. Audit
reverification compares all stored occurrence metadata with the original logs
under the caller's explicitly selected policy; the default strict verifier
does not silently accept a coalesced audit.

The report separates totals over unique epoch indices from totals over all raw
log occurrences. For example, two records of `epoch=100 applied=193/500` with
the same schedule fingerprint add 193 CP events and 500 samples to the **logged**
totals relative to the **unique** totals. Neither total proves how many optimizer
steps were executed. Equal fingerprints do not establish that checkpoint,
optimizer or LR state resumed correctly, or that each epoch ran exactly once.
The report therefore retains `training_resume_status: "unverified"`. This option
permits schedule-log comparison and existing-prediction evaluation, not a claim
that resumed training has been independently verified.

## Statistical interpretation

Patient-level pairing, metric definitions and full validation cohorts remain
unchanged. A zero-denominator permutation/bootstrap sample is not silently
discarded or assigned a made-up precision: the affected inference is unavailable
with a reason and valid/invalid resampling counts. Reported p-values are unadjusted
unless a separately specified analysis says otherwise.

Single-fold point estimates do not prove either graph level necessary or
unnecessary. Keep the prespecified primary endpoint and fold plan; do not select
a favorable metric after reading the results. In the supplied fold-0 report,
F1/false-positive improvement does not establish tumor Dice or small-lesion
segmentation improvement.

## Local debug tests versus real evaluation

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The focused tests use explicitly labelled debug numeric and byte fixtures. They
do not train a model, create a learned checkpoint, or claim medical performance.
Source/isolated-test success is separate from the target server's NIfTI smoke
test, full-cohort evaluation, and full training. Run the real evaluation only
with the actual dataset and the compatible server environment; no dataset is
downloaded by these unit tests.
