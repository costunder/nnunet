# CP input correctness repair

## Scope and baseline distinction

`Medical Data Aug.zip` is the user-provided preprocessing/basic-CP reference,
not an experiment output directory. It contains a batch script, a notebook,
and a `liver_70` image/label pair. It does not contain the failing `liver_76`.
Do not commit the archive or its medical images to the repository.

The earlier `code.txt` HierCP export is a different reference stage. Its
placement tests must not be described as proof of equivalence to that ZIP:

- ZIP batch: source padding 2, a nominal voxel-distance setting of 12,
  one requested paste, up to 4,000 proposed liver centers, non-fixed RNG.
- The batch's `uint8` inversion makes the occupied-distance calculation
  incorrect. The notebook's boolean single-case version does not have that
  inversion error. Reinstating the bug is not a valid baseline repair.
- Earlier HierCP: source padding 4, physical distance 12 mm, a 128-entry
  candidate pool and eight training candidates including the positive.
- A warning followed by an unchanged output in the old batch is not evidence
  that a particular donor was successfully pasted. Conversely, no-place
  evidence under HierCP's conditions does not prove no place under all policies.

Following the user's explicit request to repair against this archive, the
shared cache and generation profiles now restore source padding **2**, native
voxel center distance **12**, liver coverage **0.85**, and 6-neighbor occupied
clearance **2 voxels**. Physical center separation is disabled in this profile;
physical-mm distances remain physical-mm GNN features. Both unit thresholds
cannot be enabled together. Boolean mask inversion is used, not the batch's
uint8 inversion bug. Hard paste and the original HU jitter ranges are retained.

The research extension remains explicit: pool size **128**, **8** ranking
candidates, model/graph configuration, all scheduled source attempts and epochs
are unchanged. Its random proposal budget of 50,000 and exhaustive extension
are not the original script's 4,000-attempt sampler. This is restoration of the
reference geometry and no-placement behavior, not bitwise RNG equivalence to
the standalone script.
The existing donor-eligibility policy and reproducible seeded schedules also
remain research extensions, not a claim of exact standalone-script equivalence.

## No-placement behavior

The reference saves an unchanged CT/label pair when its chosen tumor cannot be
pasted. The new `retain_original` policy handles **conclusive raw full-search
zero-candidate** outcomes as an audited CP non-application, not whole-pipeline
failure. Unknown/failed searches, corrupt inputs, malformed geometry, source
resampling loss and nonzero incomplete candidate pools remain explicit errors.
No smaller tumor, different donor, duplicate center or fake graph is substituted.

- GNN cache: every scheduled attempt is resolved into a real training sample or
  a separately hashed `no_placement_entries` record. A zero-placement attempt
  has no valid ranking target and produces no `.pt` training sample. Reports
  keep materialized and resolved ratios separate; resolved 100% does **not**
  mean 100% of attempts contributed to GNN training. Empty train/validation
  graph splits are not acceptable final training inputs.
- Online bank: `source_slots_by_case` retains the complete ordered donor
  inventory, including non-applications. `entries_by_case` contains real NPZ
  entries only. Selecting an empty slot retains the original training case;
  it does not redirect probability to another lesion. Full and Basic preserve
  the paired random draw schedule and report non-applications separately.
- The actual `liver_76` files are not in the archive. This policy removes the
  structural abort for a proven zero-placement source; it does not establish
  that the new geometry creates valid placements for that patient's tumor.

The standalone GNN generator also saves an unchanged pair for an audited
first-copy zero-placement result and verifies its provenance on reuse. A
multi-copy request that pastes some but not all copies still follows its
existing explicit incomplete-output policy; the checked-in reference profile
requests one copy. Reports distinguish actual image processing from model
inference, since a retained original requires no GNN forward call.

## Fixed paths

1. **Same donor through preprocessing.** The raw selected component mask is
   mapped using the actual nnU-Net axis/crop/segmentation-resampling contract.
   A different nearest connected component is not a replacement for a donor
   which vanished or whose centroid is outside its nonconvex shape. Copy the
   mapped source mask itself, not the whole merged preprocessed component.
   Keep preprocessed CT values in float32 without a float16 round trip.
2. **GNN training input binding.** Compare current cache/source/placement and
   label settings against the graph cache metadata before using it for training.
   Do not conflate a model optimization seed with the cache-generation seed.
   Validate inputs, production cohort completeness, and device selection before
   any explicitly requested checkpoint overwrite. Failed preflight must preserve
   both the trained checkpoint and the optimizer-resume checkpoint.
3. **Exact candidate work reuse.** Reuse a completed exhaustive search for the
   same source, RNG state, conditions, and exclusion set. New exclusions still
   require the appropriate search. Failure evidence remains explicit.
4. **Exact overlap short-circuit.** Retire a candidate only after an actual
   source-mask voxel overlaps forbidden tissue. Evaluate full coverage for
   every remaining center. This changes computation, not the valid domain or
   output order. Diagnostics count both performed and avoided mask operations.

## Artifact boundaries

Existing checkpoints, datasets and experiment outputs remain evidence for their
original contracts. Do not delete them or label them valid for a changed one.
Changed training graphs require a checkpoint compatibility assessment; model
architecture code is not automatically invalidated by a placement fix.
Region-only population prototypes need not be refitted merely because candidate
conditions change, if their inputs and split remain identical and verified.
Source-mapping changes require the corresponding online bank's provenance to
match; a bank carrying the old donor-selection policy is not silently adopted.

The source-mapping policy is checked by the native bank verifier and by
`tools.train_online_feedback` for both Full and Basic, including its dry-run
preflight. The check reads the index and config bound by the verified bank
contract. The generic installed contract helper remains unchanged, but the
paired and feedback trainers now understand explicit no-placement source slots.
Their source checksums must match the installer inventory. This does not update
an independently installed or previously copied private runtime in place.

## Recovery and shared preprocessing

Changing CP geometry does not mean CT preprocessing must differ between Basic
and Full: both arms still consume the same nnU-Net preprocessing and bank
contract. The recovery transition allows only the documented old-to-reference
CP setting changes. It preserves verified splits, region features, prototypes
and source inputs; CP-dependent GNN sample tensors are regenerated because
their candidate masks and targets depended on the old geometry. Old checkpoints
and results are preserved as old-contract evidence, not relabelled as new runs.

Use a **new experiment root** for this transition and its updated private
trainers. Do not overwrite an old execution journal, refresh old runtime files
behind its hashes or present `--resume-preparation` on the old root as a
compatible retry. Recovery regression tests cover source immutability and the
distinction between shared features and CP-dependent training graphs.

## Verification limits

Synthetic regression tests are marked DEBUG and do not change production
defaults. A real-label predicate comparison is not a complete candidate census,
GNN validation, downstream segmentation evaluation, or proof about `liver_76`.
Full server training/evaluation must be reported separately. Nonzero but
insufficient candidate sets remain explicit failures, not fabricated 128-entry
banks. DEBUG tests exercise anisotropic native-voxel reference predicates,
physical GNN features, unchanged non-pasted voxels, complete attempt accounting,
no-placement source RNG slots, artifact tamper rejection and safe recovery.
The user archive, original data and old experiment results remain untouched.
