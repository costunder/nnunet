# HierCP OnlineCP custom nnU-Net trainers

The current bundle contains ten versioned trainer/helper modules: historical
paired/argmax trainers, rank-curriculum trainers, and the current Full/Basic
segmentation-feedback path with raw-target resampling/storage helpers.

The original two trainer modules came from the user-provided
`onlinecp_custom_trainers_bundle.zip` (historical archive SHA-256:
`22fb7ef5b021cf268691bbbd53128f8cfa5b512a857acac14e6b0a5634eb52c9`).
The current sources have been revised and are **not** byte-identical to that
archive. `SHA256SUMS` and the installer's `MODULES` bind all ten current sources.
Different existing files require `--overwrite`, each replaced
file gets a unique backup, writes are atomic, and failed installation restores
the files changed by that attempt. Identical installed trainers are reused.
The `.gitattributes` file preserves LF endings so Git checkout does not invalidate
the current source hashes on Windows.

## Files

### `nnUNetTrainer_OnlinePairedCP.py`

Single-pool OnlineCP implementation used by Dataset730 and the downstream level
ablation. It provides:

- `nnUNetTrainer_250epochs_OnlineBasicCP`
- `nnUNetTrainer_250epochs_OnlineHierCP` (legacy top-k-random reproduction)
- `nnUNetTrainer_250epochs_OnlineHierCPExactArgmax`
- `nnUNetTrainer_250epochs_OnlineHierCPNoPatientExactArgmax`
- `nnUNetTrainer_250epochs_OnlineHierCPNoPopulationExactArgmax`

Historical bank envelope: `hiercp_online_bank_v2`. This module also supplies the
shared loader used by current feedback subclasses; the legacy classes listed
above do not consume raw-target banks. The envelope alone is not a type check.

### `nnUNetTrainer_OnlinePairedCPArgmaxV3.py`

Multi-pool exact-argmax experiment module. It provides:

- `nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3`
- `nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3`

Expected bank format: `hiercp_online_bank_argmax_v3`.

### Current feedback and transport modules

- `nnUNetTrainer_OnlineCPCurriculum.py`: shared curriculum lifecycle plus historical
  rank-only classes; policy and provenance helpers are
  `onlinecp_curriculum_policy.py` and `onlinecp_curriculum_contract.py`.
- `nnUNetTrainer_OnlineCPFeedback.py`: current
  `nnUNetTrainer_250epochs_OnlineHierCPFeedback` and
  `nnUNetTrainer_250epochs_OnlineBasicCPFeedbackControl`.
  `onlinecp_feedback_policy.py` and `onlinecp_feedback_metrics.py` implement
  training-only difficulty selection and observation.
- `onlinecp_raw_resampling.py`: candidate-conditioned raw CT/label paste followed
  by exact native CT normalization/resampling, evaluated in the selected crop.
- `onlinecp_raw_bank.py`: immutable typed payloads, source-array sharing, mmap
  loading, full audit and checked file-witness reuse.

Current index/config declarations are `paste_contract=onlinecp_raw_target_paste_v1`,
`source_mapping_format=online_cp_raw_target_resampling_v2`, and
`entry_storage=npz_candidate_refs_raw_target_v1`. Only the current Full/Basic
Feedback classes accept this path. All raw candidates are retained, including
legitimate zero-native-support events and partially observed large lesions.
The ordinary segmentation update remains active; unavailable lesion difficulty
is not replaced with a zero observation. See [feedback design](../docs/online_cp_feedback.md).

## Installation into the active nnU-Net environment

For the current experiment, prefer the repository's verified
`tools/run_feedback_experiment.py --upgrade-bank-from ...` workflow described in
[README](../README.md). It installs into a new private runtime and preserves the
old experiment. Do not replace the installation of an active or preserved run.
The following are explicit manual installation commands, not an instruction to
overwrite the user's existing runtime.

Linux/macOS:

```bash
conda activate nnunet
python install_onlinecp_custom_trainers.py check
python install_onlinecp_custom_trainers.py apply
```

Windows PowerShell:

```powershell
conda activate nnunet
python .\install_onlinecp_custom_trainers.py check
python .\install_onlinecp_custom_trainers.py apply
```

For an nnU-Net source checkout rather than the active environment:

```powershell
python .\install_onlinecp_custom_trainers.py apply `
  --nnunet-root C:\path\to\nnunetv2
```

The target directory is:

```text
nnunetv2/training/nnUNetTrainer/
```

Run these commands from this `custom_trainers` directory. From the repository
root, prefix the script with `custom_trainers/`. If `apply` reports different
existing implementations, inspect the printed target paths before explicitly
adding `--overwrite`. Backups remain alongside the installed files for recovery.

## Runtime variables

The trainer reads:

```text
ONLINE_CP_BANK=/absolute/path/to/index.json
ONLINE_CP_SEED=42
```

These are shared base variables, not the complete current feedback launch
contract. Use `tools.train_online_feedback --help` or the experiment runner for
the policy, bank provenance, difficulty GNN and strict checkpoint arguments.

Validation does not perform Copy-Paste. The OnlineCP loader is used only for the
training loader.

## Verification

After installation:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import nnUNetTrainer_250epochs_OnlineBasicCP, nnUNetTrainer_250epochs_OnlineHierCPExactArgmax; print('OK')"
```
