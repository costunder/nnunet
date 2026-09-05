# HierCP OnlineCP custom nnU-Net trainers

This bundle restores the two custom trainer modules that were installed into
`nnunetv2` at runtime but were not stored in the project source tree.

Both trainer modules are byte-for-byte copies from the user-provided
`onlinecp_custom_trainers_bundle.zip` (archive SHA-256:
`22fb7ef5b021cf268691bbbd53128f8cfa5b512a857acac14e6b0a5634eb52c9`).
`SHA256SUMS` covers these two unchanged trainer sources. The installer has been
hardened locally: different existing files require `--overwrite`, each replaced
file gets a unique backup, writes are atomic, and failed installation restores
the files changed by that attempt. Identical installed trainers are reused.
The `.gitattributes` file preserves LF endings so Git checkout does not invalidate
the original source hashes on Windows.

## Files

### `nnUNetTrainer_OnlinePairedCP.py`

Single-pool OnlineCP implementation used by Dataset730 and the downstream level
ablation. It provides:

- `nnUNetTrainer_250epochs_OnlineBasicCP`
- `nnUNetTrainer_250epochs_OnlineHierCP` (legacy top-k-random reproduction)
- `nnUNetTrainer_250epochs_OnlineHierCPExactArgmax`
- `nnUNetTrainer_250epochs_OnlineHierCPNoPatientExactArgmax`
- `nnUNetTrainer_250epochs_OnlineHierCPNoPopulationExactArgmax`

Expected bank format: `hiercp_online_bank_v2`.

### `nnUNetTrainer_OnlinePairedCPArgmaxV3.py`

Multi-pool exact-argmax experiment module. It provides:

- `nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3`
- `nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3`

Expected bank format: `hiercp_online_bank_argmax_v3`.

## Installation into the active nnU-Net environment

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

Validation does not perform Copy-Paste. The OnlineCP loader is used only for the
training loader.

## Verification

After installation:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import nnUNetTrainer_250epochs_OnlineBasicCP, nnUNetTrainer_250epochs_OnlineHierCPExactArgmax; print('OK')"
```
