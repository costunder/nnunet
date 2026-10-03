#!/usr/bin/env bash
# Execute with bash; do not source into a login/SSH shell.
# Foreground ranking only: native40 -> nested41640 -> full21 paired evaluation.
# No model/sampler/profile edits, CP generation, or nnU-Net launch.

: "${CP_GPU:?Set CP_GPU to the GPU number displayed in this server environment}"
CP_MEDICAL_ROOT="/home/aicompetition06/Medical"
CP_COMPARISON_ROOT="${CP_COMPARISON_ROOT:-$CP_MEDICAL_ROOT/experiments/v1_native_nested416_seed42_20261003}"
CP_NATIVE="$CP_COMPARISON_ROOT/native"
CP_NESTED="$CP_COMPARISON_ROOT/nested416"
CP_ADMISSION_ARGS=()
CP_RECOVERY_ARGS=()
if [[ -n "${CP_ROI_MAX_VOXELS:-}" || -n "${CP_ROI_PROBE:-}" ]]; then
  CP_ADMISSION_ARGS=(--roi-max-voxels "${CP_ROI_MAX_VOXELS:?Explicit ROI resource guard required}"
                     --roi-budget-probe "${CP_ROI_PROBE:?Completed ROI cost probe required}")
fi
if [[ -n "${CP_RECOVERY_SOURCE:-}" ]]; then
  CP_RECOVERY_ARGS=(--recover-preparation-from "$CP_RECOVERY_SOURCE")
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
python -u tools/run_v1x_experiment.py init \
  --experiment "$CP_NATIVE" --medical-root "$CP_MEDICAL_ROOT" \
  --local-sampling native --record-epochs "${CP_ADMISSION_ARGS[@]}" "${CP_RECOVERY_ARGS[@]}" &&
python -u tools/run_v1x_experiment.py run \
  --experiment "$CP_NATIVE" --stage v1.0 --target prepare &&
python -u tools/run_v1x_experiment.py run \
  --experiment "$CP_NATIVE" --stage v1.0 --target train --gpu "$CP_GPU" &&
python -u tools/run_v1x_experiment.py init \
  --experiment "$CP_NESTED" --medical-root "$CP_MEDICAL_ROOT" \
  --local-sampling strict_nested --role-seeds 64 32 96 64 96 64 \
  --reference-experiment "$CP_NATIVE" --record-epochs "${CP_ADMISSION_ARGS[@]}" &&
python -u tools/run_v1x_experiment.py run \
  --experiment "$CP_NESTED" --stage v1.0 --target train --gpu "$CP_GPU" &&
python -u tools/evaluate_v1_sampling_pair.py \
  --native-experiment "$CP_NATIVE" --nested-experiment "$CP_NESTED" \
  --gpu "$CP_GPU" --cuda-gib 40 --rss-gib 192 \
  --bootstrap-resamples 20000 --confidence .95 --bootstrap-seed 42 \
  --output "$CP_COMPARISON_ROOT/paired21_$(date +%Y%m%d_%H%M%S)"
