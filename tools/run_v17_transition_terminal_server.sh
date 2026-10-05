#!/usr/bin/env sh
# Read-only collection and terminal report; no training or GPU allocation.
# Run with bash, never source this script into the user's SSH/login shell.
set -eu

CP_V17_CODE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
CP_V17_MEDICAL_ROOT="${CP_MEDICAL_ROOT:-/home/aicompetition06/Medical}"
CP_V17_STAMP="$(date +%Y%m%d_%H%M%S)"
CP_V17_OUTPUT="${CP_TRANSITION_OUTPUT:-${CP_V17_MEDICAL_ROOT}/experiments/v17_transition_receipt_${CP_V17_STAMP}_$$}"

python -B -u "${CP_V17_CODE_ROOT}/tools/export_v1_v22_transition_receipt.py" \
  --native-run "${CP_V17_MEDICAL_ROOT}/experiments/v22_cnn_m10_seed42" \
  --baseline "${CP_V17_MEDICAL_ROOT}/experiments/v1_m10_seed42_20261004" \
  --half-a "${CP_V17_MEDICAL_ROOT}/experiments/v15_m10_halfA_seed42_20261004" \
  --half-b "${CP_V17_MEDICAL_ROOT}/experiments/v16_m10_halfB_seed42_20261004_r3" \
  --source-root "${CP_V17_MEDICAL_ROOT}/HierCP-v1-47bdb58" \
  --source-root "${CP_V17_MEDICAL_ROOT}/HierCP-cnn-cd26119" \
  --output "${CP_V17_OUTPUT}"
