#!/usr/bin/env bash
set -euo pipefail

CP_GPU="${CP_GPU:?Set the physical GPU number for this arm}"
CP_ARM="${CP_ARM:?Set one existing arm: selected, native, native_fixed, native_listwise}"
cache_base="${CP_EXPERIMENTS_DIR:-/home/aicompetition06/Medical/experiments}"
cache_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$cache_code"
printf 'Independent arm=%s | physical GPU=%s | preserve saved progress\n' "$CP_ARM" "$CP_GPU"
printf 'Each arm owns its data, lock and checkpoints. Ctrl+C allows 10s for cooperative stop.\n'
# Route from arm, not an inherited CP_EXPERIMENT left by another terminal run.
# Custom roots are explicit --experiment options to run_comparison_arm.py.
python -B -u tools/run_comparison_arm.py \
  --gpu "$CP_GPU" --arm "$CP_ARM" --experiments-dir "$cache_base" \
  --inventory "${CP_INVENTORY:-$cache_base/v22_cnn_m10_seed42/inventory/index.json}"
