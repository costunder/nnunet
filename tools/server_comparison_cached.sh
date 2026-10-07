#!/usr/bin/env bash
set -euo pipefail

CP_GPU="${CP_GPU:-3}"
CP_ARM="${CP_ARM:?Set one existing arm: selected, native, native_fixed, native_listwise}"
cache_base="/home/aicompetition06/Medical/experiments"
case "$CP_ARM" in
  selected|native)
    cache_default="$cache_base/v18_${CP_ARM}_m10_seed42_memory"
    if [[ ! -f "$cache_default/continuation.json" ]]; then
      cache_default="$cache_base/v18_u_bridge_m10_seed42"
    fi
    ;;
  native_fixed|native_listwise)
    cache_default="$cache_base/v19_${CP_ARM}_m10_seed42"
    ;;
  *) printf 'Unknown arm: %s\n' "$CP_ARM" >&2; false ;;
esac
CP_EXPERIMENT="${CP_EXPERIMENT:-$cache_default}"
CP_INVENTORY="${CP_INVENTORY:-$cache_base/v22_cnn_m10_seed42/inventory/index.json}"
cache_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$cache_code"
printf 'Resume existing arm=%s | physical GPU=%s | experiment=%s\n' "$CP_ARM" "$CP_GPU" "$CP_EXPERIMENT"
printf 'Keep saved model/Adam/cursor, measured batch/workers, 10mm and candidate/loss policy.\n'
printf 'Reuse completed exact preprocessing; checkpoints and logs remain arm-owned.\n'
python -B -u tools/resume_comparison_cached.py \
  --gpu "$CP_GPU" --arm "$CP_ARM" --experiment "$CP_EXPERIMENT" \
  --inventory "$CP_INVENTORY" \
  --cache-sources \
    "$cache_base/v18_u_bridge_m10_seed42/data" \
    "$cache_base/v18_selected_m10_seed42_memory/data" \
    "$cache_base/v18_native_m10_seed42_memory/data" \
    "$cache_base/v19_native_fixed_m10_seed42/data" \
    "$cache_base/v19_native_listwise_m10_seed42/data"
