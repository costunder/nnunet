#!/usr/bin/env bash
set -euo pipefail

CP_GPU="${CP_GPU:-3}"
CP_ARM="${CP_ARM:-native_fixed}"
case "$CP_ARM" in
  native_fixed|native_listwise) ;;
  *) printf 'Select one additional arm: native_fixed or native_listwise. Existing v1.8 jobs are separate.\n' >&2; false ;;
esac
CP_EXPERIMENT="${CP_EXPERIMENT:-/home/aicompetition06/Medical/experiments/v19_${CP_ARM}_m10_seed42}"
CP_REFERENCE="${CP_REFERENCE:-/home/aicompetition06/Medical/experiments/v18_u_bridge_m10_seed42}"
CP_BASELINE="${CP_BASELINE:-/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004}"
CP_INVENTORY="${CP_INVENTORY:-/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42/inventory/index.json}"
CP_CUDA_GIB="${CP_CUDA_GIB:-40}"
CP_RSS_GIB="${CP_RSS_GIB:-192}"
CP_RESIDENT_GIB="${CP_RESIDENT_GIB:-128}"
CP_VALIDATION_LOCAL_CHUNK="${CP_VALIDATION_LOCAL_CHUNK:-8}"
comparison_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$comparison_root"
read -r comparison_batch comparison_workers < <(python -B tools/v19_reference_execution.py --reference "$CP_REFERENCE")
CP_WORKERS="${CP_WORKERS:-$comparison_workers}"
CP_BATCH_CANDIDATES="${CP_BATCH_CANDIDATES:-$comparison_batch}"
if [[ "$CP_WORKERS" != "$comparison_workers" || "$CP_BATCH_CANDIDATES" != "$comparison_batch" ]]; then
  printf 'Existing v1.8 lock is batch=%s workers=%s. This additional-arm launcher requires the same execution settings.\n' "$comparison_batch" "$comparison_workers" >&2
  false
fi
read -r -a comparison_candidates <<< "$CP_BATCH_CANDIDATES"
printf 'v1.9 | original v1 m10 | arm=%s | physical GPU=%s | experiment=%s\n' "$CP_ARM" "$CP_GPU" "$CP_EXPERIMENT"
printf 'One additional arm per invocation; arm-specific output/cache/checkpoint. Existing v1.8 selected/native are not started.\n'
printf 'Read-only v1.8 reference=%s | physical source batch=%s | workers=%s\n' "$CP_REFERENCE" "$comparison_batch" "$comparison_workers"
printf 'Initial calibration uses short cloned probes; only the selected arm receives 40-epoch training.\n'
python -B -u tools/run_v19_comparison.py \
  --gpu "$CP_GPU" --arm "$CP_ARM" --experiment "$CP_EXPERIMENT" \
  --baseline "$CP_BASELINE" --inventory "$CP_INVENTORY" \
  --workers "$CP_WORKERS" --batch-candidates "${comparison_candidates[@]}" \
  --validation-local-chunk "$CP_VALIDATION_LOCAL_CHUNK" \
  --cuda-gib "$CP_CUDA_GIB" --rss-gib "$CP_RSS_GIB" --resident-gib "$CP_RESIDENT_GIB"
python -B tools/summarize_v19_comparison.py --experiment "$CP_EXPERIMENT" --arms "$CP_ARM"
