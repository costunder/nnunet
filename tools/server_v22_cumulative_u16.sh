#!/usr/bin/env bash
set -euo pipefail

# New native v2.2 LocalCNN experiment; existing D/old CNN runs are untouched.
CP_GPU="${CP_GPU:-3}"
CP_MARGIN_MM="${CP_MARGIN_MM:-10}"
CP_WORKERS="${CP_WORKERS:-16}"
CP_CUDA_GIB="${CP_CUDA_GIB:-40}"
CP_RSS_GIB="${CP_RSS_GIB:-192}"
CP_RESIDENT_GIB="${CP_RESIDENT_GIB:-128}"
CP_DEVICE_CACHE_GIB="${CP_DEVICE_CACHE_GIB:-8}"
CP_NATIVE_CACHE="${CP_NATIVE_CACHE:-/home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json}"
CP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CP_ACTUAL_COMMIT="$(git -C "$CP_REPO_ROOT" rev-parse HEAD)"
CP_CODE_COMMIT="${CP_CODE_COMMIT:-$CP_ACTUAL_COMMIT}"
if [[ "$CP_CODE_COMMIT" != "$CP_ACTUAL_COMMIT" ]]; then
  printf 'Requested code differs from checkout HEAD. No experiment started.\n' >&2
  false
fi
CP_EXPERIMENT="${CP_EXPERIMENT:-/home/aicompetition06/Medical/experiments/v22_cnn_m${CP_MARGIN_MM}_cumulativeU16_gate70x2_seed42}"
cd -- "$CP_REPO_ROOT"
printf 'v2.2 LocalCNN | margin=%smm | physical GPU=%s | code=%s\n' "$CP_MARGIN_MM" "$CP_GPU" "$CP_ACTUAL_COMMIT"
printf 'New experiment=%s\nTRAIN U=16 -> 32 -> ... -> 128; all P retained; previous U retained.\n' "$CP_EXPERIMENT"
printf 'Gate: cumulative TRAIN pair-win >=0.7, mean bestP-minus-bestU >0, two consecutive epochs; Hit@1 is recorded only.\n'
printf '40 epochs; full128 validation/BEST and full observation support unchanged; physical batch32.\n'
printf 'If the gate stays below threshold, no forced promotion; incomplete128 exposure is recorded.\n'

python -B -u tools/run_local_cnn_experiment.py \
  --gpu "$CP_GPU" \
  --experiment "$CP_EXPERIMENT" \
  --cache "$CP_NATIVE_CACHE" \
  --margin-mm "$CP_MARGIN_MM" \
  --curriculum-config config/v22_cumulative_u16.json \
  --workers "$CP_WORKERS" \
  --cuda-gib "$CP_CUDA_GIB" --rss-gib "$CP_RSS_GIB" --resident-gib "$CP_RESIDENT_GIB" \
  --batch-candidates 32 \
  --support-patients 16 \
  --device-cache-gib "$CP_DEVICE_CACHE_GIB"
