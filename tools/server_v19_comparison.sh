#!/usr/bin/env bash
set -euo pipefail

CP_GPU="${CP_GPU:-3}"
CP_ARM="${CP_ARM:-all}"
CP_EXPERIMENT="${CP_EXPERIMENT:-/home/aicompetition06/Medical/experiments/v19_comparison_m10_seed42}"
CP_BASELINE="${CP_BASELINE:-/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004}"
CP_INVENTORY="${CP_INVENTORY:-/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42/inventory/index.json}"
CP_WORKERS="${CP_WORKERS:-16}"
CP_CUDA_GIB="${CP_CUDA_GIB:-40}"
CP_RSS_GIB="${CP_RSS_GIB:-192}"
CP_RESIDENT_GIB="${CP_RESIDENT_GIB:-128}"
CP_BATCH_CANDIDATES="${CP_BATCH_CANDIDATES:-1 2 4 8 16 32}"
CP_VALIDATION_LOCAL_CHUNK="${CP_VALIDATION_LOCAL_CHUNK:-8}"
comparison_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
read -r -a comparison_candidates <<< "$CP_BATCH_CANDIDATES"
cd -- "$comparison_root"
printf 'v1.9 | original v1 m10 | arm=%s | physical GPU=%s | experiment=%s\n' "$CP_ARM" "$CP_GPU" "$CP_EXPERIMENT"
printf 'all: selected -> native -> native_fixed -> native_listwise; fresh identical init per arm, full129 joint evaluation.\n'
python -B -u tools/run_v19_comparison.py \
  --gpu "$CP_GPU" --arm "$CP_ARM" --experiment "$CP_EXPERIMENT" \
  --baseline "$CP_BASELINE" --inventory "$CP_INVENTORY" \
  --workers "$CP_WORKERS" --batch-candidates "${comparison_candidates[@]}" \
  --validation-local-chunk "$CP_VALIDATION_LOCAL_CHUNK" \
  --cuda-gib "$CP_CUDA_GIB" --rss-gib "$CP_RSS_GIB" --resident-gib "$CP_RESIDENT_GIB"
python -B tools/summarize_v19_comparison.py --experiment "$CP_EXPERIMENT"
