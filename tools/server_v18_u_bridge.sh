#!/usr/bin/env bash
set -euo pipefail

CP_GPU="${CP_GPU:-3}"
CP_ARM="${CP_ARM:-both}"
CP_EXPERIMENT="${CP_EXPERIMENT:-/home/aicompetition06/Medical/experiments/v18_u_bridge_m10_seed42}"
CP_BASELINE="${CP_BASELINE:-/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004}"
CP_INVENTORY="${CP_INVENTORY:-/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42/inventory/index.json}"
CP_WORKERS="${CP_WORKERS:-16}"
CP_CUDA_GIB="${CP_CUDA_GIB:-40}"
CP_RSS_GIB="${CP_RSS_GIB:-192}"
CP_RESIDENT_GIB="${CP_RESIDENT_GIB:-128}"
CP_BATCH_CANDIDATES="${CP_BATCH_CANDIDATES:-1 2 4 8 16 32}"
CP_VALIDATION_LOCAL_CHUNK="${CP_VALIDATION_LOCAL_CHUNK:-8}"
bridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
read -r -a bridge_candidates <<< "$CP_BATCH_CANDIDATES"
cd -- "$bridge_root"
printf 'v1.8 | original v1 m10 | arm=%s | physical GPU=%s | fresh paired root=%s\n' "$CP_ARM" "$CP_GPU" "$CP_EXPERIMENT"
printf '1P+7 comparisons/update;128U rotate across epochs. BOTH arms full129 joint validation and fresh identical init.\n'
python -B -u tools/run_v18_u_bridge.py \
  --gpu "$CP_GPU" --arm "$CP_ARM" --experiment "$CP_EXPERIMENT" \
  --baseline "$CP_BASELINE" --inventory "$CP_INVENTORY" \
  --workers "$CP_WORKERS" --batch-candidates "${bridge_candidates[@]}" \
  --validation-local-chunk "$CP_VALIDATION_LOCAL_CHUNK" \
  --cuda-gib "$CP_CUDA_GIB" --rss-gib "$CP_RSS_GIB" --resident-gib "$CP_RESIDENT_GIB"
python -B tools/summarize_v18_u_bridge.py --experiment "$CP_EXPERIMENT"
