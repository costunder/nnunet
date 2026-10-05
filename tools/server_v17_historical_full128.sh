#!/usr/bin/env bash
set -euo pipefail

# Read-only evaluation. A separate code worktree/output keeps running D intact.
CP_GPU="${CP_GPU:?Set the physical GPU number}"
CP_OUTPUT="${CP_OUTPUT:?Set a NEW evaluation output directory}"
CP_WORKERS="${CP_WORKERS:-16}"
CP_CUDA_GIB="${CP_CUDA_GIB:-40}"
CP_RSS_GIB="${CP_RSS_GIB:-192}"
CP_RESIDENT_GIB="${CP_RESIDENT_GIB:-128}"
CP_PHYSICAL_BATCH_CANDIDATES="${CP_PHYSICAL_BATCH_CANDIDATES:-8 16 32}"
CP_BASELINE="${CP_BASELINE:-/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004}"
CP_HALF_A="${CP_HALF_A:-/home/aicompetition06/Medical/experiments/v15_m10_halfA_seed42_20261004}"
CP_HALF_B="${CP_HALF_B:-/home/aicompetition06/Medical/experiments/v16_m10_halfB_seed42_20261004_r3}"
CP_CROSSED_C="${CP_CROSSED_C:-/home/aicompetition06/Medical/experiments/v17_crossed_C_m10_seed42_20261005}"
CP_NATIVE_RUN="${CP_NATIVE_RUN:-/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42}"
read -r -a evaluation_batches <<< "$CP_PHYSICAL_BATCH_CANDIDATES"
evaluation_reuse=()
if [[ -n "${CP_PREPARED_CACHE:-}" ]]; then
  evaluation_reuse=(--prepared-cache "$CP_PREPARED_CACHE")
fi
printf 'Existing V1/A/B/C own-task BEST -> same whole validation P+128U.\n'
printf 'physical GPU=%s | L0 batches=%s | NEW output=%s\n' "$CP_GPU" "$CP_PHYSICAL_BATCH_CANDIDATES" "$CP_OUTPUT"
printf 'No training/checkpoint changes. V1/A recipient lesion annotation exposure is reported.\n'
python -B -u tools/evaluate_v17_historical_full128.py \
  --gpu "$CP_GPU" \
  --baseline "$CP_BASELINE" --half-a "$CP_HALF_A" --half-b "$CP_HALF_B" \
  --crossed-c "$CP_CROSSED_C" --native-run "$CP_NATIVE_RUN" --output "$CP_OUTPUT" \
  --workers "$CP_WORKERS" --physical-batch-candidates "${evaluation_batches[@]}" \
  --cuda-gib "$CP_CUDA_GIB" --rss-gib "$CP_RSS_GIB" --resident-gib "$CP_RESIDENT_GIB" \
  --minimum-free-gib 80 "${evaluation_reuse[@]}"
