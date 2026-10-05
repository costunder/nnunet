#!/usr/bin/env bash
set -euo pipefail

# Run from the reviewed new code checkout; select exactly one authorized arm.
CP_ARM="${CP_ARM:-D}"
CP_GPU="${CP_GPU:-3}"
CP_WORKERS="${CP_WORKERS:-16}"
CP_CUDA_GIB="${CP_CUDA_GIB:-40}"
CP_RSS_GIB="${CP_RSS_GIB:-192}"
CP_RESIDENT_GIB="${CP_RESIDENT_GIB:-128}"
CP_BASELINE="${CP_BASELINE:-/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004}"
CP_NATIVE_RUN="${CP_NATIVE_RUN:-/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42}"
CP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CP_ACTUAL_CODE_COMMIT="$(git -C "$CP_REPO_ROOT" rev-parse HEAD)"
CP_CODE_COMMIT="${CP_CODE_COMMIT:-$CP_ACTUAL_CODE_COMMIT}"
if [[ "$CP_CODE_COMMIT" != "$CP_ACTUAL_CODE_COMMIT" ]]; then
  printf 'Requested code commit %s differs from checkout HEAD %s. No work started.\n' "$CP_CODE_COMMIT" "$CP_ACTUAL_CODE_COMMIT" >&2
  false
fi
export CP_CODE_COMMIT
cd -- "$CP_REPO_ROOT"

case "$CP_ARM" in
  C)
    CP_PHYSICAL_BATCH_CANDIDATES="${CP_PHYSICAL_BATCH_CANDIDATES:-1 2 4}"
    CP_BATCH_UNIT="original curriculum samples; eight query candidates per sample"
    ;;
  D)
    CP_PHYSICAL_BATCH_CANDIDATES="${CP_PHYSICAL_BATCH_CANDIDATES:-32}"
    CP_BATCH_UNIT="native observation rows; two genuine original graph views per row"
    ;;
  *)
    printf 'Invalid CP_ARM=%s; choose C or D. No work started.\n' "$CP_ARM" >&2
    false
    ;;
esac

CP_RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
CP_OUTPUT="${CP_OUTPUT:-/home/aicompetition06/Medical/experiments/v17_crossed_${CP_ARM}_m10_seed42_${CP_RUN_STAMP}}"
read -r -a crossed_physical_batch_candidates <<< "$CP_PHYSICAL_BATCH_CANDIDATES"

printf 'v1.7 crossed arm=%s | code=%s\n' "$CP_ARM" "$CP_CODE_COMMIT"
printf 'batch unit=%s | explicit candidates=%s\n' "$CP_BATCH_UNIT" "$CP_PHYSICAL_BATCH_CANDIDATES"
printf 'baseline=%s\nnative=%s\nnew output=%s\n' "$CP_BASELINE" "$CP_NATIVE_RUN" "$CP_OUTPUT"
printf 'Production40 epochs; whole native P+128U benchmark; no other arm starts.\n'

python -B -u tools/run_v17_crossed_training.py \
  --arm "$CP_ARM" \
  --baseline "$CP_BASELINE" \
  --native-run "$CP_NATIVE_RUN" \
  --output "$CP_OUTPUT" \
  --gpu "$CP_GPU" \
  --workers "$CP_WORKERS" \
  --physical-batch-candidates "${crossed_physical_batch_candidates[@]}" \
  --cuda-gib "$CP_CUDA_GIB" \
  --rss-gib "$CP_RSS_GIB" \
  --resident-gib "$CP_RESIDENT_GIB"
