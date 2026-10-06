#!/usr/bin/env bash
# Read-only native30 BEST evaluation. Run with bash; no session-control commands.

native30_main() {
  local cp_repo cp_source cp_original_commit cp_output
  local -a cp_reuse_args=()
  cp_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
  cp_original_commit=a818158a81fc09b9d11d2774d53fba152bb2b453
  cp_source="${CP_ORIGINAL_SOURCE:-/home/aicompetition06/Medical/HierCP-native30-source-a818158}"
  cp_output="${CP_OUTPUT:-/home/aicompetition06/Medical/experiments/v1_native30_full128_$(date +%Y%m%d_%H%M%S)_$$}"
  if [ -z "${CP_GPU:-}" ]; then
    printf 'Set CP_GPU to the current physical GPU number.\n' >&2
    return 1
  fi
  if [ ! -d "$cp_source" ]; then
    git -C "$cp_repo" worktree add --detach "$cp_source" "$cp_original_commit" || return 1
  fi
  test "$(git -C "$cp_source" rev-parse HEAD)" = "$cp_original_commit" || return 1
  git -C "$cp_source" diff --quiet HEAD -- hiercp config/train.json || return 1
  printf 'Original V1 native ROI30/context28 BEST epoch30 | complete P+128U evaluation only\n'
  printf 'Physical GPU=%s | source=%s | output=%s\n' "$CP_GPU" "$cp_source" "$cp_output"
  if [ -n "${CP_REUSE_GEOMETRY:-}" ]; then
    cp_reuse_args=(--reuse-geometry "$CP_REUSE_GEOMETRY")
    printf 'Requested geometry reuse (read-only; checked before scoring): %s\n' "$CP_REUSE_GEOMETRY"
  fi
  python -B -u "$cp_repo/tools/evaluate_native_v1_full128.py" \
    --gpu "$CP_GPU" \
    --original-source "$cp_source" \
    --checkpoint /home/aicompetition06/Medical/HierCP/work/full/model.pt \
    --prototype /home/aicompetition06/Medical/HierCP/work/full/prototype.pt \
    --inventory /home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42/inventory/index.json \
    --output "$cp_output" \
    --lesion-policy all_observed \
    "${cp_reuse_args[@]}" \
    --workers "${CP_WORKERS:-16}" \
    --physical-batch-candidates 2 4 8 16 32 \
    --cuda-gib "${CP_CUDA_GIB:-40}" --rss-gib "${CP_RSS_GIB:-192}" \
    --resident-gib "${CP_RESIDENT_GIB:-128}"
}

native30_main
