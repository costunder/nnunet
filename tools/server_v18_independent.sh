#!/usr/bin/env bash
set -euo pipefail

CP_GPU="${CP_GPU:-3}"
CP_ARM="${CP_ARM:?Set CP_ARM to selected or native}"
CP_SOURCE_EXPERIMENT="${CP_SOURCE_EXPERIMENT:-/home/aicompetition06/Medical/experiments/v18_u_bridge_m10_seed42}"
CP_EXPERIMENT="${CP_EXPERIMENT:-/home/aicompetition06/Medical/experiments/v18_${CP_ARM}_m10_seed42_memory}"
CP_INVENTORY="${CP_INVENTORY:-/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42/inventory/index.json}"
bridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$bridge_root"
python -B -u tools/run_v18_independent.py \
  --gpu "$CP_GPU" --arm "$CP_ARM" \
  --source-experiment "$CP_SOURCE_EXPERIMENT" \
  --experiment "$CP_EXPERIMENT" --inventory "$CP_INVENTORY"
python -B tools/summarize_v18_u_bridge.py --experiment "$CP_EXPERIMENT"
