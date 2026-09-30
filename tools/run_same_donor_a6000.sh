#!/usr/bin/env bash
# Explicit manual full GNN run. No detached worker, nnU-Net launch, or old resume.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

CP_SOURCE_CACHE=/home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json
CP_INITIALIZATION=work/regions_frozen_reuse_20260929_230052/cache/index.json
CP_RUN="work/v22_same_donor_live_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9

python -u - "$CP_SOURCE_CACHE" "$CP_INITIALIZATION" "$CP_RUN" <<'PY'
import sys
from pathlib import Path
import torch
for value in sys.argv[1:3]:
    if not Path(value).is_file():
        raise FileNotFoundError(value)
if Path(sys.argv[3]).exists():
    raise FileExistsError('New output directory required: '+sys.argv[3])
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise RuntimeError('The assigned A6000 UUID must expose exactly one CUDA device')
p=torch.cuda.get_device_properties(0)
if 'A6000' not in p.name or p.total_memory<=40*2**30:
    raise RuntimeError('This launcher is for the assigned A6000 with a 40 GiB allocator budget')
print('GPU:',p.name,'VRAM GiB:',round(p.total_memory/2**30,2),flush=True)
print('Full GNN run:',Path(sys.argv[3]).resolve(),flush=True)
print('All observations; 128 comparison candidates/case; 40 epochs; physical batch 32.',flush=True)
print('Fresh SAGE/L1/L2/Adam; saved CNN initialization. No old-objective checkpoint resume.',flush=True)
print('Donor-dependent graph preparation precedes training; no EZ-SP partition preparation.',flush=True)
PY

python -u tools/prepare_same_donor.py \
  --cache "$CP_SOURCE_CACHE" --initialization "$CP_INITIALIZATION" \
  --output "$CP_RUN/cache" --workers 16 --cuda-gib 40 --rss-gib 192

python -u tools/run_fixed_regions.py train \
  --learning-policy same_donor_live_v1 --profile-policy research-report \
  --cache "$CP_INITIALIZATION" --fine-cache "$CP_RUN/cache/index.json" \
  --output "$CP_RUN/training" \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 --support-patients 16 --activation-storage retained \
  --execution-pipeline overlapped --device-cache-gib 8 --sage-workspace-mib 512
