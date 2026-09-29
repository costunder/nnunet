# Server missing torch-scatter — 2026-09-29

The server reached the first fixed-region preparation batch and failed because `torch_scatter` was absent from the active `nnunet` environment. PyG can be installed without this optional extension, while the pinned official EZ-SP merger explicitly requires it. This failure occurred before a region batch was saved or GNN training began. The original paired cache remains reusable.

The runner now performs dependency import, actual CUDA scatter sum/min and a tiny synthetic official-merge probe before reading CT/checkpoints or creating output. Missing/ABI-incompatible extensions produce an actionable environment/version error. No package is installed automatically, no vendor file is edited, and no CPU fallback is used. This probe checks kernels, not medical data or model quality.

Local verification: four tests PASS, no skips, RTX5070Ti, torch2.8.0+cu128, torch_scatter2.1.2+pt28cu128. Test suite `tests.test_region_preflight_debug`: matching wheel page, missing import, failure before output/prepare, real CUDA kernels. Test run3.252seconds, command6.44seconds. Model/partition/admission/BasicCP/L1/L2/loss/batch/candidates are unchanged. No long training was run.

Use the active environment's PyTorch version and `torch.version.cuda`, not the driver CUDA version. Official source: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html . The earlier server report was torch2.6.0+cu118/Python3.10; the official wheel index contains Linux CPython3.10 binaries at https://data.pyg.org/whl/torch-2.6.0+cu118.html . The new host's version is determined at execution time, not assumed identical.

```bash
conda activate nnunet
cd /home/aicompetition06/Medical/HierCP-regions-8580e59
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)"
python -u - <<'PY'
import subprocess, sys, torch
assert torch.cuda.is_available(), 'CUDA unavailable'
major, minor = torch.__version__.split('+')[0].split('.')[:2]
cuda = 'cu' + torch.version.cuda.replace('.', '')
url = f'https://data.pyg.org/whl/torch-{major}.{minor}.0+{cuda}.html'
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)
print('Wheel page:', url, flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
               '--only-binary=:all:', 'torch-scatter', '-f', url], check=True)
PY
```

The binary-only install deliberately fails when no matching wheel exists. Do not silently rebuild from source or replace torch. After fetching the documented fix commit, `python -m tools.v22_region_preflight` verifies the kernel on the selected GPU.

Preparation is not resumable in the current region runner: preserve `work/regions_8580e59/cache` and use a new path, `work/regions_scatterfix_20260929/cache`. This reuses the complete original paired cache; it does not recreate the earlier 14,102-pair cache. The initial-profile admission gate remains unchanged and may be the next independent stop. Installing torch-scatter does not constitute partition admission or performance validation.

The runner's source hash changes. Existing region checkpoints from an earlier source revision must not be relabelled to pass exact resume/source identity. This server attempt had not started training, so it has no region training state to resume.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 계약 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 이번에는 CUDA 커널 검사만 수행.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 원인은 의존성 누락으로 OOM 아님.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 작은 합성 입력은 명시적 커널 검사에만 사용.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번에는 모델 학습 연결 검사를 반복하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
