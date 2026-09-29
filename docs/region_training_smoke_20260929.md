# v2.2 고정 영역 SAGE 학습 연결 및 로컬 smoke

2026-09-29. 사용자의 요청: 로컬 짧은 smoke → Git push → 사용자가 A6000 서버에서 실행. 장기 학습은 자동 시작하지 않았다.

## 구현과 검사 결과

- `tools/run_fixed_regions.py`: foreground prepare/train/smoke. 기존 GAT checkpoint는 CNN snapshot만 초기화하며 GAT optimizer/support를 새 SAGE의 exact resume로 취급하지 않는다.
- `l0_regions/training_data.py`: 전체 inner-train/inner-val 순회, 원본 cache/record/view/CNN/source 결속, 동시 record 준비, 고정 partition 및 typed edge 저장. Reg 두 값은 필수 인자이며 기본값 없음.
- `l0_regions/training.py`: 전체 support, 기존 patient-group batch schedule, 기존 observed_rank_v1 loss 및 L1/L2, live CNN gradient, AdamW, support refresh, held-out validation, best 선택, best 모델의 final memory, hash로 봉인한 checkpoint와 새 프로세스 resume. Ctrl+C는 batch 경계에서 저장 후 PAUSED로 반환한다.
- CPU/GPU 회귀 **44개 PASS**, skip 없음. 실제 CT DEBUG train8/val2의 공급된 전량을 사용했다. DEBUG1epoch/4update를 정상 실행하고, 별도 실행을 step1에서 저장한 뒤 새 프로세스에서 재개했다. 최종 model/optimizer/state/RNG **네 항목의 hash가 모두 정확히 일치**했다. 이것은 전체 cohort 학습/효용 검증이 아니다.
- Git index만 별도 폴더로 추출하고 isolated Python에서 import/config/provenance/vendor pin 검증 PASS. 로컬의 다른 untracked 소스에 의존해 통과한 것으로 처리하지 않았다.

## smoke에서 실제로 발견하고 수정한 사항

1. Windows `psutil.disk_usage(Path)` 오류: 문자열 경로로 전달.
2. calibration clone의 sparse CSR deepcopy 오류: 실행용 인접행렬 캐시는 복제 모델에 새로 생성. 원본 모델의 학습 상태와 공유하지 않음.
3. Python/NumPy 난수 초기화 누락: torch와 동일 seed42로 초기화.
4. 같은 L0 입력/RNG에서도 기존 CUDA CSR 곱의 결과가 달랐다. 동일 입력 재실행에서 최초 차이는 SAGE 집계 뒤 lin_l에서 발생했고, loss 차이 약2.38e-7 및 gradient 차이가 재현되었다. 작은 차이를 통과 허용오차로 숨기지 않았다.
5. **새 영역 학습 경로에 한해** 고정 순서 segment sum을 사용한다. 같은 D^-1 A 및 transpose로 forward/backward를 수행한다. 중복 edge 질량, 관계, self/root 경로, 노드/edge 수를 변경하지 않는다. 중간 edge-feature 작업공간은64MiB 목표로 행 단위 분할하며, 단일 긴 행은 온전히 처리한다. CPU/GPU dense 수식 대조 및 반복 정확일치 검사 PASS. 기존 fine-SAGE/동적 EZ-SP 진단 경로는 그대로이다.

`r3`, `r4`의 FAIL을 보존했다. 최종 `r5`는 허용오차 완화 없이 PASS다. 새 합산 커널의 전체 cohort 속도는 미측정이며 이전 CSR 속도 측정값을 새 경로 성능으로 인용하면 안 된다.

## 학습 진입에 남은 조건 — 숨기지 않음

로컬10개 record **전부** 미검증 초기 partition profile의 상한을 넘었다. 이 사실은 구현 오류 또는 EZ-SP 자체 실패라는 뜻이 아니다. 기존1024/32768 및 bbox/분산 상한은 여전히 검증되지 않은 초기값이다.

사용자는 앞서 실패한 pair를 정상 학습 sample로 처리하지 말라고 명시했다. 따라서 DEBUG 비용/통합 검사에서는 위반을 기록하지만, **full prepare/train은 위반을 거부**한다. Reg/min_size/상한 자동조정, node drop, record skip, fine-GAT fallback, CP no-op 없음. 로컬 DEBUG cache 또는 DEBUG CNN을 full 모드로 승격하지 않는다.

즉, smoke PASS만으로 현재 partition이 전체 학습에 admitted됐다고 표시하지 않는다. 현재 로컬 결과의 `production_ready=false`는 남은 이 조건을 반영한다. 서버의 기존 non-DEBUG CNN snapshot으로도 같은 초기 상한을 넘으면 준비 단계에서 멈춘다. 상한을 연구학습의 보고 항목으로 바꾸려면 기존 사용자 지시 변경의 명시적 승인이 필요하다.

새 checkpoint는 GNN 학습용 별도 형식이다. **새 영역 encoder의 online CP/nnU-Net 최종 artifact 연결은 아직 구현되지 않았다.** 기존 Basic CP, 원본 mask 검사, 후보128, L1/L2, loss를 변경하지 않았다는 뜻이지 새 GNN의 결과를 기존 nnU-Net runner에 바로 넣을 수 있다는 뜻이 아니다.

## A6000 실행 경로

아래는 **strict admission을 유지하는** 준비/학습 명령이다. 현재 로컬 profile은 통과하지 않았으므로 무조건 학습이 시작된다고 보장하지 않는다. `--debug`를 붙여 full 학습 차단을 우회하면 안 된다. 준비 성공 후에만 train을 실행한다.

소스는 아래 branch에서 새 디렉터리로 받는다. 기존 실험 폴더나 실행 중인 checkout을 전환하지 않는다. 이 문서와 함께 기록된 Git commit으로 고정해서 실행한다. CT/기존 paired cache는 서버 경로를 그대로 참조하며 Git으로 옮기지 않는다.

```bash
conda activate nnunet
git clone --branch codex/v222-server-r6 --single-branch https://github.com/costunder/nnunet.git /home/aicompetition06/Medical/HierCP-regions-20260929
cd /home/aicompetition06/Medical/HierCP-regions-20260929
```

할당받은 A6000이 CUDA0으로 노출된 상태여야 한다. 다른 GPU나 프로세스를 종료하거나 CUDA_VISIBLE_DEVICES를 임의로 변경하지 않는다. 아래 예산은 GPU40GiB/RSS192GiB/resident128GiB, CPU16worker를 명시적으로 사용한다. 서버의 할당 자원이 부족하면 실행을 중단하고 예산을 실제 할당에 맞춰 검토한다. Physical batch32/48/64는 full query group에서 실제 측정한 처리량으로 선택하며 모델/관측 수는 줄이지 않는다.

```bash
python -u - <<'PY'
import os, json, shutil, torch, psutil
import torch_scatter, torch_geometric
p = torch.cuda.get_device_properties(0)
cpus = len(os.sched_getaffinity(0))
print(json.dumps(dict(gpu=p.name, vram_gib=p.total_memory/2**30,
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    cpu_affinity=cpus, ram_available_gib=psutil.virtual_memory().available/2**30,
    disk_free_gib=shutil.disk_usage('.').free/2**30), indent=2))
assert 'A6000' in p.name and p.total_memory > 40*2**30
assert torch.cuda.mem_get_info()[0] > 40*2**30
assert cpus >= 16 and psutil.virtual_memory().available > 192*2**30
assert shutil.disk_usage('.').free > 80*2**30
PY
```

공식 EZ-SP는 `torch-scatter`도 필요하다. 위 import가 실패하면 현재 서버의 torch/CUDA 조합에 맞는 패키지를 설치한 후 확인한다. CUDA/드라이버 또는 conda 환경을 자동 교체하지 않는다. 서버 설치는 [PyG 공식 설치 안내](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)를 따른다.

위 확인이 성공하고 스케줄러의 실제 RAM 할당도 충족할 때 다음을 실행한다.

```bash
python -u tools/run_fixed_regions.py prepare \
  --cache /home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --partition-checkpoint /home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_gnn40_resume114_20260928/training/checkpoint_latest.pt \
  --output work/regions_full_cache_20260929 \
  --prepare-batch 32 --workers 16 --reg-scale1 0.02 --reg-scale2 0.02 --view-epoch 0 \
  --cuda-gib 40 --rss-gib 192
```

위 단계가 실제로 성공하여 `index.json`이 생성된 경우에만:

```bash
python -u tools/run_fixed_regions.py train \
  --cache work/regions_full_cache_20260929/index.json \
  --output work/regions_gnn40_20260929 \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 48 64
```

기존 전체40epoch 설정을 사용한다. 새 모델의 새 학습이며 옛 GAT학습의 exact resume가 아니다. 새 영역 모델을 PAUSED한 뒤에는 같은 인자/코드/cache에서 새 output과 `--resume <영역모델의 checkpoint_latest.pt>`를 사용한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. DEBUG의 자연 tail은2, 공급된8개 support를 모두 사용했다. 서버32/48/64 실제 batch 비용은 미측정이다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실행 계약 및 calibration 보고서에 기록했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. GNN 범위이며 online CP 연결은 미완료다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
