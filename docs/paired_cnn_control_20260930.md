# Paired CNN readout 비교 경로 — 2026-09-30

## 변경 범위

`l0_cnn_control/encoder.py`에 독립적인 DEBUG 모델을 구현했다.

donor/recipient의 기존 CT 입력 `[B,1,48,48,48]` → 기존 학습 가능한 CNN
→ 기존 역할·shell 위치의 특징 readout → 기존 128D pair 표현 → 기존 L1/L2
→ 기존 same_donor_live_v1 loss → backward → AdamW update.

기존 fine SAGE와 달라진 것은 L0의 GraphSAGE 3개 block을 제거한 것이다.
해당 파라미터도 optimizer에서 빠진다. CNN 깊이·너비, 입력 영역, 역할/shell
readout, 좌표 해석(stride4), L1/L2 및 loss는 유지한다.
이것은 새로운 비교 모델이며 기존 checkpoint의 exact resume가 아니다.

**전체 CT 모델이나 모든 voxel을 pooling하는 모델은 아니다.** CNN이 기존
48³ 입력 전체를 인코딩하고, 기존 공간 위치에서 특징을 읽는다. 따라서 기존
node 선택 규칙을 학습형 선택으로 고친 작업이라고 표현하면 안 된다.
현재 로더는 원래 graph를 읽고 검증한다. CNN 경로에서 메시지 전달만 제외한다.
데이터 준비·graph 전송 비용이 제거됐다고 주장하지 않는다.

기본 학습 CLI, 서버 명령, Basic CP, 128개 후보 정책, paste mask 검사,
기존 모델 및 결과는 변경하지 않았다. DEBUG 모델의 production checkpoint
export는 명시적으로 거부한다.

## 실제 검사

- 단위 검사: `tests/test_paired_cnn_control_debug.py` 5개 통과.
- 실제 CT smoke: `work/paired_cnn_control_DEBUG_20260930/report.json`.
- RTX 5070 Ti에서 train 8개 관측(4개 case), validation 2개 관측(1개 case).
- 두 모델 모두 동일 pair 순서로 4회 update. Physical batch 2, accumulation 1,
  data-parallel worker 1, effective batch 2. 이는 명시적인 DEBUG 규모다.
- 공통 초기 parameter의 hash 동일. CNN은 기존 DEBUG 캐시의 frozen_cnn.pt에서
  초기화했으며 서버에서 학습한 최종 모델이 아니다. SAGE/L1/L2/readout은 동일
  seed 초기화다. 각 모델의 support memory를 별도로 다시 계산했다.
- CNN, readout, L1, L2의 nonzero gradient와 optimizer 후 변화 모두 확인.
  SAGE 경로에서는 SAGE 3층도 각각 확인했다.
- 표본 탈락이나 실패 시 fallback 없음. 주어진 DEBUG cohort를 모두 처리했다.

| 항목 | 기존 fine SAGE | paired CNN readout |
|---|---:|---:|
| 전체 학습 가능 parameter | 5,535,830 | 2,269,526 |
| L0 message-passing 층 | 3 | 0 (명시적 ablation) |
| 최대 PyTorch GPU 할당 byte | 996,394,496 | 385,323,008 |
| 검증 pairwise loss, 초기 → 4 update | 0.6178 → 0.7295 | 0.6558 → 0.6618 |
| 검증 MRR, 초기 → 4 update | 1.0 → 0.5 | 1.0 → 1.0 |

검증은 한 케이스의 두 후보뿐이다. 이 수치로 CNN 우월성, 일반화 성능,
128개 후보 추천 성능 또는 segmentation Dice 개선을 주장할 수 없다.

GPU에서는 별도 GOIS 학습이 실행 중이었다. 해당 작업은 중단하지 않았다.
자체 CUDA allocator 한도 3GiB, RSS 한도 24GiB, CPU worker 4개를 사용했다.
확인한 CPU는 16 logical core, 시스템 RAM은 약 64GiB였다.
시간은 report에 진단 기록으로만 남겼으며 **속도 개선 배수 및 서버 epoch
예측에 사용하지 않는다.** Physical batch 처리량 선택도 아직 수행하지 않았다.

## 실행

현재 작업 폴더에서 다음은 동일한 명시적 DEBUG 검사다. output은 존재하지
않는 새 경로여야 하며, 장기 학습이나 production checkpoint를 만들지 않는다.

```powershell
.\.venv\Scripts\python.exe -B tools/verify_paired_cnn_control_debug.py --debug --cache work/region_frozen_reuse_cli_20260929/cache/index.json --fine-cache work/same_donor_learning_DEBUG_20260930/cache/index.json --output work/paired_cnn_control_DEBUG_NEW --physical-batch 2 --support-patients 2 --workers 4 --cuda-gib 3 --rss-gib 24
```

## 완료 범위와 제한

구현·import/단위 검사·실제 데이터 GPU smoke 완료. 장기 학습, 전체 평가,
CP 적용 효과 검증, 서버 배포, production 기본 모델 교체는 하지 않았다.
학습되는 공간 선택 또는 sparse graph 재설계가 완료된 것은 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 기존 모델은 보존하고 별도 DEBUG ablation의 차이를 명시했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 원래 topology와 production 데이터 계약은 보존했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. DEBUG batch 2·worker 4, 최종 처리량 선정은 미실시다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행에는 OOM이 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
