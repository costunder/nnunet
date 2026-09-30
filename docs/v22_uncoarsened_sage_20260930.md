# v2.2: 그래프 축약 제거 — 2026-09-30

사용자 요청: “일단 v2.2에서 만들었던 그래프 축약부터 없애보자”. v1 전체로 되돌리거나 GAT로 바꾸지 않는다. 새 실행 경로는 `uncoarsened_fixed_view_sage_v1`이다.

## 변경한 계산

이전: 현재 CNN → 고정 partition별 평균 → 작은 관계 그래프 SAGE 3층 → readout → 128D.

현재: 현재 CNN → 원래 sampled fine node별 특징 → **축약 전 13종 관계 edge의 SAGE 3층** → 기존 role/shell readout → 128D → 기존 L1/L2와 ranking loss.

EZ-SP 병합, partition별 특징 평균, quotient edge 생성은 새 학습·validation·최종 후보 scoring 경로에서 호출하지 않는다. 기존 sampler/radius/hop/view 규칙은 그대로다. ‘원본 그래프’는 축약 전 sampled graph를 뜻하며 CT의 모든 voxel을 노드로 삼았다는 뜻이 아니다.

CNN 12/24/32채널, 48³ 입력, SAGE 3층·128D·residual·LayerNorm, L1/L2, loss, patient support 정책, 전체 observation, 후보128, Basic CP와 full paste mask 검사 코드는 유지한다. 새 PE나 새 특징은 추가하지 않았다.

기존 paired cache에서 원본 노드와 edge를 읽는다. region `index.json`과 `frozen_cnn.pt`는 observation 순서·고정 view·초기 가중치 출처 검증에만 사용한다. region tensor 파일이나 assignment는 학습 입력으로 읽지 않는다. 새 디스크 캐시 생성이나 기존 결과 덮어쓰기는 없다. 고정 fine batch와 canonical record는 명시적인 공용 RAM 한도에서 재사용하고, 기존 제한된 GPU 입력/CSR 캐시도 사용한다. 퇴출은 데이터 제외가 아니다.

원본 paired cache의 소스 검증은 유지한다. 이미 검토된 `hiercp_v222/model.py`의 특정 두 hash 사이 실행 변경만 기존 `compatible_core` 규칙으로 인정한다. graph/geometry 변경, record/view/hash 변조, 잘못된 edge, nonfinite, coverage 누락은 거부한다.

## 기존 checkpoint를 명시적으로 전환할 때만 사용하는 별도 기능

**현재 축약 제거 비교에는 아래 전환 기능을 사용하지 않는다.** 최근 region checkpoint는 축약 그래프에서 학습된 weights이므로, 현재 실행 안내는 기존 실험과 같은 초기화에서 step0으로 시작한다. 이전에 최근 checkpoint 재개를 기본 명령으로 제공한 안내는 철회한다.

`--resume-without-coarsening`은 일반적인 동일 조건 재개가 아니라 **입력 그래프 변경을 명시한 계속 학습**이다.

- 기존 CNN/SAGE/L1/L2의 모든 weight tensor, Adam, RNG, epoch, step, query cursor와 physical batch를 보존한다.
- 축약 그래프로 계산했던 support memory·cluster plan과 이전 best metric 선택은 무효화한다. 전체 inner-train memory를 새 그래프로 한 번 다시 계산한 다음 저장된 query 위치에서 진행한다.
- 기존 best 기록은 전환 receipt에 남기며 원본 checkpoint는 변경하지 않는다. 전환 중 일시정지해도 partial memory와 cursor를 저장한다.
- 이후에는 일반 fine checkpoint resume다. launcher가 이를 구분한다.
- 기존 학습을 이어받은 결과는 처음부터 동일 조건으로 학습한 축약 유무 ablation 결과가 아니다. 새 구조에서 성능이 좋아진다고 보장하지 않는다.

전체 40epoch 계약과 이미 소비한 step을 보존한다. 40epoch를 새로 추가하지 않는다. 완료/best-restored checkpoint를 optimization 상태처럼 바꿔 재개하지 않는다.

## 검사 결과와 한계

RTX5070Ti에서 실제 CT DEBUG train8/val2를 사용했다. 학습 query schedule의 실제 batch는2, memory/reference batch는8이며, 서버 physical32 검사로 간주하지 않는다.

- 실제 8pair: **52,666 nodes / 2,761,602 edges**, 원본 edge tensor 유지.
- 기존 fine GraphSAGE와 같은 가중치에서 출력 및 모든 비교 대상 gradient exact 일치.
- CNN/SAGE1/2/3/L1/L2 forward → loss → gradient → Adam update 연결 확인.
- 중단/새 프로세스 재개와 연속 실행의 최종 model/Adam/state/RNG hash 일치.
- 기존 region checkpoint 전환에서 모든 weight tensor/Adam/RNG/epoch/step/physical batch/query cursor 보존.
- refresh, validation, best 선택, final memory, 최종 artifact export/load, 실제 후보2개 점수 계산 통과. 이 후보2개 검사는 전체128 후보 CP 통합 성능 평가가 아니다.
- 27개 단위/회귀 검사 통과: fine 입력 무결성·전환 거부·launcher 설정 보존, 기존 SAGE 연산·gradient, checkpoint pipeline, 학습 표시.

원시 통합 report: `validation/v22_uncoarsened_sage_20260930/actual_ct_report.json`. 로컬 작업: `work/uncoarsened_sage_DEBUG_20260930_r3/`.

서버 A6000의 full physical batch32·최악 그래프 peak·전체 epoch 소요시간은 미측정이다. fine graph는 기존 region graph보다 크므로 더 빠르다고 주장하지 않는다. 목적은 축약으로 생긴 정보 손실을 제거한 경로를 확보하는 것이다. 메모리 부족 시 sample/edge/batch를 자동으로 줄이지 않는다. 장기 GNN/nnU-Net 학습과 전체 평가는 실행하지 않았다.

## A6000 서버 실행

기존 학습이 실행 중이면 Ctrl+C 후 `PAUSED`를 기다린다. 경로를 묻는 `read` 프롬프트에만 머물러 있다면 Ctrl+C 한 번으로 입력 대기를 취소한다. 학습이 종료되기 전에 checkout을 바꾸지 않는다.

아래는 **resume 없는 새 학습**이다. 기존 시작 CNN snapshot과 seed42를 사용하고 SAGE/L1/L2 및 Adam을 새로 초기화한다. region 학습이 진행된 최신 checkpoint의 weights/Adam을 가져오지 않는다. 기존 실험 physical batch32를 유지하기 위해 calibration 후보를32로 고정하며, query 환자별 마지막 잔여 batch는 기존 schedule 그대로다. support16, workers16, CUDA40/RSS192/resident128/GPU cache8GiB와 workspace512MiB를 유지한다. 40epoch 전체를 새 실행으로 진행한다.

```bash
cd /home/aicompetition06/Medical/HierCP-regions-8580e59 &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD &&
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/run_fixed_regions.py train \
  --profile-policy research-report \
  --cache work/regions_frozen_reuse_20260929_230052/cache/index.json \
  --fine-cache /home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --support-patients 16 --activation-storage retained \
  --execution-pipeline overlapped --device-cache-gib 8 --sage-workspace-mib 512 \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 \
  --output "work/v22_fine_sage_fresh_$(date +%Y%m%d_%H%M%S)"
```

처음의 calibration support는 초기 모델의 새 그래프 support embedding 계산이다. EZ-SP prepare를 다시 실행하는 단계가 아니다. 검증된 calibration memory를 첫 epoch에 재사용한다. Ctrl+C는 기존 foreground save-and-pause 방식으로 동작한다. 새 학습 경로 자체는 앞선 실제 CT GPU smoke의 uninterrupted/paused/resumed 검사에서 실행했다. 서버 전체 batch32의 fit/속도는 미측정 상태다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 설정 보존, disjoint batching/worker/RAM/GPU cache 경로 사용.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 로컬 DEBUG execution/timing 기록; 서버 전체 측정은 미실행.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검사 OOM 없음, 자동 축소 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위검사와 실제 CT smoke 구분.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
