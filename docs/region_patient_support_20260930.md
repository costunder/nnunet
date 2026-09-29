# v2.2 환자 support 미니배치 학습

사용자 승인: 전체 support를 매 update에 L1/L2로 처리하는 학습을 환자 단위 support episode로 변경한다. 이것은 연산 결과가 같은 실행 최적화가 아니라 **학습 구성 변경**이다.

## 구현

| 구간 | 적용 방식 |
|---|---|
| Query | 기존 환자별 physical batch와 순서 유지. 전체 관측을 epoch마다 정확히 한 번 query로 사용 |
| Support 선택 | inner-train만 사용. query 환자와 같은 recipient 또는 donor는 제외. 적게 선택된 환자를 우선하고 seed+epoch로 동률 순서 결정 |
| Support 크기 | `--support-patients`로 반드시 명시. 선택된 환자의 적격 관측은 전부 포함. 관측 cap·drop·fallback 없음 |
| 클래스 | support의 관측 클래스 0/1이 모두 포함되도록 선택. 불가능하면 명시적 오류. query 정답으로 선택하지 않음 |
| L0 | 기존 CNN + 고정 영역 GraphSAGE 유지. 현재 query의 L0에 gradient 전달 |
| L1/L2 | 선택된 환자들에 대해 매 update differentiable 계산. 층 수·128D·alignment loss 유지 |
| Cluster plan | query 환자 episode 시작에 선택된 support로 생성; 해당 episode 동안 고정. epoch·환자가 바뀌면 재생성 |
| Reference memory | 기존 전체 inner-train L0 memory 및 epoch refresh 유지. support memory는 기존처럼 detach |
| Ranking | 같은 case의 모든 후보 reference와 live query 조합, 기존 loss 유지. 후보128 변경 없음 |
| Validation / CP | 전체 적격 inner-train support로 환자마다 한 번 준비. 학습의 부분 환자 문맥과 추론의 전체 문맥은 명시적으로 다름 |

학습 데이터 자체를 줄인 것이 아니다. 모든 관측은 query로 학습한다. 다만 모든 관측이 모든 update의 support인 조건은 폐기했다. `support_epoch_*.json`은 완전한 예정 schedule의 query coverage 및 support 사용률을 기록하며, 실제 처리 지점은 checkpoint의 `next_batch`이다. 선택된 support 관측의 epoch coverage도 따로 기록하고 100%라고 임의 선언하지 않는다. 진행창에 `support=16p/…r`와 같이 환자/관측 수가 표시된다.

`--support-patients`를 생략하면 비교를 위한 기존 전체-support 경로다. 서버 예시의 **16환자**는 시작용 연구 설정이다: 적어도 두 환자의 증거를 요구하는 기존 clustering에서 클래스별 최대 8개의 비단일 환자 군집을 표현할 여지를 두면서, 전체 환자 cohort를 매번 처리하지 않는다. 16이 최적이거나 정확도를 보장하는 값이라는 근거는 없다. 코드에 숨긴 기본값으로 두지 않으며, 크기가 부족하면 자동 축소하지 않는다. DEBUG의 2환자는 실행 연결 및 재개 검사 전용이다.

## 기존 학습에서 전환

`--resume-support-minibatch`는 검토된 `87eafa6`, `f36518c`, `d05e4e3` 계열의 **optimization 상태**만 명시적으로 전환한다. 캐시·모델·loss·자원 계약·physical batch 후보가 같아야 한다. CUDA budget 변경이나 activation 변경과 한꺼번에 사용하지 않는다.

- 모델, Adam, RNG, epoch, step, 다음 query cursor, physical batch, 전체 memory 보존.
- 옛 전체-support cluster plan과 last-group 초기화.
- 이전 best는 원본 checkpoint에 보존하고 이전 best 지표/epoch를 이력에 남김. 새 학습 정책의 best 선택은 새로 시작.
- 이전 identity와 전환 step을 checkpoint·최종 artifact에 보존. 기존 runtime의 exact resume 또는 처음부터 미니배치로 학습한 실험이라고 표시하지 않음.
- 기존 region 캐시 재생성 없음. 새 출력 폴더에 저장. 변경 후에는 같은 설정과 `--resume`만 사용하며 migration flag는 다시 사용하지 않음.

## 검증

최종 검증 원문: `validation/region_patient_support_20260930/smoke.json` 및 `comparison.json`.

- 단위 검사 9개: 양쪽 환자 누수 제외, 선택 환자의 모든 적격 관측 포함, 전체 query coverage, 결정론, 클래스 부족 거부, 변경된 memory 결속 거부, 전체 환자 선택 시 기존 support/집계 gradient 일치, 허용되지 않은 migration 거부.
- 기존 실행 업그레이드 6개와 CUDA migration 4개 회귀 검사.
- RTX5070Ti / 실제 CT DEBUG train8·val2. 기존 query 설정 batch8이지만 환자당 관측2개라 **실제 query batch는2**. 서버 batch32 검증으로 해석하면 안 된다.
- 실제 entry의 학습, 모든 DEBUG memory refresh, validation, best 선택, 최종 artifact export 확인.
- 중단·새 프로세스 resume와 연속 실행의 model/Adam/state/RNG 완전 일치.
- 기존 whole-support checkpoint에서 다음 update로 전환. CNN/L0/L1/L2 가중치 갱신, 모든 trainable gradient 존재·유한값 확인.
- 잘못된 다른 cache로의 migration은 거부되었고, 일치하는 cache에서 통과.

짧은 비용 비교는 동일 query·초기 가중치·Adam·RNG를 반복 복원하며 실행한다. 전체 support와 미니배치는 **다른 학습 구성**이므로 loss/gradient 일치를 주장하지 않는다. 한 warmup 후 세 번의 disposable update. loader, 실제 checkpoint 저장, epoch refresh/validation은 시간에서 제외되므로 epoch 개선 배수로 쓰지 않는다. 로컬 support6→4 관측은 서버 11279개 reference와 다르다.

## A6000 서버 실행

실행 중이라면 먼저 기존 학습 창에서 Ctrl+C 후 `PAUSED`와 저장 경로를 확인한다. 같은 checkout의 실행 중 코드에 git switch를 하지 않는다. 아래 `CP_RESUME`은 대화에서 확인된 **step173 저장본**이다. 더 진행된 저장본이 있다면 그 `PAUSED` 경로를 넣어야 한다. step73 옛 파일로 되돌리지 않는다.

```bash
cd /home/aicompetition06/Medical/HierCP-regions-8580e59 &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD

CP_RESUME=work/regions_retained_resume_20260930_012003/checkpoint_latest.pt
CP_CACHE=work/regions_frozen_reuse_20260929_230052/cache/index.json
```

동일 saved query·physical batch의 짧은 비용 비교. 긴 학습이나 production checkpoint 쓰기 없음:

```bash
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/compare_region_support_debug.py \
  --cache "$CP_CACHE" --checkpoint "$CP_RESUME" \
  --output "work/support_episode_cost_$(date +%Y%m%d_%H%M%S)" \
  --support-patients 16 --workers 16 \
  --cuda-gib 40 --rss-gib 192 --resident-gib 128
```

검토된 새 학습 구성으로 이어갈 때의 실행 명령. 자동 실행하지 않았다:

```bash
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/run_fixed_regions.py train \
  --profile-policy research-report \
  --cache "$CP_CACHE" --resume "$CP_RESUME" \
  --resume-support-minibatch --support-patients 16 \
  --activation-storage retained \
  --output "work/regions_patient_episode_$(date +%Y%m%d_%H%M%S)" \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 48 64
```

## 남아 있는 범위

전체 규모 A6000 비용, 정확도 및 전체 epoch 시간은 아직 측정하지 않았다. 전체 reference memory refresh, validation, checkpoint 저장 비용은 이 변경으로 제거되지 않는다. 기존 partition quality 미검증 표시는 유지한다. Basic CP·nnU-Net·원본 mask 검사·후보128·L0 그래프·모델 층/차원 변경 없음. 장기 GNN/nnU-Net 학습은 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 support 학습 단위 변경은 위에 명시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 batch·disjoint graph·prefetch 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검사 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사와 실제 CT smoke를 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
