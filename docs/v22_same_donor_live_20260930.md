# v2.2 — 같은 donor / 현재 특징 순위 학습 수정

## 이번 변경

사용자가 로컬 비교에서 드러난 학습 조건 불일치를 고치도록 요청했다. 새 학습 정책 이름은 `same_donor_live_v1`이다. 기존 `observed_rank_v1` 및 결과는 보존한다. 새 정책은 명시적 CLI 옵션으로 선택하며, 기존 학습 checkpoint의 exact resume로 진입할 수 없다.

1. case별 donor를 label과 독립적으로 고정하고 그 donor로 모든 관측 위치의 실제 기하를 다시 생성한다. 후보 128개 및 모든 적격 양성, 원래 split·환자 배제·train donor pool을 유지한다. ID만 바꾼 기존 그래프 재사용은 금지한다.
2. 같은 case의 모든 양성 P × 미관측 U 조합을 한 epoch에 정확히 한 번씩 비교한다. physical batch 내에 두 집합을 함께 넣으며 **양쪽 CNN/SAGE에 현재 gradient**를 전달한다. ranking에 detached epoch embedding을 사용하지 않는다. L1/L2 support memory는 기존 정책을 유지한다.
3. 비교 조합을 나누면서 반복되는 관측의 CE는 반복 횟수 역수로 보정한다. batch마다 class weight가 상쇄되던 CE 대신 epoch 전체 기준의 balanced CE를 사용한다. 고정 모델에서 타일 loss의 평균 및 gradient가 전체 P/U pair 평균 + 전체 balanced CE와 일치함을 CPU/CUDA에서 검사했다. 여러 optimizer update가 하나의 giant-batch update와 같다는 뜻은 아니다.
4. best는 **MRR → R@1 → 낮은 pairwise loss** 순서로 선택한다. validation에서는 반복 tile을 쓰지 않고 각 후보를 한 번씩 평가한다.
5. 미축약 CNN→SAGE3→128D, L1/L2 층/폭·support 정책·loss 계수·원본 paste mask 검사·Basic CP는 유지한다. ranking/CE의 비교 단위와 정규화는 변경된 학습 계약이다.

## 변경되는 계산량과 제한

로컬 전체 inventory 집계: train 11,279 / validation 2,823, train P/U 67,456쌍 모두 보존. physical batch32에서 **533 update/epoch**, 15,989개 query 제시(반복 포함), 실제 batch 평균30.0·최대32·마지막 잔여 최소5. 기존405 update보다 많다. 전체 데이터 수를 늘리거나 줄인 것이 아니라 live 양쪽 비교에 필요한 반복이며 CE 반복 보정이 적용된다. 40epoch이면21,320 update이다.

현재 정책은 case당 고정 donor 하나다. 전체527개 donor pool을 그대로 선택 대상으로 두며 이 seed/전체 recipient 구성에서는97개 서로 다른 donor가 선택된다. **527개 모두를 실제 학습에 사용했다고 표현하지 않는다.** donor 다양성 감소는 이 고정 조건 비교의 제한이며, epoch별 donor 교체/다중 donor episode를 검증 없이 추가하지 않았다. 새 학습이 정확도를 개선했다는 전체 평가 결과는 아직 없다.

관측 양성은 실제 종양 위치의 대리 목표다. 외부 donor를 이식했을 때의 최적 자리 정답으로 승격하지 않는다. 현재 raw CT 입력·종양 주석 사용 규칙 및 CP 최종 overlap 제외는 유지한다.

## RAM 실패 대응

서버 실패는 GPU OOM이 아니라 프로세스 RSS 예산 초과였다. 이전 코드에서 cache tensor 한도와 프로세스 RSS 예산이 연동되지 않았고, 학습 batch 및 GPU cache의 CPU 원본을 보유한 채 refresh batch를 더 만들었다. 이것이 서버 RSS 전부의 원인이라고 원격 heap profile 없이 단정하지는 않는다.

- memory refresh/final-memory/validation 전 GPU batch cache와 등록된 fine batch cache를 해제한다. 원본 파일 cache는 재사용한다.
- refresh/validation에서 거대한 collated batch를 중복 보관하지 않는다.
- 매 fine batch 준비 전 실제 RSS에 따라 immutable input LRU를 퇴출한다. RSS와 cache 상한 차이는 workspace 여유로 남긴다. 예산 초과가 해소되지 않으면 계속 명시적으로 실패한다.
- Linux에서 지원되면 free heap 페이지를 `malloc_trim`으로 반환한다. tensor/파일/학습 상태를 삭제하는 동작은 아니다.
- `memory_timing.jsonl`에 RSS, cache bytes, eviction, 처리 완료 cursor를 기록한다. checkpoint는 완료된 memory prefix를 보존한다.
- RAM/GPU 상한을 늘리거나 physical batch·그래프를 줄이지 않는다. 전체 서버192GiB RSS 조건을 로컬 smoke로 검증했다고 주장하지 않는다.

## 로컬 검증

- RTX5070Ti, 실제 CT DEBUG train8/val2. 기존 observation을 유지하고 새 donor 기하를 실제 CT에서 다시 만들었다.
- 실제 prepare→calibration→train→refresh→validation→best→final memory→artifact export→candidate scoring 연결 확인.
- CNN/SAGE1/2/3/L1/L2의 가중치 표본 갱신 모두 확인.
- 새 프로세스에서 optimization 중단/재개, refresh 2개 관측 완료 후 중단/재개 각각 연속 실행과 **model/Adam/state/RNG exact 일치**.
- 실제 DEBUG physical2 최고 CUDA 할당0.97GiB, 관측된 update RSS 최고2.45GiB, 평균 update0.777초. 전체 batch32 또는 서버 전체 epoch 속도 수치가 아니다.
- 기존 학습 정책도 별도 실제 CT DEBUG에서 refresh·validation·최종 export까지 통과했다.
- synthetic 단위 검사는 비교쌍 전수 coverage, 양쪽 gradient 및 전체 objective 일치, class-blind train-only donor, MRR best, cache alias 및 RAM 압력 처리를 확인한다. synthetic 출력을 실제 예측으로 사용하지 않는다.

원시 GPU 결과: `validation/v22_same_donor_live_20260930/report.json`. 진단 도구: `tools/verify_same_donor_debug.py`. 모든 새 디렉터리는 기존 경로가 있으면 실패하며 결과를 덮어쓰지 않는다.

## 서버 실행 — 새 학습 계약

현재 traceback 이후 프로세스가 종료된 상태를 기준으로 한다. 실행 중인 다른 작업을 종료하거나 서버에 접속하는 명령은 포함하지 않는다. 원래 A6000 UUID를 사용한다.

아래 명령은 **새 기하 준비 성공 후 새 학습**을 시작한다. epoch1의405 update 후 refresh에서 중단된 기존 checkpoint를 새로운 학습의 resume로 넣지 않는다. 초기 CNN snapshot은 기존 region cache의 frozen CNN이며 SAGE/L1/L2/Adam은 새로 초기화한다. 전체 기하 준비 비용은 남아 있다. 기존 파일/캐시/결과는 그대로 보존한다.

```bash
cd /home/aicompetition06/Medical/HierCP-regions-8580e59 &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD &&
CP_RUN="work/v22_same_donor_live_$(date +%Y%m%d_%H%M%S)" &&
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/prepare_same_donor.py \
  --cache /home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --initialization work/regions_frozen_reuse_20260929_230052/cache/index.json \
  --output "$CP_RUN/cache" --workers 16 --cuda-gib 40 --rss-gib 192 &&
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/run_fixed_regions.py train \
  --learning-policy same_donor_live_v1 \
  --profile-policy research-report \
  --cache work/regions_frozen_reuse_20260929_230052/cache/index.json \
  --fine-cache "$CP_RUN/cache/index.json" \
  --output "$CP_RUN/training" \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 --support-patients 16 --activation-storage retained \
  --execution-pipeline overlapped --device-cache-gib 8 --sage-workspace-mib 512
```

위 명령은 사용자가 서버에서 실행할 명령이다. 이번 작업에서 로컬이나 서버의 전체 학습을 자동 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프 규칙과 전체 observation/후보128을 축소하지 않았다. donor 선택 방식의 변화와 실제 선택 수를 명시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch32 일정과 전체 pair coverage를 검토했다. 실제 GPU 검사는 별도 DEBUG physical2다.
- [x] GPU, CPU, RAM을 확인하고 DEBUG peak를 기록했다.
- [x] 모델 축소 대신 cache 수명 및 RSS 예산을 수정했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward/loss/gradient/optimizer 연결과 최종 scoring을 확인했다.
- [x] 실제 실행 설정과 변경 사항을 보고했다.
- [x] smoke와 전체 학습/평가를 구분했다. 후자는 미실행이다.
