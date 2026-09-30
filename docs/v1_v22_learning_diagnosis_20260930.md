# v1 / v2.2 학습 품질 로컬 진단 — 2026-09-30

## 판단

그래프 축약만 원인으로 지목할 수 없다. v1과 v2.2 사이에는 **donor를 고정하는 비교 단위, 정답의 의미, loss, gradient가 흐르는 후보, best checkpoint 선택 기준**까지 바뀌었다. 현재 fine GraphSAGE 경로도 이 학습 계약을 그대로 사용하므로 축약 제거만으로 해결됐다고 볼 수 없다.

우선순위는 **학습의 비교 단위를 실제 CP의 ‘동일 donor에 대한 위치 비교’와 맞추는 것**, 양성/미관측 비교가 CNN까지 함께 학습되는지 보장하는 것, 그리고 top-ranking과 checkpoint 선택을 맞추는 것이다. 이 문서는 변경 제안이며 production 학습 계약을 수정하지 않았다. 정확도 향상을 보장하거나 특정 항목을 서버 성능 저하의 단독 원인으로 확정하지 않는다.

## 실제 확인한 범위

- 코드 기준: `4389182`, `codex/v222-server-r6`.
- 보존된 v1 소스 ZIP을 직접 읽었다. loss/curriculum/local/model/pipeline 5개 파일이 현재 `hiercp/` 파일과 byte 단위 동일함을 SHA256으로 확인했다.
- 로컬 전체 paired inventory **14,102 records**를 읽어 train 11,279 / validation 2,823의 donor·관측·batch 구성을 집계했다.
- RTX 5070 Ti 16 GiB에서 보존된 **실제 CT DEBUG 8pair**의 동일 숫자 가중치로 축약/미축약을 대조했다. 순위·관측 보조·L2 loss별 gradient는 2개 query와 다른 환자의 support로 검사했다.
- 동일 초기화와 기존 **4 update DEBUG checkpoint** 두 상태를 검사했다. 새로운 optimizer update, partition 준비, production checkpoint 생성, 서버/장기 학습은 수행하지 않았다.
- 시작 자원: 논리 CPU 16개, 사용 가능 RAM 약 50.1 GB, CUDA free 약 15.7 GB. loader worker 8, DEBUG CUDA 예산 8 GiB / RSS 16 GiB. 원래 physical batch 32를 바꾸지 않은 별도 8pair 진단이다.
- 서버에서 학습한 v1 및 최신 v2.2 가중치는 이번 로컬 대조에 없다. 따라서 **같은 holdout에서 두 학습 완료 모델을 직접 평가한 결과가 아니다.**
- 전체 inventory는 `work/v222_v1_recovered2_training_20260924/cache/index.json`이다. 서버 최신 region cache를 내려받은 것은 아니므로 donor/batch 집계는 로컬 inventory 기준이며, 서버 최신 기하의 edge 순서까지 동일하다고 주장하지 않는다.

## 1. v1에서 현재까지 무엇이 달라졌나

| 항목 | 보존된 v1 | 현재 v2.2 |
|---|---|---|
| 비교 단위 | donor 하나 + 그 donor의 원래 위치와 비교 후보 | 각 observation마다 별도 train donor 배정; case별 P/U 순위 비교 |
| 양성 | source donor의 원래 anchor | recipient의 적격 소형 종양 관측 anchor |
| 음성/미관측 | easy/inter-region/intra-corruption curriculum | 주석상 미관측 위치; CP 부적합 정답은 아님 |
| 학습 후보 | ZIP 설정의 총 후보 8, 후보 pool 128 | case당 미관측 128 + 모든 적격 양성 |
| 순위 학습 | 후보 묶음의 listwise CE + pairwise margin + ordinal/mining | 관측/미관측 pairwise logistic + 관측 분류 CE + L2 |
| L0 gradient | native 후보 묶음의 현재 forward | live mini-batch만 현재 L0; 다른 후보 L0는 epoch memory detach |
| best 선택 | MRR → top1 → margin 등의 사전식 순서 | validation pairwise loss 최소 |
| 기타 변경 | 5채널·수작업 특징·종양 내부·target erase, patient/population graph | raw CT 1채널, 기존 폐기 특징 없음; data-label L1 및 clustering L2 |

v1의 모든 과거 실행이 후보 8개였다는 뜻은 아니다. 위 수치는 보존된 ZIP 설정이다. 과거 terminal 기록의 val 36개에서 top1/MRR=1.0이라는 기록은 있으나, 현재의 128개 미관측+다중 양성 관측 평가와 같은 문제가 아니다. 해당 기록은 segmentation Dice도 아니다. **이전 높은 숫자를 현재 지표와 바로 나누어 성능 저하율을 계산하면 안 된다.**

근거: `hiercp/curriculum.py:211`, `hiercp/loss.py:189`, `hiercp/pipeline.py:530`, `hiercp_v222/v1_cache.py:42`, `tools/v22_rank_objective.py:38`, `l0_regions/training.py:446`.

## 2. 가장 먼저 수정할 학습 단위 불일치

실제 CP는 donor를 하나 선택하고 그 donor를 둘 위치들을 비교한다. 현재 cache는 observation마다 donor를 독립 배정하고, ranking은 donor와 무관하게 같은 case의 양성/미관측을 비교한다.

```text
v1 및 CP에 맞는 비교       score(donor A, 위치 P) > score(donor A, 위치 U)
현재 비교에 대부분 등장   score(donor A, 위치 P) > score(donor B, 위치 U)
```

| 로컬 전체 inventory | train | validation |
|---|---:|---:|
| records | 11,279 | 2,823 |
| 양성 관측 | 527 | 135 |
| 전체 같은 case P/U pair | 67,456 | 17,280 |
| 서로 다른 donor인 P/U pair 비율 | **99.779%** | **99.838%** |
| 적격 양성이 없는 case | 19 | 5 |

이는 donor가 정답에 누수됐다는 증거는 아니다. 배정은 class 독립이다. 그러나 **위치 효과와 donor 효과를 분리한 조건부 비교를 거의 하지 않는다.** 네트워크가 donor 정보를 무시하면 관측 위치 분류에 가까워질 수 있고, donor 정보를 사용하면 위치 비교에 donor 차이가 섞일 수 있다. 어느 쪽이 실제로 발생했는지는 서버 가중치의 donor 교환 평가가 필요하다.

개선안은 같은 recipient의 비교 묶음 안에서 donor를 고정하는 것이다. 모든 observation과 후보 128개, train-only donor 및 환자 분리 계약은 유지한다. 527 donor와 모든 위치의 거대한 Cartesian product를 만들자는 뜻은 아니다. donor 선택을 episode 수준으로 옮기고, 새 pair의 기하·원본 mask 검사를 다시 결속해야 하므로 단순 ID 교체로 기존 pair cache를 재사용하면 안 된다.

또한 donor를 고정해도 recipient 관측 양성은 **그 donor를 이식한 결과의 정답**이 아니다. 여전히 ‘실제 관측 위치를 높은 점수로 학습’하는 대리 목표다. 이것과 최종 CP 효용은 구분해야 한다.

## 3. 많은 update가 현재 양성 CNN 특징과 함께 비교하지 않는다

현재 `groups()`는 환자별 edge 크기로 정렬한 뒤 32개씩 자르고 chunk 순서를 섞는다. 같은 class만 있는 chunk가 많다.

- 로컬 train 405 batch 중 **224개(55.3%)가 양성 0개**다.
- 양성이 없는 19개 case도 삭제되지 않는다. 이들은 within-case rank loss가 0이며 CE와 L2로 학습한다. 실제로 종양이 없다고 단정하는 것이 아니라 현재 적격 소형 종양 관측이 없다는 뜻이다.
- 양성이 있는 case에서도 현재 batch에 양성이 없으면, 비교 상대 양성의 **L0 embedding은 epoch memory**다. L1/L2는 현재 연산이지만 그 양성의 CNN/SAGE로 gradient는 돌아가지 않는다.
- 따라서 v1의 ‘같은 donor/현재 후보 표현을 함께 비교’와 학습 동작이 다르다. detach 자체가 불법이거나 항상 실패하는 알고리즘이라는 뜻은 아니다.

추가로 weighted CE의 기본 `mean`은 해당 batch의 class weight 합으로 나눈다. 단일 class batch에서는 그 class weight가 분자·분모에서 상쇄된다. CUDA 연산 대조에서 가중/비가중 gradient 최대 차이는 **0**이었다.

현재 전역 class weight를 적용해도 로컬 epoch의 batch별 CE 계수를 합산하면 양성 **26.8%**, 음성 **73.2%**다. 이것은 **계수 비중**이며 실제 gradient 크기 비중이 아니다. ranking과 L2까지 음성 73.2%라는 주장도 아니다.

개선 우선순위:

1. 동일 donor 비교 묶음 안의 양성·미관측을 현재 모델로 함께 비교하는 학습 경로를 설계한다. 전체 후보를 버리지 않고 계산을 묶어 재사용한다. 단순히 모든 epoch reference를 무조건 재계산하면 예전 비용 문제가 재발한다.
2. 전체 coverage와 physical batch를 유지하면서 positive-bearing case의 묶음과 reference 갱신을 감사한다. 양성을 반복 사용하면 반복 횟수에 따른 목적함수 가중을 기록해야 한다.
3. class balance의 의도가 epoch 기준인지 batch 기준인지 먼저 명시하고 CE 정규화를 맞춘다. 양성 없는 모든 batch를 버리거나 loss weight를 임의로 키우는 방식은 사용하지 않는다.

## 4. 현재 지표는 아직 강한 순위 학습을 보여주지 않는다

현재 코드와 같은 정의로 로컬 validation 21개 case / 적격 양성이 있는 16개 case / 양성 135개의 **균등 무작위 순열 기대값**을 계산했다. MRR은 case별 첫 양성 순위의 역수 평균이며, R@K는 모든 양성에 대한 micro recall이다. 서로 다른 지표이므로 MRR 0.16이 top1 16%를 뜻하지 않는다.

| 비교 | MRR | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| 로컬 구성의 무작위 순위 기대값 | 0.1605 | 0.0069 | 0.0346 | 0.0692 |
| 사용자 서버 로그 epoch 3 | 0.189 | 0.007 | 0.037 | 0.089 |
| 사용자 서버 로그 epoch 4 | 0.178 | 0.007 | 0.044 | 0.089 |
| 사용자 서버 로그 epoch 5 | 0.116 | 0.000 | 0.037 | 0.089 |

이는 현재 성능을 좋다고 보고할 근거가 약하다는 뜻이다. 현재 수치가 통계적으로 무작위보다 나쁘다는 결론은 아니다. 10,000개 독립 무작위 순위의 MRR 중앙 95% 구간은 **0.0747–0.2807**이었다. 이는 모델 성능의 신뢰구간이 아닌 null 분포 구간이다. 서버 candidate 구성을 직접 내려받아 확정한 검정도 아니다.

epoch 4는 pairwise loss가 0.6871로 낮아져 best로 선택됐지만 MRR은 epoch 3의 0.189에서 0.178로 떨어졌다. **현재 best는 추천 상위 순위가 가장 좋은 모델이라는 뜻이 아니다.** v1은 MRR/top1 우선이었다. 추천 목적에 맞는 primary ranking metric을 미리 정하고 best 선택에 연결해야 한다. 기존 결과에서 유리한 epoch만 사후 선택해 최종 성능으로 제시해서는 안 된다.

무작위 기대값 공식은 N=2..8의 모든 양성 수와 모든 배치 조합을 열거하여 검증했다. random reference는 모델 예측으로 사용하지 않았다.

## 5. 축약의 실제 영향과 gradient 검사

동일 8pair에서 fine **52,666 nodes / 2,761,602 edges**, coarse **13,868 / 379,049**였다. 노드는 73.7%, edge는 86.3% 줄었다. 압축은 실제로 발생했다.

현재 CNN 특징을 cluster 평균으로 대체했을 때 제거되는 역할 내 제곱편차 비율을 측정했다. 아래는 기존 4-update DEBUG 가중치 기준 pair별 비율 평균이다.

| 역할 | 제거되는 특징 제곱편차 평균 | 최대 |
|---|---:|---:|
| source context | 47.3% | 83.7% |
| target context | 26.4% | 60.6% |
| source liver surface | 15.8% | 29.5% |
| target liver surface | 6.4% | 13.1% |
| tumor surface | 34.3% | 54.5% |

tumor surface는 분모 분산이 0인 2pair를 `null`로 보존하고 나머지 6pair를 집계했다. 다른 역할은 8pair다. **이 비율은 정보 손실의 기하/특징 지표이며 암 관련 유효 정보 손실률이나 정확도 손실률이 아니다.**

동일 숫자 가중치를 두 구조에 넣었을 때 최종 L0 embedding cosine은 초기화에서 0.9703–0.9975, 4-update DEBUG 가중치에서 0.9853–0.9984였다. 2개 query의 rank loss는 다음과 같았다.

| 동일 가중치 상태 | coarse | fine |
|---|---:|---:|
| 초기화 | 0.7570 | 0.7583 |
| 기존 4-update DEBUG | 0.6485 | 0.6429 |

이 작은 차이로 장기 품질이 같다고 말할 수 없다. 그러나 **이번 검사만으로 축약 제거가 품질 문제를 해결했다는 결론도 낼 수 없다.** DEBUG checkpoint를 fine 모델에 대입한 것은 진단이며 coarse checkpoint의 exact resume를 허용한 것이 아니다.

gradient 검사에서는 순위 loss가 CNN/SAGE/L1/L2 모두에 전달됐다. 4-update fine 상태의 모듈별 norm은 각각 **9.0796 / 3.2437 / 1.1512 / 0.7369**였다. L2 alignment 단독 gradient는 CNN/SAGE=0이며 L1/L2에는 존재했다. 현재 L2가 detached support에서 계산되기 때문이다. 전체 학습에서 CNN gradient가 없다는 뜻은 아니다.

순위/관측 보조 gradient는 대체로 같은 방향이었다. 순위/L2 gradient cosine은 초기화에서는 약 -0.19였지만 4-update에서는 양수였다. **L2가 지속적으로 순위 학습을 방해한다고 판정할 근거는 없다.** loss 삭제나 계수 변경을 적용하지 않았다.

## 6. 다음 변경의 순서

1. **데이터 비교 단위:** 같은 donor 조건의 positive/128개 미관측 비교. 정답은 관측 위치의 대리 목표임을 유지하며 split·전체 observation·원본 paste 검사 보존.
2. **학습 연결:** 현재 positive/negative 표현의 gradient 및 epoch-memory 의존도를 함께 설계. 학습 묶음과 CE 정규화의 실제 class 기여 감사. old checkpoint exact resume가 아닌 새 학습 계약으로 분리.
3. **평가와 best:** 같은 후보·같은 donor·같은 metric으로 v1식/현재식 목표를 비교. 전체 미관측 위치 recall과 관측 위치 순위를 구분. 사전 지정한 추천 ranking metric에 best를 맞춤.
4. **구조:** 사용자 요청대로 fine SAGE를 기준으로 남긴다. 지금 PE·층 수·GAT·새 clustering을 추가하면 원인 분리가 더 어려워진다. 축약 재도입은 위 목표가 유효한지 확인한 뒤 별도 ablation.

즉, 지금 먼저 손댈 곳은 `assign_pairs()`와 `RankingContext`가 정의하는 **학습 예제/순위 비교 계약**이다. GraphSAGE를 다시 GAT로 바꾸거나 CNN을 키우는 것이 첫 조치가 아니다. 구현 전 같은 donor 조건을 지키는 실제 예제와 short GPU gradient 검사를 먼저 준비하고, 전체 규모 학습은 서버에서 수행해야 한다.

## 결과 파일과 재현

- 코드: `tools/audit_v1_v22_learning_debug.py`
- 원시 결과: `validation/v1_v22_learning_diagnosis_20260930/report.json`
- 수식/원본 대조: `validation/v1_v22_learning_diagnosis_20260930/analytical_checks.json`
- 작업 산출물 원본: `work/v1_v22_learning_audit_DEBUG_20260930/`
- 과거 v1 terminal 근거: `experiment_results/recovered_conversations_20260918/terminal_records/ee0258b7-a9e4-40b2-a0e7-506017907681.txt`

```powershell
.venv/Scripts/python.exe -B -u tools/audit_v1_v22_learning_debug.py `
  --full-index work/v222_v1_recovered2_training_20260924/cache/index.json `
  --region-cache work/region_frozen_reuse_cli_20260929/cache/index.json `
  --fine-cache work/v22_rereview_20260927_DEBUG/cache/index.json `
  --debug-checkpoint work/region_learning_display_DEBUG_20260930/overlapped/checkpoint_latest.pt `
  --output work/v1_v22_learning_audit_DEBUG_new
```

출력 폴더가 이미 존재하면 실패하며 덮어쓰지 않는다. 예시의 `_new`도 신규 경로로 지정해야 한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 전체 metadata와 명시적 8pair DEBUG를 구분했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토했다. production 32는 유지했으며 이번은 처리량 calibration이 아니다.
- [x] GPU, CPU, RAM 활용 상태를 확인했다.
- [x] 모델 축소 대신 학습 계약과 gradient를 조사했다. 이번 실행에서 OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 무작위 순위는 명시적 통계 기준선이다.
- [x] 핵심 모듈의 실제 gradient를 확인했다. 이번 읽기 전용 진단은 optimizer update를 실행하지 않았다.
- [x] 실제 실행 설정과 미검증 범위를 보고했다.
- [x] 짧은 실제 CT GPU 진단과 전체 학습/평가를 구분했다. 후자는 이번에 수행하지 않았다.
