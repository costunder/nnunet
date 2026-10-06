# v2.22 r2 — 환자 간 군집 중심으로 L2 정렬

2026-09-22. Format: `hiercp_v222_cluster_alignment_r2`. 사용자가 승인한 클러스터 기반 L2를 구현한다. r1은 `versions/v2.22/before_cluster_alignment_r2_20260922/`에 보존했다. [전체 파이프라인](pipeline_v222.md)의 L0·L1 정의는 유지한다. [References](../REFERENCES.md)에 원문·적용 위치·변형 범위를 모았다.

## 무엇을 정렬하는가

L1의 각 환자 task에는 기존 두 관측 클래스의 128차원 label 표현이 있다. 환자마다 국소 문맥이 달라 이 표현도 달라진다. **동일 관측 클래스의 환자 표현을 군집으로 모아 공통 중심을 만들고, L2 출력이 자신이 속한 중심을 다른 중심과 구분하도록 학습한다.** Class 0/1과 그 안의 문맥 군집 K는 다른 개념이다. K는 종양 종류 수나 class 수가 아니다.

L0의 CT-only CNN, 고정 가림과 전체 공간 그래프, GATv2 3층/128/4 heads 및 L1 관계 attention 2층은 r1과 같다. 종양 내부·수작업 통계·상대 좌표 특징을 추가하지 않았다. L2의 2층/4-head attention과 residual/FFN도 유지하며 그 출력에 군집 학습과 prototype scoring을 연결했다. 전체 parameter 수는 1,519,063개로 유지된다. 군집 중심은 추가 학습 파라미터가 아니다.

## 1. 먼저 query 환자를 제외한다

`support_for_query`가 같은 `patient_group`의 모든 검사·패치·종양 기록을 제거한다. 남은 검사도 환자 그룹별로 병합한다. **이후에만** `fit_support_clusters`를 호출한다. 전체 환자로 중심을 먼저 만든 다음 query만 가리는 방식은 사용하지 않는다.

학습은 inner-train 환자 group 하나를 query로 두는 episode다. 군집 입력은 나머지 inner-train support의 L1 label뿐이다. Inner-val/outer-val 데이터, query CT와 query 정답은 군집 fit 함수 인수에 없다. Inner-val은 epoch 후 checkpoint 선택에만 쓰며 이 간접적인 모델 선택 역할과 군집 fit을 구분한다. 환자 독립성은 정확한 identity manifest에 의존한다.

L1에는 클래스마다 label node가 있지만 그 환자에게 해당 클래스 관측이 없을 수 있다. 그 node는 L1 문맥 구조에 남고 **군집 근거에서는 제외**한다. 양성 관측이 없는 환자의 가상 양성 label로 양성 중심을 만들지 않는다. 이 환자의 실제 비교 클래스 근거는 유지된다. 실제 관측이 존재하는 두 클래스가 support 전체에 없으면 명시적 오류다.

## 2. 관측 클래스별 군집과 K 결정

군집에 들어가는 단위는 병변/패치마다 하나가 아니라 **실제 관측 근거가 있는 환자별 class label 하나**다. 종양이 많은 환자가 중심을 여러 표로 지배하지 않는다.

1. 해당 episode 시작 시 dropout을 끈 L1을 `no_grad`로 실행한다. 정규화된 label 방향을 teacher로 사용한다.
2. 클래스별 모든 환자 label 사이의 cosine 거리로 average-linkage 계층을 만든다. 수작업 영상 특징으로 군집화하지 않는다.
3. N명에 대해 가능한 K=2…floor(N/2)의 모든 cut을 검사한다. 모든 군집에 최소 두 독립 환자가 있어야 submode 후보로 인정한다. 이 조건은 그래프/환자 수를 잘라내는 cap이 아니며, 탈락한 환자도 최종 K=1 또는 다른 군집 안에 남는다.
4. 적격 cut 중 **양의 평균 silhouette가 가장 큰 K**를 선택한다. 동점은 작은 K다. 적격 양의 점수가 없거나 N<4이면 K=1로 기록한다. 완전히 같은 방향도 명시적 미분리 결과다. NaN·zero vector·상쇄되어 방향을 정의할 수 없는 중심은 오류로 보고한다.
5. 군집 중심은 정규화된 member label들의 평균을 다시 단위 길이로 만든다. Assignment와 teacher center는 해당 환자 episode 동안 고정하고 다음 episode에서 새 support로 계산한다.

최소 2명 조건은 ‘환자 간 반복되는 submode’라는 목적에서 정한 프로젝트 규칙이다. 클래스 전체 관측이 1명뿐인 경우 K=1은 한 명의 class 근거이며 다환자 submode라고 해석하지 않는다. Silhouette는 군집 내/간 분리의 내부 지표일 뿐, 임상적 의미나 올바른 K를 보장하지 않는다. K=1과 K>1의 가설검정도 아니다. 불균형·고차원 표현에서의 과분할 가능성은 전체 cohort에서 평가해야 한다.

## 3. L2 학습과 CP 점수

훈련 중 live L1 표현은 기존 L2의 환자 간 attention 2층을 통과한다. 같은 환자를 향하는 직접 attention은 기존대로 가린다. 군집 assignment는 고정되지만 **live L2 출력은 역전파된다.**

고정 teacher 중심을 C, live L2 표현을 z, 군집 target을 a라고 하면:

```text
alignment_logits = cosine(z, C) / 0.2
L_align = class-balanced mean CE(alignment_logits, a)
L_total = weighted query-class CE + 1.0 * L_align
```

Alignment CE는 자기 군집뿐 아니라 다른 문맥 군집과 반대 관측 클래스의 중심도 경쟁 대상으로 사용한다. 단순히 모든 벡터를 서로 가깝게 하는 loss가 아니다. 클래스별 K=1이어도 총 두 클래스 중심을 구분하는 objective다. 고정 teacher를 써서 같은 step의 양쪽 표현이 함께 이동하는 문제를 줄인다. 그렇다고 장기적인 collapse 방지가 증명되는 것은 아니다. 중심 간 최대 cosine, 클래스 내 방향 분산, 군집 점유 수와 epoch 간 ARI를 기록한다. 훈련 진행 중 군집 붕괴/불안정성은 이 기록으로 점검해야 한다.

두 CE는 각각 평균을 사용하며 가중치 1.0은 명시적인 초기 설계다. 임상적으로 최적화된 값이 아니며 query-only r1 대비 ablation이 필요하다. Temperature 0.2는 기존 값이다.

Query 분류에서는 고정 assignment로 **live L2 출력의 군집 중심**을 계산한다. 이렇게 해야 query CE도 L2로 역전파된다. Class c의 logit은 다음과 같다.

```text
live_center[k] = normalize(mean(normalize(z_i) for i assigned to k))
class_logit[c] = log sum_{k in class c} (n_k / N_c) * exp(cosine(query, live_center[k]) / 0.2)
CP_score = softmax(class_logits)[observed_small_tumor_context]
```

Softmax 기반 center 경쟁과 군집 mixture scoring을 사용한다. Assignment target 자체는 hard cluster index다. Soft assignment target이나 Sinkhorn을 구현한 것처럼 부르지 않는다. 군집 수가 많은 클래스가 중심 개수만으로 유리해지지 않도록 class 안에서 환자 수 가중치를 합 1로 정규화한다. 이 가중 방식도 프로젝트 선택이다. 점수는 **관측 문맥 순위**, 보정된 암 발생 확률이 아니다.

`L_align → L2 → support L1`과 `query CE → query L0/L1 및 support L1/L2 → live centers`가 모두 연결된다. Support L0는 기존 detached memory이므로 정렬 loss가 support CNN으로 직접 역전파되지는 않는다. 모든 inner-train patch를 query로 방문하는 CE 경로가 L0를 학습한다.

## 4. 갱신·실행·감사

- Support L0 memory: 전체 inner-train을 매 epoch 재인코딩한다. L1 teacher/군집은 각 query 환자 episode 시작 시 한 번 fit하고 batch마다 재사용한다. Episode 안에서도 live L1/L2는 optimizer로 갱신된다. Teacher와 현재 표현 사이 지연을 의도적으로 허용한 교대 학습이다.
- Evaluation: query 환자별 고정 support state를 만들고 모든 해당 query batch에서 재사용한다. Query 정답은 예측 후 metric 계산에만 쓴다.
- Frozen CP scorer: 환자 group별 support state를 캐시한다. 128개 후보 모두 같은 고정 중심을 사용한다. 후보/정답에 따라 K나 중심을 재적합하지 않는다.
- 기록: 각 epoch JSON의 `cluster_episodes`에 K 후보별 silhouette/점유 수, 선택 사유, support 환자 목록, 제외 query group, 클래스별 이전 epoch 대비 ARI, 중심 유사도·분산·fit 시간을 저장한다. ARI는 모니터링 지표이며 K 선택에 query 정답을 사용하는 통로가 아니다.
- 고정 global cluster ID를 가정하지 않는다. Query 환자 제외 집합과 epoch별 표현에 따라 K와 군집 번호가 바뀔 수 있다. 두 관측 클래스의 의미는 계속 고정된다.
- 40 GNN epochs, 250 nnU-Net epochs, 전체 support, 모든 적격 양성/정의된 비교 위치, 후보 128개, CP 확률 0.5는 유지한다. Full-cohort batch/worker는 기존 실제 측정 절차를 유지한다.
- 군집 계산은 작은 환자 label 행렬에 대한 episode별 CPU 작업이다. Query별 CT graph의 GPU 계산은 기존 batched 경로를 유지한다. 환자·패치별 GPU 직렬 forward를 추가하지 않았다.
- r1의 cache/checkpoint는 r2와 format/source hash가 다르므로 그대로 production 입력으로 받지 않는다. 버전 문자열만 바꾸는 전환은 허용하지 않는다. Native 데이터 재사용은 기존 검증된 `reuse-native` 경로를 따른다.

## 검증과 제한

검증 스크립트: `tools/verify_v222_cluster_debug.py`. 실제 원본 CT/주석 hash를 확인하고 보존된 6개 patch를 원본에서 재계산해 일치시켰다. `liver_1/5/6`의 DEBUG contract를 사용했으며 전체 cohort contract로 저장하지 않았다.

| 항목 | 결과 |
|---|---|
| 실제 입력 | 3명, 6개 CT 문맥 그래프. Query 1명, support 2명 |
| 그래프 / 모델 | 그래프당 24,174 nodes / 1,229,900 edges, 1,519,063 parameters |
| CUDA batch 2 | 3회 측정, 1.585 graphs/s, peak 611,964,928 bytes |
| CUDA batch 4 | 3회 측정, 1.716 graphs/s, peak 1,100,222,976 bytes. 실제 query 2개를 반복한 batch probe |
| Gradient/optimizer | 모든 파라미터 finite gradient, CNN/L0/L1/L2 및 L2 residual 그룹 갱신 |
| 실제 CT 군집 | 클래스당 support 2명이므로 K=1/1. 의료 다중 군집을 검증한 결과 아님 |
| 합성 DEBUG | K=2/3/4 분리, 단독 outlier 처리, 순열 불변성, missing-class 제외, query 환자 개입 불변성, 고정 teacher, 두 loss의 별도 gradient, center 변경에 따른 점수 변화, 군집 수 편향 검사 |

최종 소스와 hash가 일치하는 결과: `work/v222_20260922/cluster_r2_final_debug/result.json`. 회귀 39개 통과, 정적 검사 18파일, r1 대비 L0·L1 AST 일치 및 이전 버전 보존 확인은 `versions/v2.22/verification_cluster_r2_20260922/`에 기록했다. Frozen 추론 캐시에서 전체 support L0 buffer를 해제해도 128개 합성 query 점수가 정확히 유지되는지 추가 확인했다. 합성 자료는 알고리즘 반례 검사에만 사용했고 실제 CT 학습 결과로 제출하지 않는다.

**전체 학습·분할 평가·소형 종양 성능 개선 검증은 미실행이다.** Production 환자 identity/annotation manifest 검증과 전체 가림 범위 감사가 필요하다. 본 연구 설계에는 아직 전체 cohort에서의 K 안정성, cluster collapse 여부, 정렬 loss 가중치·점수 방식 ablation이 남아 있다. 스모크 테스트 성공으로 이를 대신하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. r1 소스를 별도 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. Full-cohort 보정은 본 학습 때 수행한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 이번 변경에서 OOM은 없었으며 모델 축소를 적용하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 검사는 명시적인 DEBUG다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
