# v1.6: 10mm v1의 B 부품만 교체하는 대조

원래 v1.4의 10mm 학습 결과를 보존하고, 원래 L0·정답·curriculum·전체 loss를 유지한 채 L1/L2/점수 계산만 legacy v2.2 경로로 교체한다. A를 함께 적용하거나 원래 baseline을 다시 학습하지 않는다.

## A에서 받은 서버 결과

사용자가 보내 준 터미널 요약은 `validation/v15_half_a_server_20261004/summary.json`에 별도로 보존했다. 서버 checkpoint나 원시 metric 파일을 로컬에서 독립 평가한 결과는 아니다.

| A validation | MRR | top1 | margin |
|---|---:|---:|---:|
| 초기 | 0.472355 | 0.250000 | -0.011617 |
| BEST epoch29 | 0.972222 | 0.944444 | 12.054579 |
| 마지막 epoch40 | 0.958333 | 0.916667 | 11.820597 |

40/40 epoch와 완료 기록이 보고됐다. 마지막 epoch의 4.05분은 평균 epoch 시간이나 baseline 대비 배속으로 해석하지 않는다. A는 고정 v1 task에서 순위를 학습했고, 기존 v2.2의 심한 학습 실패를 재현하지 않았다. 보고된 원래 v1의 MRR/top1 1.0 대비 감소는 남아 있으며 임의 허용 오차로 PASS를 만들지 않는다. 사용자의 승인에 따라 다음은 B만 검사한다.

## B의 실제 실행 경계

```text
원래 v1 10mm 입력·두 graph view
 → 원래 dense CNN + 3층 heterogeneous GAT L0
 → 원래 role/shell/relation을 모은 실제 fused128
 → legacy v2.2 RelationLayer L1 2층
 → 다른 support 환자 label의 alignment L2 2층
 → class별 cosine average-linkage prototype
 → patient-mass weighted cosine mixture의 logit1−logit0
 → 원래 전체 v1 curriculum ranking + 원래 two-view consistency loss
```

128D·4 heads를 유지한다. `hiercp_v222.model.PromptGraphModel`의 실제 legacy 연산을 사용하며 official PRODIGY reference L1을 사용하지 않는다. 혼합 query 환자는 disjoint L1와 padded batched L2로 계산한다. 같은 episode 내부의 모든 eligible support를 사용하며 query마다 원래 여덟 후보를 유지한다.

원래 L0 객체·초기 tensor·forward는 그대로이고, 원래 상위 모듈 다섯 개를 제거하고 새 상위 모듈로 교체한다. 원래 constructor가 만든 L0 초기 tensor는 seed42 기준 byte 일치한다. 새 상위 초기화는 독립 CPU RNG 구간에서 실행하여 caller의 CPU/CUDA RNG를 바꾸지 않는다. AdamW·scheduler·AMP scaler는 원래 recipe의 fresh 상태로 시작한다. 학습이 완료된 baseline 가중치를 새 B의 시작 가중치로 가져오지 않는다.

실제 parameter는 6,433,126개다. 유지한 원래 L0는 5,600,740개, 새 상위는 832,386개다. parameter 수 차이는 사용자가 지정한 부품 교체에 따른 것이며 층·차원·head를 편의상 줄인 결과가 아니다.

GT는 source tumor의 original anchor가 index0인 원래 v1 계약이다. Support class1은 그 anchor, class0은 같은 원본 curriculum의 나머지 일곱 후보다. 이것을 생물학적 종양 유무나 CP 적합/부적합으로 재정의하지 않는다. v2.2 observation CE 또는 alignment CE를 loss에 추가하지 않는다. 따라서 이 실험은 **고정 v1 task의 architecture bridge**이며 native v2.2 전체 학습과 동등하다고 주장하지 않는다.

## Support와 재개

완료된 baseline의 signed training cache 파일 전체를 별도의 deterministic loader로 읽는다. 설정 cohort84/21과 실제 materialized/eligible case·sample 수를 구분해서 기록한다. 현재 query 환자의 모든 support를 제외하며 validation 환자는 support bank에 들어갈 수 없다. Query label은 forward와 clustering 입력에 전달하지 않는다. 환자를 제외한 뒤 최소 두 개의 다른 실제 support task가 필요하므로 전체 actual training support는 최소 세 환자를 요구한다.

Memory는 원래 fixed epoch0 두 view를 eval/no-grad L0로 읽은 detached FP32 표현이다. 초기 validation 전에 만들고, 매 epoch optimization이 끝난 뒤 validation 전에 현재 L0 가중치로 전체를 갱신한다. 다음 training epoch는 그 bank를 사용한다. 재개 시에는 복원한 가중치에서 같은 fixed view bank를 재구성한다. 학습 query의 원래 epoch별 view·shuffle·worker RNG는 별도로 유지한다. Cluster plan은 refresh/excluded patient마다 detached eval teacher로 한 번 만들고, 학습 중 L1/L2는 매 forward에서 live gradient로 계산한다.

각 전체 refresh에 실제 sample/candidate 수·내용 hash·시간·VRAM·자원을 기록한다. 원래 pass peak는 refresh 전에 재설정되므로 그 값을 그대로 보존하고, `EpochPostRun.half_B_epoch_resources.whole_epoch_peak_vram_allocated_bytes`에 training·validation·support의 최대를 별도로 기록한다. `support_refresh_seconds`는 재개 시 재구성까지 포함한다. Epoch wall에는 support가 포함된다.

새 B 폴더의 `results/half_B/checkpoint_best.last.pt`가 원래 완료 epoch 재개 경로다. 같은 명령을 다시 실행하면 자신의 B 결과만 재개한다. A·baseline·DEBUG 상태는 architecture/manifest/scope/모델 recipe marker가 다르면 거부한다. `checkpoint_best.pt`는 원래 MRR·top1·margin 우선 선택 기준을 유지한다. 완료 skip 전에 실제 best/last 결속도 검사한다.

## 실제 로컬 검증

원시 결과와 source SHA 목록은 `validation/v16_half_b_20261004/manifest.json`에 결속했다.

- 메타데이터·실행 계약 UNIT: 36 PASS. 실제 archived constructor/loader 검사 안에서 9개 identity assertion도 통과했다. 이 CPU 검사는 의학 모델 성능 검사로 보고하지 않는다.
- 실제 GPU 연산 대조: 3 PASS. 혼합 query 환자, unequal episode padding, 같은 환자 반복, checkpointed training 경로를 실제 native `prepare_support/predict_embeddings`와 대조했다. 최대 score 차이 4.77e-7, query gradient 차이 5.22e-8, 상위 parameter gradient 차이 9.54e-7이다. Dropout0의 합성 128D 입력으로 연산 동등성을 검사한 것이며 실제 CT 정확도 검사가 아니다.
- 실제 CT/CUDA DEBUG: 8 optimizer update, AMP skip0. 원래 L0부터 새 L1/L2까지 677/677 trainable parameter tensor의 gradient가 연결됐고 모든 핵심 그룹이 실제로 갱신됐다.
- 학습 query는 기존 liver5/6, 별도 평가는 liver31이다. Query를 제외하고도 두 다른 support 환자가 남도록 원본 builder로 liver1의 full8/pool128 sample을 별도로 준비했다. 실제 inner-train CT만 사용했고 원래 query/GT/prototype bank는 바꾸지 않았다. DEBUG support는 세 환자이며 production의 전체 support를 검증한 결과로 승격하지 않는다.

| 8 update DEBUG | 초기 | 마지막 |
|---|---:|---:|
| train MRR | 0.1875 | 1.0000 |
| train top1 | 0.0000 | 1.0000 |
| train margin | -0.13169 | 1.00311 |
| held-out MRR | 0.14286 | 0.16667 |
| held-out top1 | 0.0000 | 0.0000 |
| held-out margin | -0.17177 | -1.65837 |

별도 환자의 추천 품질 개선을 확인하지 못했다. 이 작은 검사는 실행·gradient·짧은 학습 신호 검사이며 B의 전체 성능 판정은 서버40epoch에서 한다. Peak allocated VRAM 2.486GiB, process RSS 8.778GiB, 전체 DEBUG wall182.49초였다. Update median10.351초는 두 training sample의 기록이며 loader·전체 cohort support 갱신·checkpoint까지 포함한 서버 epoch 예측으로 사용하지 않는다.

## 서버 실행 계약

Entry: `tools/run_v1_half_b.py`. 고정 원래 baseline:
`/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004`.

새 실험 폴더:
`/home/aicompetition06/Medical/experiments/v16_m10_halfB_seed42_20261004`.

물리 GPU3, CUDA40GiB/RSS192GiB 한도에서 실행한다. Physical batch와 worker는 baseline의 실제 preflight를 읽어 같은 값으로 lock한다. 보고된 값은1/4이지만 요약 숫자를 가정하여 적용하지 않는다. 이것은 비교 조건이며 새 상위 구조의 throughput 최적값을 주장하지 않는다. 동일 자원 fingerprint도 확인한다. 입력·baseline cache·원본 source가 실제 proof와 다르면 명확한 오류로 중단하며 자동 재준비·축소·fallback을 하지 않는다.

Seed42,40epochs,84/21,outer26제외,pool128,curriculum8,원래 GT와 loss를 유지한다. Baseline·A 재학습, A+B, production CP, nnU-Net은 실행하지 않는다. 로컬에서 전체 학습 또는 전체 evaluation은 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검사에서는 OOM이 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
