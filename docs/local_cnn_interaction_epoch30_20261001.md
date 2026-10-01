# v2.2 Local-CNN — epoch 필드30의 L1 interaction 서버 결과

**서버 진단 실행은 완료됐지만, L1에 내적항을 더하는 수정으로 순위 학습 문제가 해결되지는 않았다.**
이는 현재 snapshot의 고정 가중치 비교와 두 branch의 4update에서 확인된 범위다.
새 구조를 충분히 재학습한 최종 성능까지 실패했다고 단정하는 결과는 아니다.
Production 모델·loss·checkpoint·ready 표시를 변경하지 않았다.

## 근거와 snapshot

사용자 첨부 `b764eb80-9ef2-412f-add1-5b1053f1fe50`의 전체13743bytes를
`validation/local_cnn_interaction_20261001/server_epoch30_summary.txt`에 그대로 보존했다.
SHA256는 `460049870fa321330cd02e7725266ab9ce5baff3a6d8174448c94e38794844af`다.
같은 폴더의 `server_epoch30_transcription.json`은 출력 수치와 표시 정밀도를 보존한
전사이며 **원본 서버 report JSON이 아니다.** 원본 JSON·학습 가중치는 로컬 미수신이다.

| 항목 | 출력에서 확인된 값 |
|---|---|
| 실험 | `/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42` |
| 진단 source revision | `0596e8f70f486bca69d9a1f60a768604a30a70ab` |
| epoch 필드 / step / phase | **30 / 16431 / optimization** |
| margin / physical batch / parameter 수 | 10mm / 32 / 1,125,718 |
| GPU | NVIDIA RTX A6000, 선택 index3 |
| CUDA / RSS 한도 / workers | 24GiB / 64GiB / 8 |
| 진단 peak GPU 할당 | 1.81GiB |
| checkpoint **content** SHA256 | `a303115d0643e45a3e3df1a6e50c7c01d6f284c44ca56854ee950d694aa3257e` |
| 원본 report의 서버 경로 | `/home/aicompetition06/Medical/experiments/l1_interaction_20261001_133109.json` |
| 종료 상태 | 오류 없이 report 저장 후 셸 프롬프트 복귀 |

현재 run은 **국소 CNN L0**다. 디렉터리 이름 `l0_regions`를 보고 GraphSAGE나
EZ-SP가 이번 forward에 사용됐다고 설명하지 않는다. 이번 진단 후보의 변화는
복제된 두 L1의 `existing_MLP([q,k,edge]) + beta·dot(q,k)/sqrt(head_width)`뿐이다.
기존 L0/L2/loss·층수·hidden128·physical batch32·후보128·원본 mask는 유지됐다.

Epoch22/step11872, epoch27/step14924와 이번 snapshot은 서로 다른 가중치다.
이전 수치로 이번 report의 빠진 필드를 채우거나 동일 가중치 A/B처럼 합치지 않는다.

## 고정 가중치 비교

선택 train2case, validation2case 각각의 원래 후보 전체를 사용했다. N은 관측
양성 P와 미관측 후보128의 합이다. Beta0 parity, support/query의 동일 방정식,
원본 가중치 보존은 네 case 모두 true로 출력됐다.

| case | N / P | 기본 score std | 기본 pair win → beta1 | 기본 pair loss → beta1 |
|---|---:|---:|---:|---:|
| train liver_1 | 139 / 11 | 1.117106e-5 | .5511364 → .5546875 | .6931459 → .6931459 |
| train liver_49 | 148 / 20 | 6.85948e-6 | .5406250 → .5410156 | .6931464 → .6931465 |
| validation liver_109 | 155 / 27 | 1.394262e-6 | .5121528 → .5133102 | .6931472 → .6931472 |
| validation liver_3 | 129 / 1 | 1.199617e-6 | .9687500 → .9765625 | .6931465 → .6931465 |

Beta0.25 결과 및 head별 weight variance/JS/cosine은 전사 JSON에 모두 보존했다.
내적항을 더하면 attention weight variance가 증가하지만 절대값은 여전히 매우
작다. Beta1의 둘째 L1 head variance는 대략2.75e-15~2.95e-12이며 후보쌍
head-mean cosine은 표시 정밀도에서1 또는0.9999999다. Bitwise 동일하다는 뜻은 아니다.

최종 점수 spread는 1e-6~1e-5 수준을 유지하고, pair loss는 `ln(2)` 근처다.
일부 ordering 변화가 있지만 고정 가중치에서 후보 구분 능력을 회복한 결과로
판정하지 않는다. Score spread·attention variance 자체는 정확도 지표가 아니다.

원본 L1의 정규화 후보 방향 분산은 다음과 같다.

| case | L0 | L1_1 | L1_2 | L1_1 / L1_2 |
|---|---:|---:|---:|---:|
| train liver_1 | 2.09e-6 | 1.49e-7 | 6.59e-11 | 약2,261 |
| train liver_49 | 8.14e-7 | 5.77e-8 | 2.49e-11 | 약2,317 |
| validation liver_109 | 5.85e-8 | 4.13e-9 | 1.86e-12 | 약2,220 |
| validation liver_3 | 4.91e-8 | 3.54e-9 | 1.59e-12 | 약2,226 |

L0→첫째 L1에서도 약14배 줄어든다. 이 비율은 반올림된 출력의 파생값이며
정확한 정보 손실률·정확도 감소율이 아니다. 강한 표현 수축은 재확인됐지만,
해당 수축 하나가 성능 저하의 유일 원인이라고 확정하지 않는다.

## 같은 full objective의 짧은 update 비교

두 branch는 각각 **fresh AdamW 4update, physical batch32**다. 전체 epoch
schedule의 명시적 prefix를 사용했고 고유 관측106/11,279개(약0.94%)가 포함됐다.
전체 관측을 최종 학습에서 줄인 것이 아니다. CNN·fusion·L1/L2가 현재 objective에
연결돼 있으며, CNN gradient 및 parameter 변화가 두 branch에서 출력됐다.

| 범위 / 지표 | 원본 beta0: 전 → 후 | 후보 beta1: 전 → 후 |
|---|---:|---:|
| train MRR | .06696429 → .125 | .1047619 → .07177033 |
| train pair win | .5443549 → .5602319 | .5458669 → .5357863 |
| train pair loss | .6931462 → .6931459 | .6931462 → .6931459 |
| validation MRR | .6 → .2083333 | .625 → .5625 |
| validation pair win | .5284598 → .4988839 | .5298549 → .4974889 |
| validation pair loss | .6931471 → .6931472 | .6931471 → .6931472 |

후보의 post-update MRR .5625가 원본의 .2083333보다 높다는 한 항목만으로
개선했다고 말하지 않는다. **두 branch 모두 자기 update 전 validation MRR보다
낮아졌고 pair win도 약50%로 내려갔다.** 후보는 train MRR/win도 내려갔다.
원본보다 덜 떨어진 일부 sampled MRR과 학습 효과가 확인됐다는 주장은 다르다.

`tools/v22_rank_objective.py::ranking_metrics`의 MRR은 case별 **첫 관측 양성의
순위 역수**를 평균낸 값이다. 모든 양성 각각의 MRR이나 정확도100%가 아니다.
Liver_109에는 양성27개가 있어 그중 하나만1위여도 해당 case MRR은1이다.
원본 case의 MRR은 liver_109=1, liver_3=.2다. 후자는 유일한 양성이5위라는 뜻이며,
124/128개 미관측 후보보다 높은 순위라 pair win .96875와 모순되지 않는다.
전체 양성 순위·mean rank·recall도 함께 읽어야 하나 이번 콘솔에는 출력되지 않았다.
Raw report의 실제 값을 수신하기 전에는 추정해서 전사하지 않는다.

CNN gradient norm은 원본 .0001737692~.0003023104, 후보
.0001695883~.0003077042이고 parameter delta norm은 .07675509/.07779981이다.
이는 역전파와 optimizer가 실행됐다는 근거이며 좋은 특징을 학습했다는 성능 증거는 아니다.

비교 경계는 **FP32, fresh optimizer, 양쪽 동일 saved detached L0 support table**이다.
Query CNN은 update 후 재계산했지만 전체 support refresh는 하지 않았다.
저장된 step16431의 Adam moments를 이어간 exact resume나 실제 다음 update 비교가 아니다.
Prefix seconds7.464416/3.967674는 loader/cache와 실행 순서를 포함하므로 모델 속도
약2배 개선으로 환산하지 않는다. 전체 epoch·CP 추천 효용·segmentation Dice는 미평가다.

## 현재 판단과 다음 수정 경계

**내적항 후보를 production 해결책으로 채택할 근거는 이번 결과에서 얻지 못했다.**
기존 adapter·100개 검사·로컬 CT DEBUG 결과와 후보 코드는 보존한다. 검사가 통과한
구현과 validation 순위가 개선된 모델을 구분한다. Beta를 자동 증대하거나 FFN을
없애거나 loss를 함께 바꾸지 않는다. 장기 재학습을 자동 시작하지 않는다.

다음으로 분리할 대상은 기존에 수축이 측정된 **L0 fusion과 두 번째 L1 query update의
FFN·residual·정규화 구간**이다.
이번 출력에는 fusion 내부 trace가 없어 epoch30에서 fusion 수축률을 새로 측정했다고
주장하지 않는다. Epoch22/27의 수축 근거는 이전 문서에 그대로 보존한다.
이미 구현된 `tools/local_cnn_fusion_probe.py`는 recipient 경로를 직접 보존하는
`r+lambda·Fuse(d,r,r-d,r*d)`를 원래 Fuse와 비교하며 **query와 전체 적격 support를
같은 snapshot CNN으로 함께 재인코딩**, 각 branch의 L1/L2 plan을 새로 만든다.
Query만 바꾸고 saved fused support를 그대로 쓰면 비교가 성립하지 않는다.

이 검사는 후보 차이가 약해지는 앞단을 분리하는 fixed-weight counterfactual이다.
성능이 개선될 것이라는 보장, 새로운 production 기본값, saved checkpoint의 exact
resume 패치가 아니다. 전체 support를 다시 읽는 비용이 있어 몇 초짜리 검사라고
보고하지 않는다. 이번 결과 수신 작업에서는 이 검사를 새로 실행하지 않았다.
L1 FFN0의 이전 ordering 변화도 일관되지 않았으므로 FFN 제거를 자동 적용하지 않는다.

현재 summary의 `SHADOW unavailable`과 CNN deep trace unavailable은
`--interaction-only`로 해당 검사를 요청하지 않은 결과다. Epoch27의
refresh phase로 인한 SHADOW NOT_RUN 이유를 epoch30에 가져오지 않는다.
Cloned-update 원본/RNG 보존 true와 top-level saved-payload/support-plan/RNG
unavailable도 서로 다른 출력 필드이며 누락을 보존 실패로 바꾸지 않는다.

## 이번 작업의 완료 범위

서버 콘솔 전체 수신·코드 대조·원문 byte 보존·숫자 전사·파생 비율 검산 완료.
새 production 코드 변경 없음. 기존100개 단위/CUDA 검사의 receipt는 이전 실행
그대로 보존했으며 이번 문서 작업에서 다시 실행했다고 표시하지 않는다.
실제 CT 서버 진단 결과를 읽었지만 서버 가중치·원본 JSON을 로컬에서 독립 재현한 것은 아니다.
새 장기 학습 및 전체 평가 실행 없음.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 서버 진단 batch32와 기존 검사 계약을 대조했다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 GPU/한도/peak 출력은 확인했으나 CPU 실사용·실제 RSS는 이번 첨부에 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 진단에는 OOM이 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 회귀 검사와 이번 CNN gradient/delta 출력의 범위를 구분했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
