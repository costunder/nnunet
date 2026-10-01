# v2.2 — 공식 기본 L1 대조 구현과 실제 CT 검사

현재 v2.2 LocalCNN의 L1과 공식 PRODIGY의 기본 `MetaGNNLayer`는 다른 연산이다.
이를 확인한 뒤, 같은 L0/L2 입력과 loss를 사용하는 별도 복제 모델을 구현했다.
기존 가중치로 곧바로 같은 학습을 재개하는 패치가 아니다. 새 L1이 순위 학습을
개선하는지는 학습된 서버 snapshot의 짧은 대조 결과로 판단한다.

## 구현과 보존 범위

기준 원문은 [PRODIGY commit 107ba572](https://github.com/snap-stanford/prodigy/blob/107ba57234d3188227cda5b78a2dbcfb84a1c694/models/metaGNN.py)이다.
고정 원문의 SHA256를 검사하고, 독립 검사에서 해당 클래스 AST만 추출해 원형
출력과 gradient를 대조한다. 공식 저장소 전체나 다른 optional 모델을 그대로
재현했다고 표시하지 않는다.

```text
동일한 donor/recipient native CT
  → 기존 공유 국소 CNN [12,24,32], convolutions [2,3,3]
  → 기존 multi-scale mean / projection / pair fusion, 128D
  → reference L1: support + patient labels + query, joint 2층 / 4heads
  → 기존 L2 2층, support-only cluster teacher / prototype scoring
  → 기존 same-donor P×U ranking + balanced CE + alignment
```

L1의 변경은 다음과 같다.

| 항목 | 기존 | 공식 기본 연산 대조 후보 |
|---|---|---|
| Q/K/V·edge projection | bias 없음 | bias 포함, key는 head width 제곱근으로 나눔 |
| attention 입력 | q,k,SiLU(edge), LeakyReLU MLP | k,q,ReLU(edge), ReLU MLP |
| value message | value + edge | value만, attention dropout |
| 자기 노드 | 외부 residual | attention self-loop + residual |
| output projection | 집계 뒤 | 원문처럼 각 edge에서 적용한 뒤 집계 |
| update | 두 LayerNorm + FFN residual | residual + joint BatchNorm, 층 사이 GELU |
| 관계 코드 | 프로젝트 T/F/U | 원형 T=[0,1], F=[0,-1], U=[1,0]; self=[0,0] |

Support data↔자기 환자의 label 두 개를 연결한다. Query는 모든 support label에서
메시지를 받으며 query→support/label 역방향 edge는 없다. Query 정답은 inference
API와 graph 입력에 들어가지 않는다. Train BatchNorm 통계에는 query **특징**도
포함된다. 따라서 support/query를 나누어 forward하지 않는다. Eval은 고정 running
statistics를 사용하고 query128을 physical chunk32로 나누어도 동치다.

새 후보는 기존 Q/K/V·edge·attention·output projection 텐서를 명시적으로 전이한다.
새 projection bias는 0, 새 BatchNorm은 affine 1/0, running mean/variance 0/1이다.
기존 L1 FFN/LayerNorm 텐서는 새 연산에 넣지 않는다. L0/L2/label seed는 hash로
같음을 확인한다. 전체 parameter는 1,125,718→862,806이다. 이 차이는 공식 기본
L1의 연산 구성을 따른 결과이며, dim128·heads4·L1/L2 각2층은 유지한다.

새 BatchNorm은 학습되지 않은 초기 상태다. 전이된 attention에서도 key scaling,
activation, value 및 self-loop가 바뀌므로 이전 모델과 수치적으로 같은 warmstart가
아니다. Adam moments를 전이하지 않으며 양쪽 모두 fresh AdamW를 사용한다.
같은 seed는 서로 다른 dropout graph의 난수별 동등성을 뜻하지 않는다.

Basic CP80%, split/seed42, 후보128, 전체 관측, 간 외부 입력 차단, 원본 paste mask,
L0, L2, loss 계수, support 환자 선택 및 physical batch를 production에서 변경하지
않았다. L2 alignment의 tile 반복 가중도 이번 대조에서는 수정하지 않는다.

## 실제 실행 경로

- `tools/local_cnn_reference_l1.py`: independent reference operator와 명시적 전이.
- `tools/local_cnn_reference_runtime.py`: 두 복제 모델, 같은 실제 CT tile/support,
  전체 epoch의 P×U/CE 계수와 명시적 짧은 prefix, 전후 선택 case 평가.
- `tools/diagnose_local_cnn_reference.py`: 기존 실험 manifest에서 최신 checkpoint를
  한 번 읽고 content/source/input 결속을 검증한다. 기존 checkpoint나 결과를 쓰지
  않으며 새 JSON에만 기록한다. 전체 JSON을 콘솔에 쏟지 않는다.

각 branch는 자신의 support-only teacher를 새로 fit한다. 기존 L1의 cluster plan을
새 후보에 재사용하면 명시적으로 실패한다. 학습은 원래 기록된 support episode,
평가는 전체 eligible saved epoch support를 사용한다. Saved support L0는 같은 epoch
기준으로 고정되며 live query CNN·L1·L2는 모두 학습한다. 전체 support를 새 CNN으로
refresh한 후의 최종 평가라고 표현하지 않는다.

평가 case는 model score와 무관하게 P/U가 있는 case의 정렬된 고정 spread로
선택하며 각 case의 모든 후보를 유지한다. 양성이 없는 case도 전체 학습 schedule에는
남는다. `--cases-per-split`과 `--steps`는 필수 DEBUG 범위이며 최종 학습 설정에
덮어쓰지 않는다. Epoch source·memory·원본 model/mode/gradient/RNG 불변을 검사한다.

## 검증 결과와 한계

공식 클래스 AST를 독립 oracle로 사용한 CPU/CUDA 연산 동치, train dropout 및 joint
BN 통계, topology·T/F/U·self-loop, query128/chunk32, 모든 L1 gradient, 실제 LocalCNN
extra-state 보존, teacher mode 복원, 원본 state/RNG 보존을 포함해 **18개 단위 검사
PASS**, skip0이다. 이는 CT 정확도 검사와 구분한다.

실제 CT DEBUG 실험의 완전한 짧은 schedule에서 branch별 2update를 실행했다.
원래 DEBUG physical batch2, train8/val2 관측, case별 P1/U1, 같은 native 입력과 saved
support를 사용했다. CNN/L1/L2 모두 유한한 gradient와 실제 parameter 변경이 확인됐다.
전체 모델과 기존 결과는 보존됐다. 후보128·physical32 검사를 대신한 것으로
표현하지 않는다.

추가 실제 CT 검사에서는 liver_66의 **원래 양성5 + 미관측128 = 133개**를 모두
유지하고 P5/U27의 **physical32** tile을 두 복제 모델에 각각 1update했다.
CNN/fusion/L1/L2 모든 trainable parameter에 유한 gradient가 있었고 각 모듈의
parameter 변경이 nonzero였다. 원본 checkpoint/model/support/RNG hash는 일치했다.
Support는 검증된 **DEBUG 6관측/3환자**이며 loss 정규화는 완전한 이 one-case의
640개 P×U 조합 기준이다. Production 전체 support/epoch 계수 검사의 대체물이 아니다.

| physical32 실제 CT smoke | 기존 L1 | 공식 기본 L1 대조 |
|---|---:|---:|
| update 계산 시간, 첫1회 | .333초 | .264초 |
| update peak CUDA allocated | .442GiB | .473GiB |
| 133개 train 관측 pair-win, before→after | .6516→.8016 | .4922→.5625 |
| 같은 case MRR, before→after | .0714→.1667 | .3333→.3333 |

이는 후보가 많은 실제 native 입력의 **기계적 smoke**다. 작은 saved DEBUG weights,
하나의 train case, fresh BN과 첫 update이므로 일반화·성능 향상이나 epoch 속도
검증으로 제출하지 않는다. 해당 native unique crop tensor는 `[33,1,58,50,7]`이고
더 큰 종양/높은 해상도/30mm 또는 전체 support의 최대 메모리 검증이 아니다.
GPU RTX5070Ti16GiB, FP32, CPU workers4, CUDA budget12/RSS32/resident8GiB다.

재현 스크립트는 `tools/verify_reference_l1_ct_debug.py`다. 로컬에 보존된 DEBUG
inventory와 실제 full assignment를 명시적으로 읽으며 없는 데이터에 dummy로
대체하지 않는다. 서버 comparison CLI와 달리 this smoke의 저장 support와 loss
normalization은 DEBUG임을 별도로 기록한다.

검증 원본은 `validation/reference_l1_20261001/`에 보존했다.
`unit_report.json`(18개, skip0), `native_ct_report.json`(실험 실행 CLI),
`physical32_ct_report.json`(후보128/physical32 실제 CT), `manifest.json`(각 hash 및
production source 보존 증거)을 함께 읽는다.

이 작은 DEBUG에서 reference validation MRR은 .5→.5, pair win은 0→0이었다.
**성능 개선 근거가 아니다.** 학습 완료된 서버 가중치가 로컬에 없으므로 epoch30의
모델이 이 연산에서 개선되는지는 미검증이다. 단위 통과나 후보 방향 분산 증가만으로
ranking 개선을 판정하지 않는다.

전체 update 시간은 synchronized forward/loss/backward/gradient/clip/AdamW를
측정한다. CT loading·전송·input hash·teacher fit·평가·checkpoint 저장은 제외한다.
양쪽의 raw 시간은 저장하지만 warmup 없는 몇 update의 차이를 epoch 가속으로
환산하지 않는다. 전체 diagnostic wall time과 peak CUDA/RSS는 별도 기록한다.

## 서버의 짧은 대조 명령

현재 Singularity에서 A6000 하나가 GPU0으로 보이는 세션 기준이다. 호스트에서 직접
실행한다면 `CP_GPU`를 실제 visible physical 번호로 바꾼다. UUID를 넣지 않는다.
학습 중인 checkout과 별도의 기존 진단 checkout을 사용한다.

```bash
CP_GPU=0
cd /home/aicompetition06/Medical/HierCP-diagnosis-2ce11ba &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach FETCH_HEAD &&
python -u tools/diagnose_local_cnn_reference.py \
  --gpu "$CP_GPU" \
  --run /home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42 \
  --output "/home/aicompetition06/Medical/experiments/reference_l1_$(date +%Y%m%d_%H%M%S).json" \
  --cases-per-split 2 --steps 4 --workers 8 \
  --cuda-gib 24 --rss-gib 64 --resident-gib 24
```

`--run`에 결속된 최신 checkpoint를 선택하며 원래 batch32·margin10을 상속한다.
다른 experiment의 checkpoint를 섞지 않는다. 실행은 명시적 짧은 복제 비교만 하며
장기 GNN/nnU-Net 학습이나 production ready 표시를 자동 시작하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 공식 대조의 다른 연산은 명시했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. DEBUG와 전체 실험을 구분했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 모델 축소보다 실제 상태 hash·입력·연산 및 실행 오류를 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
