# v2.2 국소 CNN — 독립 검토 반영과 세부 진단

## 검토 내용과 적용 범위

2026-10-01 사용자가 전달한 GPT 독립 검토를 전체 읽고 실제 구현과 대조했다.
검토는 현재 donor 고정/P×U tiling/CE 반복 보정/best 선택에서 산술 오류를
찾지 못했으며, 서버 측정에서 L0의 작은 후보 방향 분산과 L1의 추가 수축을
지적했다. 보조 loss의 큰 gradient norm을 곧바로 실제 Adam update 지배나
loss 충돌의 증명으로 취급하지 않도록 요구했다.

현재 반영은 **진단 코드의 확장**이다. 생산 모델, CNN readout, L1/L2,
loss weight, split, seed42, CP80%, 후보128, 전체 관측, 원본 paste mask,
physical batch 및 학습 일정은 변경하지 않았다. 변경한 실행 파일은
`tools/diagnose_local_cnn_learning.py`와 세 개의 진단 전용 helper다.

## 기존 진단의 표현 수정

- 기존 `gradient_probe`는 eval-mode에서 현재 모델로 plan을 다시 fit한
  국소 derivative다. 새 필드에 cluster plan scope, dropout, clipping/Adam
  미실행을 명시한다. 이것을 저장된 production update의 재현이라고 부르지 않는다.
- 콘솔 `episode loss`와 `full loss`는 각각 `episode_pairwise_loss`,
  `full_pairwise_loss`로 바꿨다. rank+관측 CE+alignment total과 구분한다.
- 기존 report에만 있던 eval gradient cosine과 memory drift도 콘솔에 출력한다.
- snapshot의 epoch/step/phase, 모델 구성·parameter 수·batch·precision,
  데이터 개수와 명시적 진단 subset, CPU/GPU/RAM 예산을 기록한다.

## L0 세부 측정

`tools/local_cnn_l0_probe.py`는 실제 production CNN/mean/project/fuse 모듈을
동일한 수식으로 한 번 실행한다. scale별 국소 crop 안 간 영역 전체의 mean, concat, projection,
donor/recipient 차이·곱, fusion 입력·출력의 후보 벡터를 기록한다.
scale별 map 내부 공간 분산은 **후보 간 분산과 별개의 통계**로 기록한다.

같은 map의 anchor 주변 sphere도 명시한 물리 반경으로 평균한다. native crop
origin/shape/spacing과 실제 anchor, stride1/2/4의 feature-cell center 및 organ
mask를 사용한다. 이 반경은 **진단용 readout 범위**이며 CNN의 receptive field나
production margin을 바꾸지 않는다. 반경은 CLI 필수 설정이고 기본값이 없다.
3mm를 쓰는 예시는 작은 anchor 주변과 crop 전체 평균의 차이를 살피는 진단
조건일 뿐 최적 반경 주장이나 production 설정이 아니다.

각 feature-cell에는 원래 CNN receptive field가 남아 있다. 이 검사는 새롭게
3mm CT만 CNN에 넣는 실험이 아니다. coarse stride/spacing 또는 background
bbox anchor 때문에 어떤 후보 sphere에 organ cell center가 없으면 해당 stage를
전체 case에 대해 **NOT_RUN**으로 남긴다. 유효 후보만 골라 분산을 보고하거나
0벡터로 대체하지 않는다. 원래 후보·마스크·anchor를 이동/삭제하지 않는다.

## L1 세부 측정과 counterfactual

`tools/local_cnn_l1_probe.py`는 동일 prepared support state에서 query 경로의
input, raw aggregate, projected message, scaled message, residual add,
첫 LayerNorm, FFN output, 두 번째 add, 마지막 LayerNorm을 분해한다.
raw/normalized 후보 분산, 공통 성분과 메시지/input norm 비를 기록한다.

query projected message에만 명시한 scale을 곱한다. 예시 scale0/.25/.5/1은
새 학습 설정이 아니라 현재 가중치의 inference-only 대조다. support histories,
L2, attention edge, layer 수, FFN/norm은 고정된다. scale0도 FFN과 norm을
실행하므로 L1 전체를 삭제하는 검사가 아니다. scale1과 실제 production
predict_embeddings의 logits parity를 실제 호출마다 강제한다.

첫 query L1이 읽는 history가 trained shared label seed 반복인지도 확인한다.
이는 synchronous message passing의 실제 경로 설명이며, shared seed 자체를
구현 오류라고 단정하지 않는다. 분산/score gap/pair win의 회복은 제한된
counterfactual 증거이며 학습 후 정확도 향상 보장은 아니다.

## 저장된 다음 update의 Adam 복제 대조

`tools/local_cnn_shadow_probe.py`는 original checkpoint payload와 RNG,
support/plan을 보존하고 별도 모델에서 진단한다.

- saved epoch/seed/physical batch의 **실제 다음 tile**을 선택한다. 보기 좋은
  positive tile로 대체하지 않는다. 다음 tile이 P×U가 없으면 rank 방향은
  정의되지 않을 수 있으며 그 사실을 기록한다.
- 같은 환자 episode가 이어지면 저장된 frozen plan을 사용한다. 새 episode면
  production과 같이 현재 snapshot eval teacher로 fit한 plan을 고정한다.
- train-mode FP32/dropout과 저장된 RNG로 한 forward를 공유한다. rank,
  observation CE, alignment의 실제 전역 계수로 gradient를 계산하며 그 합과
  full backward의 parity를 확인한다.
- rank-only/CE-only/alignment-only/full 네 branch에서 매번 동일 saved model과
  **동일 AdamW moments**를 복원한다. 원래 clip 값을 적용한 뒤 실제 optimizer
  step을 복제 모델에서 실행한다. 학습된 parameter delta, gradient cosine,
  rank 반대 방향 projection, clip factor를 기록한다.
- unused gradient는 None으로 유지한다. alignment-only에서 연결되지 않은 CNN에
  0 gradient를 만들어 weight decay/Adam moment 갱신을 유발하지 않는다.
- fused Adam은 parameter·gradient·moment의 실제 layout까지 맞춰야 한다.
  `autograd.grad`의 반환 layout을 실제 `.grad` accumulation과 혼동하지 않는다.
- 종료 시 caller RNG와 payload/support/plan hash 보존을 검사한다. 저장된
  optimization schedule가 끝났거나 다른 phase면 shadow는 **NOT_RUN**이다.
  임의 tile 또는 새 optimizer로 대체하지 않는다.

`opposing_projection > 1`은 auxiliary raw gradient의 rank 반대 방향 성분이
rank 자체보다 크다는 의미다. Adam moments/epsilon/decay가 반영된 실제 delta와
함께 해석해야 한다. 이 한 tile 결과를 모든 update의 성질로 일반화하지 않는다.

production optimizer update 수는 0이다. shadow에는 복제 optimizer step4회가
있으므로 보고서에 `cloned_optimizer_steps=4`로 구분한다. 진단 산출물은 새 JSON
보고서뿐이며 checkpoint/ready/training-complete 파일을 만들지 않는다.

## 사용 조건

기존 진단 호출에 `--deep --anchor-radius-mm 3 --message-scales 0 0.25 0.5 1`을
명시하면 확장 진단이 켜진다. 기존 호출은 종전 eval 진단 의미를 유지한다.
실험은 제공된 epoch0~17 이력으로 자동 식별하고 해당 실험 ledger의 checkpoint를
읽는다. 테스트 등 명시 snapshot이 필요할 때만 `--checkpoint`를 사용할 수 있으며
matched experiment 안의 파일인지와 content/source/inventory 결속을 검사한다.

Singularity의 GPU 번호는 컨테이너의 `nvidia-smi` namespace다. 사용자 직전 실행에서
해당 A6000은 컨테이너0으로 선택됐다. host 물리 번호와 같다고 주장하지 않는다.

## 검증 기록

- 단위/연산 검사 **30개 모두 PASS**, CUDA 검사 skip 없이 완료했다. L0 output
  parity, L1 scale1 parity, full gradient decomposition, 저장 Adam moments와
  production full update 대조, fused Adam layout, RNG/payload 보존을 포함한다.
- 실제 CT DEBUG 통합: 보존된 optimization step1 checkpoint의 다음 tile(physical2)을
  복제하여 4branch optimizer 대조 후 train `liver_66`/val `liver_31`을 각
  2후보 전부 검사했다. 가중치/payload/support/plan/RNG 보존 검사 PASS.
  CNN/L1/L2를 포함한 원래 모델 1,125,718 parameter, FP32를 유지했다.
- 로컬 RTX5070Ti16GiB, PyTorch2.8.0+cu128, workers4, CUDA예산12GiB,
  RSS32GiB/resident8GiB. 실제 peak CUDA0.2354GiB, RSS4.742GiB.
  입력은 명시된 기존 DEBUG cohort이며 final dataset이 아니다.
- 결과: `work/local_cnn_deep_diagnosis_DEBUG_20261001/report.json`.
  이 시점에는 서버 환경(Torch2.6.0+cu118)과 학습된 m10 weight의 세부 진단을 아직 실행하지 않았다.

이는 진단 경로 검증이며 서버 원인 판정이나 개선 정확도 측정이 아니다.
별도30mm/MIG calibration OOM은 이번 작업에서 해결하지 않았다.

## 서버 실행 완료와 콘솔 출력 보존 (2026-10-01 후속)

사용자가 제공한 콘솔 끝부분에 다음 파일의 저장과 정상 종료가 확인됐다.
`/home/aicompetition06/Medical/experiments/deep_diagnosis_20261001_101631.json`.
서버 진단은 완료됐으며, 긴 상세 JSON을 콘솔에도 출력해 터미널 앞부분이
밀려난 상황이다. Singularity 자체가 JSON 파일을 잘랐다는 증거는 없다.
로컬에는 사용자 첨부의 콘솔 끝부분만 있고 **원본 서버 JSON은 아직 없다**.
이 첨부에 없는 train/shadow 결과를 재구성하거나 완료 여부를 추정하지 않는다.

`tools/summarize_local_cnn_diagnosis.py <report.json>`은 저장된 보고서를
표준 라이브러리만으로 읽고 짧은 요약을 출력한다. GPU 초기화·CNN forward·
loss/backward·학습·재진단 없이 기존 결과를 사용한다. 선택 `--output`은
UTF-8 text 신규 파일만 기록하고 기존 파일과 원본 JSON을 덮어쓰지 않는다.
미기록 값은 unavailable, 실행되지 않은 probe는 NOT_RUN으로 표시한다.

향후 `diagnose_local_cnn_learning.py` 콘솔은 진행 막대와 짧은 요약을 기본으로
출력한다. 전체 상세는 종전과 같이 JSON에 보존하며, 필요할 때만
`--verbose-console`을 명시해 긴 출력으로 전환한다. 진단 수식·설정·범위·
실행 횟수와 production source/checkpoint 계약은 바꾸지 않는다.

남은 `liver_3` 수치에서는 fusion 입력/출력의 정규화 후보 분산이
2.42105e-4 → 5.59089e-6, L1 최종은 7.25596e-10이다. message scale0에서도
FFN 뒤 차이가 줄며, pair win은 scale0과 scale1 모두 0.0390625다.
scale0의 pairwise loss는 0.6932707로 scale1의 0.6931957보다 크다.
점수 분산 증가만을 순위 개선이나 주원인 해결로 해석하지 않는다.
원본 JSON의 train·gradient·복제 Adam 결과까지 확인해야 해석 범위가 넓어진다.

후속 검사: 기존 단위/GPU30개와 요약기8개, 총38개 PASS. 실제 CT DEBUG
두 사례의 새 기본 콘솔·전체 JSON 저장을 GPU에서 다시 확인했고,
저장 JSON을 읽어 신규 summary.txt를 쓰는 경로도 통과했다. torch import를
차단한 검사에서 요약기는 정상 작동했다. 원본 JSON·가중치·payload·
support/plan·RNG 보존과 production update0을 확인했다. 새 요약 출력은
loss 차이를 7 유효숫자로 보존하고 step/개수를 반올림하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 진단 사례 수를 명시한다.
- [x] 저장된 physical batch 및 병렬 loader를 유지했다.
- [x] 로컬 RTX5070Ti/VRAM과 CPU/RAM 및 명시적 검사 예산을 확인했다.
- [x] 복제 gradient/Adam의 layout 오류를 production 축소로 우회하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 unit 입력을 표시한다.
- [x] production forward/loss/gradient/Adam 연결과 진단 parity를 검사했다.
- [x] 실제 실행 설정과 미검증 범위를 보고했다.
- [x] smoke/단위 검사와 서버 모델 판정·전체 학습·전체 평가를 구분했다.
