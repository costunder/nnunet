# v2.2 reference L1: 전이와 BatchNorm 교란을 분리하는 고정 가중치 진단

기존 `6a02e9b`의 공식 기본 operator 대조 구현·18개 검사·실제 CT DEBUG 결과는 보존한다.
이번 변경은 production L1을 바꾸는 패치가 아니다. 제공된 독립 검토의 전이 문제를
별도 진단으로 드러내고, optimizer update 이전의 동일 입력 비교를 추가한다.

## 확인한 문제와 전이 정책

기존 프로젝트 relation은 T=[1,1], F=[1,0], U=[0,0]이다. 공식 relation은
T=[0,1], F=[0,-1], U=[1,0], self=[0,0]이다. 두 좌표계에서 같은 W를 그대로
복사하면 projected 관계 의미가 달라진다. 이전 raw 복제 후보를 삭제하거나
기존 결과를 다시 쓴 것으로 처리하지 않는다.

원래 W의 column을 a,b라 하면 새 W의 column을 `-a-b/2`, `b/2`, 새 bias를
`a+b/2`로 둔다. T/F/U의 **선형 투영 전이**는 보존된다. 새 self는 bias를
받으며 원래 graph에 대응 self가 없었다. SiLU/ReLU, key scaling, self attention,
value-only message, BatchNorm 등은 여전히 달라서 전체 함수 보존이 아니다.

진단에서는 아래 세 값을 반드시 명시해서 선택한다. 숨겨진 권장 기본값은 없다.

| 전이 정책 | 관계 투영 | 공식 edge별 output bias |
| --- | --- | --- |
| raw_columns | 이전 결과와 같은 column 복사; 의미 변화 기록 | 기존 node bias 복사 |
| affine_relations | T/F/U 선형 투영 보존 | 기존 node bias 복사 |
| affine_relations_zero_out_bias | T/F/U 선형 투영 보존 | 명시적으로 0에서 시작 |

원래 output bias는 aggregate 뒤 node당 한 번 적용됐다. 공식 out_proj는
edge마다 적용되므로 indegree×bias가 된다. Query의 indegree는 support 환자가
16명이면 2×16+1=33이다. Zero 정책은 이 증폭을 제거하지만 원래 node당 bias도
제거하므로 기능 보존 전이가 아니다. 공식 forward 안에서 bias를 degree로
나누는 수정은 하지 않는다. 실제 support/data/label/query degree와 bias norm을
각 진단에 기록한다.

복사한 attention key weight는 공식 `key/sqrt(head_dim)`을 보상하도록 rescale하지
않았다. 128D/4heads에서 sqrt(head_dim)=sqrt(32)다. 이번 세 정책은 이 차이를
해소하지 않는다. 결과는 단일 요인 인과 검증이 아니다.

## Update 이전의 같은 tile 비교

`diagnose_local_cnn_reference.py --fixed-only --transfer-policies ...`는 optimizer를
만들지 않는다. 같은 checkpoint에서 기존 L0로 native CT tile을 한 번 인코딩하고,
legacy 및 명시한 세 reference 복제 후보에 그 같은 표현과 support를 제공한다.
Optimization checkpoint라면 저장된 `next_batch`에 해당하는 정확한 tile을 선택한다.
다른 phase이면 전체 deterministic schedule의 첫 tile을 사용했다고 명시하고,
saved-next-update 또는 exact resume라고 표시하지 않는다. 양성이 없는 tile을
더 좋은 tile로 교체하지 않으며 P/U ranking이 불가능하면 NOT_EVALUABLE로 기록한다.

각 후보는 자기 L1으로 support-only eval teacher plan을 새로 만든다. Reference끼리
또는 legacy에서 만든 teacher를 다른 후보에 재사용하지 않는다. 이후 세 mode는
그 후보의 같은 plan을 유지한다.

1. `eval_fresh`: 전이 직후의 fresh BN running statistics를 사용한다.
2. `train_joint_batch_statistics_dropout_disabled`: 같은 query/support graph를 한 번
   실행해 joint BN의 batch statistics를 확인한다. 복제 모델의 dropout만 진단용으로
   0으로 두며 production 설정은 변경하지 않는다.
3. `eval_after_one_same_tile_bn_pass`: 위 한 번의 pass 후 running statistics로 eval한다.
   이것을 BN calibration 완료나 성능 검증이라고 부르지 않는다.

BN의 momentum·running mean/var·num_batches_tracked, L1 각 단계 query의
aggregate/pre-BN/operator output/GELU 또는 기존 LN/FFN 단계별 raw/normalized
centered energy를 기록한다. 후보 차이가 커져도 정확도 향상이나 특정 FFN의
원인 확정으로 해석하지 않는다. 선택 case의 모든 후보 순위 평가와 exact tile의
P/U 차이를 구분한다.

## Loss 공식 보존과 gradient 경로는 다른 주장이다

L0/L2 초기 가중치·구성, 기존 ranking/CE/alignment 공식 및 full-cohort coefficient는
보존한다. 그러나 joint train BN은 query features에 따라 support 표현을 바꾼다.
따라서 support alignment loss가 query/L0 출력까지 역전파될 수 있다. 기존 detached
support 경로와 gradient dependency가 같다고 주장할 수 없다. Query target은
graph/BN의 입력이 아니므로 이 현상 자체가 정답 누수는 아니다.

별도 복제 상태에서 원래 coefficient를 적용한 ranking/CE/alignment 각각의 query
128D gradient norm과 None 여부를 기록한다. 이 derivative 검사에는 optimizer가
없고 CNN weight를 갱신하지 않는다. CNN까지의 실제 gradient/optimizer 연결은
별도의 실제 CT smoke로 확인한다. 기존 alignment의 tile multiplicity weighting은
이번 패치에서 바꾸지 않는다.

## 검증 범위와 산출물

새 기록은 `validation/reference_transfer_20261001/`에 저장한다. 이전
`validation/reference_l1_20261001/` 원본 및 hash는 그대로 유지한다.
단위 fixture는 공식 투영·bias·BN·gradient의 참고 검사이며 실제 학습 성능이 아니다.
실제 CT DEBUG에서도 full-case 128 comparison center와 모든 관측 anchor를 유지하되,
DEBUG support와 one-case loss 정규화를 production 전체 학습으로 취급하지 않는다.
학습된 서버 epoch30 가중치의 결과는 로컬에 없으며 여전히 미검증이다.

이번 실행은 기존 연산/선택 회귀18개와 신규 전이10개·고정 비교6개, 총34개가
PASS했고 skip0이다. 공식 AST와의 CPU/CUDA output·gradient 대조를 포함한다.
실제 CT physical32 smoke는 liver_66의 P5/U128 전체133 관측을 전후 평가했다.
저장된 DEBUG support6/환자3, native tensor `[33,1,58,50,7]`, FP32,
RTX5070Ti16GiB, workers4, CUDA3/RSS32/resident8GiB의 명시적 자원 한도를 썼다.
새 affine+bias0 branch의 CNN/readout/L1/L2 gradient와 optimizer delta는 모두
유한하고 0보다 컸다. Reference update peak allocation은 507,315,712bytes다.
다른 GOIS 프로젝트가 같은 GPU에서 학습 중이었으므로 속도나 epoch 개선 근거로
사용하지 않는다.

아래 수치는 **학습 완료 서버 모델이 아닌 동일 DEBUG 가중치의 고정 tile**이다.
L0 normalized centered energy는 모든 branch에서 0.01452065로 같다.

| 초기 eval 정책 | L1 2층 후 normalized centered energy | 해당 tile pair-win |
| --- | ---: | ---: |
| legacy | 0.01436463 | 0.61481 |
| raw_columns | 0.00270644 | 0.51111 |
| affine_relations | 0.00271042 | 0.51111 |
| affine_relations_zero_out_bias | 0.01311830 | 0.50370 |

이 tile에서 bias0는 후보 방향 분산 수축을 줄이지만 pair-win 개선은 보여주지
않는다. Joint train BN에서는 분산이 더 커져도 pair-win은 더 낮았다. 따라서
분산 회복을 정확도 회복과 같게 취급하지 않는다. Alignment의 query gradient는
legacy에서 None, raw/affine/bias0에서 각각 norm 0.054898/0.054776/0.085270이었다.
이 결과는 BN의 gradient routing 차이를 확인한 것이며 성능 우열 판정이 아니다.

서버 명령과 같은 CLI의 실제 CT DEBUG end-to-end도 통과했다. 그 DEBUG는 원래
batch2/전체 DEBUG train8·val2를 상속했고 snapshot phase가 complete라
saved-next tile=False와 첫 deterministic tile 사용을 기록했다. Production
source121개는 `6a02e9b`와 동일하며 이전 18검사/CT 보고서 hash도 그대로다.

## 서버의 이전 raw 전이 4-update 결과 수신

사용자가 제공한 `reference_l1_20261001_173620.json` 실행 콘솔은
`6a02e9b`에서 raw column 전이로 완료한 이전 명령의 결과다. 사용자가 명령을
잘못 실행한 것이 아니며, 이 결과를 새 affine/BN 고정 비교로 분류하지 않는다.
물리 GPU3의 RTX A6000 선택은 실제로 성공했다. Checkpoint의 epoch/step은
콘솔에 없어 추정하지 않는다. 원본 서버 JSON은 로컬에 수신하지 않았다.

| 복제 branch / 평가 | MRR 전 → 후 | 모든 P×U pair-win 전 → 후 |
| --- | --- | --- |
| legacy / train | 0.12179 → 0.08333 | 0.49572 → 0.49320 |
| legacy / validation | 0.54167 → 0.51562 | 0.64397 → 0.68052 |
| reference / train | 0.10965 → 0.16071 | 0.49521 → 0.42742 |
| reference / validation | 0.50510 → 0.25394 | 0.45229 → 0.48661 |

이 4회 갱신은 reference 개선 근거가 아니다. Validation MRR은 약 절반으로
떨어졌고 pair-win은 증가해도 0.5 미만이다. MRR은 case별 첫 관측 종양의
reciprocal rank 평균이며 pair-win은 모든 관측/미관측 쌍의 score 대소 비교라
둘은 서로 다른 양이다. Console만으로 실제 어떤 종양 순위가 이동했는지는
알 수 없다. Fresh BN/Adam과 전이/연산 차이가 함께 들어갔으므로 공식 PRODIGY
실패 또는 특정 원인 확정으로도 해석하지 않는다. 같은 결과의 기존 모델과
reference는 초기 점수 경로가 다르므로 초기 성능도 같지 않다.

서버 콘솔의 반올림 값과 미수신 범위는
`validation/reference_transfer_20261001/server_raw4_console_summary.json`에
기록했다. 전체 데이터 학습을 다시 실행하거나 이 결과 때문에 production
모델을 자동 교체하지 않는다. 다음 전달 명령은 기존 snapshot을 읽어 update
이전 차이를 세 전이·BN mode로 분리하는 진단이다.

## 서버 실행

가장 최근 사용자가 제공한 host 터미널에서는 물리 GPU3 선택이 성공했으므로
`CP_GPU=3`을 유지한다. A6000 하나만 GPU0으로 노출된 Singularity 안에서 실행할
때만 0으로 바꾼다. 실행 중인 학습 checkout과 별도의 기존 diagnosis checkout에서
실행한다. 실제 진단 구현은 검증한 `de7d62d` commit을 고정한다.

```bash
CP_GPU=3
cd /home/aicompetition06/Medical/HierCP-diagnosis-2ce11ba &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach de7d62d &&
python -u tools/diagnose_local_cnn_reference.py \
  --gpu "$CP_GPU" \
  --run /home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42 \
  --output "/home/aicompetition06/Medical/experiments/reference_fixed_$(date +%Y%m%d_%H%M%S).json" \
  --fixed-only \
  --transfer-policies raw_columns affine_relations affine_relations_zero_out_bias \
  --cases-per-split 2 --workers 8 \
  --cuda-gib 24 --rss-gib 64 --resident-gib 24
```

기존 raw 전이의 fresh AdamW prefix 비교는 여전히 명시적인 `--steps`로 실행할 수
있다. 이번 우선 경로는 **zero update** 비교이며 장기 GNN/nnU-Net 학습,
production checkpoint 또는 ready 표시를 생성하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 공식 대조의 차이를 명시했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다.
- [x] 핵심 모듈의 forward/loss/gradient/optimizer 연결을 실제 CT smoke와 구분했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
