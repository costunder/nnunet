# v2.2 reference L1: 전이와 BatchNorm 교란을 분리하는 고정 가중치 진단

기존 `6a02e9b`의 공식 기본 operator 대조 구현·18개 검사·실제 CT DEBUG 결과는 보존한다.
이번 변경은 production L1을 바꾸는 패치가 아니다. 제공된 독립 검토의 전이 문제를
별도 진단으로 드러내고, optimizer update 이전의 동일 입력 비교를 추가한다.

후속 `99c17e59` 검토에 따라 실제 update case의 전체 후보 timeline, 동일 forward의
loss/Adam 방향, reference BN/dropout 분리를 opt-in으로 구현하고 실제 CT GPU에서
검증했다. [진단·검사·해석 경계](local_cnn_reference_causal_20261001.md)를 참조한다.
아래의 '후속 진단 미구현' 문장은 당시 수신 시점의 상태이며 현재 구현 상태는 이 후속
문서에 기록했다. 기존 L1 수축은 실제 현상이지만 주원인 확정이나 reference 악화만으로
L1 가설 전체를 배제하는 결론을 내리지 않는다.

## 최신 서버 결과: 순위 gradient는 살아났지만 추천 개선은 확인되지 않음

사용자 `211a6441` 첨부 전체를 수신했다. 실행 commit은 `da9b1bb`, 물리 GPU3
RTX A6000, snapshot은 epoch39/step21320/refresh_memory다. 원래 전체533개 tile과
11,279개 학습 관측·67,456개의 P×U 비교 정규화가 유지됐다. 선택한 원래
schedule 위치4/5/6/7은 모두 liver_117이며, 각 batch는 P16+U16=physical32,
256쌍이다. 두 branch의 모든8 update에서 순위 전용 CNN/readout/L1/L2 gradient가
비영·유한 값으로 측정됐다. 이전의 rank0 prefix 문제가 이번 실행에는 없다.

Reference의 CNN ranking gradient norm은 같은 step의 legacy보다212~423배,
readout은173~430배 크다. CNN/readout의 parameter 구조와 초기 값은 같지만
reference의 L1 연산·새 BN·새 AdamW는 다르다. Norm 증가를 유효한 특징 학습,
최종 optimizer 방향 또는 정확도 향상으로 바꿔 설명하지 않는다.

| 평가 | Legacy 전→후 | Reference 전→후 |
| --- | --- | --- |
| train2case MRR | .121795→.105556 | .196429→.102679 |
| train2case pair-win | .495716→.518901 | .535786→.500756 |
| validation2case MRR | .541667→.541667 | .507353→.504762 |
| validation2case pair-win | .643973→.648158 | .606027→.586775 |
| validation R@5 | .142857→.142857 | .178571→.071429 |
| validation R@10 | .178571→.178571 | .214286→.107143 |

Reference의 검증 R@5는 관측 종양5/28→2/28, R@10은6/28→3/28이다.
liver_3의 단일 종양은68위→105위로 밀렸다. liver_109는 여전히 첫 종양이1위라
MRR은1이지만, top5/top10에 들어간 다른 종양 수는 줄었다. 따라서 aggregate
MRR이 거의 유지된다는 이유로 추천 품질이 유지됐다고 결론 내리면 안 된다.

**이번 비교가 직접 확인한 것은 순위 gradient 경로와, 선택된 다른 case들에 대한
짧은 update의 전이 효과다.** 실제 update 대상 liver_117은 출력된 평가 목록에
없다. Train 평가 대상은 liver_1/liver_49, validation은 liver_109/liver_3다.
따라서 학습한 tile/case 자체가 더 잘 맞춰졌는지는 미측정이다. 각 update도 서로
다른 native tile이므로 step1→4 loss를 동일 batch에서의 하강 곡선으로 해석하지
않는다. 미측정 항목을 0점이나 성공으로 채우지 않는다.

표시된 weighted ranking loss 약1.37은 전체 epoch 정규화가 적용된 값이다.
256쌍인 tile에서는 평균 pair loss에 `533×256/67456=2.02277039848`을 곱한
값이다. Evaluation의 평균 loss 약.693과 직접 비교하지 않는다. 같은 계수라
branch 간 gradient norm의 차이를 이 정규화로 설명할 수도 없다.

남아 있는 구분은 다음과 같다. Code에서 확인한 계약 차이이며 이번 콘솔에서
영향 크기나 주원인이 측정됐다는 뜻은 아니다.

1. Reference train은 joint BN이고 eval은 새 BN의 running statistics를 사용한다.
   네 update 후의 통계·parameter 영향을 분리하지 않았다.
2. 네 update의 support teacher는 같은 group에서 고정된다. 평가에서는 full
   eligible support와 새 teacher를 사용한다. 원래 episode 계약은 유지되어 있다.
3. Snapshot이 refresh_memory이므로 query는 현재 CNN, support는 저장된 완료
   bank다. Refresh 완료 여부/진행 cursor가 콘솔에 없어 완전히 갱신된 epoch 평가로
   부르면 안 된다.
4. 최종 optimizer는 ranking+CE+alignment 합을 사용한다. Ranking-only norm이
   커져도 이 합의 update가 ranking을 개선한다는 보장은 없다. Loss별 gradient의
   방향이나 같은 tile의 전후 margin은 이번 출력에 없다.

다음 진단이 필요하다면 새 장기 학습에 앞서 **실제 update한 liver_117의 원래
전체 후보와 동일 native tile의 전후 margin**을 기존 case 평가와 분리해서
확인해야 한다. 같은 support/teacher에서 dropout·BN 모드를 분리하고, 동일
forward에서 rank↔CE/alignment gradient 방향을 확인하는 범위다. 기존 L0/L2/loss,
BasicCP/후보128/mask를 동시에 바꿀 이유는 이번 결과에 없다. 이 후속 진단은
이번 수신·기록 작업에서 실행하거나 구현 완료로 표시하지 않았다.

서버 원문과16case/8update 전체 숫자 및 계산한 차이는
`validation/reference_rankable_20261001/server_rankable_result_original.txt`와
`server_rankable_result_transcription.json`에 보존했다. 원본 서버 JSON은 로컬에
없으며 콘솔에 없는 BN state·full gradient cosine·update delta는 미수신이다.
모델 교체·새 GPU 실행·production 학습은 하지 않았다. 이번 작업은 **실제로
수신한 A6000 GPU 결과의 해석과 기록**이며 CPU model 검사를 새로 한 것이 아니다.

## 최신 수신: 기존 4-update에서 순위 비교가 없었던 문제 수정

`fffd1a29` 첨부의 두 서버 신호 출력 전체를 확인했다. Snapshot은
epoch39/step21320/refresh_memory다. 이전 legacy/reference의 **4회 각각 모두**
`ranking_pairs=0`, `ranking_loss=0`이었다. 관측 CE와 alignment로 변한 모델을
순위 학습 대조로 해석할 수 없다. 이것은 기존 DEBUG prefix 선택의 오류이며,
전체 production 학습이 항상 rank0이었다는 뜻은 아니다. 실제 전체 schedule에는
양성이 없는 case의 pure-U tile도 의도적으로 보존되어 있다.

새 `--update-selection rankable_full_batch_prefix`는 원래 전체 schedule에서
양성과 미관측이 함께 있는 **원래 physical batch 크기의 tile**을 순서대로 고른다.
입력 tile이나 후보를 새로 만들지 않고, 원래 schedule 위치·ID·P/U·비교 수를
기록한다. 전체 관측·원래 multiplicity·전체 cohort loss 계수는 유지한다.
선택된 prefix로 정규화를 다시 만들지 않는다. 해당 tile이 부족하면 명시적으로
실패하며 다른 batch나 작은 batch로 대체하지 않는다. 기존 complete-prefix/raw
전이 대조와 이전 결과는 그대로 보존한다.

같은 forward graph에서 순위 loss만의 CNN/readout/L1/L2 parameter gradient를
측정한 뒤 원래 전체 loss의 backward/clip/AdamW를 수행한다. CNN/L1/L2의 순위
gradient가 없거나 0 또는 비유한 값이면 성공으로 처리하지 않는다. `.grad`를
미리 채우거나 두 번째 forward를 실행하지 않는다. 이 추가 derivative 비용은
DEBUG timing에 포함되므로 production update 속도로 해석하지 않는다.

### 수신 수치의 해석

- 기존 L0 방향 에너지 `3.45e-8~1.40e-6` → L1 최종 `1.08e-12~3.87e-11`.
  네 case에서 방향 차이는 약3.1만~3.6만 배, raw 차이는 약15만~17만 배 더 작다.
  이 통계는 후보 표현의 수축이며 암 특징 손실률을 뜻하지 않는다.
- affine+out-bias0 전이는 L0 방향 에너지의 약71~73%를 보존한다. 그러나 실제
  score std는 train `7.39e-6~1.34e-5`, validation `2.06e-6~2.50e-6`이다.
  분산 회복을 순위 개선으로 승인하지 않는다.
- 기존 cross-class prototype cosine은 약 `-0.772~-0.049`다. 모든 prototype이
  같은 방향이라는 주장을 지지하지 않는다. Reference의 unavailable은 해당
  계측이 없다는 뜻이고 0이나 prototype 소실로 대체하지 않는다.
- 원본124줄을 byte 보존하고 case32행·trace32개·update8행을 숫자로 전사했다.
  원본 전체 JSON은 수신하지 않았다. 이전 임시 추정 대신 이번 실제 rank0 확인을
  기준으로 삼는다. `validation/reference_rankable_20261001/server_signals_*` 참조.

### 실제 CT CUDA smoke 완료

RTX5070Ti에서 새 `probe_updates` 선택 경로를 실제로 실행했다. 간 내 native CT,
CNN12/24/32, hidden128, L1 2층/4heads, L2 2층, FP32를 유지했다. 실제 liver_66의
양성5개와 원래 미관측128개를 모두 보존했으며 다른 bound DEBUG support case도
포함한139관측 context를 유지했다. Physical32 = P5+U27, 한 update당135쌍이다.
각 clone에1회 update를 실행하고 전후133관측 전체를 평가했다.

| 복제 branch | 순위 전용 CNN gradient norm | L1 | L2 |
| --- | ---: | ---: | ---: |
| legacy | 3.915216 | 1.251978 | 0.693539 |
| affine_relations_zero_out_bias | 24.833076 | 4.704340 | 2.302228 |

두 branch의 전체 loss optimizer에서 CNN/readout/L1/L2 parameter 변경을
확인했다. Peak allocated0.491GiB, 명시 CUDA 상한3GiB, RSS2.28GiB,
workers4다. 원본 model/gradient/mode/RNG/support/checkpoint/assignment 보존 검사가
통과했다. 실제 CT 입력·선택·forward/loss/rank-gradient/backward/optimizer를
연결한 **기계적 smoke**다. 서버 epoch39의 정확도 비교나 최종 모델 승인,
속도 개선 근거가 아니다. 보고서는 `actual_ct_cuda_report.json`이다.
입력 shape/crop 감사를 추가한 최종 동일 경로 검사도 통과했으며,
`actual_ct_cuda_final_report.json`과 `final_manifest.json`에 별도로 보존했다.

추가 helper/회귀44개도 통과했다. 이전 JSON 출력 두 종류×두 포맷은 이전
formatter와 byte 단위로 같고 production121파일 hash가 유지됐다.
기존 사용자가 수정 중인 REFERENCES/code/gpt_handoff 파일도 그대로다.
장기 GNN/nnU-Net 학습과 production checkpoint/ready 생성은 수행하지 않았다.

### 서버에서 수행할 짧은 순위 학습 대조

최근 성공했던 host 터미널의 물리 GPU3을 사용한다. A6000 하나만 노출된
Singularity 안에서는 `CP_GPU=0`으로 바꾼다. UUID를 다시 입력하지 않는다.
기존 학습 checkout과 별도인 diagnosis checkout에서 실행한다.

```bash
CP_GPU=3
cd /home/aicompetition06/Medical/HierCP-diagnosis-2ce11ba &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach FETCH_HEAD &&
python -u tools/diagnose_local_cnn_reference.py \
  --gpu "$CP_GPU" \
  --run /home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42 \
  --output "/home/aicompetition06/Medical/experiments/reference_rankable_$(date +%Y%m%d_%H%M%S).json" \
  --steps 4 \
  --update-selection rankable_full_batch_prefix \
  --update-reference-policy affine_relations_zero_out_bias \
  --cases-per-split 2 --workers 8 \
  --cuda-gib 24 --rss-gib 64 --resident-gib 24
```

화면에는 선택 schedule 위치·P/U·physical batch·rank pairs와 순위 전용
CNN/readout/L1/L2 gradient 및 before/after의 모든 선택 case 지표가 함께 출력된다.
두 branch 모두 fresh AdamW이며 reference BN은 새 통계다. Exact resume가 아니다.
이번 패치 후 해야 할 판정은 실제 서버 snapshot에서 rank 신호와 candidate spread가
유지되는지다. 좋은 결과를 만들려고 후보·loss·BasicCP를 동시에 바꾸지 않는다.

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

## 서버 고정 가중치 결과와 저장된 전체 case 평가 출력

사용자가 제공한 `reference_fixed_20261001_185138.json` 완료 콘솔은 `f803c45`의
zero-update 결과다. Actual batch32, 전체 schedule533개 중 첫 tile이며
saved-next=False다. 이 tile의 score는 NOT_EVALUABLE로 출력됐다. Helper는
한 observation class만 있는 tile에서 이 값을 기록하므로 P×U ranking 쌍이
없다는 뜻이다. 실제 P 개수·class·snapshot epoch/step은 콘솔에 없어 추정하지
않는다. 이 tile을 더 좋은 case로 바꾸거나 실패로 감추지 않았다.

동일 L0의 normalized centered energy는 8.213e-8이다. 아래 값은 콘솔 반올림
값에서 계산한 **방향 에너지 비율**이며 정확도·feature amplitude·생물학적
정보 손실률이 아니다.

| 초기 eval | L1 최종 normalized centered energy | L0 대비 비율 |
| --- | ---: | ---: |
| legacy | 2.510e-12 | 약 0.00306% |
| raw_columns | 1.648e-8 | 약 20.07% |
| affine_relations | 1.621e-8 | 약 19.74% |
| affine_relations_zero_out_bias | 6.071e-8 | 약 73.92% |

Legacy L1의 1층은 약21.80배, 2층은 추가 약1,500.80배, 전체 약32,721배
energy를 수축시킨다. Affine+bias0 reference에서는 전체 수축이 약1.35배다.
Affine relation remap만으로는 raw보다 energy가 커지지 않았다. Reference
중 bias0 차이는 이 초기화의 degree-bias 수축 영향을 보여주지만, 전체 operator
차이까지 한 요인으로 분리하거나 순위 개선으로 판정하지 않는다. L0 자체도 이
tile에서 거의 공통 방향이며, 기존 fusion/readout의 문제가 없어졌다는 뜻이 아니다.

Joint train BN에서 reference energy가 크게 늘어도 ranking 비교는 여전히
NOT_EVALUABLE다. Alignment→query gradient는 legacy에서 None, raw/affine/
bias0에서 각각 2.255521e-4/7.761518e-5/4.843674e-4였다. 기존 loss 공식과
달라진 gradient dependency를 구분한다. 전체 서버 JSON은 로컬 미수신이고
콘솔 전사는 `server_fixed_console_summary.json`에 별도로 보존한다.

**콘솔 출력에서 전체 case 지표를 빠뜨린 것을 수정했다.** 당시 진단은 이미
네 선택 case의 모든 후보 MRR·pair-win·각 관측 종양 순위를 계산해
`comparison.branches[].initial_full_case_evaluation`에 저장했다. 기존 콘솔은
tile mode/gradient만 출력해서 그 순위 결과가 보이지 않았다. 새
`summarize_local_cnn_reference.py`는 저장 JSON만 읽고, 모델/GPU/optimizer를
실행하지 않는다. 모든 branch/split/case와 전체 observed rank 목록을 출력하고,
없는 값은 unavailable/NOT_RUN으로 둔다. 일부 페이지나 case로 잘라내지 않는다.
미래 진단 CLI도 이 같은 요약 함수를 사용해 전체 case 결과를 표시한다.

요약 도구의 CPU 검사21개가 통과했다. 고정 비교와 이전 update 비교의 저장
JSON 두 종류를 `python -S`로 읽어 site-packages 없이 실행했고, 입력 파일의
SHA256이 그대로임을 확인했다. 누락 batch/update 수를 추정하지 않고,
case별 R@1/5/10과 gradient NOT_RUN 사유도 표시한다. Production source121개와
이전 검사 결과의 hash는 그대로다. 새 증거는 `saved_summary_checks.json`과
`saved_summary_unit_log.txt`에 기록했다. 이 검사는 JSON 출력 검증이며 새 GPU
smoke나 서버 성능 검증이 아니다.

## 저장된 전체 case 요약 수신: epoch39 / step21320

사용자가 `3eb0402` 요약기로 읽은 전체 콘솔을 추가로 제공했다. Snapshot은
epoch39/step21320, phase=refresh_memory다. 첫 deterministic tile은 P0/U32였고,
ranking pair/loss/query gradient는 모두0이다. 이 tile에서 빈 P×U 합이0인 것은
정상이며, 전체 학습의 ranking gradient가 끊겼다는 증거로 사용하지 않는다.
이전 최초 콘솔에는 없던 이 정보가 새 요약에서 확인됐다. 원본 서버 JSON은 여전히
로컬 미수신이며, 수신한 **요약 원문**과 분석을 별도로 보존했다.

| 경로 | train MRR | train pair-win | validation MRR | validation pair-win | liver_3 종양 순위 |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 0.121795 | 0.495716 | 0.541667 | 0.643973 | 12 |
| raw_columns | 0.109649 | 0.495212 | 0.505102 | 0.452288 | 98 |
| affine_relations | 0.266667 | 0.512349 | 0.504902 | 0.382254 | 102 |
| affine_relations_zero_out_bias | 0.196429 | 0.535786 | 0.507353 | 0.606027 | 68 |

Reference의 normalized energy 회복은 **일관된 순위 개선으로 이어지지 않았다.**
Bias0의 train pair-win은 높아졌지만 validation MRR/pair-win 및 단일 양성
case에서는 legacy보다 낮다. 이 reference는 zero-update 전이 대조이며 새 L1을
학습 완료한 모델이 아니다. 즉시 전이 성능으로 PRODIGY 자체의 학습 가능성을
판정하거나 production L1을 자동 교체하지 않는다.

현재 MRR는 각 case의 **첫 관측 종양 순위의 역수**를 case별 평균한다. 모든
branch에서 liver_109는 관측 종양27개 중 첫 종양이1위라 MRR=1이다. 그 case의
R@1은1/27=0.037037이다. Liver_3의 종양은1개뿐이며 reference에서는68~102위다.
따라서 validation MRR 약0.5를 후보 정확도50%로 읽으면 안 된다. Validation
pair-win도 liver_109의3456쌍이 전체3584쌍의96.43%를 차지하므로 case별 결과를
함께 봐야 한다. 같은 N/P의 무작위 순서에서 기대되는 first-positive MRR는
두 train case 평균 약0.267638, 두 validation case 평균 약0.206263이다.
이는 조합식으로 계산한 비교 기준이며 실제 모델 실행 결과나 모집단 검정이 아니다.

표시된 모든 pairwise loss는 ln2=0.69314718056에서 약-1.73e-6~+6.16e-8 이내다.
Pairwise loss 개선은 매우 작지만, 이 평균만으로 모든 점수 차이가 작다거나
모든 후보가 동점이라고 단정하지 않는다. 특히 단일 양성 liver_3의 순위와 strict
pair-win을 대조하면 exact P/U tie는 최소 legacy1/raw5/affine5/bias0 3개가
존재한다. 전체 tie율이나 FP32 반올림의 영향 크기는 저장된 score 통계로 확인한다.

원문은 `server_fixed_saved_summary_original.txt`, 전사는
`server_fixed_saved_summary_analysis.json`에 저장했다. 네 branch/두 split/전체
16case 행과 모든 observed rank를 보존했고, 무작위 기준 및 tie 하한은 **계산된
비교 값**으로 표시했다. Production 모델·loss·mask·가중치는 변경하지 않았다.

다음 확인도 새 GPU 학습이 필요 없다. 동일 서버 JSON에 이미 저장된 case별
score_std/min/max, positive-minus-unobserved, exact_tie_rate, 전체 case의 L0/L1
stage energy/norm을 읽는다. 이전4-update JSON에는 매 update의 ranking_pairs와
ranking_loss가 있으므로, 그4회에서 ranking 신호가 실제 들어갔는지도 확인할 수
있다. 고정 P0 tile만으로 이전4회 모두 ranking=0이라고 추정하지 않는다.

이를 위한 stdlib `--signals` 출력 검사를 추가했고 CPU 검사29개가 통과했다.
기존 기본 요약은 byte 단위로 그대로다. 고정/update JSON 모두에서 저장된
score6항목, case별 stage energy/norm, 기록된 prototype cosine(없는 reference는
unavailable), fixed query ranking pair/gradient와 이전 **모든** update의
rank pair/loss/CE/alignment를 출력한다. 새 점수나 순위를 계산하지 않는다.
검사 결과는 `saved_signal_checks.json`, 로그는 `saved_signal_unit_log.txt`다.
입력 JSON·이전 증거·production121파일 hash가 보존됐으며 새 GPU 실행은 없다.

원본 JSON 두 개의 추가 신호를 한 번에 읽는 명령은 다음과 같다.

```bash
cd /home/aicompetition06/Medical/HierCP-diagnosis-2ce11ba &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach FETCH_HEAD &&
python -u tools/summarize_local_cnn_reference.py \
  /home/aicompetition06/Medical/experiments/reference_fixed_20261001_185138.json --signals &&
python -u tools/summarize_local_cnn_reference.py \
  /home/aicompetition06/Medical/experiments/reference_l1_20261001_173620.json --signals
```

기존 결과를 읽는 명령은 다음과 같다. Production 모델이나 기존 결과를 바꾸지
않고 새 summarizer만 설치한다. GPU 선택 입력은 필요 없는 파일 읽기 명령이다.

```bash
cd /home/aicompetition06/Medical/HierCP-diagnosis-2ce11ba &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach FETCH_HEAD &&
python -u tools/summarize_local_cnn_reference.py \
  /home/aicompetition06/Medical/experiments/reference_fixed_20261001_185138.json
```

## 새 GPU 진단 실행이 필요한 경우의 원래 명령

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
