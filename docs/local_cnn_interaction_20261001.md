# v2.2 Local-CNN — epoch필드27 결과와 명시적 q–k interaction 진단 후보

상태: **clone-only 수정 후보 구현, 로컬 GPU/실제 CT smoke 확인**. 아래 서버 증거의 정리와 새로운 진단 후보를
구분한다. Production 기본 모델·loss를 바꾸거나 장기 학습을 실행하지 않았다.
전체 회귀 검사 결과와 실행 명령은 아래에 별도로 기록한다. 서버27epoch 가중치의 개선이나 전체 성능을 확인한 상태는 아니다.

## 서버 snapshot과 원문 보존

사용자 첨부 `e7338fde-8bfa-4e4a-b80a-ec0b15319982` 전체를 읽고 다음 파일에
원본 bytes 그대로 보존했다.

- `validation/local_cnn_interaction_20261001/server_epoch27_summary.txt`
- `validation/local_cnn_interaction_20261001/server_epoch27_transcription.json`

원문 파일 SHA256:
`7f82579591f3e03570a25ff767a53b6f01ae3b32598d7827d53492df63d19231`.
원문 크기6886bytes. JSON은 **화면에 출력된 수치의 전사**이며 원본 서버 report가
아니다. 표시 문자열과 숫자를 함께 기록하고, 파생 비율은 반올림된 요약값에서
계산했다. 출력 정밀도보다 정확한 값이 있는 것처럼 표시하지 않는다.

| 항목 | 이 첨부에서 확인된 값 |
|---|---|
| 실험 | `/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42` |
| 원본 report의 서버 경로 | `/home/aicompetition06/Medical/experiments/counterfactual_20261001_122757/report.json` |
| 저장 epoch 필드 / step | **27 / 14924** |
| phase | **refresh_memory** |
| margin / physical batch | **10mm / 32** |
| checkpoint **content** SHA256 | `62fda7c8dffd542d51c3f6c4cc752de703ac44712e6f722c5dd150b70c0b699b` |
| 선택된 case | train2개 / validation2개; 출력된 각 case의 관측+미관측 후보 전체 |
| diagnostic / production update / full evaluation | true / 0 / false |
| 최대 GPU 할당 | 3.44GiB |

Epoch 필드를 완료 epoch나 화면 표시 epoch로 재해석하지 않는다. GPU 이름, source
commit, 원본 JSON, 서버 학습 가중치는 이 첨부에 없으므로 미수신으로 기록한다.
첫 줄의 터미널 경로가 일부 잘린 것도 그대로 보존했다.

이전 epoch필드22 / step11872의 hash
`d397703cc5d21eaa6a899c62c8313ab5e53fa7fc5b580e6b6bdfac9a7c287e02`와
**다른 snapshot**이다. 두 결과를 동일 고정 가중치의 A/B 검사로 합치지 않는다.

## FFN scale 결과: spread 증가와 순위 개선은 다르다

이번 비교는 message1을 유지하고 **두 번째 query L1 FF residual**의 scale만
고정 가중치에서 바꾼 것이다. Scale1 production parity는 네 case에서 모두 true다.

| case | 기본 score std | FF0 / 기본 std 비 | 기본 pair win → FF0 pair win | 판정 범위 |
|---|---:|---:|---:|---|
| train liver_1 | 1.820107e-5 | 14.49 | 0.5553977 → 0.5319602 | spread 증가, ordering 하락 |
| train liver_49 | 5.364994e-6 | 14.42 | 0.4332031 → 0.4503906 | 일부 개선, loss는 조금 악화 |
| validation liver_109 | 2.560169e-6 | 14.64 | 0.6296296 → 0.6229745 | spread 증가, ordering 하락 |
| validation liver_3 | 2.355397e-6 | 14.16 | 0.7265625 → 0.7265625 | spread 증가, ordering 동일 |

FF0에서 분산과 margin 크기는 커지지만 순위가 일관되게 좋아지지 않는다.
따라서 FFN을 약하게 하거나 없애는 것이 추천 실패의 해결책이라고 승인할 근거가
없다. 이는 fixed-weight counterfactual이며 FFN 변경 후 재학습의 성능을 측정한
것도 아니다. 서버 loss가 ln2 근처인 사실과 현재 loss의 gradient가 사라진다는
주장도 구분한다.

현재 baseline의 MRR은 case별 train0.5/0.25, validation1/0.02857143이다.
Liver_109의 MRR1은 **해당 case의 첫 관측 양성이 1위**라는 순위 지표이며 전체
validation 정확도100%나 CP 적합성100%라는 뜻이 아니다.

## Attention 결과: 후보에 따른 source 가중치 차이가 매우 작다

네 case 모두 첫 L1 head별 candidate-weight 평균 분산은 약1e-20~1e-19이고,
둘째는 약2e-16~8e-13이다. 표시된 모든 후보쌍 평균 cosine은1이다. 첫째 L1의
JS는0~1e-15 수준, 둘째도 최대 약6.18e-9 수준이다. Entropy와 JS는 natural-log
nats 단위다. 출력된 cosine1은 반올림된 값이므로 개별 확률이 bitwise 같다는
주장으로 바꾸지 않는다.

이는 **현재 고정 snapshot에서 attention source weighting이 사실상 후보에
민감하지 않다**는 가설을 강하게 지지한다. Message 분산만 본 이전 결과보다
원인을 좁힌 측정이다. 다만 이 통계만으로 기존 attention에 구현 오류가 있다고
단정하거나, q–k interaction을 더하면 검증 성능이 오른다고 보장하지 않는다.
첫 층 shared label seed의 존재 자체도 버그로 취급하지 않는다.

기존 source scoring은 `MLP([q,k,edge]) → source softmax`다. 같은 nonlinear
activation 구간에서는 query 기여가 source 전체에 같은 additive shift가 될 수
있고, softmax가 이를 지운다. 이는 source weighting이 후보 차이를 잘 사용하지
못하는 구체적인 구조적 가능성이다. 현재 통계는 그 가능성과 일치한다.

Fusion도 계속 수축한다. 이번 네 case의 fusion input→output 정규화 후보 분산
비는 요약값 기준 약102~174배다. Epoch22의39~120배와 같은 snapshot의 값처럼
섞지 않는다. CNN map spatial variance는 별도 통계이며 candidate variance가
아니다. 방향 분산의 수축률을 종양 정보 손실률이나 정확도 저하율로 표시하지 않는다.

## Direct scalar control과 아직 없는 검사

Frozen recipient project 특징에 새 scalar head/fresh Adam을 사용한 rank-only
진단은100회 update, CNN update0이다.

| 범위 | MRR 전 → 후 | Pair win 전 → 후 | Pair loss 전 → 후 |
|---|---:|---:|---:|
| 선택 train cases | 0.1666667 → 0.1979167 | 0.5693044 → 0.5884577 | 0.6927898 → 0.6908739 |
| 선택 validation cases | 0.1310241 → 0.03635204 | 0.4383371 → 0.3618862 | 0.6933103 → 0.6935976 |

작은 train fit은 있지만 validation은 악화했다. 이것을 CNN 특징의 유용한 CP
표현이나 학습 실패의 단일 원인 입증으로 삼지 않는다. 새 head는 원래 L1/L2
scorer나 saved full-objective optimizer history와 다르다. 또한 v1 전체 모델을
재현한 실험이 아니다. 네 case의 이 결과는 전체 평가가 아니다.

**SHADOW는 NOT_RUN**이다. Snapshot이 refresh_memory라 다음 production
optimizer tile이 정의되지 않았다는 이유다. 따라서 이 snapshot의 Adam
history-only/full/rank delta 결과는 없다. Epoch22의 shadow 값을 이번 결과에
채워 넣지 않는다. `weights=true`는 출력됐지만 saved-payload/support-plan/
caller-RNG는 unavailable이다. 미표시는 보존 실패가 확인됐다는 뜻이 아니다.

Fusion identity-bypass branch는 이 요약에 출력되지 않았다. 원본 JSON이 없으므로
실행 여부를 추정하지 않고 **NOT_REPORTED / unavailable**로 기록한다.
Liver_49의 anchor scale3도 NOT_RUN이다. 지원 feature cell이 없는 candidate를
제외한 분산이나0값으로 대신하지 않는다.

## Alignment와 목적함수 경계

서버 감사 출력은84group / 533tile, 그룹별 K_g4~38이며 현재/균등-group weight
ratio 최소0.630394 / 최대5.988743이다. 현재 alignment coefficient mass는
tile 수에 비례한다. Per-group 전체 항목은 이 요약에 없으며 이전 로컬 감사의
세부 case값으로 서버 항목을 메우지 않는다.

Step-weighted와 group-balanced 중 의도한 L2 목적은 여전히 명시적으로 결정해야
한다. **이번 candidate에서 loss를 자동 균등화하지 않는다.** 모델과 dropout은
tile 사이에 변하므로 매번 같은 scalar loss를 복제했다는 설명도 하지 않는다.
Basic CP, 후보128, 전체 관측, 원본 paste mask, split/seed42는 보존한다.

## 구현한 최소 interaction 후보

기존 L1의 additive MLP를 보존하고 다음 항을 **더하는** clone-only 후보를 구현했다.

```text
logit = existing_MLP([q,k,edge])
      + beta * dot(q,k) / sqrt(head_width)
```

Beta는 명시적인 진단 설정이다. 자동 production default를 만들지 않는다.
Q/K/V와 MLP 가중치를 같은 checkpoint에서 가져오고, support와 query에 같은
source scoring 규칙을 사용한다. L2·loss·FF/message scale·L0/CNN 크기·physical
batch·전체 후보를 동시에 바꾸지 않는다. Beta0 baseline의 원래 출력/gradient
대조와 실제 query/source 결속 검사가 필요하다.

이 후보는 query와 key 사이에 source-dependent interaction을 넣어 후보 차이를
source weighting에서 사용할 직접 경로를 만든다. 단순 FF scale 조절의 실패를
해결됐다고 포장하지 않고 attention sensitivity와 검증 순위가 함께 반응하는지
측정하려는 것이다. 새로운 학습 방정식이므로 **기존 checkpoint의 exact resume
버그패치가 아니다.** Clone 상태에서 짧은 실제 scheduled tile의 full objective
A/B와 단위/CUDA 검증을 수행하는 단계이며 성능 개선은 아직 확정되지 않았다.

원본 optimizer/model/memory/plan/RNG 보존, finite loss·gradient·clip, rank gradient의
CNN/readout 도달, candidate attention·score sensitivity, 같은 작업량의 비용과
메모리를 함께 기록해야 한다. 무조건 attention 분산을 키우는 것이 성공 조건이
아니다. 최종 판정에는 validation 순위와 실제 CP/nnU-Net 비교가 별도로 필요하다.
여기서 장기 학습을 자동 시작하지 않는다.

## 실제 실행 경로와 비교 계약

- `tools/local_cnn_interaction_candidate.py`: 같은 state dict를 가진 독립 복제본의 두 L1을 교체한다. beta0 출력/gradient가 원본과 같아야 한다. 두 support history와 teacher partition은 branch별로 새로 계산한다. 기존 L2·MLP·FFN·Q/K/V·edge parameters는 실행 경로에 그대로 있다.
- `tools/local_cnn_interaction_runtime.py`: 원래 관측 ID·owner·class·donor 결속, 양쪽 query 환자 배제, 선택 환자의 전체 적격 관측을 검사한다. 평가 시에는 선택 case의 모든 원래 후보를 쓴다. CNN이 갱신되면 이전 query embedding을 재사용하지 않는다.
- `tools/local_cnn_interaction_updates.py`: 원본과 후보에 같은 full epoch schedule의 **명시된 짧은 prefix**와 fresh AdamW를 적용한다. 기존 loss/정규화와 saved physical batch, 원래 support 환자 선택을 보존한다. 각 branch의 CNN·fusion·L1·L2까지 실제 raw CT objective로 갱신한다. 저장된 Adam history를 잇는 exact resume가 아니다.
- `tools/diagnose_local_cnn_learning.py --interaction-only`: 이전 FF/fusion/direct-head 검사 전체를 반복하지 않고 이 비교만 실행한다. 이 옵션은 계산 종류만 선택하며, 선택 case의 관측을 버리는 옵션이 아니다.

고정 가중치의 MLP/dot logit 크기·attention 감도·점수/순위와 짧은 update 전후의 train/validation MRR·pair win/loss를 분리해서 기록한다. Support는 양쪽 모두 같은 **saved detached L0 table**을 사용한다. CNN update 후 support 전체를 갱신한 epoch 평가를 수행했다고 주장하지 않는다. 후보 beta1은 단위/진단 비교값이며 최적값이나 새 production default가 아니다.

같은 prefix라도 첫 branch의 CT cache가 차갑고 두 번째가 따뜻할 수 있다. `synchronized_update_seconds`는 forward/backward/optimizer 구간이고 `prefix_seconds`는 provider/검사/teacher 준비를 포함한다. 두 prefix의 총시간을 그대로 모델 속도 배수로 비교하지 않는다.

## 로컬 실제 CT DEBUG 결과

최종 보고서 원문: `validation/local_cnn_interaction_20261001/final_actual_CT_DEBUG_report.json`, 검사 기록: 같은 폴더의 `debug_receipt.json`. SHA256는 `9d428603bfaa7fa8fdb48df4647238eb2bae396c085dac1372593f9601c6a319`이다. Ledger가 선택한 snapshot은 DEBUG attempt0002, epoch필드1/step4/phase complete이며 서버27epoch 가중치가 아니다. RTX5070Ti, 모델1,125,718 parameters, hidden128·L1/L2 각각2층. 이 DEBUG inventory의 train8/validation2관측, physical batch2를 그대로 유지했다. 최종 batch를2로 바꾸지 않았다.

선택 train/validation 각각1case의 **모든 DEBUG 후보2개**를 비교했고, full schedule4tile 중 명시된2tile를 baseline/candidate 각각 갱신했다. 원래 loss와 fused AdamW, gradient clip5, LR0.0001을 사용했다. 모든 CNN/fusion/L1/L2에 finite gradient와 parameter 변화가 확인됐고 beta0 원본 parity 및 원본 가중치/모드/메서드/gradient buffer/RNG 보존이 통과했다. Production update/checkpoint는0이다. Peak 할당은 약0.24GiB다.

고정 가중치 beta1은 attention 감도를 높였지만, 2update 이후 선택 DEBUG validation MRR은 두 branch 모두1→0.5였다. **이 결과에서 validation 개선은 확인되지 않았다.** 해당 tiny DEBUG의 수치를 실제14,102관측 정확도나 CP 성능으로 제출하지 않는다. 서버 학습된 snapshot에서 같은 후보 비교가 필요하며, 이 검사를 통과했다는 이유만으로40epoch 학습을 자동 시작하지 않는다.

총100개 단위·CUDA 회귀 검사 통과(0실패/0skip, 13.788초). 별도의 full LocalCNN/L1/L2 native synthetic physical32 CUDA 검사는 실제 CT 정확도와 구분했다. Caller AMP가 활성화돼도 복제 update는 FP32를 유지하는 검사, 먼저 CUDA가 초기화된 suite에서의 검사도 통과했다. 결정론 비교는 환경을 먼저 설정한 독립 CUDA child에서 수행하며 원본 실행의 결정론 설정을 바꾸지 않는다. 문법9파일과 tracked whitespace 검사도 통과했다.

## 서버에서 수행할 짧은 비교

Git에 올라간 이 버전의 별도 checkout에서 아래 진단을 수행한다. 현재 A6000 Singularity 출력의 단일 visible GPU 번호0을 기준으로 했으며 다른 환경에서는 현재 보이는 번호를 넣는다.

```bash
python -u tools/diagnose_local_cnn_learning.py \
  --gpu 0 \
  --run /home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42 \
  --output "/home/aicompetition06/Medical/experiments/l1_interaction_$(date +%Y%m%d_%H%M%S).json" \
  --cases-per-split 2 --workers 8 \
  --cuda-gib 24 --rss-gib 64 --resident-gib 24 \
  --interaction-only --interaction-scales 0 0.25 1 \
  --interaction-update-steps 4 --interaction-lr 0.0001 \
  --interaction-training-scale 1
```

선택한 train2/validation2 case의 전체 후보에서 beta0/0.25/1을 고정 비교하고, 원래 physical batch로 baseline/candidate 각각4update만 기존 full objective를 실행한다. Refresh phase에서도 fresh diagnostic prefix로 정의하며 saved next update라고 부르지 않는다. JSON은 새 파일에만 저장하고 원본 checkpoint는 저장하지 않는다. Case 처리와 두 branch의 tqdm에 loss·시간·실제 batch를 표시한다. 개선이 없는 후보를 production에 자동 적용하는 처리는 없다. 이 짧은 비교는 최종 CP/segmentation 평가나40epoch 학습이 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검사에서는 OOM이 발생하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
