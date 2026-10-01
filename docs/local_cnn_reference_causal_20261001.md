# v2.2 — 학습 대상·loss/Adam·BN 원인 분리 진단

사용자 `99c17e59` 검토 전문을 읽고 기존 원문·결과를 보존했다. 이번 변경은
`--causal-probe`로 켜는 독립 DEBUG 계측이며 production 모델 또는 최종 학습 패치가 아니다.

## v1과 현재 모델의 실제 차이

| 항목 | 보존된 v1 코드·설정 | 현재 v2.2 LocalCNN |
| --- | --- | --- |
| 양성 정의 | donor의 원래 종양 anchor 1개를 원래 자리에 재구성 | 독립 train donor를 고정하고 recipient의 실제 관측 종양 anchor들을 양성으로 사용 |
| 비교 | pool128에서 한 training sample의 후보8, 선별·난이도 curriculum | 관측 양성 P와 미관측128 U의 전체 P×U 비교를 원래 physical tile로 분할 |
| L0 | CNN+당시 수작업 node 특징, heterogeneous GAT, 역할·shell attention readout | 간 내 native CT CNN12/24/32, 각 scale masked mean, project128, donor/recipient 차이·곱 fusion128 |
| 점수 | L0 표현을 상위 문맥과 함께 learned scalar MLP에 직접 입력 | fusion→두 L1 query update→L2 정렬 label의 prototype cosine mixture→class1−class0 |
| loss | listwise CE + difficulty margin pairwise + ordinal + 단계별 mining | softplus P/U pair ranking + 전역 균형 observation CE + support alignment |

근거: `hiercp/curriculum.py:211`, `config/train.json:69`,
`hiercp/model.py:1237`와 `:1462`, `hiercp/loss.py:177`,
`l0_local_cnn/model.py:55`, `hiercp_v222/model.py:287`,
`hiercp_v222/clustering.py:122`, `l0_regions/donor_learning.py:74`.
이는 실제 코드 차이이며 특정 차이가 성능 저하를 단독으로 일으켰다는 증명은 아니다.
v1의 anchor 재구성 top1=1과 현재 cross-patient 관측 ranking은 같은 정답·후보
조건의 성능 대조가 아니다. 현재 same-donor/live P·U gradient/best-MRR 수정은
이미 적용되어 있으므로 과거 Sep30 진단의 donor 불일치 등을 현재 오류로 재사용하지 않는다.

## 검토 해석의 수정

기존 L1에서 후보 방향 분산이 약3만 배 줄고 reference에서 약71~73%가 남는
것은 관측된 현상이다. 분산은 조직/CP 적합성을 구별하는 신호의 양과 같지 않다.
Reference에서 gradient가212~423배 커진 사실도 실제 AdamW 이동이나 정확도
향상을 뜻하지 않는다. 앞선 결과에는 실제 update case liver_117 평가가 없었다.

따라서 **L1 수축을 주원인으로 확정하지 않는다.** 동시에 reference는 operator,
fresh joint BN, query→BN→support alignment gradient 경로와 teacher geometry까지
함께 바뀌므로, 악화된4-update를 L1 원인 가설 전체를 배제하는 단일 개입 반례로
취급하지 않는다. 새로운 진단은 이 공백을 채우며 L1/BN/정렬을 production에서 바꾸지 않는다.

## 연결한 계측

1. 원래 schedule에서 실제 선택된 모든 update case를 기존 비교 case에 **추가**한다.
   매 update 후 해당 case의 원래 전체 후보를 평가한다. 기존 train/validation
   비교 case는 별도 aggregate를 유지하여 과거 지표 분모를 바꾸지 않는다.
   전체 P×U margin·pair-win·loss·observed ranks·R@1/5/10·score std를 저장한다.
2. 같은 native tile의 전후를 같은 episodic support·frozen teacher에서 평가한다.
   dropout을 끈 eval clone으로 측정하며 실제 train forward의 점수도 따로 기록한다.
3. 원래 **한 forward**의 weighted rank/CE/alignment/full 각각에서 parameter
   gradient를 구한다. 모듈 norm, rank↔CE/alignment/full cosine과 clipping을 기록한다.
   현재 cloned optimizer의 moments를 그대로 복제한 rank-only/auxiliary/full
   AdamW shadow를 순차 실행한다. Full shadow delta hash가 실제 original full
   backward/clip/step delta와 정확히 일치하지 않으면 실패한다. Step2 이후 history도 유지한다.
4. Reference 첫 선택 tile에 동일 weight/teacher/input/RNG의 BN·dropout 2×2
   비교를 수행한다. BN의 actual forward 수·counter·running statistics와
   alignment→query/CNN gradient dependency를 측정한다. 별도 BN-eval/dropout-off
   clone의1회 full update도 기록한다. 원래 branch의 BN/RNG/gradient는 보존한다.
5. L0/L1_1/L1_2에서 raw와 L2-normalized P/U class means·within-class variance·
   Fisher-like ratio를 원래 label binding으로 계산한다. Zero denominator나
   결측 class는 명시적 undefined로 남긴다. 이 통계는 held-out accuracy가 아니다.

`local_cnn_reference_target_signal.py`의 frozen linear probe는 별도 명시 API로
구현·UNIT 검사했지만 이번 실제 CT/서버 CLI에서 자동 fit하지 않는다. 실제 서버의
train-only frozen-feature fit/held-out 평가 결과는 아직 없다. 분산이나 Fisher ratio만으로
L0의 정보가 충분하거나 부족하다고 확정하지 않는다.

## 짧은 실제 CT GPU 검사

- RTX5070Ti, FP32, CNN12/24/32·hidden128·L1 2층/4heads·L2 2층 유지.
- 실제 liver_66 양성5+미관측128=133 모두 평가. 다른 bound DEBUG support 관측을
  포함한139관측 LiveContext의 원래 계수와 multiplicity 유지.
- Physical32=P5+U27,135쌍. 기존 원래 schedule0/1에서 clone당2 update.
- 두 branch4개 actual full step 모두 shadow delta와 exact hash 일치.
- 모든 fitted-case timeline은133관측을 유지하고 L0/L1 target statistics를 기록.
- 실제 GPU 검사18.22초(초기 checkpoint/CT 로딩 제외), peak allocated837,882,368bytes
  =0.780GiB, 명시 CUDA 상한3GiB, RSS2.30GiB, workers4. Epoch 속도 측정이 아니다.
- 추가23개 검사 및 기존44개 회귀 총67 PASS, skip0. 새 model/gradient numerical
  검사는 CUDA에서 수행했다. Admission·JSON 입출력 검사는 일반 코드 검사다.

| 완전한 fitted case | 초기 pair-win | 2-update 후 pair-win | 초기 mean pair loss | 이후 loss |
| --- | ---: | ---: | ---: | ---: |
| Legacy DEBUG | .651563 | .837500 | .688906 | .645775 |
| Reference DEBUG | .509375 | .815625 | .692696 | .682848 |

두 branch 모두 이 DEBUG에서 자기 case를 더 잘 맞췄다. 이것은 서버 epoch39
가중치의 개선이나 held-out 일반화 증거가 아니다. 서버 checkpoint는 로컬에 없으며,
사용한 checkpoint는 이전의 명시적 짧은 DEBUG다. Production checkpoint·ready·
장기 GNN/nnU-Net 학습을 만들지 않았다.

네 mode에서 reference alignment→query gradient는 eval/dropout-only에서0,
BN-train-only에서.08527, current-training에서.06882였다. 이는 해당 DEBUG에서
joint BN의 dependency 경로를 확인한 값이고 서버 악화의 원인 크기는 아니다.

## 해석 경계

- Frozen tile의 전후에는 실제 update가 바꾼 parameter와 BN running buffers가 함께 반영된다.
- Whole fitted case는 full support·새 teacher, tile control은 episodic support·frozen teacher다.
  Tile 개선/전체 case 악화를 바로 objective inconsistency로 확정하지 않는다.
- BN-eval cloned update는 dropout도 끈다. BN 효과는 같은 dropout 상태의
  fixed-weight factorial modes끼리 비교해야 한다.
- Mode control은 reference 첫 선택 tile1회이며 모든 tile/legacy로 외삽하지 않는다.
- Gradient/delta cosine은 즉시 방향이다. Rank-only shadow의 finite-step ranking
  metric 자체는 이번에 평가하지 않았다.
- 저장 snapshot phase·refresh completed/parts·bank hash·query local hash를 기록한다.
  Pending refresh parts를 완료 bank로 바꾸거나 완전히 갱신된 평가라고 표시하지 않는다.

증거는 `validation/reference_causal_20261001/actual_CT_DEBUG_report.json`,
`actual_CT_DEBUG_report.summary.txt`, `unit_receipt.json`과 원본 검토 텍스트다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 짧은 DEBUG 범위를 명시했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 데이터로 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
