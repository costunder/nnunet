# v2.2 — 학습된 서버 모델의 동일 BN finite 비교 결과

## 정답 계약 정정 — 사용자 지시 우선

사용자가 이후 명시적으로 정정했다. **v2.2의 P/U를 donor-specific CP 적합성
정답으로 해석하거나 v1의 정답으로 되돌리면 안 된다.** 앞선 보고의 “먼저
donor–위치 적합성과 학습 정답을 연결하는 기준을 바로잡아야 한다”는 결론을 철회한다.
첨부 `b28b3e51`의 donor mismatch→GT 복원 제안도 이 사용자 정정을 따르며 실행하지 않는다.

| 고정 항목 | 현재 의미 |
| --- | --- |
| P / GT=1 | 원본 CT에서 실제 관측된 적격 종양 anchor |
| U / GT=0 | 기존 comparison/unobserved center |
| ranking 목표 | 같은 case에서 s(P)>s(U) |
| observation CE | 제공된 P/U observation class를 예측 |
| donor | 한 recipient case에서 고정한 조건 입력; donor별로 GT를 새로 부여하지 않음 |
| CP suitability | 별도 GT 없음; P/U와 같은 의미가 아님 |

U를 CP 부적합, P를 특정 donor의 적합 위치로 부르지 않는다. Donor를 바꿔도 P를
U로 바꾸거나 U를 P로 바꾸지 않는다. TP/FN/TN/FP를 쓰면 위 **observation label**과
명시한 classification decision rule에 대한 값이며 CP 성공/실패가 아니다.
현재 구현은 이미 이 계약으로 `target`을 원래 observation과 대조하고 ranking/CE에
사용한다. 코드에 label bug가 발견된 것이 아니므로 production GT·assignment·loss를
수정하지 않는다. 아래 v1 차이는 실험의 차이를 설명하며 GT 변경의 근거가 아니다.

## 수신한 실행과 보존 범위

사용자 첨부 `47930b33` 전문을 새로 실행된 실제 GPU 진단 결과로 보존했다.
Commit `7df8731dc48252cbbf0aef1a4077b9fce2fad527`, `ece-a6gpu2`, 물리 GPU3의
RTX A6000이다. 원문은 `validation/reference_finite_server_20261002/server_console_original.txt`,
전사는 같은 폴더의 `server_transcription.json`이다. 원문 SHA256:
`0823ba7c05b6f372995b0406750a9757199da0b33bb6fa79e8ac21a156cc2b9c`.

서버 JSON은 `/home/aicompetition06/Medical/experiments/reference_finite_20261002_010234.json`에
있지만 그 파일 자체는 로컬에 전달되지 않았다. Snapshot epoch·step·content hash와
backend 정책은 콘솔에 없으므로 UNAVAILABLE로 둔다. 이전 출력과 수치가 같아도
checkpoint identity가 독립적으로 확인된 것으로 쓰지 않는다.

원래533 tile 중 schedule4/5/6/7을 선택했다. 모두 `liver_117`, P16+U16,
physical/effective batch32, accumulation1, tile당256쌍이다. 두 clone의 누적
full-objective update는 각각4회다. 전체 case199=P71+U128,9,088쌍을 별도로
평가했다. P71은 관측 record 수이며 서로 다른 종양71개라는 뜻이 아니다.
CNN12/24/32, L1두 층·128D·4heads, L2두 층과 원래 loss를 유지했다.

8회 모두 full shadow와 실제 optimizer 가중치 변화의 exact parity가 true다.
Finite readout은 총24회이며 동일 pre-forward BN buffer·dropout-off·native CT·
support·frozen teacher로 CNN부터 다시 계산했다. 완료된 실행은 코드의 검사 통과를
보여주지만, 원본 JSON이 없어 각 field hash를 로컬에서 독립 대조한 것은 아니다.

## 실제 가중치 이동이 후보를 분리했는가

다음은 같은 tile에서 관측 양성의 점수가 미관측보다 높은 비율이다. 동점은0.5로
계산했다. 임상 정확도나 segmentation Dice가 아니다.

| 4tile 평균 | update 이전 | rank-only 한 step | full 한 step | rank-only > full인 tile |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 46.14% | 48.05% | 47.27% | 2/4 |
| Reference | 49.56% | 49.61% | 50.10% | 2/4 |

Rank-only가 update 이전보다 높아진 tile도 각 branch에서2/4다. 큰 후보 분리의
회복은 확인되지 않았다. 특히 첫 Legacy rank-only step은 공통 점수 평균이
`+0.0114257354` 이동해도 P−U 차이 변화는 `+6.3329935e−8`뿐이다.
기존 평가 점수 표준편차는 여전히 약1e−6 수준이고 pair loss도 ln2 근처다.

이 첫 step은 **fresh AdamW**라 과거 생산 학습의 moments도, 앞선 clone full
update의 moments도 없다. 따라서 공통 점수 이동을 전부 예전 optimizer history나
현재 CE 때문이라고 설명할 수 없다. CE·alignment를 빼면 곧바로 해결된다는
production 변경 근거도 이번 결과에서 얻지 못했다.

단, 네 rank-only arm은 **4회 연속 rank-only 학습이 아니다.** 각 full branch의
현재 상태에서 한 step씩 갈라지는 shadow다. Step2~4에는 앞선 clone full update의
moments가 들어간다. 전체 case의 after0→4 timeline은 누적 **full-objective**
학습 결과이며 rank-only 전체 case 성능으로 붙이면 안 된다.

| 전체199 record의 누적 full update | Legacy 전→후 | Reference 전→후 |
| --- | ---: | ---: |
| 동점 반점 pair win | 46.53%→46.62% | 48.97%→46.93% |
| score std | 7.199e−7→7.993e−7 | 2.024e−6→1.305e−6 |
| top5 관측 양성 수 | 2→2 | 2→1 |
| top10 관측 양성 수 | 3→4 | 2→2 |

First-positive MRR=1은71개 관측 중 첫 양성이1위라는 뜻이다. 위의 전체 순위
분리나 일반화가100%라는 뜻으로 사용하지 않는다.

## BN 효과와 기존 모델을 구분

Reference 첫 step에서 같은 pre-forward BN으로 full 가중치만 바꿨을 때 공통
점수 이동은 `+0.0002993`이다. 기존 tile-after는 약`+0.01523` 이동한다.
둘은 같은 post-weight·query·teacher이므로 새 reference의 **BN buffer 갱신이
추가적인 공통 점수 이동**을 만든다. Step2~4에도 같은 현상이 있다.

그러나 BN을 고정한 arm에서도 후보 분리는 회복되지 않았다. 기존 Legacy는
LayerNorm 경로이며 finite-full과 기존 after 수치가 일치한다. 이 reference의
새 BN 현상을 Legacy production 실패의 원인으로 옮기지 않는다.
원래 train-mode gradient를 eval/dropout-off 점수로 읽으므로 두 함수는 다르다.
즉 이번 결과로 장기 rank-only 학습의 가능성이나 CNN 능력 자체를 부정하지 않는다.

## 이미 실행된 직접 head 검사를 중복하지 않는다

기존 epoch필드27/step14924, phase `refresh_memory`, checkpoint content hash
`62fda7c8dffd542d51c3f6c4cc752de703ac44712e6f722c5dd150b70c0b699b`의
직접 scalar head100-update 기록이 이미 있다. 원문·전사는
`validation/local_cnn_interaction_20261001/server_epoch27_summary.txt`와
`server_epoch27_transcription.json`, 설명은 `docs/local_cnn_interaction_20261001.md`다.

| Frozen recipient project-r + fresh nonlinear scalar head | train | validation |
| --- | ---: | ---: |
| 엄격 pair win 전→후 | 56.93%→58.85% | 43.83%→36.19% |
| pair loss 전→후 | .692790→.690874 | .693310→.693598 |

Train fit은 조금 있지만 held-out 개선이 없다. 이 대조도 v1 전체 모델 재현이나
CNN 재학습이 아니다. 이번 finite snapshot과 동일하다고 주장하지 않는다.
이미 있는 이 결과를 누락하고 “직접 head가 아직 미실행”이라고 설명하거나 같은
검사를 다시 만드는 것은 잘못이다. 이번 작업에서 신규 head/linear 진단은 추가하지 않았다.

## v1과 실제로 달라진 정답의 의미

코드에서 확인한 차이는 L0 연산과 후보 수만이 아니다.

| 항목 | v1 | 현재 same-donor LocalCNN |
| --- | --- | --- |
| donor와 recipient | 같은 case에서 종양을 선택 | recipient와 다른 patient group의 train donor를 선택 |
| 양성 | source 종양의 원래 `anchor_center` 한 곳 | recipient에서 실제로 관측된 종양의 anchor들 |
| 비교 | source 문맥·영역·prototype·난이도로 선별된 curriculum 후보 | 같은 독립 donor에 대해 모든 recipient P×128 U 비교 |
| 정답이 직접 뜻하는 것 | source의 원래 주변 문맥을 찾는 compatibility | recipient에서 종양이 관측됐다는 status; 해당 donor의 새 위치 적합성은 직접 관측되지 않음 |

근거는 `hiercp/curriculum.py:211`의 positive center=source.anchor_center,
`hiercp/pipeline.py:2615`에서 같은 case의 종양 선택,
`l0_regions/donor_data.py:23–36`의 independent donor assignment다.
후자는 target을 donor에 맞춰 새로 부여하지 않고 원래 recipient observation target을 유지한다.

v1 target CT는 source footprint를 변환해 양성·음성 모두에서 지우는 경로다
(`hiercp/local.py`, `hiercp/spatial.py`). 따라서 위 차이를 근거 없이 “v1은 정답
종양을 그대로 보여줘서 누수로100%를 만들었다”로 바꾸면 안 된다. 기존 shortcut
masking도 있다. 확인한 것은 **학습 target과 donor 조건의 차이**이며, 성능 차이의
기여율이나 단일 원인을 측정한 것이 아니다.

현재 P/U는 **관측 위치 순위화라는 objective의 정답으로 유지한다.** 특정 donor에
대한 CP 적합성 GT가 없다는 사실이 이 observation GT를 잘못된 label로 만들지는
않는다. v1은 원래 관계와 curriculum 관계, v2.2는 observed P와 unobserved U를
각자의 objective로 분리하는 능력을 비교한다. 차이가 있다는 것과 성능 저하의
원인이 supervision 오류라는 주장은 별개다. 이전의 GT 복원 우선 판단은 철회했다.

## P/U를 유지한 학습 경로 분석

`l0_regions/donor_learning.py:71–82`는 query target을 원래 row와 대조하고,
두 class의 현재 CNN 특징을 함께 계산한 뒤 `s=logit1−logit0`를 사용한다.
Loss는 `softplus(s(U)−s(P))`의 전체 pair 기준 정규화다. Margin이0일 때
정규화 계수를 적용하기 전 단일 항의 pair margin 미분은−0.5이므로 **ln2 근처의 loss 자체가 gradient를
포화시켜서 없앤다는 설명은 맞지 않는다.** GPU 결과에도 rank→CNN gradient가 있다.

Rank loss는 모든 점수에 같은 상수 c를 더해도 변하지 않는다. 따라서 score의
공통 이동은 순위 학습의 성공 지표가 아니다. Parameter gradient가 연결되고
가중치가 바뀌더라도 **eval에서 P−U 차이가 커지는지**를 따로 봐야 한다.
원래 train/dropout gradient와 eval readout의 차이, Adam preconditioning과
nonlinear parameter 변화 때문에 rank-only parameter step에서도 공통 이동은
생길 수 있다. 이것은 GT 불일치나 rank loss 오류의 증거가 아니다.

현재 scorer(`hiercp_v222/clustering.py:122`)는 정규화한 query와 live class
prototype의 cosine을 class별 logsumexp한 차이다. Query의 후보 차이가 있어도
그 방향이 **두 class prototype의 대비 방향**에 반영되지 않으면 score 차이는
작을 수 있다. Fusion·L1에서의 전체 variance 감소량만으로 target signal이
사라진 위치를 확정할 수 없다. 이 구조적 가능성의 기여율은 아직 미측정이며
prototype이 같거나 CNN이 학습 불가능하다고 단정하지 않는다.

이번 whole-case score min/max는 두 branch의 update 전후 모두0보다 작다.
**CE의 통상 argmax(=logit1−logit0>0이면 P)로 분류한다는 조건에서만**
이199개 record는 TP0/FN71/TN128/FP0으로 유도된다. 이는 콘솔 range에서
유도한 선택 case의 observation confusion이며 전체 평가나 CP 실패율이 아니다.
임의 threshold 전부에서 분류할 수 없다는 주장은 하지 않는다. 작은 score
amplitude만으로 분류 불가능을 증명할 수도 없다. 실제 P/U pair win이 거의
chance 수준이라는 측정과 이 지정 decision rule의 confusion을 구분한다.

유지할 원인 분석 질문은 **“P/U GT를 그대로 둔 현재 모델이 왜 그 두 집합의
score 차이를 학습하지 못하는가?”**다. 현재 증거는 L1만 교체·CE 제거·직접 head
교체가 해결책임을 보이지 않았다. 같은 기존 P/U와 자기 objective에 대한
fit·held-out 분리, stage→score의 target-sensitive 경로, train/eval 차이를
분석한다. 새로운 정답·donor-specific label·CP suitability label을 만들지 않는다.

## 현재 판단과 변경 상태

- 같은 BN의 finite 점수까지 검사했지만 현재 후보의 안정적인 구분 개선은 확인되지 않았다.
- L1 reference 교체, FFN 수축 완화, 즉시 CE 제거, frozen 직접 head 중 어느 것도 현재 자료에서 해결책으로 검증되지 않았다.
- 후보 분산이 있다는 것과 target과 관련된 일반화 가능한 특징이 있다는 것을 구분한다. **P/U 정답을 고정하고** L0 fusion·L1·prototype score·train/eval의 학습 경로를 분석한다. v1과의 target 차이를 현재 GT 오류로 해석하지 않는다.
- Production 모델·loss·Basic CP80%·split/seed42·후보128·전체 관측·physical batch·원본 mask는 변경하지 않았다. 장기 GNN/nnU-Net 학습을 자동 시작하지 않았다.
- 이번에는 원문·전사·판단 기록을 추가했다. 새로운 model 수치 검사나 smoke를 실행한 것으로 보고하지 않는다. 수신한 서버 GPU 실행의 결과와 이전 로컬 smoke를 구분한다.

## 저장된 GPU 결과의 ranking gradient 출력

`tools/summarize_local_cnn_reference.py --causal --mode-ranking`은 기존 JSON의
BN/dropout 네 모드에 이미 저장된 ranking weighted loss·query gradient norm·
CNN parameter gradient norm을 출력한다. 원래 `per_loss_query_CNN_gradients.ranking`
field만 읽고, 없는 값은 UNAVAILABLE, 실제0은0으로 표시한다. 기본 출력은 유지한다.
이것은 **저장된 GPU 측정의 텍스트 출력**이며 CPU 모델 검사나 추가 GPU 학습이 아니다.

서버의 저장 JSON에서 네 행만 한 번에 출력하는 명령이다. 이 도구가 포함된 commit으로
코드를 갱신한 뒤 실행한다. GPU 선택과 checkpoint 재로드는 필요 없다.

```bash
python tools/summarize_local_cnn_reference.py \
  /home/aicompetition06/Medical/experiments/reference_finite_20261002_010234.json \
  --causal --mode-ranking |
python -c 'import sys; print("".join(line for line in sys.stdin if line.startswith("    BN/dropout ")), end="")'
```

현재 서버 JSON 전문은 미수신이므로 이 네 모드의 **학습된 서버 ranking gradient
값을 새로 확인했다고 주장하지 않는다.** 보존된 실제 CT 로컬 GPU DEBUG JSON으로
field 연결과 출력은 검증했으며, 이것을 학습된 서버 모델 결과로 바꾸어 쓰지 않는다.
출력 회귀45개가 통과했다. 모델·optimizer 실행은 없으며 기존 실제 GPU 검사를
재실행한 것으로 보고하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 새 결과 폴더 사용.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 모델 변경 없음.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. Production 변경 없음.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 수신 DEBUG 범위를 명시.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존32 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 A6000 실행 확인; 자원 상세 미표시.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 수신8회 exact full parity 및 기존 GPU 검사 범위.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
