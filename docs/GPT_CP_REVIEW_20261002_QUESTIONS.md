# GPT 검토 요청 — v1처럼 유효한 CP 추천으로 작동하지 않는 이유와 수정 우선순위

작성일: 2026-10-02. 이 문서는 새 검토 ZIP에 넣을 최신 질문이다. 과거 `code.txt`, `gpt_handoff.md`, 연구 코드와 기존 결과를 덮어쓰지 않는다. ZIP의 최신 코드 지도·원시 측정 JSON·원본 v1 archive·아래 과거 증거를 함께 읽어 판단해 달라.

## 검토할 실제 문제

우리가 원하는 것은 graph 모양이나 실행 가능한 예제가 아니라, **현재 입력을 사용해 유효한 위치를 추천하고 원본 donor mask로 실제 CP까지 연결되는 모델**이다. v1에는 학습·추천·CP 경로와 과거 ranking 성공 기록이 있다. 현재 모델도 최신 DEBUG에서 실제 학습과 CP 실행 경로를 연결했지만, 별도 CT의 추천 상위 순위 개선은 확인되지 않았다.

질문은 다음과 같다.

**“현재 P/U 정답과 전체 실험 계약을 유지할 때, 현재 구현이 v1처럼 유효한 CP 추천으로 작동하지 않는 가장 타당한 이유는 무엇이며, 실제 코드와 현재 반례를 설명하는 가장 좁은 수정은 무엇인가?”**

v1과 현재 모델의 과거 수치는 같은 후보·정답·split·metric에서 평가한 두 모델의 성능 비교가 아니다. v1의 과거 Top1/MRR=1 기록을 현재 CP 정확도100%나 segmentation Dice로 해석하지 않는다. 반대로 현재 DEBUG의 기계적 CP 성공을 추천 품질 회복으로 해석하지 않는다.

## 변경하면 안 되는 정답·실험 계약

| 항목 | 현재 고정한 의미 |
|---|---|
| P / target=1 | 원본 CT에서 실제 관측된 적격 종양 anchor |
| U / target=0 | 기존 comparison의 미관측 위치 |
| Ranking 목표 | 같은 recipient case에서 `s(P) > s(U)` |
| Donor | 같은 recipient case의 P/U에 공통으로 고정한 train-only 조건 입력 |
| CP suitability | 현재 P/U와 같은 정답이 아니며, 별도의 donor-specific suitability GT는 없음 |

**U는 CP 부적합 정답이 아니다. P도 특정 donor를 이식했을 때의 적합성 정답이 아니다. Donor를 바꿨다는 이유로 P/U label을 다시 부여하지 않는다.** 관측 위치를 높은 점수로 학습하는 현재 objective가 CP 효용의 어느 부분을 대리하는지 분석하되, 이를 GT 오류로 단정하거나 v1의 source-anchor GT로 복원하는 수정안을 내지 말아 달라. 과거 문서의 donor mismatch→GT 복원 우선 제안은 사용자 정정으로 철회됐다.

Basic CP80%, split/seed42, 원본 관측 ID·native anchor·모든 P·case당 U128·원본 donor component·원본 paste mask·CNN 채널과 깊이·hidden128·L1/L2·전체 loss·physical32를 유지한다. 모델·graph·데이터를 작게 만들거나 hidden subset/cap을 추가하는 방식으로 해결하지 않는다. 최종 학습 계약과 아래 명시적 DEBUG를 구분한다.

## 최신 실제 CUDA 비교 — 읽어야 할 원시 증거

최신 runner는 `tools/compare_latest_sparse_cp_learning_debug.py`다. 완성된 원시 결과는 `work/latest_sparse_cp_learning_DEBUG_20261002/report.json`, 요약은 같은 폴더의 `compact_summary.json`이다. ZIP에서는 이 원시 JSON을 포함한 최신 증거 경로를 코드 지도에서 확인할 수 있다. 원시 score·case별 관측 순위·update 기록을 요약값 대신 직접 읽어 달라.

비교 범위는 기존 로컬 DEBUG의 train4case와 held-out1case다. Train은 P15+U512=527관측, held-out은 P8+U128=136관측이며 총663/14,102=4.70%다. 전체84 train/21 validation 실험이나 전체 segmentation 평가가 아니다. CNN과 기존 fusion/L1/L2의 출처는 **로컬 DEBUG saved step4**이며, 학습 완료 서버 checkpoint를 이어받은 비교가 아니다.

두 branch는 같은 원본 observation schedule에서 fresh AdamW로 **각 branch 총60회의 연속 update(20회×3cycle)**를 수행했다. 초기/20/40/60update에서 train 전체 support를 현재 L0로 갱신하고 query native CT도 현재 CNN으로 다시 인코딩했다. Validation은 support·teacher fitting·optimizer update에 들어가지 않는다. Cycle마다 모든 train527관측을 사용하고 전체 P×U1,920쌍을 정확히 한 번씩 비교했다. Physical/effective batch32, accumulation1을 유지하며 자연스러운 tail25/20/15/15를 명시했다.

| Branch / 범위 | P/U pair-win 초기→60 | MRR 초기→60 | R@1 초기→60 | R@5 초기→60 | Pairwise loss 초기→60 | P−U 평균 margin 초기→60 |
|---|---:|---:|---:|---:|---:|---:|
| Native CNN / train | 54.58%→82.97% | .132639→.508333 | .000→.066667 | .066667→.466667 | .687423→.443847 | .012122→1.016601 |
| Native CNN / held-out | 40.43%→53.22% | .111111→.071429 | .000→.000 | .000→.000 | .694541→.739231 | −.002735→−.019803 |
| Footprint graph / train | 42.50%→92.71% | .055907→.687500 | .000→.133333 | .000→.533333 | .696878→.264581 | −.007109→2.220735 |
| Footprint graph / held-out | 56.93%→50.78% | .100000→.062500 | .000→.000 | .000→.000 | .691090→.953766 | .004410→−.039390 |

Pair-win은 동점에0.5를 주는 P/U 쌍 승률이다. MRR는 case별 **첫 관측 양성의 순위 역수**이며 R@K는 관측 양성에 대한 micro recall이다. Native held-out pair-win은 증가했지만 MRR·loss·평균 margin은 악화했다. 모든 지표가 악화했다고 단순화하지 않는다. Footprint graph는 train fitting을 강화했지만 held-out MRR·pair-win·loss 개선을 보이지 않았다. 이 한 CT의1,024쌍을 독립1,024개 case로 세거나 모집단 일반화 실패율로 바꾸지 않는다. 종료60step은 사전 고정했으며 validation에서 유리한 시점을 고른 best 결과가 아니다.

Native CNN은1,125,718 parameters, footprint graph는1,785,558 parameters다. 최신 graph는 native CNN8Conv12/24/32의 현재 특징과 후보 상대 좌표, donor/recipient 역할, near/mid/wide 정보를 쓰고, 완전한 donor footprint에 따른 물리 sampling/relay 경로, 3층128D relation-aware mean GraphSAGE, 역할별 attention readout, 기존 pair fusion과 L1/L2를 연결한다. Band별 context seed32는 이 DEBUG profile이며 relay가 붙는 실제 graph의 최종 node cap이나 검증된 최적 크기가 아니다. 이것은 v1 전체 GAT·입력 특징·patient/population 모델의 정확한 재현이 아니다.

실제 자원은 RTX5070Ti16GiB 한 개, logical CPU16, reader workers8, CUDA12GiB/RSS40GiB 한도다. 실제 precision은 FP32/deterministic/TF32 off다. 이 short DEBUG의 fresh AdamW backend와 production AMP/fused 실행을 동일하다고 주장하지 않는다.

## 최신 CP 결과 — 실제 실행 성공과 품질을 분리

최종 두 branch에서 같은 held-out CT, 같은 train donor, 같은 원래 U128 위치를 현재 model score로 순위화하고 기존 full-mask 필터와 `PlacementSpec.paste`를 실제 실행했다. Footprint 추론 경로는 `l0_sparse_feature/cp_diagnostic.py`, native 경로는 `l0_local_cnn/recommendation.py`, 공통 gate는 `tools/v22_rank_recommendation.py::rank_then_filter`다.

- 두 event 모두 후보128 전체를 사용했고 score override 없이 점수 순서의 첫 유효 후보를 선택했다. 기존 eligibility 기준은 전체 paste가 CT 안에 있음, 종양 overlap0, liver coverage≥0.85다.
- Native 선택 index21은 coverage.9423077, graph 선택 index126은 coverage1.0이며 두 선택의 종양 overlap은0이다. 순위 index 자체가 위치 품질이나 임상 적합성의 정답은 아니다.
- 두 event 모두 recipient spacing의 정확한 paste mask260voxels를 사용했다. Donor CT값·label2·mask 밖의 CT/label 보존과 원본 입력/모델 보존 검사를 통과했다. Runner의 `original_mask_sha256`은 **recipient-spacing paste mask hash**이며 donor-native full-mask hash와 구분한다.
- Graph 추론은 observation target을 받지 않고 원본 source·환자·native anchor·spacing/affine·원본 donor footprint와128개 실제 placement의 결속을 검사했다. Recipient organ mask와 최종 annotation overlap 필터까지 사용하지 않았다는 뜻은 아니다.
- `mechanical_pass=true`, `quality_evaluated=false`, `production_ready=false`, `production_checkpoint_written=false`다. 새 trained artifact/ready marker나 nnU-Net 실행은 없다.

따라서 “CP 코드가 끝까지 실행되지 않는다”는 설명은 이 최신 경로에는 맞지 않는다. 해결되지 않은 질문은 **학습된 점수가 관측 위치의 상위 순위를 일반화하는가, 그리고 그 대리 목표가 CP의 실제 효용을 충분히 설명하는가**다. 두 번의 기계적 paste 성공으로 후자의 답을 대신하지 말아 달라.

## 과거 서버·반사실 검사에서 이미 확인한 반례

최신 원시 JSON과 다음 원본 수신 증거를 함께 읽는다. 서버의 전체 report JSON·학습 가중치는 로컬 미수신인 경우가 있다. 보존된 콘솔/JSON reader 출력은 실제 수신 원문이지만 원본 서버 JSON인 것처럼 취급하지 않는다. Snapshot이 다른 결과를 같은 고정 가중치 A/B로 합치지 않는다.

| 기존 검사 | 확인된 사실 | 이 결과로 단정하면 안 되는 것 |
|---|---|---|
| Epoch필드22/step11872 deep | 선택4case에서 fusion 방향 분산39–120배, L0→마지막 L1 약7700–8160배 감소. CNN map/recipient project에는 차이가 존재. Message scale0도 ordering을 회복하지 못함 | CNN 출력 전체가 상수, 분산 감소율=종양 정보/정확도 손실률, mean pooling 또는 L1 하나가 확정된 주원인 |
| Epoch필드27/step14924 | FFN0에서 score spread 약14배 증가하나 validation ordering은 하락/동일. Attention weighting의 후보 감도는 매우 작음 | FFN 제거만으로 해결, attention spread 증가=추천 정확도 개선 |
| 같은 epoch27 direct head100updates | Frozen recipient project-r에 fresh nonlinear scalar head를 fit. Train pair-win56.93→58.85%, validation43.83→36.19%. CNN update0, donor/fusion/L1/L2/support 없음, rank-only | 이식에 유효한 CNN 특징 입증, CNN 무용성 입증, end-to-end scalar scorer 재학습의 실패 입증, v1 재현 |
| Epoch필드30/step16431 interaction | 기존 additive MLP에 dot(q,k) 항을 더한 clone의4 full updates에서 validation MRR/pair-win은 자기 초기값보다 낮아짐 | q–k 항 추가가 검증된 해결책, 더 큰 beta나 장기 학습이면 해결된다는 보장 |
| Epoch39/step21320 reference fixed | Zero-update 전이에서 legacy validation MRR/pair-win .541667/.643973, bias0 reference .507353/.606027. 단일 양성 liver_3 순위12→68이며 다른 reference는98~102위. Normalized energy 회복이 순위 회복으로 이어지지 않음 | 고정 전이의 energy 회복=학습된 추천 개선, 새 reference의 학습 가능성 반증, MRR 약.5=후보 정확도50% |
| Epoch39/step21320 reference rankable | 순위 전용 CNN gradient가 크게 회복됐지만 reference validation R@5는5/28→2/28, R@10은6/28→3/28. 새 operator·BN·전이와 fresh Adam을 함께 비교 | Gradient norm 증가=좋은 특징/유효 update, reference 즉시 전이 실패=L1 가설 전체 배제 |
| Causal/동일-BN finite 서버 검사 | Liver_117 P71/U128 전체에서 Legacy pair-win46.53→46.62%, reference48.97→46.93% 후에도 score std 약1e−6. Rank-only shadow의 후보 구분 회복 없음. 일부 step의 rank/CE 충돌과 반대 방향 step도 존재 | CE가 모든 step에서 방해, CE 삭제·과거 Adam history 제거·BN 고정만으로 해결, 모든 후보 완전 동점 |

Epoch39 fixed의 첫 deterministic tile은 P0/U32였고, 최초4-update prefix에도 ranking_pairs=0인 검사만 있었다. 후속 rankable prefix로 교정했다. Pure-U tile의 ranking loss0은 정상이며 전체 production gradient 단절의 근거로 확대하지 않는다. Fixed validation의 pair-win은 liver_109가 전체3,584쌍 중3,456쌍을 차지하므로 case별 결과도 함께 읽는다. 최신 finite의 네 rank-only arm은 각 evolving full branch에서 갈라진 **단일 step shadow**이며4회 연속 rank-only 학습이 아니다. Reference의 새 BN 현상을 LayerNorm인 Legacy production의 원인으로 옮기지 않는다.

`softplus(s(U)-s(P))`는 margin0에서도 단일 pair margin에 대한 미분이−0.5다. `ln2` 근처 loss가 곧 gradient 포화·부재라는 설명은 맞지 않는다. Rank loss는 모든 score에 같은 상수를 더해도 불변이므로 공통 score 이동은 학습 성공이 아니다. 기존 cross-class prototype이 모두 같은 방향이라는 주장도 측정된 cosine과 맞지 않는다. Live prototype에는 L2/support L1 쪽 gradient가 있으며 teacher assignment/center의 detach와 score prototype의 detach를 혼동하지 않는다.

## GPT에게 요청하는 코드 검토 질문

1. **실제 구현 오류가 있는가?** 최신 입력→native crop/footprint→CNN→node 선택/relay→SAGE/readout→fusion→L1→L2/prototype score→loss→backward→Adam→현재 support refresh→평가→CP 필터/paste의 경로를 코드와 원시 기록으로 확인해 달라. 좌표·role·mask·edge 방향·layer history·support 제외·gradient·coverage·schedule 결속에서 결과를 설명할 구체적인 결함이 있으면 정확한 파일/함수/행과 반례를 제시하라. 단순히 graph가 작거나 node가 disconnected라는 도식만으로 결함을 확정하지 말라.

2. **표현과 최종 score가 target 차이를 사용하는가?** Current query와 live class prototype의 cosine-mixture 대비가 실제 P/U 차이에 민감한 방향을 만드는지 수식과 미분 경로로 검토하라. Fusion/두 번째 query FFN/정규화/attention source weighting의 작은 차이와 공통 방향 증폭을 구분하고, raw centered/common energy·class-sensitive contrast·실제 score 차이로 가설을 연결하라. 국소 crop masked mean을 간 전체 평균으로 설명하거나, 현재 graph 개선이 mean-pool 가설을 자동 증명한다고 하지 말라.

3. **현재 full objective와 update schedule이 원하는 학습을 하는가?** `l0_regions/donor_learning.py`의 전체 pair mean, 반복 P의 CE inverse multiplicity, support alignment의 tile별 계수와 optimizer update를 실제 schedule에 대입해 검산하라. CE/ranking/alignment의 gradient 방향·Adam preconditioning·train dropout/eval readout 차이를 구분하라. Alignment가 step-weighted인 사실과 의도하지 않은 weighting bug를 구분하고, 의도 확인 없이 loss를 삭제·재정의하지 말라. Saved full-objective Adam shadow와 fresh Adam, 지속 fitting과 한 step 반사실 비교도 구분하라.

4. **Fit은 되는데 held-out 상위 추천이 나아지지 않는 이유를 어디까지 말할 수 있는가?** Native와 footprint branch의 실제 train/held-out 초기→최종 지표를 모두 설명하라. Native pair-win 개선과 MRR/loss 악화가 동시에 가능한 이유, 모든 양성의 순위와 첫 양성 MRR의 차이를 분석하라. Case1개로 확정할 수 없는 내용을 표시하고, 작은 cohort의 과적합 양상을 서버 전체 실패의 단일 원인으로 옮기지 말라.

5. **관측 위치 ranking과 CP 효용 사이에 무엇이 아직 검증되지 않았는가?** 현재 P/U 목표를 유지한 채 v1의 source-anchor/curriculum/target erase·learned scalar score와 현재 독립 donor/관측 ranking/prototype score의 차이를 코드에서 비교하라. 이 차이가 코드 결함인지 objective 차이인지, 실제 기여율이 미측정인지 구분하라. 새로운 donor-specific suitability label을 만들어 현재 GT를 대체하지 말고, 현재 점수/필터/paste만으로 알 수 있는 결과와 CP 효용에 필요한 추가 증거를 구분하라.

6. **가장 좁고 근거 있는 수정 하나를 우선한다면 무엇인가?** 위 반례를 설명할 수 있는 후보를 우선순위로 제시하라. 기존 CNN/graph/L1/L2/loss/전체 데이터 계약을 동시에 바꾸는 광범위한 재설계, graph quota sweep, CNN 확장, 다른 clustering 또는40epoch 재실행부터 제안하지 말라. Architecture/objective 변경이 필요하면 exact resume가 아닌 새 비교임을 밝히고, 최종 production 적용 전에 수행할 최소 CUDA 반증 검사를 구체화하라.

## 원하는 답변 형식과 판단 기준

- 각 주요 주장에 실제 **파일·함수·1-based 행 번호**를 붙이고, 입력 tensor/coordinate/role·score 수식·loss normalization·gradient·physical batch schedule을 설명한다. 제공되지 않은 파일/가중치를 읽었다고 주장하지 않는다.
- “코드에서 확인한 사실 / 측정으로 확인한 사실 / 아직 가능한 가설 / 반례로 지지되지 않는 설명 / 미검증 범위”를 구분한다. 단일 root cause가 입증되지 않았으면 그렇게 말한다.
- 수정 후보는 우선순위와 함께 변경 전후 방정식, 영향을 받는 모듈·학습 계약, 이 가설이 틀렸을 때 나올 관찰, 필요한 최소 CUDA 검사와 사전 판정 지표를 제시한다. 실행되지 않은 proof를 작성하거나 수치가 없는 성공을 가정하지 않는다.
- 새 검사는 현재 실제 native CT와 기존 same-donor/P/U128/physical32/full objective·현재 support refresh를 보존한 독립 DEBUG로 정의한다. 쌍 수나 전체 모델을 줄여 개선을 보이는 방식은 금지한다. 기존 검사와 중복되는 frozen head/fixed-weight sweep은 새로 하지 않아도 되는 근거를 먼저 확인한다.
- 최소 검사는 한 가지 가설을 분리해야 한다. 모든 trainable core의 실제 forward/loss/gradient/optimizer 연결, current-query/current-support 평가, train과 복수 held-out case의 before/final 상위 순위·loss·margin, CP 필터/paste 보존을 무엇까지 확인할지 명시한다. Case 수를 더 확보해야 하면 기존 split과 실제 자원에 근거해 제안하고 production 데이터를 축소하지 않는다.
- **장기 학습을 자동 제안하거나 성공을 보장하지 않는다.** Short DEBUG의 통과/실패를 읽은 뒤 필요한 후속 범위를 정한다. Full training·전체 holdout·segmentation·CP 효용이 측정되지 않았으면 미검증으로 남긴다.

이번 문서 작업 자체는 저장된 코드·실제 GPU 결과의 검토 요청 정리다. 새 모델 실행, 서버 접속, 학습 재개, GT 수정 또는 기존 결과 변경을 수행한 문서가 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 새 질문 문서만 작성했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실제 physical32·tail·worker 기록을 대조했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 새 실행 없이 최신 실제 GPU 결과의 자원 기록을 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 문서 작업에 OOM이나 모델 실행은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 최신 실제 update 기록과 소스에서 확인된 범위를 명시했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 이번 작업은 새 검토 문서 작성이며 연구 구현 변경이 아니다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
