# HierCP 현재 코드 검수 및 수정 설계 — 2026-09-12

## 0. 결론과 이번 작업 범위

사용자가 다시 붙여 준 11개 절의 검토 요약을 **사용자가 준 프롬프트 원문**으로 취급했다. 별도 프롬프트를 요구하거나, 보고서가 어렵다는 이유로 일부 항목을 제외하지 않았다. `feedback`의 상세 보고서·159개 파일 대장·증거 ZIP과 현재 코드를 대조했다.

**현재 파이프라인을 검증 완료 상태로 승인할 수 없다.** 웹 보고서의 운영·평가 반례는 현재 코드에서도 재현된다. 추가로 실제 PyG 상위 계층에서 source-host 연결 변경이 L1·L2·최종 점수를 바꾸는 경로를 확인했다. 반면 핵심 GNN, raw-target resampling, feedback 관측이 가짜 구현이거나 전부 무효라는 결론도 맞지 않는다.

이 문서는 **검수 결과와 수정 설계**다. 이번 작업에서는 production 코드·설정·trainer·기존 실험 결과를 수정하거나 Git에 푸시하지 않았다. 아래 구현안과 승인 기준은 아직 구현·실서버 검증 완료를 뜻하지 않는다. 진단은 명시적인 합성 입력/CPU DEBUG이며 의료 데이터, 실서버 GPU, 전체 cohort 학습을 대체하지 않는다.

특히 다음 세 가지를 구분한다.

1. **확정 결함:** 실제 함수 반례 또는 명확한 실행 경계로 확인한 문제.
2. **구조적 위험/연구 해석:** 정보 경로·측정 범위·대조군 때문에 아직 보장할 수 없는 부분. 실제 성능 피해나 환자 누수를 자동으로 뜻하지 않는다.
3. **미실행 검증:** 실제 데이터/GPU/checkpoint가 필요한 검사. 생략하지 않고 구체적인 실행·승인 조건을 아래에 정의했다.

## 1. 검토 자료, 버전, 증거의 신뢰 범위

| 항목 | 확인 결과 |
| --- | --- |
| 웹 검토 대상 | `ec2d8388f39c6b03ed32a3677fbdd6eaafed4731` |
| 현재 검토 대상 | `d904eeb346da8592ea5f04029e8aaa475ec9126e` |
| 사용자 프롬프트 | 다시 붙여 준 11절 요약 및 해당 첨부 원문 |
| 상세 보고서 | `feedback/HierCP_code_review_20260912.md` 전체 읽음 |
| 파일 대장 | `feedback/HierCP_review_inventory_20260912.md` 전체 읽음 |
| 증거 무결성 | ZIP의 체크섬 목록 18개 모두 일치 |
| 외부 보고서와 ZIP 내부본 | 두 Markdown 모두 byte-identical |
| 기준 snapshot | 대장 159개 파일 집합 및 각 SHA를 실제 기준 Git blob과 대조, 모두 일치 |
| 현재 코드 재현 | 읽고 검토한 웹 진단 8개를 현재 코드에 다시 실행, 보고된 정상/실패 시나리오 모두 재현 |

기준 이후 변경 파일은 `code.txt`, `docs/online_cp_feedback.md`, `gpt_handoff.md`, bank 디스크 재시도 관련 helper/tests, `tools/online_cp_benchmark.py`다. 보고서의 주요 모델·feedback·계약 발행·평가·offline 실행 코드는 그대로다. **이전 디스크 재개 수정이나 기존 테스트 통과가 이번 지적을 해결했다는 근거는 아니다.**

이번에는 보고서 전체와 영향받는 활성 경로를 깊게 추적했다. 이를 “159개 파일 모든 분기를 현재 환경에서 전부 실행했다”거나 “실데이터 전체 검증을 완료했다”로 표현하지 않는다. ZIP의 모든 긴 테스트 로그를 새로 전부 실행·수동 증명한 것도 아니다. 웹의 기존 344개 성공 기록과 이전 로컬 전체 회귀 결과는 출처를 구분해야 한다.

### 현재 코드에서 다시 확인한 8개 시나리오

| 시나리오 | 현재 결과 | 증거의 한계 |
| --- | --- | --- |
| source-host topology | 마스킹 뒤에도 두 후보의 동일 source-region 2-hop 여부가 `[True, False]` | 해당 웹 진단은 graph container/일부 feature 경계 DEBUG 대체물 사용 |
| 계약 발행 실패 | 최종 검증 `OSError` → 최종 계약 파일 잔존 → 재시도 `FileExistsError` | 실제 publisher, 검증 경계에 의도적인 실패 주입 |
| offline full 경계 | baseline 경계 뒤 CP 누수 방지 예외 | 실제 학습 대신 기록용 DEBUG 경계 사용 |
| streaming 집계 | CPU float64 forward 오차 0, autograd gradcheck 통과, empty-edge 정상 | 전체 GPU 학습 동등성은 아님 |
| raw cubic/label mixture | global resize 대비 최대 절대 오차 약 `2.22e-15`, label mixture 일치 | 합성 입력 operator 검증 |
| 마스킹 raw 열 | `[0,1,2,4]` 해당 열의 data gradient 0 | 전체 모델 연결 없음이라는 뜻은 아님 |
| surviving support | crop 내부 support를 available로 평가 | crop 밖 원본 전체 병변 보존의 증거는 아님 |
| quality-aware matching | TP=1에서 유효 Dice 0.45 대신 0.30 선택 | 전체 TP나 whole-volume Dice 전체가 틀렸다는 뜻은 아님 |

8개 시나리오 재현은 **결함 재현을 포함한 결과**다. “8개가 통과했으므로 프로젝트 정상”으로 요약하면 반대 의미가 된다.

진단 JSON: `C:/Users/lock1/AppData/Local/Temp/HierCP_current_review_DEBUG_g__2y0j8/web_probes_current.json`. 원본 ZIP 및 제공된 보고서는 변경하지 않았다.

## 2. 프롬프트 전체 대응표

| 프롬프트 절 | 검토 및 설계 위치 | 판정 |
| --- | --- | --- |
| 1. 어디까지 확인했는가 | §1, §14 | 기준 SHA·증거·실행 한계 분리 |
| 2. 전체 구조 연결 | §3 | 실제 연결 있음, 최종 평가까지 완결은 아님 |
| 3. L1 source 위치 | §4 | topology 경로 확인, native 상위 점수 영향 추가 확인 |
| 4. 계약/캐시 발행 | §6 | 실패 후 재개 결함 확인, 두 종류 commit 분리 설계 |
| 5. 병변 매칭 | §7 | 반례 재현, 평가 버전 분리 필요 |
| 6. offline full | §8 | 누수 guard 보존, 시작 전 거절 필요 |
| 7. Difficulty 자원 | §9 | prefix 대표성·host 예산·I/O·state peak 상세 설계 |
| 8. 재개/최종 평가 | §10–11 | fresh journal 및 prediction provenance 미완결 |
| 9. 제대로 된 복잡한 부분 | §3, §14 | 보존할 구현과 실제 검증 범위 명시 |
| 10. 연구 설계 | §4–5, §11 | context 의미·대조군·support·환자 grouping 분리 |
| 11. 수정 순서 | §12–13 | 재사용 범위, 호환성, 단계별 승인 기준 정의 |

## 3. 현재 모델과 데이터 경로: 무엇을 보존해야 하는가

현재 활성 경로는 outer split → outer train 내부 inner split → region/context prototype와 Quality GNN → 공통 nnU-Net planning/preprocessing → raw-target CP bank 및 frozen quality → feedback 계약 → Full/Basic nnU-Net이다. Full에서는 정상 nnU-Net forward의 pre-update 예측으로 CP 오차를 관측하고, epoch 종료 시 EMA·별도 Difficulty GNN을 갱신해 다음 epoch 선택에 사용한다.

Quality와 Difficulty는 별도 목적·모델·상태다. Difficulty는 실제 hierarchy를 사용하고 출력 head를 교체하며, 관측 위치의 BCE → backward → clipping → AdamW로 연결된다. segmentation loss/backward를 feedback이 대체하지 않는다. 현재 코드를 단순 random/dummy/toy 예제로 분류할 근거는 없다. 다만 “실제 모델”과 “연구 목적을 올바르게 학습했음”은 별개다.

기준 구조와 실험 규모를 유지한다: hidden 128, head 4, L0/L1/L2 3/2/2 blocks, patient regions 24, population prototypes 16, Quality 40 epochs, nnU-Net 250 epochs, bank의 완전한 128-candidate 계약. Quality 학습의 8개 후보와 online 128개 후보는 다른 workload다. source별 exhaustive no-placement 0은 별도 유효 상태이고, 부족한 1–127개 후보나 검색 오류를 128개 성공처럼 승인하지 않는다.

현재 donor의 등가 지름 `0 < diameter ≤ 20 mm` 같은 명시적인 연구 대상 조건과, 오류를 피하려고 몰래 큰 sample을 버리는 조건을 구분한다. 전자는 현재 계약을 유지하고 사용자 승인 없이 임의 확대/축소하지 않는다. ROI/node/edge의 fail-closed 자원 상한 역시 넘는 입력을 조용히 잘라 성공시키는 cap과 다르다. 상한 초과가 실제로 발생하면 전체 입력 통계와 대안을 보고해야 하며 graph를 버리거나 cap을 숨겨 승인해서는 안 된다.

다음은 다시 작성하거나 약화할 대상이 아니다.

- L0의 dense 48³와 full-shape geometry 분리: dense patch만 보고 전체 종양을 잘랐다고 판단하지 않는다.
- context 384는 seed 수이며 interface/hop closure 뒤 최종 전체 node cap과 동일하지 않다.
- v3의 최종 L1 region/lesion/liver와 L2 prototype 상태는 candidate-conditioned readout에 연결된다.
- raw-target CP는 target별 원본 CT/전체 label mixture를 native resampling에 반영한다. 미리 resample한 종양을 단순 이동시키는 경로로 되돌리지 않는다.
- native 신규 support 0이어도 CP draw와 segmentation 학습을 유지하고 feedback unavailable로 구분한다. 다른 donor로 바꾸거나 관측값 0을 만들어 넣지 않는다.
- typed raw/legacy bank, 정확한 source/event/appearance 계약, donor no-placement 구분을 유지한다.

`Medical Data Aug`의 전처리·CP 의미도 보존 대상이다. 현재 문서에는 source pad 2 voxel, separation 12 voxel, coverage 0.85, clearance 2 mm, hard paste 등이 구분돼 있다. 예전 batch의 uint8 inversion 문제를 재현성이라는 이유로 복구해서는 안 된다. 현재 128개 후보/full search 연구 확장과 원래 실행이 bitwise 같은 것은 아니므로, “원본과 완전히 동일”이라는 표현 역시 쓰지 않는다.

## 4. F1 — L1 source 주소 단서: 확인과 설계

### 4.1 확인한 경로

근거: `hiercp/hierarchy.py`의 `build_patient_graph`(410–465 부근), `hiercp/model.py`의 upper feature masking(161–187)과 `PatientRegionPyGEncoder.forward_raw`(869–915), `hiercp/curriculum.py`의 원래 위치 positive 생성(205–222).

source/candidate의 위치 raw feature와 일부 tumor edge attribute는 지워진다. 그러나 실제 source anchor region으로 향하는 `tumor → hosted_by → region`과 역방향 연결은 남는다. 모델은 이 edge index를 실제 message passing에 사용한다. 따라서 원래 region → 해당 region의 candidate 경로가 존재한다.

`upper_position_noise`는 숫자 feature/pos/edge attribute를 바꾸지만 이 host 연결을 재배치하지 않는다. 이 검사만으로 topology shortcut까지 차단됐다고 승인할 수 없다.

### 4.2 실제 PyG 상위 모델 진단으로 추가 확인

합성 case의 실제 graph builder, PyG L1/L2, readout, ranking loss를 사용했다. hidden 128/head 4/24 regions/16 prototypes, physical B=2의 명시적 DEBUG 진단이며 후보 8개다. **L0 embedding은 고정된 합성 경계 입력**이다. 전체 native L0 학습이나 실데이터 bank 검사가 아니다.

숫자 feature를 고정하고 한 환자의 source host와 역방향 연결만 region 8→9로 바꿨다. patient graph는 각각 35 nodes, 362/372 edges였으며 무작위 초기화 모델의 eval 상태에서 실제 ranking backward를 수행했다. optimizer step은 수행하지 않았다. 진단 코드는 `feedback/debug_information_contract_20260912.py`이며 production 실행 경로에 연결하지 않았다.

| 측정 | 최대 절대 변화 |
| --- | ---: |
| 최종 후보 점수 | 0.00987194851 |
| 후보 간 상대 점수 | 0.01878063008 |
| L1 candidate 상태 | 0.607703745 |
| L2 candidate 상태 | 0.994724274 |
| L2 prototype 상태 | 0.529852092 |
| 건드리지 않은 다른 환자 점수 | 0 |

기존 숫자 feature의 금지 필드 perturbation은 점수 변화 0이었다. 즉 숫자 차단은 작동하지만 **topology의 정보 전달 경로는 실제 점수 계산에 살아 있다.** 무작위 초기화 모델에서 확인한 결과이므로 학습된 모델의 shortcut 사용량, 성능 하락, held-out 환자 누수 발생을 주장하지 않는다.

### 4.3 원문보다 정확하게 한정해야 할 부분

- L2 graph builder에 source/tumor node나 직접 source→prototype edge가 있는 것은 아니다. L1 candidate/region 상태가 L2로 전달되고 최종 readout도 L1 상태를 사용하므로 source 정보가 간접 전파된다.
- `_lesions`는 source full mask를 제외한다. source를 별도 lesion node로 그대로 중복 포함한다고 주장하지 않는다.
- region descriptor에는 per-region 종양 개수/점유율 열이 없다. 전역 liver raw에는 종양 component 수와 전체 CT 통계가 있어 donor 정보의 일부를 포함할 수 있지만, 이것은 정확한 host region 주소와 같지 않다.
- **원래 위치 positive를 포함한 모든 target의 L0 context에서 `erase_target=True`가 적용된다.** source branch는 종양을 유지한다. “positive target만 종양 CT 정답을 그대로 본다”는 결함은 확인되지 않았다.
- source와 원래 positive의 주변 context/geometry가 매우 비슷해 self-context matching을 학습할 가능성은 별도 연구 문제다. topology 주소 단서와 동일한 주장으로 합치지 않는다.

추가 정적 확인: `hiercp/schema.py:31–35`의 source/candidate absolute coordinates 금지라는 포괄적 설명도 현재 경로보다 강하다. `hierarchy.py:318–320`의 candidate→region delta는 마스킹되지 않으며 region position도 남는다. 따라서 `candidate_position = region_position − edge_delta`로 **목적지 좌표를 복원할 수 있다.** 이는 source-host 주소 문제와 별도다. 기존 position-noise 검사는 저장된 일부 tensor 열의 비사용을 확인할 뿐 모든 공간 정보의 부재를 증명하지 않는다.

recipient scene 정의도 명시해야 한다. L1 lesion 목록은 donor를 제외하고, region CT는 모든 tumor 내부를 대체하며, liver summary는 원래 모든 tumor를 포함한다. L0 target은 virtual footprint만 지우므로 negative의 넓은 문맥에는 footprint 밖 원래 donor가 남을 수 있다. 이것은 실제 within-patient recipient context일 수도 있다. 현재 서로 다른 branch의 scene 정의를 숨기거나 곧바로 누수로 단정하지 않고, 각 branch의 포함/제외 의미를 먼저 계약화해야 한다.

### 4.4 권고 정보 계약

목표는 source 출신을 통계적으로 절대 추론할 수 없게 만드는 것이 아니다. 그러면 사용자가 원하는 **비슷한 생물학적 context에 전이하는 기능**까지 제거할 수 있다.

| 정보 | 계약 |
| --- | --- |
| 종양 morphology, source 주변 조직, 상대적 종양-조직 관계 | 허용: 모델이 배워야 할 생물학적 내용 |
| destination의 간 내부 위치/context와 다른 병변 상태 | 허용하되 train/held-out 경계와 donor 제외 정책 명시 |
| 원래 source region ID, anchor 기반 host index, 정답 positive 표식 | scoring 경로에서 금지 |
| 전역 종양 burden/CT 통계 | 주소가 아닌 임상 context로 사용할지 별도 명시; 무조건 삭제하지 않음 |
| source와 target의 context가 비슷해서 생기는 상관 | 허용할 수 있음; 원래 주소를 따라가는 shortcut과 구분하여 평가 |

권고 구조는 source 생물학적 embedding을 **모든 recipient region/prototype에 동일한 내용 조건으로 제공**하고, 어느 region이 원래 donor의 host였는지 index로 주입하지 않는 방식이다. source-conditioned compatibility는 destination context를 보고 계산한다. L1 전체, region graph, lesion context, L2, block 수/폭은 유지한다.

목적지 해부학적 위치를 허용하자는 위 표는 **권고 설계이며 기존 schema의 금지 문구를 이미 충족했다는 판정이 아니다.** 구현 전에 금지 대상이 source의 정답 주소인지, destination의 전역 좌표까지인지 확정해야 한다. 후자까지 유지하려면 region/prototype의 위치 열과 모든 상대 delta의 조합, global reference를 함께 점검해 제거/재표현해야 하며 단순 raw masking으로 해결되지 않는다. 이 경우에도 목적지의 depth·국소 geometry 등 허용되는 해부학적 관계는 유지할 수 있도록 정보 계약을 먼저 정의한다. 기존 요구사항을 문구만 완화해 통과시키지 않는다.

대안은 후보 위치마다 가상 source를 배치한 candidate-conditioned scene이다. 이때 원래 host가 아니라 **각 후보의 가상 배치 관계**를 사용한다. 전체 region states를 공유하면서 후보별 상태를 분리해야 하고, 정확한 vectorization/streaming과 전체128개 입력의 자원 측정이 필요하다. 단순히 비싸다는 이유로 후보 수나 L1을 줄이는 대안은 아니다. 두 설계를 동시에 섞어 의미를 흐리지 말고, 첫 권고안을 기준으로 수식·입력 계약을 확정해야 한다.

필수 검사는 좌표/attribute/topology를 별도 조작하는 counterfactual, region ID의 일관된 permutation, source host/reverse edge만 바꾸는 검사, L1→L2 전파 검사, 환자 간 disjoint-batch isolation이다. source biology와 destination 내용은 고정한 상태에서 bookkeeping 주소만 바꿨을 때 점수가 불변이어야 한다. 반대로 합법적 context를 바꿨을 때 score·gradient가 반응해야 한다. 무조건 모든 상위 입력에 불변인 상수 모델로 이 검사를 통과시키면 실패다.

원래 anchor positive는 “자연 발생 위치”라는 pretext supervision이지 새 위치 CP의 병리학적 정답은 아니다. 같은 context의 다른 위치/다른 환자에 대한 일반화와 위치별 anchor 편중을 별도로 평가해야 한다. 실제 환자 split 누수 여부는 환자 mapping과 fold fitting 자료로 따로 확인한다.

negative 구성은 source region/prototype 및 scale/orientation corruption과 연관된다. corruption category 자체를 model 입력으로 주는지와 실제 합법적 geometry feature로 구분한다. same-context/different-address, same-address/corrupted-context의 교차 검증과 단순 nuisance-only probe가 필요하다. 또한 8개 training slate와128개 deployment slate에서 candidate 간 message passing이 달라질 수 있으므로 candidate permutation/equivariance 및 slate 민감도를 검사한다. 128개 중 8개만 남겨 이 차이를 없애는 방법은 허용하지 않는다.

## 5. F8 및 L2 의미 — 실제 연결과 불필요한 parameter를 분리

### 5.1 고정 0 열

현재 raw 14개 열 가운데 `[0,1,2,4]`를 항상 0으로 만든 뒤 기존 Linear에 넣는다. 실제 PyG DEBUG backward에서 다음을 확인했다.

- tumor/candidate projection, prototype candidate bridge, 최종 score raw 부분의 총 **3,584개 weight cell**이 고정 0 data gradient를 받는다.
- source-incident edge projection 12개 행렬의 해당 금지 attribute 열에서 **7,680개 cell**이 추가로 고정 0 data gradient다.
- 합계 **11,264개 cell**. 해당 parameter tensor 전체의 gradient는 0이 아니므로 `grad is not None` 또는 tensor norm 검사만으로 검출할 수 없다.

이는 hierarchy 전체가 loss와 끊겼다는 뜻이 아니다. 그러나 항상 사용하지 않는 열을 그대로 trainable parameter로 남긴 것은 AGENTS의 불필요한 parameter 금지 원칙과 맞지 않는다. AdamW weight decay로 값이 움직이는 것과 입력/목표로부터 학습하는 것도 다르다.

설계: 의미가 있는 raw/edge 열만 명시적으로 pack하여 projection에 전달한다. hidden width·head·blocks·node/edge를 줄이지 않는다. packing만 바꾸는 별도 변경에서는 기존 weight와 Adam moment의 열 mapping을 정의하고 scalar step을 보존한다. CPU float32의 네 raw projection packing 진단은 출력 오차 0이었지만, optimizer/GPU까지 완전한 migration 승인이 끝난 것은 아니다.

source-incident relation은 남는 edge 입력7개, 다른 relation은 기존12개를 유지해야 한다. 공통 `edge_dim`을 일괄7로 바꾸면 정상 feature까지 삭제되므로 relation별 allowlist/입력 폭을 구현해야 한다. 해당 `lin_edge` 경로는 `model.py:282–290`, 기존 전체 tensor gradient 검사는 `pipeline.py:2047–2050`에 있다.

**F1 topology 변경과 이 함수 보존 packing을 같은 “완전 호환 migration”으로 취급하지 않는다.** F1은 의미 변경이고, packing은 동등성 검사를 통과한 범위에서만 호환 변경이다.

### 5.2 L2가 실제로 하는 일

`hiercp/prototype.py`는 inner-train region의 16차원 context descriptor를 표준화해 16개 prototype으로 clustering하고, candidate/region과 prototype 간 관계를 구성한다. region context에서 종양 CT를 대체하므로 이는 **암 종류나 종양 phenotype을 직접 clustering하는 모델이 아니라 간/주변 조직 context의 population prototype 모델**이다.

사용자가 말한 “암들이 없던 구역이라도 context가 비슷하면 CP”는 hard-valid 위치와 context 기반 ranking이라는 현재 방향과 부합한다. 하지만 prototype 존재만으로 그런 전이가 실제 학습됐거나 성능상 유효하다는 보장은 없다. source 주소 경로를 고친 뒤 novel-region/원래 종양이 없던 region의 후보 선택과 downstream 효과를 따로 확인해야 한다.

종양 phenotype clustering까지 추가하려면 morphology/local lesion embedding 기반의 별도 학습 대상·train-only bank·loss·readout이 필요하다. 현재 기능을 그 기능으로 이름만 바꾸거나, 사용자 승인 없이 연구 범위를 추가하지 않는다. 이전 fold 0의 Level2 제거 성능이 더 좋았다는 결과도 현재 새 설계의 유효성/불필요성을 확정하는 증거로 재사용할 수 없다.

## 6. F2 — 계약·그래프 캐시의 안전한 발행과 재개

### 6.1 현재 결함

`tools/online_cp_curriculum.py::publish`는 최종 `feedback_contract.json`을 만든 다음 verifier를 실행한다. 실패하면 파일이 남고 다음 실행은 존재 여부만 보고 거절한다. artifact 발행 후 journal 완료 저장 사이에 실패해도 같은 문제가 생길 수 있다.

`hiercp/feedback.py`의 graph cache는 최종 `.pt`와 receipt를 별도로 쓴다. 불완전 pair를 정상으로 받아들이지 않는 guard는 옳다. 그러나 자신의 중단으로 남은 상태를 안전하게 완성할 경로가 없는 것은 별도 운영 결함이다.

### 6.2 저장 transaction과 stage transaction은 다르다

```text
검증된 입력 digest + 고유 attempt
  → 자기 staging에 전체 artifact 작성
  → payload/의미/hash 검증
  → no-clobber commit 발행
  → 최종 artifact 재검증
  → stage journal 완료 기록
```

`staging → rename`만 추가해서 끝내면 안 된다. 현재 verifier는 최종 sidecar 경로를 참조하므로 **payload 검증과 최종 경로 lookup을 분리**해야 한다. 또한 두 파일 `.pt`/receipt를 각각 rename하면 그 사이 crash window는 남는다.

권고는 immutable generation 또는 content object들과 **하나의 commit manifest**다. reader는 commit manifest가 가리키는 완전한 generation만 읽는다. 같은 filesystem의 자기 staging에서 작성하고, 지원되는 환경에서 flush/fsync 및 directory durability를 적용한다. 최종 파일이 이미 있으면 기존 사용자 파일을 `os.replace`로 덮어쓰지 않고 no-clobber로 처리한다. NFS에서는 실제 서버의 publish/read semantics를 시험해야 하며 fsync만으로 모든 장애 복구를 보장한다고 말하지 않는다.

### 6.3 모든 실패 상태의 처리

| 발견 상태 | 필요한 동작 |
| --- | --- |
| 아무 artifact 없음 | 검증된 동일 입력으로 새 attempt 시작 |
| 본인 attempt의 미완성 staging | 원본 보존, live owner 검사 후 별도 staging에서 필요한 artifact만 재생성 |
| `.pt`만 있거나 receipt만 있음 | 정상 재사용 금지. legacy orphan은 소유권/입력 recipe 증거 없이 자동 승인하지 않음 |
| 완전 artifact, journal 미완료 | 현재 입력·hash·verifier 재검증 후 journal을 reconcile. 학습/생성을 무조건 반복하지 않음 |
| 동일 계약의 완전 artifact | 엄격한 idempotent reuse와 증거 기록 |
| 다른 계약/손상 artifact | 기존 보존, 원인과 새 namespace 필요성을 보고 |
| 동시 동일 publisher | 하나만 no-clobber 발행, 다른 실행은 발행된 내용을 전부 검증 |
| running/소유자 불명 | PID뿐 아니라 host/process start/attempt identity 확인. 모르는 프로세스 종료나 lock 강제 삭제 금지 |

검증 실패나 ENOSPC 때 기존 파일을 지워 guard를 통과시키는 복구는 금지다. 입력 의미가 변했다면 동일 root를 억지로 재사용하지 않는다. 반대로 receipt 없는 일부 파일 때문에 이미 검증된 전처리/GNN 전체를 재생성하지도 않는다.

필수 회귀는 serialization 도중, file sync 전후, commit 직전/직후, 최종 검증, journal 기록 전후 실패 주입과 concurrent publisher다. 기존 파일 SHA와 live owner가 보존되는지까지 검사한다.

독립 CPU DEBUG도 현재 실제 publisher로 확인했다. 쓰기 전 실패는 재시도 가능했지만, JSON 직렬화 도중 남은10바이트 final, late-verifier 실패 뒤 final, 정상 발행 뒤 재호출은 모두 `FileExistsError`였다. 결과는 `C:/Users/lock1/AppData/Local/Temp/hiercp_review_online_fb63c7c5aea747f8a405c923e3a44558/audit_online_debug_results.json`에 보존했다. bank/native training 경계는 명시적인 DEBUG 대역이며 실데이터 검증이 아니다. 정확한 현행 경계는 publisher `:45`, `:103–109`, final basename verifier `custom_trainers/onlinecp_curriculum_contract.py:214–216`, stage executor `tools/run_feedback_experiment.py:514–559`다.

## 7. F5 — 평가 매칭의 정확한 수정 범위

현재 `tools/online_eval_v2.py`는 `valid * bonus + clipped_score`를 assignment objective로 사용한다. threshold 미달 edge의 score까지 secondary objective에 남는다.

threshold 0.25, Dice `[[0.45, 0.24], [0.30, 0.00]]`에서 TP=1인 두 선택 중 버릴 0.24가 개입해 유효 0.30을 선택한다. 올바른 valid-only secondary에서는 유효 0.45를 선택한다. 이 반례는 실제 가능한 component voxel/intersection 수로 재현했다.

독립 진단의 네 row/column 순열에서도 같은 현상을 확인했다. 합성 GT voxel 수100/200, prediction100/200, intersection `[[45,36],[45,0]]`다. voxel volume5 mm³를 주면 두 GT 지름은 약9.847/12.407 mm라 ≤10 mm recall 귀속이 바뀔 수 있음을 구체화한다. 실제 환자 결과가 얼마나 바뀌는지는 재평가 전에는 알 수 없다. 현행 objective 위치는 `tools/online_eval_v2.py:392–402`다.

수정 설계는 `secondary = where(valid, clip(score,0,1), 0)`이고 primary cardinality를 secondary 전체보다 우선하도록 bonus의 bound를 명시한다. invalid assignment는 최종 metric에도 들어가지 않아야 한다. rectangular/empty matrix, zero intersection, threshold 경계, 같은 TP·다른 품질, 서로 다른 병변 크기 구간을 검사한다.

정확히 같은 유효 quality의 여러 최적 matching은 원래 모호할 수 있다. geometry 기반 canonical tertiary ordering 또는 ambiguity 기록을 정하고, 실제 품질 차이를 덮는 임의 epsilon이나 작은 병변을 유리하게 만드는 tie-break를 넣지 않는다. 완전 대칭에서 불가능한 환자/ID 독립의 유일 matching을 약속하지 않는다.

영향은 어떤 GT가 검출됐는지, GT별 matched quality와 크기별 recall 귀속이다. 이 반례의 전체 TP는 유지된다. whole-volume Dice나 별도 `max_dice_match()`까지 전부 틀렸다고 확대하지 않는다.

evaluator/metric 정의를 새 버전으로 발행한다. 검증된 기존 prediction은 새 output directory에서 재평가할 수 있다. 과거 evaluation JSON/Markdown을 덮어쓰거나 같은 metric version으로 결과만 바꾸지 않는다.

## 8. F3 — 지원하지 않는 offline full의 조기 차단

`run.py full`은 앞선 준비·학습·생성 후 `tools.nnunet all`로 진입한다. 이 경로는 baseline 뒤 `hierarchical_copy_paste`를 시도하지만 해당 `train_one`은 fold-specific validation 독립성을 증명하지 못해 의도적으로 거절한다.

이 누수 방지 guard를 제거하면 안 된다. 수정 대상은 지원 불가능한 전체 경로를 지원한다고 설명하는 entrypoint/docs와 늦은 실패다.

최상위 preflight에서 unsupported offline full을 **파일 생성·baseline 학습 전에** 거절하고, 지원되는 fold-specific feedback/paired 경로를 안내한다. baseline-only는 실제 지원되는 범위로 별도 유지할 수 있다. 기존 명령을 새 실험으로 조용히 redirect하지 않는다. 테스트는 금지 경로에서 subprocess/training/artifact writer가 한 번도 호출되지 않는지를 검증해야 한다.

## 9. F4 — Difficulty GNN 자원·캐시·병목 설계

### 9.1 현재 확인한 사실과 잘못 확대하면 안 되는 사실

`hiercp/feedback.py::_calibrate`(476–578 부근)는 batch size별로 `entries[:size]`를 측정한다. 실제 update는 전체 observed entry, predict는 전체 bank를 사용한다. **학습 데이터를 prefix로 잘라 버리는 결함이 아니라 승인할 자원 범위보다 측정 표본이 좁은 문제**다.

현재 측정에는 실제 BCE/backward/AdamW probe가 있고, 같은 프로세스에 nnU-Net network/optimizer가 남아 있어 이 resident VRAM은 allocator peak에 포함된다. “nnU-Net 없는 상태만 측정한다”는 비판은 틀리다. epoch 끝에서 segmentation gradient는 먼저 해제한다. 그러나 augmenter/prefetch queue가 남을 수 있고, host RAM·cgroup·전체 source/128개 후보 크기·최악 batch·모든 optimizer state의 상한을 현재 방식으로 승인하지는 못한다.

probe optimizer는 실제 optimizer를 CPU에 둔 상태에서 별도로 만들며 lr=0이다. 모델 hash와 RNG 복원을 검사하는 장치가 있다. 이를 실제 optimizer를 calibration으로 학습시킨다는 결함으로 오해하지 않는다. 다만 현재 prefix에서 사용되지 않은 relation의 moment가 probe에 생성되지 않을 수 있어, 이전 epoch에서 누적된 실제 Adam state 전체보다 작게 측정될 가능성은 별도 budget으로 처리해야 한다.

또한 repeat마다 새 DataLoader를 만들므로 `persistent_workers=True`여도 repeat 간 동일 worker를 계속 사용하는 측정은 아니다(`feedback.py:425–431`, `:503–509`). worker 시작·hash·read·collate가 현재 throughput에 섞인다. cold/warm 구분 없이 이 값을 GPU 계산 처리량으로 해석하면 안 된다.

`BankGraphProvider._build`는 source별로 case/region 준비를 반복하며 `cache_dir=None`을 사용한다. graph cache에서도 SHA 검사와 일반 `torch.load`가 반복된다. 동일 환자의 다중 source에서 공유할 준비·I/O가 남는다. 실제 서버 병목의 시간 비율이나 실제 OOM 발생까지 이번 진단으로 확정하지는 않았다.

### 9.2 보존하면서 효율화할 경계

1. **case-level single-flight:** 같은 환자의 raw/geometry/region 준비는 동시 source 작업 사이에 한 번 수행하고 검증된 결과를 공유한다. RAM 예산에 맞춘 cache/mmap과 명시적 eviction을 사용한다. source별 종양/후보 graph는 의미가 다르면 별도로 유지한다.
2. **기존 region cache 재사용:** 기존 metadata는 path/mtime까지 비교하므로 Medical/Data와 nnU-Net raw copy가 byte-identical이어도 경로만 넘기면 거절될 수 있다. 원래 metadata를 고치지 않고 SHA, shape, spacing, labels, clip, seed, graph semantics가 같은 것을 증명하는 새 content-alias receipt가 필요하다.
3. **feedback 전용 storage adapter:** quality cache와 feedback의 `binding/entry_id/sample` wrapper가 다르다. quality inventory/mmap 코드를 그대로 호출하지 않고 format·shared tensor storage를 이해하는 adapter를 둔다.
4. **mmap/witness:** 시작 시 필요한 전체 무결성 검사를 하고 검증 witness/stat 변경 감지를 통해 같은 immutable 파일의 불필요한 전체 재읽기를 줄인다. hash 검증 자체를 없애거나 변경 파일을 무조건 재사용하지 않는다.
5. **128개 유지:** source별 canonical graph와 후보 materialization을 공유/streaming하되 전 후보 forward/loss 의미를 보존한다. quality의 8-candidate worker 설정을 128-candidate workload의 측정 근거로 사용하지 않는다.

### 9.3 전체 자원 inventory와 실제 측정

사전 inventory에는 모든 source의 node/edge 분포, candidate count, unique backing storage bytes, 두 view/materialization, graph decode/collate 임시 메모리, worker·prefetch·pinned copy, nnU-Net queue, GNN parameter/gradient/전체 Adam state, hash/state_dict/checkpoint 직렬화 피크를 포함한다. RAM은 host `available`뿐 아니라 cgroup limit/current와 scheduler allocation을 함께 확인한다. NFS의 free space와 quota/실제 writing throughput도 별도 기록한다.

predict는 전체 bank에서 최대/혼합 graph batch를 실측할 수 있다. update는 **그 epoch의 실제 관측 label이 있는 entry 집합** 안에서 최대/혼합 batch를 실측해야 한다. 미관측 source를 측정에 넣으려고 가짜 difficulty label을 만들면 연구 목표를 바꾸는 것이다. 미관측 대형 source의 host 입력 상한은 inventory로 검사하고, 다음 epoch에 처음 관측되면 해당 update workload를 재승인한다.

후보 physical batch size를 실제로 여러 개 측정한다. 허용 메모리에서 처리량이 가장 좋은 값을 선택하며, physical/effective batch와 accumulation을 따로 기록한다. 크기 편차는 full-data bucket/dynamic batching으로 다루고 모든 source가 처리됐음을 증명한다. 모델/graph/후보/epoch 축소나 큰 graph skip을 해결책으로 넣지 않는다.

측정은 nnU-Net resident 상태에서 진행한다. cold preparation·cold Adam allocation과 warm repeated update/predict를 분리하고 CPU/RAM/cgroup, allocated/reserved/외부 VRAM, H2D, forward/backward/optimizer, checkpoint/hash, graphs/s 및 candidates/s를 기록한다. 기존 nnU-Net prefetch와 GNN worker가 동시에 차지하는 메모리도 실측한다. 사용자에게 할당된 GPU만 사용하고, 여러 GPU가 실제 할당돼 있으면 독립 case/preparation 또는 data parallel을 검토한다. 보이는 GPU 전체를 마음대로 점유하지 않는다.

현재 train augmenter 정리는 다음 epoch 시작(`nnUNetTrainer_OnlinePairedCP.py:867–885`), validation queue 교체도 다음 validation 시작(`nnUNetTrainer_OnlineCPCurriculum.py:302–309`)이다. 선택지는 완료된 epoch의 자기 augmenter를 정상 정리한 뒤 기존 epoch/thread seed·warm-up discard 규칙을 그대로 재현하거나, 그대로 둔다면 아직 채워질 queue까지 자원에 예약하는 것이다. 전자는 성능 변경이 schedule을 바꾸지 않는다는 검증이 필요하며, 광범위 process 종료를 사용하지 않는다.

mmap/cache eviction도 batch와 prefetch가 backing storage를 참조하는 동안 강제 close하면 안 된다. active lease가 끝난 뒤 해제하고 shared tensor의 in-place 변경이 없는지 검사한다. serialization bytes SHA와 deterministic tensor/semantic hash를 분리해 압축·pickle 표현 변화와 실제 graph 의미 변화를 구별한다.

### 9.4 calibration과 epoch state의 불변성

probe 전후 모델 parameter/buffer, 실제 optimizer state, caller Python/NumPy/Torch/CUDA RNG, 학습 step counter가 보존돼야 한다. CUDA OOM뿐 아니라 CPU allocation, I/O, serialization 실패 시 복원/실행 잠금도 검사한다. calibration 후 성공한 척 다음 step을 진행하지 않는다.

현재 CPU DEBUG의 실제 calibration→update→predict 테스트 1개는 통과했다(7.792초). 200,253 parameter의 명시적인 작은 DEBUG 구성으로 RNG isolation, 전체 parameter 연결, 중복 epoch 거절을 확인한 것이며 production128/실제 GPU/multiworker/native checkpoint 검증은 아니다.

### 9.5 자원 수정도 현재 checkpoint를 막을 수 있음

`feedback.py`의 graph binding에 코드 SHA가 포함되고 runtime identity에는 config/worker 등이 묶인다. trainer도 `hiercp/*.py` 전체 SHA를 runtime identity로 검사한다. 따라서 캐시 helper나 worker 최적화만 해도 기존 native checkpoint resume이 거절될 수 있다.

설계상 **semantic graph/model/state identity**와 **execution/resource revision**을 분리해야 한다. graph, logit, loss, gradient, optimizer state, RNG 경로가 동등함을 검증한 변경에만 명시적 migration receipt를 제공한다. 문서나 전체 Git HEAD가 바뀌었다는 이유로 동일 의미의 모든 산출물을 무효화해서도 안 된다. 반대로 실제 의미 변경을 “병목 수정”으로 포장해 old hash를 바꾸거나 guard를 끄지 않는다.

## 10. F6 — fresh/recovery/upgrade 재개와 native checkpoint

ordinary fresh는 recovery/upgrade와 같은 durable journal 경로가 아니다. 기존 root를 fresh로 다시 실행하면 거절하고, 기존 `--resume-experiment`도 원래 recovery/upgrade 조건을 요구한다. “어떤 실행이든 같은 인자로 재개 가능”은 현재 사실이 아니다.

독립 DEBUG에서 실제 `build_plan/execute_plan`의 첫 stage 경계에 실패를 주입했다. launch plan/runtime은 남고 journal은 없었으며, 일반 재시도는 `FileExistsError`, `--resume-experiment`는 recovery identity를 요구하는 `ValueError`였다. 이는 실학습 중단이 아니라 실행 state-machine 반례다. 현행 분기는 runner `:687–692`, `:747–751`이다.

새 설계는 fresh/recovery/upgrade의 **입력 검증과 허용 작업은 구별하되 stage execution/state machine을 공유**한다. stage는 planned → running(attempt/source/input digest) → artifact committed → evidence verified → completed로 구분한다. 실패와 모호한 running은 별도로 기록한다.

- artifact 성공/완료 journal 누락이면 증거를 재검증해 reconcile한다.
- 학습이 시작됐다면 정확한 native checkpoint 재개만 허용한다. checkpoint가 없거나 의미가 다르면 새 학습으로 fallback하지 않는다.
- 과거 journal 없는 fresh root는 새 schema로 이름만 바꿔 자동 승인하지 않는다. 생성 출처/완료 증거를 독립 감사한 import/recovery 경로가 필요하다.
- Python 실행 경로가 base/nnunet으로 바뀐 경우 등은 실제 dependency identity 문제와 단순 path alias를 구분한다. current plan과 저장 plan 차이를 항목별로 출력해야 하며 checksum을 덮어써 통과시키지 않는다.

native checkpoint 저장에는 이미 좋은 경계가 있다. `custom_trainers/nnUNetTrainer_OnlineCPCurriculum.py`의 저장(400–444 부근)은 complete train+val epoch만 허용하고 자기 temp → `torch.save` → flush/fsync → atomic replacement를 사용한다. 이를 graph cache의 직접 final write와 혼동하지 않는다.

남은 검증은 복합 state restore다. 현재 일부 extension 검증 후 nnU-Net/optimizer/current_epoch를 먼저 복원하고 나중에 feedback/GNN의 strict identity/state를 복원한다. GNN 검증 실패 시 예외로 중단되지만 이미 in-memory 일부가 바뀌었을 수 있다. **shadow validation 또는 새 trainer instance에서 전체 상태를 먼저 검증하고 실행을 잠근 채 restore**해야 한다. malformed GNN identity/shape/optimizer/RNG, 중간 epoch, validation 미완료, mixed epoch state를 주입해 정상 실행을 막는지 검사한다.

실제 서버에서 uninterrupted와 epoch checkpoint resume의 다음 epoch를 비교해야 한다. event/source/appearance, target 선택, EMA 관측 수, Difficulty snapshot age/state, optimizer step, 각 RNG, 출력/gradient의 허용 수치 오차를 기록한다. 이 검증을 CPU smoke 통과로 대신하지 않는다.

## 11. F7 — 평가 출처, 대조군, support와 환자 독립성

### 11.1 이미 있는 것과 없는 것

runner에는 final 250-epoch checkpoint와 bank identity를 확인하는 증거 검사가 있다. 따라서 “학습 완료 증거가 전혀 없다”는 표현은 부정확하다. generic evaluator도 trainer를 지정할 수 있다.

그러나 현재 runner 끝에 **feedback 결과 → 그 checkpoint에서 생성된 prediction → paired 평가/통계 → 최종 완료 receipt**가 이어지지는 않는다. prediction 파일 hash만으로 어느 checkpoint를 실제 로드해 생성했는지는 증명할 수 없다.

필요한 연결은 checkpoint SHA/epoch/arm/bank/policy → 실제 native inference command·로드한 checkpoint·plans/reader/axis/precision/ensemble/postprocess → 입력 case SHA와 prediction SHA → GT/split/patient mapping → evaluator/metric version → paired 통계/완료 receipt다. case 누락·추가·중복, shape/affine/label 오류, 잘못된 fold, best/final 혼동을 거절한다.

기존 prediction의 출처가 불명확하면 hash를 새로 계산해 과거 실행을 입증했다고 하지 않는다. 검증된 checkpoint로 새 directory에 inference를 다시 수행하는 방법이 있다. 이것은 모델 재학습과 다르다. 기존 결과는 유지한다.

### 11.2 Full 대 Basic의 해석

| 비교 | 알 수 있는 효과 |
| --- | --- |
| 현재 Full vs Basic(all128 hard-valid uniform) | quality gate + feedback curriculum의 묶음 효과 |
| Full vs 동일 quality gate + uniform | feedback package의 추가 효과 |
| Full vs 동일 gate/정책의 EMA-only | Difficulty GNN의 추가 효과를 분리하는 데 필요 |

추가 arm은 설계 제안이지 이번에 실행하거나 사용자 승인 없이 기존 Basic 정의를 바꾸는 작업이 아니다. 별도 arm identity와 전체250 epochs, 같은 공유 전처리/합법 source/event/appearance schedule이 필요하다. 후보 선택이 다르므로 augmentation 뒤 support 생존율까지 같다고 주장하지 않는다. L0/L1/L2 ablation과 fold/seed 확대도 기존 계약을 보존하고 실제 주장에 필요한 비교를 별도 승인·계획한다.

### 11.3 difficulty의 정확한 의미

현재 값은 현재 nnU-Net이 실제 받은 crop/augmentation에서 **남은 CP attribution support**에 대한 pre-update 오차다. 전체 원본 병변, crop 밖 component, CP의 생물학적 타당성, 미래 epoch의 절대 난이도와 동일하지 않다.

raw full-source/native-new/crop/augmentation support의 voxel 수와 availability/status를 함께 보고한다. 크기별·epoch별·선택 정책별 관측률, 마지막 관측 age, predicted-only 비율을 포함해 missingness 편향을 드러낸다. unavailable을 error 0으로 채우거나 survival 조건을 몰래 바꾸지 않는다. 원본 전체 병변 난이도를 목표로 바꾸려면 별도 target contract가 필요하다.

case-ID 분리 검사는 실제 환자 단위 독립성을 자동 보장하지 않는다. 한 환자의 여러 scan/ID가 있는 데이터면 환자 grouping mapping과 split 생성 증거가 추가로 필요하다. 자료가 없다는 이유로 실제 누수가 있었다고 단정하지도, 없었다고 승인하지도 않는다.

## 12. 무엇을 재사용하고 무엇을 새 버전으로 만들어야 하는가

**전처리 공유와 조건이 다른 GNN/bank의 무조건 재사용 금지는 모순이 아니다.** 산출물마다 실제 의존하는 의미가 다르므로 묶음을 나눠야 한다.

| 산출물 | 재사용 조건 / 처리 |
| --- | --- |
| 원본·환자/split 목록 | 데이터·환자 mapping·split 계약 동일하면 보존/공유 |
| nnU-Net planning/preprocessing | 데이터·plans·native resampling 의미 동일하면 공유. F1이나 evaluator 수정만으로 재계산할 이유 없음 |
| patient region cache | partition/descriptor/입력 의미 동일하면 공유; path alias는 독립 receipt로 검증 |
| context prototype | fitting inner-train ID·descriptor·seed·cluster 의미 동일하면 공유 가능 |
| L0 geometry/raw source/candidate payload | 원본·전체 source geometry·CP 조건·정확한 후보 순서/중심이 동일하면 공유 가능 |
| Quality GNN checkpoint | F1 정보 구조 변경 후 새 설계 검증 완료 모델로 재표기 금지. 기존 비교 모델로 보존 |
| frozen quality score | 새 architecture/model이면 재산출하여 별도 score overlay/receipt에 발행 |
| typed bank | geometry/data와 quality 계약을 분리하되 전체128/zero 상태를 exact 검증. 기존 index/hash를 편집해 새 의미로 승인하지 않음 |
| Difficulty/native Full checkpoint | policy/정보 구조 의미 변경이면 동일 실험 재개 아님. 기존 보존, 승인된 새 arm/학습 필요 |
| Basic checkpoint | hard-valid geometry/event/source/appearance/augmentation/plans/seed/epoch trajectory와 출처가 모두 동일한 경우만 명시적 재사용 검토. 이름만 같다고 재사용 금지 |
| 기존 predictions | checkpoint lineage 확인 시 새 평가 버전에서 재사용. 불명확하면 새 inference |
| 기존 metrics | 정의가 바뀌면 새 directory에서 재평가, 원본 보존 |

raw case 수십 GB를 score version마다 물리 복사할 필요는 없다. immutable 공통 object 또는 검증된 same-filesystem hardlink import를 설계할 수 있다. 현재 bank의 상대경로/경계 guard를 `../`로 우회하지 않고 실제 content import receipt를 만든다. 사용 가능 filesystem 기능과 디스크 여유는 서버에서 확인한다.

resource-only 수정은 의미 동등성을 증명한 migration 또는 검증된 구 runtime 고정으로 재개할 수 있게 설계한다. architecture 변경은 새 version의 full-scale 학습이다. warm start를 사용하더라도 원래 실험을 그대로 resume한 것처럼 보고하지 않는다.

## 13. 구현 순서와 완료 승인 조건

| 단계 | 구현 대상 | 반드시 통과할 검사 |
| --- | --- | --- |
| A | atomic/idempotent 계약·cache 발행 및 stage reconcile | 모든 crash window, concurrent publish, 동일/다른 입력, orphan/live owner, 기존 파일 보존 |
| B | valid-only evaluator + offline 조기 guard | 실제 matching 반례/경계값/크기별 귀속, unsupported full에서 학습 호출 0회, metric 새 버전 |
| C | semantic/execution identity 및 공유 준비 cache | old metadata 보존, 동일 content alias, 실제128 input inventory, resource-only parity/migration |
| D | source 정보 계약·L1/L2 경로·고정0 열 정리 | 숫자/주소 topology 분리 counterfactual, 허용 context 반응, full batch isolation, forward/loss/gradient/optimizer 연결 |
| E | 실제 nnU-Net resident Difficulty 자원 측정 | 전체 source inventory, 관측 update/전체 predict의 worst/mixed batch, host/cgroup/worker/prefetch/VRAM/throughput, 모든128개 유지 |
| F | unified fresh/recovery/upgrade와 native resume | 완전 epoch state, interrupted/uninterrupted 비교, checkpoint 없는 fresh fallback 금지 |
| G | provenance-bound inference/evaluation/통계 | checkpoint→prediction→GT→metric 전체 연결, 누락/중복/다른 fold 거절, 새 결과 폴더 |
| H | 승인된 full-scale 연구 비교 | 전체 cohort/40·250 epochs, 원래 Basic 보존, 필요한 대조군·fold/seed·환자 grouping 명시 |

독립적인 A/B 작업은 병렬 검증할 수 있다. C의 identity 설계와 D의 새 architecture 경계를 먼저 정하지 않고 각각 patch만 더하면 다시 재개 충돌이 생길 수 있다. E/F는 코드가 있다는 사실이 아니라 실제 서버 측정/재개 증거가 완료 기준이다. 어려운 부분을 “추후 TODO”로 처리하고 파이프라인 전체 완료를 선언하지 않는다.

## 14. 검증 상태와 미실행 항목

### 이번에 확인한 것

- 제공 문서/대장 전체, 기준159파일 Git SHA 및18개 증거 checksum 대조.
- 현재 코드에서 웹의8개 정상/실패 시나리오 재현.
- 실제 PyG 상위 계층의 source-host 개입과 점수 전달, 고정0 parameter cell 진단.
- 실제 CPU Difficulty calibration/update/predict/RNG isolation DEBUG 1개 통과.
- raw resampling/native trainer/streaming/feedback metric/storage/shared-source 관련 추가 CPU 회귀 결과는 아래 실행 기록으로 구분한다.

### 실행 기록

추가 회귀 첫 실행은 외부 nnU-Net의 hostname 탐색 과정에서 Windows 문자열 decoding 오류로 **테스트 discovery 전 실패**했다. 프로젝트 모델 결함이나 테스트 성공으로 집계하지 않았다. vendor 코드·시스템 설정은 수정하지 않았고, 공식적으로 제공되는 `nnUNet_n_proc_DA` 환경 입력을 해당 DEBUG 프로세스에만 지정해 재실행했다. 이 값은 production worker 설정을 바꾸지 않는다.

두 번째 실행은 53개 항목 중2개가 테스트 파일 간 sibling import 경로 때문에 오류였다(10.978초). production 코드를 고친 것이 아니라 기존 테스트 방식에 맞게 `tests`를 해당 프로세스의 import path에 넣고 아래6개 모듈을 다시 실행했다.

| 추가 CPU 회귀 모듈 | 범위 |
| --- | --- |
| `test_raw_cp_resampling_debug` | native resampling 및 raw cubic/label/support parity |
| `test_raw_cp_trainer_native_debug` | 실제 nnU-Net API를 사용하는 합성 trainer 통합 경계 |
| `test_edge_attention_streaming_debug` | streaming forward/backward 및 hierarchy checkpoint/dropout/optimizer 경계 |
| `test_online_cp_feedback_metrics` | CP surviving-support metric/status |
| `test_raw_bank_storage_debug` | raw bank 저장/typed 계약 |
| `test_raw_bank_shared_sources_debug` | source payload 공유 및 검증 |

최종 실행은 **55개 모두 성공, 23.014초, exit0**이었다. 출력의 native parity16개 비교에서 CT 최대 절대 오차0, label/support exact equality가 확인됐다. 별도의 Difficulty RNG/calibration 테스트1개와 합쳐 이번 선택 회귀는 **56개 성공**이다. 재실행 전 오류를 성공으로 숨기거나 중복 성공 개수를 더하지 않았다.

실데이터 root가 지정되지 않았다는 nnU-Net 안내와 matplotlib 기본 cache 경로 권한 경고도 있었다. matplotlib은 자기 임시 cache를 사용했고 최종 회귀는 성공했다. 이것은 실제 데이터가 없어도 production을 승인했다는 뜻이 아니다. 의존성/시스템 설치/기존 설정을 수정하지 않았다.

로컬 검사 환경은 PyTorch2.6.0+cpu/PyG2.6.1, CUDA 없음, 논리 CPU16개다. 검사 종료 후 RAM snapshot은 총18,918,256,640 bytes, available2,227,011,584 bytes였다. 이는 종료 후 상태이며 test peak RAM이나 원격 서버 용량이 아니다. 제공된 두 원본 Markdown을 ZIP 내부본과 마지막에 다시 비교해 불변임을 확인했다. tracked 파일의 Git diff도 없었다. 새 산출물은 이 검수·설계 문서와 명시적인 DEBUG 진단 스크립트다.

### 실제 서버에서 아직 확인하지 않은 것

- 전체 의료 cohort의 새 bank 생성 및 전 source/128 candidate 무결성.
- 실제 GPU, NFS, cgroup, nnU-Net resident 상태의 Difficulty 자원/worker/throughput 측정.
- 새 정보 계약의 full40-epoch Quality GNN과250-epoch Full/Basic 학습.
- native checkpoint interruption/resume 수치·schedule 동등성.
- source shortcut에 대한 기존 학습 checkpoint의 민감도와 실제 성능 영향.
- 새 evaluator 정의의 전체 prediction 재평가, provenance와 paired 통계.
- 실제 환자 단위 독립성과 추가 대조군의 연구 결과.

이 항목들은 설계·승인 조건을 정의했지만 실행 완료가 아니다. 모델·데이터 축소, 가짜 입력을 production으로 사용, 오류 무시로 대신하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. production 최적값 승인은 미실행이다.
- [x] 로컬 CPU/RAM·CUDA 가용 범위와 DEBUG 자원 한계를 확인했다. 원격 실험 자원은 미측정이다.
- [x] 자원 문제는 모델 축소보다 메모리·I/O·계측 범위 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 production으로 사용하지 않았다. 합성 DEBUG 경계는 명시했다.
- [x] 핵심 forward/loss/gradient/optimizer 경로를 검토하고 일부 실제 DEBUG 연산으로 확인했다. 전체 production 실행 승인은 아니다.
- [x] 실제 실행 설정과 미구현 설계를 구분했다.
- [x] smoke/CPU DEBUG와 전체 학습·평가를 구분했다.
- [ ] 위 수정 설계의 production 구현 완료.
- [ ] 실제 서버 전체 학습·재개·평가 검증 완료.
