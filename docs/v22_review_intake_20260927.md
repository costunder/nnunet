# v2.2 독립 검토 3문서 독해 및 현재 코드 대조 기록

작성일: 2026-09-27. 사용자 요청: 첨부 세 문서를 누락 없이 읽고 복잡한 내용을 자세히 확인한다. 이번 작업은 독해·대조이며 production 코드 수정, 새 학습, 서버 실행, Git push는 하지 않았다.

## 읽은 원문과 확보 범위

| 문서 | 첨부 디렉터리 ID | 전체 읽은 줄 수 | SHA256 |
|---|---|---:|---|
| 검토 요약 및 발견 사항 | 3bdc3ea7-8cc9-4b66-9682-21b57b3768a4 | 529 | 7f4abb5bef05aedd3accbe93b113a9c949943e3b2b2159d4d7a53a4f9bb9c376 |
| 상세 독립 검토 보고서 | 5077b6b7-9f67-4521-8b1f-9300792119a4 | 399 | 50c9599149e4f76c385872d29d1c94b9b210e615827c539261af5df263925e39 |
| 수정·검증 지시의 설명 | bf9f9aee-3318-4c6f-9b7a-75e1b429e6ae | 359 | 2861e42b8117a03239de8c8636cc8c2e47e93989b9c97368e78e4a7c23b96158 |

각 파일명은 `붙여넣은 텍스트.txt`이며 총1,287줄. 처음 합쳐 읽은 출력이 잘려서 이후 각 파일을 연속 구간으로 나눠 전부 읽었다. 상세 보고서의 부록 검사 ID와 제3문서의 마지막 시작 프롬프트까지 포함한다.

원문 안의 sandbox 링크는 다른 GPT 세션의 파일이다. 현재 첨부와 저장소에서는 `CODEX_REPAIR_SPEC.md`, `ACCEPTANCE_MATRIX.json`, `CODEX_START_HERE.md`, 검토 재현 ZIP, 수정 지시 ZIP 원본을 찾지 못했다. **60개 검수 항목 전체를 읽거나 실행했다고 주장하지 않는다.** 본 기록은 제공된 본문 기준이며 원본 검수표를 임의로 재구성해 대체하지 않는다.

## 소스 기준 대조

현재 HEAD와 검토 기준은 모두 `94596b9f6945015dfa14f3721d6763ce262d4a9d`. 전달 ZIP SHA256은 `2ce54b461c216d8e86945481ffadb15faa64872362b770faa721f1c0e832382b`. 이번에 manifest544파일을 현재 파일과 다시 비교했고 해시 불일치0이다.

manifest는 tracked413개/untracked131개를 포함한다. tracked 파일의 raw Git blob 대비 바이트 차이184개는 Git-clean checkout에서도 지정된 CRLF 변환으로 발생한다. 따라서 `matches_base_commit=false` 자체를 임의 코드 수정의 증거로 보지는 않는다. 반대로 untracked131개까지 전부 GitHub에 올라갔다고 해석해도 안 된다. 현재 tracked 변경은 기존 사용자 `code.txt`뿐이고 전달본에서 제외됐다.

physical32 증거가 현재 revision 전체의 검증이라는 주장도 성립하지 않는다. 상세 보고서가 지적한 trainer/recommendation의 현재 해시는 원본 보고서와 일치한다. 검사별 당시 해시와 최상위 최종 해시는 구분해야 한다.

## F01–F09 누락 없는 대조표

| ID | 내용 | 이번 확인 수준 | 후속 수정·검사에서 지킬 사항 |
|---|---|---|---|
| F01/P1 | process report가 calibration list를 dict로 강제 변환 | 원본 중첩 report 함수를 AST 그대로 실행하고 합성9필드 payload로 ValueError 재현. trainer의 list writer 및 reuse와 DEBUG wrapper 우회도 코드 확인 | list 스키마 유지, mapping에만 metadata, 입력 불변 및 충돌 거부. 실제 process 새 실행/reuse/저장 실패/최종 산출물까지 검증 |
| F02/P1 | spacing 변환 후 explicit anchor 유실 | 원본 regrid 함수+합성 donor 실행: shape11³, anchor6³, midpoint5³. pair_record→mask-only→spatial 중심정렬 경로 확인 | graph/filter/paste 좌표를 explicit anchor로 결속. +1 하드코딩·centroid 이동·crop 금지. 영향받는 geometry/cache/embedding 계약 갱신 |
| F03/P2 | 다중 양성 동점에서 Recall/MRR 오류 | 현재 ranking_metrics 직접 실행: scores[1,1,0], truth[1,1,0] → ranks[2,2], R@1=0, MRR=.5 | GT와 독립된 candidate key로 단일 순서, 실제 ranks[1,2], R@1=.5, MRR1. positive-first record 순서 금지. best 기준은 pair loss 유지 |
| F04/P2 | rolling 재개가 외부 best_path에 의존 | trainer finalization 및 saver/state GPU 복원 코드 확인. 이번에 파일 이동/삭제 실험은 미실행 | CPU 전용 불변 best snapshot 또는 검증된 명시적 의존성. best 누락 시 latest 대체 금지. best weights와 final memory 일치 |
| F05/P2 | 추천 artifact validator 불충분 | 현재 loader gate와 final writer 대조. 이번에 변조 artifact inference 실험은 미실행 | resume/epoch/final/DEBUG 유형별 공통 validator. runtime·identity·donor split·coverage·중복·dtype·유한성·모델-memory 결속 |
| F06/P2 | query_group과 recipient identity 결속 없음 | recommend에서 전달값을 바로 grouped_support에 넣는 경로 확인 | 검증된 manifest로 group 결정, caller 값은 대조. 양쪽 support 제외 유지. 신규 case 독립성을 문자열만으로 가정 금지 |
| F07/P2 | scored geometry / filter mask / 실제 paste 변환이 별개 | identity candidate 생성과 추천 API의 별도 mask/anchor 입력 코드 확인 | 공통 PlacementSpec에 identity/frame/affine/spacing/transform/CT/mask/anchor/hash. 실제 변환 및 nnU-Net 입력까지 좌표 대조 |
| F08/P2 | CT와 GT physical grid 및 finite spacing 검증 부족 | common.load_case의 shape 및 spacing<=0 검사 확인. affine 비교/finite gate 없음 | mismatch 거부. 검증 없이 GT resample 금지. 전체 실제 cohort audit 별도 수행 필요 |
| F09/P3 | 처음부터 빈 후보는 예외, 전부 탈락만 no-op | rank_then_filter/recommend의 nonempty gate 확인 | 정상 빈 proposal/전부 탈락은 원본 유지. 잘못된 모델·identity·NaN·graph·renderer는 오류. RNG draw 유지 |

위 CPU 반례는 원본 함수 본문을 변경하지 않았지만 synthetic 입력을 사용한 분리 검사다. 실제 CT/GPU/전체 process 실행 재현으로 확대 해석하지 않는다. F04–F09의 외부 검토 재현 결과와 이번 정적 확인도 구별한다.

## 설계 위험 D01–D04와 변경 금지 경계

1. **D01 CE 대 rank:** 둘 다 양성 score를 올리는 방향이므로 구조적으로 반대라는 단정은 틀리다. 상대 순위와 전체 관측 분류/무양성 CT 학습의 차이는 존재한다. loss별 L0/L1/L2 norm/cosine과 rank-only/CE-only/joint는 별도 objective/run에서 검사한다. CE 삭제를 버그 수정으로 처리하지 않는다.
2. **D02 pair 가중과 memory 근사:** [P,U1], [U2]의 epoch 손실계수 .5 대1.5는 균등 중복이 아니다. 이번 현재 rank_loss의 고정0점 CPU 검사에서 첫 batch score gradient[-.5,.25,.25], 둘째[-.5,0,.5]를 얻었다. pair 수로 나눠 scalar 합이 맞아도 각 live L0 endpoint gradient가 달라질 수 있다. 중복 pair 제거·loss/2·reference L1/L2 detach 금지. 현 estimator를 보존하고 빈도/가중/graph-size 상관을 측정한 뒤 재가중은 별도 실험한다.
3. **D03 shortcut/fixed donor:** 종양 내부 node가 없어도 CNN 수용영역과 normalization은 본체 정보를 담는다. 현재 비교 s(d_p,r_p)>s(d_u,r_u)는 CP event의 s(d,r_p)>s(d,r_u)와 다르다. 실제 donor 무시나 성능 실패를 이미 입증한 것은 아니다. fixed-donor 전체 후보, donor swap, appearance/geometry 분리, recipient-only, 중심노출 통제, CP 대조군을 분리한다. 중심 가림/donor schedule 변경을 기본에 몰래 적용하지 않는다.
4. **D04 최대 비용 calibration:** group 전체 edge 합 최대가 physical batch 최대 VRAM을 보장하지 않는다. reference 수/support/L2/live range를 포함한 실제 worst-profile 배치와 전체 support로 측정한다. 아직 full-scale OOM이 재현됐다는 뜻은 아니다. graph/model/batch 축소로 우회하지 않는다.

## 기하·순위·checkpoint에서 놓치면 안 되는 세부 조건

- paste voxel = recipient center + transformed mask voxel - explicit transformed anchor. anchor가 tumor 내부일 필요는 없다.
- spacing 변환 자체는 voxel count를 바꿀 수 있다. 동일 변환 결과를 downstream에서 자르거나 이동시키지 않는 것이 보존 조건이다. source branch donor 물리좌표와 recipient virtual footprint를 구별하고 이중 resampling 금지.
- 동점 수정은 하나의 label-independent total order에서 지표를 계산한다. Recall은 positive micro, MRR은 positive-case macro, pair loss는 전체 P×U sum/count. 무양성과 양성만 있는 case 구별, 전체 pair0은 best loss0으로 위장하지 않는다.
- 기존 state 전체가 resume에서 CUDA로 이동된다. best snapshot을 그대로 state에 넣으면 안 된다. CPU payload 분리와 alias 불변성, epoch/metric/run/hash 결속이 필요하다.
- final validator를 rolling에 그대로 적용하면 재개 가능 checkpoint까지 잘못 거부한다. 유형별 필수 상태와 provenance 범위를 명확히 구분한다.
- 검증된 source/runtime migration만 허용하며 geometry 의미가 바뀐 cache/checkpoint를 이름만 바꿔 재사용하지 않는다. 기존 정확한 stride4/양쪽 group 제외/owner 병합/worker guard는 되돌리지 않는다.

## GT와 환자 독립성의 정확한 표현

CT-only 채널이지만 GT 간 union(label1 또는2)으로 graph geometry를 만든다. 따라서 GT가 forward 구성에 전혀 관여하지 않는 모델이 아니다. query 관측 target/overlap을 직접 score에 넣지 않는 것과 구분한다.

검사는 CT·간 union·후보·donor·support를 고정한 채 종양 label만 바꿔 raw score/raw rank는 유지되고 eligibility만 바뀌는지 확인해야 한다. 후보를 label에서 새로 만들면서 이 불변성을 주장하면 안 된다. outer-val inventory/hash를 읽는 것과 optimizer/support로 쓰는 것도 별개다. published_case_only와 patient_independence_verified=False를 유지한다.

## 온라인 CP 연결과 정상 no-op

새 API는 아직 native online 통합이 아니다. bank의 legacy scorer, candidate pool의 종양 선제 제외, native_adapter의 raw argmax 강제, trainer의 scores 재선택, raw renderer와 최종 전처리를 함께 추적해야 한다. adapter의 raw argmax 강제와 trainer 재선택을 현재 코드에서 확인했다.

새 선택 계약: 모델 score+GT 독립 동점 순서 → 모든 후보 scoring 완료 → 전체 true footprint로 eligibility → 최고 eligible 또는None. 제외 후보 score를 -inf로 변조해 기존 argmax를 통과시키지 않는다. renderer bbox/crop 미지원은 부적합 후보로 바꿔 차순위를 재추첨할 이유가 아니다.

정상 no-op은 원본 CT/label 및 RNG draw schedule을 유지한다. 모델/identity/record/NaN/graph/transform/renderer 오류는 숨기지 않는다. CP80%는 시도율이며 실제 붙인 비율과 별도 기록한다. donor 재추첨이나 사용하지 않는 RNG draw 제거도 표본 분포를 바꾸므로 금지한다.

graph→filter→실제 raw paste→동일 nnU-Net transpose/crop/resample/augmentation→최종 trainer input까지 검사해야 한다. 기존 legacy paste 성공을 새 ranking 통합 증거로 제출하지 않는다.

## 허용되는 성능 개선과 검증 경계

case별 전체 P×U sum/count, full mask chunk, generation/device별 정적 index cache, frozen inference support reuse, CPU snapshot 불변 공유, 실제 producer/prefetch 병목 개선을 검토한다. pair sampling/top-k/cap/작은 mask 생략/모델·해상도·physical batch 축소는 우회안이 아니다.

chunk mean을 평균내지 않는다. chunk forward 후 autograd가 모든 중간값을 보존하면 backward 메모리가 충분히 줄지 않을 수 있다. 업데이트 후 train L1/L2 출력을 재사용하지 않는다. 저장 빈도를 몰래 낮추지 않는다. 수학적 동등성과 bitwise exact resume는 별개이며 reduction 순서 변경을 측정·version 관리한다. 위 항목은 아직 측정된 성능 개선이 아니다.

## 완료 단계와 현재 상태

| 단계 | 요구 | 이번 상태 |
|---|---|---|
| G0 | 기준 snapshot·활성 경로·반례 확인 | 부분 확인:544해시 대조, 주요 경로 정적 대조, F01/F02/F03 및 D02 CPU 반례. 공식60항목 원본 미확보 |
| G1 | 실행·기하·지표·artifact·identity 수정 검증 | 접수 당시 NOT_RUN. 후속 코드 수정과 실제 검증은 [수정 기록](v22_review_repairs_20260927.md) 참조 |
| G2 | 실제 process·실제 모델 gradient·pause/resume | NOT_RUN |
| G3 | 전체 support/데이터 coverage·physical batch | NOT_RUN |
| G4 | 새 bank/RPC/adapter/trainer·실제 paste·최종 입력 | NOT_RUN |
| G5 | 전체 학습과 CP 대조 실험의 연구 효용 | NOT_RUN |

독립 검토의29 passed/1 skipped/1 deselected 및21분리 검사는 검토자가 수행한 범위다. 본 작업에서 그대로 재실행하지 않았다. 기존58회귀·32physical DEBUG·8/2 lifecycle·2후보 mask 검사는 전체 support/production main/G5를 증명하지 않는다. 수정 완료 보고에는 실제 revision별 증거·migration·미해결·확인한 실행 명령을 구분해야 한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번은 독해/CPU 반례이며 새 처리량 측정은 하지 않았다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 이번 작업에서 자원 calibration은 실행하지 않았다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. 이번 실행에서 OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다. 합성 CPU 반례를 실제 CT 검증으로 표현하지 않았다.
- [x] dummy, placeholder, random fallback을 production에 추가하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 코드 연결은 대조했지만 이번에 전체 모델 역전파를 재검증하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. production 코드 변경 없음.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
