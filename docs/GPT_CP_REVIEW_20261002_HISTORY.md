# GPT 검토용 구현·실험 진행 이력 — 근거 기준 2026-10-02 KST

자료 정리일: 2026-10-03 KST. 파일명의 날짜는 포함된 최신 실행 근거의 날짜다.

이 문서는 보존 소스, 당시 수정 기록, 복구된 서버 콘솔, 실제 CT DEBUG 결과를 근거로 진행 과정을 정리한다. 이번 문서 작성에서 모델·optimizer·GPU 실행, 서버 접속, 전체 학습 또는 평가를 새로 수행하지 않았다. 과거 문서에 있던 “미구현”은 그 문서 작성 시점의 상태이며, 후속 구현과 정정을 함께 읽어야 한다.

현재 가장 최신의 확인 결과는 **native CNN과 footprint sparse graph의 각 60회 연속 DEBUG 학습, 별도 CT의 관측 위치 순위 평가, 각 한 번의 실제 CP 실행**이다. Train 순위 분리는 학습됐지만, 별도 CT의 상위 순위 개선은 확인되지 않았다. 실제 paste는 성공했으나 추천 품질과 segmentation 효용은 미평가다. 새 graph는 diagnostic-only CP bridge로 연결됐으며 production artifact/export/온라인 trainer 교체는 완료되지 않았다.

근거: [최신 실행 기록](<D:/AI project/nnunet/docs/sparse_cp_recommender_DEBUG_20261002.md:3>), [원시 report](<D:/AI project/nnunet/work/latest_sparse_cp_learning_DEBUG_20261002/report.json>), [집계 JSON](<D:/AI project/nnunet/work/latest_sparse_cp_learning_DEBUG_20261002/compact_summary.json>). JSON에서는 `full_training=false`, `full_evaluation=false`, `production_ready=false`, `production_checkpoint_written=false`, `P_U_GT_changed=false`를 확인했다.

## 1. 버전 이름과 현재 정답 계약

- **v1**은 보존된 기존 HierCP 파이프라인이다. 보존 revision은 `74dcc2cf03d2d40d1f582223321d96004333f661`; 내부 모델 revision `v5`와 파이프라인 번호를 혼동하지 않는다. [manifest](<D:/AI project/nnunet/versions/v1/manifest.json:2>)와 [보존 ZIP](<D:/AI project/nnunet/versions/v1/pipeline_v1_source.zip>)이 있다.
- `hiercp_v2/`와 `versions/v2/`에는 폐기한 view-only v2.0과 후속 cross-patient v2.1의 이력이 함께 있다. 경로 이름만으로 두 구현을 같은 모델로 취급하지 않는다.
- 후속 기록의 **v2.2**, **v2.22**, **v222**, `l0_regions`는 보고 제목·artifact·패키지 이름이 섞여 있다. 실제 encoder와 learning-policy를 확인해야 한다. 예를 들어 `l0_regions` 패키지를 쓰는 최근 서버 run의 L0는 native CNN이며 SAGE/EZ-SP를 사용하지 않는다. [서버 snapshot 설명](<D:/AI project/nnunet/docs/local_cnn_interaction_epoch30_20261001.md:29>)

현재 고정 계약은 **P=원본 CT에서 관측된 적격 종양 anchor, U=기존 미관측 비교 중심**이다. Donor는 조건 입력이며 donor를 바꾼다고 P/U를 바꾸지 않는다. P를 특정 donor의 CP 적합 위치 정답, U를 CP 부적합 정답으로 해석하지 않는다. 제공된 observation GT를 오류로 간주하거나 v1 GT/curriculum으로 되돌리라는 이전 제안은 사용자 정정에 따라 철회됐다. [최우선 정정](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:3>)

## 2. 구현이 어떻게 진행됐는가

| 시점·경로 | 구현 또는 수정 | 당시 확인 범위와 남은 문제 | 근거 |
| --- | --- | --- | --- |
| 기존 v1, 9/19 보존 | 국소 graph L0, patient/region L1, population/prototype L2, feedback/curriculum·online CP 보존 | 보존일은 원래 실험일이 아니다. 옛 학습 완료 checkpoint와 prediction은 현재 handoff에서 독립 재평가하지 않았다. | [PATCH_NOTES:872](<D:/AI project/nnunet/PATCH_NOTES.md:872>), [v1/current 대조:7](<D:/AI project/nnunet/docs/v1_current_comparison_20260925.md:7>) |
| v2.0, 폐기 | 동일 데이터의 두 stochastic view에 K-means label을 두고 view 간 일치로 L2를 구현 | 실제 환자의 독립 label 공간을 정렬하라는 요구와 달랐다. 최종 기준선으로 사용하지 않는다. | [PATCH_NOTES:860](<D:/AI project/nnunet/PATCH_NOTES.md:860>) |
| v2.1, 9/19 | 실제 patient task의 독립 label, 다른 train 환자 관측 T의 L2 전달, target group 제외 | 통계 유사도 teacher는 연구 가정이다. 전체 학습을 시작했다는 기록과 학습 완료는 구분한다. | [PATCH_NOTES:837](<D:/AI project/nnunet/PATCH_NOTES.md:837>) |
| v2.1 실행 정정 | 자기 donor가 있는 case와 없는 case에 비대칭 후보 정책이 적용되는 문제 확인·중단, 이후 공통 train-only donor 경로로 수정 | 초기 source inventory는 RAM admission으로 실패했다. Lazy source 및 무손실 저장을 추가했으며, 실패 기록과 부분 출력은 보존됐다. | [PATCH_NOTES:799](<D:/AI project/nnunet/PATCH_NOTES.md:799>), [PATCH_NOTES:810](<D:/AI project/nnunet/PATCH_NOTES.md:810>), [PATCH_NOTES:816](<D:/AI project/nnunet/PATCH_NOTES.md:816>) |
| v2.2 r1/r2, 9/20 | CNN을 제거하고 원본 CT/통계/기하 node 특징→MLP→기존 GNN 경로 추가 | r1 DEBUG를 r2 완성 증거로 옮겨 쓰면 안 된다. 통계 descriptor teacher·특징 조합의 직접 연구 근거와 전체 objective가 미검증으로 정정됐다. | [PATCH_NOTES:757](<D:/AI project/nnunet/PATCH_NOTES.md:757>), [PATCH_NOTES:779](<D:/AI project/nnunet/PATCH_NOTES.md:779>) |
| v2.2 r3→r4→r5, 9/22 | CT-only CNN 특징으로 변경; 함께 제거했던 기존 L2 복구; 폐기된 tumor-interior node 및 관련 활성 모듈 제거 | “CNN 단순화=L2 제거”가 아니었다. T/U-only 관측 loss에 균등 점수의 반례가 남았고, data-label 단위/T-F-U conditioning은 당시 미검증이었다. | [PATCH_NOTES:708](<D:/AI project/nnunet/PATCH_NOTES.md:708>), [PATCH_NOTES:724](<D:/AI project/nnunet/PATCH_NOTES.md:724>), [PATCH_NOTES:738](<D:/AI project/nnunet/PATCH_NOTES.md:738>) |
| v2.21→v2.22, 9/22 | T/F/U data-label 의미를 geometry gate와 분리. v2.22에서 CNN12/24/32+spatial GATv2 3층128D/4heads, observed-class L1, train-support L2와 CE 연결 | v2.21 relation-only 합성 검사와 v2.22 통합 DEBUG는 다른 범위다. v2.22의 detached train-support L0는 epoch마다 갱신한다. | [PATCH_NOTES:694](<D:/AI project/nnunet/PATCH_NOTES.md:694>), [PATCH_NOTES:680](<D:/AI project/nnunet/PATCH_NOTES.md:680>) |
| 원본 CT / PPR-A* / pooling 검토, 9/23–24 | raw CT 공간 graph, CNN-PPR/A*, 조기 선택·지연 pooling·U-Net형 L0를 순차 검토 | PPR/A* r4 설계는 반려되고 학습이 차단됐다. 과거 대안을 현재 활성 모델로 설명하지 않는다. CNN-only 해석의 불일치도 정정했다. | [PATCH_NOTES:552](<D:/AI project/nnunet/PATCH_NOTES.md:552>), [PATCH_NOTES:570](<D:/AI project/nnunet/PATCH_NOTES.md:570>), [PATCH_NOTES:611](<D:/AI project/nnunet/PATCH_NOTES.md:611>), [PATCH_NOTES:522](<D:/AI project/nnunet/PATCH_NOTES.md:522>) |
| v1 방식 dense L0, 9/24–28 | v1 local graph 방식의 CT-only L0, full-cohort paired runner, 저장/재개·정적 입력/CSR 재사용·실행 병목 수정 | L0를 v1처럼 바꾼 것은 v1의 전체 loss/target/patient-prototype 학습을 복원한 것이 아니다. Query 관측 수와 support 재인코딩 비용이 크게 달랐다. | [PATCH_NOTES:509](<D:/AI project/nnunet/PATCH_NOTES.md:509>), [PATCH_NOTES:516](<D:/AI project/nnunet/PATCH_NOTES.md:516>), [v1/current 대조:15](<D:/AI project/nnunet/docs/v1_current_comparison_20260925.md:15>) |
| Fine GAT→mean GraphSAGE, 9/29 | 동일 fine graph의 관계 convolution을 CSR mean-SAGE로 교체, CNN/readout/L1/L2·3층128D 유지 | 실제 CT DEBUG8pair의 update2.283→1.203초, refresh0.247→0.406초. 계산 비교이며 추천 정확도나 전체 서버 epoch 개선 검증이 아니다. | [GAT/SAGE:7](<D:/AI project/nnunet/docs/l0_gat_sage_comparison_20260929.md:7>), [비용:30](<D:/AI project/nnunet/docs/l0_gat_sage_comparison_20260929.md:30>) |
| Dynamic EZ-SP adapter, 9/29 | 공식 GPU merger, fine-node mass 집계, 13종 quotient, 2+1 GAT 및 두 scale readout | 단위 연결 검사는 통과했다. 실제 CT8pair는 node/edge/bbox/variance admission 실패로 full update 미실행. 거부 시점 메모리를 full-update 절감으로 해석하지 않는다. | [EZ-SP:13](<D:/AI project/nnunet/docs/l0_ezsp_adapter_20260929.md:13>), [실제 실패:37](<D:/AI project/nnunet/docs/l0_ezsp_adapter_20260929.md:37>) |
| Frozen-region+SAGE, 9/29 | 기존 CNN snapshot으로 partition을 미리 고정; downstream의 현재 CNN 특징을 집계해 SAGE 및 L1/L2 학습 | 로컬 partition 출처는 DEBUG saved step4이다. 무작위 대체가 없다는 것과 충분한 partition 학습은 다르다. 거대영역/미미한 2차 병합이 확인돼 사용자 요청으로 단일 scale 및 research-report 실행 경로를 추가했다. | [고정 구조:5](<D:/AI project/nnunet/docs/l0_fixed_regions_20260929.md:5>), [진단:7](<D:/AI project/nnunet/docs/region_partition_inspection_20260929.md:7>), [PATCH_NOTES:209](<D:/AI project/nnunet/PATCH_NOTES.md:209>), [PATCH_NOTES:216](<D:/AI project/nnunet/PATCH_NOTES.md:216>) |
| 축약 제거, 9/30 | partition 평균/quotient를 제거하고 원래 sampled fine-node/13종 edge에서 SAGE3 실행 | 실제 DEBUG 출력·gradient 동등성과 pause/resume 검사는 통과했다. CNN48³·원래 sampling은 유지하므로 CT의 모든 voxel을 노드로 복원한 것이 아니다. | [미축약 경로:7](<D:/AI project/nnunet/docs/v22_uncoarsened_sage_20260930.md:7>) |
| 학습 비교 단위 수정, 9/30 | `same_donor_live_v1`: case donor 고정, 모든 P×U 비교, 양쪽 live L0 gradient, balanced CE/multiplicity 보정, MRR→R@1→loss best | P/U 의미는 유지했다. 전체 schedule은 physical32,533updates/epoch,40epoch=21,320updates다. 선택 donor97개와 전체 pool527개를 구분한다. 앞선 tiny DEBUG에서 train 개선·validation 악화도 있었다. | [same-donor 정책:15](<D:/AI project/nnunet/docs/v22_same_donor_live_20260930.md:15>), [계산량:25](<D:/AI project/nnunet/docs/v22_same_donor_live_20260930.md:25>) |
| Native local CNN L0, 10/1 | 사용자 승인으로 L0 graph를 native-resolution crop CNN8Conv12/24/32, organ-masked multi-scale mean, paired fusion으로 대체 | `margin_mm`는 donor occupied bbox 밖 추가 거리이며 필수 입력. Unique crop batch·raw/crop cache·prefetch를 사용한다. L1/L2와 same-donor objective·후보128·원본 mask는 유지. 새 CNN 학습이며 옛 GAT/SAGE exact resume가 아니다. | [native CNN:3](<D:/AI project/nnunet/docs/v22_native_local_cnn_20261001.md:3>), [입력·모델:7](<D:/AI project/nnunet/docs/v22_native_local_cnn_20261001.md:7>) |
| 학습된 서버 CNN 진단, 10/1–2 | L1 interaction/reference·frozen 직접 head·rank-only/finite shadow 등의 복제 진단 | 제출 콘솔의 epoch27/30 및 finite 실행은 서로 다른 snapshot이다. 표현 분산 유지나 가중치 변화가 안정적인 P/U 후보 분리·held-out 개선으로 이어진 증거를 얻지 못했다. | [epoch30:10](<D:/AI project/nnunet/docs/local_cnn_interaction_epoch30_20261001.md:10>), [finite:50](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:50>) |
| Sparse feature graph, 10/2 | 현재 CNN68D 특징+상대 위치, query/near/mid/wide별 FPS, spatial3NN+feature1NN, SAGE3·역할 readout | 최초 feature/spatial 단위 문제를 정규화로 수정. 48/96/192 DEBUG 크기 비교 및 이후 각60연속 update를 수행했지만 별도 CT 순위는 자기 초기값 대비 악화했다. | [희소 경로:8](<D:/AI project/nnunet/docs/sparse_feature_graph_l0_DEBUG_20261002.md:8>), [연속 학습:7](<D:/AI project/nnunet/docs/sparse_feature_learning_DEBUG_20261002.md:7>) |
| Typed relational sparse, 10/2 | donor/recipient의8종 directed 관계별 mean-SAGE·역할 attention을 추가하여 fusion 이전 donor message 전달 | v1의 필수 fine 이웃/hop은 아직 복원하지 않았다. 모든133pair에서 연결 분리·고립과3-hop message reachability 한계가 확인됐다. | [관계 구조:8](<D:/AI project/nnunet/docs/relational_sparse_L0_DEBUG_20261002.md:8>), [연결 진단:48](<D:/AI project/nnunet/docs/relational_sparse_L0_DEBUG_20261002.md:48>) |
| Native6 relay, 10/2 | 유효 native voxel의 parent forest와 누적 물리 길이≤6mm인 mandatory bidirectional relay를 보존 | seed48/96/192는 최종 node cap이 아니다. 399pair의 weak component1/isolate0을 확인했으나 모든 wide 정보가 SAGE3에서 query에 도달하지는 않는다. 모든 규모의 비용이 빨라지지도 않았다. | [relay 구성:24](<D:/AI project/nnunet/docs/relay_sparse_L0_DEBUG_20261002.md:24>), [결과·한계:51](<D:/AI project/nnunet/docs/relay_sparse_L0_DEBUG_20261002.md:51>) |
| Full-footprint physical graph, 10/2 | bbox crop는 CNN 입력으로 유지하면서 graph domain을 완전한 donor-mask voxel에서 거리≤10mm인 영역으로 분리; conservative26/supercover 경로 | 박스 모서리 편향이 줄었지만 native 격자/5mm Z는 남았다. Coverage 악화, 특정 record의 미도달 seed, first-build/refresh 비용 증가를 기록했다. | [footprint 정의:24](<D:/AI project/nnunet/docs/footprint_sparse_L0_DEBUG_20261002.md:24>), [실제 한계:58](<D:/AI project/nnunet/docs/footprint_sparse_L0_DEBUG_20261002.md:58>) |
| 최신 CP bridge+matched60updates, 10/2 | annotation target 없이 새 graph로128후보 점수→기존 full-mask 필터→실제 paste를 연결; native CNN과 각60연속 update 대조 | 기계적 CP PASS, 추천 품질 미평가, production 교체 미완료. Held-out1case의 상위 관측 순위 개선 없음. | [최신 CP 기록:7](<D:/AI project/nnunet/docs/sparse_cp_recommender_DEBUG_20261002.md:7>) |

## 3. v1이 빨랐고 높은 수치가 나왔다는 기록의 증거 수준

사용자가 보고한 “v1이 빠르고 잘 학습됐다”는 경험은 비교의 출발점이다. 다만 과거 v1의 실제 checkpoint/cache/config와 같은 서버 자원의 전체 epoch를 복원해 속도 원인을 확정한 것은 아니다. 보존 소스 대조, 전달된 과거 console, 새 로컬 비용 probe를 각각 구분한다.

직접 읽은 복구 console에는 서로 다른 두 실행이 있다.

| 보존 원문 | 출력에서 확인되는 값 | 그 출력이 증명하지 않는 것 |
| --- | --- | --- |
| [full run console:7](<D:/AI project/nnunet/experiment_results/recovered_conversations_20260918/terminal_records/f4e3b499-a931-484a-832c-d2aac078fd54.txt:7>) | train105/val26, graph cache235, `training complete epoch=40/40`; best epoch29의 MRR .9869 | 서버의 현재 상태·모델 파일 hash·전체 nnU-Net/의료 평가 완료. 같은 출력의 generated/validated/dataset은0이다. |
| [full run의 제한 audit:41](<D:/AI project/nnunet/experiment_results/recovered_conversations_20260918/terminal_records/f4e3b499-a931-484a-832c-d2aac078fd54.txt:41>) | val8sample의 clean top1 .8750/MRR .9167 | 전체 validation 성능, 모든 실행의 결과, 최신 revision의 품질 |
| [paired fold0 console:15](<D:/AI project/nnunet/experiment_results/recovered_conversations_20260918/terminal_records/ee0258b7-a9e4-40b2-a0e7-506017907681.txt:15>) | cache187 재사용, 40/40 완료 skip, val36에서 clean top1/MRR1.0 | segmentation Dice1.0, CP 적합성100%, 현재128U+다중P 문제의100% 일반화 |

과거 제공 fold0 exact-argmax ablation의 tumor Dice는 Basic CP .6454, Full M3 .6722, w/o L1 .6239, w/o L2 .6977로 기록돼 있다. 이 수치는 원래 prediction으로 이번에 재계산한 결과가 아니다. Full−w/oL2의 기록된 CI와 p값으로 해당 단일 fold에서 L2 이득이 입증됐다고 말할 수도 없다. [과거 결과의 출처·한계](<D:/AI project/nnunet/docs/results_summary_20260918.md:24>)

보존 v1 설정은 donor 하나의 원래 source anchor와 curriculum 후보 묶음을 비교하고 direct candidate ranking/view consistency를 학습한다. 현재는 독립 train donor를 조건으로 recipient의 관측 P와 기존 U를 순위화한다. Candidate 수, 관측 단위, L1/L2와 loss·best 선택이 달라서 이전 top1/MRR를 현재 R@1 및 pair-win과 같은 분모의 성능으로 비교할 수 없다. [학습 계약 대조](<D:/AI project/nnunet/docs/v1_v22_learning_diagnosis_20260930.md:22>), [후속 정정](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:116>)

보존 v1에는 source footprint를 변환해 positive와 curriculum 비교 위치의 target CT를 가리는 경로가 있다. 따라서 원래 source-anchor를 positive로 썼다는 차이만으로 “v1이 정답 종양을 그대로 보여줘100%를 만들었다”는 누수 결론을 내리지 않는다. 현재 source/target 계약 차이의 성능 기여율은 측정하지 않았다. [v1 masking과 인과 한계](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:127>)

같은 실제 CT 위치의 **초기 가중치 L0 비용 probe**에서는 batch16 기준 보존 v1 forward2.592/backward6.177초, 당시 현재 경로 forward1.914/backward5.234초였다. Native ranking loss가 아닌 명시적 비용 probe이고 토폴로지가 완전 동일하지 않지만, “현재 L0 하나가 더 커서 느린 것이 원인”을 입증하는 수치는 아니다. 당시 전체 workload는 normal epoch에 train11,279+support refresh11,279+val2,823=25,381pair forward였다. Cache entry235/187과14,102pair를 단순 나눠 속도 차이로 환산하지 않는다. [workload·측정 경계](<D:/AI project/nnunet/docs/v1_current_comparison_20260925.md:26>)

## 4. 최신 60-update 대조: 실제로 학습한 것과 실패한 것

두 branch는 기존 로컬 DEBUG saved step4의 CNN/fusion/L1/L2를 같은 숫자 가중치로 복제했다. Graph/readout은 seed42의 새 초기값이다. 서버의 장기 학습 CNN을 복원한 비교가 아니다. Graph는 context band당32seed, SAGE3층128D, CNN12/24/32, 기존 L1/L2·full objective를 유지했다. 전체/trainable parameter는 native CNN1,125,718 / footprint graph1,785,558이다.

Train은 liver_66/71/72/75의 원래 P15+U512=527관측, validation은 liver_31의 P8+U128=136관측이다. 합663/전체14,102=4.70%의 별도 DEBUG다. 각 cycle의 train P×U1,920쌍을 정확히 한 번씩 비교하고3cycle/60update를 수행했다. Physical/effective32, accumulation1이며 자연스러운 마지막 잔여 tile도 보존했다. Train-only support와 평가 query를 현재 모델로 재인코딩했고 held-out case를 support/teacher/optimizer 입력에 쓰지 않았다. 마지막60update를 사전 종료점으로 사용했으며 held-out을 보고 best를 선택하지 않았다. [실행 조건](<D:/AI project/nnunet/docs/sparse_cp_recommender_DEBUG_20261002.md:15>) 및 원시 report의 `limitations`, `full_loss`, `train_only_support`.

| Branch / 평가 | Pair-win 초기→60update | MRR 초기→60update | R@1 초기→최종 | R@5 초기→최종 | Rank loss 초기→최종 |
| --- | --- | --- | --- | --- | --- |
| native CNN / train4case | 54.58%→82.97% | .132639→.508333 | 0→.066667 | .066667→.466667 | .687423→.443847 |
| native CNN / held-out1case | 40.43%→53.22% | .111111→.071429 | 0→0 | 0→0 | .694541→.739231 |
| footprint graph / train4case | 42.50%→92.71% | .055907→.687500 | 0→.133333 | 0→.533333 | .696878→.264581 |
| footprint graph / held-out1case | 56.93%→50.78% | .100000→.062500 | 0→0 | 0→0 | .691090→.953766 |

Pair-win은 P 점수가 U 점수보다 높은 쌍의 비율이며 동점은0.5로 센다. 분류 정확도·CP suitability 정확도가 아니다. MRR는 case별 첫 관측 P의 역순위, R@k는 모든 관측 P에 대한 micro recall이다. 위 표는 집계 JSON의 `branches.<name>.train/validation.initial/final`을 직접 읽어 대조했다. Graph 초기 MRR는 원시 JSON의 .0559074573을 사용했다. 최신 요약 문서의 .055908 표기와 작은 차이가 있어 이 표는 원시 값을 우선한다.

Held-out의 마지막 첫 P는 CNN14위, graph16위이고 두 branch 모두 R@1/5/10=0이다. CNN의 pair-win 일부 상승과 상위 순위·loss 악화가 함께 있다. Graph는 train-fit이 강해졌으나 held-out pair-win/MRR/loss가 자기 초기값보다 악화했다. 마지막 held-out score std .377470/1.120450이므로 이번 결과를 단순 상수 점수 문제와 동일시하지 않는다. 한 held-out CT와60update로 최종 일반화나 유일 원인을 확정할 수 없다. [결과 해석](<D:/AI project/nnunet/docs/sparse_cp_recommender_DEBUG_20261002.md:29>)

실제 CP 이벤트는 recipient liver_31 / train donor liver_100 component4다. Source mask766voxel→기존 spacing 변환의 full paste mask260voxel을 사용했다. 두 branch 모두128후보를 점수화하고 기존 filter에서102개를 허용했다. CNN centre[346,330,33], graph centre[341,355,77]에 각각260voxel을 paste했다. `mechanical_pass=true`, `quality_evaluated=false`, `CP_efficacy_established=false`이며 입력 원본과 mask 밖 값을 보존했다. **서로 다른 위치가 선택됐다는 사실은 더 좋은 위치를 학습했다는 뜻이 아니다.** [CP 실행 범위](<D:/AI project/nnunet/docs/sparse_cp_recommender_DEBUG_20261002.md:40>)

실행 장비는 RTX5070Ti16GB1개, CPU16logical, workers8, 시작 가용 RAM41.86GiB다. CUDA12/RSS40/resident24GiB 예산이며 FP32/deterministic/TF32off다. Compute update 중앙값은 CNN .225690초 / graph1.768926초, peak allocated1.821/3.675GiB였다. Graph의 속도 우위나 서버3시간/epoch 해결을 확인한 결과가 아니다. [비용·범위](<D:/AI project/nnunet/docs/sparse_cp_recommender_DEBUG_20261002.md:51>)

## 5. GPT가 원인 판정에서 빠뜨리면 안 되는 반증

1. **연산 교체와 학습 품질은 다르다.** SAGE의 작은 DEBUG update는 빨라졌지만 memory refresh는 느려졌고 full-support 서버 epoch와 CP 효용은 검증하지 않았다. Fine graph를 복원한 것만으로 학습 target/loss 불일치가 해결된 것도 아니다.
2. **EZ-SP 결과의 연결성은 해부학적 유효성을 보증하지 않는다.** 저장된8pair의 scale1은52,666→13,868nodes, scale2는13,849nodes였다. 선택 recipient region 하나는 원래3,150node 중2,496개를75.24mm bbox diagonal의 한 region으로 모았으며, scale2는 해당 role231→231로 변하지 않았다. Bbox/variance는 공식 merger 내부 제약이 아니라 사후 검사다. [partition inspection:12](<D:/AI project/nnunet/docs/region_partition_inspection_20260929.md:12>)
3. **서버의 거의 공통 점수 현상은 단일 수정으로 회복되지 않았다.** Epoch30 L1 내적항 개입은 score spread1e-6–1e-5와 ln2 부근 pair loss를 유지했다. Finite shadow에서 동일 BN을 고정하거나 CE를 제외한 한 step도 안정적인 분리 개선을 보여주지 못했다. 네 rank-only arm은 현재 full branch에서 갈라진 한 step씩이며4회 연속 rank-only 학습이 아니다. 첫 fresh Adam step에도 공통 이동이 있어 모두 과거 moments 탓이라고 설명할 수 없다. Reference BN의 공통 이동 현상을 LayerNorm인 legacy 실패 원인으로 옮겨 쓰지 않는다. [epoch30:43](<D:/AI project/nnunet/docs/local_cnn_interaction_epoch30_20261001.md:43>), [finite:55](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:55>), [BN 구분:85](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:85>)
4. **직접 head는 이미 검사했다.** Frozen recipient project-r+fresh nonlinear scalar head100update의 엄격 pair-win은 train56.93→58.85%, validation43.83→36.19%였다. CNN 재학습·v1 재현 결과가 아니며 이번 finite snapshot과도 같지 않다. “직접 head 미실행”으로 다시 설명하지 않는다. [기존 head 증거](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:98>)
5. **Mean pooling을 유일 원인으로 확정하지 않는다.** 국소 crop의 mean이며 간 전체 평균이 아니다. Actual anchor를 readout에서 쓰지 않는 표현력 제한은 코드에서 확인했지만, frozen head 실패는 CNN/mean/project를 모두 고정했으므로 손실을 mean에 특정하지 못한다. [mean의 측정 한계](<D:/AI project/nnunet/docs/local_cnn_finite_server_20261002.md:220>)
6. **노드 증가만으로 일반화가 해결되지 않았다.** 단순 sparse의48/96/192 각각60update에서 train pair-win91.35/92.40/94.79%까지 올랐지만 held-out pair-win43.26/48.44/54.88%는 모두 자기 초기70.51/73.54/73.05%보다 낮았다. [노드 수 연속 학습](<D:/AI project/nnunet/docs/sparse_feature_learning_DEBUG_20261002.md:9>)
7. **대표점 관계 추가만으로 연결은 보존되지 않았다.** Relational graph133pair의 weak component19–53, isolate10–36이 남았다. Relay로 weak component1이 돼도3-layer directed query ancestor와 readout 경로를 따로 검사해야 한다. [relational:52](<D:/AI project/nnunet/docs/relational_sparse_L0_DEBUG_20261002.md:52>), [relay:57](<D:/AI project/nnunet/docs/relay_sparse_L0_DEBUG_20261002.md:57>)
8. **Footprint 모양 개선에는 측정된 대가가 있다.** 같은 원본 전체 pool의 spatial coverage/CNN feature deficit은 새 domain에서 악화했다. `liver_66:65`의 recipient seed1개는 모든 밀도에서 physical root에 미도달했다. Seed192에서 typed joint graph가 연결돼도 physical forest의 미도달은 남는다. Seed48/96은 실제 component2/isolate1이다. 이 record는3update tile에 들어 있지 않았다. [footprint:68](<D:/AI project/nnunet/docs/footprint_sparse_L0_DEBUG_20261002.md:68>), [coverage:74](<D:/AI project/nnunet/docs/footprint_sparse_L0_DEBUG_20261002.md:74>)
9. **Foreign donor footprint와 P/U는 별개다.** Footprint 검사133recipient 중46record에서 full footprint의 간 밖 voxel이 있었고 최대475voxel이다. P/U를 바꾸거나 clip/record skip하지 않았다. 실제 CP는 기존 full-mask admission을 거쳐야 한다. [footprint:32](<D:/AI project/nnunet/docs/footprint_sparse_L0_DEBUG_20261002.md:32>)
10. **Cache update의 짧은 이득을 full epoch로 환산하지 않는다.** Footprint geometry-cache의 같은 tile update는 일부 단축됐지만 first-build가 남고 refresh는 오히려 느렸다. FPS와[B,N,N] graph construction 비용도 남는다. 최신 matched 비교의 graph는 native CNN보다 느렸다. [footprint 비용 경계](<D:/AI project/nnunet/docs/footprint_sparse_L0_DEBUG_20261002.md:84>)

## 6. 상태를 정확히 구분한 현재 인계

| 대상 | 완료 또는 확인 | 미완료·미검증 |
| --- | --- | --- |
| 보존 v1 | 소스/설정 보존, 복구 console의 과거40/40 및 순위 숫자, 별도 L0 비용 대조 기록 | 원래 학습 완료 가중치의 현재 holdout 재평가, 동일 서버 전체 epoch 비교, 최신 실험으로의 숫자 이전 |
| 현재 native CNN production 경로 | Native crop/CNN/fusion/L1/L2/train·resume·export/scorer 구현 및 DEBUG 연결 증거; 사용자가 제공한 epoch27/30 서버 snapshot 진단 | 최신 서버 전체40epoch 완료·전체 CP/nnU-Net250epoch·전체 segmentation metric·Basic CP 대비 효용을 이 문서에서 확인한 것 아님 |
| 최신 footprint sparse | 별도 L0 구현, 실제 CT forward/loss/backward/continuous optimizer,60update train/held-out 관측 평가,128후보 CP bridge·실제 paste | Production artifact/load/online trainer 채택, 충분한 독립 환자 평가, CP 추천 품질·segmentation 효용, 최적 node/radius·서버 epoch 비용 |
| 이번 이력 문서 | 기존 기록과 선택 JSON 필드의 읽기·대조·새 문서 작성 | 신규 syntax/unit/smoke/model/GPU 검사와 장기 실행은 수행하지 않음 |

GPT에게는 **P/U를 고정한 현재 objective를 어떻게 학습하고 다른 CT로 일반화할 것인지**를 묻는다. V1의 GT나 curriculum으로 교체하는 제안, 이미 수행한 직접 head/finite 검사의 반복, 그래프가 연결됐다는 이유만으로 추천 품질 PASS를 주는 결론은 이 자료의 결론과 맞지 않는다. V1과의 차이가 있다는 사실을 GT 오류나 단일 실패 원인으로 단정하지 않는다.

검토 답변에는 제안별로 근거 artifact, 반증, 현재 증거로 확정할 수 없는 부분, 변경할 코드 경로와 정답·규모를 보존하는 대조 조건을 적어야 한다. 특히 train-fit과 held-out 실패를 구분하고, 서버의 거의 공통 점수 현상과 최신 sparse의 분산 있는 점수/held-out 실패를 같은 현상으로 합치지 않아야 한다.

## 작업 완료 체크리스트

아래는 **이번 자료 정리 작업**의 범위다. 기존 GPU 실행 결과를 확인한 것은 새 GPU 측정이나 재실행이 아니다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 새 문서만 추가했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 모델 변경 없음.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 과거 DEBUG 규모와 전체 계약을 구분했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 저장된 physical/effective32·workers·batch 경로를 읽어 확인했으며 새 실행은 하지 않았다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 저장된 최신 report의 자원·peak·시간 필드를 확인했다. 새 현재 장비 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 문서 작업에 OOM은 없으며 과거 RAM admission 실패·cache/refresh 병목 및 측정 경계를 기록했다.
- [x] 디버그 설정과 최종 설정을 분리했다. 로컬 DEBUG step4, 서버 학습 snapshot, 최신60update를 혼합하지 않았다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 초기화/합성 UNIT/실제 CT/과거 console을 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 실제 CT DEBUG의 확인 범위를 인용했으며 새 연결 검사를 수행했다고 주장하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 반려·철회·실패·미도달·coverage 악화·held-out 악화를 포함했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
