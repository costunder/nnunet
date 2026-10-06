# v1 그래프 크기 대조와 변경 묶음 비교 — 2026-10-03

## 실제 학습 실행기 연결

전용 DEBUG sampler에만 연결돼 있던 strict-nested를 `run_v1x_experiment.py init --local-sampling strict_nested --role-seeds <6개> --reference-experiment <native>`로 선택할 수 있게 연결했다. native와 nested는 같은 canonical preparation과 측정된 batch/worker lock을 공유하며 train·validation·generation에 동일한 고정 sampler 계약을 적용한다. 기존 누적 stage v1.1/v1.2/v1.3은 nested profile을 뜻하지 않는다. 첫 sampling 대조는 양쪽 모델 stage v1.0으로 한다.

실제 CT/CUDA의 spawned worker·full loss/gradient/update·validation·canonical inference·전체 596 voxel paste·checkpoint reload smoke는 통과했다. 전체 84/21 학습, 순위 품질 유지, production 128-candidate generation과 전체 epoch 비용은 아직 실행하지 않았다. 상세 구현·검사·명령은 `docs/v1_native_nested_runtime_20261003.md`에 기록했다. 이전 sampler 및 raw evidence는 그대로 보존한다.

## 최신 독립 검토 반영: 원본 sampled view 내부에서만 줄인다

기존 canonical-resample 결과에서 compact node의 약10~12%가 원본 sampled view 밖에서 새로 선택됐다. 그래프 크기와 선택 분포가 함께 바뀐 대조였다. 기존 코드와 결과는 보존하고, `hiercp_v1x/nested_graph_size.py`에 원본 `build_local_view()`가 만든 그 sampled view 내부에서만 줄이는 별도 대조를 추가했다. 모델·GT·loss·CNN 입력을 유지하며 모든 induced edge와10D attributes를 원본 순서대로 보존한다.

`graph_flow_audit.py`는 실제3-layer GAT와 같은1/2/3-hop directed 도달성, zero-in-degree, relation별 degree, role/shell coverage, directed shortestpath와 correspondence coverage를 검사한다. Weak component 연결 성공을 정보 보존으로 해석하지 않는다. 실제 결과와 검사 범위는 `docs/v1_strict_nested_review_correction_20261003.md`에 기록한다.

아래 canonical104/416 결과와 이전 누적 v1.0~v1.3 결과는 보존 기록이다. 새 strict-nested 결과와 혼합하지 않는다. Group search는quality(local_features/local_operator)와runtime(activation_storage)를 분리한v2 계약을 사용하며 구v1 계획을 자동 재해석하지 않는다. Small-v1의 전체84/21 순위 학습 품질과 모든 독립 조합 구현이 완료됐다는 뜻은 아니다.

## 최신 사용자 지시: 작은 v1 그래프를 먼저 검증하고 변경 묶음을 좁힌다

이 문서의 아래 네 단계 코드는 보존된 구현·검사 기록이다. 최신 진행 우선순위는 **v1의 그래프 크기를 먼저 줄인 대조군 → 변경 요소를 절반씩 나눈 대조 → 의심 항목과 상호작용 확인**이다. 네 단계를 무조건 차례대로40epoch 학습하는 계획으로 사용하지 않는다.

첫 비교에서는 v1의 positive/negative 의미, curriculum, 모든 loss, 전체 모델, scalar scoring, L1/L2, CNN patch와 관찰 범위를 유지하고 node selection만 변경한다. 실제 작은 v2.2 graph의 첫32pair 측정은 pair당 N388~437(mean416.34), typed E2,069~2,325(mean2,214.69)다. 이것을 크기 비교의 참조로 사용하며, branch seed96을 최종 node cap으로 착각하지 않는다.

작은 v2.2 graph는8role/8relation이고 원본 v1은6role/16relation이다. v2.2 graph를 그대로 넣으면 '크기만 바꾸는' 실험이 아니다. v1의 원래 역할·shell과 관계를 보존하는 별도 sampling 경로가 필요하다. 원본 `hiercp/sample.py`의 `build_local_view` 선택 단계 뒤에 `_subset_node`, `_induced_edge_index`, 원본10D edge attributes를 유지하는 경계가 있다. 현재 context seed384를96으로 바꾸는 것만으로는 필수 interface 이웃·2hop 확장 및 다른4role의 전량 노드가 남아서 최종 graph가 작아지지 않는다.

첫 작은-graph 대조의 완료 조건은 같은 record의 N/E와 role/shell/연결성분, 동일 원본 mask와 CNN 입력, forward/backward/optimizer, 처리시간·VRAM, 동일 held-out 후보의 순위다. 실행만 통과한 결과를 추천 품질 유지로 판정하지 않는다. 범위를 줄이는 실험과 같은 범위에서 sampling 밀도를 줄이는 실험도 분리해야 한다.

그 다음은 변경 집합A/B를 절반으로 나눠 `기준+A`, `기준+B`, `기준+A+B`를 대조하고, 악화가 재현되는 묶음을 다시 나눈다. 학습 GT가 바뀌는 observed P/U 전환은 이 고정-GT 묶음에 섞지 않는다. 변경 효과에 단조성이나 독립성이 보장되지 않으므로, 양쪽 묶음에서는 괜찮고 합칠 때 나빠지는 경우를 상호작용으로 남긴다. 마지막 의심 요소는 단독과 의심 조합으로 재확인한다.

**현재 상태:** 작은-v1 sampling과 같은 원본 모델의 실제 CT/CUDA DEBUG 대조를 구현·실행했다. 전체 순위 학습·held-out 추천 품질과 변경 묶음의 학습 결과는 아직 없다. 기존 ZIP `HierCP_v1x_REVIEW_20261003.zip`은 방향 변경 전의 네 단계 구현 기록으로 보존한다.

## 그래프 크기만 바꾼 실제 CT / GPU 검사

`hiercp_v1x/graph_size.py`는 원본 canonical pool에서 역할별 seed를 선택한다. Context는 원본 shell을 나누고 각 shell 안에서 FPS를 한다. 원본의 비어 있지 않은 각 관계에 endpoint witness를 보존한다. 선택한 점들이 원래 같은 weak component에 있을 때 **원래 topology의 BFS 경로에 있는 relay node**를 추가한다. 새 edge, kNN, 영역 평균, radius 변경, 모델 변경은 없다. 최종 edge는 남긴 원본 노드 사이의 모든 induced edge다. Weak connectivity 검사는 방향별 GNN message 도달성 검증을 뜻하지 않는다.

두 명시적 설정은 원본 역할 순서 `tumor_surface / tumor_interior / source_context / source_liver_surface / target_context / target_liver_surface`에서 각각 `64/32/96/64/96/64`와 `16/8/24/16/24/16`이다. 전자는 합계416 seed를 작은 v2.2의 실제 평균416노드와 비교하기 위한 첫 설정이다. 연결을 유지하면 seed 이상으로 커지는 것을 확인하여 **별도 104-seed 대조**도 실행했다. 이 숫자는 최적값이나 production 기본값이 아니다. 최종 노드 cap으로 잘라 결과를 맞추지 않았다.

| 실제 48개 graph view의 평균 | 원본 v1 | 416-seed 대조 | 104-seed 대조 |
| --- | ---: | ---: | ---: |
| N / graph | 11,335.2 | 1,094.7 | 362.3 |
| typed E / graph | 783,205.9 | 18,798.9 | 3,108.9 |
| 추가 relay N / graph | 해당 없음 | 685.0 | 258.3 |
| 측정된 전체 DEBUG update 평균 | 21.87초 / 22.65초¹ | 1.86초 | 1.59초 |
| 최대 PyTorch GPU allocated | 6.29GiB | 0.65GiB | 0.48GiB |

¹ 각 대조 실행에서 같은 원본 branch를 따로 2 update 측정했다. 서로 다른 실행의 baseline 수치를 섞어 배속을 계산하지 않는다. 104-seed 대조에서 N은267~527, E는2,397~4,047이었다. v2.2의 첫32pair N388~437 / E2,069~2,325와 비슷한 규모이지만 **완전히 동일한 크기·관계 구조는 아니다.**

실제 CT는 train `liver_5/6`, held-out `liver_31`이며, train case만으로 fit한 기존 DEBUG prototype bank를 재사용했다. Physical batch2 sample ×8 candidate=16개 graph/view, 두 view, 원본 모델10,434,532 parameters, CNN5채널48³, hidden128·local/patient/prototype3/2/2·heads4, 원본 전체 loss와 original checkpointing을 그대로 썼다. RTX5070Ti16GiB 한 개, worker4, 명시적 CUDA12GiB/RSS32GiB 한도로 실행했다. 모델 깊이·폭·CNN 입력·상위 graph·정답·전체 paste mask는 두 branch에서 동일하다. 축소 sampler는 새로운 공간을 만들지 않지만 원본 sampled view와 다른 canonical node를 뽑을 수 있으므로 **엄격히 같은 node set의 nested subset만 비교한 실험은 아니다.**

두 branch는 같은 seed42 원본 **무작위 초기화 전체 state hash**를 공유한다. 실제 학습 완료 checkpoint를 재현한 검사가 아니다. 각 branch2 update에서 원본 loss→CNN/L0/readout/L1/L2/scalar gradient→AdamW 갱신, held-out scoring, 전체596voxel source mask paste·mask밖 보존을 검사했다. 104-seed branch의 train2case MRR은0.333→0.75, held-out1case MRR은0.25→0.167이었다. 해당 대조의 원본은 train0.3125→1.0, held-out0.20→0.143이었다. 이렇게 작은 분모와2 update로 정확도 유지·하락이나 일반화 원인을 판정하지 않는다. 생성 후보128 전체 CP inference와 segmentation 학습은 실행하지 않았다.

새 측정은 원본 `set_seed/configure_runtime`의 `cudnn.deterministic=True`, TF32 off, benchmark off를 적용했다. 비용은 GPU 전송 이후 synchronized forward/loss/backward/gradient 검사/clip/optimizer까지다. Loader, graph preparation, 전체 validation과 checkpoint IO를 포함한 epoch 시간이 아니다. 원본/축소 graph materialization도 별도 기록했다. 아직 canonical CSR를 두 view마다 다시 만드는 부분이 남아 있으므로 모든 전처리 중복을 제거했다고 주장하지 않는다.

원시 결과는 `work/v1_graph_size_CUDA_DEBUG_20261003/report.json`과 `work/v1_graph_size104_CUDA_DEBUG_20261003/report.json`에 있다. 같은 record의 축소 전 sampled N/E, canonical N/E, role/shell 수, 관계별 edge, 연결성분, seed/relay 및 원본 ID는 각각 `sampling_audit.json`에 저장했다. 첫 실행을 덮어쓰지 않았다. 두 대조 모두 전체 학습·추천 품질·production ready는 false다.

추가 coverage 대조에서48graphview 중416-seed의 canonical weakcomponent 일부 미선택은0건, isolated node를 포함한 view는4건이었다. 104-seed에서는 각각2건과16건이었다. 선택된 원래 weakcomponent 안의 연결성을 유지했다는 것과 원본의 모든 component·방향별 메시지 도달성을 보존했다는 것은 다르다. 원본 sampled view에 없었던 canonical node를 새로 선택한 합계는416-seed6243개,104-seed1809개다. Sampling 위치 변경의 혼입도 후속 검토 대상으로 남긴다.

## 작은 v1 이후의 절반 묶음 비교 구현

`hiercp_v1x/group_search.py`는 `기준 / 기준+A / 기준+B / 기준+A+B`의 비교 계획·근거 계약을 만든다. 현재 독립 요인은 CNN activation 저장 방식, local handcrafted16열 사용 여부, local GAT/SAGE 연산이다. 이전의 누적 v1.1~v1.3을 모든 독립 조합의 구현으로 간주하지 않는다. 특정 조합의 실제 adapter·기계적 검증 hash가 있어야 그 조합을 등록한다. 정답 체계를 바꾸는 observed P/U 전환은 이 집합에 넣지 않는다.

허용할 품질 차이와 비용 개선 기준은 필수 비교 설정이고 임의 기본값이 없다. 짧은 동일 조건의 복제 실험(`cloned_diagnostic`)과 전체 epoch 비용(`epoch_cost`)을 구분한다. 이전 일반 비용 양식의 refresh 항목은 v2.2 support memory에 해당하며 v1에는 없다. 실제 v1 비교 비용은 sampler/loader/H2D/optimization/validation/save로 기록한다. 짧은 대조에 전체 epoch 실행을 강제하지 않으며 짧은 계산 시간을 전체 epoch 해결 근거로 쓰지도 않는다. 상호작용·상쇄·양쪽 실패·근거 누락을 별도로 남기고 단일 원인을 자동 확정하지 않는다. 실제 작은-v1 학습의 held-out 품질 대조와 절반 조합 실행은 NOT_RUN이다.

새 metadata/topology 단위 검사35개가 통과했다: sampler9, group search19, probe input/scope7. 이것은 CUDA 결과와 구분한다. 장기 GNN/nnU-Net 학습, 서버 변경, Git push는 이번 그래프 크기 DEBUG 검사에서 실행하지 않았다.

## 구현 목적과 기준

v2.2의 관측 P/U 학습을 임의로 재정의하지 않고, 보존된 v1에서 어떤 변경이 학습과 CP 추천에 영향을 주는지 분리한다. `versions/v1/pipeline_v1_source.zip`의 revision `74dcc2cf03d2d40d1f582223321d96004333f661`과 202개 파일을 먼저 검증한다. 원본 archive SHA256은 `5157bafe641e9189824826532a3b055ea560f5c5dfc299374842a3bd1d20a22e`이다.

원본 파일을 편집하거나 현재 `hiercp`를 원본으로 가정하지 않는다. 실행기가 실험 디렉터리에 검증된 원본을 풀고, 선언한 모델 변경만 격리된 overlay로 적용한다. 실행 전에 원본·overlay·설정·분할·캐시 결속을 다시 검사한다.

현재 생성한 최종 실행 디렉터리는 `work/v1x_suite_r4_20261003`이다. 앞선 디렉터리와 실패 결과는 그대로 보존했다. r4는 아래 최종 실행 정책과 단계별 checkpoint serialization adapter를 사용한다.

## 버전과 비교 순서

| 버전 | 직전 버전 대비 변경 | 유지하는 것 |
| --- | --- | --- |
| v1.0 | 원본 코드·모델·설정 | 원본 positive anchor, curriculum, loss, scalar score |
| v1.1 | CNN activation 보관; 큰 L0 graph block의 checkpointing 유지 | 수식·가중치 구조·입력·그래프·batch |
| v1.2 | L0 node projection의 handcrafted 16열 제거 | CNN 32D, 그래프, role/shell pooling, 상위 계층 |
| v1.3 | L0의 관계별 GAT를 edge-conditioned mean GraphSAGE로 교체 | 같은 모든 node/edge, 16종 관계, residual/LN/FFN, 상위 계층 |

`v1.2 - v1.1`, `v1.3 - v1.2`처럼 직전 변경과 비교하고, 공통 v1.0 대비 차이도 함께 계산한다. 여러 변경을 한꺼번에 적용한 v1.3 결과만으로 각 변경의 효과를 판정하지 않는다. 새 모델 단계는 seed42로 새로 시작하며, 다른 단계의 checkpoint를 exact resume로 받아들이지 않는다.

v1.1의 초기 all-retained 진단은 실제 큰 CT 그래프에서 12GiB 한도를 초과했다. 실패 결과를 보존했고 노드·edge·batch를 줄이지 않았다. 최종 정책은 `checkpoint_dense_encoder=False`, `checkpoint_local_blocks=True`다. CNN의 재계산만 제거하며, 큰 GAT activation을 모두 저장하는 설정이 검증됐다고 주장하지 않는다. v1.3도 이 실행 정책을 그대로 유지하여 연산 교체 이외의 차이를 섞지 않는다.

v1.2에서 제거하는 범위는 **L0의 handcrafted projection 입력**이다. 원본 5-channel CNN 입력, shell/role metadata, 10D edge attributes, 상위 계층의 raw candidate 입력까지 동시에 제거하지 않는다. 따라서 이 단계 이름을 '모든 기하 정보 제거'라고 해석하면 안 된다.

v1.3은 PyG SAGEConv를 그대로 꽂은 모델이 아니라 원본 heterogeneous graph용 관계별 mean GraphSAGE 구현이다. 관계별 `mean(W_r h_source + E_r edge_attr + b_r)`를 합치며 자기 정보는 기존 block residual에서 한 번 전달한다. 방향·중복 edge의 multiplicity·관계별 degree를 보존한다. sparse CSR aggregation을 GPU에서 수행하고 같은 batch의 정적 topology와 edge-attribute mean을 세 local layer에서 재사용한다. 학습 중인 node feature는 캐시하지 않는다. L0 attention head는 SAGE에서 해당하지 않으며, L1/L2의 원래 4 heads는 유지한다.

## 전체 파이프라인

```text
원본 CT / 원본 label
  → train-only region / prototype bank 준비
  → 원본 source anchor + curriculum 후보의 두 graph view
  → 5-channel 3D CNN의 위치별 특징
  → L0: 6 roles / 16 directed relations / 3 blocks
  → role·context shell attention readout와 pair fusion
  → 기존 patient L1 2 blocks
  → 기존 population/prototype L2 2 blocks
  → 기존 직접 scalar score
  → 원본 CE + pairwise + ordinal + mined + consistency loss
  → backward / AdamW
  → 고정 validation curriculum의 MRR·top1·margin
  → 학습 완료 best checkpoint로 명시적 CP generate
  → 후보128 scoring / 전체 source mask eligibility / 실제 paste
```

v1의 positive는 source 종양의 실제 original anchor이고 negative는 원래 easy/inter/intra/relation-corruption curriculum이다. v2.2의 `P=실제 관측 종양 위치`, `U=미관측 비교 위치`와 다른 supervision이다. U를 CP 부적합으로, P를 외부 donor와의 적합 정답으로 바꾸지 않는다. observed P/U 목표로 옮기는 실험은 별도 v2 방향으로 기록할 사항이며 이번 v1.x 구현에는 포함하지 않았다.

## 고정 설정과 데이터

Hidden128, local/patient/prototype depth3/2/2, GAT heads4, dense CNN base12/output32, native patch48³, context28mm 및 shell4/12/28mm, 원본 sampling·hop·radius·mask 규칙을 유지한다. 기존 원본 admission 상한을 완화하거나 새 node/edge cap을 넣지 않았다. 원래 graph가 큰 경우에도 원본 결과를 그대로 계산하거나 명시적 오류로 보고한다.

전체 설정은 보존된 `config/train.json`에서 생성한다. 40epochs, lr1e-4, weight decay1e-4, 원본 curriculum과 모든 loss 가중치, seed42, candidate pool128, **학습 sample당8 candidates**, case당2 samples 및 두 view를 유지한다. 학습의8 candidates와 생성의128 candidates는 다른 단계다.

공통 분할은 기존 CP80 fold0의 inner train84 / inner validation21이다. outer validation26은 준비·학습·generate에서 제외한다. 원본 CT/label131개 모두 분할에 포함되거나 outer exclusion으로 명시되어야 한다. prototype은 train84에서만 fit한다. generate 입력은 해당105개 파일의 link만 만들고 CT를 다시 복사하지 않는다.

입력 분할 파일에는 split 생성 seed가 기재되어 있지 않다. 그 파일의 고정 case 목록을 SHA로 결속하고 그대로 사용한다. 정규화된 원본 형식의 `seed42`는 archive에 명시된 **학습·실험 seed**이며, 기존 분할 목록이 seed42 난수로 만들어졌다고 추정하는 기록이 아니다.

이는 **새 공통 분할에서의 원본 v1.0 대조군**이다. 역사적 v1의 train105/validation26 구성과 결과를 재현했다고 주장하지 않는다. 당시 MRR1을 이번 CP 정확도100%로 옮겨 해석하지 않는다.

Basic CP 및 기존 CP80 비교 계약은 수정하지 않았다. 이 실행기는 nnU-Net 또는 segmentation 비교 실험을 자동 시작하지 않는다. 원본 CP `generate`는 사용자가 명시적으로 실행하는 단계이며, 전체 마스크 검사와 원본 coverage/occupied-clearance 규칙을 유지한다.

## 캐시·batch·체크포인트

region/prototype/training cache를 `shared/`에 **한 번만** 준비한다. 네 단계는 이를 읽고 결과는 `results/v1.0`부터 `results/v1.3`에 각각 저장한다. 다른 버전의 latest/best를 자동 탐색하여 이어받지 않는다.

v1.0의 원본 GPU physical batch/loader worker calibration을 실제 실행하여 선택한 값을 `shared/execution_lock.json`에 결속한다. 이후 단계는 이 **같은 physical batch와 worker 수**를 사용한다. effective batch는 physical sample batch × accumulation1 × GPU worker1이다. 각 sample 안의8 candidate graph는 함께 batching한다. 현재 한 GPU만 할당되어 있다.

각 단계의 source/config/cache/split 및 execution 조건은 `launch_contract.json`과 `run_contract.json`에 기록한다. 원본 모델은 별도의 best와 epoch-boundary resume state를 사용한다:

- `results/<stage>/checkpoint_best.pt`: best MRR→top1→margin→loss 기준
- `results/<stage>/checkpoint_best.last.pt`: 마지막 저장된 **완료 epoch**의 optimizer/scheduler/scaler/RNG 상태
- `results/<stage>/checkpoint_best.pt.preflight.json`: 해당 실행의 calibration 기록

원본 checkpoint loader의 geometry/patient 검사를 그대로 호출하면서 새 단계의 architecture ID와 state revision도 확인하는 serialization adapter를 붙였다. 모델을 바꾸고 저장만 가능하게 만든 채 `generate`에서 로드되지 않는 연결 오류를 막았다. 생성 전에도 완료 결과의 source/cache/단계/원본 학습 계약을 검사한다.

원본 pipeline의 resume는 epoch 경계이다. Ctrl+C가 foreground Python 작업을 중단하지만 **진행 중 epoch를 매 update 저장한다고 주장하지 않는다**. 해당 epoch 도중 중단하면 마지막 완료 epoch부터 다시 진행한다. 부모 shell/SSH/session을 종료하는 명령을 사용하지 않는다.

## 실행 방법

저장소 Python 환경에서 실행한다. 아래 `run ... train`은 실제40epoch 실행이므로 이번 로컬 검사에서는 실행하지 않았다. `plan`은 명령만 출력하며 학습하지 않는다.

```bash
python -u tools/run_v1x_experiment.py init \
  --experiment work/v1x_suite \
  --medical-root /home/aicompetition06/Medical

python -u tools/run_v1x_experiment.py plan \
  --experiment work/v1x_suite --stage v1.0 --target prepare

python -u tools/run_v1x_experiment.py run \
  --experiment work/v1x_suite --stage v1.0 --target prepare

python -u tools/run_v1x_experiment.py run \
  --experiment work/v1x_suite --stage v1.0 --target train --gpu 3
```

v1.0 calibration이 저장된 후 같은 명령에서 `--stage v1.1`, `v1.2`, `v1.3`으로 각각 실행한다. 각 단계가 새 모델 실험이므로 v1.0의 optimizer를 넘기지 않는다. 각 단계 내 재실행은 그 단계의 원본 last-epoch state만 사용한다. `--gpu`는 해당 터미널에서 보이는 물리 GPU 번호를 선택하며, 현재 컨테이너에 GPU0만 보이면0을 사용해야 한다.

완료한 단계의 원본 checkpoint metric을 읽는 `collect`와, 동일한 평가 조건만 비교하는 `compare`를 제공한다. CPU에서 읽는 것은 checkpoint metadata이며 neural forward/backward 검사는 CUDA에서 수행한다. 완료되지 않은 학습·DEBUG 결과·다른 정답/분할/후보/분모를 full result로 합치지 않는다.

```bash
python -u tools/run_v1x_experiment.py collect \
  --experiment work/v1x_suite --stage v1.0

python -u tools/run_v1x_experiment.py collect \
  --experiment work/v1x_suite --stage v1.1

python -u tools/run_v1x_experiment.py compare \
  --baseline work/v1x_suite/results/v1.0/comparison_report.json \
  --candidate work/v1x_suite/results/v1.1/comparison_report.json
```

v1.2 이후에는 `--predecessor`로 바로 전 단계 report도 전달한다. `generate`는 원본 full-training checkpoint 요구를 유지하며, 학습을 완료하지 않은 DEBUG weight를 자동 채택하지 않는다.

이 명령은 새 파일이 있는 checkout 기준이다. 이번 작업에서 GitHub 업로드나 서버 장기 실행은 하지 않았다.

## 검증과 해석 범위

로컬 장비는 RTX5070Ti16GiB, CPU16logical/8physical, RAM63.93GiB이며 시작 시 RAM42.76GiB·VRAM 약12.78GiB 여유를 확인했다. 실제 CT DEBUG 준비는 liver5/6만 prototype fit/train에 사용하고 liver31을 held-out으로 두었다. 모델 깊이·너비·native graph·sample당8 candidates·두 view는 그대로이고, DEBUG case 수와 update 수만 별도 checker에 명시했다.

실제 CT sample 두 개를 동시에 처리했고, sample당8 candidates로 view마다16개 graph를 disjoint-union batching했다. 첫 view의 N210,222/E14,237,065, 두 번째 view의 N209,701/E14,195,726을 전 단계에서 동일하게 유지했다. 작은 그래프만 골라서 SAGE의 시간 이득을 보고하지 않았다.

| 단계 | 전체 parameters | 실제 DEBUG update 1 / 2 | 최대 GPU allocated |
| --- | ---: | ---: | ---: |
| v1.0 | 10,434,532 | 23.754 / 22.377초 | 5.749GiB |
| v1.1 | 10,434,532 | 22.218 / 21.601초 | 6.386GiB |
| v1.2 | 10,422,244 | 21.844 / 21.000초 | 6.378GiB |
| v1.3 | 9,617,380 | 1.855 / 1.070초 | 5.370GiB |

각 단계에서 원본 CE·pairwise·ordinal·mined·consistency loss, CNN/L0/L1/L2/scalar gradient, AdamW 가중치 변경, held-out CT scoring, 원본 마스크596voxels 전체 paste를 실제 실행했다. 붙여넣은 mask 밖의 영상은 보존됐고 해당 예의 간 coverage는1이었다. 이것은 **8개 DEBUG candidate에 대한 paste 검사**다. production의128개 후보 생성·scoring 전체를 실행했다는 뜻이 아니다.

이 시간은 두 update의 제한된 GPU 계산 비용이다. loader·전체 준비·validation·저장·서버 hardware를 포함한 전체 epoch 속도로 환산하지 않는다. 네 branch의 공통 DEBUG backend에서는 `cudnn.deterministic=False`였고 TF32와 cudnn benchmark는 꺼져 있었다. production 설정의 `deterministic=True` runtime을 통째로 재현한 benchmark라고 주장하지 않는다. 다음 checker 실행은 원본 configure_runtime을 적용하도록 수정했으며, 기존 원시 결과를 덮어쓰지 않았다.

원본 baseline 결과는 `work/v1x_real_CUDA_DEBUG_20261003/v1.0/report.json`, 최종 세 단계 결과는 `work/v1x_real_CUDA_DEBUG_20261003_final/report.json`에 있다. 그 사이 all-retained 및 stale helper import로 실패한 실행도 별도 디렉터리에 남겨 현재 profile의 성공 결과와 구별한다. 최종 checker는 실제 variant module 경로와 SHA를 결속하여 추출된 과거 helper가 현재 코드를 가리는 문제를 차단한다.

단위 검사는 총64개 고유 case를 통과했다: contract14, model10(CUDA8·metadata2), DEBUG scope7, runtime20, completed-result metadata10, serialization3. Metadata 검사는 파일·설정·정답·분모·체크포인트 계약을 CPU에서 검사한다. 신경망 연산·gradient·optimizer는 별도 실제 CUDA 검사에서 확인했다. Serialization 검사는 네 snapshot을 각각 새 subprocess에서 import하고 **native checkpoint loader와 strict state reload**까지 확인했으며 model forward나 학습은 실행하지 않았다.

`prepare → train → generate`의 잘못된 checkpoint·source·cache·stage 결속, 중복 실행, 준비 중 partial publication, 다른 batch의 resume를 차단한다. 완료 epoch40의 원본 best/last에 연결된 metric만 자동 수집한다. 같은 평가 조건과 분모가 아니면 비교를 거부한다. 정상 생성에서도 먼저 이 완료 계약을 검사한 후 원본 pipeline을 실행한다.

짧은 CUDA smoke는 실행 연결·gradient·optimizer·full-mask paste 확인이다. **v1.0의 역사적 성능 재현, 새 단계의 held-out 추천 개선, 40epoch 학습 완료, CP80 nnU-Net Dice 개선을 뜻하지 않는다.** 단계 채택은 동일 조건의 실제 완료 결과를 비교해서 판단하며 자동 승격하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
