# v2.22 r2 — 정답과 입력을 분리한 문맥 관계 학습과 군집 정렬

2026-09-22. 사용자가 승인한 두 관측 클래스와 누수 차단 설계를 별도 패키지 `hiercp_v222`에 구현한다. v2.21 및 이전 소스·설정·실험 결과는 변경하지 않는다. 구현/단위 검사와 전체 임상 성능 검증을 구분한다.

현재 format은 `hiercp_v222_cluster_alignment_r2`다. r1 소스는 `versions/v2.22/before_cluster_alignment_r2_20260922/`에 보존했다. [L2 군집 정렬 상세](pipeline_v222_cluster_r2.md), [References](../REFERENCES.md)에서 실제 적용과 논문 참고 범위를 구분한다.

## 과제와 해석

Data node 하나는 한 위치의 CT 주변 문맥 그래프 표현이다. Label 0은 주석에서 종양이 관측되지 않은 비교 위치, label 1은 기존 소형 병변 조건을 만족한 실제 종양 위치의 문맥이다. F는 support가 해당 관측 클래스에 속하지 않는다는 뜻이며 기하적 불가능성이나 미래 발생 불가능성이 아니다. U인 query에는 두 클래스 정답을 모두 숨긴다.

기존 소형 병변 기준인 **체적 등가 구 직경 20mm 이하**를 유지한다. 이는 최대 길이가 20mm 이하라는 뜻이 아니다. 큰 병변은 이번 소형 문맥 과제의 양성 anchor에서만 제외하고 목록·개수를 기록한다. 원본 데이터와 nnU-Net 분할 학습의 큰 병변은 유지한다. 소형 병변이 없는 환자도 비교 문맥과 online CP 수혜 대상에 포함한다. 모든 적격 양성 anchor 및 환자당 128개 비교 위치를 사용하며, 비교 위치 수는 기존 128 후보 예산을 계승한 구현 선택이지 검증된 최적값이 아니다.

출력 softmax는 구성된 관측 과제의 점수다. 실제 암 발생 확률로 보정되었다거나 donor-specific CP 적합성의 정답이라고 해석하지 않는다. 후보의 순위에 사용하고 CP 효과는 별도의 전체 분할 평가로 검증해야 한다. 양성 위치와 비교 위치의 표집 차이가 만드는 편향도 성능 검증 대상이다.

## L0: CT → CNN → 고정 공간 그래프

- CT 한 채널, 48³ 입력. 기존 `PatchFeatureEncoder3D`의 12/24/32 채널과 encoder 깊이를 유지한다.
- 노드 특징은 CNN의 32차원 출력만 받아 128차원으로 투영한다. CT 평균/표준편차, 곡률, 상대 좌표 등 수작업 벡터를 결합하지 않는다.
- 정답 종양 내부·표면과 간 정답 경계를 이용해 query 노드/엣지를 만들지 않는다. 가림 영역 바깥의 고정 물리적 격자 전체를 사용한다. KD-tree 반경 연결이며 기존 문맥 격자 간격 2.5mm, 연결 반경 6mm를 유지한다. 좌표는 샘플링/연결에만 사용한다.
- GATv2 3층, hidden 128, 4 heads, residual/FFN과 attention pooling. 학습 편의를 위한 노드/엣지 상한이나 임의 샘플링은 없다. 여기서 GATv2는 합의 이전 코드의 공간 인코더를 유지한 것이며 PRODIGY 원문 복제라고 주장하지 않는다.
- CT 입력과 graph topology 함수에는 query 정답·환자 ID·donor 모양·장기 마스크 인수가 없다. CT는 보간 전에 가려진다.

가림 반경은 모든 inner-train 적격 병변의 anchor 기준 실제 voxel 범위를 포함하는 최대 반경에 기존 2mm 여유를 더해 **한 번만** 정한다. 이를 모든 위치에 동일하게 적용한다. 문맥은 그 반경 바깥으로 기존 28mm 폭을 유지한다. fit 환자·병변 개수·반경을 캐시와 checkpoint에 저장한다. Inner-val 병변이 고정 범위를 넘으면 감사 파일을 남기고 실패하며, validation에 맞춘 자동 확대/샘플 누락을 하지 않는다. 이 조건이 실패하면 학습을 시작하기 전에 설계를 검토해야 한다.

고정 topology의 CNN 특징 샘플링은 삼선형 보간이다. CUDA `grid_sample` 역전파의 결정론 미지원 문제를 피하도록 고정 인덱스·보간 가중치를 사용한다. CPU reference의 forward와 gradient 일치 검사를 포함한다.

결정론 모드의 전체 edge tensor 역전파에서 DEBUG batch 2의 할당량이 29.49GB로 증가한 것을 측정했다. 모델/그래프를 줄이지 않고 GATv2의 동일한 수식을 노드 범위별로 계산하고 activation checkpointing으로 재계산한다. 범위 크기는 실제 여유 VRAM으로 정하고 로그에 남긴다. 모든 batch 샘플과 범위 내 이웃은 tensor로 병렬 처리하며 전체 노드·엣지를 빠짐없이 방문한다. 원래 PyG GATv2와 출력·입력 gradient·파라미터 gradient가 일치하는 단위 검사를 추가했다. 범위 크기는 실행 tiling이며 그래프 cap이 아니다.

## L1: 실제 support 관계에 조건화한 환자별 표현

각 support 환자 task에 관측 클래스 2개를 두고, 실제 클래스와의 연결은 T, 반대 클래스와의 연결은 F로 만든다. 엣지 특징 `[is_support, true_relation]`을 attention에 넣는다. Query는 `[0,0]`으로, known-F `[1,0]`과 다르다.

2층 edge-conditioned attention을 사용한다. Support data↔label 메시지 전달로 환자별 label 표현이 달라진다. 클래스 seed는 공유된 학습 파라미터이며 환자 ID별 암기 테이블을 두지 않는다. 2개 label이라는 선택과 128차원 벡터 너비는 별개다. 128은 기존 모델 너비를 유지했으며 최적 차원이라는 주장은 하지 않는다.

Query는 label에서만 메시지를 받고 support/label로 메시지를 보내지 않는다. 매 층 support 상태를 먼저 계산하고 query가 그 상태를 읽는다. Query 정답 인수는 model forward에 존재하지 않는다. Query가 속한 **전체 환자 그룹**의 모든 기록은 support에 넣지 않는다. 이는 같은 병변의 증강본·겹치는 패치뿐 아니라 다른 검사에서도 답이 재유입되는 것을 막는 보수적인 episode 구성이다. 같은 환자의 여러 검사는 L2에서도 한 task로 합친다.

## L2: 다른 환자의 문맥 표현 정렬

L1이 만든 환자별 2×128 label 표현에 2층/4-head cross-patient attention을 적용한다. 같은 환자 task를 직접 참조하는 attention은 막고 residual로 본인의 표현을 유지한다. Query 환자의 정답은 정렬 메모리에 들어가지 않는다.

Query 환자를 제거한 support의 L1 label을 관측 클래스별로 군집화한다. 실제 해당 클래스 관측이 없는 환자 label은 중심 fit에서 제외한다. Cosine average-linkage의 비단독 군집 cut 중 양의 silhouette가 가장 큰 K를 고르며 근거가 부족하면 K=1을 기록한다. 임의 고정 K나 수작업 영상 특징을 사용하지 않는다. Assignment와 L1 teacher 중심은 환자 episode 동안 고정한다.

L2 출력은 고정 teacher 중심에 대한 class-balanced assignment CE로 정렬한다. Query 분류에는 live L2 출력으로 만든 군집 중심과 환자 수 가중 log-sum-exp 점수를 사용한다. Total loss는 weighted query CE + alignment CE다. Query CE는 L2/L1/query CNN·GNN, alignment CE는 L2/support L1에 연결된다. 이 군집 선택·loss 조합은 논문 재현이 아니라 문서화된 프로젝트 설계이며 임상적 타당성은 별도 검증이 필요하다. SwAV의 view-only consistency나 균등 Sinkhorn을 복구하지 않았다.

Support L0 임베딩은 inner-train 전체를 매 epoch 재인코딩한 **detached memory**다. Query forward에서 모든 학습 샘플을 매 epoch 한 번씩 방문하므로 L0 전체는 query loss로 학습된다. Support branch의 CNN까지 매 query마다 다시 역전파하는 joint objective는 아니다. 이 memory 기반 학습 선택과 epoch 내 표현 지연을 명시한다. 검증 전·최종 선택 가중치 저장 전에도 전체 support memory를 재계산한다.

## 데이터 경계와 실행 경로

`run_v222.py`에 `check`, `prepare`, `train`, `reuse-native`, `nnunet-plan`, `bank`, `nnunet-train`, `predict`, `evaluate`가 연결되어 있다.

`prepare`와 신규 `nnunet-plan`은 기존 split 외에 명시적 환자 identity manifest를 요구한다. 형식은 `format=hiercp_patient_identity_v1`, `cases[case_id]` 아래 `patient_group`, `identity_basis`, `annotation_complete=true`다. 이름만 보고 자동으로 환자 독립성이나 주석 완전성을 보장하지 않는다. 원본 출처에 근거한 manifest가 없으면 오류로 중단한다. 파일 재압축/이름 변경을 잡기 위해 decoded CT hash도 비교한다. 해시가 다른 재촬영 환자 식별은 manifest의 정확성에 의존한다.

분할은 inner-train/inner-val/outer-val 환자 그룹이 겹치지 않아야 한다. Inner-train은 가림 규칙 fit, 모델 최적화, support memory, donor의 출처다. Inner-val은 모델 선택과 관측 과제 검증에만 사용한다. Outer-val은 GNN 입력 캐시에 생성하지 않는다. nnU-Net planning/학습은 outer-train, 최종 분할 평가는 outer-val이다. 분할과 source/config/checkpoint hash를 각 단계에서 대조한다.

Patch `.npy`에는 CT tensor만 저장하고 정답은 index의 별도 필드에 둔다. 정답은 그래프 입력으로 전달되지 않는다. 학습은 40 epochs, nnU-Net은 기존 250 epochs를 유지한다. Physical batch는 실제 full-graph forward/backward 후보 측정으로 정하고 worker 0/2/4/8도 데이터 읽기 시간을 비교한다. 0은 비교 후보이지 무조건 최종값이 아니다. Query loader는 환자별 batch sampler와 persistent workers로 구성하며 마지막 잔여 batch를 버리지 않는다. Static topology와 보간 표는 캐시한다. Gradient accumulation은 1이다.

## Online CP와 평가

Frozen v2.22 scorer는 128개 기하 적격 후보 각각에 동일한 가림/문맥 모델을 적용한다. 기하 필터는 붙여넣기 가능성을 판정할 뿐 L1의 F를 만들지 않는다. Score API는 CT patch만 허용하며 donor·query GT를 받지 않는다. 따라서 현재 점수는 donor와 무관한 위치 문맥 점수이고 donor 모양은 기하 후보 선정·원본 paste 경로에만 사용된다.

모든 outer-train 환자에게 같은 inner-train donor pool을 제공한다. 이벤트 발생 확률 0.5, 후보 128개 전체 점수의 exact argmax와 기존 raw target paste 전송을 유지한다. 실제 CT/정답의 paste는 native training dataloader에서 온라인으로 실행한다. Validation loader는 기본 nnU-Net loader이며 CP를 적용하지 않는다. 구 checkpoint와 DEBUG checkpoint는 bank 생성에서 거부한다. 새 CLI는 이전 모델이 차단된 상태를 이름만 바꿔 통과시키지 않는다.

분할 평가는 기존 v5 metric 정의를 보존해 `case_metrics.csv`, `lesion_metrics.csv`, `summary.csv`, `size_metrics.csv`를 출력한다. 크기별 recall, lesion Dice, 전체 Tumor Dice와 FP를 구분한다. 전체 학습·평가를 하지 않았다면 이 파일의 실험 성능값을 만들어 내지 않는다.

## 검증 기록과 남은 범위

**r2 검증:** 군집·누수·loss 검사 8개를 추가해 회귀 39개가 통과했다. 정적 검사 18파일 및 r1 대비 L0·L1 AST 보존도 확인했다. 실제 CT 3명/6개 전체 그래프에서 batch 2/4의 모든 파라미터 finite gradient 및 주요 그룹 optimizer 갱신을 확인했다. 최대 CUDA 할당량은 611,964,928 / 1,100,222,976 bytes다. 이 fixture는 클래스당 support 2명이므로 K=1/1이며 실제 다중 군집 성능 검증이 아니다. 최종 소스 해시와 일치하는 로그는 `work/v222_20260922/cluster_r2_final_debug/result.json`, 회귀 검증은 `versions/v2.22/verification_cluster_r2_20260922/`다. 전체 학습·평가는 미실행이다.

아래 수치와 오류 대응 기록은 **r1 당시 검증 이력**이다. r2 성능 결과로 재사용하지 않는다.

- `tests/test_v222_leakage_debug.py`: 가림 내부 CT 개입 불변성, query 정답 API 차단, support 불변성, 환자 alias/중복 CT 차단, train-only 가림 fit, 모든 층의 gradient/optimizer 갱신, T/F와 L2의 출력 영향, sampler 전체 방문, 구 checkpoint/bank 거부, 삼선형 보간과 전체 GATv2의 forward/backward reference 비교.
- `tools/verify_v222_real_debug.py`: 지정한 inner-train 실제 CT 3개에서 6개 문맥 그래프를 사용하는 명시적 DEBUG. CNN/GNN 너비·깊이와 물리적 그래프 규칙은 그대로다. Batch-4 probe는 동일한 실제 query 2개를 반복한 자원 측정이며 독립 환자 4개라고 보고하지 않는다.
- 최종 검사: 새 누수/수치/학습 연결 단위 검사 12개, 기존 버전까지 포함한 회귀 검사 총 31개 통과. 별도 runtime에 `nnUNetTrainer_250epochs_OnlinePromptGraphV222` 등록/import 통과. 트레이너 import 성공은 실제 전체 online CP 실행 성공과 다르다.
- 실제 CT: inner-train `liver_1`, `liver_5`, `liver_6`의 6개 DEBUG 그래프. 노드 24,174개/엣지 1,229,900개(별도 self-loop 제외), 전체 파라미터 1,519,063개. DEBUG 반경 17.7142mm는 3개 환자의 적격 병변 17개로 산정했으므로 production 반경으로 사용하지 않는다.
- 최종 결정론 CUDA 검사: physical batch 2/4 모두 통과. CUDA 최대 할당량 605,463,040 / 1,090,839,552 bytes. 측정 forward/backward 0.480/1.206초 및 0.638/1.944초. 이는 워밍업·전체 support 규모를 통제한 최종 처리량 결과가 아닌 DEBUG probe다. 그래프 규모 축소 없이 계산 범위를 256/128개 노드씩 나누었고, 모든 graph 노드를 처리했다.
- 모든 CNN/L0/L1/L2 파라미터에 유한한 gradient가 존재하고 각 주요 파라미터 그룹이 optimizer step으로 갱신됨을 확인했다. 실제 CT 중심값 개입 불변성, query 정답 변경 시 예측 불변성, query 변경 시 support 불변성도 통과했다. 최종 로그는 `work/v222_20260922/real_debug4_chunked/` 및 `versions/v2.22/verification_20260922/checks.json`이다.
- 최초 실제 CT DEBUG 통과 후 결정론 설정에서 `grid_sample` backward 미지원 오류, 이어 전체 edge tensor의 29.49GB 메모리 사용을 확인했다. 오류를 숨기거나 결정론을 해제하지 않고 보간과 GAT 계산을 수정했다. 실패·중단 로그/폴더도 보존한다.
- 전체 cohort의 identity/annotation 근거 확정, 전체 가림 범위 감사, 40-epoch GNN, 250-epoch nnU-Net, online CP 전체 실행, 전체 소형 병변 성능 평가는 이번 DEBUG 성공으로 대체하지 않는다. 새 목적함수·class 수·고정 가림의 성능 타당성은 미검증이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험 명령을 사용하지 않았다. 멈춘 본 작업의 unittest 단일 PID만 확인·보고 후 중단했다.
- [x] 사용자 파일과 기존 실험 결과를 파괴적으로 변경하지 않았다. v2.21 소스와 기존 handoff/patch notes를 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 승인된 관측 클래스 정의 변경과 공간 노드 의미 변경을 별도로 기록했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 모든 적격 양성 및 정의된 비교 위치·고정 격자 전체를 처리한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch와 병렬화 경로를 구현하고 실제 CT DEBUG batch 2/4를 측정했다. 최종 full-cohort batch는 별도 보정한다.
- [x] GPU·CPU·RAM 정보를 확인하고 DEBUG 자원 로그를 남겼다.
- [x] 설정을 낮추는 대신 CUDA 보간의 결정론 호환성을 수정했다.
- [x] DEBUG 설정·자료 범위와 production 실행을 분리했다.
- [x] Dummy 입력을 실제 CT로 보고하거나 가짜 결과/checkpoint를 생성하지 않았다.
- [x] CNN/L0/L1/L2가 forward, loss, gradient, optimizer에 연결됨을 단위 검사했다.
- [x] 실제 실행 설정·새 설계 선택·미검증 범위를 기록했다.
- [x] smoke test와 전체 학습/평가를 구분했다. 전체 학습·평가는 실행하지 않았다.
