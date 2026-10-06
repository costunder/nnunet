# v2.22 r3 — 원본 CT와 전체 공간 그래프

2026-09-23. 사용자가 승인한 전체 수정안을 구현했다. Format은 `hiercp_v222_raw_ct_cluster_r3`, 입력 계약은 `v222_raw_ct_full_ball_v1`이다. 이전 코드는 `versions/v2.22/before_raw_ct_r3_20260923/`에 보존했다. 이 문서는 현재 구현 기준이며, 과거 r1/r2 가림 설정을 다시 적용하지 않는다.

## 이번에 바꾼 것

실제 CT와 전체 L0 노드를 조작해 보는 화면 및 VRAM admission 추가 수정은 [시각화·실행 검증 기록](v222_visual_runtime_verification_20260923.md)에 정리했다. 현재 실행은 `gnn_vram_affine`이며 전체 학습·평가는 미완료다. 아래 구조는 그대로 유지한다.

| 항목 | 현재 구현 |
| --- | --- |
| CT 입력 | 원본 CT 1채널, 기존 HU [-200,250] 정규화와 48³ 보간. 중심을 상수로 가리지 않음 |
| L0 노드 | 물리 반경 55.290656mm 안의 일반 공간 격자 전체. 2.5mm 간격, KD-tree 6mm 반경 연결 |
| 그래프 규모 | 45,385 nodes / 2,383,482 directed edges, self-loop 제외. 기존 외곽 39,936 nodes와 2,040,356 edges 전부 보존 |
| 비교 위치 | 제공된 GT label1인 간 좌표 중 양성 anchor와 정확히 같은 좌표만 제외. 종양 거리 제한 없음. case당 128개 비복원 표집 |
| 모델 | CNN 12/24/32, L0 GATv2 3층·hidden128·heads4, L1 2층, L2 2층 유지. 1,519,063 trainable parameters |
| 실험 | 같은 split, seed42, CP 시도율80%, GNN40epoch, nnU-Net250epoch 유지 |

내부에 추가한 5,449개 노드는 종양 마스크를 따라 생성한 특별한 종양 노드가 아니다. 모든 위치에서 같은 격자를 쓴다. 이 격자는 중심이 정상 위치든 종양 위치든 동일하다. CT 평균·표준편차·곡률·상대 좌표를 특징 벡터로 추가하지 않는다.

## L0 — 영상에서 위치 표현까지

`context_patch(image, spacing, center, contract)`는 CT와 좌표만 받는다. GT, donor, 환자 ID는 인수가 아니다. 이미지 밖은 기존 HU 최솟값으로 패딩하며 중심 안쪽과 바깥쪽 모두 같은 보간을 적용한다.

CNN이 32차원 영상 특징맵을 만든다. 고정 공간 격자에서 삼선형 보간으로 특징을 읽고 128차원으로 투영한다. 좌표는 특징 샘플링과 이웃 연결에만 사용한다. 전체 격자에 GATv2 3층과 residual/FFN을 적용하고 attention pooling하여 한 위치의 128차원 data 표현을 얻는다.

외곽 시야는 기존 r2에서 정했던 범위를 그대로 보존한다. 모든 inner-train 적격 병변의 최대 extent 25.290656mm에 기존 2mm와 28mm를 더한 55.290656mm다. 이제 이 값은 전체 시야를 고정하는 데만 사용하며 내부를 지우는 반경은 없다. Validation 병변에 맞춰 반경을 바꾸지 않는다.

전체 엣지를 한꺼번에 확장할 때의 메모리 증가를 피하기 위해 기존 정확한 GATv2 식을 노드 범위별로 계산하고 activation checkpointing을 사용한다. 범위마다 전체 batch와 이웃을 tensor로 처리한다. 노드·엣지를 버리지 않으며 PyG 기준 출력/gradient와 일치하는 검사를 유지한다.

## L1 — data와 관측 label의 관계

Data 하나는 위 L0의 위치 표현이다. 각 support case에는 두 관측 label이 있다. Label0은 주석상 간 비교 위치, label1은 기존 소형 종양의 anchor다. 양성 anchor는 연결 성분 bounding-box의 중심이며 복잡한 모양에서는 중심 voxel 자체가 종양 밖일 수 있다. 이런 좌표를 음성 표본으로 다시 사용하지 않도록 제외한다.

종양 기준은 기존 체적 등가 구 직경20mm 이하이며 모든 적격 anchor를 사용한다. 큰 종양은 원본 GT와 nnU-Net 학습에 그대로 남는다. 소형 종양이 없는 case도 128개 비교 위치와 online CP의 recipient로 참여한다.

Support의 관측 클래스 연결이 T, 반대 클래스 연결이 F다. F는 기하적으로 불가능한 위치나 생물학적으로 암이 절대 없는 위치라는 뜻이 아니다. Query 관계는 U이며 정답을 forward에 전달하지 않는다. 2층 attention에서 support data↔label 관계를 학습하고 query는 support label 상태를 읽는다. Query에서 support로 역방향 메시지를 보내지 않는다.

Label seed는 공유 trainable 128차원 파라미터다. 환자 ID별 암기 테이블이 아니다. 128은 기존 표현 너비를 보존한 값이며 검증된 최적값이라는 주장은 하지 않는다.

## L2 — case별 label 표현의 군집 정렬

Query와 같은 그룹의 모든 기록을 먼저 support에서 제거한다. 나머지 support L1 label을 관측 클래스별로 cosine average-linkage 군집화한다. 실제 해당 클래스 관측이 없는 case의 가상 label은 군집 근거에서 제외한다. 가능한 비단독 군집 cut을 평가해 양의 silhouette가 가장 큰 K를 선택하고 근거가 부족하면 K=1을 기록한다. 임의로 K를 고정하거나 환자를 잘라내지 않는다.

L2는 기존 2층·4-head case 간 attention을 유지한다. Episode 동안 고정한 L1 teacher 중심에 대한 assignment CE로 L2 출력을 정렬한다. Query 분류는 live L2 출력의 군집 중심과 case 수 가중 log-sum-exp를 사용한다. Loss는 weighted query CE + alignment CE다. Query CE는 query CNN/L0·L1·L2까지, alignment CE는 support L1·L2까지 역전파된다.

Support CNN 임베딩은 전체 inner-train을 매 epoch 재계산한 detached memory다. 모든 학습 패치를 query로 방문하는 CE가 CNN/L0를 학습한다. Support CNN을 매 query마다 다시 역전파하는 구조라고 설명하지 않는다. 군집과 loss의 상세 정의·선행연구 적용 범위는 [L2 문서](pipeline_v222_cluster_r2.md)와 [References](../REFERENCES.md)에 보존했다. 그 문서의 r2 입력 가림·CP50%는 역사적 설정이며 현재는 이 문서가 우선한다.

## 데이터 분할과 누수 차단 범위

MSD 원본 CT131case를 같은 split으로 사용한다. Outer-train105case 안에서 inner-train84 / inner-val21이며 outer-val26은 최종 분할 평가용이다. 시야 fit·CNN/GNN 최적화·support·donor는 inner-train을 기준으로 한다. Inner-val은 관측 과제 평가와 checkpoint 선택에 사용하며 outer-val patch는 GNN 캐시에 생성하지 않는다. nnU-Net은 outer-train으로 학습한다.

현재 실행은 공개 case ID 단위 benchmark다. 같은 환자의 재검사 여부가 확인됐다고 표시하지 않는다. `hiercp_public_case_benchmark_v1`에 `patient_independence_verified=false`, `annotation_complete=null`을 기록한다. 확인된 환자 매핑을 사용하는 별도 계약은 유지한다. 중복 decoded CT, split 교차, raw hash, cache/checkpoint source identity를 검사한다. 서로 다른 CT인 같은 환자를 해시만으로 식별할 수는 없다.

GT는 표본 정의와 loss target, donor 추출 및 실제 CP label 갱신에 쓰인다. GT를 CNN 채널로 전달하거나 GT 경계로 L0 그래프를 만들지 않는다. Target은 CT patch와 분리 저장한다. 중심 CT에 실제 종양 appearance가 보이는 것은 이번 승인된 입력의 특성이다. Appearance shortcut 가능성과 관측 분류에서 CP 배치로의 전이는 학습 후 평가해야 한다.

## Online CP와 점수 해석

GNN은 붙이기 전 recipient의 원본 CT 위치를 평가한다. Donor의 모양은 기하 후보 생성과 실제 paste에 사용되지만 CNN/GNN 점수의 입력에는 없다. 기하 필터는 붙일 수 있는지 검사하며 F의 학습 정답을 만들지 않는다.

전체128개 후보를 평가해 관측 클래스 점수의 exact argmax를 선택한다. CP gate가 통과한 training sample은 원본 recipient에 donor를 온라인으로 붙인 CT/GT로 학습하고, 미시도 또는 배치 실패 시 원본 sample을 사용한다. 원본과 CP 결과를 무조건 두 sample로 복제하는 설정이 아니다. Validation에는 CP를 적용하지 않는다.

CP 시도율은 양쪽 모두80%이며 seed42와 공통 gate schedule을 사용한다. 실제 붙이기 성공률은 시도율과 따로 기록한다. Basic CP와 GNN 방법 사이에 donor·크기·crop 차이가 남아 있으므로 최종 비교는 전체 방법 비교이며 GNN 위치 선택만의 인과 효과로 보고하지 않는다.

출력은 실험적인 관측 유사도 순위다. 종양 발생 확률이나 검증된 배치 적합도 확률이 아니다. 중심/주변 appearance 개입 진단과 동일 outer-val의 전체 nnU-Net 비교가 남아 있다. 성능 평가는 기존 `case_metrics.csv`, `lesion_metrics.csv`, `summary.csv`, `size_metrics.csv`로 크기별 recall, lesion Dice, Tumor Dice, FP를 구분한다.

## 실행과 검증 기록

최신 실행 보완과89검사 결과는 [runtime 재점검 보고서](v222_runtime_review_20260923.md)를 기준으로 한다. 같은 case 묶음으로 CPU worker를 보정하고 GAT 실행 타일을 실측에 맞춰 개선했다. Frozen CP scorer는 nnU-Net의 RNG/backend 상태를 저장·복원한다. 아래 batch2/4 수치는 최초 r3 검증 이력이다.

- 구현·회귀44검사 통과: `work/v222_raw_ct_r3_20260923/unit_tests.log`.
- 실제 CT `liver_1/5/108`, 전체45,385노드 그래프의 GPU DEBUG 통과: `work/v222_raw_ct_r3_20260923/gpu_debug/result.json`. 중심·주변 개입이 L0 특징을 바꾸며 모든 파라미터에 finite gradient가 있고 CNN/L0/L1/L2 그룹이 optimizer로 갱신됐다.
- Batch2/4, 3회 측정: 0.825/0.885 graphs/s, peak CUDA 할당1,065,028,608 / 1,982,396,416bytes. Batch4는 두 실제 query의 반복이며 최종 training batch 선택값이 아니다. 전체 support로 재보정한다.
- `liver_108` 비교 후보0개 실패 해소: label1 및 anchor 제외 후3,402,279개 중128개 선택. 원본 GT를 변경하지 않았다.
- 전체131case 새 사전 검사는 완료됐다.105case 모두128개 비교표본을 확보했다. 상세 근거는 `work/v222_raw_ct_r3_20260923/preflight/summary.json`과 [r3 검증 기록](v222_raw_ct_r3_verification_20260923.md)에 기록한다.
- 전체 GNN40epoch, nnU-Net250epoch 학습·평가 성능은 이 DEBUG로 검증되지 않았다. 가짜 checkpoint나 성능 지표를 생성하지 않았다.

기존 cache/checkpoint는 새 format과 source hash 검사에서 거부한다. 학습 명령은 `tools/start_v222_case_training.py`, 개별 단계는 `run_v222.py`의 prepare/train/bank/nnunet-train/predict/evaluate로 연결되어 있다. 기존 결과를 덮어쓰지 않는 새 output 폴더가 필요하다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 문제에 그래프 축소 없이 기존 tiling/checkpointing 경로를 유지했다. 새 DEBUG에서 OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
# 최신 구현: v2.22 r4 — PPR/A* L0

**r4는 방사형 구조를 강제하는 설계 결함으로 반려됐고 학습/production checkpoint 사용이 차단됐다.** 실행 가능한 완성 기준 모델로 취급하지 않는다. 아래 구현 설명과 과거 검증은 이력을 위한 기록이다.

현재 기본 설정은 `hiercp_v222_ppr_astar_cluster_r4`다. [r4 실제 경로·설정·검증·한계](v222_ppr_astar_r4_20260923.md)를 우선한다. 원본 CT와 전체 공간 후보를 유지하되 CNN 특징 기반 PPR와 A*로 최종 GNN 부분 그래프를 선택한다. 아래 r3의 고정 전체 그래프 설명은 이전 이력이다. L1/L2·CP80·seed42·전체 데이터·모델 깊이/폭은 유지한다.
