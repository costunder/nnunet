# v2.2 희소 L0: 48 / 96 / 192 context node 실제 CT 비교

2026-10-02. 별도 DEBUG 구현과 비용·공간 표현 검사. 기존 `64b82e4`의 48 context node 검사와 결과는 보존한다. 사용자가 노드 충분성 비교를 승인한 뒤 96·192 설정을 추가했다. Production 기본 모델이나 최종 노드 수를 결정한 결과는 아니다.

## 결과와 판단

같은 CT, 같은 CNN 특징 pool, 같은 초기 가중치와 physical batch32에서 세 설정을 실제 CUDA로 실행했다. 원래 liver_66의 P5와 U128을 전부 사용했다. Recipient133개의 crop에 있는 유효 위치 전체를 집계한 값은 다음과 같다.

| Context nodes | 실제 scene nodes | 전체266 scene의 무방향 edge 범위 | CNN cosine deficit p95 | 공간 거리 p95, mm | Joint 거리 p95 | Update 중앙값, 초 |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 49 | 122–156 | 0.315167 | 15.007869 | 1.154220 | 0.481047 |
| 96 | 97 | 229–280 | 0.240168 | 11.692280 | 0.840885 | 0.659644 |
| 192 | 193 | 453–510 | 0.180033 | 9.980477 | 0.612717 | 0.994302 |

48→96에서 CNN cosine deficit p95가23.80%, 96→192에서25.04% 감소했다. 이 검사에서는48에서 coverage가 포화됐다는 증거가 없다. 192에서도 개선이 남으므로192가 최적이거나 충분하다는 결론도 낼 수 없다. 대표점을 늘리면 최근접 거리가 감소하는 것은 nested selection의 예상 성질이다. 이 감소를 암 관련 정보 보존율, CP 정확도 상승률 또는 학습 완료로 해석하지 않는다.

Update는 CNN부터 L1/L2, 전체 loss, backward, finite 검사·clip·AdamW까지 포함한다. 입력은 미리 GPU에 전달된 physical32이다. Loader·전송·checkpoint 저장·진단용 CPU hash/delta 복사는 제외했다. 따라서 서버의 전체 epoch 비용과 같은 범위가 아니며 기존3시간/epoch 문제가 해결됐다고 보고하지 않는다.

## 고정한 계약과 실제 구성

- P는 실제 관측된 적격 종양 anchor, U는 기존 미관측 비교 위치다. Donor별 CP 적합/부적합 GT를 만들지 않았다. 같은 recipient·같은 donor의 기존 목록을 사용했다.
- 원래 Basic CP80%, seed42, split, observation, 후보128, 원본 paste mask, L1/L2 연산·차원·층수와 loss/multiplicity 계산을 보존했다.
- L0는 기존 native organ-only 국소 CNN8Conv `[12,24,32]`, feature stride1/2/4, CNN68D 특징 및 상대 좌표, 3층128D mean GraphSAGE, query/near/mid/wide readout, 기존 paired fusion, 출력128D다. 세 설정 모두 전체/trainable parameters **1,242,198**로 같다.
- 이 입력의 국소 범위는 원래 donor bbox와 margin10mm로 정한 crop이다. Near≤5mm, mid `(5,10]`mm, wide는 crop 안의 나머지 유효 위치다. Margin10을 전체 수용 반경10mm라고 해석하지 않는다. Query readout 반경3mm도 고정했다.
- 각 band의 context quota만16/32/64로 바꿔 context48/96/192를 비교한다. Abstract query1개가 더해져 실제 scene49/97/193, donor＋recipient pair98/194/386개다. 이 세 quota는 명시적인 DEBUG 후보이며 production cap/default 변경이 아니다.
- 모든 실제 sampled context node는 간 안에 있다. 간 밖 CT는 organ-aware CNN 입력에서 차단된다. 원본 anchor는 이동하지 않으며 bbox 중점이 간 밖인 경우도 abstract query로 유지한다. 원본 종양 주석은 화면 overlay로 별도 표시하며 노드 선택에 넣지 않는다.

## 노드 선택과 edge

위치만 균일 배치하지 않는다. 각 band에서 anchor에 가장 가까운 위치로 시작하고, CNN 특징과 공간의 합성 거리로 farthest-point 대표점을 고른다. 두 성분은 band의 **전체 pool** 분산으로 정규화하며 quota에 따라 분모를 바꾸지 않는다. 작은 quota의 선택 집합이 큰 quota에 포함되는지 검사했다.

선택 index는 이산적이고 detach되지만 선택된 현재 CNN 특징은 다시 gather하여 CNN까지 gradient를 전달한다. 이것은 CP loss로 연속 좌표나 분기 방향을 학습한 탐색기가 아니다. 암에 유용한 위치를 골랐다는 의미도 아니다.

Edge는 기존 spatial3NN＋feature1NN의 symmetric union이다. 연결이 끊겼을 때만 기존 MST 연결 검사 경로가 있으나 이번 세 설정의266 scene 모두 이미 연결돼 추가 MST edge는0이었다. 화면의 edge는 feature message 전달 관계이며 혈관이나 CT를 읽는 선분이 아니다. Crop 단위 공유 CNN과 batched scene/message passing을 사용한다. CT 전체를 환자당 한 번만 encode했다고 주장하지 않는다.

## Coverage의 정확한 분모와 식

Recipient133 crop의 **유효 fine 위치1,938,845개 occurrence**를 사용했다. Crop이 겹치면 같은 원본 voxel이 다른 candidate의 위치로 다시 포함된다. 이것은 간 전체의 unique voxel 수나133개의 후보 중심 거리 통계가 아니다. 반복 donor는 recipient 통계에 섞지 않고 unique donor1scene의9,326개 위치로 따로 집계한다.

원래 pool 중 세 feature scale에서 모두 지원되는 위치를 사용한다. Coarse support를 만족하지 않는 위치는 기존 규칙대로 별도 count/비율로 기록했다. 실제 가장 큰 제외 비율은14.2231%이며 세 quota에서 pool·좌표·지원 mask·band가 정확히 같다. 새 quota에 맞춰 추가 제외하거나 sampling하지 않았다.

각 fine 위치i와 같은 band의 selected context j 사이에서 다음을 계산한다. Abstract query는 대표점 통계에서 제외한다.

```text
feature_deficit(i) = 1 - max_j cosine(f_i, f_j)
spatial_mm(i)     = min_j ||x_i - x_j||_2
joint(i)          = min_j [||x_i-x_j||² / D_space
                           + ||normalize(f_i)-normalize(f_j)||² / D_feature]
D_space, D_feature = 해당 band 전체 pool의 2 × 평균 제곱 편차
```

Joint는 공간과 특징의 최근접 j를 각각 골라 더하는 식이 아니다. 두 성분을 **같은 j**에서 계산한 뒤 최소를 구한다. Spatial은 native-mm Euclidean 거리이며 간 내부 geodesic 거리라고 주장하지 않는다. Selected 위치 자체의0거리도 포함한다. Scene·band별 mean/p95/max와 전체 recipient pooled mean/p95/max를 각각 저장했다. Scene별p95의 평균은 pooled p95와 다른 값이다.

256MiB의 명시적 workspace 안에서 CUDA bmm로 chunking하되 모든 fine 위치와 모든 selected node를 처리한다. NaN/Inf·norm overflow·zero norm·잘못된 index는 명시적 오류이며 부분 sample, 빈 값, CPU fallback으로 숨기지 않는다.

## 가중치와 측정 공정성

CNN 및 기존 pair fusion의 출처는 로컬 DEBUG `work/local_cnn_experiment_resume_DEBUG_20261001/resumed/attempts/0002/checkpoint_latest.pt`, **saved step4**다. SHA256은 `9b36e85a57819042026b152e7bcc7236ed9ad0b372fa32b5d5c321d74b50c1f6`이다. 서버의 epoch22/39 학습 모델을 검사한 결과가 아니다. 새 node projection·GraphSAGE·graph readout은 seed42 초기값이다.

세 설정의 전체 초기 state hash는 동일한 `6bd04b2259a547967e27ee90eeba81344e93bfddeb525fb72450c0f37160b047`이다. 각 설정은 기존 DEBUG support의 유효6record를 새 L0로 재인코딩하고 teacher를 다시 fit한다. Old mean-pooling memory와 새 graph query를 혼합하지 않았다. 같은 support IDs를 사용했다.

원본 case의133관측을32/32/32/32/5로 forward하여 설정마다 donor＋recipient266scene를 검사했다. Update 비용은 동일 P5＋U27 tile, physical/effective32, accumulation1, ranking135쌍, 전체 DEBUG context139의 기존 정규화다. 각 설정에서 warmup1회 후3회 측정했다. **매번 같은 초기 model로 복원하고 fresh AdamW를 사용**하며 dropout/RNG 조건도 일치시킨다. 3회 연속 학습한 뒤의 성능을 비교한 것이 아니다.

각 warmup 및 측정 update에서 CNN/node projection/SAGE/graph readout/fusion/L1/L2의7개 gradient와 실제 parameter delta가 finite이고0보다 큰지 확인했다. Initial state, 원본 checkpoint, assignment와 production121파일을 보존했다. 초기 전체 case 점수는 실행 확인으로 raw report에 보관하지만 `accuracy_evidence=false`이며 후보 수 선택의 정확도 근거로 사용하지 않는다.

## 비용과 자원

GPU는 RTX5070Ti16GiB1개, logical CPU16, 시작 시 available RAM 약43.65GiB다. FP32, autocast/TF32 false, deterministic true, workers4, CUDA12GiB/RSS32GiB/resident12GiB 제한을 명시했다. 데이터는 기존 RAM/crop 재사용과 batch donor 중복 제거를 사용한다. 입력 해상도·CNN 폭/깊이·GNN 깊이·physical batch를 줄이지 않았다. 현재 작업은 로컬 Windows이며 SSH나 스케줄러에 접근하지 않았다.

| Context nodes | Forward 평균, 초 | Backward 평균, 초 | Finite·clip·optimizer 평균, 초 | CUDA 선택 평균, 초 | CUDA edge 구성 평균, 초 | CUDA SAGE3 평균, 초 | Update peak allocated, GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.351899 | 0.086118 | 0.037510 | 0.189375 | 0.021630 | 0.002231 | 2.176200 |
| 96 | 0.535433 | 0.085473 | 0.037871 | 0.357533 | 0.040040 | 0.001785 | 2.176230 |
| 192 | 0.871110 | 0.084070 | 0.037296 | 0.674556 | 0.069740 | 0.001771 | 2.176291 |

CUDA CNN phase 평균은 각 설정 약0.009초였다. Phase값은 CUDA event, update값은 GPU 동기화를 포함한 wall time이어서 측정 경계가 완전히 같지 않다. 이번 새 경로에서는 **대표점 선택이 큰 비용**을 차지하며 GNN만 더 가볍게 해도 이 부분은 사라지지 않는다.

Coverage 진단 전체의 peak allocated는2.634728GiB, 프로세스RSS는 약5.785GiB다. 위 표의 update peak와 구분한다. Support 재encode＋teacher 합계37.724초도 별도로 기록했다. 전체 server memory refresh·validation·checkpoint I/O, 여러 case의 크기 편차·worst-case, 전체 epoch 처리량은 미측정이다.

## 실행과 검증 기록

- GPU unit **17PASS**, 4.904초. Same-j joint의 독립oracle, retain-all, nested coverage, chunk 경계, padding, 비유한값, quota, 전체 모듈gradient 등을 포함한다.
- 실제CT의 전체133관측·각266scene, 동일pool/weight/support, nested selection, 전체 위치 거리의 비증가, 전체loss/backward/optimizer를 검사했다.
- 3D 화면 **14PASS**, runtime errors0. 세 설정을 같은 native-mm 축과 camera로 비교하며 P/U·donor 전환, 전체 간/국소, 원본 종양 overlay, node의 실제mm, 회전, 360px 화면을 확인했다. 표시는P5＋U3와 donor지만 metric/forward는 전체P5＋U128을 사용했다.
- 별도agent가 raw report와 구현6SHA, production121SHA, 기존dirty3SHA, checkpoint, HTML payload/bytes/hash를 독립 확인해PASS. 기존 결과를 덮어쓰지 않았다.
- 장기GNN/nnU-Net/서버 학습, production checkpoint/ready, 전체dataset 평가는 실행하지 않았다.

재현 명령은 기존 로컬DEBUG run/assignment가 있는 환경용이다. 매번 새로운 output을 지정한다. 필요한 입력이 없으면 명시적으로 실패한다. 서버 전체 학습을 시작하는 명령이 아니다.

```powershell
.\.venv\Scripts\python.exe tools/compare_sparse_feature_budgets_debug.py `
  --run work/local_cnn_experiment_resume_DEBUG_20261001/resumed `
  --assignment work/v222_v1_full_training_20260924/cache/pair_assignment.json `
  --assignment-receipt validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json `
  --case liver_66 `
  --output work/sparse_feature_budget_CT_DEBUG_20261002_new `
  --visual "C:/Users/user/.codex/visualizations/2026/09/17/01a0ae06-ab24-7fe0-bdb8-ef2751a1c7be/ct-sparse-budget-comparison-new.html" `
  --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 4 --cuda-gib 12 --rss-gib 32 --resident-gib 12 `
  --coverage-workspace-mib 256 --measure-repeats 3
```

원시 기록은 `validation/l0_sparse_feature_budget_DEBUG_20261002/`의 report.json, unit_checks.txt, browser_checks.json, publication_receipt.json이다. 실제CT/mask/weights/해부학3Dpayload와 이미지는 Git에 공개하지 않는다. 로컬 표시의 SHA256은 `dab91d0ef95b50f3f10285ddef0b24f87aabbd09aa9d453f40a4e9d6026fdb03`, 458,959bytes다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인한48/96/192 DEBUG 비교와 전체133관측을 사용했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존32 유지, batched GPU node/edge/message 계산.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행은 OOM 없고 workspace·peak·phase 비용을 계측했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 새 학습 가능 가중치의 초기화와 실제 학습 결과를 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
