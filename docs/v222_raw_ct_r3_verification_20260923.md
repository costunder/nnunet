# v2.22 r3 원본 CT 수정·검증 기록

2026-09-23. 사용자 응답 “이 전체 수정안으로 구현·검증 진행”에 근거하여 적용했다. 이전의 자동 승인 차단은 이 전체 변경안의 명시적 승인을 받은 뒤 해소됐다.

**최신 상태:** 전체131case 검사 완료. 이후 CPU/GPU 실행과 frozen scorer 전역 상태 결함을 추가 수정하고89검사를 통과했다. 아래 기록은 최초 r3 적용 시점이며 현재 실행·변경 파일은 [runtime 재점검](v222_runtime_review_20260923.md)을 우선한다.

## 변경과 보존

- `hiercp_v222/inputs.py`: 중심 치환을 제거하고 원본 CT를 보간한다. 고정 outer FOV 안쪽까지 일반 공간 격자로 포함한다.
- `hiercp_v222/data.py`: label1 간 위치에서 동일 양성 anchor를 제외하여 case당128개를 고른다. 모든 종양에서의 거리 제한과 그 거리 변환 계산을 제거했다. 모든105case의 비교 표본 검사를 patch 생성 전에 수행한다.
- 새 input/pipeline format으로 이전 masked cache·checkpoint 혼용을 차단한다. CNN/L1/L2 규모, 전체 cohort, seed42, CP80%를 유지한다.
- 이전 구현·설정·검사·문서는 `versions/v2.22/before_raw_ct_r3_20260923/`에 보존했다.
- `model.py`, `clustering.py`, `training.py`, `scoring.py`, `bank.py`는 수정 전과 바이트 단위로 일치한다. Frozen v1 revision `74dcc2cf03d2d40d1f582223321d96004333f661` 검사 통과.

## 확인된 결과

| 검증 | 실제 결과 |
| --- | --- |
| 정적 검사 | Python18파일 통과 |
| 단위·회귀 |44검사 통과. CT 보간 독립 reference, 기존 외곽 전체 노드·엣지 보존, 근처 간 비교 표본, query/support 분리, 두 loss 경로 포함 |
| 실제 CT GPU DEBUG |liver_1/5/108, 6개 그래프. 전체 inner-train으로 정한 반경55.290656mm와 전체45,385노드/2,383,482엣지 사용 |
| 모델 |전체1,519,063파라미터, CNN/L0/L1/L2 gradient 유한·optimizer 갱신 확인 |
| 입력 개입 |중심5mm 및 10mm 옆 주변 영역 개입이 각각 CT 입력·L0 표현에 도달 |
| liver_108 |기존 거리 후보0 → 새 조건3,402,279개 중128개 선택. 원본 GT 변경 없음 |
| 전체 cohort |raw CT/GT131case 검사 완료.105case 모두128개 비교 표본, outer-val26case 표집 제외 |
| 본 학습·평가 |새 GNN40epoch 및 nnU-Net250epoch/전체 평가 아직 미실행 |

GPU DEBUG는 세 case의 두 query를 사용한 실행·자원 검사다. Batch4에서는 두 query를 반복했다. Support가 클래스당 두 case여서 K=1이며 다중 군집 임상 검증은 아니다. 최종 training batch를2나4로 확정하지 않았다.

| physical / effective batch | 반복 | 처리량(graphs/s) | peak CUDA 할당(bytes) |
| --- | --- | --- | --- |
|2 / 2|3|0.825170|1065028608|
|4 / 4|3|0.885002|1982396416|

자원은 RTX5070Ti 16GB 한 대, RAM 약64GiB, CPU 할당16논리코어다. GPU 결과의 CPU/RAM 측정과 시작 소스 해시는 receipt에 기록했다. 전체 graph 크기를 유지한 실행 tiling/checkpointing을 사용했고 이번 DEBUG에서 OOM은 없었다.

전체 cohort 감사의 worker1/2 파동은 서로 다른 case로 측정되어 절대적인 최적 worker 증거가 아니다. 크기 차이 때문에 worker1이 선택된 이번 읽기 전용 감사 결과를 최종 학습 loader 설정으로 재사용하지 않는다. 최종 학습은 기존 full-topology physical batch 보정과 worker0/2/4/8 실제 읽기 측정을 별도로 수행한다.

## 원본 근거

- [회귀 로그](../work/v222_raw_ct_r3_20260923/unit_tests.log)
- [GPU 실제 CT 결과](../work/v222_raw_ct_r3_20260923/gpu_debug/result.json)
- [입력·중심/주변 개입 기록](../work/v222_raw_ct_r3_20260923/gpu_debug/inputs.json)
- [정적 검사·보존 확인](../work/v222_raw_ct_r3_20260923/static_and_preservation.json)
- [전체 cohort 완료 receipt](../work/v222_raw_ct_r3_20260923/preflight/summary.json)
- [현재 파이프라인](pipeline_v222.md)

## 남은 검증

관측 분류 점수가 좋은 CP 위치를 고른다는 효용은 아직 입증하지 않았다. 전체 GNN 학습 후 중심 appearance/주변 context 개입 진단, 전체 nnU-Net 학습과 동일 outer-val의 Basic CP80 비교가 필요하다. 현재 데이터는 공개 case 단위이며 재검사 환자 독립성과 주석 완전성을 보장하지 않는다. Donor·크기·crop 차이가 있는 전체 방법 비교라는 한계를 유지한다.

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
