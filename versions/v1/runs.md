# v1 실험 파일 위치

봉인된 실험 경로는 유지합니다. 이동한 보존물은 원래 이름과 현재 위치를 relocation receipt로 추적합니다.
이름만으로 학습 완료·품질·삭제 가능 여부를 판정하지 않습니다. 생성된 UNIT fixture 묶음은 실험 실행 수가 아닙니다.

<details><summary>work (23)</summary>

| 내용 | 위치 |
| --- | --- |
| native30 full128 DEBUG | [열기](../../work/native30_full128_DEBUG_20261006/) |
| native30 worktree relocation actual DEBUG | [열기](../../work/native30_worktree_relocation_actual_DEBUG_20261006/) |
| v1 bundle STANDALONE DEBUG | [열기](../../work/v1_bundle_STANDALONE_DEBUG_20261003/) |
| v1 current comparison | [열기](../../work/v1_current_comparison_20260925_DEBUG/) |
| v1 epoch telemetry CUDA DEBUG | [열기](../../work/v1_epoch_telemetry_CUDA_DEBUG_20261003/) |
| v1 epoch telemetry CUDA DEBUG | [열기](../../work/v1_epoch_telemetry_CUDA_DEBUG_20261003_r2/) |
| v1 epoch telemetry CUDA DEBUG | [열기](../../work/v1_epoch_telemetry_CUDA_DEBUG_20261003_r3/) |
| v1 epoch telemetry CUDA DEBUG | [열기](../../work/v1_epoch_telemetry_CUDA_DEBUG_20261003_r4/) |
| v1 graph size104 CUDA DEBUG | [열기](../../work/v1_graph_size104_CUDA_DEBUG_20261003/) |
| v1 graph size CUDA DEBUG | [열기](../../work/v1_graph_size_CUDA_DEBUG_20261003/) |
| v1 nested104 CUDA DEBUG | [열기](../../work/v1_nested104_CUDA_DEBUG_20261003/) |
| v1 nested416 CUDA DEBUG | [열기](../../work/v1_nested416_CUDA_DEBUG_20261003/) |
| v1 sampling runtime416 CUDA DEBUG | [열기](../../work/v1_sampling_runtime416_CUDA_DEBUG_20261003/) |
| v1 sampling runtime CUDA DEBUG | [열기](../../work/v1_sampling_runtime_CUDA_DEBUG_20261003/) |
| v1 sampling runtime CUDA DEBUG | [열기](../../work/v1_sampling_runtime_CUDA_DEBUG_20261003_r2/) |
| v1x real CT DEBUG | [열기](../../work/v1x_real_CT_DEBUG_20261003_prepare/) |
| v1x real CUDA DEBUG | [열기](../../work/v1x_real_CUDA_DEBUG_20261003/) |
| v1x real CUDA DEBUG | [열기](../../work/v1x_real_CUDA_DEBUG_20261003_CNNretained/) |
| v1x real CUDA DEBUG | [열기](../../work/v1x_real_CUDA_DEBUG_20261003_final/) |
| v1x suite | [열기](../../work/v1x_suite_20261003/) |
| v1x suite r2 | [열기](../../work/v1x_suite_r2_20261003/) |
| v1x suite r3 | [열기](../../work/v1x_suite_r3_20261003/) |
| v1x suite r4 | [열기](../../work/v1x_suite_r4_20261003/) |

</details>

<details><summary>validation (4)</summary>

| 내용 | 위치 |
| --- | --- |
| native30 all lesions reuse DEBUG | [열기](../../validation/native30_all_lesions_reuse_DEBUG_20261006/) |
| native30 full128 DEBUG | [열기](../../validation/native30_full128_DEBUG_20261006/) |
| native30 geometry relocation DEBUG | [열기](../../validation/native30_geometry_relocation_DEBUG_20261006/) |
| v1x progressive | [열기](../../validation/v1x_progressive_20261003/) |

</details>

<details><summary>exports (1)</summary>

| 내용 | 위치 |
| --- | --- |
| HierCP v1x REVIEW | [열기](../../exports/HierCP_v1x_REVIEW_20261003/) |

</details>

<details><summary>archived_tests (8)</summary>

| 내용 | 위치 |
| --- | --- |
| bytecode (4개 fixture) | [열기](../../work/archive/tests/v1/bytecode/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| roi-parser (10개 fixture) | [열기](../../work/archive/tests/v1/roi-parser/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| roi-recovery (5개 fixture) | [열기](../../work/archive/tests/v1/roi-recovery/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| roi-reuse (40개 fixture) | [열기](../../work/archive/tests/v1/roi-reuse/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| runtime (9개 fixture) | [열기](../../work/archive/tests/v1/runtime/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| scope-controller (26개 fixture) | [열기](../../work/archive/tests/v1/scope-controller/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| serialization (1개 fixture) | [열기](../../work/archive/tests/v1/serialization/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| telemetry (18개 fixture) | [열기](../../work/archive/tests/v1/telemetry/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |

</details>


## 작업 완료 체크리스트

이 파일은 탐색 인덱스다. 학습·평가·자원 benchmark와 데이터 이동·삭제는 실행하지 않는다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. (인덱스 생성에 해당 없음)
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. (인덱스 생성에 해당 없음)
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (인덱스 생성에 해당 없음)
- [x] 디버그 설정과 최종 설정을 분리해서 안내했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (새 실행 검사 없음)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 기존 구현·기록으로 연결한다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
