# v1.7 실험 파일 위치

봉인된 실험 경로는 유지합니다. 이동한 보존물은 원래 이름과 현재 위치를 relocation receipt로 추적합니다.
이름만으로 학습 완료·품질·삭제 가능 여부를 판정하지 않습니다. 생성된 UNIT fixture 묶음은 실험 실행 수가 아닙니다.

<details><summary>work (19)</summary>

| 내용 | 위치 |
| --- | --- |
| v17 actual nonempty reuse DEBUG | [열기](../../work/v17_actual_nonempty_reuse_DEBUG_20261005_155506/) |
| v17 actual nonempty reuse FINAL DEBUG | [열기](../../work/v17_actual_nonempty_reuse_FINAL_DEBUG_20261005_160121/) |
| v17 actual storage reuse DEBUG | [열기](../../work/v17_actual_storage_reuse_DEBUG_20261005_143301/) |
| v17 admitted reuse STORAGE DEBUG | [열기](../../work/v17_admitted_reuse_STORAGE_DEBUG_20261005_144456/) |
| v17 C cuda DEBUG | [열기](../../work/v17_C_cuda_DEBUG_20261005_r1/) |
| v17 C cuda DEBUG | [열기](../../work/v17_C_cuda_DEBUG_20261005_r2/) |
| v17 C cuda DEBUG | [열기](../../work/v17_C_cuda_DEBUG_20261005_r3/) |
| v17 C cuda DEBUG | [열기](../../work/v17_C_cuda_DEBUG_20261005_r4/) |
| v17 crossed native DEBUG | [열기](../../work/v17_crossed_native_DEBUG_20261005/) |
| v17 D cuda DEBUG | [열기](../../work/v17_D_cuda_DEBUG_20261005_r1/) |
| v17 D cuda DEBUG | [열기](../../work/v17_D_cuda_DEBUG_20261005_r2/) |
| v17 D cuda DEBUG | [열기](../../work/v17_D_cuda_DEBUG_20261005_r3/) |
| v17 empty context actual DEBUG | [열기](../../work/v17_empty_context_actual_DEBUG_20261005_after/) |
| v17 empty context actual DEBUG | [열기](../../work/v17_empty_context_actual_DEBUG_20261005_before/) |
| v17 empty context CUDA DEBUG | [열기](../../work/v17_empty_context_CUDA_DEBUG_20261005_r1/) |
| v17 fastprep D cuda DEBUG | [열기](../../work/v17_fastprep_D_cuda_DEBUG_20261005_r1/) |
| v17 fastprep D cuda DEBUG | [열기](../../work/v17_fastprep_D_cuda_DEBUG_20261005_r2/) |
| v17 preparation execution DEBUG | [열기](../../work/v17_preparation_execution_DEBUG_20261005/) |
| v17 reuse checkout provenance DEBUG | [열기](../../work/v17_reuse_checkout_provenance_DEBUG_20261005_155956/) |

</details>

<details><summary>validation (14)</summary>

| 내용 | 위치 |
| --- | --- |
| v17 crossed training | [열기](../../validation/v17_crossed_training_20261005/) |
| v17 historical eval DEBUG | [열기](../../validation/v17_historical_eval_DEBUG_20261005/) |
| v17 historical eval DEBUG | [열기](../../validation/v17_historical_eval_DEBUG_20261005_r2/) |
| v17 historical eval DEBUG | [열기](../../validation/v17_historical_eval_DEBUG_20261005_r3/) |
| v17 historical eval DEBUG | [열기](../../validation/v17_historical_eval_DEBUG_20261005_r4/) |
| v17 historical eval DEBUG | [열기](../../validation/v17_historical_eval_DEBUG_20261005_r5/) |
| v17 historical eval DEBUG | [열기](../../validation/v17_historical_eval_DEBUG_20261005_r6/) |
| v17 historical full128 server | [열기](../../validation/v17_historical_full128_server_20261005/) |
| v17 preparation execution | [열기](../../validation/v17_preparation_execution_20261005/) |
| v17 recipient context | [열기](../../validation/v17_recipient_context_20261005/) |
| v17 reference reuse identity | [열기](../../validation/v17_reference_reuse_identity_20261005/) |
| v17 terminal | [열기](../../validation/v17_terminal_20261005/) |
| v17 transition | [열기](../../validation/v17_transition_20261005/) |
| v17 transition server | [열기](../../validation/v17_transition_server_20261005/) |

</details>

<details><summary>archived_tests (5)</summary>

| 내용 | 위치 |
| --- | --- |
| metadata (47개 fixture) | [열기](../../work/archive/tests/transition/metadata/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| parallel-failure (4개 fixture) | [열기](../../work/archive/tests/transition/parallel-failure/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| parallel-pass (4개 fixture) | [열기](../../work/archive/tests/transition/parallel-pass/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| prepare (10개 fixture) | [열기](../../work/archive/tests/transition/prepare/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |
| storage (75개 fixture) | [열기](../../work/archive/tests/transition/storage/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |

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
