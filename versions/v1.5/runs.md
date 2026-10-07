# v1.5 실험 파일 위치

봉인된 실험 경로는 유지합니다. 이동한 보존물은 원래 이름과 현재 위치를 relocation receipt로 추적합니다.
이름만으로 학습 완료·품질·삭제 가능 여부를 판정하지 않습니다. 생성된 UNIT fixture 묶음은 실험 실행 수가 아닙니다.

<details><summary>work (2)</summary>

| 내용 | 위치 |
| --- | --- |
| v1 half a learning DEBUG | [열기](../../work/v1_half_a_learning_DEBUG_20261004_r1/) |
| v1 half a learning DEBUG | [열기](../../work/v1_half_a_learning_DEBUG_20261004_r2/) |

</details>

<details><summary>validation (2)</summary>

| 내용 | 위치 |
| --- | --- |
| v15 half a | [열기](../../validation/v15_half_a_20261004/) |
| v15 half a server | [열기](../../validation/v15_half_a_server_20261004/) |

</details>

<details><summary>archived_tests (1)</summary>

| 내용 | 위치 |
| --- | --- |
| half-a (8개 fixture) | [열기](../../work/archive/tests/v1/half-a/) · [원래→현재](../../work/archive/organization/tests-20261007-234732/plan.json) |

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
