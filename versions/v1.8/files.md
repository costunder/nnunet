# v1.8 파일 안내

원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.
게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.

<details><summary>구현 (8)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| host memory | [열기](../../hiercp_v1x/host_memory.py) | 게시 대상 |
| u bridge calibration | [열기](../../hiercp_v1x/u_bridge_calibration.py) | 게시 대상 |
| u bridge continuation | [열기](../../hiercp_v1x/u_bridge_continuation.py) | 게시 대상 |
| u bridge data | [열기](../../hiercp_v1x/u_bridge_data.py) | 게시 대상 |
| u bridge experiment | [열기](../../hiercp_v1x/u_bridge_experiment.py) | 게시 대상 |
| u bridge fields | [열기](../../hiercp_v1x/u_bridge_fields.py) | 게시 대상 |
| u bridge training | [열기](../../hiercp_v1x/u_bridge_training.py) | 게시 대상 |
| u bridge upper | [열기](../../hiercp_v1x/u_bridge_upper.py) | 게시 대상 |

</details>

<details><summary>설정 (1)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| v18 u bridge | [열기](../../config/v18_u_bridge.json) | 게시 대상 |

</details>

<details><summary>실행·분석 (6)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| run v18 independent | [열기](../../tools/run_v18_independent.py) | 게시 대상 |
| run v18 u bridge | [열기](../../tools/run_v18_u_bridge.py) | 게시 대상 |
| server v18 independent | [열기](../../tools/server_v18_independent.sh) | 게시 대상 |
| server v18 u bridge | [열기](../../tools/server_v18_u_bridge.sh) | 게시 대상 |
| summarize v18 u bridge | [열기](../../tools/summarize_v18_u_bridge.py) | 게시 대상 |
| v18 memory debug | [열기](../../tools/verify_v18_memory_debug.py) | 게시 대상 |

</details>

<details><summary>검사 (9)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| host memory | [열기](../../tests/test_host_memory.py) | 게시 대상 |
| u bridge calibration | [열기](../../tests/test_u_bridge_calibration.py) | 게시 대상 |
| u bridge continuation | [열기](../../tests/test_u_bridge_continuation.py) | 게시 대상 |
| u bridge data | [열기](../../tests/test_u_bridge_data.py) | 게시 대상 |
| u bridge experiment | [열기](../../tests/test_u_bridge_experiment.py) | 게시 대상 |
| u bridge fields | [열기](../../tests/test_u_bridge_fields.py) | 게시 대상 |
| u bridge training | [열기](../../tests/test_u_bridge_training.py) | 게시 대상 |
| u bridge upper | [열기](../../tests/test_u_bridge_upper.py) | 게시 대상 |
| v18 independent | [열기](../../tests/test_v18_independent.py) | 게시 대상 |

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