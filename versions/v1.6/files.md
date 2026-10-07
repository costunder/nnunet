# v1.6 파일 안내

원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.
게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.

<details><summary>구현 (5)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| half b allocation | [열기](../../hiercp_v1x/half_b_allocation.py) | 게시 대상 |
| half b entry | [열기](../../hiercp_v1x/half_b_entry.py) | 게시 대상 |
| half b model | [열기](../../hiercp_v1x/half_b_model.py) | 게시 대상 |
| half b support | [열기](../../hiercp_v1x/half_b_support.py) | 게시 대상 |
| half b training | [열기](../../hiercp_v1x/half_b_training.py) | 게시 대상 |

</details>

<details><summary>실행·분석 (3)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| prepare v1 half b support debug | [열기](../../tools/prepare_v1_half_b_support_debug.py) | 게시 대상 |
| run v1 half b | [열기](../../tools/run_v1_half_b.py) | 게시 대상 |
| v1 half b learning | [열기](../../tools/verify_v1_half_b_learning.py) | 게시 대상 |

</details>

<details><summary>기록 (3)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| v16 half b allocation fix | [열기](../../docs/v16_half_b_allocation_fix_20261004.md) | 게시 대상 |
| v16 half b learning | [열기](../../docs/v16_half_b_learning_20261004.md) | 게시 대상 |
| v16 half b server result | [열기](../../docs/v16_half_b_server_result_20261005.md) | 게시 대상 |

</details>

<details><summary>검사 (7)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| v1 half b allocation | [열기](../../tests/test_v1_half_b_allocation.py) | 게시 대상 |
| v1 half b allocation entry | [열기](../../tests/test_v1_half_b_allocation_entry.py) | 게시 대상 |
| v1 half b entry | [열기](../../tests/test_v1_half_b_entry.py) | 게시 대상 |
| v1 half b epoch reporting | [열기](../../tests/test_v1_half_b_epoch_reporting.py) | 게시 대상 |
| v1 half b learning | [열기](../../tests/test_v1_half_b_learning.py) | 게시 대상 |
| v1 half b model | [열기](../../tests/test_v1_half_b_model.py) | 게시 대상 |
| v1 half b training | [열기](../../tests/test_v1_half_b_training.py) | 게시 대상 |

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