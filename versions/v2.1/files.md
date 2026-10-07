# v2.1 파일 안내

원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.
게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.

<details><summary>구현 (15)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
|   init   | [열기](../../hiercp_v2/__init__.py) | 게시 대상 |
| bank | [열기](../../hiercp_v2/bank.py) | 게시 대상 |
| contracts | [열기](../../hiercp_v2/contracts.py) | 게시 대상 |
| data | [열기](../../hiercp_v2/data.py) | 게시 대상 |
| donors | [열기](../../hiercp_v2/donors.py) | 게시 대상 |
| evaluation | [열기](../../hiercp_v2/evaluation.py) | 게시 대상 |
| model | [열기](../../hiercp_v2/model.py) | 게시 대상 |
| native adapter | [열기](../../hiercp_v2/native_adapter.py) | 게시 대상 |
| nnunet | [열기](../../hiercp_v2/nnunet.py) | 게시 대상 |
| nnUNetTrainer OnlinePromptGraphV2 | [열기](../../hiercp_v2/nnUNetTrainer_OnlinePromptGraphV2.py) | 게시 대상 |
| parallel | [열기](../../hiercp_v2/parallel.py) | 게시 대상 |
| scoring | [열기](../../hiercp_v2/scoring.py) | 게시 대상 |
| storage | [열기](../../hiercp_v2/storage.py) | 게시 대상 |
| training | [열기](../../hiercp_v2/training.py) | 게시 대상 |
| volumes | [열기](../../hiercp_v2/volumes.py) | 게시 대상 |

</details>

<details><summary>설정 (2)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| prompt graph v2 | [열기](../../config/prompt_graph_v2.json) | 게시 대상 |
| runtime windows v21 | [열기](../../config/runtime_windows_v21.txt) | 로컬 |

</details>

<details><summary>실행·분석 (8)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| run v2 | [열기](../../run_v2.py) | 게시 대상 |
| prepare local v21 dataset | [열기](../../tools/prepare_local_v21_dataset.py) | 로컬 |
| run local v21 gnn | [열기](../../tools/run_local_v21_gnn.py) | 로컬 |
| prompt graph v2 debug | [열기](../../tools/verify_prompt_graph_v2_debug.py) | 로컬 |
| v21 cardinality debug | [열기](../../tools/verify_v21_cardinality_debug.py) | 로컬 |
| v21 cuda smoke | [열기](../../tools/verify_v21_cuda_smoke.py) | 로컬 |
| v21 real cuda debug | [열기](../../tools/verify_v21_real_cuda_debug.py) | 로컬 |
| v21 shared cp real debug | [열기](../../tools/verify_v21_shared_cp_real_debug.py) | 로컬 |

</details>

<details><summary>기록 (3)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| local v21 5070ti run | [열기](../../docs/local_v21_5070ti_run.md) | 게시 대상 |
| pipeline v2 | [열기](../../docs/pipeline_v2.md) | 게시 대상 |
| shared donor leakage v21 | [열기](../../docs/shared_donor_leakage_v21.md) | 게시 대상 |

</details>

<details><summary>검사 (3)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| prompt graph v2 debug | [열기](../../tests/test_prompt_graph_v2_debug.py) | 로컬 |
| prompt graph v2 io debug | [열기](../../tests/test_prompt_graph_v2_io_debug.py) | 로컬 |
| shared donor leakage debug | [열기](../../tests/test_shared_donor_leakage_debug.py) | 로컬 |

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