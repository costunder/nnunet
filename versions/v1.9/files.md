# v1.9 파일 안내

원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.
게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.

<details><summary>구현 (13)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| arm process | [열기](../../hiercp_v1x/arm_process.py) | 게시 대상 |
| comparison data | [열기](../../hiercp_v1x/comparison_data.py) | 게시 대상 |
| comparison data timing | [열기](../../hiercp_v1x/comparison_data_timing.py) | 게시 대상 |
| comparison execution | [열기](../../hiercp_v1x/comparison_execution.py) | 게시 대상 |
| comparison experiment | [열기](../../hiercp_v1x/comparison_experiment.py) | 게시 대상 |
| comparison inputs | [열기](../../hiercp_v1x/comparison_inputs.py) | 게시 대상 |
| comparison progress | [열기](../../hiercp_v1x/comparison_progress.py) | 게시 대상 |
| comparison runtime | [열기](../../hiercp_v1x/comparison_runtime.py) | 게시 대상 |
| comparison training | [열기](../../hiercp_v1x/comparison_training.py) | 게시 대상 |
| comparison views | [열기](../../hiercp_v1x/comparison_views.py) | 게시 대상 |
| owned continuation | [열기](../../hiercp_v1x/owned_continuation.py) | 게시 대상 |
| preparation reuse | [열기](../../hiercp_v1x/preparation_reuse.py) | 게시 대상 |
| preparation reuse | [열기](../../l0_regions/preparation_reuse.py) | 게시 대상 |

</details>

<details><summary>설정 (2)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| comparison cp80 | [열기](../../config/comparison_cp80.json) | 게시 대상 |
| v19 comparison controls | [열기](../../config/v19_comparison_controls.json) | 게시 대상 |

</details>

<details><summary>실행·분석 (13)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| report comparison runtime | [열기](../../tools/report_comparison_runtime.py) | 게시 대상 |
| resume comparison cached | [열기](../../tools/resume_comparison_cached.py) | 게시 대상 |
| run comparison arm | [열기](../../tools/run_comparison_arm.py) | 게시 대상 |
| run v19 comparison | [열기](../../tools/run_v19_comparison.py) | 게시 대상 |
| server comparison cached | [열기](../../tools/server_comparison_cached.sh) | 게시 대상 |
| server v19 comparison | [열기](../../tools/server_v19_comparison.sh) | 게시 대상 |
| stop comparison arm | [열기](../../tools/stop_comparison_arm.py) | 게시 대상 |
| summarize v19 comparison | [열기](../../tools/summarize_v19_comparison.py) | 게시 대상 |
| v19 reference execution | [열기](../../tools/v19_reference_execution.py) | 게시 대상 |
| arm launch debug | [열기](../../tools/verify_arm_launch_debug.py) | 게시 대상 |
| comparison cache debug | [열기](../../tools/verify_comparison_cache_debug.py) | 게시 대상 |
| comparison input debug | [열기](../../tools/verify_comparison_input_debug.py) | 게시 대상 |
| comparison progress debug | [열기](../../tools/verify_comparison_progress_debug.py) | 게시 대상 |

</details>

<details><summary>기록 (1)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| comparison runtime | [열기](../../docs/comparison_runtime.md) | 게시 대상 |

</details>

<details><summary>검사 (18)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| arm process | [열기](../../tests/test_arm_process.py) | 게시 대상 |
| comparison arm launch | [열기](../../tests/test_comparison_arm_launch.py) | 게시 대상 |
| comparison cache logging | [열기](../../tests/test_comparison_cache_logging.py) | 게시 대상 |
| comparison cached resume | [열기](../../tests/test_comparison_cached_resume.py) | 게시 대상 |
| comparison execution | [열기](../../tests/test_comparison_execution.py) | 게시 대상 |
| comparison experiment | [열기](../../tests/test_comparison_experiment.py) | 게시 대상 |
| comparison inputs | [열기](../../tests/test_comparison_inputs.py) | 게시 대상 |
| comparison progress debug | [열기](../../tests/test_comparison_progress_debug.py) | 게시 대상 |
| comparison reference | [열기](../../tests/test_comparison_reference.py) | 게시 대상 |
| comparison runtime report | [열기](../../tests/test_comparison_runtime_report.py) | 게시 대상 |
| comparison server | [열기](../../tests/test_comparison_server.py) | 게시 대상 |
| comparison summary | [열기](../../tests/test_comparison_summary.py) | 게시 대상 |
| comparison training | [열기](../../tests/test_comparison_training.py) | 게시 대상 |
| comparison views | [열기](../../tests/test_comparison_views.py) | 게시 대상 |
| owned continuation | [열기](../../tests/test_owned_continuation.py) | 게시 대상 |
| preparation reuse | [열기](../../tests/test_preparation_reuse.py) | 게시 대상 |
| preparation reuse hotpath | [열기](../../tests/test_preparation_reuse_hotpath.py) | 게시 대상 |
| stop comparison arm | [열기](../../tests/test_stop_comparison_arm.py) | 게시 대상 |

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