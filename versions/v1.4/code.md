# V1.4 코드

| 역할 | 실제 파일 |
| --- | --- |
| 사용자 진입점 | [run_v14_scope_training.py](../../tools/run_v14_scope_training.py) |
| 기존 초기화·요청 controller | [run_v1_bounded_training.py](../../tools/run_v1_bounded_training.py) |
| 원본 범위 adapter | [bounded_scope.py](../../hiercp_v1x/bounded_scope.py) |
| prepare/train entry | [scope_training_entry.py](../../hiercp_v1x/scope_training_entry.py) |
| 비교 계획 | [v14_m10_component_search.json](../../config/v14_m10_component_search.json) |
| 공통 P+128U 평가 | [server_v17_historical_full128.sh](../../tools/server_v17_historical_full128.sh) |
| 기존 서버 명령·재개 계약 | [전체 실행 설명](../../docs/v1_scope_learning_20261004.md) |

도움말은 저장소 root에서 `python -B tools/run_v14_scope_training.py --help`로 확인한다. 실제 실행에는 `--gpu --margin-mm --source-experiment --experiment --cuda-gib --rss-gib`가 필요하다. 기존 서버 experiment는 `/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004`이며 결과 stage는 `results/v1.0`다.

같은 실험·설정·source/helper bytes로 재실행하면 마지막 완료 epoch에서 재개한다. 새 이름을 붙이기 위해 source·checkpoint·cache·결과 폴더를 이동하지 않는다. 이 파일은 경로 안내이며 새로운 실행이나 config가 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 계약을 안내했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 기록만 참조하며 새 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 문서 정리에 OOM은 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결에 관한 기존 증거를 구분했다. 새 학습 검사는 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 원본 실행 파일은 유지했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
