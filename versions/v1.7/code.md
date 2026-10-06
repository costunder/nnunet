# V1.7 코드

| 역할 | 실제 파일 |
| --- | --- |
| C/D 사용자 진입점 | [run_v17_crossed_training.py](../../tools/run_v17_crossed_training.py) |
| 서버 launcher | [server_v17_crossed.sh](../../tools/server_v17_crossed.sh) |
| 전환 계약 | [v17_crossed_training.json](../../config/v17_crossed_training.json) |
| C 학습 구현 | [transition_c_training.py](../../hiercp_v1x/transition_c_training.py) |
| D 학습·공통 평가 callback | [transition_runtime.py](../../hiercp_v1x/transition_runtime.py) |
| D 진행 조회 | [watch_v17_d_learning.py](../../tools/watch_v17_d_learning.py) |
| 기존 V1/A/B/C BEST 공통 평가 | [evaluate_v17_historical_full128.py](../../tools/evaluate_v17_historical_full128.py) |

도움말은 저장소 root에서 `python -B tools/run_v17_crossed_training.py --help`다. 서버에서는 검토한 checkout의 launcher에 `CP_ARM=C` 또는 `CP_ARM=D`와 고정된 `CP_OUTPUT`을 명시한다. 필수 학습 옵션은 `--arm --baseline --native-run --output --gpu --workers --physical-batch-candidates --cuda-gib --rss-gib --resident-gib`다.

launcher 기본 D는 GPU3/workers16/physical32 observations/CUDA40GiB/RSS192GiB/resident128GiB다. 한 observation에 원래 두 graph view가 있다. C의 calibration 후보1/2/4는 curriculum sample 수이며 sample당8후보다. 실행 shape 측정으로 admission하며 더 작은 모델·graph·batch로 자동 전환하지 않는다.

`CP_OUTPUT`을 생략하면 timestamp가 들어간 새 experiment를 만든다. 정확한 resume에는 같은 output·arm·source·data·자원 계약이 필요하다. D의 `--prepared-cache`/`CP_REUSE_PREPARATION`은 완료된 전체14102관측 cache의 읽기 전용 재사용이며 subset cache는 production에서 거부한다.

C의 알려진 서버 experiment는 `/home/aicompetition06/Medical/experiments/v17_crossed_C_m10_seed42_20261005`다. D는 실제 사용한 고정 output에서 조회해야 한다. [기존 기록](../../docs/v17_crossed_training_20261005.md)에 없는 최신 checkpoint 경로나 완료 수치를 만들지 않는다.

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
