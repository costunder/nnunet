# V1.6 코드

| 역할 | 실제 파일 |
| --- | --- |
| 사용자 진입점 | [run_v1_half_b.py](../../tools/run_v1_half_b.py) |
| 상위 계층 bridge | [half_b_model.py](../../hiercp_v1x/half_b_model.py) |
| curriculum support | [half_b_support.py](../../hiercp_v1x/half_b_support.py) |
| 계약·checkpoint controller | [half_b_training.py](../../hiercp_v1x/half_b_training.py) |
| 학습 entry | [half_b_entry.py](../../hiercp_v1x/half_b_entry.py) |
| 공통 P+128U 평가 | [server_v17_historical_full128.sh](../../tools/server_v17_historical_full128.sh) |
| 서버 결과 기록 | [summary.json](../../validation/v16_half_b_server_20261005/summary.json) |

도움말은 저장소 root에서 `python -B tools/run_v1_half_b.py --help`다. 필수 옵션은 `--gpu --baseline --experiment --cuda-gib --rss-gib`다. `--allocation-policy`는 명시한 자원 admission 정책을 선택한다. 기존 experiment는 `/home/aicompetition06/Medical/experiments/v16_m10_halfB_seed42_20261004_r3`, 결과는 `results/half_B`다.

동일 source/helper·baseline·자원·실험 계약에서 자신의 마지막 완료 epoch를 재개한다. A와 baseline 결과는 보존하며 새 준비·A+B 자동 실행은 없다. 기존 SHA-bound source·checkpoint·cache의 이름이나 bytes를 바꾸지 않는다.

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
