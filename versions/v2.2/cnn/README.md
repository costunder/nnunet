# 국소3D CNN L0

간 mask 안의 native-spacing CT를 donor 종양 bbox 바깥 margin 범위로 읽습니다. CNN12/24/32 → scale별 masked readout → donor/recipient fusion128D → 기존 L1/L2로 이어집니다. margin은 각 면 바깥 여유거리이며 전체 cube 너비가 아닙니다. 기존 서버 학습에서 후보 순위 분리가 약했습니다.

| 찾을 것 | 파일 |
| --- | --- |
| 실행 | [run.py](run.py) — 기존 CLI 인자 그대로 |
| 설정 원본 | [config.md](config.md) |
| 구현 | [code.md](code.md) |

세부 기록: [기록1](../../../docs/local_cnn_learning_diagnosis_20261001.md), [기록2](../../../docs/v22_native_local_cnn_20261001.md).

P는 실제 관측된 적격 종양 위치이고 U는 미관측 비교 위치입니다. CP suitability 정답으로 재정의하지 않습니다.


## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 설정을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 자원 기록을 연결했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 정리에서는 학습을 실행하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 구현 경로를 그대로 연결했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
