# 구현 파일

원본 v1 모델을 수정하지 않고, 비교 위치·공통 loss·전체 후보 평가와 실행 관리를 연결한다. 보존된 원본 source는 byte identity를 검사한 뒤 가져온다.

| 파일 | 역할 |
| --- | --- |
| [run_v18_u_bridge.py](../../tools/run_v18_u_bridge.py) | GPU 선택, 입력 결속, 동일 초기값, 두 군의 공통 batch 선정, 실행·재개 |
| [u_bridge_experiment.py](../../hiercp_v1x/u_bridge_experiment.py) | signed baseline 입력과 원본 sample 목록, 실험 계약·잠금 관리 |
| [u_bridge_data.py](../../hiercp_v1x/u_bridge_data.py) | 원본 P·selected 좌표·고정 U, 10 mm canonical graph와 두 sampled view, 실제 batch |
| [u_bridge_training.py](../../hiercp_v1x/u_bridge_training.py) | pairwise loss, AMP backward/update, exact checkpoint, 전체 129개 후보의 joint 평가 |
| [u_bridge_calibration.py](../../hiercp_v1x/u_bridge_calibration.py) | 원본 source 비용이 큰 문제를 먼저 측정. 실제 학습 순서는 유지 |
| [u_bridge_fields.py](../../hiercp_v1x/u_bridge_fields.py) | 원본 전체 CT 거리 배열을 정확한 dtype·shape·값으로 저장·재사용 |
| [u_bridge_upper.py](../../hiercp_v1x/u_bridge_upper.py) | 원본 source/lesion upper 함수의 정적 반환 배열을 저장·재사용 |
| [server_v18_u_bridge.sh](../../tools/server_v18_u_bridge.sh) | 고정된 서버 경로와 물리 GPU 번호로 두 군을 실행하고 결과 요약 |
| [summarize_v18_u_bridge.py](../../tools/summarize_v18_u_bridge.py) | 저장된 JSON만 읽어 초기·best·최근의 학습/전체 평가 점수를 구분해 출력 |

실제 입력 → 원본 graph → 전체 모델 forward → 공통 loss → backward → optimizer update 연결은 CT/CUDA smoke에서 확인했다. trainable tensor 1,085/1,085개의 gradient와 5개 모듈 그룹의 가중치 변경을 확인했고, 로컬 회귀 검사 74개가 PASS했다.

이 smoke는 train 2명·validation 1명의 DEBUG이며, 두 군 각각 2 epochs를 완료했다. selected의 첫 update 후 중단·재개와 두 군의 완료 후 재실행을 확인했다. 완료 후에는 추가 update 없이 model·optimizer·scheduler·RNG를 보존했다. 각 군에서 초기·epoch 1·epoch 2의 고정된 전체 129개 후보를 joint upper graph로 평가했다. DEBUG 2 epochs의 source별 고유 비교 위치는 selected 7개·native 14개로 순환 계약과 일치한다. [smoke 증거](../../validation/v18_u_bridge/smoke.json)와 [results.json](results.json)에 기록한다. 서버 40 epochs 학습과 추천 품질 검증은 완료하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
