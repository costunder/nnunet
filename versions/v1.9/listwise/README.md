# v1.9 listwise — native_listwise

native와 source·epoch별로 같은 순환 U 7개를 사용하고, P가 index 0인 8후보 cross entropy로 학습합니다.

원본 **v1 m10 모델**을 유지합니다: 10 mm 범위, L0/L1/L2 3/2/2층, 128D, 4 heads, 48³ 입력, 10,434,532 parameters. 한 source 문제의 학습은 원래 P anchor 1개와 비교 위치 7개이며, 목적함수는 8후보 cross entropy(target=0) + 원본 두 view consistency × 0.1입니다. 평가는 동일 P 1개와 frozen U 128개 전체를 함께 처리합니다. U는 미관측 비교 위치이며 CP 부적합 정답으로 새로 정의하지 않습니다.

각 arm은 기존 데이터·checkpoint·optimizer·RNG·cursor와 원래 40 epoch 계약을 유지해 독립 재개합니다. 이번 안내 작업에서는 서버 실행·학습·성능 검증을 하지 않았습니다.

`CP_GPU`에 배정받은 물리 GPU 번호를 설정한 뒤 저장소에서 이 래퍼를 실행합니다.

```bash
bash versions/v1.9/listwise/server.sh
```

래퍼는 `CP_ARM=native_listwise`을 고정하고 [공유 launcher](../../../tools/server_comparison_cached.sh)에 연결합니다. 다른 `CP_ARM` 값이나 위치 인수는 거부합니다. `CP_GPU`, 선택적 `CP_EXPERIMENTS_DIR`·`CP_INVENTORY` 등 기존 환경변수는 그대로 전달합니다. 래퍼 자체는 현재 작업 디렉터리에 의존하지 않고 저장소 위치를 찾습니다.

공유 launcher와 기존 실행 기록에서 확인한 기본 서버 경로는 다음과 같습니다. `CP_EXPERIMENTS_DIR`를 지정하면 그 아래 같은 이름을 사용합니다.

- 실험: `/home/aicompetition06/Medical/experiments/v19_native_listwise_m10_seed42`
- 저장 데이터: `/home/aicompetition06/Medical/experiments/v19_native_listwise_m10_seed42/data`
- 최근 checkpoint: `/home/aicompetition06/Medical/experiments/v19_native_listwise_m10_seed42/native_listwise/checkpoint_latest.pt`
- best checkpoint: 위 checkpoint 폴더의 `checkpoint_best.pt`
- 원본 frozen inventory: `/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42/inventory/index.json`

## 작업 완료 체크리스트

이번 작업은 실행 안내·래퍼 추가이며 학습을 실행하지 않았습니다. 계산·학습 검증 항목은 해당 없음으로 표시합니다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. (안내 작업에 해당 없음)
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. (안내 작업에 해당 없음)
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (안내 작업에 해당 없음)
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (이번 작업에서 학습 검증 안 함)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
