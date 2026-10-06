# v1.8 실제 CT/CUDA 검사

[smoke.json](smoke.json)은 원본 v1 전체 모델의 실제 CT/CUDA 실행 기록을 요약한다. [evidence_sources.json](evidence_sources.json)은 보존된 로컬 원본 로그·JSON receipt의 SHA256과 크기를 기록한다. CT 영상과 모델 checkpoint 자체는 Git에 포함하지 않았다.

원본 10,434,532개 parameter를 유지한 채 train CT 2명·validation CT 1명에서 두 군을 각각 DEBUG 2 epochs 실행했다. selected 군은 첫 update 뒤 저장·중단하고 정확한 cursor에서 재개했다. 두 군 모두 완료 후 같은 명령으로 재실행하여 model·optimizer·scheduler·RNG와 update 수가 바뀌지 않는 것을 확인했다.

74개 회귀 검사가 PASS했다. 실제 CUDA update 4개에서 모든 trainable tensor 1,085개의 gradient와 CNN/L0/L1/L2/scalar 그룹의 가중치 변경을 확인했다. 초기 및 매 epoch validation은 동일한 P 1개와 고정 U 128개를 사용했고, 원본 upper graph는 129개 후보 전체를 한 번에 처리했다. 두 DEBUG epochs에서 native 군이 사용한 U는 source당 14개이며, 128개 전체 학습 완료로 표시하지 않았다.

이 결과는 실행·연결·중단/재개 검증이다. 서버 전체 cohort의 40 epochs 학습, 전체 환자 평가, ranking 품질, 실제 CP 추천 품질을 증명하지 않는다. 서버에서 사용할 physical batch는 양 군의 실제 calibration으로 선정한다. Windows 검사이므로 Linux의 mmap page release 효과와 서버 전체 cohort의 자원 적합성은 아직 측정하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 10 mm 조건을 유지하고 DEBUG만 별도로 표시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
