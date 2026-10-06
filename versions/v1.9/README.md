# v1.9 — 비교 위치와 순환·loss 대조

실행 중인 v1.8 selected/native 두 실험을 기준으로, 고정 U와 listwise loss 두 실험만 추가한다. 기존 두 실험은 계속 진행한다. 새 두 실험은 원본 v1 m10 모델·P·source 목록을 유지하고 seed 42 초기 가중치에서 각각 시작한다.

| 군 | 학습 비교 위치 7개 | 목적함수 |
| --- | --- | --- |
| selected | 기존에 선택된 원본 좌표 7개, 기하 변형 제거 | 평균 pairwise softplus |
| native | 같은 환자의 고정 U 128개에서 에포크마다 7개 순환 | 평균 pairwise softplus |
| native_fixed | 동일 U 128개 중 처음 7개를 계속 사용 | 평균 pairwise softplus |
| native_listwise | native와 source·에포크별로 정확히 같은 순환 7개 | P를 index 0으로 둔 8후보 cross entropy |

모든 군에 원본 두 sampled view의 consistency를 0.1로 더한다. 7개는 source 문제 한 개의 비교 위치 수이며 physical batch는 완전한 source 문제 여러 개다. 네 군의 실제 CUDA calibration을 통과하는 공통 batch를 사용한다. source별 업데이트 수와 40 epochs를 유지하며, U 128개를 한 에포크 안에서 모두 순회하는 추가 업데이트를 만들지 않는다.

네 군 모두 동일 원본 P 1개와 고정 U 128개를 평가한다. L0만 chunk로 encoding한 다음 후보 129개가 있는 patient/prototype graph를 함께 처리한다. Best는 전체 129개 후보의 patient-macro MRR·top1·pairwise 평가 loss 순으로 선택한다. 학습의 7개 비교 지표와 전체 129개 평가 지표를 구분한다.

selected/native는 이미 실행 중인 v1.8 기준군이다. `native_fixed`와 기존 `native`는 고정 위치 반복 노출과 순환 노출을 비교한다. `native_listwise`와 기존 `native`는 동일한 위치 노출에서 목적함수 차이를 비교한다. 두 추가 실험끼리는 위치 노출과 loss가 함께 달라지므로 한 변인의 효과를 판정하는 직접 대조가 아니다. 기존 v1의 original8 MRR 1.0은 보존 참고값이며 새 selected의 성능으로 간주하지 않는다.

`selected`와 `native_fixed`는 모두 동일한 7개를 반복하므로 비교 위치의 분포를 대조한다. 기존 selected/native 비교에서는 위치 분포와 누적 노출 개수가 함께 달라졌고, 새 fixed 군이 그 둘을 나눠 확인하게 한다. 고정 U 7개는 전체 frozen 목록의 처음 7개를 그대로 사용하며 난이도로 재선별하지 않는다.

Listwise CE는 원래 식의 평균을 그대로 사용한다. 평균 pairwise softplus와 loss·gradient의 자연스러운 크기가 다르므로, 결과는 두 목적함수의 실제 학습 경로 비교로 해석한다. 경쟁 방식만의 효과라고 단정하지 않는다. 모든 군의 optimizer·학습률·consistency 가중치는 동일하다.

원본 source의 case/sample/component/anchor와 P 메타데이터, target erase, 전체 L0/L1/L2/scalar head, prototype bank를 유지한다. P를 다른 종양이나 다른 환자의 donor로 전환하지 않는다. U는 미관측 비교 위치다. Listwise 학습도 U를 CP 부적합 정답으로 재지정하지 않으며, hidden filter나 재추출을 적용하지 않는다.

Signed 목록은 train 151개·validation 36개 source 문제다. 설정 validation은 21명이지만 실제 signed materialized validation은 18명이다. 누락된 3명은 실행 계약과 로그에 명시한다. 새 sample을 만들어 21명이라고 표시하지 않는다.

서버 launcher는 호출당 추가 실험 한 개만 시작한다. `CP_ARM=native_fixed` 또는 `CP_ARM=native_listwise`를 지정한다. `all`과 기존 selected/native는 이 launcher에서 받지 않는다. 별도 GPU와 터미널에서 각각 실행할 수 있다. `CP_GPU`는 물리 GPU 번호다.

기본 저장 위치는 각각 `/home/aicompetition06/Medical/experiments/v19_native_fixed_m10_seed42`와 `/home/aicompetition06/Medical/experiments/v19_native_listwise_m10_seed42`다. 모델·optimizer·RNG·cursor·캐시·잠금·체크포인트가 서로 분리된다. 같은 명령은 해당 실험의 상태에서 재개한다. 활성 v1.8 캐시를 공유하거나 기존 체크포인트를 새 실험에 옮기지 않는다.

`CP_REFERENCE` 기본값은 `/home/aicompetition06/Medical/experiments/v18_u_bridge_m10_seed42`다. 이곳의 experiment/calibration JSON만 읽어 이미 측정한 physical batch와 worker 수를 맞춘다. 초기 calibration의 짧은 clone probe는 유지하며, 정식 40 epoch 학습은 선택한 추가 군 하나만 수행한다. 네 군 새 장기 학습을 모두 요구했던 이전 실행 안내는 이 방식으로 수정했다. 동시 실행의 epoch 시간에는 CPU·I/O 경합이 섞이므로 단독 실행 처리량으로 해석하지 않는다.

실행 조건은 [config.md](config.md), 구현 연결은 [code.md](code.md), 검증 상태는 [results.json](results.json)에 기록한다. 기존 CUDA evidence는 commit `c430a88`의 구현에 해당하며, 이번 수정은 서버 launcher·기준 실행 설정 읽기·요약 출력에 한정한다. 학습 엔진·모델·후보 규칙의 결속 파일은 유지한다. Bash 구문 검사는 로컬 실행기가 없어 미검증이다. 실제 CT/CUDA에서 네 군 각각 두 DEBUG epoch·두 update와 전체 129개 평가를 확인한 기존 결과를 보존했다. Listwise의 update1 중단·정확 재개와 네 군의 완료 후 0 추가 update도 통과했다. [검증 기록](../../validation/v19_comparison/README.md)은 당시 GPU·RAM·graph 규모와 파일 SHA를 포함한다. 서버의 40 epochs 및 추천 성능은 아직 미검증이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 10 mm·7개 비교 조건을 유지한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. DEBUG 네 군에서 source batch2·32 graph views를 측정했다. 서버는 후보 batch를 다시 측정한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제 CUDA update에 peak VRAM·RSS·CPU와 구간 시간을 기록했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 DEBUG에서는 OOM 없이 원본 모델을 유지했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 update마다 1,085개 tensor gradient와 주요 모듈 가중치 변화를 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
