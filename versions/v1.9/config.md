# 고정 조건

[실험 정의](../../config/v19_comparison_controls.json)는 기존 두 군과 추가 두 군의 비교 위치와 목적함수를 명시한다. 원본 signed source/config/prototype와 raw CT·고정 U·초기 모델·실행 코드 SHA를 실행 계약에 결속한다. 서버 추가 실험은 기존 v1.8의 측정 batch·worker를 읽어 맞추고, 해당 batch의 실제 calibration을 확인한다.

| 항목 | 조건 |
| --- | --- |
| 모델 | 원본 L0/L1/L2/scalar scorer, 10,434,532 parameters |
| 깊이·너비 | L0/L1/L2 3/2/2층, 128D, 4 heads |
| 국소 범위·입력 | margin/context outer 10 mm, 원본 48³ dense input |
| source/P | 원본 case/sample/component/anchor 한 개. 다른 환자 donor로 바꾸지 않음 |
| 데이터 | signed train 151개·validation 36개 source sample, 실제 validation 18명 |
| selected | 원본 선택 좌표 7개, geometry corruption 없음 |
| native | 고정 U 128개 중 7개, offset `((epoch−1)×7) mod 128` |
| native_fixed | 같은 고정 U 목록의 첫 7개, 모든 에포크에서 동일 |
| native_listwise | native와 동일 순환 위치·노출, P index 0의 8후보 CE |
| pairwise loss | source별 `mean softplus(sU−sP)` |
| 공통 consistency | 원본 두 view consistency × 0.1 |
| 학습 | fresh seed 42, 동일 초기값, 각 군 40 epochs |
| sample 순서 | 원본 RandomSampler, 전용 seed `42 + 2003`, 모든 원본 문제 매 에포크 1회 |
| 평가·best | 고정 P 1개 + U 128개, joint upper graph, patient-macro MRR/top1/pair-loss |
| 배치 | 기존 v1.8 calibration의 measured physical source batch를 추가 두 군에도 사용 |
| 축적 | gradient accumulation 1, physical/effective sample batch 별도 기록 |
| 서버 | 물리 GPU 번호 지정, `native_fixed`/`native_listwise` 각각 독립 실험 루트 |

P의 coverage 1·source patch ring 통계·자기 source를 제외한 다른 종양 거리와, U의 label 1 coverage·모든 관측 종양 거리를 유지한다. 입력과 후보를 줄이지 않고 정확한 canonical-local·whole-case distance·upper static cache를 재사용한다. Listwise는 순위 목적함수의 변경이며 U의 GT를 만드는 정책이 아니다.

실제 validation 18명과 설정 21명의 차이를 로그에 공개한다. Historical original8 MRR 1.0과 새 네 군의 full129 지표는 다른 평가 조건이다. 실제 CT/CUDA DEBUG 네 군 검사와 정확 재개는 통과했고, 서버 40 epochs와 전체 signed cohort의 자원 측정·품질은 대기 중이다.

추가 군을 각각 실행해도 source 목록·40 epochs·평가 후보 129개는 유지한다. 같은 GPU나 쓰기 가능한 캐시를 두 실행에 공유하지 않는다. 기본 root가 군별로 분리되므로 기존 v1.8 두 군과 새 두 군의 상태가 섞이지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 네 군의 실제 DEBUG source batch2를 측정했다. 서버는 전체 후보 batch를 다시 측정한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제 update 자원·구간 시간을 기록했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. DEBUG에서는 OOM 없이 원본 규모를 유지했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 네 군 실제 update의 전체 trainable gradient와 주요 모듈 변화를 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
