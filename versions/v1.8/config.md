# 고정 조건

기준은 보존된 v1.4 10 mm의 signed source/cache/config/prototype다. [실험 정의](../../config/v18_u_bridge.json)와 실행 시 생성하는 `experiment.json`에 원본 source·raw CT·bank·U 128개·초기 모델·코드 SHA를 결속한다. CLI 자원 설정과 physical batch는 실제 calibration으로 결정하며 두 군에 같은 값을 적용한다.

| 항목 | 고정 조건 |
| --- | --- |
| 모델 | 원본 L0/L1/L2/scalar scorer, 10,434,532 parameters |
| 깊이·너비 | L0/L1/L2 3/2/2층, 128D, 4 heads |
| 국소 범위·입력 | margin 10 mm, context outer 10 mm, 원본 48³ dense input |
| 데이터 | signed train 151개·validation 36개 source sample. 실제 validation 18명 |
| 양성 P | 원본 source 종양의 anchor 1개 |
| selected 비교 위치 | 원본에 선택된 좌표 7개. 기하 변형 제거 |
| native 비교 위치 | 같은 환자의 고정 U 128개 중 7개를 에포크별로 순환 |
| 학습 | fresh seed 42, 동일 초기값·새 optimizer, 각 군 40 epochs |
| Loss | source 문제별 평균 pairwise softplus + 원본 view consistency × 0.1 |
| 평가·best | 매 에포크 P 1개 + U 128개. 전체 joint upper graph, patient macro MRR/top1/pair-loss |
| 배치 | 완전한 source 문제 `1 / 2 / 4 / 8 / 16 / 32`개를 측정하고 양 군 공통 값을 선택 |
| 축적·샘플 순서 | gradient accumulation 1, 원본 `RandomSampler`, 전용 seed `42 + 2003` |

P/U의 의미는 CP 적합/부적합 GT로 바꾸지 않는다. selected도 공통 pairwise loss를 사용하므로 과거 v1 결과의 exact reproduction이 아니다. 이 대조는 비교 위치 분포와 위치별 노출 빈도를 함께 바꾼다.

로컬 회귀 검사 74개와 실제 CT/CUDA smoke가 PASS했다. DEBUG는 train 2명·validation 1명, 두 군 각각 2 epochs를 완료했다. 동일 초기값, 총 4번의 update에서 gradient 1,085/1,085개·5개 그룹 가중치 변경, 두 군 각각 3번의 전체 129개 후보 평가를 확인했다. 중단·재개와 완료 후 재실행의 추가 update 0회도 검증했다. 서버 40 epochs 품질은 아직 미검증이다.

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
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 CT/CUDA update에서 gradient와 가중치 변경을 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
