# V1.4 — 10mm 기준선

V1.0에서 **L0 물리 범위만 10mm로 변경**한 분기다. V1.1~V1.3의 계획 변경을 누적한 모델이 아니다. 완전한 transformed donor footprint의 bbox에 양쪽10mm를 더하며 fixed48³ 입력, 원래 L0/L1/L2·두 view·loss·source-anchor GT·선별8후보/pool128은 유지한다.

[실행](run.py) · [설정](config.md) · [코드](code.md) · [결과](results.json) · [범위 계약](../../docs/v1_scope_learning_20261004.md)

| 설정 | 계약 |
| --- | --- |
| 기준 소스 | [V1 보존 ZIP](../v1/README.md)의 74dcc2 snapshot |
| 학습 | seed42, 40epoch, configured84train/21validation, outer26 제외 |
| 원래 평가 | source anchor를 정답으로 하는 sample당8후보 |
| 공통 평가 | 모든21case의 모든 P+128U. zero-P case도 score한다. |

| 기록된 결과 | MRR | top1 / Hit@1 |
| --- | ---: | ---: |
| own 8후보 BEST epoch11 | 1.000000 | top1 1.000000 |
| own 8후보 마지막 epoch40 | 1.000000 | top1 1.000000 |
| 공통 P+128U의 `V1` arm | 0.233839 | Hit@1 0.125000 |

서버 own-task40/40 완료와 공통 평가 숫자는 **사용자 제공 요약**이며, 서버 checkpoint·raw metric을 이 정리에서 독립 확인하지 않았다. 공통 P+128U는 과제·donor 조건·지표 정의까지 바뀐 평가이며 단순히8을128로 바꾼 정확도 대조가 아니다. P는 관측 종양 위치, U는 미관측 위치다.

[16-update actual CT/CUDA DEBUG](../../docs/v14_matched_learning_20261004.md)는 별도 작은 고정 batch의 학습 경로 검사다. 전체 품질이나 서버40epoch 완료를 그 DEBUG로 대신하지 않는다. 기존 `results/v1.0` 내부 경로는 모델 stage 명칭이므로 보존한다.

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
