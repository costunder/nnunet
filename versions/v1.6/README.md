# V1.6 — B: 상위 계층 교체

[V1.4](../v1.4/README.md)의 **원래 L0를 유지하고 L1/L2/scorer를 legacy V2.2 연산으로 교체**한 대조다. 원래 source-anchor 정답·curriculum8/pool128·ranking loss·두 view consistency는 유지한다.

[실행](run.py) · [설정](config.md) · [코드](code.md) · [결과](results.json) · [B 계약](../../docs/v16_half_b_learning_20261004.md)

B support는 원래 curriculum label을 사용한다. 현재 native P/U의 전체 비교·observation CE·alignment·support16 episode로 전환한 모델은 아니다.

| 기록된 결과 | MRR | top1 / Hit@1 |
| --- | ---: | ---: |
| own 8후보 BEST epoch20 | 1.000000 | top1 1.000000 |
| own 8후보 마지막 epoch40 | 1.000000 | top1 1.000000 |
| 공통 P+128U의 `B` arm | 0.226487 | Hit@1 0.125000 |

own-task40/40 완료와 공통 평가 숫자는 사용자 제공 서버 요약이다. 이번 정리에서 checkpoint·raw score를 독립 평가하지 않았다. B만 교체하면 원래8후보 과제를 학습한다는 기록이며, 모든 입력·과제에서 상위 계층이 무해하다는 증거는 아니다.

seed42/40epoch/configured84train·21validation/outer26 제외와 실제 signed candidate population을 유지한다. [서버 결과 설명](../../docs/v16_half_b_server_result_20261005.md)은 8후보 top1과 공통 P+128U의 case Hit@1을 구분한다.

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
