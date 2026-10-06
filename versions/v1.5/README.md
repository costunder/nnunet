# V1.5 — A: L0 교체

성공한 [V1.4 10mm](../v1.4/README.md)의 **L0 encoder/readout/fusion만 CNN으로 교체**한 대조다. 원래 fixed48³ resampling, target erasure, role/shell 구성, source-anchor GT, curriculum8/pool128, 두 view와 complete loss·상위 계층은 유지한다.

[실행](run.py) · [설정](config.md) · [코드](code.md) · [결과](results.json) · [A 계약](../../docs/v15_half_a_learning_20261004.md)

정확한 입력 bridge는 `v2CNN_on_preserved_v1_denseROI`다. native spacing organ-only crop을 쓰는 현재 V2.2 LocalCNN과 동일한 입력 계약이 아니다.

| 기록된 결과 | MRR | top1 / Hit@1 |
| --- | ---: | ---: |
| own 8후보 BEST epoch29 | 0.972222 | top1 0.944444 |
| own 8후보 마지막 epoch40 | 0.958333 | top1 0.916667 |
| 공통 P+128U의 `A` arm | 0.146013 | Hit@1 0.062500 |

own-task40/40 완료와 공통 평가 수치는 사용자 서버 요약을 그대로 옮겼다. 서버 checkpoint·raw metric의 독립 확인은 아니다. A는 원래 과제에서 순위를 학습했지만 기준선과 동등한 품질 PASS나 실패 원인 확정을 붙이지 않는다. 공통 평가는 recipient annotation을 사용하는 외부 donor adapter이며 blind CP 추천 품질이 아니다.

설정은 seed42/40epoch/configured84train·21validation/outer26 제외다. 기존 actual CT/CUDA8-update DEBUG와 전체 서버 결과는 [증거](../../validation/v15_half_a_server_20261004/summary.json)에서 구분한다.

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
