# v1.6 B 서버 결과: 고정 v1 학습에서 상위 계층 교체가 순위를 학습함

사용자가 2026-10-05에 보낸 서버 요약에서 B(r3)는 40/40 epoch와 전체 학습 완료 기록을 남겼다. BEST epoch20 및 마지막 epoch40 validation의 MRR와 top1이 모두 1.0이다. 따라서 legacy v2.2 L1/L2/scorer 교체만으로 원래 v2.2의 심한 순위 학습 실패가 재현되지는 않았다.

입력은 사용자 터미널 요약이며 서버 checkpoint·원시 score 파일을 로컬에서 독립 평가하지 않았다. 수치는 [서버 요약](../validation/v16_half_b_server_20261005/summary.json)에 보존한다. 구현과 이전 CUDA 증거는 [B 실행 계약](v16_half_b_learning_20261004.md)에 있다. 마지막 stderr 발췌는 진행 표시이며 traceback이 보이지 않는다. 전체 로그를 검토했다는 뜻은 아니다.

| B 결과 | loss | MRR | top1 | positive-best-other margin |
|---|---:|---:|---:|---:|
| 초기 validation | 3.864780 | 0.249074 | 0.027778 | -0.084152 |
| BEST epoch20 validation | 0.130164 | 1.000000 | 1.000000 | 10.340888 |
| 마지막 epoch40 train | 0.006447 | 1.000000 | 1.000000 | 7.862640 |
| 마지막 epoch40 validation | 0.068598 | 1.000000 | 1.000000 | 8.808204 |

초기 validation에서 양성 점수가 가장 높은 비교 후보보다 평균적으로 낮았으나, 마지막에는 양의 margin을 크게 형성했다. 이는 실제 장기 학습 증거다. 여기서 top1은 **원래 여덟 curriculum 후보 중 source anchor를 1위로 놓은 비율**이다. 미관측 위치의 CP 부적합 판정, 특정 외부 donor와의 적합도 정답, 실제128후보 CP 추천 정확도로 확대 해석하지 않는다.

## A/B 이분 대조의 현재 판정

| 조건 | BEST validation MRR | BEST top1 | 마지막 MRR | 마지막 top1 |
|---|---:|---:|---:|---:|
| 기존 v1.4 10mm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| A: CNN L0 교체, 원래 v1 상위 계층 | 0.972222 | 0.944444 | 0.958333 | 0.916667 |
| B: 원래 v1 L0, legacy v2.2 상위 계층·score | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

A는 감소가 남았지만 순위를 학습했고, B는 보고된 validation 순위 지표가1.0에 도달했다. 임의 tolerance로 A를 원래 v1과 동등한 PASS로 처리하지 않는다. **각 절반의 부품을 v1 task 안에 단독으로 넣었을 때 둘 다 심한 실패를 재현하지 않았으므로, 현 단계에서 어느 단일 부품이 원인인지 이분 분할을 계속할 근거가 없다.** 각 부품을 모든 입력·목표·조합에서 무해하다고 증명한 것은 아니다.

A와 B 모두 native v2.2 전체를 그대로 재현한 것이 아니다.

- A는 원래 v1의 adaptive ROI를 fixed48로 resample한 입력, target erasure, 원래 role/shell 구성을 유지하는 bridge를 사용한다. native-spacing organ-only crop과 단일 fused128 출력의 완전한 대조가 아니다.
- B는 원래 v1 source anchor, 선별·관계 corruption curriculum8, 원래 ranking 및 two-view consistency loss를 유지한다. v2.2 P/U 전체 비교, observation CE, alignment loss, tile schedule을 적용하지 않았다.
- B support는 v1의 training-only curriculum label을 사용하고 query 환자 전체를 제외한다. 원래 v2.2 관측 label과 episode/memory gradient·refresh 계약까지 같다는 뜻은 아니다.

다음 원인 분할은 아직 대조하지 않은 **입력·readout 계약**과 **supervision·loss·schedule·support 계약**을 명시적으로 구분해서 진행해야 한다. 우선 실패했던 서버 v2.2 실험의 실제 recipe/source/cache 결속을 확인한 뒤, 검증된 B의 입력·GT·후보를 고정하고 loss/학습정책 그룹을 별도로 대조하는 것을 검토한다. 실패가 재현되면 ranking/observation CE 쪽과 alignment/정규화·schedule 쪽으로 다시 이분한다. 이 대조는 원래 anchor GT에 적용하는 학습정책 대조이며 native P/U 과제 전체 재현이 아니다. 기존 P=관측 적격 종양 위치, U=미관측 비교 위치의 의미를 CP 적합/부적합이나 donor-specific GT로 바꾸지 않는다. 이 결과만으로 P/U objective가 불가능하다거나 CNN/legacy L1이 원인에서 완전히 제외됐다고 주장하지 않는다. 부품 사이 상호작용도 미검증으로 남는다. A+B를 자동 실행하거나 기존 baseline/A/B를 다시 학습하지 않았다.

## 시간과 검증 범위

A/B scorer 및 objective scale이 다르므로 margin/loss 절대값만으로 구조 우열을 판정하지 않는다. B의 마지막 epoch 시간은15.83분이다. A의 마지막 epoch4.05분과 GPU/allocation이 같다는 독립 증거와 모든 epoch 시간 분포를 받지 않았으므로 정확한 배속 대조로 사용하지 않는다. B의 전체 support refresh는 epoch wall에 포함된다. 이 출력만으로 상위 계층·support·L0 각각의 병목 기여를 분리할 수 없다.

이번에는 새 GPU 학습·단위 검사·전체 평가를 시작하지 않았다. 기존 실제 CT/CUDA 검증은 보존하고, 새로 받은 것은 서버 전체 학습 완료의 사용자 보고다. 전체21 case별 paired evaluation,128후보 production CP, Basic CP/nnU-Net 실험은 이 결과로 완료 처리하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 계약을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 admission 증거를 보존하며 이번 서버 요약에 없는 자원 수치를 만들지 않았다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 결과 보고에는 OOM이 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 실제 CT/CUDA 증거를 보존했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
