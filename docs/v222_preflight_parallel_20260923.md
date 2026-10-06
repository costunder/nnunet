# v2.22 준비 단계 수정과 검증 — 2026-09-23

## 원인과 적용 범위

기존 비교 표본은 label=1인 위치 중 모든 종양에서 blind radius와 반 voxel 대각선을 더한 거리보다 멀리 떨어진 곳으로 제한돼 있었다. liver_108에서는 요구 거리27.9431mm보다 최대 거리23.5397mm가 작아 후보0개였다. 종양이 없는 간 주석 자체가 없는 것은 아니다. 기존 준비는 이 검사를 각 case의 패치 생성과 함께 수행해86case를 처리한 뒤 실패했다.

이번에는 모든 outer-train105case의 표본 선택을 패치 생성 전에 완료하도록 순서를 바꿨다. 기존 거리 기준·난수·128개·가림·모델은 보존한다. 사전 검사 후 원본 CT/GT hash가 바뀌면 패치 생성을 거부한다. 이 수정은 실패 시 낭비를 줄이지만, 후보0개 문제를 해결하지는 않는다.

거리 제한을 없애고 label=1 중심에서 양성 anchor와 같은 좌표를 제외하는 변경안은 자동 승인 심사에서 거부됐다. 사유는 AGENTS.md의 기존 실험 계약 변경 금지와 정확한 규칙에 대한 명시적 승인 부재였다. 사용자에게 적용 내용을 질문했으며 응답 전에는 변경하지 않는다. 신규 GNN 전체 준비/학습을 실행하지 않았다.

## 병렬 처리

기존 GNN 준비 자원 기록에는 worker4개가 실제로 선택돼 있다. 병렬화가 전혀 없었던 것은 아니지만, 선택은 첫 case의 CPU/RAM 측정에 기반했다. 새 `hiercp_v222/parallel.py`는 메모리와 CPU 범위 내에서 동시 실행 수를 늘려 실제 대기 중인 전체 case 작업으로 처리량을 측정한다. 처리한 case는 재실행하지 않는다. 이후 선택한 폭의 작업 큐는 완료 즉시 보충한다.

최대 전체 입력 메모리 추정과 실제 관측 RSS 증가량을 사용하고 가용 RAM50%를 남긴다. 이 admission은 추정이며 모든 환경에서 OOM이 없다는 보증은 아니다. 명시적으로 요청한 worker 수가 안전 범위를 넘으면 조용히 변경하지 않고 실패한다. `.admissionN.json`, `.waveN.json`, 최종 자원 JSON을 기록한다. 서로 다른 크기의 case로 측정하므로 전역 최적 worker 수라는 주장은 하지 않는다.

## 실행한 검사

- 신규 병렬 DEBUG5개: 실제2개 작업 동시 실행, 모든 작업 정확히1회 처리, auto worker 측정, 실패 전파와 성공 출력 보존, RAM/CPU 초과 admission 거부.
- 신규 사전 검사 DEBUG3개: 기존 거리/난수와 동일한 위치 선택, 후보0개 명시적 오류, 실패 시 패치와 완료 receipt 미생성.
- 기존 DEBUG24개: query/group 격리, 입력 가림 개입 검사, graph chunking 값과 gradient 일치, 모든 학습 표본 방문, L1/L2 출력 영향, loss·gradient·optimizer 연결, 군집 정렬, case provenance.
- 정적 구문 검사 및 frozen v1 확인. Basic CP 실행 source identity도 중단 전 receipt와 동일하다.
- 첫 병렬 테스트 실행은 sandbox의 임시 폴더 쓰기 권한 때문에 실패했다. scoped escalation 후 동일 검사가 통과했다. 코드 성공으로 오기록하지 않는다.

## 실제 CT·GPU DEBUG

`work/v222_20260923_parallel_debug/result.json`에 실제 CT3명·6그래프 검증을 저장했다. 보존된 원본 CT/GT hash와 입력 재생성 일치를 확인했다. 모델은1,519,063parameters이며 CNN/L0/L1/L2 주요 모듈의 유한 gradient와 optimizer 갱신을 확인했다.

| Physical/effective batch | Peak allocated VRAM | 처리량 |
| --- | ---: | ---: |
| 2 / 2 | 611,964,928 bytes | 1.344 graphs/s |
| 4 / 4 | 1,100,222,976 bytes | 1.431 graphs/s |

Gradient accumulation1, GPU RTX5070Ti1개. 기존 DEBUG 그래프24,174nodes/1,229,900edges를 그대로 사용했다. Production의39,936nodes/2,040,356edges와 다르며, 이 값을 최종 학습 batch 선택 근거로 대신하지 않는다. Batch4는 실제 query2개 반복 fixture다. 다중 군집 임상 효능, 전체 GNN40epoch, nnU-Net250epoch, 전체 평가는 검증하지 않았다. GPU 검사 후 scheduler의 admission 로그 필드만 추가했으며 모델·입력·학습·군집 소스 해시는 동일하게 확인한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch2/4와 작업 병렬화를 실제 DEBUG로 확인했다. Production 보정은 남아 있다.
- [x] GPU, CPU, RAM 상태 및 DEBUG 처리량을 측정했다.
- [x] 모델 축소로 성능 문제를 우회하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] synthetic scheduler 작업을 실제 학습 결과로 표현하지 않았다.
- [x] 실제 CT DEBUG에서 핵심 모듈의 forward/loss/gradient/optimizer 연결을 확인했다.
- [x] 적용된 변경과 차단된 변경을 구분했다.
- [x] DEBUG와 전체 학습·평가를 구분했다.
