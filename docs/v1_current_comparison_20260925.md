# v1과 현재 v2.22 비교 — 코드 대조와 실제 CT L0 측정

사용자 요청은 v1과 현재 실행이 왜 다른지 비교하는 것이었다. 앞선 64→256MiB 실험은 현재 모델 내부의 실행 정책 비교였고, 이 요청의 대체물이 아니다. 이번에는 보존 v1과 현재 코드를 직접 대조하고 동일한 실제 CT 위치에서 두 L0를 실행했다. 과거 서버의 실제 체크포인트·캐시·epoch 소요시간을 복원했다고 주장하지 않는다.

## 비교 기준과 원문

- 보존 v1: `versions/v1/manifest.json`의 revision `74dcc2cf03d2d40d1f582223321d96004333f661`. 내부 모델 revision은 v5. 실행 전에 보존 소스/아카이브 SHA 검증 통과.
- 현재: `config/prompt_graph_v222_v1_l0.json`, `hiercp_v222/v1_local.py`, `v1_execution.py`. 기존 production 캐시 provenance 검증 통과.
- 과거 실제 완료 기록: `experiment_results/recovered_conversations_20260918/terminal_records/f4e3b499-a931-484a-832c-d2aac078fd54.txt`는 full cache 235개, GNN 40/40 epoch 완료. `ee0258b7-a9e4-40b2-a0e7-506017907681.txt`는 paired cache 187개 재사용과 40/40 완료. 이 두 기록은 서로 다른 실행이다.
- 복구한 `attachments/469001ae70c826c4_code_summary.txt`의 LocalTumorContextPyGEncoder 주변 HeteroGATv2Block은 stock GATv2Conv를 사용한다. 보존 v1은 compatibility gate 및 streamed edge 실행을 포함한다. 따라서 보존 v1의 측정을 과거 학습 체크포인트의 정확한 실행시간으로 대신할 수 없다.
- 로컬 work에서 prototype/원본 model.pt/옛 graph-cache manifest는 찾지 못했다. 현재 v2 계열 체크포인트가 있다는 사실을 v1 가중치 보유로 해석하지 않는다. 서버는 이번 조사에서 접근·변경하지 않았다.

## 실제 차이

| 항목 | 보존 v1 코드/설정 | 현재 v2.22 코드/캐시 |
|---|---|---|
| 전체 파라미터 | 10,434,532 | 5,550,806 |
| L0 파라미터 | 5,600,740 | 4,718,420 |
| L0 입력 | 5채널, 수작업 특징, 내부 노드, target 가림 포함 | CT 1채널, 내부 노드/수작업 입력/target 가림 제외 |
| 케이스당 준비 | 2묶음 × 8후보 = 16쌍 설정 | 비교 위치 128개 + 적격 실제 종양 위치 전부 |
| 캐시 한 entry | donor 하나와 여러 후보, patient/prototype graph | donor–후보 한 쌍 |
| graph view | 후보당 2개, CNN feature map은 두 view가 공유 | 쌍당 1개 |
| 상위 학습 | patient/prototype graph와 후보 ranking·view consistency | data–label 관계와 관측 class CE·정렬 CE |
| 전체 support 재인코딩 | 해당 학습 루프에 없음 | 초기, 매 epoch 뒤, 최종 선택 모델에서 수행 |

이는 보존 소스와 현재 소스의 비교다. 보존 설정의 8후보를 과거 서버 실행 때 실제 적용된 값으로 확정하지 않는다. 235/187 entry를 14,102 pair와 직접 나눠 속도 차이라고 주장하지 않는다.

현재 실제 캐시는 inner-train 84case/11,279쌍(84×128+527), inner-val 21case/2,823쌍(21×128+135)이다. 정상 epoch의 L0 forward는 학습 11,279 + support refresh 11,279 + 검증 2,823 = **25,381쌍**, backward 대상은 학습 11,279쌍이다. 초기·최종 memory는 합계 22,558쌍의 추가 forward다. 이는 쌍 처리 횟수이며 kernel 수나 시간 배수가 아니다. checkpoint 재계산 및 runtime calibration 비용은 별도다.

graph cache는 정적 CT/노드/엣지 재사용이다. 학습으로 변하는 CNN/GNN embedding까지 고정하는 캐시가 아니므로 두 버전 모두 캐시가 있어도 GPU forward/backward는 남는다. 현재는 support embedding 갱신 작업까지 추가된다. 다만 이 구조 차이만으로 실제 서버 지연의 비중을 단정하지 않는다.

추가 발견: `v1_execution.py`의 epoch 결과 `seconds`는 `state['epoch_seconds']`이며 optimization 구간에서 누적된다. support refresh·validation 및 일부 checkpoint 비용을 포함한 전체 epoch wall time이 아니다. 이 필드를 전체 epoch 시간으로 인용하면 안 된다. 이번 작업은 관측 도구/기록만 추가했고 기존 실행 소스나 cache provenance를 변경하지 않았다.

## 실제 CT에서 측정한 L0 비용

RTX5070Ti, PyTorch allocator 9,000,000,000byte 상한, bf16, 64MiB edge workspace, 기존 3층/128D/4head 유지. 실제 liver_5와 liver_6, 동일 donor component와 동일 비교 위치 16개. 4개 CPU worker로 입력 구성, GPU에서는 disjoint graph batch. 각 버전의 native 특징·전체 sampled graph·모든 induced edge를 유지했다. 두 버전의 sampling seed 규칙도 각 구현 그대로여서 topology가 완전히 동일한 실험은 아니다.

| physical pair batch | 구현 | 노드 / 엣지 | L0 forward | probe backward | peak allocated |
|---|---|---|---:|---:|---:|
| 8 | 보존 v1 | 63,873 / 4,442,445 | 1.267초 | 3.073초 | 1.479GB |
| 8 | 현재 | 62,954 / 4,360,377 | 1.037초 | 2.508초 | 1.247GB |
| 16 | 보존 v1 | 135,114 / 9,641,027 | 2.592초 | 6.177초 | 3.010GB |
| 16 | 현재 | 134,448 / 9,539,684 | 1.914초 | 5.234초 | 2.546GB |

각 프로세스 3회 중 warmup을 제외한 2회 평균. L0의 fused output 제곱평균으로 역전파·optimizer update를 수행하는 **명시적 비용 probe**이며 native ranking/관측 CE loss가 아니다. 미학습 초기 가중치, 같은 케이스 내 쌍, single view 조건이다. v1 612/612, 현재 479/479 parameter tensor에 유한 gradient 확인. 이 결과는 전체 모델 학습 정확도나 CP 효용 검증이 아니다. 8쌍은 첫 case, 16쌍은 두 case이므로 batch 배수에 대한 순수 scaling 비교도 아니다.

이 입력에서는 현재 L0가 보존 v1보다 크거나 느리다는 증거가 없다. 확대된 관측 수와 추가 support pass를 포함하는 전체 실행을 비교해야 한다. 현재 모델 내부의 64→256MiB 개선 결과는 별도 문서이며 이 표에 섞지 않았다.

재현 도구: `tools/compare_preserved_v1_l0_debug.py`. 새 출력 경로로 `prepare`, 이어서 `measure --variant old/new --batch 8/16`, 마지막 `report`를 실행한다. 기존 출력 덮어쓰기를 거부한다. 원본 SHA, 위치, 설정, 실행 결과는 `work/v1_current_comparison_20260925_DEBUG/` 및 `validation/v222_r6/v1_current_comparison_20260925_DEBUG.json`에 보존한다. 측정용 748,794,807byte 임시 graph fixture는 측정 후 제거하고 실제 CT 원본·기존 학습 캐시는 유지한다.

## 완료 범위

입력 제작과 비교 도구 구현, 실제 CT L0 실행 4조건, batch8/16 및 9GB 상한 확인, source/입력 일치/gradient 검증, 전체 코드와 workload 대조 완료. **과거 서버 native 전체 epoch 재현과 현재 서버 전체 epoch 실측은 미완료**다. A100 MIG 처리량·정확도·학습 완료를 이 결과로 주장하지 않는다. 현재 서버에 성능 패치를 적용하거나 재시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 별도 명시적 DEBUG만 실행했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. batch8/16, CPU worker4.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 측정은 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 실제 CT와 명시적 미학습 비용 probe.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번 범위는 L0 probe이며 전체 native loss 검증은 아니다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
