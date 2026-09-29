# v2.2 L0 공식 EZ-SP adapter — 구현 및 DEBUG 검사

2026-09-29. **adapter·단위 검사·실제 pair 비용 측정 경로 구현. Production 학습 승인 아님.**

## 입력 문서와 우선순위

사용자가 제공한 `L0_EZSP_Reconciled_Handoff.zip`을 `work/l0_ezsp_reconciled_handoff_20260929`에 해제하고 `00_READ_FIRST.md`, `01_EZSP_CORRECTION.md`, 내부 원본 ZIP의 실제 명세·소스 anchor·CPU reference를 읽었다. 외부 manifest와 내부 ZIP CRC를 확인했다. 내부 ZIP SHA256: `612418212f0614d829fa60185bc9779ea22fe6b258eddc944c383158fb0405dc`.

정정 문서가 원본의 mutual-best/pair-only/연쇄 병합 금지보다 우선한다. 원본 SOURCE_IDENTITY의 8개 anchor는 현재 checkout과 일치한다. 원본 CPU reference는 수학 참고 검사이며 공식 GPU 병합 검증과 별도로 기록한다.

## 구현

- `l0_ezsp/vendor/torch_graph_components`: [공식 commit e3db9f3](https://github.com/drprojects/torch-graph-components/blob/e3db9f352fae52dff416616742b3c7ff1378451d/src/torch_graph_components/merge.py)의 변경 없는 MIT 소스. 실행 전 SHA256/Git blob 검증. merge blob `f37f803137e2cb28d17b75800e239a5b6406f778`.
- `config/l0_ezsp_unresolved.json`: 첨부 프로파일 그대로. reg 두 값은 null이며 실행 API/CLI에서 둘 다 필수. 본 파일의 blockers는 전달 당시 원문이며 현재 검사 결과는 이 문서를 따른다. 실제 진단의 0.02를 기본값으로 저장하지 않았다.
- `data.py`: 기존 seed·fine view·48³ CT 유지. mm 좌표와 canonical ID는 별도 sidecar. PyG disjoint-union batching과 CPU thread 병렬 materialization. 학습 특징에 기하·통계 채널 추가 없음.
- `partition.py`: role별 호출, pair/shell별로 연결이 분리된 GPU 입력. FP32 L2 정규화. 최초 unique undirected adjacency의 W=1, 다음 단계는 fine adjacency 경계 가중치의 합. 공식 함수를 그대로 호출하며 연쇄 병합 허용. 호출의 RNG는 dropout/증강 RNG와 분리한다. stable ID 입력 정렬은 커널 교체가 아니다. 모든 GPU/버전에서 bitwise 결정론을 보장한다는 뜻은 아니다.
- `ops.py`: live feature의 원본-node mass 가중 집계, 13개 방향성 relation의 quotient·중복 합산·witness. 같은 role 내부 edge만 제거하고 cross-role의 같은 정수 ID edge는 보존한다. kNN 재연결 없음.
- `encoder.py`: **CNN32 → 공식 병합 → projection128/GAT2 → 공식 병합 → GAT1 → mass-aware 두 scale readout 평균 → 기존 fuser →128D**. 3 GAT blocks/4heads/128D 유지. partition은 activation checkpoint 바깥에 있다. 총 trainable L0 parameters는 기존과 동일한 **4,718,420**.
- admission은 **완성된 partition 뒤, 각 scale GAT 앞**에서 bbox·normalized SSE/mass·role/shell 및 pair별 node/edge 상한·연결성·group 분리를 검사한다. bbox는 모든 fine member의 min/max로 이어진다. 위반 시 명시적 오류. node drop/record skip/fine GAT fallback/CP no-op 없음.
- `build_model`은 기존 `PromptGraphModel`의 명시적 encoder 주입 지점을 사용한다. 합성 DEBUG에서 기존 L1/L2·supervised loss와 연결을 검사했다. 본학습 runner/online CP에 기본 등록하거나 새 학습을 시작하지 않았다. production 경로 전체 연결을 검증했다는 주장은 하지 않는다.
- `identity.py`와 module extra state가 old checkpoint의 exact resume를 거부한다. parent `load_state_dict(strict=False)`로도 identity 검사를 우회하지 못한다. 별도 명시적 CNN-only 초기화 함수는 CNN 이외 가중치·optimizer·memory·teacher plan을 읽지 않는다. 새 support memory/plan은 재생성이 필요하다.

기존 Basic CP, L1/L2, loss, 전체 observation, 후보128, physical batch 설정, 전체 paste mask 검사 및 원본 캐시는 수정하지 않았다. 새 파일은 core provenance 디렉터리 밖에 있다. 기존 캐시를 새 구조의 검증 완료 캐시라고 재명명하지 않는다.

## 검사 결과

`python -m unittest tests.test_l0_ezsp_debug -v`: **9개 통과**. CPU 집계/gradient·typed edge·mass-aware readout·reg 필수·구형 resume 거부; CUDA 공식 연쇄 병합·repeat/순열·단독/배치·shell 분리·고립/빈 context·상한 거부; 전체 크기의 CNN/3 GAT/L1/L2/loss/backward/Adam update를 합성 DEBUG로 검사했다. Backward 중 partition 추가 호출 0. Admission 실패 시 첫 GAT 호출 0.

원본 `reference_checks.py`는 별도 DEBUG 복사본에서 실행해 통과했다. 이 결과를 공식 GPU 검증에 포함시키지 않는다. 모든 합성 데이터는 합성 DEBUG로 표시하며 실제 CT 성능 결과로 취급하지 않는다.

보존된 원시 결과: `validation/l0_ezsp_20260929/summary.json`, `unit_tests.txt`, `cpu_reference.json`, `actual_pair1.json`, `actual_pair8.json`. 최종 정적 검사10파일, 단위 검사9개(실패0/skip0), 원본 anchor8개 해시 일치. 실제 비용 측정 이후 추가한 checkpoint load identity 차단은 최종 단위 검사에 포함되며 이전 측정 report의 당시 source hash를 덮어쓰지 않는다.

## 실제 CT 짧은 비용 측정

RTX5070Ti16GB, torch2.8.0+cu128, FP32, CPU workers8. 프로젝트 venv에 torch-scatter2.1.2+pt28cu128 설치; 시스템 CUDA/드라이버 변경 없음. 기존 실제 DEBUG cache의 inner_train8 records 사용. 초기화 seed42, 학습되지 않은 CNN, reg1=reg2=0.02를 **명시적 진단 후보**로만 사용했다. 동일 physical batch의 baseline과 adapter를 각각 실행했다. 손실은 출력 제곱 평균의 L0 비용 진단이며 observation rank 학습이 아니다.

| 측정 | 실제 pair1 | 실제 pair8 |
|---|---:|---:|
| fine nodes | 7,955 | 52,666 |
| fine directed edges | 317,595 | 2,761,602 |
| 기존 L0 첫 update | 1.074초 | 5.003초 |
| 기존 L0 두 번째 update | 미실행 | 2.633초 |
| EZ-SP 거부까지 | 0.788초 | 1.728초 |
| EZ-SP full update | **admission 실패로 미실행** | **admission 실패로 미실행** |

8pair 실행에서 CNN0.016초, sampling/입력 adjacency0.061초, scale1 partition 및 역할 검사1.612초, typed quotient/전체 admission0.038초. baseline peak allocated1.039GiB, 새 경로는 거부 시점0.297GiB. 중간 거부까지의 메모리를 full-update 메모리 감소로 해석하면 안 된다. CPU graph/materialization1.193초, H2D0.006초. cold start와 1회 추가 update만 측정했으므로 안정적인 평균 처리량·epoch 추정으로 사용하지 않는다.

첫 pair의 scale1 결과는 **1,611 nodes /39,309 edges**, 상한 **1,024/32,768**을 넘는다. source context bbox 최대71.036mm(허용10mm), variance0.1946(허용0.05). 8pair별 node 수는 `[1611,951,5155,1603,1271,1896,3306,2064]`, edge 수는 `[39309,24694,140592,59909,52838,58048,97478,61063]`. 작은 graph만 택해서 통과시키지 않았다. pair2가 총량 이하인 것만으로 bbox/variance/role 조건까지 통과한 것은 아니다.

이는 이 초기 feature와 reg 후보가 첨부 admission 조건을 만족하지 못했다는 결과다. 모든 reg/학습된 feature에서 불가능하다는 결론은 아니다. 제한 변경·사후 분할·자동 reg 재시도 없이 거부 결과를 보존했다.

## 실행 경로

아래는 **짧은 로컬 DEBUG** 명령이며 서버 본학습 명령이 아니다. 동일 output을 재사용하면 덮어쓰지 않고 거부한다. reg 값은 이 보고서에 기록한 진단 후보이며 production 추천값이 아니다.

```powershell
.venv\Scripts\python.exe -m unittest tests.test_l0_ezsp_debug -v
.venv\Scripts\python.exe tools/profile_l0_ezsp_debug.py --cache work/v22_rereview_20260927_DEBUG/cache/index.json --debug-cache --indices 0 1 2 3 4 5 6 7 --physical-batch 8 --workers 8 --repeats 2 --reg-scale1 0.02 --reg-scale2 0.02 --baseline --output work/NEW_UNIQUE_EZSP_DEBUG_OUTPUT
```

다른 cache는 `--debug-cache`를 빼면 기존 production cache 내용/provenance 검사를 거친다. 지정 indices 수와 physical batch가 다르면 거부한다. subset 사용은 이 진단 도구에서 명시되며 최종 학습 설정에는 반영되지 않는다. 상세 report에는 cache·adapter·공식 소스·core hash, 설정, 규모, 실제/사용 수, 시간과 VRAM이 있다. 실패 시 nonzero 종료, 성공한 DEBUG도 training_ready=false.

## 남은 검증

실제 데이터에서 두 scale admission을 통과할 명시적 reg 설정, 학습된 feature의 안정성, 여러 seed/순열/graph의 광범위한 GPU 결정론, MIG10GB/production physical32, full-support L1/L2 update·memory refresh·validation·checkpoint/hash 비용 및 전체 epoch 개선은 **미검증**이다. 실제 scale2/GAT/backward 비용은 앞 단계가 거부되어 측정하지 않았다. online CP·nnU-Net 통합 및 효용은 이 새 L0로 검증하지 않았다. 장기 GNN/nnU-Net 학습·전체 평가를 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 요청한 공식 partition 적용 및 명시적 DEBUG 측정만 수행했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 요청한 상한은 거부 조건이다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. DEBUG1/8 측정, production batch 불변.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행에는 OOM이 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사와 실제 CT 측정을 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 합성 DEBUG 범위이며 production 연결 검증 아님.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
