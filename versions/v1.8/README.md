# v1.8 — 비교 후보 조건 대조

성공한 v1.4 10 mm의 원본 모델과 source 종양·anchor를 유지하고, 비교 후보가 바뀔 때 학습이 유지되는지 확인한다.

| 군 | 한 문제의 학습 입력 | 평가 |
| --- | --- | --- |
| selected | 원본 P 1개 + 기존에 선택된 비교 좌표 7개. 기하 변형 제거 | 동일 P 1개 + 고정 U 128개 |
| native | 원본 P 1개 + 같은 환자의 고정 U 128개 중 7개. 에포크마다 순환 | 동일 P 1개 + 고정 U 128개 |

두 군을 seed 42의 **동일 초기 가중치**에서 새로 학습한다. 원본 전체 L0/L1/L2/scalar scorer, 3/2/2층·128D·4 heads, 48³ dense input, 두 sampled view, target erase, prototype bank, 40 epochs를 유지한다. 두 군 모두 `mean softplus(sU−sP) + 0.1 view consistency`를 쓴다. 기존 difficulty margin·ordinal·mining·CE·geometry corruption은 두 군에서 제거한다.

각 문제의 P는 해당 source 종양의 원래 anchor 한 개다. native의 U는 같은 환자에 고정된 미관측 비교 위치다. 원본 `RandomSampler`와 전용 seed `42 + 2003`을 유지하며 매 에포크 모든 원본 source 문제를 한 번씩 사용한다.

7개는 한 source 문제의 비교 위치 수다. physical batch는 완전한 source 문제 여러 개이며, 두 군의 실제 CUDA 처리량을 측정해 공통 값을 선택한다. native 군도 에포크당 원본 샘플을 한 번씩 사용한다. U 128개를 한 에포크에 19묶음 모두 처리하는 방식이 아니다.

48 GB 서버 기본 calibration 후보는 source 문제 `1 / 2 / 4 / 8 / 16 / 32`개다. source 문제 하나마다 후보 query 8개와 두 sampled view를 함께 GPU에 넣는다. OOM·자원 거부 결과를 기록하고 두 군에 모두 수용되는 실제 처리량 최선 값을 공통으로 선택한다. 큰 batch를 검증하지 않은 채 VRAM이 남는 작은 값을 고정하지 않는다.

전체 평가에서는 후보 129개의 L0를 chunk encoding한 후 **모든 후보가 있는 patient/prototype graph를 한 번** 처리한다. Best는 전체 129개 후보의 MRR·top1·pair-loss 순으로 선택한다. 환자별 source 문제 평균을 먼저 계산한 뒤 환자 평균으로 기록한다. 학습의 비교 위치 7개 점수와 평가의 후보 129개 점수를 섞지 않는다.

원본 signed source 목록의 train 151개·validation 36개 sample을 유지하는 계약이다. 기존 환자 split 설정은 84/21이지만 실제 materialized validation은 18명이며, 누락 3명을 manifest/stdout에 명시한다. 새로운 anchor나 가짜 sample을 생성해 21명이라고 표시하지 않는다. 기존 v1의 MRR 1.0은 이전 8후보 task의 보존 참고값이며, 이 selected 군의 결과로 대체하지 않는다.

U는 종양이 관측되지 않은 비교 위치다. CP 부적합 정답이나 latent negative 확률을 새로 만들지 않는다. Frozen U를 원본 placement filter로 제외·재추출하지 않는다. 형상이 생성되지 않으면 case/component/center와 함께 명시적으로 실패한다.

P의 source coverage = 1, source patch ring 통계, 자기 source를 제외한 다른 종양까지의 거리는 원본 v1 규칙으로 유지한다. U는 원본 비교 후보처럼 label 1 coverage와 모든 관측 종양까지의 거리를 사용한다. 원본 canonical-local 계산은 이 upper metadata를 사용하지 않으므로 중심·변환·원본 CT가 일치하는 저장된 local graph를 재사용할 수 있다.

정적 case 거리 배열과 원본 upper source/lesion 함수의 반환 배열도 입력·원본 코드에 결속해 저장한다. 값·dtype·shape·후보를 바꾸지 않고 반복 계산을 제거한다. 저장량·초기 준비 시간·재개 시간은 별도로 기록한다. Calibration은 원본 source 비용이 큰 문제부터 양군의 실제 update를 측정한다. 이 순서는 새 U의 모든 미래 입력에 대한 메모리 보장이 아니며, 실제 자원 한도는 학습·평가 내내 검사한다.

실제 구현과 파일별 역할은 [code.md](code.md), 고정 조건은 [config.md](config.md), 검증 상태는 [results.json](results.json)에 기록한다.

처음 구현한 `server_v18_u_bridge.sh`는 공통 root의 잠금 아래 selected 다음 native를 실행한다. 이 공통 root에서 두 프로세스를 동시에 실행하면 안 된다. 기존 결과는 그대로 보존한다.

2026-10-07의 별도 실행은 `CP_GPU=3 CP_ARM=selected bash tools/server_v18_independent.sh`, 다른 터미널에서는 `CP_GPU=5 CP_ARM=native bash tools/server_v18_independent.sh`를 사용한다. 출력 root는 각각 `experiments/v18_selected_m10_seed42_memory`, `experiments/v18_native_m10_seed42_memory`다. 각 root의 `selected/` 또는 `native/`에 checkpoint를 저장하고 `data/`·pipeline/data/arm 잠금도 분리한다. GPU 번호는 실행할 서버의 물리 번호로 바꾼다. 동일 명령을 재실행하면 해당 군의 저장된 정확한 cursor·optimizer·scaler·RNG에서 재개하며, 완료 군은 update를 다시 수행하지 않는다.

`run_v18_independent.py`는 실패한 기존 root의 봉인된 입력 계약·원본 helper SHA·공통 initial·calibration을 직접 검사한다. 기존 군의 checkpoint와 로그는 새 root에 독립 복사한다. 완료된 불변 cache는 SHA/size와 publication 구조를 검사한 뒤 hardlink한다. 원본 core는 cache를 실제 사용할 때 기하 binding을 다시 검사한다. 기존 절대 `prepared_data_root`는 역사 계약 안에 그대로 보존하고, 별도 `continuation.json`에 새 실행 위치를 결속한다. 새 entry는 위치가 바뀐 historical contract를 원래 controller에서 다시 생성하거나 덮어쓰지 않는다. 캐시 재계산은 피하지만 검증 I/O는 발생한다. 기존 root가 실행 중이면 복사를 거부한다.

서버에서 전체 129후보 초기 평가 중 process RSS 192 GiB 한도를 넘었다는 오류가 보고됐다. resident cache 128 GiB는 cache 참조만 계산하고, 전체 후보의 CPU canonical graph·두 sampled view·collate 복사·worker scratch와 기존 readonly mmap resident page는 별도로 존재한다. GPU L0 chunk=8은 이 CPU 입력 전체를 나누지 않는다. 제공된 로그만으로 각 항목이 차지한 실제 비율이나 단일 live batch 크기를 확정하지 않는다.

`host_memory.py`는 기존 RSS/cache headroom으로 회수 시점을 정한다. RSS192/cache128 설정이면 실제 RSS160 GiB 초과 시 provider LRU 참조를 해제하고, Linux readonly mmap에 `MADV_DONTNEED`를 요청하며 GC와 가능한 allocator trim을 실행한다. mapping을 닫거나 입력 값을 변경하지 않는다. 새 `_get` allocation 전과 batch 전후에도 검사한다. 실제 회수 전후 RSS·해제 cache 참조·OS hint 적용 여부를 `host_memory.jsonl`에 기록한다. RSS192·CUDA40·resident128·측정된 physical batch·worker·모델·10 mm·후보·40 epochs는 유지한다. 활성 batch와 작업공간 자체가 한도를 넘으면 실제 수치를 출력하고 계속 실패한다. 서버 오류 해결을 로컬 DEBUG만으로 보장하지 않는다.

실행 전 모델·설정·전체/실제 사용 샘플·GPU/CPU/RAM/I/O/컨테이너 제한·batch를 출력한다. `update_timing.jsonl`, `validation_epoch_XXX.json`, `curve.jsonl`, `checkpoint_latest.pt`, `checkpoint_best.pt`, coverage에 실제 학습과 시간·메모리·순위를 기록한다. DEBUG는 별도 실제 CT fixture로만 실행하며 성능 검증이나 전체 학습으로 승격하지 않는다.

결과 확인은 `python tools/summarize_v18_u_bridge.py --experiment /home/aicompetition06/Medical/experiments/v18_u_bridge_m10_seed42`로 한다. 초기·best·최근의 전체 129개 후보 지표와 학습 7개 비교 위치 지표를 구분해 짧게 출력한다. 중단·완료 후 재실행의 검증 결과와 자원·cache 사용량은 `invocations/`에 새 receipt로 남긴다.

이 대조는 비교 좌표 분포와 개별 비교 위치의 노출 빈도를 바꾼다. selected도 과거 v1과 다른 공통 pairwise loss를 쓰므로, selected의 학습 유지부터 확인해야 한다. native가 나쁘다는 결과만으로 128이라는 수 자체를 원인으로 단정하지 않는다.

로컬 회귀 검사 74개와 실제 CT/CUDA smoke가 PASS했다. train 2명·validation 1명으로 parameter 10,434,532개의 원본 모델을 두 군 각각 2 epochs 학습했다. 동일 초기값, 총 4번의 update에서 trainable tensor 1,085/1,085개의 gradient 연결과 5개 모듈 그룹의 실제 가중치 변경을 확인했다. 두 군 모두 초기·epoch 1·epoch 2의 전체 129개 후보 평가를 마쳤다. selected는 첫 update 후 중단·재개했고, 완료 후 재실행에서도 두 군 모두 추가 update 없이 model·optimizer·scheduler·RNG를 보존했다. [smoke 증거](../../validation/v18_u_bridge/smoke.json)에 범위와 검증 결과를 기록했다. 이 작은 smoke는 서버의 40 epochs 학습이나 추천 품질 검증을 뜻하지 않는다.

독립 실행·RAM 회수 패치의 회귀 검사 40개가 PASS했다. 추가 실제 CT/CUDA DEBUG에서는 같은 원본 전체 모델·physical source batch2·worker4로 native 2 epochs의 실제 update2회와 초기/epoch1/epoch2의 전체129 joint 평가3회를 마쳤다. 매 update gradient1085/1085 및 CNN/L0/L1/L2/scalar의 실제 가중치 변경을 확인했다. 완료 후 같은 명령을 재실행했을 때 checkpoint 바이트가 같고 추가 update는0회였다. 기존 파일은 보존했다. 측정된 update RSS 최대13.99 GiB, CUDA allocated 최대2.18 GiB는 이 작은 DEBUG 입력의 값이다. Linux page/allocator hint와 서버192 GiB 한도에서의 전체 cohort 재개는 아직 검증하지 않았다. [추가 실행 증거](../../validation/v18_memory/execution.json)에 범위를 명시한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 10 mm·7개 비교 조건을 명시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 두 군 실제 CUDA calibration을 요구한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실행 전 resource snapshot을 기록한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. candidate별 실제 실패·peak를 보존한다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 CT/CUDA update에서 gradient 1,085/1,085개와 5개 그룹의 가중치 변경을 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 서버 40 epochs 성능은 아직 미검증이다.
