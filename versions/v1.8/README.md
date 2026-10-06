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

고정된 실행 코드를 Git으로 가져온 서버 checkout에서 `CP_GPU=3 CP_ARM=both bash tools/server_v18_u_bridge.sh`를 사용한다. 먼저 selected 40 epochs, 이어서 native 40 epochs를 실행한다. 같은 명령을 재실행하면 각 군의 저장된 정확한 cursor·optimizer·scaler·RNG에서 재개한다. 결과는 `experiments/v18_u_bridge_m10_seed42/{selected,native}`로 구분된다. 단일 GPU 비교라 shared pipeline lock으로 중복 실행을 차단한다. 완료 군은 업데이트를 다시 수행하지 않는다.

실행 전 모델·설정·전체/실제 사용 샘플·GPU/CPU/RAM/I/O/컨테이너 제한·batch를 출력한다. `update_timing.jsonl`, `validation_epoch_XXX.json`, `curve.jsonl`, `checkpoint_latest.pt`, `checkpoint_best.pt`, coverage에 실제 학습과 시간·메모리·순위를 기록한다. DEBUG는 별도 실제 CT fixture로만 실행하며 성능 검증이나 전체 학습으로 승격하지 않는다.

결과 확인은 `python tools/summarize_v18_u_bridge.py --experiment /home/aicompetition06/Medical/experiments/v18_u_bridge_m10_seed42`로 한다. 초기·best·최근의 전체 129개 후보 지표와 학습 7개 비교 위치 지표를 구분해 짧게 출력한다. 중단·완료 후 재실행의 검증 결과와 자원·cache 사용량은 `invocations/`에 새 receipt로 남긴다.

이 대조는 비교 좌표 분포와 개별 비교 위치의 노출 빈도를 바꾼다. selected도 과거 v1과 다른 공통 pairwise loss를 쓰므로, selected의 학습 유지부터 확인해야 한다. native가 나쁘다는 결과만으로 128이라는 수 자체를 원인으로 단정하지 않는다.

로컬 회귀 검사 74개와 실제 CT/CUDA smoke가 PASS했다. train 2명·validation 1명으로 parameter 10,434,532개의 원본 모델을 두 군 각각 2 epochs 학습했다. 동일 초기값, 총 4번의 update에서 trainable tensor 1,085/1,085개의 gradient 연결과 5개 모듈 그룹의 실제 가중치 변경을 확인했다. 두 군 모두 초기·epoch 1·epoch 2의 전체 129개 후보 평가를 마쳤다. selected는 첫 update 후 중단·재개했고, 완료 후 재실행에서도 두 군 모두 추가 update 없이 model·optimizer·scheduler·RNG를 보존했다. [smoke 증거](../../validation/v18_u_bridge/smoke.json)에 범위와 검증 결과를 기록했다. 이 작은 smoke는 서버의 40 epochs 학습이나 추천 품질 검증을 뜻하지 않는다.

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
