# v2.1 공통 donor와 누수 방지 계약

2026-09-19. 사용자가 지적한 donor/recipient 혼동을 수정한다. 기존 모델 깊이·폭, KD-tree/graph 규칙, 후보 128개, GNN 40 epoch, nnU-Net 250 epoch, CP 확률 0.5는 유지한다. 이전의 잘못된 환자별 후보 분기는 `versions/v2/pre_shared_donor_fix_20260919/`에 보존한다.

## 모든 학습 환자가 증강 대상

소형 종양 조건은 donor 선택 조건이다. 정답 mask label=2의 기존 6-connectivity component 중 등가 구 직경 ≤20 mm를 사용한다. 원래 소형 종양이 있는지 여부로 recipient를 제외하거나 서로 다른 donor 규칙을 적용하지 않는다.

native CP는 각 이벤트에서 공통 donor 풀의 source 하나를 균등 추출한다. 풀에는 모든 inner-train 적격 source가 들어간다. 해당 recipient의 원래 소형 종양 수는 선택 함수의 입력이 아니다. CP 이벤트 확률 0.5를 유지하므로 모든 minibatch sample에 무조건 종양을 추가한다는 의미는 아니다.

선택된 donor를 recipient의 실제 spacing으로 변환하고, 기존 hard geometry 조건을 통과하는 128개 위치를 평가한다. frozen GNN의 argmax 위치에 donor CT와 mask를 함께 반영한다. 원본 CT/GT는 변경하지 않는다. 유효 후보가 부족하면 오류를 기록하며 다른 donor나 축소된 후보 풀로 조용히 대체하지 않는다.

## 분할별 경계

| 단계 | 학습/수신 환자 | donor 또는 관측 evidence | 금지 |
|---|---|---|---|
| GNN optimizer/support memory | inner_train 84명 | inner_train의 실제 T만 | inner_val·outer_val의 학습 gradient, donor, support 편입 |
| GNN 구조적 검증 | inner_val 21명 | frozen inner_train evidence; target context는 U | 검증 GT로 parameter 갱신, 검증 환자의 T를 donor evidence로 전달 |
| 최종 nnU-Net 학습 CP | outer_train 105명 전부 | 동일한 inner_train donor 풀 | outer_val·test를 donor 또는 recipient로 사용 |
| 최종 nnU-Net 검증/예측 | outer_val 26명 | CP 없음 | 증강 loader/CP service 연결, 정답을 모델 예측 입력으로 사용 |

inner_val 21명은 GNN의 검증 환자이면서 최종 segmentation 단계에서는 outer_train의 일부다. 두 단계의 train/validation 경계를 혼동하지 않는다. native fingerprint·planning·정규화 추정은 outer_train으로 수행한 기존 계획을 검증해서 사용한다.

split 중복, donor membership, checkpoint의 실제 patient parameters/support IDs, 원본 CT/GT SHA, native 계획 및 split SHA, catalog/entry/원시 paste payload SHA를 검증한다. 파일 이름만 바꾼 동일 원본 영상이 inner/outer 분할을 넘는 경우도 거부한다. 이 검사는 환자 ID와 원본 파일 동일성 계약을 검사하며, 서로 다른 ID·파일로 제공된 동일인의 다른 검사까지 식별하는 환자 재식별 기능은 아니다.

## GNN context와 실제 CP 이벤트를 분리

GNN은 모든 실제 소형 종양의 원위치 T anchor를 유지한다. U context는 recipient의 소형 종양 유무와 무관하게 동일한 배정 함수를 사용한다. 각 분할에서 donor 목록과 환자 목록을 seed로 섞고 순환 배정하며, 한 round의 이벤트 수는 `max(donor 수, recipient 수)`다. 모든 donor와 모든 환자가 참여하고 환자별 이벤트 수 차이는 최대 1이다. 각 이벤트는 전체 128개 후보를 사용한다.

현재 원본에서는 donor 527개, 실제 T anchor 662개다. GNN train context는 67,983개, validation context는 67,591개, 총 135,574개다. 중단한 잘못된 1,704,342개 Cartesian 분기를 교체한 결과이며, 환자·donor를 버리거나 graph 내부 node/edge/반경을 줄인 것이 아니다. 이 균형 배정은 명시적인 실험 설계 선택이며 의료 성능 우월성이 입증됐다는 뜻은 아니다.

GNN context 데이터의 균형 배정과 native CP의 매 이벤트 균등 추출은 서로 다른 단계다. native CP에서 각 환자는 전체 donor 풀에 접근한다. 특정 donor를 환자별로 고정하거나 매 epoch 한 개로 제한하지 않는다.

## 요청 시 생성과 자원 처리

bank `index.json`은 전체 recipient/donor catalog다. `complete=true`의 범위는 catalog의 검증 완료이며 전체 donor×환자의 후보 파일 생성 완료가 아니다. `completion_scope`와 `materialization` 필드를 함께 읽어야 한다.

native augmentation worker가 실제 CP 이벤트에서 선택한 pair만 중앙 생성기에 요청한다. 전체 128개 graph를 메모리에서 만들고 모두 점수를 계산한다. 128개 점수·좌표와 argmax로 확정된 한 위치의 실제 raw CT/GT paste payload를 저장해 같은 pair에서 재사용한다. 선택 규칙이 고정 argmax이므로 사용되지 않을 127개 paste 파일을 만드는 작업만 제외한다. 후보 평가나 graph 크기를 줄이지 않는다. inner/outer split 밖의 요청은 영상 로딩 전에 거부한다. validation loader는 기존 일반 nnU-Net loader로 유지한다.

중앙 frozen GNN의 CUDA 연산과 segmentation forward/backward/validation은 공통 lock으로 겹치지 않게 한다. 실제 native 동시 모델 메모리·처리량은 본학습 checkpoint가 생긴 뒤에도 검증해야 한다. cold pair 생성 대기 시간이 사라졌다고 주장하지 않는다.

CPU queue는 작업 하나가 끝나면 다음 작업을 즉시 배정한다. 원본 영상·organ depth·component map은 RAM 예산이 있는 cache로 재사용한다. L1은 모든 context를 유지한 activation checkpointing과 CUDA attention의 65,535-row 제한에 맞춘 실행 분할을 사용한다. 이 분할은 graph/data 개수를 제한하지 않는다.

## 검증 상태

누수·동일 donor 정책 DEBUG 13개와 기존 회귀 26개, 총 39개가 모두 통과했다(실패·skip 0). 인증된 worker 통신의 오류 전달, 일반 validation loader 유지, native preparation 저장·재로딩 및 선택한 raw payload의 실제 paste도 검사한다. 증거는 `work/shared_donor_fix_20260919/debug_final/`이다. 실제 84명에 대응하는 67,983개 context cardinality의 합성 L1/L2 역전파 검사도 통과했다. 이 검사는 L0 또는 전체 의료 학습의 완료 증거가 아니다.

실데이터 외부 donor → 소형 donor 없는 recipient → 128개 위치의 실제 GNN 계산 → native CT/GT paste 검사가 통과했다. `liver_1`의 component 1을 원래 적격 소형 source가 0개인 `liver_2`에 적용했고, 128개 점수의 argmax index 63에 native 종양 support 3,165 voxel을 생성했다. CT는 유한값이며 붙인 영역 밖 GT 및 원본 CT/GT SHA는 변하지 않았다. native baseline CT 최대 오차 0 / segmentation exact도 확인했다.

본훈련 가중치가 없어 이 도구는 명시적으로 미학습 모델을 주입한다. `debug=true` catalog를 production 학습에 넣으면 거부한다. 모델 성능이나 전체 학습 완료의 증거가 아니다. 검사 전체는 581초였으며 원본 검증·후보 준비·batch/worker calibration·native transport를 포함한다. 전체 학습 support와 segmentation 모델이 함께 있는 production 처리량은 별도로 측정해야 한다. 성공 증거는 `work/shared_donor_fix_20260919/real_cp_DEBUG3/verification.json`이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 잘못된 donor 조합 정책 변경은 위에 별도로 명시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 및 병목 원인을 먼저 조사했다. attention kernel 제한을 실행 분할로 처리했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결을 DEBUG로 검증했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
