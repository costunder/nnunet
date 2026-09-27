# v2.2 B01–B03: exact-resume 무결성 보강 (2026-09-27)

검토 기준은 `758275b59d188c459ea40ef25a07b12d04367043`이다. 사용자 첨부 텍스트 전체를 읽었다. 첨부 텍스트 안의 sandbox 링크에 있는 별도 ZIP/MD를 내려받아 읽었다고 주장하지 않는다. 검토가 확인한 R01–R06 수정은 유지하고, 새로 지적한 조건부 손상 checkpoint admission 세 곳을 수정했다. 정상 trainer가 실제로 잘못된 가중치나 plan을 만들었다는 증거로 해석하지 않는다.

## 구현

| 항목 | 변경 | 유지한 정상 의미 |
|---|---|---|
| B01 final-memory | 현재 model의 모든 state_dict 항목을 embedded best hash와 비교. 부분 support의 내용 hash와 run/phase/epoch/step/model generation 검사. 완료 support는 생성 generation을 이어받아 final artifact에 결속 | optimization에서는 현재 가중치와 best 또는 epoch reference L0가 다를 수 있음. best를 강제로 load하지 않음 |
| B02 cluster plan | immutable CPU checkpoint에 plan·generation hash 저장. 생성 당시 teacher 모델 hash/step, support generation/content, query group 결속. active/assignment/membership/mass/classes/centers/prior/alignment weights/지원 입력 검사 | resume 때 teacher를 다시 fit하지 않음. 같은 그룹의 이후 update에도 처음 만든 plan 유지. 다음 그룹에서 정상 생성 |
| B03 RNG | training device, 논리 CUDA 개수/순서/UUID/VRAM/state shape를 identity에 기록. 실제 재개 장치와 identity 비교. Torch/CUDA/NumPy/Python state 내용 hash, 길이·형식·layout 검사 | 빠진 RNG를 seed로 복구하지 않음. CPU 전용 artifact의 CUDA 0개는 허용. inference final은 RNG 불필요 |

구현 위치: `tools/v22_resume_integrity.py`, `tools/v22_artifacts.py`, `tools/v22_ranking_training.py`, `tools/v222_runtime_execution.py`, `tools/v222_support_snapshot.py`. 새 helper를 runtime hash 목록에 추가했다.

hash는 파일 손상과 state/generation 불일치를 검출하는 장치이며, 공격자가 모든 metadata와 hash를 함께 다시 쓰는 상황을 막는 전자서명이 아니다. CPU snapshot을 만든 뒤 seal하고, no-update support pass의 model/Adam CPU snapshot은 기존처럼 재사용한다. teacher hash는 그룹당 한 번 생성한다. 속도 개선을 측정했다고 주장하지 않는다.

artifact 계약은 `observed_rank_artifact_v4`다. v3에는 새 증거가 없으므로 metadata만 덧붙여 exact resume하지 않는다. 기존 checkpoint/결과는 보존했다. 이번에는 **core graph/source/geometry 계약을 바꾸지 않았으므로** 758275b에서 재생성한 동일 provenance/content-bound graph cache를 그대로 검증해서 사용했다. 더 오래된 geometry/content-binding 없는 cache는 여전히 불가하다. runtime이 바뀌었으므로 calibration은 새로 측정했다.

## 검증 결과와 한계

- 단위/회귀: **82개 통과**. 기존 72개 + 새 10개 test method이며 내부 반례 수를 중복 합산하지 않았다. 이전 best가 epoch 1이고 마지막 epoch가 2인 CPU fixture에서 final-memory 시작/부분/완료 prefix 허용, finite weight/buffer/prefix 손상 거부를 검사했다. CPU fixture는 학습 성능 증거가 아니다.
- 실제 CT + RTX 5070 Ti + 전체 5,550,806 parameter 모델: 명시적 DEBUG 8 train/2 validation, 1epoch/4 updates. 기존 전체 그래프 규칙과 L0/L1/L2 규모를 유지했다. 첫 update 저장 후 재개와 연속 실행에서 다음 update의 loss·544개 parameter tensor의 gradient·모델·Adam이 bitwise 동일했고, 마지막 Torch/CUDA/NumPy/Python RNG와 support도 동일했다.
- actual final-memory 시작/부분/완료 checkpoint 각각 재개하여 최종 모델/support bitwise 동일. rolling 파일 하나만 복사하고 외부 best 파일 읽기를 금지한 portable resume도 동일.
- 실제 GPU가 만든 checkpoint에 CUDA RNG empty/missing/truncated/dtype, finite prototype class/center swap, teacher step, final weight/prefix/generation을 변조한 **10개 admission 반례를 거부**했다. 손상 파일로 실제 optimizer를 실행한 것이 아니라 실행 전 CPU admission에서 거부한 검사다.
- 같은 episode 수치 검사는 실제 CT batch를 **명시적으로 한 번 더 사용하는 별도 DEBUG probe**다. teacher_step=0/current_step=1인 plan의 재생성을 금지하고 직렬화 전후 다음 loss·L0/L1/L2 gradient·Adam·모델·RNG의 bitwise 동일성을 확인했다. 정상 lifecycle DEBUG는 그룹당 batch 하나이므로 이 probe를 여러 batch가 있는 전체 epoch 검사로 표현하지 않는다. 복수 batch/다음 그룹의 teacher cursor 계약은 별도 CPU fixture로 검사했다.
- 실제 CT run은 1epoch이므로 **실제 CT에서 earlier-best < last-epoch를 선택하는 다중 epoch 검증은 하지 않았다**. 해당 선택/가중치/buffer 계약은 위 CPU fixture에서 확인했다.

검증 driver: `tools/verify_v22_integrity_process_debug.py`, `tools/verify_v22_same_episode_debug.py`, 기존 `tools/verify_v22_portable_resume_debug.py`. 원문 로그/결과/체크포인트는 `work/v22_integrity_20260927_DEBUG/`, 소스·실행·출력 hash와 결과 요약은 `validation/v222_r6/resume_integrity_20260927_DEBUG.json`이다.

자원: RTX 5070 Ti 16GB, CPU 8 physical/16 logical. loader 0/2/4/8을 측정했고 warm 처리량 기준 4를 선택했다. DEBUG의 그룹당 2개 관측에 physical=effective batch 2, accumulation 1. 학습 calibration peak allocated 408,728,064 bytes / 2.308 graphs/s는 **이 작은 support DEBUG의 측정**이며 production physical batch나 전체 support 메모리 근거로 사용하지 않는다. RAM/가용 메모리와 실제 graph node/edge 통계는 실행 기록에 보존했다. 전체 CPU/GPU를 수치상 100% 점유하도록 바꾸지 않았다.

## 남은 연구·통합 검증

pair estimator(.5/1.5 포함), CE/rank/정렬 loss, donor schedule, seed42, CP80%, 모델·그래프·전체 데이터 규모는 바꾸지 않았다. 종양 본체 shortcut 가능성, GT liver union 의존성, fixed-donor 효용도 해결됐다고 주장하지 않는다.

G3 전체 support/전체 physical batch 및 worst-batch 검증, G4 native bank/RPC/adapter/trainer와 nnU-Net 최종 입력 연결, G5 CP 대조 성능 실험은 미완료다. 이번 작업은 full cohort 학습 또는 실제 nnU-Net 학습을 시작하지 않았다. 이전 2,368voxel raw paste 증거는 이전 revision의 별도 검사이며 이번에 재실행했다고 합산하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 명시적 DEBUG만 분리했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번 측정은 DEBUG 범위다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. 이번 실행 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 production에 사용하지 않았다. CPU fixture는 별도 표시했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 모델 gradient 비교로 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
