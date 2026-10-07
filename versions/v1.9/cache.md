# 기존 네 군의 캐시 재사용과 정확 재개

네 군을 새로 학습하지 않는다. `resume_comparison_cached.py`는 기존 v1.8/v1.9 manifest와 원본 helper SHA를 검사하고, 기존 controller 또는 이미 생성된 v1.8 독립 continuation을 호출한다. 각 군의 최신 checkpoint에서 model, AdamW, scaler, scheduler, RNG, epoch와 batch cursor를 복원하는 학습 엔진은 바꾸지 않는다.

모델 10,434,532 parameters, L0/L1/L2 3/2/2층, 128D, 4 heads, 10mm, 두 view, source별 P 1개와 비교 7개, 전체 P+128U 평가, 원래 loss, 40epochs를 유지한다. 저장된 physical batch와 worker 수, CUDA/RSS/resident 한도 및 평가 chunk를 그대로 읽는다. 새 batch를 임의로 선택하거나 재측정하지 않는다.

## 확인된 비용

제공된 native_fixed epoch1 첫 6 update는 전체 경과 383초에 비해 `sec` 합이 약 63.59초였다. 이 값은 transfer/forward/backward/optimizer/checkpoint 등 update 내부 시간이다. 데이터 준비 대기는 밖에 있다. 로그에는 환자별 whole-case fields를 10.4~117.0초 동안 새로 만드는 작업이 함께 나온다. Prefetch와 겹칠 수 있으므로 각각의 시간 합을 서로 겹치지 않는 wall-time 비율로 계산하지 않는다.

10mm는 L0 ROI 범위다. 원본 feature 정의의 간 경계까지 거리와 종양까지 거리는 전체 CT에서 준비한다. 별도 실험 data 디렉터리에 완성 캐시가 없으면 같은 거리 변환을 다른 군에서도 다시 계산했다. source를 제외한 종양까지 거리를 포함하는 정적 upper feature도 새 캐시에서는 최초 준비가 필요했다.

첫 epoch ETA에는 이 준비 비용이 포함된다. 다음 epoch가 30분이라는 측정은 아직 없으며, 새로운 순환 U의 canonical graph, 원본 sampled view, collate, GPU update와 전체129 평가 비용은 남는다.

## 실행 수정

완료된 동일 binding의 `whole_case_fields`, `upper_static/source_raw`, `upper_static/lesions`, `canonical_local`만 다른 군의 data에서 가져온다. SHA·size·publication 검사 뒤 각 군의 data에 hardlink하거나, 다른 filesystem이면 검증된 바이트를 복사한다. 원래 loader가 binding과 실제 배열 검사를 계속 수행한다. 다른 군이 실행 중이어도 이미 원자적으로 게시된 불변 캐시만 읽는다. 원본 checkpoint, 로그, 잠금과 미완료 파일을 공유하거나 변경하지 않는다.

현재 군에 이미 있는 캐시는 교체하지 않는다. 다른 군에 동일 캐시가 없으면 원래 factory가 계산하며, `cache_miss`로 기록한다. 아직 어느 군에도 완성되지 않은 동시에 시작된 최초 계산을 중앙에서 하나로 묶는 기능은 없다. 모든 전처리가 즉시 끝나거나 모든 병목이 제거됐다고 표현하지 않는다.

동일 입력의 저장 방식만 바꾸므로 네 군의 후보·정답·학습 변인은 그대로다. 공유되는 것은 불변 전처리 바이트이며, 각 군의 mutable 학습 상태는 독립이다. helper의 전역 alias는 한 process 안에서만 잠시 연결하고 예외 후 복원한다.

v1.9에도 기존 `PressureBudget` 실행 정책을 연결한다. RSS192GiB/resident128GiB이면 실제 RSS160GiB 초과 시 cache 참조와 가능한 readonly page residency를 회수하고 다시 측정한다. 모델·입력·batch를 줄이지 않는다. 활성 입력 자체가 hard limit을 넘으면 실제 오류를 유지한다. Linux hint와 서버 전체 cohort의 시간 개선은 서버 실행 후 판정한다.

## 서버 사용

네 군은 서로 다른 GPU·터미널에서 각자 재개한다. 새 launcher는 지정된 군 하나만 실행하고, 기존 모델·Adam·RNG·cursor·측정 batch·worker·40epoch를 복원한다. 진행된 실험을 새 초기 가중치로 다시 시작하지 않는다.

```bash
CP_GPU=3 CP_ARM=native_fixed bash tools/server_comparison_cached.sh
```

`CP_ARM`은 selected/native/native_fixed/native_listwise 중 하나다. `CP_GPU`는 사용자가 할당한 물리 GPU 번호이며 반드시 지정한다. 네 터미널에서 GPU를 각각 지정한다. 현재 터미널에 남은 `CP_EXPERIMENT`가 다른 군을 같은 폴더로 보내지 않도록 새 Bash launcher는 이 변수를 읽지 않는다. 사용자 지정 root는 `tools/run_comparison_arm.py --experiment ...`로 명시한다.

| 군 | 독립 실험 폴더 | checkpoint 위치 |
| --- | --- | --- |
| selected | `v18_selected_m10_seed42_memory` | `selected/checkpoint_latest.pt` |
| native | `v18_native_m10_seed42_memory` | `native/checkpoint_latest.pt` |
| native_fixed | `v19_native_fixed_m10_seed42` | `native_fixed/checkpoint_latest.pt` |
| native_listwise | `v19_native_listwise_m10_seed42` | `native_listwise/checkpoint_latest.pt` |

각 root의 `data/`, `.pipeline.lock`, checkpoint가 독립이다. selected/native 독립 폴더가 없으면 비활성 원본에서 해당 군의 저장 상태를 정확히 복사한다. 원본 공유 root로 되돌아가는 fallback은 제거했다. 최초 복사에는 원본의 활성 writer가 없어야 한다. 이미 복사가 끝난 군은 자기 receipt와 immutable 바이트를 검증하므로, 다른 군이 과거 source의 잠금을 사용해도 재개를 막지 않는다. v19의 기존 root가 없으면 임의로 새로운 학습을 시작하지 않는다.

동일 군·동일 root에 검증된 작업이 이미 있으면 PID·명령·checkpoint를 출력하고 `ALREADY_RUNNING`으로 돌아온다. 두 번째 writer는 시작하지 않는다. 다른 군은 자기 root에서 계속 실행할 수 있다. 출처나 소유자가 다른 잠금은 지우지 않는다.

`preparation_reuse.jsonl`은 reused/cache_miss/binding_miss·읽은 source·destination·bytes·시간을, `execution_overrides/`는 요청한 실행 helper SHA와 기존 neural contract SHA를 기록한다. 요청 영수증은 학습 완료 증거가 아니다. 독립 continuation의 기존 memory 정책과 원래 manifest를 다시 쓰지 않는다.

## 중단 처리

기존 Ctrl+C handler는 pause flag만 바꾸고 `future.result()`와 전처리 thread 종료를 기다렸다. 그래서 GPU update가 4.75초여도 긴 CPU/I/O 작업에 걸리면 pause 메시지만 반복될 수 있었다. 이 상태를 checkpoint 저장 완료라고 표현하지 않는다.

새 launcher가 생성한 단일 Python 자식을 감독한다. Ctrl+C 한 번이면 해당 PID에 SIGINT를 보내 기존 저장 처리를 10초 기다린다. 응답하지 않으면 PID·부모 PID·생성 시각·정확한 argv·UID를 다시 대조하고 그 자식 하나에 SIGTERM을 보낸다. 반복 Ctrl+C는 대기 시간을 초기화하거나 신호를 반복하지 않는다. SSH·부모 셸·프로세스 그룹·다른 군은 신호 대상이 아니다.

각 update의 checkpoint는 atomic publication이다. 강제 중단 시 보존되는 것은 마지막 성공한 checkpoint이며, 저장하지 않은 현재 작업은 재개 때 다시 수행한다. 현재 batch를 반드시 새로 저장했다고 주장하지 않는다. SIGKILL은 사용하지 않는다. 커널의 중단 불가능한 I/O 상태에서는 SIGTERM도 지연될 수 있으므로, 10초는 종료 완료 보장 시간이 아닌 cooperative grace 시간이다.

이미 구형 launcher로 실행되어 Ctrl+C에 묶인 작업은 같은 서버의 별도 터미널에서 `tools/stop_comparison_arm.py --arm native_fixed --experiment /home/aicompetition06/Medical/experiments/v19_native_fixed_m10_seed42`로 중단할 수 있다. 잠금·host·UID·명령·arm·root·생성 시각을 확인해 정확한 Python 하나만 종료한다. 잠금·checkpoint·cache는 지우지 않으며, 종료가 확인되지 않으면 추가 신호와 재개를 거부한다.

## 검증 결과

회귀 검사 97개가 통과했다. 실제 CT/CUDA DEBUG는 train2명/validation1명, 동일 전체 모델·physical source batch2·worker4로 2epoch와 실제 update2회를 수행했다. 매 update 1,085개 tensor 모두 finite gradient가 있었다. 초기/epoch1/epoch2의 전체129 joint 평가3회를 마쳤다. 145 canonical graph, 3 whole-case field, 6 static upper publication을 가져왔으며 세 종류의 factory 재계산은 모두 0회였다.

완료 후 원본 엔진으로 재개했을 때 추가 update0회이며 model·optimizer·scaler·scheduler·RNG·cursor를 포함한 checkpoint content digest가 동일했다. torch 재직렬화의 archive byte SHA는 이 검사에서 달라졌다. 이를 가중치 변경으로 오인하거나 파일 바이트가 같았다고 기록하지 않는다. 새 wrapper를 실제 GPU에서 별도로 실행한 완료 재개도 통과했다. 기존 reference·cache와 원본 helper17개는 바이트가 보존됐다.

이는 실행 연결과 불변 캐시 재사용을 확인한 DEBUG다. 서버40epoch 학습, 추천 품질 개선 또는 서버 epoch 시간 개선을 증명하지 않는다. Linux `renameat2`·page residency·allocator hint는 Windows 호스트에서 실제 실행하지 않았다. [검증 기록](../../validation/comparison_cache/execution.json)에 현재 helper SHA와 범위를 보존한다.

새 독립 launcher는 별도 actual-CT DEBUG checkpoint로 selected/native의 실제 GPU 복원을 네 번 확인했다. 추가 update는 0이며, 한 번은 historical source에 살아 있는 잠금을 두어도 자기 폴더에서 그대로 재개됐다. 원본 checkpoint와 frozen neural/input helper17개를 보존했다. 총 회귀 156개가 통과했다. 네 실제 CPU process가 독립 잠금 8개를 동시에 유지·해제했고, 별도 자식의 정상 종료와 요청한 중단도 확인했다. Windows CPU 검사에서 SIGINT forwarding만 mock이며 단일 자식 종료는 실제 수행했다. Linux 터미널의 실제 SIGINT forwarding 및 네 GPU 동시 서버 실행은 검증했다고 주장하지 않는다. [새 실행 검증](../../validation/arm_launch/execution.json)에 범위를 분리했다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 CUDA calibration lock을 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 제공 로그와 별도 DEBUG를 구분한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 원본 엔진을 유지하고 실제 연결을 별도 DEBUG에서 확인한다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 서버 전체40epoch 및 시간 개선은 미검증이다.
