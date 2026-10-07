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

기존 작업을 사용자가 Ctrl+C로 안전하게 중단하고, 엔진의 저장 완료와 prompt 복귀를 확인한 뒤 새 코드 worktree에서 실행한다. 자동으로 서버 process를 종료하지 않는다.

```bash
CP_GPU=3 CP_ARM=native_fixed bash tools/server_comparison_cached.sh
```

`CP_ARM`만 selected/native/native_fixed/native_listwise 중 현재 재개할 군으로 지정한다. `CP_GPU`는 해당 서버의 물리 GPU 번호다. selected/native는 이미 만들어진 `_memory` continuation이 있으면 이를 선택하고, 없으면 기존 v18 root를 선택한다. 추가 두 군은 원래 v19 root를 선택한다. 다른 경로에서 실행했던 경우 `CP_EXPERIMENT`에 그 기존 root를 명시한다. 실험이나 checkpoint를 새 이름으로 자동 선택하지 않는다.

각 호출은 지정된 군 하나만 재개한다. 기존 v18 공통 root로 돌아가는 selected/native는 공통 잠금을 사용하므로 동시에 실행할 수 없다. 이미 분리한 `_memory` root는 각자의 잠금을 사용한다. `preparation_reuse.jsonl`은 reused/cache_miss/binding_miss·읽은 source·destination·bytes·시간을, `execution_overrides/`는 요청한 실행 helper SHA와 기존 neural contract SHA를 기록한다. 요청 영수증은 학습 완료 증거가 아니다. 독립 continuation의 memory 정책과 원래 manifest를 다시 쓰지 않는다.

## 검증 결과

회귀 검사 97개가 통과했다. 실제 CT/CUDA DEBUG는 train2명/validation1명, 동일 전체 모델·physical source batch2·worker4로 2epoch와 실제 update2회를 수행했다. 매 update 1,085개 tensor 모두 finite gradient가 있었다. 초기/epoch1/epoch2의 전체129 joint 평가3회를 마쳤다. 145 canonical graph, 3 whole-case field, 6 static upper publication을 가져왔으며 세 종류의 factory 재계산은 모두 0회였다.

완료 후 원본 엔진으로 재개했을 때 추가 update0회이며 model·optimizer·scaler·scheduler·RNG·cursor를 포함한 checkpoint content digest가 동일했다. torch 재직렬화의 archive byte SHA는 이 검사에서 달라졌다. 이를 가중치 변경으로 오인하거나 파일 바이트가 같았다고 기록하지 않는다. 새 wrapper를 실제 GPU에서 별도로 실행한 완료 재개도 통과했다. 기존 reference·cache와 원본 helper17개는 바이트가 보존됐다.

이는 실행 연결과 불변 캐시 재사용을 확인한 DEBUG다. 서버40epoch 학습, 추천 품질 개선 또는 서버 epoch 시간 개선을 증명하지 않는다. Linux `renameat2`·page residency·allocator hint는 Windows 호스트에서 실제 실행하지 않았다. [검증 기록](../../validation/comparison_cache/execution.json)에 현재 helper SHA와 범위를 보존한다.

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
