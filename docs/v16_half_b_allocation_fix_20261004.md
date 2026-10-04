# v1.6 B: 다른 동일 GPU 서버의 최초 실행 자원 검사 수정

사용자가 보내 준 서버 로그에서 B는 `bind_execution_resources`의 전체 fingerprint 동등성 검사에서 중단됐다. 학습 update, 초기 validation, support refresh 전에 발생했다. 해당 로그는 `validation/v16_half_b_allocation_20261004/server_failure.txt`에 원문 그대로 보존했다. 로그에는 현재 fingerprint 전체가 없으므로 어떤 필드가 달랐는지는 단정하지 않는다.

원래 fingerprint에는 GPU UUID 환경변수, Linux platform, CPU 개수·affinity·cpuset, RAM 총량·cgroup 한도, 스케줄러 작업 정보 등이 포함된다. GPU 종류와 실제 실행 가능한 용량이 충분하더라도 다른 서버나 GPU UUID면 전체 비교가 실패할 수 있었다. Free/used 메모리 같은 변동값이 원인이었다고 주장하지 않는다. 이 오류는 L0 그래프, 학습 loss, L1/L2 연산 오류가 아니다.

## 변경

`--allocation-policy current_allocation`을 명시한 **새 B 폴더의 최초 실행**에서만 baseline과 다른 allocation을 허용한다. GPU name·VRAM 총량·SM 개수·compute capability·MIG 여부는 실제 측정된 baseline과 같아야 한다. GPU UUID·서버 CPU 구성·스케줄러 identity의 차이는 그대로 기록한다.

원래 verified baseline의 physical batch, workers, gradient accumulation을 유지한다. 이 셋이 실제 config와 다르면 거부한다. 현재 단일 CUDA device, GPU 잔여 공간과 자체 할당량, CPU affinity·cgroup v1/v2·스케줄러 제약을 반영한 CPU 용량과 RAM 잔여 공간을 확인한다. 선언한 GPU/RSS 한도가 현재 capacity를 초과하면 명확하게 실패하며 batch·worker·모델을 자동으로 줄이지 않는다. 이 검사는 시간별 capacity의 영구 보장이 아니며 실행 중 기존 ResourceBudget 검사도 유지한다.

Native pipeline에는 **현재 실제 fingerprint**를 반환한다. Baseline fingerprint로 치환하지 않는다. `execution_lock.json`과 invocation별 `allocations/admission_*.json`에 이전·현재 자원과 바뀐 필드를 기록한다. 처리량을 새로 측정했다는 표시, 같은 서버의 시간 측정이라는 표시를 만들지 않는다. 최초 실행 이후 같은 B를 재개할 때는 자신의 실제 allocation lock과 native training signature를 계속 검사한다. 이미 학습한 B를 임의의 다른 allocation으로 옮기는 기능은 추가하지 않았다.

이 수정으로 기존 source-bound manifest가 달라지므로, 실패한 B 폴더를 덮어쓰지 않는다. 재시도 폴더는 다음과 같다.

```text
/home/aicompetition06/Medical/experiments/v16_m10_halfB_seed42_20261004_r2
```

실패는 학습 전이므로 잃는 B 학습 update가 없다. 원래 완료된 10mm baseline의 cache, prototype bank, source, 결과를 그대로 사용한다. A 또는 baseline 재학습과 cache 재준비는 실행하지 않는다. 모델·GT·loss·10mm 범위·두 view·후보8/pool128·seed42·40epoch는 바꾸지 않았다.

## 검증

- Resource validator와 entry/recipe integration, 기존 B 실행 계약을 합쳐 68 UNIT PASS. 자원 부족, 다른 GPU 종류, 다중 GPU, CPU fallback, worker/batch/accumulation 변경, 기존 실행 lock 변경을 거부하는 반례를 포함한다. CPU 검사이며 모델 정확도 검사로 보고하지 않는다.
- 실제 RTX 5070 Ti의 CUDA 연산과 archived pipeline의 resource collector를 사용해 수정된 entry 검사 경로를 실행했다. 현재 GPU/CPU/RAM 수치는 실제 값이다. 이전 UUID/platform만 명시적인 DEBUG counterfactual로 바꿔 최초 allocation 변경을 검사했으며 서버 실패 전체를 독립 재현했다고 주장하지 않는다. Native fingerprint가 실제 현재 allocation이고 batch/workers가 그대로임을 확인했다. Optimizer update는0이다.
- 기존 실제 CT 8 update와 CUDA parity 근거는 `validation/v16_half_b_20261004`에 보존했다. Neural 모델·support·native prompt/clustering 모듈 byte가 수정되지 않은 것을 확인했다. 이번 자원 검사 수정으로 그 근거를 새 서버 장기 학습 또는 새로운 추천 품질 검증으로 승격하지 않는다.
- 로컬 UNIT 두 개는 workspace 임시 폴더 생성 권한에서 대기해 중단했다. 정확한 PID·명령을 확인한 직접 생성 검사 worker22328/37064만 중단한 뒤, 쓰기 권한을 맞춘 검사는 모두 통과했다. 서버나 셸·SSH 프로세스는 중단하지 않았다.

현재 상태는 실행 검사 수정 완료, UNIT/CUDA resource probe 완료, 서버 재시도 전이다. 전체 B 학습 및 전체 B 평가 결과는 아직 없다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 원래 측정한 비교 조건을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 오류는 OOM이 아닌 identity 검사였다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. UNIT/DEBUG 반례 입력은 실제 성능 결과로 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 실제 CT/CUDA 근거와 core source byte를 보존했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
