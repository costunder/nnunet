# v2.22 support 실행 병목과 CPU producer 분리

이전의 “최적화 완료” 설명은 검증 범위를 넘어섰다. 당시 확인한 것은 대형 실제 학습 6 batch와 일부 support를 이용한 검사였다. 전체 support 처리량 또는 A100 MIG의 전체 epoch 시간을 검증한 것이 아니었다.

## 이번에 확인한 병목

RTX5070Ti에서 실제 inner-train 관측을 원래 순서로 처리했다. 전체 모델 5,550,806 parameters, physical batch32, CNN/GNN 깊이·너비·노드·엣지, bf16, 256MiB workspace, CPU decode workers8, 매 batch checkpoint, `release_unused=True`를 유지했다. CUDA allocator 상한은 9,000,000,000 bytes이며 드라이버 메모리까지 포함하는 MIG 에뮬레이션은 아니다.

기존 threaded loader의 첫 2,368개/74 batch에서 배치 시간 합계는 412.270초였다. GNN CUDA-event 구간 373.046초, CNN 2.686초, loader 대기 4.063초, checkpoint 제출 8.547초였다. checkpoint 제출 시간은 fsync 완료 시간이 아니다.

**이 GNN 구간은 GPU 커널의 순수 연산 시간과 같지 않다.** 커널 사이 CPU launch 지연도 포함한다. 같은 그래프를 반복해서 계산하고 CPU 준비를 분리하자 이 구간이 크게 줄었다. 기존에는 같은 Python 프로세스 안에서 다음 그래프를 만드는 스레드와 GPU 연산을 호출하는 주 스레드가 간섭했다. GIL과 CPU 스케줄링 각각의 기여율까지 분리 측정한 것은 아니다.

## 적용한 실행 변경

- `tools/v222_process_loader.py`: CPU producer 2개가 각 4개 decode thread를 사용한다. 전체 physical batch를 disjoint graph로 조립하고 PyTorch CPU 공유 storage를 통해 전달한다. GPU에는 동일한 batch32가 들어간다.
- producer는 support/optimization/validation 사이에 유지된다. canonical cache를 단계마다 폐기하지 않는다. producer들의 합산 cache 예산은 생성 시 가용 RAM의 20%다. 이는 프로세스 전체 RSS 상한은 아니다.
- 최대 2개 batch를 순서대로 prefetch한다. CPU producer에서 CUDA 초기화나 pinning을 하지 않고 주 프로세스에서 pinning·전송한다. 오류를 숨기거나 thread/CPU 학습으로 자동 대체하지 않는다.
- `tools/run_v222_process_runtime.py`: 명시적인 loader 전환 진입점이다. 직전 그대로의 runtime SHA 또는 자신의 정확한 SHA만 허용한다. 원본 checkpoint 파일을 수정하지 않고 기존 실행기에 전달한다. 가중치·Adam·RNG·epoch·batch cursor·부분 support·cluster plan과 수치/allocator 정책은 보존한다.
- 서버 진입점의 `--runtime process`로 선택한다. 원본 run 생존 차단과 tqdm의 Ctrl+C 저장 중단을 유지한다. 기본 `optimized` 경로의 runtime 소스 세 파일은 바꾸지 않아 이전 checkpoint의 SHA를 불필요하게 무효화하지 않는다.

첫 2,368개 동일 관측 비교: **412.270 → 185.302초, 약 2.22배**. 신규 프로세스 시작 비용을 포함한 배치 시간 합계다. 두 실행 모두 한 번 측정한 로컬 결과이며 A100 MIG 속도를 보장하는 수치는 아니다.

## 분리해서 검사한 대안

1. `empty_cache`를 매 batch 호출하지 않는 경우: 같은 2,368개에 약 406초로 큰 개선이 없었고 reserved가 8.80GB까지 증가했다. 해당 실행 중 짧은 CPU IPC 검사가 겹쳤으므로 정밀한 속도 차이는 해석하지 않는다. 기존 allocator 정책을 유지한다.
2. CSR message 합산: 실제 대형 graph에서 단일 forward가 빨라졌으나 BF16 최종 embedding 최대 절대 차이 0.0078125가 발생했다. 합산 순서도 달라진다. **최종 실행에는 적용하지 않았다.** 원래 attention·합산·gradient 연산을 유지한다.
3. producer 1개×8 thread와 2개×4 thread를 실제 8 batch에서 비교했다. 초기화 포함 36.68초와 27.59초였다. 선택한 경로는 2개이며 이후 전체 support를 별도로 검사한다.

## 검증 범위

정책·cache·writer·중복 재개 차단·Ctrl+C·epoch 표시 회귀 32개 통과. 실제 cache 2개에서 epoch3의 CT·노드·엣지·배치 순서가 기존 loader와 정확히 일치하고 loader를 닫아도 producer pool이 유지됨을 확인했다.

**전체 support 11,279개/353 batch 완료: 925.646초(15분26초), OOM 없음.** 최대 allocated 3.505GB, reserved 5.090GB. 입력은 실제 전체 inner-train cohort이고 weights는 기존 실제 CT DEBUG 6-step checkpoint다. 최종 의료 성능 평가나 40epoch 학습 완료가 아니다. CPU 회귀 검사는 비교 prefix 이후의 전체 실행 일부 구간과 겹쳤다. 실행 중 한 시점의 주/producer RSS는 4.155/7.146/7.049GB, 가용 RAM31.126GB였다. RSS 합계에는 공유 페이지가 중복 집계될 수 있다.

기존 경로와 공통인 **2,368개 support embedding 전체가 bitwise 일치**했다. 각 batch의 처리 위치·노드·엣지 수도 일치했다.

**실제 대형 query 6 batch 학습 완료:** batch32, 전체 모델, 실제 저장된 support prefix1,216개, 9GB allocator, source/cache identity 검증. 6 step 합계51.848초, step2–6 평균6.924초, 최대 allocated6.403GB/reserved8.009GB. 모든 parameter gradient가 존재하고 유한하다. 기존 256MiB 실행과 비교한 step2의 loss·전체 가중치·Adam 상태가 bitwise 일치했다. 이는 전체 학습/전체 epoch 또는 A100 MIG 검증을 대체하지 않는다.

Git 보존 증거: [support_process_20260926_DEBUG.json](../validation/v222_r6/support_process_20260926_DEBUG.json). 재현 도구: `profile_v222_support.py`, `verify_v222_process_loader_debug.py`, `verify_v222_runtime_debug.py --mode process`, `record_v222_support_validation_debug.py`. 서버 학습에는 아직 적용하지 않았다.

기존 baseline 진단은 2,368개를 저장한 뒤 소유한 로컬 STOP marker로 중단했다. 당시 진단 harness가 정상 pause 반환을 assert로 처리하여 AssertionError를 출력했다. 완료된 74개 profile 행과 durable checkpoint는 남아 있으며, 이를 전체 support 완료로 세지 않는다. 진단 harness는 pause를 명시적으로 기록하도록 수정했다. 서버 학습은 중단하거나 변경하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 full-model 업데이트와 전체 유한 gradient 확인.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
