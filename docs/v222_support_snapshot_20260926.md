# v2.22 r6 support 복사·준비 대기 개선

2026-09-26. 변경 대상은 실행 계층이다. v1 paired L0와 v2.22 L1/L2의 모델, 그래프, split, 관측 수, 학습 목표는 변경하지 않는다.

## 수정 내용

1. Support pass 동안 가중치와 Adam 상태는 바뀌지 않는다. 이전에는 매 배치마다 같은 tensor를 GPU에서 CPU로 복사했다. `v222_support_snapshot.py`는 pass 시작에 한 번 복사하고 종료 시 폐기한다. 가중치 갱신이 있는 optimization 단계는 기존대로 매번 새 snapshot을 만든다. Tensor version 변경을 감지하면 저장을 거부한다.
2. 매 배치 standalone checkpoint 저장은 그대로다. 부분 support, 진행 위치, RNG 등 바뀌는 내용은 매번 새로 복사한다. 원자 저장과 fsync, durable 상태, 쓰기 실패 전파, STOP_AFTER_BATCH 후 저장 완료 대기를 유지한다. 디스크에 완전한 파일을 매번 쓰는 비용까지 제거한 것은 아니다.
3. CPU graph 준비를 2 process × 4 decode threads에서 4 process × 2 decode threads로 변경한다. 총 decode workers8과 physical batch32를 유지한다. CPU 공유 storage 전달 후, 주 프로세스의 별도 thread가 최대4개 선행 배치를 pin한다. 소비 순서를 보존하고 오류를 전파한다.
4. Producer와 bounded canonical cache는 support/optimization/validation 사이 유지한다. 합산 cache 예산은 생성 시 가용 RAM의20%다. 가중치에 의존하는 embedding은 optimizer 갱신을 넘어 재사용하지 않는다.

## 재개와 재현

`--runtime process`가 이 경로를 선택한다. 이전 optimized 3파일 해시 및 실제 배포된 683a9f7의 process 5파일 해시에서만 명시적 전환을 허용한다. 알 수 없는 runtime 변경은 거부한다. 이전 체크포인트를 수정하거나 새 체크포인트로 위장하지 않는다.

기존 graph cache와 모델 source identity는 유지한다. 저장된 batch, workspace, allocator 정책, optimizer, RNG, epoch, batch cursor, 부분 support를 상속한다. 실행 중인 서버는 이 패치로 자동 변경되지 않는다. 저장 중단 후 새 코드 checkout에서 재개하는 명령은 [SERVER_V222.md](../SERVER_V222.md) 상단에 있다.

검증 도구:

- `tools/profile_v222_support.py --loader process --producer-processes 4 --fixed-support-snapshot`: 실제 전체11,279 support, 전체 모델5,550,806parameters, batch32, workers8, bf16, workspace256MiB, allocator9GB, 매 배치 checkpoint. 명시적 DEBUG 진단이며 최종 학습이 아니다.
- `tools/verify_v222_runtime_debug.py --full-support-profile ...`: 완성된 전체 support 및 해당 가중치의 SHA 검증 후 실제 대형6batch를 forward/loss/backward/Adam update한다. 비교 실행과 loss/전체 가중치/Adam 일치를 검사한다.
- `tools/verify_v1_resume.py --fixed-support-snapshot`: 실제 CT DEBUG fixture에서 optimizer 및 부분 support 저장·재개 결과 일치를 검사한다.
- `tools/record_v222_snapshot_validation_debug.py`: 전체 support embedding/모델/Adam·토폴로지·배치 순서 일치와 학습·재개 증거를 확인한 뒤 결과 JSON을 작성한다.

## 측정 결과

RTX5070Ti, 실제11,279개 support/353batch 전체를 완료했다. 직전 process 경로와 같은 가중치, 입력 순서, batch32, workspace256MiB, allocator9GB, release_unused=True다. 전체 embedding과 저장된 model/Adam tensor가 bitwise 일치한다.

| 측정 항목 | 직전 process 경로 | 이번 수정 |
| --- | ---: | ---: |
| 전체 support 경과 시간 | 925.65초 | 722.25초 |
| 배치 준비 대기 합계 | 229.59초 | 11.47초 |
| checkpoint 복사·제출 합계 | 57.00초 | 4.99초 |
| GNN CUDA-event 구간 합계 | 535.87초 | 595.68초 |
| peak allocated | 3.505GB | 3.505GB |
| peak reserved | 5.090GB | 5.090GB |

전체 시간은 약22% 단축됐다. GNN 계측 구간은 오히려 길어졌으며 이를 숨기거나 모든 연산이 빨라졌다고 주장하지 않는다. 전체 대기·복사 감소가 이 증가를 상쇄했다. 원인별 독립 실험으로 GNN 구간 증가를 분해하지 않았고, 이 구간에는 host launch 간격도 포함된다.

실행 중 자원1초 표본은 주 GPU 프로세스 CPU170.7%, producer 중3개가 각각230~254%, available RAM27.94GB, GPU 사용률77%였다. CPU100%는 한 논리코어 기준이다. 순간 표본이며 평균 점유율로 해석하지 않는다. 공유 storage의 RSS는 프로세스 사이 중복 집계될 수 있다.

관련 단위 회귀36개 통과. 전체 support11,279개를 사용하는 실제 대형6query-batch(각32)의 forward/loss/backward/Adam update도 통과했다. Recipient group 안에서는 원래 학습 경로처럼 같은 support plan을 유지했다. Batch당22.24~24.37M edges이며 모델/그래프를 줄이지 않았다.

| 실제 optimizer DEBUG | 기존 thread 경로 | 수정 process 경로 |
| --- | ---: | ---: |
| 6step 시간 합계 | 79.10초 | 66.83초 |
| 초기 준비 제외 step2~6 평균 | 12.09초 | 8.83초 |
| peak allocated | 6.416GB | 6.416GB |
| peak reserved | 6.671GB | 6.671GB |

모든 학습 파라미터 gradient가 존재하고 유한함을 확인했다. Step2뿐 아니라6회 갱신 후 전체 model/Adam tensor와 각 loss가 bitwise 일치했다. 그래프 준비·프로세스 시작 비용이 포함되는 첫 배치는 수정 경로가 더 오래 걸렸다. 전체 epoch 시간을 위6개로 확정하지 않는다.

실제 CT6관측 fixture를 쓰는 별도 DEBUG에서 optimizer 중단·복구 후 loss/전체 parameter/Adam과 부분 support 재개 결과가 연속 실행과 bitwise 일치했다. Validation allocator callback 전후 metric도 같았다. 이 fixture의 작은 batch는 재개 검사 전용이며 최종 batch 설정을 변경하지 않는다.

Python/NumPy 전역 RNG는 독립 profile 시작 상태가 같지 않아 두 run의 최종 상태를 동일하다고 비교하지 않았다. Torch/CUDA RNG와 실제 출력은 동일하다. Python/NumPy는 각 저장 시점의 상태를 그대로 담는다는 별도 단위 검사로 확인했다. 이 구분 없이 모든 독립 run의 RNG까지 같다고 주장하지 않는다.

Git에 보존하는 근거: [support_snapshot_20260926_DEBUG.json](../validation/v222_r6/support_snapshot_20260926_DEBUG.json). 로컬 전체 support 경로는 `work/v222_support_snapshot_full_20260926_DEBUG`, 전체 support 학습 검사는 `work/v222_runtime_snapshot_fullsupport_20260926_DEBUG`, 재개 검사는 `work/v222_snapshot_resume_20260926_DEBUG`다. 기존 결과를 덮어쓰지 않았다.

## 남은 비용과 검증 범위

GNN의 전체 edge attention, 메시지 합산, 메모리 제한에 따른 chunk 실행 비용은 남는다. CUDA-event로 잰 GNN 구간에는 CPU의 kernel launch 대기 간격도 포함되므로 순수 GPU kernel 시간이라고 해석하지 않는다. 구조나 그래프 규모를 줄여 속도를 올리지는 않았다.

로컬 단일 실행 비교이며 반복 실험의 평균이나 A100 MIG 처리량이 아니다. 전체 support 검사는 전체40epoch 학습, 전체 validation 및 segmentation Dice 검증과 구분한다. Native online CP bank/nnU-Net 연결은 이 수정의 완료 범위가 아니다.

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
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 전체 support 기반6batch 및 재개 검증 통과.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
