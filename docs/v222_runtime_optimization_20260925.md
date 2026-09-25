# v2.22 r6 실행 중복 제거 — 2026-09-25

모델·관측·loss·split을 유지한 실행 최적화다. 보존 v1 및 `hiercp*` 연구 소스는 수정하지 않았다. 전체 5,550,806 parameters, L0 3층/L1·L2 각 2층, 128차원·4 heads, CT 48³, 128개 비교 위치, seed42, CP80%, GNN40epochs, train11,279/val2,823을 유지한다. 서버에는 아직 적용하지 않았다.

## 구현

| 확인한 비용 | 적용한 수정 |
|---|---|
| 동일한 준비용 graph를 worker별로 생성하고 버린 뒤 다시 생성 | `v222_prepare_optimized.py`: 실제 미완료 작업으로 동시성을 측정하고 각 결과를 즉시 저장. 관측마다 한 번만 실행하며 donor·pair 전체 개수 검증 |
| shared donor 읽기·압축 해제 중 전역 lock 점유 | donor별 single-flight. 서로 다른 파일은 lock 밖에서 병렬 로딩 |
| 같은 epoch의 graph view 반복 생성 | 파일 경로·SHA·stat·epoch 기준 RAM LRU. tensor/NumPy/dataclass 실제 저장공간을 세고 alias는 중복 집계하지 않음. cgroup 가용 RAM 반영 |
| 같은 donor의 다른 CPU 사본을 별도 CNN 입력으로 취급 | 검증된 source content SHA로 batch 내 donor map 공유. recipient graph·target CT는 모두 유지 |
| support prefix를 GPU에서 clone한 뒤 CPU checkpoint로 복사 | prefix view에서 직접 불변 CPU snapshot. 다음 batch가 저장 내용에 영향을 주지 않도록 복사 완료 후 진행 |
| 매 batch의 checkpoint 직렬화·fsync 대기 | CPU snapshot 뒤 writer 1개가 비동기 원자 저장. pending snapshot 최대 1개, 다음 제출 전 이전 writer 오류/완료 확인. STOP·종료 때 flush |
| 작은 edge workspace의 반복 chunk 연산 | 새 실행은 실측한 256MiB. 기존 checkpoint 재개는 원래 workspace·allocator 정책 상속 |

학습된 embedding을 가중치 변경 이후까지 재사용하지 않는다. 초기/매 epoch/최종 support 재인코딩은 모델 의미를 유지하기 위해 그대로 수행한다. epoch별 sampling seed도 유지한다. 모든 가능한 중복 비용이 사라졌다고 주장하지 않는다.

`run_v222_server.py` 기본 경로를 최적화 backend에 연결했다. `--cache`는 기존 완성 index를 직접 사용하며, `--cache ... --resume ...`는 raw 준비·graph 생성·DEBUG profile을 반복하지 않는다. 새 출력 경로만 허용한다. 학습 batch는 기존 실제 calibration으로 결정하며 DEBUG batch32를 production에 강제하지 않는다.

## 실제 측정

RTX5070Ti, CPU16 logical cores/RAM64GiB, bf16, 전체 모델, physical32/effective32, loader8, 큰 실제 CT graph 6batch. batch당 약323k–354k nodes, 22.2M–24.4M edges. PyTorch allocator 상한9GB. support는 실제 저장된 1,216개 prefix이며 전체11,279개는 아니다.

| 경로 | 6 step 합계 | step2–6 평균 | checkpoint 제출 평균 | 같은 32쌍 warm reload | 최대 allocated / reserved |
|---|---:|---:|---:|---:|---:|
| 기존 64MiB | 158.91s | 25.57s | 0.328s | 3.376s | 6.403 / 6.717GB |
| 중복 제거 64MiB | 161.95s | 26.13s | 0.122s | 0.307s | 6.401 / 8.166GB |
| 중복 제거 256MiB | 64.86s | 9.98s | 0.128s | 0.277s | 6.403 / 8.009GB |

64MiB 조건에서는 전체 step 속도 개선이 관측되지 않았다. 캐시 재로딩 약11배, 저장 제출 대기는 감소했으나 주비용은 GPU 연산이었다. workspace 변경까지 포함한 새 실행의 6step 합계는 약2.45배 개선됐다. 비동기 제출 시간은 fsync 완료 시간이 아니며 durable writer 시간은 별도로 기록한다.

각 조건 1회 측정이고 초반 warmup·파일 캐시 영향이 있다. 최종 256MiB batch 무렵 독립 CPU 준비 검증이 시작됐다. 엄격한 반복 성능 실험이나 A100 MIG 속도 측정이 아니다. CUDA context 등 allocator 밖 메모리도 9GB 상한에 포함되지 않는다. 전체 epoch 시간·40epoch 완료·분할 정확도를 이 결과로 추정하지 않는다.

64MiB 기존/수정 경로의 2step loss·전체 가중치·optimizer 상태는 bitwise 일치했다. 256MiB는 gradient 누적 순서가 달라질 수 있어 bitwise 동일을 주장하지 않는다. 따라서 기존 64MiB checkpoint를 256MiB로 조용히 바꾸지 않는다. 기존 run의 재개에서는 위 2.45배 속도를 보장하지 않는다.

## 검증과 기록

- 캐시 병렬성·예산·무결성, writer 오류/STOP, 재개 정책, server 단계, tqdm, 자원 스케줄러 등 단위/회귀47개 통과.
- 실제 CT의 optimizer 중단·재개: loss·전체 parameters·optimizer·부분 support memory bitwise 동일, 모든 parameter gradient 존재·유한.
- 실제 CT 6관측의 별도 DEBUG: pause→resume→epoch support refresh→validation→best checkpoint 완료. train/validation fixture 재사용은 제어 흐름 검사이며 의료 성능 결과가 아니다.
- 실제 원본 CT에서 새 준비 함수로 양성/비교 관측 2개 생성: 기존 cache의 CT·canonical nodes·edges·bounds 모두 동일. 전체14,102개 재생성 실행은 하지 않았다.
- 준비·학습 source provenance와 기존 전체 cache 호환성 확인. raw/cache payload를 Git에 넣지 않는다.

원시 로그: `work/v222_runtime_20260925_DEBUG/`. Git 보존 증거: `validation/v222_r6/runtime_optimization_20260925_DEBUG.json`. 최초 legacy 실행은 첫 step 기록 없이 종료됐고 원인 로그가 없었다. 성공 결과는 새 `legacy64_r2` 실행이며 최초 실패를 성공으로 세지 않았다.

정확한 재현 도구는 `verify_v222_runtime_debug.py`, `verify_v222_prepare_debug.py`, `verify_v1_resume.py --optimized`, `smoke_v1_execution_lifecycle.py --optimized`다. 모든 축소 검사는 별도 DEBUG 출력이며 최종 설정에 반영되지 않는다.

## 저장·시간 해석

- `checkpoint_status.json`: 디스크에 원자적으로 저장 완료된 checkpoint. `checkpoint_durability.jsonl`에 모든 commit 이력.
- `runtime_events.jsonl`: committed step과 pending 여부를 구분한다. epoch 시간은 optimization + support refresh + validation의 활성 실행 시간이며 pause 대기와 초기 support는 별도다.
- 보존 실행기의 `step_timings.jsonl` 및 `first_optimizer_step.json`에서 `checkpoint_step`은 **제출한 step**이다. `epoch_NNN.json`/`metrics.csv`의 `seconds`는 여전히 optimization 누적이다. 이를 durable step이나 전체 epoch 시간으로 읽지 않는다. 새 run의 `runtime_log_schema.json`에도 명시한다.
- 강제 프로세스 종료 시 아직 pending인 마지막 저장은 소실될 수 있다. 이전 원자 checkpoint는 유지한다. 정상 STOP/종료는 writer 완료 후 반환한다.
- rolling checkpoint에 실행 backend SHA를 넣는다. 모델/cache provenance는 그대로 보존한다. 변경된 backend checkpoint의 임의 재개는 거부하며, workspace는 checkpoint 내부 기록과 sibling receipt 일치 여부를 확인한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번32, 앞선 동일 모델16/32/64 메모리 검사와 구분.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번6batch OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 전체 학습/평가 및 서버 적용은 미실행.
