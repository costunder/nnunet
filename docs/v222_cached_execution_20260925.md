# 캐시 이후 실제 L0 실행 병목 측정 — 2026-09-25

사용자는 v1도 캐시를 사용하는데 현재 실행이 왜 느린지 물었다. 캐시 생성량과 support 재계산만으로 원인을 설명한 이전 답변은 실측을 뒷받침하지 못했다. 이번에는 기존 실제 CT 캐시를 읽은 뒤 GPU 연산을 구간별로 측정했다. 서버 프로세스와 캐시는 변경하지 않았다.

## 실측

RTX5070Ti, PyTorch allocator9GB 상한, 전체5,550,806 parameters, 실제 physical/effective batch32, accumulation1, CPU workers8, bf16. train11,279개 중 가장 큰 두 recipient 그룹의 실제32개씩을 사용한 명시적 DEBUG다. support는 저장된 실제 initial-memory prefix1,216개로 전체 support 실험이 아니다. 각 프로세스에서 같은 초기 가중치/RNG/입력으로 두 번 forward/backward하고 마지막에 한 번 optimizer update했다. 아래는 profiler를 켜지 않은 두 번째 pass이며 입력 로딩과 체크포인트 저장은 별도다.

| 실제 입력 | 노드 | 엣지 | 작업 메모리64MiB | 256MiB | 개선 |
|---|---:|---:|---:|---:|---:|
| liver_39 32개 | 346,135 | 23,857,190 | 22.152초 | 6.710초 | 3.30배 |
| liver_117 32개 | 343,880 | 24,373,716 | 21.905초 | 6.411초 | 3.42배 |

liver_39의 L0는6.107→1.951초, 역전파15.857→4.533초였다. L1/L2는0.023→0.045초였다. 이 support prefix 조건에서 L1/L2가 주병목이라는 증거는 없다. 전체 support일 때의 비용은 이 수치로 확정하지 않는다. 로딩3.8~4.0초, 저장0.27~0.44초. peak allocated는두 입력 모두6.30~6.33GB, reserved는6.54~6.59GB였고9GB 상한에서 OOM은 없었다. 512MiB 추가 측정은첫 입력6.452초로256MiB 대비추가이득이 작아 우선256MiB를 제공한다.

원인은 보존된 `hiercp/model.py`의64MiB 고정 작업 메모리가 hidden128에서16,384-edge 조각을 만들고, 전방/체크포인트 재계산/역전파에서 이를 반복하는 실행 비용이다. 새256MiB 선택은65,536개씩 처리한다. 모든 노드·엣지·softmax 정규화 범위·레이어·loss·물리 batch를 유지하며, graph를 줄이거나 캐시를 재생성하지 않는다. 단일 CUDA 커널별 비중을 확정한 것은 아니며, 같은 전체 입력의 작업 메모리 A/B 실험으로 실행 비용 차이를 확인했다.

두 입력에서 loss 값은 각기 동일했고 모든544 parameter tensor의 gradient가 존재하고 유한했다. 하지만 gradient는 비트 동일하지 않다. 누적 순서 변경으로 parameter별 최대 상대L2차이는0.004761/0.003921, 최대절대차이는0.000977/0.000488이다. 이를 동일 학습 궤적이나 임상 성능 동등성으로 주장하지 않는다. 전체epoch/A100 MIG 처리량/최종성능은 이번 검증 범위 밖이다.

첫 상세 profiler 실행은 GPU 연산 후 이벤트 통계 집계에서 RAM18.6GB까지 증가했다. 해당 작업에서 만든PID24896의 명령·생성시각을 재확인하고 사용자에게 보고 후 그 프로세스만 종료했다. 그 측정은 속도 비교에서 제외하고, profiler 없는 CUDA-event 실험으로 다시 측정했다. 서버/셸/다른 사용자 프로세스에는 신호를 보내지 않았다.

## 구현 및 실행 범위

- `tools/profile_v222_cached_step_debug.py`: 기존 실제 캐시/초기support 체크포인트의 provenance를 확인하고 동일 입력·loss·gradient·CUDA 구간 시간을 비교한다. DEBUG 출력만 새 경로에 저장한다.
- `tools/v222_gpu_workspace.py`: 현재 프로세스의 실행 상수만 선택한다. 보존된 모델 소스, config, cache provenance는 바꾸지 않는다. 각 출력 옆 `*.workspace.json`에 정책을 기록한다. resume 시 이전 정책과 다르면 거절한다. 기존64MiB checkpoint를256MiB의 정확한 재개로 조용히 취급하지 않는다.
- 서버 실행기의 `--edge-workspace-mib 256`은 graph DEBUG/profile/train에만 연결된다. observations/paired cache와 모델/학습 규모는 그대로다. 기존 실행과 재개를 보호하기 위해 기본값은64MiB로 남겼다. 실행 중인 서버에 적용했다고 주장하지 않는다.
- 이미 존재하는 cache의 GPU 단계만 실행할 수도 있다. `python tools/v222_gpu_workspace.py --workspace-mib 256 run_v222_v1_l0.py train --cache <기존완료cache/index.json> --output <새학습경로> --release-unused` 형식이다. 이는 새 학습을 시작하므로 현재 서버 작업과 중복 실행하면 안 된다. 이 조사 중 서버에서 실행하지 않았다.

과거 원문에는 full GNN의 graph cache235개와40epoch 완료, paired 실험의cache187개 재사용 기록이 있다. 옛 cache entry는 여러 후보를 묶으므로 현재14,102개 pair record와 숫자를 나눈 값을 속도 배수로 사용하지 않는다. 동일 조건의 v1 epoch 시간은 복구한 원문에서 확인하지 못했다. 본 실측은 현재 구현의 비용을 검증한 것이며 과거 v1 대비3.4배라는 의미가 아니다.

근거: `validation/v222_r6/cached_step_workspace_20260925_DEBUG.json`, `work/v222_cached_profile_20260925/`. 실행 정책/재개 보호/서버 실행·viewer 회귀검사14개 통과. 새 wrapper를 통해 실제 CT3case/6개 전체 크기 graph를 생성하고 전체 모델의 유한 gradient, L0/L1/L2 각 모듈 update, donor·recipient CT 변화에 대한 점수 민감도까지 통과했다(`work/v222_workspace_verify_20260925_DEBUG/result.json`). 이는별도 DEBUG이며 전체 학습과 segmentation 평가는 미실행.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 측정은 별도 DEBUG이며 production 계약 유지.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존측정 batch32와workers8 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 및 병목 원인을 먼저 조사했다. 이번 비교에서 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제batch검증.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
