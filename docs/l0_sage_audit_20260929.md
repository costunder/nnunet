# GraphSAGE 관계·refresh·서버 시간 범위 감사

2026-09-29. GraphSAGE를 다음 비용 개선 후보로 유지한다. EZ-SP adapter와 결과는 보존하고 현재 주력 학습 경로에 추가하지 않는다. 이번 작업은 **기존 SAGE 구현을 바꾸지 않은 감사와 짧은 GPU 계측**이다. Production 전환, exact resume, 전체 학습·평가 승인이 아니다.

## 1. 관계와 자기 노드 경로

실제 `l0_sage/encoder.py`, `hiercp/model.py`, 설치된 PyG 2.6.1 `SAGEConv`/`spmm` 소스를 대조했다. 13개 관계가 각자 다른 파라미터와 degree를 갖는다. 모든 관계의 이웃을 한 번에 평균내지 않는다.

관계 r의 출력은 `W_neighbor,r × mean_r(x_source) + b_r + W_root,r × x_target`이다. 대상 역할별로 이 출력을 합친 뒤 기존 block의 `LayerNorm(x_target + dropout(message))`, FFN과 두 번째 residual을 적용한다.

| 대상 역할 | 들어오는 관계 수 / 독립 root 변환 수 | message 앞 기존 identity residual |
|---|---:|---:|
| tumor_surface | 2 | 1 |
| source_context | 3 | 1 |
| source_liver_surface | 2 | 1 |
| target_context | 4 | 1 |
| target_liver_surface | 2 | 1 |

**자기 정보가 관계별 변환과 residual 양쪽으로 들어가는 것은 사실이다.** 같은 가중치를 실수로 4번 호출하는 구조는 아니며 13개의 독립 root 행렬이다. 이웃이 없는 대상도 SAGE root 변환을 받는다. 이는 표준 관계별 SAGE를 sum 결합한 결과지만, 종전 GAT의 target-dependent attention 및 `add_self_loops=False`와 같은 수식은 아니다. 이 구조가 추천 정확도에 적합하다는 뜻도 아니다. `root_weight=False`로 바꾸거나 관계 수로 나누면 또 다른 모델이므로 이번 감사에서 조용히 변경하지 않았다.

새 검사는 **실제 block을 명시적 관계별 mean + 각 root + residual + LayerNorm/FFN 수식과 대조**한다. 5개 역할의 출력, 입력 gradient 및 13개 root weight gradient가 일치했다. 기존 검사는 raw COO PyG mean과 normalized CSR의 출력·전체 gradient, 중복 edge, 고립 node, 빈 edge를 대조한다.

## 2. 희소 경로는 이미 사용 중

관계별 `D^-1 A`를 native Torch CSR로 만든 후 PyG `message_and_aggregate → spmm → torch.sparse.mm`을 사용한다. `aggr='sum'`은 이미 정규화한 값을 합하기 위한 설정이다. 두 번 degree로 나누지 않는다. 행은 target, 열은 source이며, 중복 edge의 값은 합쳐 원래 multiplicity를 보존한다.

현재 배치에 대해 13개 CSR을 한 번 만들고 3층과 activation checkpoint 역전파에서 재사용한다. edge identity/version 또는 node 수가 바뀌면 재생성한다. 새로운 refresh 배치는 새로운 tensor이므로 새 CSR을 만든다. 한 배치의 cache hit를 전체 support pass의 cache hit라고 해석하면 안 된다. 학습 중 변하는 node 표현을 캐시하는 구조도 아니다.

CSR을 다시 적용하는 변경, integrity 검사 제거, root 변경, 자동 cap·skip·fallback은 하지 않았다.

## 3. 실제 8pair refresh 비용 분해

도구: `tools/profile_sage_refresh_debug.py`. 실제 DEBUG inner_train **8개 전체, 52,666 nodes / 2,761,602 edges**, disjoint-union physical8, FP32, CNN chunk4, 같은 초기 seed42 local 가중치. 학습 checkpoint는 사용하지 않았다. RTX5070Ti16GB, CPU workers8, CUDA allocator6GiB/RSS12GiB/180초의 명시적 한도. 모델 복제, 입력 materialization/H2D는 refresh timer 밖이다. 가중치는 forward에서 변하지 않았고 cold/hot 출력 일치도 확인했다.

각 arm 워밍업1회 후 교대 순서로 3회 측정. Cold는 새 encoder의 첫 batch라 CSR이 없다는 뜻이며, 전역 CUDA/CNN 커널까지 완전히 처음이라는 뜻이 아니다. Hot은 **동일 모델·동일 8pair tensor** 재실행이다. `eval/no_grad`만 수행하고 update·loss·optimizer·checkpoint는 실행하지 않았다.

1차 분리 결과: [refresh_subphases.json](../validation/l0_sage_audit_20260929/refresh_subphases.json).

- GAT cold 0.241초, SAGE cold 0.430초.
- SAGE 입력 무결성 검사 0.018초, CSR 생성 0.019초, 나머지 encoder 0.393초.
- 같은 batch hot에서 CSR 준비는 0.000162초, SAGE 전체 0.272초였다.

남은 encoder 비용을 분리한 별도 진단: [refresh_modules.json](../validation/l0_sage_audit_20260929/refresh_modules.json).

| 동기화된 진단 구간, 3회 평균 | GAT cold | SAGE cold | SAGE hot |
|---|---:|---:|---:|
| 전체 refresh | 0.213초 | 0.389초 | 0.368초 |
| 추가 integrity 검사 | 별도 wrapper 없음 | 0.0176초 | 0.0183초 |
| CSR 준비 | 해당 없음 | 0.0186초 | 0.000245초 |
| CNN 두 입력 인코딩 | 0.0101초 | 0.0101초 | 0.0087초 |
| 그래프 block 0 | 0.0652초 | 0.0812초 | 0.0750초 |
| 그래프 block 1 | 0.0608초 | 0.0705초 | 0.0634초 |
| 그래프 block 2 | 0.0545초 | 0.1700초 | 0.1806초 |
| sampling·project·readout 및 계측 잔여 | 0.0224초 | 0.0203초 | 0.0218초 |

SAGE의 그래프 block 합계는 0.322초, GAT는 0.181초다. **이번 no-grad refresh에서는 CSR 생성보다 그래프 블록 실행 비용이 컸다.** CNN이나 CSR 생성만을 원인으로 지목할 근거는 없다. 세 번째 블록의 비용이 특히 컸지만, 개별 커널/수치 분포/allocator 등 어느 요인이 이를 만들었는지까지 이 module 계측으로 확정하지 않는다.

각 구간에 CUDA synchronize를 추가한 진단이므로 기존의 비계측 update/refresh 평균과 동일한 benchmark가 아니다. 두 진단 간 편차도 남겼다. Hot에서 줄어드는 나머지 시간 전부를 CSR 재사용 효과로 계산하면 틀린다. 실제 production refresh는 여러 다른 batch를 지나므로 이 hot 결과를 11,279개에 곱하지 않는다.

이전 update 2.283→1.203초 결과는 별도로 보존한다. 이번 refresh-only 결과를 새 update 개선율로 보고하거나 EZ-SP의 별도 2.649초 baseline과 섞지 않는다.

## 4. 서버 epoch와 연결한 작업량

로컬에 남아 있는 이전 전체 cache metadata `work/v222_v1_recovered2_training_20260924/cache/index.json`을 읽었다. graph tensor는 만들거나 다시 읽지 않았다. **이전 metadata의 개수 확인이며 현재 서버 cache의 내용·provenance 검증은 아니다.**

| 단계 | 작업량, physical32 기준 |
|---|---|
| optimization | inner_train 11,279개, 84 recipient groups → 405 update |
| support refresh | 같은 11,279개 → memory physical32일 때 353 batch |
| validation | inner_val 2,823개, 21 groups → query physical32일 때 100 batch |

실제 optimization batching은 recipient별 edge 크기 정렬 후 분할하며, 마지막 partial batch를 버리지 않는다. 따라서 `ceil(11279/32)=353`을 update 개수로 쓰면 틀린다. 405×40=16,200은 사용자가 붙여 준 서버 진행표의 total step과도 맞는다. Memory batch는 별도 calibration 값이므로 353은 **memory batch도32인 경우**다.

사용자가 제공한 서버 step29–78 평균 **22.104초**가 405개 update 전체를 대표한다는 조건을 두면, optimization만 **8,952.12초 = 149.20분**이다. 이것은 조건부 환산이며 실제 완주 epoch 측정값은 아니다. 여기에 refresh와 validation 및 경계 비용이 필요하다. 현재 이 감사에는 서버 raw JSONL이 없으므로 그 시간과 SAGE 서버 epoch 예상은 **null/미확인**으로 남겼다. 과거 다른 실행의 4.35item/s를 합치지 않았다.

로컬 1.203초는 production과 다음 조건이 다르다.

- RTX5070Ti/torch2.8/PyG2.6.1/FP32 대 A100 MIG1g.10gb/torch2.6/PyG2.7/BF16.
- Query physical8/CNN chunk4 대 production query32/CNN chunk32.
- DEBUG8개 detached support와 4개 recipient task 대 전체11,279개 memory에서 제외 규칙을 적용하는 production task.
- 새 Adam의 첫 update 대 이어서 학습 중인 Adam.
- 로컬 update에는 loader/H2D, support refresh·plan 준비, 실제 checkpoint 쓰기 완료, validation이 포함되지 않는다.

따라서 1.90배를 서버149분 또는3시간에 곱하지 않는다. [server_scope.json](../validation/l0_sage_audit_20260929/server_scope.json)에 근거 출처·가정·미확인 값을 분리했다.

## 5. 로그만 읽는 비용 집계 경로

`tools/summarize_v22_cost_readonly.py`는 torch도 import하지 않고 metadata, `training_started.json`, `step_timings.jsonl`, `runtime_events.jsonl`만 읽는다. 모델·cache 생성이나 학습·checkpoint 쓰기가 없다. stdout에 JSON을 출력한다.

- 명시한 query/memory batch가 저장된 실행 계약과 다르면 오류.
- 여러 재개 run을 자동 합쳐 step을 중복 계산하지 않는다. 한 run씩 읽으며 중복 step/epoch 완료·손상 JSON은 오류.
- `epoch_NNN.json`의 `seconds`는 전체 epoch로 취급하지 않는다.
- 전체 epoch는 `runtime_events`의 `epoch_complete.runtime_phase_seconds`가 있는 경우만 집계한다. 이것도 runtime이 보고한 active phase 시간이며 initial/final memory·중단 시간·현재 비동기 저장의 최종 durable 완료 전부를 보장하는 wall clock은 아니다.
- Loader/save는 이미 phase 구간 안에 들어 있으므로 별도로 다시 더하지 않는다. `checkpoint_seconds` 역시 현재 파일 쓰기 완료 전체와 같지 않다.
- 부분 로그밖에 없으면 전체 epoch를 빈 값으로 둔다. 부분 값이나0으로 채워서 통과시키지 않는다.

기존 로컬 DEBUG 완주 로그로 CLI를 실행했다. 4update 합계5.523초와 runtime phase 합계9.297초가 분리되어 기록된다. **로컬 로그 집계 검사 결과이며 서버 수치가 아니다.** [local_log_reader_check.json](../validation/l0_sage_audit_20260929/local_log_reader_check.json).

새 도구가 들어 있는 checkout에서 실행하는 형식:

```text
python tools/summarize_v22_cost_readonly.py --cache <paired_cache/index.json> --run <training directory> --physical-batch <saved query batch> --memory-batch <saved memory batch> --epoch 1
```

## 6. 검증과 남은 경계

기존19개 + 관계/root 수식 검사1개 + 로그/스케줄 검사3개 = **23개 통과, 실패0, skip0**. 정적 AST 검사, 실제 CUDA refresh 진단2종, metadata CLI 검사를 완료했다. 원본 core identity와 SAGE encoder hash가 이전 검증본과 같고, EZ-SP 보존 ZIP23개 파일의 SHA/CRC도 다시 확인했다. 기존 측정·검사 파일을 덮어쓰지 않았다.

Basic CP·L1/L2·loss·후보128·원본 mask·production batch와 observation·깊이/너비는 변경하지 않았다. 장기 학습·전체 평가 미실행. SAGE checkpoint export는 계속 금지다. BF16·production wrapper 속성 전달·실제 batch32/full support·MIG의 동일 조건 시간과 CP 효용은 검증되지 않았다. GAT checkpoint exact resume나 production-ready로 표시하지 않는다.

남은 속도 확인 대상은 CSR 재도입이 아니라 **실제 SAGE 그래프 연산의 refresh 비용과 production 조건에서의 update/validation 비중**이다. 이번에 확인한 backward 이득을 인정하되, 3시간 문제가 해결됐다고 선언하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 기존 명시적 DEBUG8개 전체 사용.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 동일 batch8, 전체 metadata batch32 스케줄 별도 집계.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 수식/로그 합성 검사와 실제 CT 측정 구분.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 연결 회귀검사 유지.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
