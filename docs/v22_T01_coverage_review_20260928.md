# v2.2 T01 — 통합 검사의 누락 검출과 검토 범위

검토 기준은 `a2c98b6d0886611c81e760672e3f09375298e75c`이며, 사용자가 전달한 독립 검토문 전체를 읽고 활성 코드와 대조했다. 보고서에 링크된 `sandbox:/mnt/data`의 별도 ZIP/MD는 이번 로컬 첨부 파일에 포함되지 않았으므로 그 파일들까지 읽었다고 주장하지 않는다.

## 놓쳤던 조건과 수정

이전 DEBUG 검사 두 곳은 `all(... for row ... if batch == 1)`만 검사했다. 원래 Git 소스의 두 `if` AST를 그대로 실행해 빈 목록과 warmup만 있는 목록이 모두 통과함을 재현했다. 실제 이전 GPU 기록에는 세 update의 양수 대조 기록이 있었으므로 그 실행 결과가 가짜였다는 뜻은 아니다. 하지만 이 음성 검사를 빠뜨린 채 검증 완료라고 보고한 것은 불충분했다.

이번 변경은 `tools/v22_integrated_smoke_debug.py`의 DEBUG 증거 판정과 관측 식별자에 한정한다. production GNN, loss, graph, catalog, cache, segmentation trainer는 바꾸지 않는다.

1. 실제 loader의 batch 생성 직전에 `stage / role / epoch / batch`를 기록한다. `comparison_batch` 증가 전의 번호가 반환되는 실제 batch에 붙는다. batch1을 하드코딩해 학습이라고 추정하지 않는다.
2. 실제 transform 호출 순서로 sample ordinal을 센다. CP 여부와 관계없이 모든 sample을 세고, paste 전 원본과 동일 난수로 대조한 CP sample만 완료 기록을 남긴다. transform 수와 실제 batch 크기가 다르거나 미처리 paste가 남으면 실패한다.
3. 실제 optimizer 입력의 `online_cp_applied` 배열에서 필요한 CP sample 목록을 별도로 만든다. 대조 기록으로 기대 목록을 만들어 자기 자신과 비교하지 않는다.
4. `initial / continuous / resumed`를 분리한다. 특히 continuous와 resumed의 epoch/batch가 같아도 서로 대신할 수 없다. warmup 기록은 update coverage에 포함하지 않는다.
5. `checked_train_step`은 기대 목록과 관측 목록의 정확한 일치, 중복 없음, CT/추가 종양 target 차이 양수를 먼저 확인한 뒤 실제 production `trainer.train_step`을 호출한다. 정상 반환한 update stage도 별도로 기록한다.
6. 세 update가 끝난 뒤 전체 coverage와 완료 stage를 다시 검사해야 `CP_survives_training_augmentation=True`를 반환한다.

대조용 영상은 학습 입력을 대체하지 않는다. 기존 observer의 Python/NumPy/Torch CPU RNG 복구와 실제 표준 증강은 유지한다.

## 검사 범위

새 회귀 모듈 `tests/test_v22_integrated_coverage_debug.py`는 18개 test method다. 빈 기록, warmup만 존재, 각 stage 누락, 중복, 잘못된 sample/batch/epoch, continuous로 resumed 대체, CT 또는 target 차이 0/음수/NaN, 기대 stage 누락, 실제 CP flag와 불일치, 정상 입력을 검사한다. 하위 조합은 별도 검사 수에 더하지 않는다.

실제 호출 경계도 검사한다. observer가 누락되면 production update를 호출하지 않고, 세 정상 호출만 완료 stage로 기록하며 네 번째 호출을 거부한다. 이 단위검사의 Mock은 호출 여부 검사용이며 실제 학습 성공의 근거는 별도의 CUDA 통합 실행이다.

이번 회귀 실행은 기존 71개와 새 18개를 합한 **89개**다. 과거 실행 수나 독립 GPT의 검사 수는 합산하지 않는다. 원래 모델과 실제 CT를 쓰는 GPU 통합 재실행의 수치·source/log/checkpoint hash는 `validation/v222_r6/T01_coverage_20260928_DEBUG.json`에 기록한다.

## 실제 재실행 결과

판정은 **PASS_LOCAL_SHORT_T01_VALIDATION**이다. 구현 수정, 구문 검사, 회귀89개, 실제 CT/CUDA 통합 smoke를 완료했다. 전체 학습·전체 평가·서버 학습은 실행하지 않았다.

| 검사 | 결과 |
|---|---|
| 기대 CP sample과 실제 대조 | initial `(0,1,0)`, continuous `(1,1,0)`, resumed `(1,1,0)`의 epoch/batch/sample이 정확히 일치 |
| 제외한 warmup | 3개, update 증거에 미포함 |
| 세 update의 CP 영향 | 각각 CT 차이25,600 voxel / 추가 종양 target4,124 voxel |
| 실제 GPU 결과를 변조한 음성 검사 | 빈 목록/누락/중복/오표기/차이0 등9조건 모두 거부; 89개 unittest와 별도 구분 |
| 연속/재개 다음 update | 입력·RNG·상태 hash 동일, model/SGD/gradient 최대 차이0 |
| 다음 loss | 양쪽 `1.1939997673034668` |
| 수정 전 성공 실행과 대조 | 세 batch 입력·loss·최종 상태 hash 동일 |
| Raw→native oracle | CT 최대 차이0, segmentation 동일 |
| 추천 | index57 / raw rank1 / eligible116 / 후보128 |
| 모델/입력 | GNN5,550,806 + native102,350,575 parameters, native batch2×128³ |
| 자원 | RTX5070Ti, GPU peak allocated6,597,423,616 bytes(6.14GiB), sampled peak RSS8,164,003,840 bytes(7.60GiB) |
| 측정 시간 | 232.12초; 모델 초기화와 사전 warm SGD update는 측정 구간 밖이며 전체 프로그램 시간·event 처리량이 아님 |
| 새 산출물 | 약2.26GiB; 로컬 한 case 검증용 결과/체크포인트. 전체 graph cache 아님 |
| 보존 확인 | core83/runtime20/online15 SHA256 일치 |

이전 `exports/v2.2_local_final_a2c98b6_20260928`은 당시 결과로 보존한다. 최신 로컬 짧은 검사 판정은 이번 T01 수정본과 JSON에 한정한다. 저장 시점의 소스·로그·결과·체크포인트 hash를 함께 남겼다.

## 독립 보고서의 다른 항목과 대조

| 항목 | 실제 코드와 대조한 판단 |
|---|---|
| S01 설치 부모의 종료 분기 | 로컬 캡처 함수는 best/logger 처리와 epoch 증가 뒤 정상 반환한다. `OnlineCheckpointMixin`은 부모가 반환한 뒤 저장한다. 서버 설치본·간접 종료·OS 신호 안전성을 이 결과로 증명하지 않는다. 임의의 SystemExit 패치는 추가하지 않는다. |
| 실제 CP 학습 연결 | 기존 actual driver와 native loader에서 실제 owner/RPC/paste/transform/train_step을 사용한다. 이번에는 그 optimizer 입력별 대조 기록의 존재도 강제한다. |
| Production 초기화 우회 | DEBUG support8와 기존 native fixture 때문에 owner/bank/trainer admission 일부를 우회한다. 점수·loss 대체는 아니지만 production admission 통과로도 볼 수 없다. |
| 재개 범위 | 같은 trainer에 checkpoint를 로드하고 loader/transform을 재생성한다. owner/RPC/receipt는 유지된다. 새 프로세스의 runner resume, cold/warm receipt, worker queue를 검사한 것이 아니다. |
| Support / worker / compile | support8, augmentation worker0, compile off인 명시적 DEBUG다. native 모델과 batch2×128³는 유지한다. full-support peak VRAM의 상한으로 해석하지 않는다. |
| CP 확률 | metadata .8 유지. DEBUG draw .1/.9로 CP와 no-CP 두 분기를 보장한다. 실제 production 확률 schedule 또는 성공률 검증은 아니다. |
| 관측 GT와 추천 | `rank_then_filter`는 score를 그대로 정렬한 뒤 전체 paste footprint의 겹침/간 포함률/경계를 검사한다. 1위를 무조건 버리지 않는다. |
| 실제 학습 경로 | `v22_ranking_steps.optimizer_step`의 loss→backward→clip→AdamW와 native CUDA train_step 연결을 유지한다. |
| Support 제외 | `v1_local.support_for_recipient`가 query group을 recipient와 donor 양쪽에서 제외한다. case ID가 실제 환자 독립성을 보장한다는 뜻은 아니다. |

## 해결했다고 볼 수 없는 연구 질문

- `rank_loss`는 live query에 닿는 P/U pair를 해당 update의 pair 수로 나눈다. 미니배치 분할에 따른 pair 비균등 가중 위험은 남는다. 이를 단순 loss/2로 바꾸면 L0 gradient까지 바뀌므로 이번 DEBUG 수정에 섞지 않는다.
- `RankingContext.reference`의 epoch embedding은 detached이며 live query 위치만 `net.local(query)`로 교체한다. 전체 CT end-to-end backpropagation이라고 부르지 않는다.
- 관측 종양 감지 shortcut, 고정 donor 의존 여부, CE/rank gradient의 관계는 성능과 ablation으로 확인할 연구 질문이다. 저장·재개 일치로 해결됐다고 할 수 없다.
- CT 특징을 학습하더라도 GT 간 union으로 만든 geometry를 사용한다. annotation-free가 아니다.
- Recall/MRR은 제외 전 관측 종양 순위다. 실제 CP 효용은 no-CP / valid-random-CP / proposed-CP 비교와 segmentation 평가가 필요하다.

**로컬 짧은 통합 검사와 전체 production 검증을 구분한다.** 전체 support G3, production admission/worker/compile/새 프로세스 재개 G4, 비교 성능 G5는 이번 실행에 포함하지 않는다. 서버 학습을 시작하지 않는다. 이전 결과는 보존하며 이 문서는 T01을 반영한 후속 기록이다.

## 재현 명령

```powershell
.\.venv\Scripts\python.exe -u tools/verify_v22_online_actual_debug.py --checkpoint work/v22_model_integrity_20260927_DEBUG/uninterrupted/checkpoint.pt --native work/local_v21_5070ti_20260919/native/native.json --output work/v22_integrated_T01_20260928_DEBUG --integrated-segmentation
```

기존 결과 폴더를 덮어쓰지 않으므로 재실행 때는 새로운 output 경로를 쓴다. 전체 graph cache를 생성하는 명령이 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 batch2와 graph worker 측정을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. CUDA/자원 telemetry를 JSON에 기록한다.
- [x] OOM 회피를 위한 모델 축소를 하지 않았다. 이번 T01은 메모리 결함이 아니다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
