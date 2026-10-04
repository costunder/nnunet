# v1.4 — v1.0 기반 10mm 범위 축소 순위 학습

실험 버전은 **v1.4**, 재사용한 원본 모델·소스 버전은 **v1.0**이다. 기존 실험 계획의 v1.1(activation 저장), v1.2(handcrafted 제거), v1.3(SAGE 교체)와 번호가 겹치지 않도록 구분했다. **v1.4는 v1.3에서 누적 변경한 모델이 아니라 v1.0에서 범위만 바꾼 분기**다. v2.2의 L1/L2·관측 P/U 학습으로 전환한 상태가 아니다.

`tools/run_v14_scope_training.py`는 terminal 시작·종료에 v1.4를 표시하고, 실제 학습 계약 hash에 연결한 `experiment_version.json`을 별도로 기록한다. 이미 만들어진 실험의 source/helper·manifest·checkpoint byte와 `results/v1.0` 내부 경로는 유지한다. 이 경로의 v1.0은 원본 모델 stage를 뜻한다. 표시 변경 때문에 기존 8c7e914 실험을 다시 초기화하거나 checkpoint를 다른 모델로 바꾸지 않는다.

30mm와 기존 native 결과는 보존한다. 이번 서버 명령은 **10mm arm 하나만** 준비하고 학습한다. native/nested416 두 arm이나 10/20/30mm를 자동 실행하지 않는다.

질문은 “v1 L0의 공간 범위를 줄여도 원래 source-anchor 순위 학습이 유지되는가?”다. v2.2의 observed P/unobserved U 정답을 바꾸거나 CP 적합성 정답을 새로 만드는 실험이 아니다.

## 실제 로컬 결과

아래 결과는 이전 2-update smoke다. 이후 **native 원시 기록과 view·loss·optimizer 조건을 맞춰10mm16회 실제 CUDA update를 완료**했다. 최신 결과와 AMP overflow 진단은 [원본 조건 대조 보고서](v14_matched_learning_20261004.md)를 우선 읽는다. 최신 고정 train MRR은0.35→1.0, loss는3.8925→1.0871로 개선됐다. 작은 batch의 학습 경로 확인이며 전체 validation 품질 승인과 구분한다.

실제 CT train 2환자(liver_5/liver_6), 별도 validation 1환자(liver_31)를 사용했다. RTX 5070 Ti CUDA에서 physical sample batch2, 후보 graph batch16, 원본 10,434,532 parameter 모델과 두 view로 **명시적 2-update smoke**를 수행했다. scheduler는 원본 T_max=40을 유지했다.

| 지표 | 초기 | 1 update 뒤 | 2 update 뒤 |
|---|---:|---:|---:|
| 고정 train MRR | 0.3500 | 1.0000 | 1.0000 |
| 고정 train top1 | 0.0000 | 1.0000 | 1.0000 |
| train positive−best-other margin | −0.00754 | +0.00755 | +0.01358 |
| 별도 환자 MRR | 0.2500 | 0.1667 | 0.1667 |
| 별도 환자 top1 | 0.0000 | 0.0000 | 0.0000 |
| 별도 환자 margin | −0.02817 | −0.04022 | −0.04678 |

**10mm에서도 train 순위를 실제로 올리는 경로는 확인됐다. 별도 환자 성능 개선은 확인되지 않았다.** 전체 1,085개 trainable parameter tensor에 gradient가 전달됐다. 소수 환자 2-update를 전체 정확도나 일반화 성능으로 제출하지 않는다. GPU를 다른 프로젝트와 공유했으므로 측정 시간으로 서버 epoch 시간을 환산하지 않는다.

원시 결과는 `validation/v1_scope_learning_20261004/`에 보존한다. 새 30mm arm은 실행하지 않았으므로 동일 초기 상태의 새 30/10mm 성능 대조를 측정했다고 주장하지 않는다.

## 서버 전체 학습

사용자용 진입점은 `tools/run_v14_scope_training.py`다. 기존 `tools/run_v1_bounded_training.py`의 초기화·검증·요청 생성 함수를 그대로 호출하고 `hiercp_v1x.scope_training_entry`의 prepare → train을 한 번씩 실행한다. 기존 bound helper 파일을 변경하지 않으므로 동일한 GPU·범위·자원·실험 경로로 재실행할 때 기존 checkpoint 재개 계약을 유지한다.

물리 GPU3, 10mm, 기존에 안내한 실험 폴더를 그대로 사용한다. 아래 명령은 게시된 branch에서 새 진입점을 가져온다. 재실행 때 실행 helper의 bytes가 기존 실험과 달라지면 조용히 재개하지 않고 오류를 낸다. 같은 계약의 준비/학습 결과가 있으면 이어간다. 새 v1.4 이름을 붙이려고 폴더를 바꾸거나 checkpoint를 복사하지 않는다.

```bash
CP_GPU=3
CP_MARGIN_MM=10
CP_REVISION=origin/codex/v222-server-r6
CP_EXPERIMENT="/home/aicompetition06/Medical/experiments/v1_m${CP_MARGIN_MM}_seed42_20261004"

conda activate nnunet &&
cd /home/aicompetition06/Medical/HierCP-v1-47bdb58 &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach "$CP_REVISION" &&
python -u tools/run_v14_scope_training.py \
  --gpu "$CP_GPU" \
  --margin-mm "$CP_MARGIN_MM" \
  --source-experiment /home/aicompetition06/Medical/experiments/v1_native_nested416_seed42_20261003/native \
  --experiment "$CP_EXPERIMENT" \
  --cuda-gib 40 \
  --rss-gib 192
```

버전 표기 변경은 별도 7개 단위 검사로 확인했다. 기존 manifest/checkpoint byte 보존, sidecar의 계약 결속, prepare 실패 시 train 미실행, prepare/train 각각 한 번의 동일 요청, 기존 8c7e914 helper SHA 보존을 검사했다. 이 검사는 GPU 학습이나 새 정확도 평가를 의미하지 않는다.

- 원본 v1 ZIP의 202파일을 별도 snapshot으로 꺼내 실제 bytes/inventory를 검사한다. 원본 파일 수정 없이 범위 adapter를 메모리에서 적용한다.
- 원본 84train/21validation/outer26 제외, seed42, 40epochs, 후보 pool128, sample당 curriculum8, 두 view, 모델·optimizer·loss·L1/L2를 유지한다.
- `adaptive_roi_margin_mm`과 `context_outer_radius_mm`만 10mm로 설정한다. ROI는 **완전한 transformed tumor footprint의 bbox + 양쪽10mm**다. 10mm가 전체 변 길이나 종양 내부만 보는 크기를 뜻하지 않는다.
- CNN 입력48³·channel/depth, GAT3층·128D·4heads, 관계별 edge/hop 규칙, shell 정의·거리 normalization을 유지한다. 새 node/edge cap은 없다. 물리 범위가 작아져 실제 문맥이 달라지는 실험이다.
- native에서 준비한 population bank와 region cache를 바이트 그대로 복사한다. 영역 분할이나 prototype를 다시 학습하지 않는다.
- 원본 성공 sample은 같은8후보·transform·GT·difficulty/corruption·patient/prototype graph를 검증하고 L0만 재생한다. supervision digest가 달라지면 실패한다.
- 원본 sample이 없는 실패/누락 요청은 원래 curriculum 준비 규칙을 적용하고 `scope_replay.jsonl`에 `paired_native=false`를 기록한다. 이 요청에 동일 후보 재사용을 주장하지 않는다. 실패를 skip하거나 작은 sample로 대체하지 않는다.
- 전체 새 캐시가 완성되지 않으면 학습을 시작하지 않는다. 기존 실패 native 캐시는 읽기 참조이며 training cache로 승격하지 않는다.
- preparation은 원본 case 병렬 runtime을 사용한다. training physical batch/worker는 실제 CUDA 처리량·메모리 calibration으로 결정한다. local smoke의 batch2/workers2를 최종 기본값으로 복사하지 않는다.

## 점수·시간·checkpoint

학습 전 전체 validation21의 초기 MRR/top1/margin을 `results/v1.0/initial_validation.json`에 저장한다. optimizer/scheduler/best selection을 변경하지 않으며 root RNG와 validation worker generator를 복원한다.

각 epoch는 train loss/ranking loss/MRR/top1/margin, validation MRR/top1/margin을 출력한다. `epoch_telemetry_*.jsonl`에는 epoch wall time, loader 대기, worker sampling, GPU compute, validation, checkpoint 저장, peak VRAM을 기록한다. 병렬 worker 시간의 합을 epoch wall time에 단순 합산하지 않는다.

원본 best 선택 기준과 optimizer·scheduler·AMP scaler·RNG를 포함한 last-epoch resume를 유지한다. 실제 neural state의 scope digest buffer와 architecture suffix로 native/다른 margin checkpoint를 거부한다. config/source/split/population/region/helper의 실제 bytes도 재실행 전에 검증한다.

Ctrl+C는 foreground 작업을 중단한다. 같은 명령을 재실행하면 **마지막 완료 epoch**에서 이어간다. 현재 epoch의 모든 update를 저장한다고 주장하지 않는다. 기존 checkpoint나 SSH 셸을 종료·덮어쓰는 명령은 없다.

## 검증 범위와 미검증

CSV 입력, 전체 cohort 요청, 원본 복사·checkpoint marker·worker·epoch 기록은 단위 검사 및 별도 실제 CT/CUDA worker 검사로 확인한다. 실제 서버105환자 새 cache와40epoch는 로컬에서 실행하지 않았다. 128-candidate production CP와 nnU-Net은 자동 시작하지 않고 `quality_verified=false`를 유지한다.

v2.2를 “전혀 학습되지 않는다”로 단정하지 않는다. 최신 짧은 검사에서는 train separation이 개선됐지만 held-out 성능은 개선되지 않았다. support 갱신 간격 등은 원인 후보이며 이10mm 결과로 단일 원인을 확정하지 않는다. 먼저 이 arm의 전체 학습 곡선을 기존 v1 결과와 비교한다.

## 2026-10-04 서버 전체 학습 결과 — 사용자 터미널 기록

실험 `/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004`의 서버 기록에서 40/40epoch 완료와 보존된 40개 epoch 기록이 보고됐다. 다음 값은 사용자가 전달한 터미널 요약이며, 로컬에서 서버 checkpoint와 원시 JSONL을 새로 읽어 독립 평가한 수치가 아니다.

| 서버 validation | MRR | top1 | positive−best-other margin |
|---|---:|---:|---:|
| 초기 | 0.301521 | 0.055556 | −0.014954 |
| best epoch11 | 1.000000 | 1.000000 | +14.371202 |
| 마지막 epoch40 | 1.000000 | 1.000000 | +13.398003 |

이 결과는 **10mm의 원본 source-anchor/curriculum8 순위 목표가 서버 학습에서도 학습됐다는 기준선**으로 사용한다. 앞의 로컬 DEBUG held-out 하락을 서버40epoch 결과로 잘못 일반화하지 않는다. MRR/top1=1은 이 후보 집합과 평가 방식에서의 수치이며, 전체128 후보의 관측 P/U 순위나 실제 CP 적합성·segmentation 정확도100%를 뜻하지 않는다. 기존 `quality_verified=false` smoke receipt는 수정하지 않는다. 설정된84/21 cohort와 실제 eligible sample/case 수 및 metric 집계 단위는 원시 cache manifest·epoch JSONL을 새 대조의 실행 계약에 결속해 확인한다.

epoch 시간은 아직 전달되지 않았다. 위 수치만으로 30mm 대비 시간 단축률이나 다른 하드웨어 대비 가속을 계산하지 않는다. 기존 cache/weights/optimizer/epoch 기록은 보존하고 native/30mm/10mm 기준선을 자동 재학습하지 않는다.

## 다음 대조 — 성공한10mm v1에서 v2.2 변경 묶음 찾기

사용자가 요청한 방식은 한 부품씩 순차 교체하는 전체 탐색이 아니라 **변경 묶음의 절반과 나머지 절반을 대조하고, 성능이 나빠지는 묶음을 다시 나누는 실험**이다. 구체적 상태와 코드 경계는 `config/v14_m10_component_search.json`에 기록했다. 첫 절반 A의 구현·짧은 실제 CT/CUDA 검사·서버 실행 연결은 [v1.5 half A 기록](v15_half_a_learning_20261004.md)에 별도로 정리했다. 이 설계 JSON 자체는 전체 학습 완료나 성능 승인 receipt가 아니다.

첫 architecture 비교의 기준선은 위의 완성된10mm v1.4다. A는 L0 입력·encoder·readout·pair fusion, B는 L1 support/prompt·L2 prototype·score 경로다. **먼저 `baseline+A`만 검사한다. 성능 저하가 재현되면 A를 다시 절반으로 나누고, 재현되지 않으면 나머지 `baseline+B`를 검사한다.** 각 단계에서 문제를 재현한 묶음만 좁힌다. 사용자의 이분 탐색 요청에 따라 A+B 조합 실험은 제외하고 기준선도 재학습하지 않는다. 모든 arm은 **v1 source-anchor GT, 후보8/pool128, 후보 순서·변형·difficulty/corruption, 원래 loss와40epoch를 유지**한다. 기존 v2.2의 P/U bank를 v1 정답으로 이름만 바꾸거나 GT를 donor compatibility로 재해석하지 않는다.

10mm source/cache·split·bank·region 및 기준선의 실제 batch/worker lock을 검증해 재사용한다. 유지하는 모듈은 동일 seed42 초기 가중치로 맞추고, 교체 모듈의 초기화 출처·신규 parameter·neural hash를 기록한다. 모든 새 arm은 fresh optimizer/scheduler/scaler로 시작하며 epoch11/40 best를 새 모델의 exact resume로 쓰지 않는다. 기존과 다른 update schedule이 필요한 support 경로는 숨겨서 바꾸지 않고, support 출처·gradient/detach·refresh cadence를 명시한 별도 factor로 다룬다. 같은 물리 batch를 맞춰도 후보당 처리량과 optimization step 수의 의미가 같아야 한다.

매 epoch 초기 대비 train/validation loss·MRR·top1·margin, case별 결과, 실제 epoch wall time·loader·GPU compute·validation·checkpoint·peak VRAM을 비교한다. worker 시간 합은 wall time과 중복 합산하지 않는다. 공통 validation 환자의 paired difference를 기록하며 임의 tolerance나 자동 PASS를 만들지 않는다. 어느 단계에서든 두 절반 모두 저하를 재현하지 못하면 **해당 조건에서 원인을 좁히지 못했다고 보고**한다. 임의 부품을 범인으로 정하거나 추가 조합·상호작용 실험을 자동 시작하지 않는다.

큰 묶음이 좁혀지면 A를 입력/encoder와 readout/fusion으로, B를 L1과 L2/scorer로 나눈다. P/U supervision·balanced CE/alignment·P×U schedule 전환은 architecture 검색과 분리한 다음 실험이다. 다른 GT의 MRR 차이를 동일 정답에 대한 부품 고장으로 제출하지 않는다.

코드 조사에서 실제 adapter 경계가 확인됐다. 원본 L0는 여러 의미의128D dictionary를 반환하고 현재 LocalCNN은 단일128D tensor를 반환한다. tensor 하나를 dictionary 모든 key로 복제해서 연결하면 원래 의미를 보존하지 못한다. 또한 원본 fixed48 ROI와 현재 native-spacing crop의 정규화·target erasure 계약이 다르다. **v2 CNN을 원본 dense ROI에 연결하는 대조와 실제 native-spacing v2.2 입력은 별도 이름과 계약**으로 구분한다. L1 patient graph와 prompt support의 topology도 별도 bridge가 필요하다.

기존 `hiercp_v1x/group_search.py`는 metadata planner이고 v1.0→v1.3 runtime은 누적 변경이다. 이번에 별도 `half_a_model.py`, `half_a_training.py`, `half_a_entry.py`, `tools/run_v1_half_a.py`를 추가해 첫 A만 실행하도록 구현했다. 기존 SHA에 결속된 v1.4 helper와 202개 원본 source는 변경하지 않았다. A는 원본 fixed48 ROI·target erasure·role/shell 입력을 유지한 CNN bridge이며 native-spacing v2.2 전체 복제가 아니다. 실제 CT/CUDA 8-update 검사와 46개 단위·실행 계약 검사를 완료했다. B·A+B·40epoch 전체 학습·production CP·nnU-Net은 자동 실행하지 않았다. 기존 DEBUG receipt를 전체 품질 승인으로 승격하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자 요청 범위 변경만 별도 arm에 적용했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 명시적 local smoke는 분리했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
