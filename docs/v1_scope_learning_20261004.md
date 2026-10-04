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
