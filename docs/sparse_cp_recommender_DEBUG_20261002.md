# v2.2 — CP 추천 실행과 실제 CT 순위 학습 DEBUG

2026-10-02. **v1처럼 유효한 Copy-Paste 추천 시스템을 복원한 결과가 아니다.** 최신 footprint 그래프를 전체 128개 후보 점수 → 기존 원본 mask 필터 → 실제 paste에 연결한 DEBUG 구현은 실행됐다. 그러나 별도 CT의 관측 위치 순위는 개선되지 않았다. 그래프 생성·gradient·paste 성공을 추천 품질로 대체하지 않는다.

## 확인한 연결 문제와 이번 구현

기존 online CP 경로는 `train_v22_online_rank` → `nnUNetTrainer_OnlineRankV22` → `RankedEntryBuilder` → 추천 artifact → `rank_then_filter` → 원본 payload 검증 → `OnlinePairedCP`다. 기존 native CNN artifact에는 이 경로가 있지만, 새 `FootprintPhysicalSparseL0`를 복원하는 production artifact/export/trainer 경로는 없었다. 새 그래프의 3D 화면을 만들었다는 사실만으로 production CP 연결을 주장할 수 없었다.

이번에 별도 `l0_sparse_feature/cp_diagnostic.py`를 추가했다. 기존 학습 reader가 요구하는 observation target 및 고정 학습 record 목록과 분리하여, 실제 CP 이벤트의 recipient·train-only donor·component·원본 CT·mask·spacing·anchor·128개 PlacementSpec을 검증한다. Scorer 입력에는 `target`이나 `observation_target`을 넣지 않는다. recipient와 donor의 patient group, full-mask 내용 및 좌표 frame을 확인하고, 동일 donor의 128개 후보를 physical batch32로 점수화한 뒤 기존 `rank_then_filter`를 사용한다. 점수 변경, record skip, 실패 시 CP no-op은 없다.

이것은 **diagnostic-only inference bridge**다. Production artifact·온라인 trainer를 새 그래프로 교체하지 않았고, 새 학습 가중치를 저장·배포하지 않았다. 기존 native CNN/L1/L2/loss/Basic CP/원본 mask 검사 구현은 변경하지 않았다.

## 실제 CT CUDA 대조 조건

`tools/compare_latest_sparse_cp_learning_debug.py`에서 기존 native CNN과 최신 footprint 그래프를 대조했다.

- 초기 CNN·fusion·L1/L2는 기존 로컬 DEBUG checkpoint saved step4에서 동일하게 복제했다. 서버에서 오래 학습한 모델의 재현 결과가 아니다. SAGE·graph readout은 seed42의 새 가중치다.
- 기존 assignment의 train case `liver_66/71/72/75` 전체 527개 관측(P15 + U512), 별도 validation case `liver_31` 전체 136개 관측(P8 + U128)을 사용했다. 전체 원본 14,102개 중 663개, 4.70%의 명시적 DEBUG이며 production subset으로 저장하지 않았다.
- 같은 case donor를 고정했고 P/U GT를 변경하지 않았다. **P=관측된 적격 종양 anchor, U=미관측 비교 위치**다. U를 CP 부적합 정답, P를 특정 donor의 CP 적합 정답으로 해석하지 않는다.
- 기존 `same_donor_live_v1`의 전체 ranking·balanced observation CE·L2 alignment objective, L1/L2, 후보128, mask 검사를 유지했다. 각 cycle의 train P×U 1,920쌍을 정확히 한 번씩 사용했다.
- branch별 3 DEBUG cycle, 60회 연속 AdamW update를 실행했다. Physical/effective batch32, accumulation1. 자연스러운 마지막 tile의 일부 batch만 32보다 작다.
- 최초와 cycle 경계에서 현재 모델의 train-only support를 재인코딩했다. Held-out case는 support·teacher·optimizer 입력에 들어가지 않았다. 평가도 현재 query를 재인코딩했다.
- Gradient finite·각 모듈 연결·parameter 변화·Adam 연속 상태를 확인했다. 기본 모델 깊이·채널·후보·전체 데이터 계약을 바꾸지 않았다.

최신 그래프는 3개 band 각32개의 context seed와 query 및 검증된 relay 경로를 사용한다. 96은 최종 node cap이나 검증된 최적 크기가 아니다. CNN 채널은 [12,24,32], SAGE는 3층128D, L1은 2층128D/4head, 기존 L2를 유지했다. Parameter 수는 native CNN 1,125,718, footprint 그래프 1,785,558이며 모두 trainable이다.

## 순위 결과

| 경로·평가 데이터 | P/U pair-win 초기→최종 | MRR 초기→최종 | R@1 초기→최종 | R@5 초기→최종 | Pairwise loss 초기→최종 | 평균 P−U 점수 초기→최종 |
|---|---:|---:|---:|---:|---:|---:|
| native CNN / train4case | 54.58%→82.97% | .132639→.508333 | 0→.066667 | .066667→.466667 | .687423→.443847 | .012122→1.016601 |
| native CNN / held-out1case | 40.43%→53.22% | .111111→.071429 | 0→0 | 0→0 | .694541→.739231 | −.002735→−.019804 |
| footprint graph / train4case | 42.50%→92.71% | .055908→.687500 | 0→.133333 | 0→.533333 | .696878→.264581 | −.007109→2.220735 |
| footprint graph / held-out1case | 56.93%→50.78% | .100000→.062500 | 0→0 | 0→0 | .691090→.953766 | .004410→−.039390 |

Pair-win은 P 점수가 U보다 높은 쌍의 비율이며 동점은 0.5로 센다. MRR는 case별 첫 관측 종양 순위의 역수, R@k는 관측 종양에 대한 micro recall이다. 분류 정확도·CP 적합성 정확도가 아니다. 1,024개의 held-out P×U쌍을 1,024명의 독립 환자처럼 세지 않는다.

Train 순위는 학습됐지만, 별도 CT에서 native CNN의 첫 관측 종양은 14위, 그래프는 16위였다. 두 경로 모두 최종 R@1/5/10=0이다. 최종 held-out score 표준편차는 각각 .377470/1.120450이므로 이 DEBUG의 실패를 단순한 상수 점수로 설명할 수 없다. **후보 점수 차이가 생겨도 원하는 순위가 다른 환자에게 이어지지 않았다.** 60 update와 held-out1case만으로 최종 일반화 성능·유일한 원인을 확정하지 않는다.

## 실제 CP 실행 검사

Held-out recipient `liver_31`, 원래 assignment의 train donor `liver_100` component4로 각 branch의 한 이벤트를 실행했다. 원본 source mask766복셀을 기존 spacing 변환으로 recipient의 260복셀 full paste mask로 변환했다. Bbox 또는 clipped mask로 대체하지 않았다.

| 경로 | 점수화 후보 | 기존 mask 필터 허용 | 선택 centre(native index) | 실제 paste 복셀 |
|---|---:|---:|---|---:|
| native CNN | 128 | 102 | [346,330,33] | 260 |
| footprint graph | 128 | 102 | [341,355,77] | 260 |

선택은 모델 점수의 순서와 기존 full-mask 규칙으로 결정됐다. Original CT/mask/anchor/spacing/frame 검증, paste 후 mask 내부 값 및 외부 불변성, 원본 입력 보존을 통과했다. **붙여넣기 실행 성공과 위치가 다르게 선택됐다는 사실은 좋은 CP 위치를 학습했다는 증거가 아니다.** `mechanical_pass=true`, `quality_evaluated=false`, `production_ready=false`를 유지했다. Downstream segmentation 학습·Dice/작은 병변 recall·Basic CP 대비 효용은 측정하지 않았다.

## 비용·자원·범위

RTX5070Ti 16GB 한 개, CPU16logical, 시작 RAM 여유41.86GiB, workers8, CUDA budget12GiB/RSS40GiB/resident24GiB로 실행했다. 최종 process RSS2.38GiB다. 입력·불변 geometry의 CPU 준비는 prefetch하고 실제 모델/gradient/optimizer 검사는 CUDA에서 수행했다.

| 경로 | compute update 중앙값 | peak GPU allocated |
|---|---:|---:|
| native CNN | .225690초 | 1.821GiB |
| footprint graph | 1.768926초 | 3.675GiB |

이 경계는 server의 전체 epoch·전체 support·checkpoint 저장 비용과 다르다. 작은 DEBUG에서 그래프가 CNN보다 느렸으며 3시간/epoch 해결이나 그래프의 속도 우위를 주장하지 않는다. `nvidia-smi`의 reserved/context/다른 앱 포함 수치와 PyTorch allocated peak를 혼동하지 않는다.

첫 실제 support physical32의 CNN 입력은 [33,1,58,50,7]이었다. 그래프 pair당 node388–437, edge 약1,950–2,295, component1/isolates0을 확인했다. 이것은 전체 cohort의 connectivity 증명이 아니다. 이전 `liver_66:65`의 native 미도달 seed 진단을 이번 첫32개 검사로 덮지 않는다.

## 검사와 보존

- 정적 AST/CLI 검사 완료. 새 inference bridge의 synthetic UNIT 6개 metadata 검사 및 전체128 점수·기존 필터·정확한 paste CUDA UNIT1개 PASS. UNIT 데이터를 실제 CT로 보고하지 않았다.
- 실제 CT: 두 branch 각60연속 update, 별도 CT 순위 평가, 두 실제 CP 이벤트 완료. **추천 품질 PASS는 아니다.**
- 기존 checkpoint·assignment·20개 사용 raw image/label hash를 보존했다. Production 121파일 및 기존 사용자 수정 파일은 별도 byte 감사 대상으로 확인한다.
- 새 결과는 `D:/AI project/nnunet/work/latest_sparse_cp_learning_DEBUG_20261002/report.json`, 요약은 같은 폴더의 `compact_summary.json`이다. 원본 결과를 덮어쓰지 않았다.
- 새 그래프의 production export/load/online-trainer 연결, 전체 학습·전체 평가·CP 효용·서버 epoch 속도는 미검증이다. Production checkpoint/ready 표시·장기 GNN/nnU-Net 학습·원격 작업·Git push를 수행하지 않았다.

현재 증거로 수정 대상을 그래프 모양에만 한정할 수 없다. 다음 설계 판단은 고정된 P/U 정답에서 donor 조건·공간 표현·점수 경로가 어떻게 유효한 순위에 기여하는지 다뤄야 한다. V1의 direct score와 curriculum을 현재 목표의 정답까지 교체하는 방식으로 복원하거나, 후보·모델을 줄여 PASS를 만드는 변경은 이 결과에 포함하지 않았다.

## 재현 명령 — 별도 DEBUG 출력

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe tools/compare_latest_sparse_cp_learning_debug.py `
  --run work/local_cnn_experiment_resume_DEBUG_20261001/resumed `
  --assignment work/v222_v1_full_training_20260924/cache/pair_assignment.json `
  --assignment-receipt validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json `
  --output work/latest_sparse_cp_learning_DEBUG_20261002_rerun `
  --debug-cycles 3 --context-nodes-per-band 32 `
  --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 8 --cuda-gib 12 --rss-gib 40 --resident-gib 24
```

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 전체 계약과 분리된 명시적 DEBUG만 실행했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 physical32를 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 CUDA 실행은 OOM 없이 완료됐다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. UNIT synthetic fixture와 실제 CT 실행을 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
