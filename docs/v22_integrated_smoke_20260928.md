# v2.2 로컬 GPU 한 프로세스 통합 smoke

이전에는 추천/CP 입력 검사와 native GPU 학습 검사를 따로 실행했다. 이번에는 **실제 GNN 추천 → 인증 RPC → raw CP → native crop → 표준 영상 증강 → nnU-Net forward/loss/backward/SGD → epoch checkpoint → 로더 재생성/재개 → 다음 업데이트 비교**를 같은 프로세스에서 실행했다.

기준 commit은 `085e76cc5479ceee25113357a0249ff24a864bc7`이다. production core83/runtime20 및 online identity15는 변경하지 않았다. 추가된 것은 DEBUG 통합 검사와 기록이다. 전체 support와 장기 학습은 서버에서 실행하는 방침을 유지한다.

## 실제 실행과 결과

| 항목 | 결과 |
|---|---|
| GPU | RTX5070Ti 16GiB |
| 원본 CT / donor | liver_31 / liver_73 component1, 원본 내용 hash 확인 |
| validation 입력 | outer validation의 liver_0, 실제 validation transform/step |
| GNN | 기존 전체 모델5,550,806 parameters, DEBUG support8, physical batch2 |
| 후보 | 생산 규칙의128개 모두 실제 채점, 유효 최고점 index57 선택 |
| 점수 검사 | 반복 일치, recipient tumor annotation을 liver로 바꿔도 raw score/rank 동일 |
| GNN dense CT 입력 | source/target 각각 `[1,48,48,48]` |
| branch graph node 수 | 최소2,769 / 최대14,169 / 평균7,023.625 |
| branch graph edge 수 | 최소212,112 / 최대1,109,343 / 평균469,240.637 |
| segmentation 모델 | 기존 ResidualEncoderUNet102,350,575 parameters, 전체 trainable |
| segmentation 입력 | 초기 crop205³ → 표준 증강 → `[2,1,128,128,128]` |
| batch / accumulation / effective batch | 2 / 1 / 2 |
| optimizer | 실제 SGD momentum0.99/Nesterov, CUDA autocast/GradScaler |
| 실제 CP footprint | raw2,368 voxels → native/crop4,124 voxels |
| 증강 후 실제 학습 CT 변화 | 같은 증강의 no-paste 대조와25,600 voxels 차이 |
| 증강 후 추가 종양 정답 | 같은 대조보다4,124 voxels 증가 |
| 첫 CP 학습 update | 모델 내용 hash 변경 확인; loss1.3216165304 |
| 재개 전후 다음 입력 | 로더/epoch seed를 새로 구성해 생성; data/target/CP flag/token exact |
| 다음 loss | 양쪽1.1939997673034668 |
| model/SGD momentum/gradient 최대 차이 | 모두0 |
| model/optimizer/scaler/gradient hash 및 RNG | exact |
| 독립 raw→native oracle | CT 최대 차이0, segmentation exact |
| CUDA peak allocated | 6,597,423,616 bytes, 약6.14GiB |
| sampled peak RSS | 8,872,103,936 bytes |
| 전체 측정 시간 | 218.18초; 학습 epoch 처리량이 아님 |
| 최종 DEBUG 폴더 저장량 | 약2.26GiB; raw native 준비와checkpoint 포함, 전체 graph cache 없음 |

그래프 통계는128쌍의 source/target 총256개 **논리 branch** 통계다. 같은 donor branch의 재사용을 RAM의256개 독립 복제본으로 해석하면 안 된다. 원래 graph 구조를 줄이지 않았다. graph 생성은 기존 자원 측정 scheduler가 worker1/2/4 wave를 비교하고2를 선택했다. 여러 모델의 memory를 따로 측정해 더한 수치가 아니라, 두 모델을 함께 올린 동일 실행의 peak다. 다만 support8이므로 full-support peak는 아니다.

GNN 호출 전에 native warm update를1회 실행해 실제 SGD momentum/gradient buffer를 만들었다. 이후 train loader의 첫 missing receipt가 실제 RPC/owner를 호출했다. owner의 점수와 선택을 덮어쓰지 않았고 실제 native paste를 소비했다. 그 다음 epoch0에서 실제 CP update1회, epoch1의 연속/재개 next update를 각각 실행했다. warm-up으로 버린 loader batch를 optimizer update 수에 포함하지 않는다.

검사에는 production OnlineRank trainer의 `train_step`, CP 통계/transport 검사, GPU lock, 부모 epoch lifecycle, checkpoint save/load와 `PairedTrainerMixin._reset_comparison_epoch`를 사용했다. 다음 입력을 이전 tensor로 재생하지 않고 실제 로더와 표준 증강에서 다시 생성했다. 첫 입력과 다음 epoch 입력 hash는 다르며, 연속/재개 next 입력은 같다.

## 증강 후 CP 영향의 별도 확인

단순히 `cp_applied=1`만 확인하지 않았다. 실제 paste 직전 crop을 대조용으로 보관하고, production `get_training_transforms`가 만든 동일 transform에 같은 Python/NumPy/Torch CPU 난수를 적용했다. 대조 출력은 학습에 넣지 않으며 대조 후 실제 난수 상태를 복구한다. 실제 학습 batch에서 CT 차이와 추가 종양 정답이 남는지 확인했다. 추가 대조가 없었던 앞선 성공 실행과도 학습 입력 및 다음 model/optimizer 상태 hash가 동일했다.

이 데이터에서 확인한4,124 voxels는 이 smoke의 결과다. 모든 augmentation/모든 CP 이벤트에서 붙인 종양이 항상 남는다는 보장은 아니다. 표준 crop/변환 규칙을 바꾸지 않았다.

## 명시적 DEBUG 조건

- 최신 완전한 production artifact 대신 기존 실제 학습된 **DEBUG GNN checkpoint/support8**을 사용했다. full cohort/catalog admission은 검증 대상이 아니다. DEBUG 초기화에서만 base OnlineCP constructor와 기존 native 자료를 연결했으며 production validator를 완화하지 않았다.
- training case1개와 실제 outer validation case1개, 짧은 update로 구성했다. 설정의250epoch horizon은 유지하지만250epoch 학습을 실행한 것이 아니다. Dice/loss는 임상 성능 지표가 아니다.
- compile은 DEBUG에서 off다. CPU 증강 worker0의 명시적 single-process 경로로 로더/난수 경계를 검사했다. production worker 설정은 변경하지 않았으며 multiprocessing augmentation/compile 조합의 합격을 주장하지 않는다.
- CP 메타데이터 확률은80% 그대로다. 짧은 검사에서 적용/미적용 분기를 모두 거치도록 batch2의 draw를 `.1/.9`로 지정했다. 로그의 `applied=1/2, rate=0.5`는 이 통제 입력의 결과이며 **CP 확률을50%로 변경한 것이 아니다.** donor draw도 실제 liver_73 component1을 지정한 DEBUG 입력이다.
- 원래128개 후보, node/edge 규칙, L0/L1/L2, source patch, nnU-Net 모델/해상도/batch는 유지했다. 점수·선택·paste·augmentation 결과·loss를 dummy로 대체하지 않았다.
- segmentation은 global deterministic=False, cuDNN deterministic=True, benchmark/TF32=False다. GNN은 기존 scoped strict runtime을 사용한다. 이번 exact 결과를 모든 CUDA CE 실행의 결정론 보장으로 확대하지 않는다.

## 실행 및 증거

로컬 실행 명령(새 output 경로가 필요하며 기존 output 덮어쓰기 거부):

```powershell
.\.venv\Scripts\python.exe -u tools/verify_v22_online_actual_debug.py --checkpoint work/v22_model_integrity_20260927_DEBUG/uninterrupted/checkpoint.pt --native work/local_v21_5070ti_20260919/native/native.json --output work/v22_integrated_smoke_20260928_final_DEBUG --integrated-segmentation
```

`tools/v22_integrated_smoke_debug.py`가 새 검사 본체다. `validation/v222_r6/integrated_smoke_20260928_DEBUG.json`에 실제 결과, 입력/state hash, graph 자원 측정, source hash, 로그/체크포인트 hash를 기록했다. 최종 GPU 실행 통과 및 정적 구문 검사를 완료했다. 이전71개/21개 회귀검사 숫자를 이번 결과에 더하지 않았고, 이번에는 production 불변을 확인해 그 단위검사를 반복하지 않았다.

첫 실행은 DEBUG progress JSON에 NumPy int64가 들어가 실패했다. 로그 출력 값만 Python int로 변환해 수정했다. 두 번째 실행에서 통합 경로가 통과했고, 최종 실행에서 증강 대조와 graph 통계를 더해 다시 통과했다. 실패/이전 성공 기록을 보존했다.

## 서버 검증과의 구분

**이제 로컬의 한 프로세스 통합 smoke는 완료다.** 서버에서 남은 것은 full-support G3, production catalog/전체 support/worker/compile 조건의 G4, 동일 조건 비교 학습 및 평가 G5다. 이번 추가로 catalog나 전체 graph cache를 새로 만들어야 하는 변경은 없다. 로컬 전체 데이터 학습은 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버/원격 세션 종료 명령이나 신호를 사용하지 않았다.
- [x] 사용자 파일과 기존 실험 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이·너비·graph 규칙·production 데이터·batch를 축소하지 않았다.
- [x] 숨겨진 production subset/cap/fast mode를 추가하지 않았다.
- [x] 실제 GPU/CPU/RAM과 동시 모델/optimizer 상태를 측정했다.
- [x] graph 병렬 worker 측정과 DEBUG augmentation worker 조건을 구분했다.
- [x] 실제 입력→추천→CP→증강→forward/loss/backward/SGD→저장·재개를 연결했다.
- [x] CP가 증강 후 실제 학습 CT/정답에 영향을 주는지 대조했다.
- [x] dummy 예측이나 random fallback을 사용하지 않았다.
- [x] DEBUG 통제 draw와 production CP80%를 구분했다.
- [x] 로컬 통합 smoke와 서버 전체 학습/평가 미완료를 구분했다.
