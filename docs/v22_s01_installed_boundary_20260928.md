# v2.2 S01 설치본 확인과 CUDA epoch 경계 검증

사용자 첨부 `3aa7fcc8-a2d2-4100-a385-1583962c26bb` 전체를 읽었다. 첨부 본문의 sandbox 보고서/ZIP 링크는 별도 파일이 제공되지 않아 열어 보았다고 주장하지 않는다. 이번 기준 구현은 `c989ae1ed6944d9f9625e66e610bb29ab9858fc1`이다.

## 결론과 변경 범위

**조건부 S01의 직접 종료 분기는 로컬 설치 nnU-Net에 없다.** 실제 `on_epoch_end`는 latest 요청, best 결정/요청, logger plot, `current_epoch += 1` 뒤 정상 반환한다. `SystemExit`를 처리하는 production 패치를 임의로 추가하지 않았다. 서버 설치본은 이번 로컬 조사로 확인된 것이 아니며, 신호 종료 분기가 있는 다른 버전의 S01 위험을 해결했다고 주장하지 않는다.

`tools/inspect_installed_epoch_boundary.py`를 추가했다. 실제 OnlineRank trainer의 MRO, 각 `on_epoch_end` override, 설치 부모의 `on_epoch_end/on_train_end/run_training/save_checkpoint` 원문·시작 줄·파일 hash를 출력한다. Trainer를 생성하지 않고 신호를 보내지 않으며 로컬 검사 종료 후에도 CUDA 미초기화를 확인했다. AST의 직접 호출 목록은 간접 호출까지 안전하다는 증명이 아니므로 원문 검토와 구분한다.

상속 경로는 OnlineRank trainer → OnlineCheckpointMixin → PairedTrainerMixin → OnlineCP trainer → 설치 nnUNetTrainer다. 실제 epoch override는 checkpoint mixin과 nnUNetTrainer 두 개였다. 함수 원문을 `validation/v222_r6/installed_epoch_boundary_20260928.json`에 포함해 이전 전달본의 ‘설치 hash만 있고 코드 bytes 없음’ 한계를 보완했다.

학습/저장/추천 production 코드는 변경하지 않았다. core83/runtime20, online identity15, F03/U01, 모델·그래프·loss·batch·CP80%·seed42를 유지한다. 이번 검증 도구 추가 때문에 catalog나 graph cache를 다시 만들 필요는 없다. 이전 c989ae1보다 오래된 catalog의 호환성을 새로 인정한다는 의미는 아니다.

## 실제 GPU 검사 범위

`verify_v22_segmentation_cuda_debug.py --epoch-boundary-audit`로 실제 RTX5070Ti에서 기존 native ResidualEncoderUNet(102,350,575 parameters), SGD(momentum0.99/Nesterov), CUDA GradScaler, 실제 liver_31 crop `[2,1,128,128,128]`를 사용한다. 모델 규모와 physical batch2를 유지한다. CPU는 파일 I/O·상태 hash·손상 입력 대조에 사용한다.

추가 도구 `verify_v22_epoch_boundary_debug.py`는 그 **실제 CUDA optimizer update를 거친 모델**에 설치 부모의 epoch 함수를 호출한다. 통제 입력으로 첫 epoch1, 중간125, 마지막250 및 best 갱신/비갱신을 확인한다. 이 epoch 번호와 metric은 경계 테스트 입력이며 250epoch 학습이나 Dice 성능이 아니다. fresh 첫 epoch의 ‘이전 best 비갱신’은 유효한 시나리오가 아니므로 만들지 않는다.

각 경계의 실제 latest/best 파일을 저장하고 다시 읽어 cursor와 best 결속을 검사한다. parent logger 오류가 save 요청 뒤에 발생하면 저장하지 않아야 한다. 디스크 여유0 주입 때는 실제 80GiB reserve와 writer를 그대로 두고 오류 전파와 기존 파일 보존을 검사한다. 실제 설치 `on_train_end`의 final cursor250도 확인한다. 원래 driver의 latest→best 사이 쓰기 실패/복구와 다음 optimizer update 대조도 함께 실행한다.

이번 실행은 다음과 같이 통과했다.

| 검사 | 결과 |
|---|---|
| 실제 CT 다음 update의 loss/model/SGD/gradient/scaler/RNG | 동일; loss1.174100399017334, tensor 최대 차이0 |
| 통제 epoch1/125/250의 latest/best 재개 | 5개 조건 통과 |
| latest/best 사이 쓰기 실패 | 정상 latest에서 best 복구 |
| 부모 logger 오류 | 예외 전파, 미완료 epoch 저장 없음 |
| 디스크 여유0 | reserve80GiB 유지, 오류 전파, 이전 파일 bytes 보존 |
| 실제 부모 final 저장 | cursor250 및 실제 load 통과 |
| 기존 저장/온라인 연결 회귀 | 21개 통과; 이전71개와 합산하지 않음 |
| CUDA peak allocated/reserved | 6,680,252,416 / 7,652,507,648 bytes (6.22 / 7.13GiB) |
| sampled peak RSS | 9,457,766,400 bytes |
| 전체 검증 시간 | 207.13초 |
| 새 DEBUG checkpoint 보존 | 6개, 합계4,914,412,858 bytes; 전체 cache 생성 없음 |

실행 결과·VRAM·RAM·시간·상태 hash·파일 hash는 `validation/v222_r6/s01_GPU_boundary_20260928_DEBUG.json`에 기록한다. 명령 및 결과 로그도 이 파일에 해시와 함께 연결한다. 시간에는 직렬화·해시·실패 주입이 포함되므로 학습 throughput으로 사용하면 안 된다.

이번 통제 logger는 epoch125/250의 index/plot 검사를 위해 history를 채운다. 현재 epoch 종료 timestamp를 부모가 덮어쓸 때 나오는 logger의 `maybe some logging issue!?` 알림은 이 DEBUG 통제 입력에서 발생한다. 실제 전체 학습 로그를 가장한 기록이 아니다.

## 서버에서 읽기 전용 확인

실제로 학습에 사용하는 conda 환경을 활성화한 뒤 최신 checkout 안에서 실행한다.

```bash
python tools/inspect_installed_epoch_boundary.py
```

학습·GPU 초기화·OS 신호 전송·세션 종료를 수행하지 않는다. 서버 원문에 종료 분기가 있으면 best 결정 완료 시점과 epoch 증가 여부를 그 코드로 확인한 뒤 S01 수정 및 별도 경계 검사가 필요하다. 로컬 원문과 hash가 다르다는 이유만으로 결함이라고 판단하지 않는다.

## 제한과 실패 기록

- 실제 OS 종료 신호는 보내지 않았다. ‘SIGTERM 정상 저장 완료’ 검사가 아니다. 로컬 부모에 해당 분기가 없어 임의의 종료 handler를 설치하지 않았다.
- production ranked-bank constructor/full-support owner/CP/표준 전체 증강/segmentation update를 하나의 실행으로 합친 G4는 미완료다. 전체 cohort G3와 CP 효용 G5도 미완료다.
- CUDA CE의 strict deterministic 미지원 조건은 이전과 동일하다. 이번 실행의 동일성 관찰을 모든 실행의 결정론 보장으로 확대하지 않는다.
- 처음 회귀 실행은 `tests` 보조 모듈 경로 누락으로 21개 중3개 import 오류가 발생했다. 경로를 포함한 기존 실행 방식으로 다시 검사하며 실패 로그도 보존했다. production import 오류로 분류하지 않는다.
- 독립 검토가 제시한 pair 비균등 가중, shortcut, fixed-donor 목적, GT 간 union 의존, Recall/MRR와 CP 효용 구분은 여전히 연구 위험이다. checkpoint 검증에 연구 설계 변경을 섞지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험 명령과 신호를 사용하지 않았다.
- [x] 사용자 파일과 기존 실험 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이·너비·그래프·데이터·production batch를 축소하지 않았다.
- [x] 숨겨진 subset/cap/fast mode를 production에 추가하지 않았다.
- [x] GPU·CPU·RAM을 확인하고 실제 batch2 CUDA update를 사용했다.
- [x] DEBUG 경계 번호/metric과 실제 학습 성능을 구분했다.
- [x] dummy/random 출력을 예측 결과로 사용하지 않았다.
- [x] 실제 forward/loss/backward/SGD 경로와 저장·재개 검사를 구분해 기록했다.
- [x] 로컬 설치본과 미확인 서버 설치본을 구분했다.
- [x] smoke와 전체 학습/평가 미완료를 명시했다.
