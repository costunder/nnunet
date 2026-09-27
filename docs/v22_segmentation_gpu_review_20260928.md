# v2.2 F03 / U01 수정과 실제 native CUDA 재개 검사

사용자의 `095c007f-1cd4-4447-b48f-8a0c96f6dc04` 검토문 전체를 읽었다. 외부 sandbox 링크의 별도 보고서/ZIP은 첨부되지 않아 그 파일을 읽었다고 주장하지 않는다. F01/F02를 다시 미해결로 분류하지 않는다. 이번 수정은 새 segmentation checkpoint 경계에 한정하며 기존 GNN core83/runtime20, loss, 그래프, 데이터, production batch는 유지했다.

## 실제 설치본과 수정

로컬 `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`를 import해 확인했다. optimizer는 **SGD(momentum=0.99, Nesterov=True)**이며 upstream on_epoch_end는 **latest 저장 → best EMA 갱신 → best 저장** 순서였다. 설치 파일 경로와 hash는 검증 JSON에 기록했다.

**F03:** `tools/v22_seg_state.py`를 추가했다. 새 저장 때 CPU의 불변 snapshot을 한 번 만들고 모델·optimizer·scaler·logger·RNG·epoch·best·실행 계약·optimizer parameter mapping의 내용 hash를 기록한다. 복구는 저장 당시 hash를 비교하며 읽은 파일을 새 hash로 보정하지 않는다. 모델과 momentum의 유한성, shape/dtype, 실제 모델 이름과 optimizer group의 parameter 연결, 초기화된 state의 누락, 실제 optimizer 종류와 hyperparameter를 확인한다. 정상적으로 사용되지 않아 아직 momentum이 없는 parameter는 저장 시 coverage에 명시해 허용한다. CUDA GradScaler의 필수 값/범위도 확인한다. GNN AdamW validator를 native SGD에 재사용하지 않았다.

**U01:** 설치된 부모 on_epoch_end의 metric·best 결정·logger 갱신을 그대로 실행하되, 내부 save 요청은 결정이 끝날 때까지 지연한다. 완료된 epoch cursor와 best 결정을 담은 latest를 먼저 원자적으로 저장하고 best를 저장한다. 마지막 epoch에도 latest를 남겨 final 저장 전 중단에 대비한다. latest가 자신이 best인 epoch의 가중치·metric·hash를 포함하므로 두 저장 사이에 중단되면 그 정상 latest에서 best를 복구한다. 이전 epoch의 best가 필요한 경우에는 파일의 내용과 결속을 확인하고, 없거나 다르면 오류를 낸다. best에서만 재개하거나, parent의 best 저장을 누락시키거나, 손상 파일을 새 hash로 인정하지 않는다.

새 segmentation 계약은 `online_rank_epoch_resume_v2` / `online_segmentation_state_v2`다. 보호 정보 없는 v1 segmentation checkpoint를 exact하게 검증할 수 있다고 재표시하지 않는다. GNN artifact v5와 GNN 재개 경로는 변경하지 않았다. online source hash가 달라져 이전 catalog를 조용히 재사용하지 않으며 production final GNN/native에서 새 catalog가 필요하다. catalog 생성과 전체 graph cache 재생성은 다른 작업이다.

## GPU 검증

실행기: `tools/verify_v22_segmentation_cuda_debug.py`. 실제 설치 nnU-Net의 network construction, optimizer, GradScaler, train_step, validation_step, on_epoch_end를 사용했다. DEBUG 객체는 native constructor를 사용하지만 production ranked-bank admission/owner 생성자를 통과한 실행은 아니다.

| 항목 | 실제 실행 |
|---|---|
| 장치 | RTX 5070 Ti, 16 GiB |
| 입력 | 실제 전처리 CT liver_31의 crop, `[2,1,128,128,128]` |
| 모델 | 기존 native plans의 ResidualEncoderUNet, 102,350,575 parameters |
| 구조 | stages6, channels32/64/128/256/320/320, blocks1/3/4/6/6/6, decoder1/1/1/1/1 |
| optimizer | 설치본 SGD, momentum0.99, Nesterov, weight_decay3e-5 |
| precision | 실제 CUDA autocast/GradScaler, 저장 scale65536, growth_tracker2 |
| 정상 재개 | 초기 실제 update2회 → epoch 경계 저장 → 추가로 뽑은 세 번째 CT crop batch update → 복구 → 같은 다음 batch update |
| model / momentum / gradient | 최대 절대 차이 모두0.0 |
| model+optimizer+scaler+gradient 내용 hash | 연속/재개 모두 `00c905537964cbe4fa137a783958a74c370ae58920f0815d2111f4bcfc3092cb` |
| Python/NumPy/CUDA RNG | 비교 draw 일치 |
| next loss | 연속/재개 모두1.174100399017334; 이번 최종 실행은 bitwise 동일 |
| 손상 거부 | 실제 full native payload의 11종, 실제 load 메서드에서 GPU state 적용 전에 거부 |
| best lifecycle | 실제 부모 저장 순서에서 controlled EMA0.8→재개→0.7에도 이전 best 보존 |
| 두 파일 사이 중단 | EMA0.9의 latest 저장 후 best 쓰기 실패 주입 → latest에서 best 복구 |
| CUDA peak allocated / reserved | 6,680,252,416 / 7,652,507,648 bytes (약6.22 / 7.13 GiB) |
| 전체 검사 시간 | 74.78초, 직렬화·hash·손상11종·best 복구 포함 |
| sampled peak RSS | 7,902,199,808 bytes; full state CPU 복사/손상 대조 포함 |

손상11종: momentum 전체 누락, model NaN/Inf, momentum NaN/shape/dtype, parameter group 순서 변경, scaler key 누락/None/NaN/음수 counter. 파일 손상 입력은 저장된 실제 full-model payload를 메모리에서 변형한 것이며 변형 payload를 디스크에 11개 복제하지 않았다. 저장/복구 정상 경로는 실제 파일을 사용했다. 별도 단위 검사로 원자적 교체 실패 때 이전 checkpoint bytes 보존, unused parameter의 정상 빈 state, 내용이 바뀐 유한 tensor, 비어 있는 scaler 거부를 확인했다.

CPU는 파일 hash/직렬화/입력 준비/거부 검사에 사용하고, 실제 network forward·loss·backward·SGD·GradScaler·validation은 GPU에서 실행했다. 단위/회귀 **71개 통과**이며 GPU driver는 이 숫자와 별도로 보고한다. 최종 코드는 빈 scaler 거부와 shared tensor snapshot 회귀 검사를 포함하며, 이 코드로 CUDA 전체 검사를 다시 실행했다.

불변 snapshot은 같은 storage/view의 tensor 별칭을 보존한다. nnU-Net이 encoder를 여러 경로로 등록한 것을 각각 복제하던 저장 중복을 제거했다. 이번 DEBUG checkpoint 4개의 합계는 7.24GiB에서 3,276,246,012 bytes(약3.05GiB)로 줄었다. 최종 checkpoint는 보존하고 이번 작업에서 직접 만든 이전 DEBUG checkpoint 4개만 경로·크기·SHA256 기록 후 정리했다. 기존 실험 결과는 건드리지 않았다. 정리 기록은 검증 JSON에 포함했다.

## 실패 기록과 한계

첫 DEBUG 실행은 표준 padding label `-1→0` 처리를 빠뜨려 CUDA index 오류가 났다. 검사 입력 준비를 native train/validation의 동일 매핑으로 고쳤다. 두 번째 실행은 설치된 CUDA CrossEntropy가 strict `torch.use_deterministic_algorithms(True)`를 지원하지 않아 명시적으로 실패했다. 세 번째는 CUDA loss를 유지하고 global deterministic=False, cuDNN deterministic=True, benchmark/TF32=False에서 수치 비교했다. GNN scorer의 기존 strict runtime은 바꾸지 않았다. 세 번째 실행에서는 loss에 허용오차 내 차이가 있었지만, 최종 코드의 추가 CT crop batch 실행은 loss와 상태 모두 bitwise 동일했다. 설치 CUDA CE의 일반적인 bitwise 결정론을 보장한다는 뜻은 아니다. 실패/이전 성공 로그도 보존했다.

GPU driver의 epoch는 두 update를 사용한 DEBUG 경계이며 전체250-iteration epoch가 아니다. compile은 DEBUG에서 비활성화했으며 모델/해상도/batch를 줄이지 않았다. 표준 무작위 영상 증강 전체나 CP owner를 이번 driver에 합치지 않았다. best 시험의 0.8/0.7/0.9는 의도적인 lifecycle 입력이며 실제 Dice 성능이 아니다. full GNN support + segmentation optimizer 공존, production 초기화, 전체 cohort G3/G4, CP 효용 G5는 미검증이다. 이번 실행의6.22GiB를 그 공존 peak로 사용하면 안 된다.

모델·optimizer·scaler 단독 CUDA 정상 재개 증거는 추가됐지만, 전체 실험 완료나 임상 성능 검증은 아니다. 연구 위험(pair 비균등 가중, detached reference, CE/rank, shortcut, fixed-donor 효용)은 변경하지 않았다. 전체 cache와 장기 학습은 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버/원격 세션 종료 위험 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이·너비를 축소하지 않았다.
- [x] 그래프/전체 데이터 규모를 변경하지 않았다.
- [x] 숨겨진 subset/cap/fast mode를 production에 추가하지 않았다.
- [x] native plans의 physical batch2를 그대로 GPU에서 실행했다.
- [x] GPU/CPU/RAM과 실제 peak를 확인했다.
- [x] 모델 축소/CPU loss fallback으로 오류를 숨기지 않았다.
- [x] DEBUG 입력·반복 횟수·compile 조건과 production 설정을 구분했다.
- [x] synthetic lifecycle metric을 실제 성능으로 보고하지 않았다.
- [x] 실제 모델 forward/loss/backward/SGD/GradScaler 경로를 실행했다.
- [x] 실패와 재실행 조건 및 미검증 항목을 기록했다.
- [x] GPU smoke·회귀와 전체 학습/평가를 구분했다.
