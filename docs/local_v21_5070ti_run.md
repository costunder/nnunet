# v2.1 로컬 RTX 5070 Ti 실행 기록

2026-09-19 KST. 사용자 요청으로 로컬 데이터 준비와 전체 GNN 학습 실행을 진행한다. 실행 루트는 `work/local_v21_5070ti_20260919/`다. 첫 준비 실행과 잘못된 donor 정책의 `attempt2/`는 중단·보존했다. 현재 학습 완료나 의료 성능을 선언하지 않는다. 실제 진행은 아래 단계별 로그·완료 파일로 판단한다.

## 현재 수정본 — 공통 donor / 누수 방지

**재실행: 2026-09-19 22:21 KST**, `work/local_v21_5070ti_20260919/attempt3_shared_donor/`. 현재 소스 SHA와 일치하는 회귀·CUDA preflight를 보존하고, 완료된 131명 원본을 다시 검증한 뒤 전체 context 준비→GNN 40 epoch를 수행하는 launcher PID 26652를 시작했다. 설정은 subset/debug=false다. 실제 단계는 `pipeline_started.json`, `gnn_prepare_started.json`, `gnn_prepare.log`, `gnn_train_started.json`으로 확인한다. 준비 중인 상태를 optimizer 학습 시작이나 완료로 보고하지 않는다. native 250 epoch·전체 평가는 후속 단계이며 아직 실행되지 않았다.

최종 증거는 `versions/v2/verification_shared_donor_20260919/`에 모았다. 정적 22개·회귀 39개 통과, 전체 핵심 모듈 BF16 CUDA optimizer 1 step 통과, 67,983 context L1/L2 backward 재검증 통과(peak 7,445,368,320 bytes). 실제 외부 donor/native paste 성공 증거도 포함한다. 모두 같은 runtime source identity를 확인했다. 이전 code.txt는 ZIP으로 보존하고 235개 현재 소스로 최신화했다.

모든 outer-train 105명이 같은 inner-train donor 527개를 사용한다. 소형 종양 조건은 donor 조건이며 recipient 제외 조건이 아니다. GNN context는 균형 round의 train 67,983 / validation 67,591개다. native CP 이벤트는 전체 donor 풀에서 균등 추출하며, 실제 pair에 대해 128개 위치를 모두 평가하고 선택한 한 위치의 raw CT/GT payload를 저장한다. 설계·분할별 경계는 `docs/shared_donor_leakage_v21.md`를 따른다.

- `work/shared_donor_fix_20260919/debug_final/`: 회귀 **39/39 통과**, 실패·skip 0. validation donor/support 차단, 0-small recipient 포함, native validation loader 유지, 실제 raw payload 저장·로딩·paste를 검사했다.
- `task_cardinality_DEBUG_after.json`: 합성 pre-encoded 67,983개 context / 84명 / 원래 L1/L2 폭과 깊이의 backward 통과. peak allocated 7,445,368,320 bytes. L0를 포함한 전체 학습 메모리 측정은 아니다. 이전 attention kernel 65,535-row 한계 오류를 모든 context를 유지하는 실행 분할과 activation checkpointing으로 해결했다.
- `real_cp_DEBUG1/`: 실제 native baseline CT 오차 0, segmentation exact 확인 후, 재로딩한 runtime에서 preparation subtree가 없어 실패했다. subtree를 별도로 보존·복원하도록 수정했다.
- `real_cp_DEBUG2/`: native 저장·batch 측정을 추가 수정하면서 오래된 코드로 실행 중인 본인 DEBUG PID 38204만 중단했다. `stopped_superseded_debug.json`에 보존했다. 완료 증거가 아니다.
- `real_cp_DEBUG3/`: 실제 외부 donor `liver_1/component1` → 적격 소형 source 0개의 `liver_2` → 전체 128후보 GNN → native CT/GT paste **통과**. argmax index 63, native support 3,165 voxel, 붙인 영역 밖 GT 및 원본 CT/GT SHA 불변, finite CT. native baseline 최대 CT 오차 0 / segmentation exact. 전체 검사 581.03초에는 원본 hash 검증·후보 준비·batch 측정이 포함된다. 미학습 모델을 쓰는 명시적 DEBUG이며 production 진입점에서 이 catalog를 거부한다. 증거는 해당 `verification.json`과 bank의 entry receipt다. JSON의 `peak_vram`은 내부 calibration에서 peak counter를 재설정한 이후 값이므로 전체 검사 최고 VRAM으로 해석하지 않는다. 단계별 측정값은 receipt의 `scoring_resources`를 따른다.

아래는 수정 전 실행 이력이다. 1,704,342개 context와 `attempt2` 실행 중 설명은 현재 설정이 아니다.

## 확인된 상태와 실패 복구

- **최신 정정 — 2026-09-19 donor 정책 검토로 중단:** 사용자가 지적한 비대칭을 확인했다. 81명은 자기 환자의 소형 source만 사용하지만, 자기 소형 source가 없는 24명에게만 전체 inner-train source 527개를 적용한다. L2 환자 간 정렬 요구가 이 후보 생성 분기를 정당화하지 않는다. 이 정책의 타당성이 확립되지 않아 직접 시작한 graph 준비 PID 37104만 중단했다. 본훈련은 시작하지 않았고 cache는 미완성이다. 데이터·부분 결과는 보존한다. 중단 근거와 명령은 `attempt2/stopped_donor_policy_review.json`에 기록했다. native 전처리는 131/131명 완료했고 `native/native.json`을 보존했다. 아래 실행 시작 기록은 중단 이전 이력이다. 후보 정책을 재설계하지 않은 상태로 자동 재실행하지 않는다.
- 전체 archive 다운로드·MD5 검증·추출 완료. CT/GT 131명 검증 완료. outer train/val 105/26, inner train/val 84/21. 실행 원본은 `dataset/medical/Data/`, 원본 SHA와 검증 결과는 `dataset/dataset_ready.json`이다.
- 첫 GNN 준비는 source inventory 37/105에서 사용 가능한 RAM 대비 측정된 case peak가 커 `MemoryError`로 중단됐다. 당시 native 준비와 병행 중이었다. `gnn_prepare.log`, `gnn_prepare_failed.json`을 보존했다. 본훈련 optimizer step은 발생하지 않았다.
- 한 case의 모든 source mask를 동시에 유지하던 구현을 하나씩 생성하는 반복 가능한 collection으로 수정했다. 공통 source tensor는 content SHA를 기준으로 한 번 저장하고 graph payload는 gzip으로 무손실 압축한다. 로딩 tensor의 정확한 동등성과 SHA 위조 검출을 검사했다. 모델/graph/data/candidate 규모를 축소하지 않았다.
- 실제 원본 전체를 센 결과 inner-train eligible source 527개, T가 없는 outer-train 환자 24명이다. 이 24명도 모든 donor×128개 후보를 사용한다. 총 예정 graph **1,704,342개**. split의 26-connectivity 통계와 source eligibility의 기존 6-connectivity 통계는 서로 다른 목적이다.
- `attempt2/`는 완료된 원본 131명 SHA를 다시 검증하고 graph 준비부터 시작했다. 부분 cache/실패 결과를 완료된 cache로 재사용하지 않는다.
- nnU-Net train-only 105명 fingerprint/planning 완료. ResEncM 계획은 batch 2, patch 128³, spacing [1, 0.7578125, 0.7578125], 6 stages, channels 32/64/128/256/320/320이다. 131명 native 전처리는 `native/`에서 별도로 실행한다. 이 batch 2는 planner 결과이며 실제 augmentation을 포함한 최종 처리량 측정은 남아 있다.

## 실제 CT를 사용한 CUDA DEBUG

`tools/verify_v21_real_cuda_debug.py`로 inner-train의 liver_1/liver_5/liver_6에 대해 각 source의 전체 후보 128개를 생성한 뒤 원위치 T 1개와 U 1개씩, 총 6개 graph를 별도 DEBUG batch로 실행했다. 최종 학습 데이터의 subset 설정은 바꾸지 않았다.

BF16에서 실제 forward → loss → backward → AdamW update 1회 성공. 7,086,156 parameters, loss 3.203799247741699, 모든 gradient 유한값, 모든 핵심 모듈 갱신 확인. peak allocated VRAM 1,086,012,416 bytes, 실데이터 준비를 포함한 검사 시간 201.142초. checkpoint는 생성하지 않았다. 이 loss는 검증용이며 종양 Dice나 실험 성능 점수가 아니다.

최신 증거: `preflight/real_medical_cuda_DEBUG.json`, `preflight/storage_debug2/verification.json` 및 `tests.txt`, `preflight/storage_cuda_smoke.json`. 소스 SHA와 함께 `versions/v2/verification_storage_real_cuda_20260919/`에도 보존한다. 정적 검사 17개, 회귀 26/26, 실패·skip 0개.

## 남은 전체 규모 검증

전체 cache의 최종 디스크 용량·처리시간, 170만여 개 context를 유지하는 L1/L2의 실제 peak VRAM, 전체 graph의 physical batch/worker 측정은 아직 완료되지 않았다. 작은 DEBUG batch의 성공으로 이를 대신하지 않는다. GNN 40 epoch, frozen bank, native 250 epoch 학습과 전체 outer-val 의료 평가는 아직 완료되지 않았다. 오류가 있으면 해당 단계 실패 파일을 확인하고 규모를 줄이는 fallback 없이 수정해야 한다.

## 환경과 실행 계약

- GPU: RTX 5070 Ti 16,303 MiB 1개, compute capability 12.0, 드라이버 591.86. MIG 미지원. 초기 GPU 사용 메모리 약 1,397 MiB.
- CPU: 16 logical processors. RAM: 68,640,653,312 bytes. 초기 D: 여유 공간 약 1.90 TB.
- 전용 `.venv`: PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, PyG 2.6.1, nnU-Net 2.8.1. 버전 제약은 `config/runtime_windows_v21.txt`, 전체 설치 목록은 실행 루트의 `preflight/environment_freeze.txt`.
- 기존 v2.1 모델·그래프·후보 수·epoch 설정을 유지한다. outer fold 0, GNN 40 epoch, nnU-Net 후속 목표 250 epoch, source당 128개 후보. 실제 GNN batch와 loader worker 수는 전체 의료 그래프로 측정한다. DEBUG batch 4를 최종 batch로 지정하지 않는다.

## 데이터 출처와 분할

[MONAI 공식 데이터셋 코드](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/datasets.py)에 기록된 [Task03 Liver 전체 archive](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar)를 사용한다. 크기 28,925,891,584 bytes, 공식 MD5 `a90ec6c4aa7f6a3d087205e23d4e6397`.

다운로드는 `datasets/msd_liver/`에 보존한다. 전체 archive를 받아 검증·추출한다. 공개 정답이 있는 131명 전부로 nested split을 만들며 정답이 없는 public test 영상을 학습에 넣지 않는다. 이미지·label을 수정 없이 실행 루트의 `dataset/medical/Data/`에 복사하고 SHA, shape, affine, label 집합을 검증한다.

서버 split 파일의 로컬 사본은 발견하지 못했다. 기존 `tools.paired_benchmark`의 26-connectivity lesion profile, stratified `balanced_folds`, outer seed 270869, inner seed 1042로 fold 0을 재구성한다. **서버 split 및 CT/GT와 byte 단위 동일성은 미확인**이다. 따라서 과거 점수와 엄밀한 paired 비교로 보고하지 않는다. L0 source 적격성의 기존 6-connectivity는 변경하지 않는다.

## 사전 검증과 수정

첫 CUDA BF16 검사에서 `index_add_(): self (BFloat16) and source (Float)` 오류가 발생했다. L1 누적 버퍼를 weighted message의 승격 dtype에 맞춰 수정했다. 이전 모델과 테스트는 `versions/v2/pre_cuda_dtype_fix_20260919/`에 보존했다.

수정 후 실제 GPU full-width 합성 DEBUG 검사: parameter 7,086,156개(환자 3명 fixture), patch 48³, physical graph batch 4, BF16, optimizer 1 step, 모든 gradient 유한값, peak allocated VRAM 313,470,464 bytes. 전체 의료 그래프의 메모리·처리량을 대표하는 측정은 아니다.

CPU/CUDA 회귀 26/26 통과, 실패·skip 0개. nnU-Net private runtime의 v2.1 trainer import도 통과했다. import 성공을 native 전체 학습 검증으로 간주하지 않는다. 증거는 실행 루트의 `preflight/cuda_smoke.json`, `preflight/debug/verification.json`, `preflight/debug/tests.txt`, `preflight/native_api/import_result.json`에 있다.

## 실행과 진행 확인

순서: 전체 다운로드 검증 → 131명 데이터 검증·split → 모든 outer-train 환자의 L0 graph 준비 → 실제 graph batch/worker 측정 → GNN 40 epoch.

`tools/run_local_v21_gnn.py`는 위 데이터 검증부터 GNN 단계까지 연결한다. 새 실행 루트에서만 시작하며 실패하면 해당 로그와 실패 JSON을 남기고 다음 단계로 넘어가지 않는다. 부분 graph cache를 완성된 것으로 사용하거나 자동 resume하지 않는다. `--prepared-dataset`은 완료된 131명 원본을 모두 다시 SHA 검증해 재사용하는 옵션이며 graph cache/checkpoint resume가 아니다. 현재 GNN 단계 로그·출력은 아래 파일명 앞에 `attempt2/`를 붙여 확인한다. 원본 `dataset/`은 첫 실행 루트에 보존한다.

| 확인할 것 | 실행 루트 안의 파일 |
|---|---|
| 실행 시작·설정·GPU·PID | `pipeline_started.json` |
| 데이터 검증 진행 / 완료 | `dataset.log` / `dataset/dataset_ready.json` |
| 그래프 생성 진행 / 전체 완료 | `gnn_prepare.log` / `gnn_cache/index.json`의 `complete=true` |
| 측정 후 선택된 실제 batch·workers | `gnn_train/execution.json` |
| 실제 학습 epoch·loss·VRAM·처리량 | `gnn_train/metrics.csv`, `gnn_train.log` |
| GNN 40 epoch 완료 | `gnn_train/completion.json`, `gnn_train/model.pt` |
| 단계 실패 | `<stage>_failed.json`과 해당 `.log` |

nnU-Net 전처리는 GNN 준비와 독립적으로 먼저 진행한다. 로그는 `native_prepare.log`, `native/preprocess.log`, 완료 증거는 `native/native.json`이다. bank·250 epoch segmentation 학습·평가는 GNN 다음 단계다. 현재 GNN 실행기는 이 후속 학습 단계를 자동 실행하지 않는다. native 환경의 실제 batch/augmentation worker 처리량 측정과 end-to-end 검증도 남아 있다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 최종 batch/worker 선택은 실데이터 측정 후 기록한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. CPU admission 실패 후 mask 누적과 반복 저장을 수정했다. GPU DEBUG 오류는 OOM이 아닌 dtype 문제였다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 데이터나 결과로 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결을 실제 GPU 합성 DEBUG 및 별도 실데이터 DEBUG로 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
