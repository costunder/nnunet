# v2.22 공개 case 기준 학습 실행 (2026-09-22)

**최신 상태: context 준비 실패.** `liver_108`에서 고정 가림 반경 기반 거리 조건을 만족하는 비교 중심이0개여서128개를 만들지 못했다. `failed.json`과 stderr에 보존했다. GNN optimizer 및 nnU-Net 학습은 시작하지 않았다. 아래 착수 설명은 이 실패 전의 기록이다. Case 제외나 조건 완화를 적용하지 않았다.

사용자의 “이렇게 학습하게 해” 지시에 따라 `work/v222_train_20260922/case_benchmark_run1/`에 전체 context 준비 → GNN 40 epoch 실행을 시작했다. 실행 로그는 같은 상위 폴더의 `case_benchmark_run1.stdout.log`와 `.stderr.log`다. Wrapper PID와 명령은 `.launch.json`에 기록했다. 이 문서의 시작 시점 상태는 **context 준비 중**이며 optimizer update가 시작됐다는 뜻이 아니다.

## CP 50%의 의미와 근거

CP 시도 확률0.5는 기존 프로젝트에서 이어진 설정이다. 최적값으로 검증됐다는 실험 근거는 없다. nnU-Net 각 학습 샘플 방문에서 CP를 시도할 확률이며, 유효 배치를 만들지 못하면 실제 적용률은 더 낮을 수 있다. 데이터셋을 절반만 사용한다는 뜻이 아니며 GNN 관계 학습에 50% CP를 적용하는 것도 아니다. 원본 Basic CP는 매 방문마다 시도한다. 이번에는 사용자 승인대로 v2.22의 0.5를 유지했고 원본 Basic CP는 변경하지 않았다.

## 고정 조건

- 로컬 전체131 CT; outertrain105/outerval26. GNN innertrain84/innerval21. Outer validation은 GNN patch cache에 넣지 않는다.
- GNN seed42, 40epoch, 모든 training query를 매 epoch 사용한다. 최종 nnU-Net 계획은 seed42/250epoch/ResEncM/patch128³/batch2이며 CP0.5다. Native worker 수는 실제 처리량 측정 후 Basic CP와 맞춘다.
- CNN/L0/L1/L2 깊이·폭 및 전체 그래프 규칙은 보존한다. 군집 정렬 r2 모델1,519,063parameters. 현재 전체 데이터에서 가림 반경27.290655635700848mm, 그래프39,936nodes/2,040,356edges다.
- train 비교 중심은 case당 기존128개이며 적격 종양527개 전부를 포함한다. 예상 GNN train11,279patches, validation2,823patches. 이 수는 cache의 실제 레코드와 대조한다.

## 환자 정보와 주석을 사실대로 기록

기존 `hiercp_patient_identity_v1`은 확인된 환자 정보와 완전한 주석을 요구했다. 공개 case ID만 있는 이번 데이터에 이를 True로 꾸며 넣지 않았다. 별도 `hiercp_public_case_benchmark_v1`은 `patient_independence_verified=false`, `annotation_complete=null`, `patient_group=case:<ID>`를 명시한다. 정답0은 제공된 마스크에 종양이 관측되지 않은 비교 위치이며 생물학적으로 종양이 절대 없다는 증명이 아니다.

다른 case 사이에 동일 환자가 없다는 보장은 없다. CT 내용 중복 검사는 전체131개에 수행했으며 이미 알려진 case의 split 혼합과 train/validation 입력 혼합은 계속 차단한다. 기존 verified-patient 계약을 사용하면 원래의 엄격한 검사가 그대로 적용된다. 새 benchmark 형식에서 verified/complete를 주장하거나 다른 case의 group을 재사용하면 오류다.

## 실행 및 성능 측정

전체 cache 준비 후 L0 full-graph inference batch 후보를 실측하고, 전체 cache 로딩으로 worker0/2/4/8을 비교한다. 선택된 값으로 전체 support memory를 만든 다음 실제 L0/L1/L2 forward/backward batch 보정을 수행한다. 추론 batch와 학습 physical batch는 따로 기록한다. 후보 OOM은 해당 batch 측정 실패로 기록하며 모델·그래프·입력 해상도·데이터를 줄이지 않는다.

이전에는 첫 support memory를 batch2/worker0으로 전부 만든 뒤 보정을 시작했다. 이번에는 측정을 먼저 수행한다. 직전 validation에 사용한 최신 memory를 다음 epoch 시작에 재사용하여, optimizer update 없이 동일 전체 CT를 다시 인코딩하는 중복만 제거했다. 매 epoch 종료 후 최신 가중치로 전체 memory 갱신은 유지한다.

초기 자원 확인: RTX5070Ti16GB, VRAM14866MiB free, CPU16logical/RAM64GiB. 준비 단계의 실측 RAM/CPU/worker 기록은 `context/inventory_resources.json`, `context/patch_resources.json`, GNN의 batch·worker·VRAM 측정은 `gnn/*calibration.json`에 생성된다.

학습 시작 helper는 이번 **GNN 단계까지만** 실행한다. 이후 bank 생성·native 처리량 보정·nnU-Net 학습이 필요하며, 이 단계까지 이미 실행됐다고 기록하지 않는다. `gnn/training_started.json`은 보정 완료 후 생성되고, `stage=optimization` 로그가 첫 optimizer update 증거다. 최종 완료는 `gnn/training_complete.json` 및 `gnn_complete.json`으로 구분한다.

## 검증

Case scope 4개를 포함한 v2.2~v2.22 회귀43개와 변경 Python 정적 검사를 통과했다. 이전 source snapshot은 `versions/v2.22/before_case_benchmark_training_20260922/`에 보존했다. 첫 정적 검사에서 helper의 괄호 오타를 발견해 실행 전에 수정했다. 학습 source는 실행 이후 바꾸지 않으며 docs/export 갱신만 한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토하고 실측 보정 경로를 연결했다. 최종 실측값은 실행 중 기록한다.
- [x] GPU, CPU, RAM 자원을 확인했다. 사용률·peak는 실행 기록으로 확인한다.
- [x] OOM 시 모델 축소 없이 물리 batch 측정 실패를 기록하도록 했다.
- [x] DEBUG 검사와 전체 데이터 실행을 분리했다.
- [x] dummy, placeholder, random fallback을 production에 사용하지 않았다.
- [x] 핵심 모듈의 forward/loss/gradient/optimizer 회귀 검사를 통과했다.
- [x] 설정과 변경 사항, case 단위 독립성 한계를 명확하게 기록했다.
- [x] 정적·DEBUG 검사, 준비, 실제 학습, 전체 완료를 구분해서 보고했다.
