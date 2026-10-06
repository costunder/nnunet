# v2.22 r3 실행 경로 재점검 — 2026-09-23

사용자가 CPU/RAM/GPU 저활용과 전체 구현의 신뢰성을 지적한 뒤 확인한 실제 결함과 수정이다. 초기44검사 성공을 전체 학습·online CP 검증으로 확대 해석하지 않는다.

## 확인하고 고친 문제

| 문제 | 원인과 영향 | 수정·검증 |
| --- | --- | --- |
| CPU worker1 선택 | 크기가 다른 case의 cases/s를 비교하여 작은 첫 case가 유리했다. 이후 전체 준비가 직렬로 실행됐다 | 같은 전체 case 묶음을 warm-up 후 worker1/2/4에서 반복 측정한다. 검사용 probe는 캐시를 게시하지 않으며 실제 전체 작업은 모두 한 번씩 실행한다. 준비 queue는 완료 즉시 보충 |
| 작은 GPU 타일 | 여유 VRAM의5%에32배 보수 계수를 적용하여 실행 타일이 지나치게 작아졌다 | 전체 그래프를 유지한 tile/batch 실측 후25%와4개의 FP32-equivalent workspace로 초기 실행 타일을 계산한다. 실제 peak VRAM과 full-support batch 보정은 계속 적용 |
| Scorer의 전역 상태 변경 | 같은 프로세스의 frozen GNN 초기화가 torch seed와 TF32/결정론 설정을 바꾸고, 대체될 모델 초기 가중치 생성도 난수를 소비했다 | 기존 공유 GPU lock 안에서 CPU/CUDA RNG와 backend/CUBLAS 환경을 저장·복원한다. 정상·예외·공개 scoring 진입 경로 검사 통과 |
| 비교 계약 검사 누락 | 기본값은42/128이지만 config의 seed·비교 표본 수 변경을 강제 차단하지 않았다 | GNN config의 seed42/128 및 nnU-Net 비교 seed42를 검사하고 다른 값은 실행 전에 오류 처리 |

모델 학습 알고리즘·CNN/L1/L2 깊이와 너비를 바꾸지 않았다. `PromptGraphModel`과 GAT 메시지 계산식은 이전 AST와 동일하다. L0 실행 메모리 타일과 runtime isolation만 보완했다. 전체45,385노드/2,383,482엣지를 유지하고 노드·샘플·epoch를 줄이지 않았다.

## 실제 자원 측정

같은 CT4개(liver_0/43/86/130)를 각 worker 후보에서 처리했다. 이는 준비 처리량 보정이며 training subset이 아니다. 메모리 admission은 최대 case의 전체 CT/GT/임시 배열 상한6,209,667,072bytes와 가용 RAM의50% 여유를 기준으로 한다. 현재 그 조건에서 검사한 worker 후보는1/2/4다.4가 전역 최적값이라고 단정하지 않는다.

| CPU worker | 같은4case 처리량(case/s) | peak process RSS(bytes) |
| --- | --- | --- |
|1|0.193260|3,671,027,712|
|2|0.321230|5,442,449,408|
|4|0.348162|5,430,329,344|

현재 full-run은4worker로131case inventory를 처리한다. 실행 중 측정은 CPU398.4%(한 코어100% 기준), RSS6,240,301,056bytes였다.64GiB RAM을 모두 채우는 것을 완료 조건으로 삼지 않으며 현재 메모리 상한과 실제 측정을 구분한다.

GPU DEBUG는 같은 실제 CT query를 반복하여 자원을 비교했다. 전체 물리 그래프와1,519,063파라미터를 사용한다. 초기4batch/128tile은 약0.89graphs/s였다. 수동 tile sweep의 최고는batch16/tile2048에서3.18graphs/s다. 이후 수정된 기본 경로를 별도로 검사한 값은 아래와 같다.

| 실제 기본 코드의 physical/effective batch | 실행 tile | 처리량(graphs/s) | peak CUDA 할당(bytes) |
| --- | --- | --- | --- |
|4 / 4|4096|2.696197|2797319680|
|8 / 8|2048|3.013626|3805808128|
|16 / 16|1024|3.127862|7471685632|
|32 / 32|512|2.191564|14788444160|

Batch32는 VRAM을 더 쓰면서 느려졌다. 최종 physical batch는 이3-case DEBUG 값을 고정하지 않고 전체 support/query를 사용하는 production calibration에서 다시 선택한다. 두 관측 클래스당 support2case인 DEBUG이므로 큰 L1 support의 성능과 임상 효용을 검증한 수치가 아니다. Gradient accumulation은1을 유지한다.

## 데이터·기능 경로 검사 결과

- 전체 원본 CT/GT131case 해시 검증 완료. 중복 decoded CT 없음.
- Outer-train105case 모두 비교 표본128개를 생성할 수 있다. 가장 작은 후보 집합도421,735개다. Outer-val26case는 비교 표본 생성에서 제외한다.
- Inner-train84case의 적격 양성527개와 inner-val21case의135개를 유지한다. 전체 patch 예정14,102개 중 train11,279 / inner-val2,823이다.
- `liver_108`은 새 조건에서3,402,279개 중128개 선택한다. 원본 GT나 병변 수를 임의로 고치지 않았다.
- 단위·회귀89개 통과: 확장84검사 중83개가 처음 통과했고 git safe.directory 접근으로 실패한1건은 저장소 범위의 임시 설정으로 재검사하여 통과했다. RNG/backend/seed 관련5개를 추가 통과했다. 전역 git 설정은 변경하지 않았다.
- 실제 CT의 중심·주변 정보가 입력과 L0 특징에 전달되고 모든 파라미터의 gradient가 유한함을 확인했다. CNN/L0/L1/L2 optimizer 갱신은 별도 real-CT DEBUG에서 확인했다.
- Synthetic donor/bank/native-transport 검사는 원본 CT 전체 online CP 실행과 구분한다. 임상 성능으로 보고하지 않는다.
- Frozen v1 검증 통과. r3 전 코드와 이번 runtime 수정 전 코드는 각각 별도 snapshot에 보존했다.

## 현재 실제 실행 상태

**후속 수정:** 아래 PID6256 실행은 이후 VRAM 보정 오류로 중단했다. CT 캐시는 완료·보존했고 현재는 `gnn_vram_affine`에서 재실행 중이다. [VRAM 오류 두 가지와 시각화 검증](v222_visual_runtime_verification_20260923.md)이 최신 기록이며 아래 문단은 당시 실행 이력이다.

새 실행은 `work/v222_raw_ct_r3_training_20260923/`다. Wrapper PID23268, Python PID6256으로 시작했다. 전체105case 입력 준비 → 전체 GNN40epoch 순서다. 실행 로그의 stage가`context_preparation`인 동안 optimizer 학습이 시작됐다고 보고하지 않는다. 이후 batch/worker 보정과 support memory 작성도 optimization과 구분한다.

- [실행 stdout](../work/v222_raw_ct_r3_20260923/full_training.stdout.log)
- [실행 stderr](../work/v222_raw_ct_r3_20260923/full_training.stderr.log)
- [전체 cohort 검증](../work/v222_raw_ct_r3_20260923/preflight/summary.json)
- [실제 기본 GPU 실행 측정](../work/v222_raw_ct_r3_20260923/production_execution_benchmark/result.json)
- [최종 코드·데이터 검증 receipt](../work/v222_raw_ct_r3_20260923/review_verification.json)
- [89검사 중 확장84검사 로그](../work/v222_raw_ct_r3_20260923/runtime_review_tests.log)
- [접근 오류1건 재검사 및 추가5검사](../work/v222_raw_ct_r3_20260923/runtime_final_tests.log)

## 아직 검증하지 못한 부분

전체 GNN40epoch 수렴, 전체 nnU-Net250epoch online CP 실행, 전체 평가와 Basic CP80 대비 이득은 아직 검증되지 않았다. 관측 분류가 실제로 좋은 붙여넣기 위치를 선택하는지도 전체 실험과 중심/주변 개입 진단이 필요하다. 공개 case ID만으로 재검사 환자 독립성을 보장할 수 없고 원본 주석의 누락 가능성도 유지한다. Donor·크기·crop 차이가 남은 전체 방법 비교다. 이 기록은 전체 코드에 오류가 없다는 보증이 아니다.

## 작업 완료 체크리스트


- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 문제에 그래프 축소 없이 기존 tiling/checkpointing 경로를 유지했다. 새 DEBUG에서 OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
