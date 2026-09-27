# v2.2 observed_rank_v1 검토 후 실제 수정 — 2026-09-27

기준 revision은 `94596b9f6945015dfa14f3721d6763ce262d4a9d`이다. 세 첨부 검토문 전체를 읽은 기록은 [검토 접수 문서](v22_review_intake_20260927.md)에 남겼다. 이번 변경은 그 이후의 코드 수정이다. 첨부되지 않은 별도 ZIP·공식 60항목 acceptance 파일까지 검사했다고 주장하지 않는다.

## 수정 내용과 경계

| 항목 | 코드 변경 | 검증 범위 |
|---|---|---|
| F01 calibration list 오류 | list형 calibration JSON을 dict로 변환하지 않고 보존. dict에만 의미 계약을 붙이고 충돌은 오류 | 단위 검사 및 실제 process fresh calibration |
| F02 donor anchor 이동 | spacing 변환 후 occupied voxel 상대 좌표를 그대로 보존하는 padding과 explicit anchor. shape 중간점으로 anchor를 재해석하지 않음 | 홀수·짝수·비등방성·변환 단위 검사, 실제 CT 전체 그래프 재생성 |
| F03 동점 순위 | GT·입력 순서와 무관한 case/donor/component/좌표 키로 단일 순서를 만들고 지표와 추천이 공유 | 여러 양성 동점과 순열 검사 |
| F04 외부 best 파일 의존 | rolling checkpoint 최상위에 불변 CPU best 가중치·epoch·metric·run identity·hash 저장. CUDA로 이동되는 state에는 메타데이터만 저장 | CPU snapshot alias/hash 검사, 실제 저장·재개 |
| F05 artifact 검증 | resume/epoch/final 유형 분리. format/runtime/source/geometry/완료 epoch, support owner·donor·group·coverage·유한값·선택 모델 결속 검사 | 누락·오염 단위 검사, 실제 final 생성 |
| F06 query identity | manifest의 recipient group과 query_group 일치, donor inner-train 및 동일 환자 제외 검사 | 실제 추천 경로의 잘못된 group 거부 검사 |
| F07 동일 placement | CT·mask·anchor·좌표계·변환을 PlacementSpec에 묶고 graph metadata와 비교. 전체 true voxel filter 및 raw paste가 공유 | 회전·scale·mirror·경계 및 실제 CT 검사. **native bank/RPC/nnU-Net 최종 입력은 미연결** |
| F08 물리 좌표 | CT/GT shape·affine·finite spacing을 확인. CT voxel 기준 8개 모서리 오차 검사 | 실제 131/131 CT header 통과 |
| F09 정상 빈 후보 | 빈 proposal과 전부 탈락은 keep_original. NaN/identity/placement 오류는 예외 | API 단위 검사. **native trainer RNG/no-op 종단 검증은 미완료** |

물리 좌표 허용 오차는 `1e-4 CT voxel`이다. 최초 절대 mm 오차 기준에서는 NIfTI 헤더 정밀도 차이 9건이 걸렸고, 전체 코호트의 최대 모서리 오차는 `5.355902703740867e-05 voxel`이었다. 데이터를 옮기거나 resampling해서 맞춘 것이 아니다. 비유한 affine, 실제 위치·spacing 불일치는 계속 거부한다.

Windows venv redirector가 자신의 학습 프로세스를 중복 실행으로 오인하는 오류도 실제 진입점 실행 중 확인하여 수정했다. 직접 부모 PID, Python home, argv, cwd 및 동일 output이 모두 맞는 현재 redirector만 제외한다. 다른 실행이나 프로세스를 종료하지 않는다.

## 바꾸지 않은 연구 계약

모델 5,550,806 parameters, hidden128, L0/L1/L2 깊이와 그래프 규칙, 원래 전체 split·donor 배정·40epochs·seed42·CP80% 설정은 축소하지 않았다. 관측 CE와 L2, 현재 pairwise estimator도 유지했다. pair 중복 제거, loss/2, reference L1/L2 detach를 적용하지 않았다. 새 pair_estimator_audit는 실제 배치의 pair 빈도와 scalar 계수를 기록할 뿐 loss를 바꾸지 않는다.

CE/rank gradient 상충 여부, donor 무시·중심 종양 shortcut, fixed-donor 효용, 전체 support에서 최대 메모리 배치, CP 대조군 성능은 여전히 별도 검증이 필요하다. 특히 한 case의 전체 edge 합 최대가 모든 학습 배치의 최대 VRAM이라는 보장은 없다. 이번 DEBUG 수치를 A100 MIG 전체 코호트 성능으로 대체하지 않는다.

## 실행 증거와 실패 기록

`work/v22_review_repair_20260927_DEBUG` 아래에 실패한 실행과 성공 실행을 모두 별도 폴더로 보존했다.

- `cohort_grid_audit_r2.json`: 실제 131 CT/GT 물리 header 검사. 영상 전체 voxel 검사나 모든 graph 재생성을 의미하지 않는다.
- `cache_r3`: 실제 train8/validation2 관측, 원래 전체 그래프를 재생성한 명시적 DEBUG 자료. 4개 train case에서 donor 양쪽 제외 후에도 두 클래스가 남도록 기존 배정 행을 선택했다. 원래 코호트나 donor 배정 자체는 수정하지 않았다.
- `paused_r3`: production process 진입점 + 명시적 review_repair profile. 실제 worker0/2/4/8 경로, memory/training calibration list 저장, 첫 optimizer update, durable checkpoint 중단.
- `resumed_r3`: step1에서 재개하여 총4 updates, 8/8 query 방문, validation, CPU best snapshot, selected-model final support 및 final artifact 저장 완료.
- `portable_r3`: rolling 파일 하나만 복사하고 외부 best 파일 읽기를 강제로 거부한 상태에서 실제 process 재개 통과. selected weights와 final support가 원래 완료 결과와 bitwise 일치하고 run identity 유지.
- `reuse_r3`: 저장된 loader/memory/training calibration JSON 재사용 경로로 첫 update 및 durable pause 통과.
- `selection_r4`: 실제 CT·1 voxel donor에 대해 원본 반복 대조군과 GT 변경 대조군 raw score/rank bitwise 일치. 실제 raw paste 1 voxel 귀속 일치.
- `selection_multivoxel`: 실제 donor의 2,368 voxel 전체 footprint 사용. 관측 후보의 종양 겹침 35 voxel, 다른 후보의 간 coverage 부족을 각각 확인하여 정상 no-op. 종양 GT만 제거한 대조군에서는 같은 score/rank를 유지하면서 eligibility만 바뀜. 이 대조군은 annotation 변조 실험이며 원래 의료 정답을 대체하지 않는다.
- `first_gradient_check.json`: 모든 parameter gradient 존재·유한, ranking loss의 L0 gradient 비영.
- DEBUG는 full model이지만 physical batch2, support8, validation2, 1epoch이다. production 설정을 덮어쓰지 않았다. 이 표본의 순위 점수는 연구 성능으로 제출하지 않는다.
- 초기 `cache`는 지나치게 엄격한 affine 기준에서 중단되었다. `cache_r2`의 DEBUG 행 구성은 양쪽 환자 제외 후 한 클래스가 사라져 `resumed`에서 중단되었다. 검사 완화 없이 DEBUG 행 구성을 수정했다.
- 첫 `selection_r3` 검사는 deterministic CUDA 설정을 누락하여 raw score 완전 일치 assertion에서 실패했다. 학습과 같은 runtime 설정 및 변경 없는 입력의 반복 대조군을 추가한 `selection_r4`와 `selection_multivoxel`에서는 완전 일치를 확인했다. 추천 API 사용자는 학습과 같은 runtime 정책을 설정해야 하며, API가 호출자의 RNG를 임의로 재설정하지 않는다.

최종 집계와 각 실행의 SHA는 `validation/v222_r6/review_repairs_20260927_DEBUG.json`에 기록한다. 원래 revision의 physical32 및 전체 support 측정은 이번 수정 revision의 검증 증거가 아니다.

관련 단위·회귀 검사 **64개가 통과**했고 `git diff --check`도 통과했다. fresh calibration과 reuse calibration의 첫 update 가중치·Adam state·support가 bitwise 일치했다. 보존된 v1 manifest 검증도 통과했다. 단위 검사 속도를 학습 처리량으로 보고하지 않는다.

## 캐시와 checkpoint 이전

`paired_explicit_anchor_symmetric_padding_v2`와 `observed_rank_artifact_v2`를 도입했다. 기존 raw CT/GT는 그대로 쓴다. 이전 geometry graph/cache와 기존 artifact를 이름이나 provenance만 바꿔 새 의미로 재사용하면 안 된다. 새 graph를 만들고 새 학습으로 시작해야 한다. 이전 결과·가중치·코드는 비교와 보존을 위해 남긴다. 같은 새 계약의 rolling checkpoint는 저장한 optimizer/RNG/cursor와 함께 재개한다.

## 남아 있는 단계

- G1: 실행·기하·순위·artifact/API 수정과 단위 검사를 수행했다.
- G2: 실제 CT, full model, 실제 process DEBUG 저장·재개·완료를 확인했다.
- G3: 새 geometry 전체 14,102 graph 및 11,279 support의 실제 full-scale calibration은 미실행.
- G4: 새 rank API와 native bank/RPC/adapter/trainer, nnU-Net transpose/crop/resample/최종 입력 연결은 미완료. raw paste API가 이를 대체하지 않는다.
- G5: 전체40epoch·nnU-Net 비교·CP 임상/분할 성능은 미실행.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. DEBUG 표본은 별도 경로다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번 실행은 명시적 DEBUG 범위다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. 이번 수정 검증에서는 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 production에 사용하지 않았다. 단위 검사의 synthetic 입력은 명시했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 full-model DEBUG update 검사.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
