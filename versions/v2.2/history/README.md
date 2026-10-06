# v2.2 이력

과거 snapshot과 검증 기록을 세 묶음으로 정리했습니다. `early`는 초기 CT·CNN·L2 변경, `relations`는 T/F/U 관계 계약, `observed`는 관측 과제와 실행 변경 이력입니다. 옛 `v2.21`, `v2.22` 폴더를 별도 현재 실험으로 취급하지 않습니다.

34개 폴더를 실제 이동했고, 156개 파일의 내용·크기·SHA256과 내부 디렉터리 목록을 전후 대조했습니다. ZIP member와 manifest 내용도 바꾸지 않았습니다. 23개 ZIP의 CRC와 manifest에 결속된 582개 member도 확인했습니다. 이동한 전체 파일 목록은 [moves.json](../../moves.json), 검증 결과는 [history.json](../../../validation/version_layout/history.json)과 [archives.json](../../../validation/version_layout/archives.json)에 있습니다. 원본 기록 안의 예전 경로는 당시 출처로 남겨두었습니다.

현재 실행 안내는 [버전 목록](../../README.md)을 따릅니다. 아래 파일은 보존 이력이며 새 학습 결과가 아닙니다.

| 묶음 | 새 폴더 | 이전 폴더 이름 |
| --- | --- | --- |
| early | [before-cnn](early/before-cnn) | before_cnn_only_r3_20260922 |
| early | [before-l2](early/before-l2) | before_l2_restore_r4_20260922 |
| early | [before-features](early/before-features) | before_physical_features_r2_20260920 |
| early | [before-no-interior](early/before-no-interior) | before_retired_interior_removal_r5_20260922 |
| early | [before-relations](early/before-relations) | before_v221_20260922 |
| early | [cnn-l0](early/cnn-l0) | cnn_l0_20260924 |
| early | [checks](early/checks) | verification_20260920 |
| early | [checks-cnn](early/checks-cnn) | verification_cnn_only_20260922 |
| early | [checks-l2](early/checks-l2) | verification_l2_restore_20260922 |
| early | [checks-no-interior](early/checks-no-interior) | verification_no_interior_20260922 |
| relations | [before-observed](relations/before-observed) | before_v222_20260922 |
| relations | [checks](relations/checks) | relation_contract_verification_20260922 |
| observed | [before-cases](observed/before-cases) | before_case_benchmark_training_20260922 |
| observed | [before-centers](observed/before-centers) | before_center_fix_20260923 |
| observed | [before-clusters](observed/before-clusters) | before_cluster_alignment_r2_20260922 |
| observed | [before-seed](observed/before-seed) | before_comparison_seed_fix_20260922 |
| observed | [before-cp80](observed/before-cp80) | before_cp80_comparison_20260922 |
| observed | [before-sampling](observed/before-sampling) | before_ppr_astar_r4_20260923 |
| observed | [before-raw-ct](observed/before-raw-ct) | before_raw_ct_r3_20260923 |
| observed | [before-runtime](observed/before-runtime) | before_runtime_review_20260923 |
| observed | [before-v1-l0](observed/before-v1-l0) | before_v1_l0_apply_20260924 |
| observed | [before-vram](observed/before-vram) | before_vram_admission_20260923 |
| observed | [before-ezsp](observed/before-ezsp) | ezsp_before_diagnostics_20260929 |
| observed | [v1-runtime](observed/v1-runtime) | v1_execution_r6_20260924 |
| observed | [v1-train](observed/v1-train) | v1_full_training_20260924 |
| observed | [v1-empty-context](observed/v1-empty-context) | v1_full_training_empty_context_20260924 |
| observed | [v1-l0](observed/v1-l0) | v1_l0_20260924 |
| observed | [checks](observed/checks) | verification_20260922 |
| observed | [checks-cases](observed/checks-cases) | verification_case_benchmark_training_20260922 |
| observed | [checks-clusters](observed/checks-clusters) | verification_cluster_r2_20260922 |
| observed | [checks-seed](observed/checks-seed) | verification_comparison_seed_20260922 |
| observed | [checks-cp80](observed/checks-cp80) | verification_cp80_20260922 |
| observed | [checks-preflight](observed/checks-preflight) | verification_preflight_parallel_20260923 |
| observed | [checks-resume](observed/checks-resume) | verification_resume_20260923 |

초기 이력의 이전 부모는 `versions/v2.2`, 관계 계약은 `versions/v2.21`, 관측 과제는 `versions/v2.22`였습니다. `moves.json`에는 이전·현재 전체 경로가 함께 있습니다. 세 보조 스크립트의 archive 위치만 갱신했고, 현재 학습의 import package·config·실험 snapshot·checkpoint 경로는 유지했습니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 계약을 변경하지 않았다. 이번 실행은 파일 정리뿐이다.
- [x] 파일 수와 저장량을 확인했다. 새 GPU·학습 CPU/RAM 측정은 이번 작업 범위가 아니다.
- [x] OOM 회피나 모델 축소를 수행하지 않았다.
- [x] 정리 검증과 실제 학습 결과를 구분했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 기존 모델의 forward, loss, gradient, optimizer 코드를 수정하지 않았다.
- [x] 실제 이동 경로와 보존 hash를 기록했다.
- [x] 전체 학습·전체 평가를 이번 작업에서 실행하지 않았음을 명시했다.
