# v1 파일 안내

원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.
게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.

<details><summary>구현 (55)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
|   init   | [열기](../../hiercp/__init__.py) | 게시 대상 |
| cache | [열기](../../hiercp/cache.py) | 게시 대상 |
| cache migration | [열기](../../hiercp/cache_migration.py) | 게시 대상 |
| causality overlap | [열기](../../hiercp/causality_overlap.py) | 게시 대상 |
| common | [열기](../../hiercp/common.py) | 게시 대상 |
| contracts | [열기](../../hiercp/contracts.py) | 게시 대상 |
| curriculum | [열기](../../hiercp/curriculum.py) | 게시 대상 |
| data | [열기](../../hiercp/data.py) | 게시 대상 |
| donor preflight | [열기](../../hiercp/donor_preflight.py) | 게시 대상 |
| feedback | [열기](../../hiercp/feedback.py) | 게시 대상 |
| feedback calibration | [열기](../../hiercp/feedback_calibration.py) | 게시 대상 |
| feedback patient | [열기](../../hiercp/feedback_patient.py) | 게시 대상 |
| feedback resources | [열기](../../hiercp/feedback_resources.py) | 게시 대상 |
| feedback storage | [열기](../../hiercp/feedback_storage.py) | 게시 대상 |
| geometry | [열기](../../hiercp/geometry.py) | 게시 대상 |
| hierarchy | [열기](../../hiercp/hierarchy.py) | 게시 대상 |
| local | [열기](../../hiercp/local.py) | 게시 대상 |
| loss | [열기](../../hiercp/loss.py) | 게시 대상 |
| model | [열기](../../hiercp/model.py) | 게시 대상 |
| nifti geometry | [열기](../../hiercp/nifti_geometry.py) | 게시 대상 |
| pipeline | [열기](../../hiercp/pipeline.py) | 게시 대상 |
| preparation runtime | [열기](../../hiercp/preparation_runtime.py) | 게시 대상 |
| prototype | [열기](../../hiercp/prototype.py) | 게시 대상 |
| region | [열기](../../hiercp/region.py) | 게시 대상 |
| sample | [열기](../../hiercp/sample.py) | 게시 대상 |
| schema | [열기](../../hiercp/schema.py) | 게시 대상 |
| spatial | [열기](../../hiercp/spatial.py) | 게시 대상 |
| split | [열기](../../hiercp/split.py) | 게시 대상 |
| tensor | [열기](../../hiercp/tensor.py) | 게시 대상 |
| training resources | [열기](../../hiercp/training_resources.py) | 게시 대상 |
|   init   | [열기](../../hiercp_v1x/__init__.py) | 게시 대상 |
| budget recovery entry | [열기](../../hiercp_v1x/budget_recovery_entry.py) | 게시 대상 |
| cache budget recovery | [열기](../../hiercp_v1x/cache_budget_recovery.py) | 게시 대상 |
| competition analysis | [열기](../../hiercp_v1x/competition_analysis.py) | 게시 대상 |
| contracts | [열기](../../hiercp_v1x/contracts.py) | 게시 대상 |
| epoch telemetry | [열기](../../hiercp_v1x/epoch_telemetry.py) | 게시 대상 |
| experiment | [열기](../../hiercp_v1x/experiment.py) | 게시 대상 |
| graph flow audit | [열기](../../hiercp_v1x/graph_flow_audit.py) | 로컬 |
| graph size | [열기](../../hiercp_v1x/graph_size.py) | 게시 대상 |
| group search | [열기](../../hiercp_v1x/group_search.py) | 로컬 |
| models | [열기](../../hiercp_v1x/models.py) | 게시 대상 |
| native30 checkpoint | [열기](../../hiercp_v1x/native30_checkpoint.py) | 게시 대상 |
| native30 data contract | [열기](../../hiercp_v1x/native30_data_contract.py) | 게시 대상 |
| native30 geometry | [열기](../../hiercp_v1x/native30_geometry.py) | 게시 대상 |
| native30 upper | [열기](../../hiercp_v1x/native30_upper.py) | 게시 대상 |
| nested execution | [열기](../../hiercp_v1x/nested_execution.py) | 로컬 |
| nested graph size | [열기](../../hiercp_v1x/nested_graph_size.py) | 게시 대상 |
| results | [열기](../../hiercp_v1x/results.py) | 게시 대상 |
| roi budget probe | [열기](../../hiercp_v1x/roi_budget_probe.py) | 게시 대상 |
| roi failure replay | [열기](../../hiercp_v1x/roi_failure_replay.py) | 게시 대상 |
| roi probe reuse | [열기](../../hiercp_v1x/roi_probe_reuse.py) | 게시 대상 |
| sampling entry | [열기](../../hiercp_v1x/sampling_entry.py) | 게시 대상 |
| sampling runtime | [열기](../../hiercp_v1x/sampling_runtime.py) | 게시 대상 |
| snapshot inventory | [열기](../../hiercp_v1x/snapshot_inventory.py) | 게시 대상 |
| telemetry entry | [열기](../../hiercp_v1x/telemetry_entry.py) | 게시 대상 |

</details>

<details><summary>설정 (1)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| train | [열기](../../config/train.json) | 게시 대상 |

</details>

<details><summary>실행·분석 (38)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| run | [열기](../../run.py) | 게시 대상 |
| run v1 | [열기](../../run_v1.py) | 게시 대상 |
| audit v1 empty context | [열기](../../tools/audit_v1_empty_context.py) | 로컬 |
| audit v1 pair resampling | [열기](../../tools/audit_v1_pair_resampling.py) | 로컬 |
| build v1 graph first review | [열기](../../tools/build_v1_graph_first_review.py) | 로컬 |
| build v1 nested graph review | [열기](../../tools/build_v1_nested_graph_review.py) | 로컬 |
| build v1 portfolio | [열기](../../tools/build_v1_portfolio_20260930.py) | 로컬 |
| build v1 runtime416 review | [열기](../../tools/build_v1_runtime416_review.py) | 로컬 |
| build v1x review bundle | [열기](../../tools/build_v1x_review_bundle.py) | 로컬 |
| v1 empty context | [열기](../../tools/check_v1_empty_context.py) | 로컬 |
| v1 recovery | [열기](../../tools/check_v1_recovery.py) | 로컬 |
| v1 visual | [열기](../../tools/check_v1_visual.cjs) | 로컬 |
| v1 visual data | [열기](../../tools/check_v1_visual_data.py) | 로컬 |
| evaluate native v1 full128 | [열기](../../tools/evaluate_native_v1_full128.py) | 게시 대상 |
| evaluate v1 sampling pair | [열기](../../tools/evaluate_v1_sampling_pair.py) | 게시 대상 |
| freeze v1 training | [열기](../../tools/freeze_v1_training.py) | 게시 대상 |
| inspect native v1 checkpoint | [열기](../../tools/inspect_native_v1_checkpoint.py) | 게시 대상 |
| migrate v1 execution cache | [열기](../../tools/migrate_v1_execution_cache.py) | 게시 대상 |
| probe v1 roi budget | [열기](../../tools/probe_v1_roi_budget.py) | 게시 대상 |
| profile v1 execution | [열기](../../tools/profile_v1_execution.py) | 게시 대상 |
| profile v1 pair preparation | [열기](../../tools/profile_v1_pair_preparation.py) | 로컬 |
| retire old v1 graphs | [열기](../../tools/retire_old_v1_graphs.py) | 게시 대상 |
| run v1 native nested416 server | [열기](../../tools/run_v1_native_nested416_server.sh) | 게시 대상 |
| run v1x experiment | [열기](../../tools/run_v1x_experiment.py) | 게시 대상 |
| server native v1 full128 | [열기](../../tools/server_native_v1_full128.sh) | 게시 대상 |
| smoke v1 execution lifecycle | [열기](../../tools/smoke_v1_execution_lifecycle.py) | 게시 대상 |
| smoke v1 memory limit | [열기](../../tools/smoke_v1_memory_limit.py) | 게시 대상 |
| smoke v1 paired training | [열기](../../tools/smoke_v1_paired_training.py) | 로컬 |
| supervise v1 execution | [열기](../../tools/supervise_v1_execution.py) | 로컬 |
| supervise v1 training | [열기](../../tools/supervise_v1_training.py) | 로컬 |
| v1 server | [열기](../../tools/v1_server.py) | 게시 대상 |
| native30 full128 cuda debug | [열기](../../tools/verify_native30_full128_cuda_debug.py) | 게시 대상 |
| v1 epoch telemetry cuda debug | [열기](../../tools/verify_v1_epoch_telemetry_cuda_debug.py) | 게시 대상 |
| v1 graph size debug | [열기](../../tools/verify_v1_graph_size_debug.py) | 로컬 |
| v1 nested graph size debug | [열기](../../tools/verify_v1_nested_graph_size_debug.py) | 로컬 |
| v1 resume | [열기](../../tools/verify_v1_resume.py) | 게시 대상 |
| v1 sampling runtime debug | [열기](../../tools/verify_v1_sampling_runtime_debug.py) | 로컬 |
| v1x cuda debug | [열기](../../tools/verify_v1x_cuda_debug.py) | 로컬 |

</details>

<details><summary>기록 (11)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| native30 all lesions geometry reuse | [열기](../../docs/native30_all_lesions_geometry_reuse_20261006.md) | 게시 대상 |
| native v1 30mm full128 evaluation | [열기](../../docs/native_v1_30mm_full128_evaluation_20261006.md) | 게시 대상 |
| v1 bundle standalone review | [열기](../../docs/v1_bundle_standalone_review_20261003.md) | 게시 대상 |
| v1 current comparison | [열기](../../docs/v1_current_comparison_20260925.md) | 게시 대상 |
| v1 native nested416 server | [열기](../../docs/v1_native_nested416_server_20261003.md) | 게시 대상 |
| v1 native nested runtime | [열기](../../docs/v1_native_nested_runtime_20261003.md) | 게시 대상 |
| v1 r3 review followup | [열기](../../docs/v1_r3_review_followup_20261003.md) | 게시 대상 |
| v1 r5 review closed | [열기](../../docs/v1_r5_review_closed_20261003.md) | 게시 대상 |
| v1 roi budget recovery | [열기](../../docs/v1_roi_budget_recovery_20261003.md) | 게시 대상 |
| v1 strict nested review correction | [열기](../../docs/v1_strict_nested_review_correction_20261003.md) | 게시 대상 |
| v1x progressive experiments | [열기](../../docs/v1x_progressive_experiments_20261003.md) | 게시 대상 |

</details>

<details><summary>검사 (46)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| native30 checkpoint | [열기](../../tests/test_native30_checkpoint.py) | 게시 대상 |
| native30 data contract | [열기](../../tests/test_native30_data_contract.py) | 게시 대상 |
| native30 evaluation entry | [열기](../../tests/test_native30_evaluation_entry.py) | 게시 대상 |
| native30 geometry | [열기](../../tests/test_native30_geometry.py) | 게시 대상 |
| native30 geometry relocation | [열기](../../tests/test_native30_geometry_relocation.py) | 게시 대상 |
| native30 upper | [열기](../../tests/test_native30_upper.py) | 게시 대상 |
| native v1 checkpoint inspection | [열기](../../tests/test_native_v1_checkpoint_inspection.py) | 게시 대상 |
| old v1 graph cleanup | [열기](../../tests/test_old_v1_graph_cleanup.py) | 게시 대상 |
| v1 bounded scope | [열기](../../tests/test_v1_bounded_scope.py) | 게시 대상 |
| v1 cache budget recovery | [열기](../../tests/test_v1_cache_budget_recovery.py) | 게시 대상 |
| v1 deterministic sampling | [열기](../../tests/test_v1_deterministic_sampling.py) | 게시 대상 |
| v1 empty context | [열기](../../tests/test_v1_empty_context.py) | 게시 대상 |
| v1 epoch telemetry | [열기](../../tests/test_v1_epoch_telemetry.py) | 게시 대상 |
| v1 execution | [열기](../../tests/test_v1_execution.py) | 게시 대상 |
| v1 graph size probe | [열기](../../tests/test_v1_graph_size_probe.py) | 로컬 |
| v1 nested execution | [열기](../../tests/test_v1_nested_execution.py) | 로컬 |
| v1 nested review bundle | [열기](../../tests/test_v1_nested_review_bundle.py) | 로컬 |
| v1 paired training | [열기](../../tests/test_v1_paired_training.py) | 게시 대상 |
| v1 recorded experiment | [열기](../../tests/test_v1_recorded_experiment.py) | 게시 대상 |
| v1 review bundle input | [열기](../../tests/test_v1_review_bundle_input.py) | 로컬 |
| v1 roi admission contract | [열기](../../tests/test_v1_roi_admission_contract.py) | 게시 대상 |
| v1 roi budget probe | [열기](../../tests/test_v1_roi_budget_probe.py) | 게시 대상 |
| v1 roi failure replay | [열기](../../tests/test_v1_roi_failure_replay.py) | 게시 대상 |
| v1 roi probe failure display | [열기](../../tests/test_v1_roi_probe_failure_display.py) | 게시 대상 |
| v1 roi probe reuse | [열기](../../tests/test_v1_roi_probe_reuse.py) | 게시 대상 |
| v1 roi recovery suite | [열기](../../tests/test_v1_roi_recovery_suite.py) | 게시 대상 |
| v1 roi replay admission | [열기](../../tests/test_v1_roi_replay_admission.py) | 게시 대상 |
| v1 runtime416 review bundle | [열기](../../tests/test_v1_runtime416_review_bundle.py) | 로컬 |
| v1 sampling metric cuda | [열기](../../tests/test_v1_sampling_metric_cuda.py) | 로컬 |
| v1 sampling pair metrics | [열기](../../tests/test_v1_sampling_pair_metrics.py) | 게시 대상 |
| v1 server | [열기](../../tests/test_v1_server.py) | 게시 대상 |
| v1 snapshot bytecode | [열기](../../tests/test_v1_snapshot_bytecode.py) | 게시 대상 |
| v1 telemetry entry | [열기](../../tests/test_v1_telemetry_entry.py) | 게시 대상 |
| v1x contracts | [열기](../../tests/test_v1x_contracts.py) | 게시 대상 |
| v1x cuda debug | [열기](../../tests/test_v1x_cuda_debug.py) | 로컬 |
| v1x experiment | [열기](../../tests/test_v1x_experiment.py) | 게시 대상 |
| v1x graph flow audit | [열기](../../tests/test_v1x_graph_flow_audit.py) | 로컬 |
| v1x graph size | [열기](../../tests/test_v1x_graph_size.py) | 로컬 |
| v1x group search | [열기](../../tests/test_v1x_group_search.py) | 로컬 |
| v1x models | [열기](../../tests/test_v1x_models.py) | 로컬 |
| v1x nested graph size | [열기](../../tests/test_v1x_nested_graph_size.py) | 로컬 |
| v1x results | [열기](../../tests/test_v1x_results.py) | 로컬 |
| v1x sampling comparison | [열기](../../tests/test_v1x_sampling_comparison.py) | 로컬 |
| v1x sampling experiment | [열기](../../tests/test_v1x_sampling_experiment.py) | 게시 대상 |
| v1x sampling runtime | [열기](../../tests/test_v1x_sampling_runtime.py) | 로컬 |
| v1x serialization | [열기](../../tests/test_v1x_serialization.py) | 로컬 |

</details>


## 작업 완료 체크리스트

이 파일은 탐색 인덱스다. 학습·평가·자원 benchmark와 데이터 이동·삭제는 실행하지 않는다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. (인덱스 생성에 해당 없음)
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. (인덱스 생성에 해당 없음)
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (인덱스 생성에 해당 없음)
- [x] 디버그 설정과 최종 설정을 분리해서 안내했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (새 실행 검사 없음)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 기존 구현·기록으로 연결한다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.