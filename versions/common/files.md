# common 파일 안내

원본 경로와 SHA는 유지합니다. 아래는 짧은 이름으로 찾는 실행·코드·설정·기록 목록입니다.
게시 여부는 Git index 기준입니다. 로컬 파일은 서버 checkout에 없을 수 있습니다.

<details><summary>구현 (24)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
|   init   | [열기](../../basic_cp_online/__init__.py) | 게시 대상 |
| nnUNetTrainer BasicCP80Online | [열기](../../basic_cp_online/nnUNetTrainer_BasicCP80Online.py) | 게시 대상 |
| nnUNetTrainer OriginalBasicCPOnline | [열기](../../basic_cp_online/nnUNetTrainer_OriginalBasicCPOnline.py) | 게시 대상 |
| prepare | [열기](../../basic_cp_online/prepare.py) | 게시 대상 |
| reference | [열기](../../basic_cp_online/reference.py) | 게시 대상 |
| runtime | [열기](../../basic_cp_online/runtime.py) | 게시 대상 |
| code | [열기](../../code.txt) | 로컬 수정 |
| comparison randomness | [열기](../../comparison_randomness.py) | 게시 대상 |
|   init   | [열기](../../custom_trainers/__init__.py) | 게시 대상 |
| install onlinecp custom trainers | [열기](../../custom_trainers/install_onlinecp_custom_trainers.py) | 게시 대상 |
| nnUNetTrainer OnlineCPCurriculum | [열기](../../custom_trainers/nnUNetTrainer_OnlineCPCurriculum.py) | 게시 대상 |
| nnUNetTrainer OnlineCPFeedback | [열기](../../custom_trainers/nnUNetTrainer_OnlineCPFeedback.py) | 게시 대상 |
| nnUNetTrainer OnlinePairedCP | [열기](../../custom_trainers/nnUNetTrainer_OnlinePairedCP.py) | 게시 대상 |
| nnUNetTrainer OnlinePairedCPArgmaxV3 | [열기](../../custom_trainers/nnUNetTrainer_OnlinePairedCPArgmaxV3.py) | 게시 대상 |
| onlinecp curriculum contract | [열기](../../custom_trainers/onlinecp_curriculum_contract.py) | 게시 대상 |
| onlinecp curriculum policy | [열기](../../custom_trainers/onlinecp_curriculum_policy.py) | 게시 대상 |
| onlinecp feedback metrics | [열기](../../custom_trainers/onlinecp_feedback_metrics.py) | 게시 대상 |
| onlinecp feedback policy | [열기](../../custom_trainers/onlinecp_feedback_policy.py) | 게시 대상 |
| onlinecp raw bank | [열기](../../custom_trainers/onlinecp_raw_bank.py) | 게시 대상 |
| onlinecp raw resampling | [열기](../../custom_trainers/onlinecp_raw_resampling.py) | 게시 대상 |
| debug information contract | [열기](../../feedback/debug_information_contract_20260912.py) | 게시 대상 |
| pyproject | [열기](../../pyproject.toml) | 게시 대상 |
| requirements-debug | [열기](../../requirements-debug.txt) | 게시 대상 |
| requirements | [열기](../../requirements.txt) | 게시 대상 |

</details>

<details><summary>설정 (7)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| comparison cp80 | [열기](../../config/comparison_cp80.json) | 게시 대상 |
| nnunet | [열기](../../config/nnunet.json) | 게시 대상 |
| online cp curriculum | [열기](../../config/online_cp_curriculum.json) | 게시 대상 |
| online cp feedback | [열기](../../config/online_cp_feedback.json) | 게시 대상 |
| online cp feedback gnn | [열기](../../config/online_cp_feedback_gnn.json) | 게시 대상 |
| server gpu allocations | [열기](../../config/server_gpu_allocations.json) | 게시 대상 |
| split cp80 fold0 | [열기](../../config/split_cp80_fold0.json) | 게시 대상 |

</details>

<details><summary>실행·분석 (92)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| run basic cp online | [열기](../../run_basic_cp_online.py) | 게시 대상 |
|   init   | [열기](../../tools/__init__.py) | 게시 대상 |
| ablation | [열기](../../tools/ablation.py) | 게시 대상 |
| assemble | [열기](../../tools/assemble.py) | 게시 대상 |
| audit | [열기](../../tools/audit.py) | 게시 대상 |
| audit population bank | [열기](../../tools/audit_population_bank.py) | 게시 대상 |
| audit prompt graph contract debug | [열기](../../tools/audit_prompt_graph_contract_debug.py) | 게시 대상 |
| audit relational connectivity debug | [열기](../../tools/audit_relational_connectivity_debug.py) | 게시 대상 |
| audit relay connectivity debug | [열기](../../tools/audit_relay_connectivity_debug.py) | 게시 대상 |
| audit relay shape metadata | [열기](../../tools/audit_relay_shape_metadata.py) | 로컬 |
| basic bank equivalence | [열기](../../tools/basic_bank_equivalence.py) | 게시 대상 |
| basic reuse provenance | [열기](../../tools/basic_reuse_provenance.py) | 게시 대상 |
| build cp recommender review bundle | [열기](../../tools/build_cp_recommender_review_bundle.py) | 로컬 |
| build gnn pipeline pdf | [열기](../../tools/build_gnn_pipeline_pdf_20260918.py) | 로컬 |
| build version guides | [열기](../../tools/build_version_guides.py) | 게시 대상 |
| build version layout | [열기](../../tools/build_version_layout.py) | 게시 대상 |
| case | [열기](../../tools/case.py) | 게시 대상 |
| causality | [열기](../../tools/causality.py) | 게시 대상 |
| causality resources | [열기](../../tools/causality_resources.py) | 게시 대상 |
| ct only visual | [열기](../../tools/check_ct_only_visual.cjs) | 로컬 |
| footprint shape visual | [열기](../../tools/check_footprint_shape_visual.cjs) | 로컬 |
| native ct annotation | [열기](../../tools/check_native_ct_annotation.py) | 로컬 |
| paired cnn visual | [열기](../../tools/check_paired_cnn_visual.cjs) | 로컬 |
| relay shape visual | [열기](../../tools/check_relay_shape_visual.cjs) | 로컬 |
| cleanup verified storage | [열기](../../tools/cleanup_verified_storage.py) | 로컬 |
| compress lossless storage | [열기](../../tools/compress_lossless_storage.py) | 로컬 |
| ct-only-liver.template | [열기](../../tools/ct-only-liver.template.html) | 로컬 |
| download task03 liver | [열기](../../tools/download_task03_liver.py) | 로컬 |
| downstream level ablation | [열기](../../tools/downstream_level_ablation.py) | 게시 대상 |
| env | [열기](../../tools/env.py) | 게시 대상 |
| evaluate feedback experiment | [열기](../../tools/evaluate_feedback_experiment.py) | 게시 대상 |
| export ct only visual | [열기](../../tools/export_ct_only_visual.py) | 로컬 |
| export gpt handoff | [열기](../../tools/export_gpt_handoff.py) | 게시 대상 |
| export paired cnn visual debug | [열기](../../tools/export_paired_cnn_visual_debug.py) | 로컬 |
| feedback bank upgrade | [열기](../../tools/feedback_bank_upgrade.py) | 게시 대상 |
| feedback basic reuse | [열기](../../tools/feedback_basic_reuse.py) | 게시 대상 |
| feedback fresh execution | [열기](../../tools/feedback_fresh_execution.py) | 게시 대상 |
| feedback preparation recovery | [열기](../../tools/feedback_preparation_recovery.py) | 게시 대상 |
| feedback preprocessing reuse | [열기](../../tools/feedback_preprocessing_reuse.py) | 게시 대상 |
| feedback stage execution | [열기](../../tools/feedback_stage_execution.py) | 게시 대상 |
| fixed-region-inspector.template | [열기](../../tools/fixed-region-inspector.template.html) | 게시 대상 |
| footprint-shape.template | [열기](../../tools/footprint-shape.template.html) | 로컬 |
| index versions | [열기](../../tools/index_versions.py) | 게시 대상 |
| inspect installed epoch boundary | [열기](../../tools/inspect_installed_epoch_boundary.py) | 게시 대상 |
| install | [열기](../../tools/install.py) | 게시 대상 |
| l0-full-edge-inspector.template | [열기](../../tools/l0-full-edge-inspector.template.html) | 로컬 |
| l0-methods-comparison.template | [열기](../../tools/l0-methods-comparison.template.html) | 로컬 |
| l0-spatial-encoding.template | [열기](../../tools/l0-spatial-encoding.template.html) | 로컬 |
| local-cnn-fov.template | [열기](../../tools/local-cnn-fov.template.html) | 로컬 |
| measure same donor learning debug | [열기](../../tools/measure_same_donor_learning_debug.py) | 게시 대상 |
| nnunet | [열기](../../tools/nnunet.py) | 게시 대상 |
| online bank disk retry | [열기](../../tools/online_bank_disk_retry.py) | 게시 대상 |
| online bank preparation | [열기](../../tools/online_bank_preparation.py) | 게시 대상 |
| online bank progress | [열기](../../tools/online_bank_progress.py) | 게시 대상 |
| online cp argmax benchmark | [열기](../../tools/online_cp_argmax_benchmark.py) | 게시 대상 |
| online cp benchmark | [열기](../../tools/online_cp_benchmark.py) | 게시 대상 |
| online cp curriculum | [열기](../../tools/online_cp_curriculum.py) | 게시 대상 |
| online eval provenance | [열기](../../tools/online_eval_provenance.py) | 게시 대상 |
| online eval v2 | [열기](../../tools/online_eval_v2.py) | 게시 대상 |
| online raw bank preparation | [열기](../../tools/online_raw_bank_preparation.py) | 게시 대상 |
| online scoring | [열기](../../tools/online_scoring.py) | 게시 대상 |
| online trainer contract | [열기](../../tools/online_trainer_contract.py) | 게시 대상 |
| paired-cnn-readout.template | [열기](../../tools/paired-cnn-readout.template.html) | 로컬 |
| paired-local-cnn.template | [열기](../../tools/paired-local-cnn.template.html) | 로컬 |
| paired benchmark | [열기](../../tools/paired_benchmark.py) | 게시 대상 |
| prepare same donor | [열기](../../tools/prepare_same_donor.py) | 게시 대상 |
| profile sage refresh debug | [열기](../../tools/profile_sage_refresh_debug.py) | 게시 대상 |
| qa gnn pipeline pdf | [열기](../../tools/qa_gnn_pipeline_pdf_20260918.py) | 로컬 |
| regress | [열기](../../tools/regress.py) | 게시 대상 |
| relay-shape.template | [열기](../../tools/relay-shape.template.html) | 로컬 |
| resume basic cp80 preparation | [열기](../../tools/resume_basic_cp80_preparation.py) | 로컬 |
| run feedback experiment | [열기](../../tools/run_feedback_experiment.py) | 게시 대상 |
| run fixed regions | [열기](../../tools/run_fixed_regions.py) | 게시 대상 |
| run same donor a6000 | [열기](../../tools/run_same_donor_a6000.sh) | 게시 대상 |
| run uncoarsened sage | [열기](../../tools/run_uncoarsened_sage.py) | 게시 대상 |
| smoke | [열기](../../tools/smoke.py) | 게시 대상 |
| snapshot pipeline v1 | [열기](../../tools/snapshot_pipeline_v1.py) | 로컬 |
| train online curriculum | [열기](../../tools/train_online_curriculum.py) | 게시 대상 |
| train online feedback | [열기](../../tools/train_online_feedback.py) | 게시 대상 |
| v1-l0-actual-visual.template | [열기](../../tools/v1-l0-actual-visual.template.html) | 로컬 |
| validate | [열기](../../tools/validate.py) | 게시 대상 |
| candidate curriculum debug | [열기](../../tools/verify_candidate_curriculum_debug.py) | 게시 대상 |
| original basic cp online debug | [열기](../../tools/verify_original_basic_cp_online_debug.py) | 로컬 |
| paired cnn control debug | [열기](../../tools/verify_paired_cnn_control_debug.py) | 로컬 |
| reference l1 ct debug | [열기](../../tools/verify_reference_l1_ct_debug.py) | 게시 대상 |
| reference rankable ct debug | [열기](../../tools/verify_reference_rankable_ct_debug.py) | 게시 대상 |
| same donor debug | [열기](../../tools/verify_same_donor_debug.py) | 게시 대상 |
| storage cleanup | [열기](../../tools/verify_storage_cleanup.py) | 로컬 |
| storage compression | [열기](../../tools/verify_storage_compression.py) | 로컬 |
| uncoarsened sage debug | [열기](../../tools/verify_uncoarsened_sage_debug.py) | 게시 대상 |
| version layout | [열기](../../tools/verify_version_layout.py) | 게시 대상 |
| visualize ct vessels | [열기](../../tools/visualize_ct_vessels.py) | 로컬 |

</details>

<details><summary>기록 (38)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| AGENTS | [열기](../../AGENTS.md) | 게시 대상 |
| README | [열기](../../custom_trainers/README.md) | 게시 대상 |
| 2026-09-05-levels-readonly-audit | [열기](../../docs/audits/2026-09-05-levels-readonly-audit.md) | 게시 대상 |
| comparison seed audit | [열기](../../docs/comparison_seed_audit_20260922.md) | 게시 대상 |
| cp80 comparison | [열기](../../docs/cp80_comparison_20260922.md) | 게시 대상 |
| cp80 comparison center decision | [열기](../../docs/cp80_comparison_center_decision.md) | 게시 대상 |
| cp80 resume | [열기](../../docs/cp80_resume_20260923.md) | 게시 대상 |
| cp input repair | [열기](../../docs/cp_input_repair.md) | 게시 대상 |
| design | [열기](../../docs/design.md) | 게시 대상 |
| feature evidence audit | [열기](../../docs/feature_evidence_audit_20260922.md) | 게시 대상 |
| feedback recovery | [열기](../../docs/feedback_recovery.md) | 게시 대상 |
| full edge training | [열기](../../docs/full_edge_training.md) | 게시 대상 |
| GPT CP REVIEW | [열기](../../docs/GPT_CP_REVIEW_20261002_CODE_MAP.md) | 게시 대상 |
| GPT CP REVIEW | [열기](../../docs/GPT_CP_REVIEW_20261002_HISTORY.md) | 게시 대상 |
| GPT CP REVIEW | [열기](../../docs/GPT_CP_REVIEW_20261002_QUESTIONS.md) | 게시 대상 |
| graph pooling evidence | [열기](../../docs/graph_pooling_evidence_20260923.md) | 게시 대상 |
| level v3 implementation | [열기](../../docs/level_v3_implementation.md) | 게시 대상 |
| notion research index | [열기](../../docs/notion_research_index_20260929.md) | 게시 대상 |
| notion research snapshot | [열기](../../docs/notion_research_snapshot_20260929.json) | 게시 대상 |
| notion version consolidation | [열기](../../docs/notion_version_consolidation_20260929.md) | 게시 대상 |
| online bank performance | [열기](../../docs/online_bank_performance.md) | 게시 대상 |
| online cp curriculum | [열기](../../docs/online_cp_curriculum.md) | 게시 대상 |
| online cp feedback | [열기](../../docs/online_cp_feedback.md) | 게시 대상 |
| online evaluation verification | [열기](../../docs/online_evaluation_verification.md) | 게시 대상 |
| original basic cp online | [열기](../../docs/original_basic_cp_online.md) | 게시 대상 |
| paired cnn control | [열기](../../docs/paired_cnn_control_20260930.md) | 게시 대상 |
| prodigy current pipeline audit | [열기](../../docs/prodigy_current_pipeline_audit_20261001.md) | 게시 대상 |
| results summary | [열기](../../docs/results_summary_20260918.md) | 게시 대상 |
| storage cleanup | [열기](../../docs/storage_cleanup_20260923.md) | 게시 대상 |
| storage compression | [열기](../../docs/storage_compression_20260923.md) | 게시 대상 |
| HierCP code review | [열기](../../feedback/HierCP_code_review_20260912.md) | 게시 대상 |
| HierCP current review design | [열기](../../feedback/HierCP_current_review_design_20260912.md) | 게시 대상 |
| HierCP review inventory | [열기](../../feedback/HierCP_review_inventory_20260912.md) | 게시 대상 |
| gpt handoff | [열기](../../gpt_handoff.md) | 로컬 수정 |
| PATCH NOTES | [열기](../../PATCH_NOTES.md) | 로컬 수정 |
| README | [열기](../../README.md) | 게시 대상 |
| REFERENCES | [열기](../../REFERENCES.md) | 로컬 수정 |
| START | [열기](../../START.md) | 게시 대상 |

</details>

<details><summary>검사 (107)</summary>

| 내용 | 원본 | Git |
| --- | --- | --- |
| ablation comparability debug | [열기](../../tests/test_ablation_comparability_debug.py) | 게시 대상 |
| bank geometry preparation debug | [열기](../../tests/test_bank_geometry_preparation_debug.py) | 게시 대상 |
| bank parallel preparation debug | [열기](../../tests/test_bank_parallel_preparation_debug.py) | 게시 대상 |
| bank progress debug | [열기](../../tests/test_bank_progress_debug.py) | 게시 대상 |
| bank scoring canonical debug | [열기](../../tests/test_bank_scoring_canonical_debug.py) | 게시 대상 |
| bank scoring resources debug | [열기](../../tests/test_bank_scoring_resources_debug.py) | 게시 대상 |
| basic bank equivalence debug | [열기](../../tests/test_basic_bank_equivalence_debug.py) | 게시 대상 |
| basic reuse provenance debug | [열기](../../tests/test_basic_reuse_provenance_debug.py) | 게시 대상 |
| batch calibration debug | [열기](../../tests/test_batch_calibration_debug.py) | 게시 대상 |
| cache recovery debug | [열기](../../tests/test_cache_recovery_debug.py) | 게시 대상 |
| cache training contract debug | [열기](../../tests/test_cache_training_contract_debug.py) | 게시 대상 |
| candidate curriculum | [열기](../../tests/test_candidate_curriculum.py) | 게시 대상 |
| candidate diagnostics debug | [열기](../../tests/test_candidate_diagnostics_debug.py) | 게시 대상 |
| candidate pool reuse debug | [열기](../../tests/test_candidate_pool_reuse_debug.py) | 게시 대상 |
| causality bank contract debug | [열기](../../tests/test_causality_bank_contract_debug.py) | 게시 대상 |
| causality checkpoint contract debug | [열기](../../tests/test_causality_checkpoint_contract_debug.py) | 게시 대상 |
| causality overlap debug | [열기](../../tests/test_causality_overlap_debug.py) | 게시 대상 |
| causality preflight debug | [열기](../../tests/test_causality_preflight_debug.py) | 게시 대상 |
| causality resources debug | [열기](../../tests/test_causality_resources_debug.py) | 게시 대상 |
| causality transform memory debug | [열기](../../tests/test_causality_transform_memory_debug.py) | 게시 대상 |
| causality transforms debug | [열기](../../tests/test_causality_transforms_debug.py) | 게시 대상 |
| comparison randomness debug | [열기](../../tests/test_comparison_randomness_debug.py) | 로컬 |
| competition analysis | [열기](../../tests/test_competition_analysis.py) | 게시 대상 |
| cp80 comparison debug | [열기](../../tests/test_cp80_comparison_debug.py) | 로컬 |
| cp exact masks debug | [열기](../../tests/test_cp_exact_masks_debug.py) | 게시 대상 |
| curriculum bank contract | [열기](../../tests/test_curriculum_bank_contract.py) | 게시 대상 |
| curriculum launch debug | [열기](../../tests/test_curriculum_launch_debug.py) | 게시 대상 |
| curriculum training | [열기](../../tests/test_curriculum_training.py) | 게시 대상 |
| donor geometry debug | [열기](../../tests/test_donor_geometry_debug.py) | 게시 대상 |
| donor preflight debug | [열기](../../tests/test_donor_preflight_debug.py) | 게시 대상 |
| donor usage debug | [열기](../../tests/test_donor_usage_debug.py) | 게시 대상 |
| downstream level audit | [열기](../../tests/test_downstream_level_audit.py) | 게시 대상 |
| downstream reuse | [열기](../../tests/test_downstream_reuse.py) | 게시 대상 |
| edge attention streaming debug | [열기](../../tests/test_edge_attention_streaming_debug.py) | 게시 대상 |
| feedback bank upgrade debug | [열기](../../tests/test_feedback_bank_upgrade_debug.py) | 게시 대상 |
| feedback basic reuse debug | [열기](../../tests/test_feedback_basic_reuse_debug.py) | 게시 대상 |
| feedback evaluation producer debug | [열기](../../tests/test_feedback_evaluation_producer_debug.py) | 게시 대상 |
| feedback experiment debug | [열기](../../tests/test_feedback_experiment_debug.py) | 게시 대상 |
| feedback gnn debug | [열기](../../tests/test_feedback_gnn_debug.py) | 게시 대상 |
| feedback graph binding debug | [열기](../../tests/test_feedback_graph_binding_debug.py) | 게시 대상 |
| feedback launch bridges debug | [열기](../../tests/test_feedback_launch_bridges_debug.py) | 게시 대상 |
| feedback launch debug | [열기](../../tests/test_feedback_launch_debug.py) | 게시 대상 |
| feedback preprocessing reuse debug | [열기](../../tests/test_feedback_preprocessing_reuse_debug.py) | 게시 대상 |
| feedback stage execution debug | [열기](../../tests/test_feedback_stage_execution_debug.py) | 게시 대상 |
| feedback storage resources debug | [열기](../../tests/test_feedback_storage_resources_debug.py) | 게시 대상 |
| feedback trainer integration | [열기](../../tests/test_feedback_trainer_integration.py) | 게시 대상 |
| footprint domain cuda | [열기](../../tests/test_footprint_domain_cuda.py) | 로컬 |
| footprint graph cuda | [열기](../../tests/test_footprint_graph_cuda.py) | 로컬 |
| full edge view debug | [열기](../../tests/test_full_edge_view_debug.py) | 게시 대상 |
| gpt handoff export debug | [열기](../../tests/test_gpt_handoff_export_debug.py) | 게시 대상 |
| hierarchy loss debug | [열기](../../tests/test_hierarchy_loss_debug.py) | 게시 대상 |
| hierarchy model debug | [열기](../../tests/test_hierarchy_model_debug.py) | 게시 대상 |
| historical c checkpoint | [열기](../../tests/test_historical_c_checkpoint.py) | 게시 대상 |
| historical checkpoint | [열기](../../tests/test_historical_checkpoint.py) | 게시 대상 |
| historical evaluation | [열기](../../tests/test_historical_evaluation.py) | 게시 대상 |
| historical patient graph | [열기](../../tests/test_historical_patient_graph.py) | 게시 대상 |
| medical aug reference debug | [열기](../../tests/test_medical_aug_reference_debug.py) | 게시 대상 |
| medical aug source import debug | [열기](../../tests/test_medical_aug_source_import_debug.py) | 게시 대상 |
| no placement cache debug | [열기](../../tests/test_no_placement_cache_debug.py) | 게시 대상 |
| online bank disk retry debug | [열기](../../tests/test_online_bank_disk_retry_debug.py) | 게시 대상 |
| online bank wiring debug | [열기](../../tests/test_online_bank_wiring_debug.py) | 게시 대상 |
| online cp curriculum | [열기](../../tests/test_online_cp_curriculum.py) | 게시 대상 |
| online cp feedback metrics | [열기](../../tests/test_online_cp_feedback_metrics.py) | 게시 대상 |
| online cp feedback policy | [열기](../../tests/test_online_cp_feedback_policy.py) | 게시 대상 |
| online eval io regression | [열기](../../tests/test_online_eval_io_regression.py) | 게시 대상 |
| online eval provenance | [열기](../../tests/test_online_eval_provenance.py) | 게시 대상 |
| online eval statistics | [열기](../../tests/test_online_eval_statistics.py) | 게시 대상 |
| online no placement debug | [열기](../../tests/test_online_no_placement_debug.py) | 게시 대상 |
| online raw bank preparation debug | [열기](../../tests/test_online_raw_bank_preparation_debug.py) | 게시 대상 |
| online source mapping debug | [열기](../../tests/test_online_source_mapping_debug.py) | 게시 대상 |
| online support dispatch debug | [열기](../../tests/test_online_support_dispatch_debug.py) | 게시 대상 |
| original basic cp online debug | [열기](../../tests/test_original_basic_cp_online_debug.py) | 로컬 |
| paired cnn control debug | [열기](../../tests/test_paired_cnn_control_debug.py) | 로컬 |
| paired gnn cli debug | [열기](../../tests/test_paired_gnn_cli_debug.py) | 게시 대상 |
| patient task preparation debug | [열기](../../tests/test_patient_task_preparation_debug.py) | 로컬 |
| physical relay cuda | [열기](../../tests/test_physical_relay_cuda.py) | 로컬 |
| population metric edges debug | [열기](../../tests/test_population_metric_edges_debug.py) | 게시 대상 |
| population publication debug | [열기](../../tests/test_population_publication_debug.py) | 게시 대상 |
| preparation recovery debug | [열기](../../tests/test_preparation_recovery_debug.py) | 게시 대상 |
| preparation runtime debug | [열기](../../tests/test_preparation_runtime_debug.py) | 게시 대상 |
| preprocessed storage debug | [열기](../../tests/test_preprocessed_storage_debug.py) | 게시 대상 |
| prototype fit v2 debug | [열기](../../tests/test_prototype_fit_v2_debug.py) | 게시 대상 |
| ranking history debug | [열기](../../tests/test_ranking_history_debug.py) | 게시 대상 |
| raw bank consumer guards debug | [열기](../../tests/test_raw_bank_consumer_guards_debug.py) | 게시 대상 |
| raw bank shared sources debug | [열기](../../tests/test_raw_bank_shared_sources_debug.py) | 게시 대상 |
| raw bank storage debug | [열기](../../tests/test_raw_bank_storage_debug.py) | 게시 대상 |
| raw bank type boundary debug | [열기](../../tests/test_raw_bank_type_boundary_debug.py) | 게시 대상 |
| raw cp resampling debug | [열기](../../tests/test_raw_cp_resampling_debug.py) | 게시 대상 |
| raw cp trainer debug | [열기](../../tests/test_raw_cp_trainer_debug.py) | 게시 대상 |
| raw cp trainer native debug | [열기](../../tests/test_raw_cp_trainer_native_debug.py) | 게시 대상 |
| recovery orchestration debug | [열기](../../tests/test_recovery_orchestration_debug.py) | 게시 대상 |
| reference causal integration | [열기](../../tests/test_reference_causal_integration.py) | 게시 대상 |
| reference causal summary | [열기](../../tests/test_reference_causal_summary.py) | 게시 대상 |
| reference finite summary | [열기](../../tests/test_reference_finite_summary.py) | 게시 대상 |
| reference mode ranking summary | [열기](../../tests/test_reference_mode_ranking_summary.py) | 게시 대상 |
| reference rankable control | [열기](../../tests/test_reference_rankable_control.py) | 게시 대상 |
| reference rankable summary | [열기](../../tests/test_reference_rankable_summary.py) | 게시 대상 |
| relay connectivity json audit | [열기](../../tests/test_relay_connectivity_json_audit.py) | 게시 대상 |
| resume basic cp80 debug | [열기](../../tests/test_resume_basic_cp80_debug.py) | 로컬 |
| retained generation debug | [열기](../../tests/test_retained_generation_debug.py) | 게시 대상 |
| review publication matching debug | [열기](../../tests/test_review_publication_matching_debug.py) | 게시 대상 |
| same donor learning debug | [열기](../../tests/test_same_donor_learning_debug.py) | 게시 대상 |
| smoke view contract debug | [열기](../../tests/test_smoke_view_contract_debug.py) | 게시 대상 |
| source content contract debug | [열기](../../tests/test_source_content_contract_debug.py) | 게시 대상 |
| training resources debug | [열기](../../tests/test_training_resources_debug.py) | 게시 대상 |
| uncoarsened sage debug | [열기](../../tests/test_uncoarsened_sage_debug.py) | 게시 대상 |
| version layout | [열기](../../tests/test_version_layout.py) | 게시 대상 |

</details>
