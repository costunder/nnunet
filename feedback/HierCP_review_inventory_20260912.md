# HierCP 파일별 검토 대장 — 159개

기준: 첨부 code.txt의 ec2d8388f39c6b03ed32a3677fbdd6eaafed4731. 원격 저장소/서버 파일은 별도 확인하지 않았다.

**전수 검사는 파일 인벤토리·Python/JSON 구문·AST 참조 범위다. 각 파일의 모든 분기를 native 실행한 기록이나, 모든 줄에 동일한 깊이의 수동 의미 검증을 수행했다는 주장이 아니다.** 복잡한 활성 raw transport, L0/L1/L2, batching/autograd, feedback, recovery 및 평가 경계는 본 보고서에서 별도로 추적했다.

테스트 수는 unittest callback/event 수이므로 setup/subtest 오류와 단순 합산하지 않는다. 오류가 있는 모듈을 통과로 표시하지 않았다.

| # | 파일 | 본문 줄 | code.txt 시작 | 함수/클래스 | 검토 내용/실행 상태 |
|---:|---|---:|---:|---:|---|
| 1 | `.gitignore` | 56 | 190 | 0/0 | 의료영상·배열·체크포인트·work 제외, 소스·설정 추적 경계. |
| 2 | `AGENTS.md` | 408 | 250 | 0/0 | 보존·전체 규모·실측 자원·DEBUG/최종 실행 구분 규칙을 감사 기준으로 대조. |
| 3 | `README.md` | 296 | 662 | 0/0 | 현재 raw-feedback 경로와 offline 경로 분리. offline full 완료 설명은 F3와 충돌. |
| 4 | `config/nnunet.json` | 41 | 962 | 0/0 | ResEncM/250 epochs/5 folds 및 source equivalent diameter 0–20 mm 조건. 현재 단일 outer-fold feedback과 구분. |
| 5 | `config/online_cp_curriculum.json` | 16 | 1007 | 0/0 | 과거 고정 rank-band 정책. 현재 실제-error feedback과 혼합 금지. |
| 6 | `config/online_cp_feedback.json` | 26 | 1027 | 0/0 | 품질 gate, EMA, 20/20/60 난이도 혼합. 임상·효능이 검증된 값으로 해석하지 않음. |
| 7 | `config/online_cp_feedback_gnn.json` | 19 | 1057 | 0/0 | 별도 difficulty GNN, epoch당 관측 전체 1회 순회, 매 epoch 예측, worker는 quality 측정값 상속. |
| 8 | `config/train.json` | 182 | 1080 | 0/0 | full_v22/v3, 48³, 128 hidden, 3/2/2, 8/128 후보, 40 epochs, auto batch/worker 보존. |
| 9 | `custom_trainers/.gitattributes` | 3 | 1266 | 0/0 | trainer LF 보존과 hash 일치 경계. |
| 10 | `custom_trainers/README.md` | 132 | 1273 | 0/0 | 새 feedback와 legacy trainer의 typed bank 경계 및 설치 계약. |
| 11 | `custom_trainers/SHA256SUMS` | 10 | 1409 | 0/0 | 10개 source bytes 및 installer MODULES와 모두 일치. |
| 12 | `custom_trainers/__init__.py` | 1 | 1423 | 0/0 | portable helper package marker. |
| 13 | `custom_trainers/install_onlinecp_custom_trainers.py` | 324 | 1428 | 10/0 | 10개 모듈 SHA 검증, 배타적 설치·백업·실패 원복. 실제 nnU-Net 설치/import는 미실행. |
| 14 | `custom_trainers/nnUNetTrainer_OnlineCPCurriculum.py` | 533 | 1756 | 25/7 | typed bank/cohort 검증, epoch별 loader 재시작, complete-epoch checkpoint와 엄격한 restore. |
| 15 | `custom_trainers/nnUNetTrainer_OnlineCPFeedback.py` | 464 | 2293 | 23/6 | 5-draw event/choice 분리, 실제 loss observer, epoch 경계 EMA/GNN 갱신. native 전체 lifecycle 미실행. |
| 16 | `custom_trainers/nnUNetTrainer_OnlinePairedCP.py` | 1216 | 2761 | 43/10 | legacy/raw dispatch, 고정 크기 crop, native baseline seg 대조, 전체/native/crop support 감사. |
| 17 | `custom_trainers/nnUNetTrainer_OnlinePairedCPArgmaxV3.py` | 761 | 3981 | 26/6 | 과거 multi-pool exact-argmax 전용. 새 raw-target bank를 명시적으로 거절. |
| 18 | `custom_trainers/onlinecp_curriculum_contract.py` | 280 | 4746 | 11/0 | 파일 SHA·live 전처리·cohort·NPZ unpack 비교. 계약은 암호서명이나 실제 환자 grouping 증명이 아님. |
| 19 | `custom_trainers/onlinecp_curriculum_policy.py` | 177 | 5030 | 12/1 | 과거 rank-band/minimum-choice/temperature 규칙 및 일정 digest. 현재 feedback 정책과 별개. |
| 20 | `custom_trainers/onlinecp_feedback_metrics.py` | 303 | 5211 | 9/1 | 실제 최고해상도 pre-update 측정, detached/NaN 상태, shared transform. whole-lesion 보존 증명은 아님. |
| 21 | `custom_trainers/onlinecp_feedback_policy.py` | 484 | 5518 | 28/2 | frozen epoch snapshot, 관측 우선·예측 freshness, quality gate, tie-aware quantile, 무관측 exploration. |
| 22 | `custom_trainers/onlinecp_raw_bank.py` | 360 | 6006 | 20/2 | pickle 없는 JSON/NPZ·공유 donor NPY·mmap·SHA/stat witness. incomplete publication은 fail-closed. |
| 23 | `custom_trainers/onlinecp_raw_resampling.py` | 509 | 6370 | 17/1 | 전역 cubic basis/delta, 전체 label-mixture, HU→normalize→resample, separate-axis/clip 경계. CPU 수학 helper 재현. |
| 24 | `docs/audits/2026-09-05-levels-readonly-audit.md` | 185 | 6883 | 0/0 | 과거 commit 감사. 과거 결함을 최신 소스에 그대로 재인용하지 않고 현재 구현과 대조. |
| 25 | `docs/cp_input_repair.md` | 139 | 7072 | 0/0 | Medical Data Aug 기준·후보 검색/no-placement·raw-target 표현 변경의 범위. |
| 26 | `docs/design.md` | 247 | 7215 | 0/0 | 전체 설계/보존/실행규칙. 일부 과거 effective-batch 표현은 현재 train.json보다 오래됨. |
| 27 | `docs/feedback_recovery.md` | 295 | 7466 | 0/0 | 검증된 source의 보존·recovery/continuation 조건. 모든 실패가 자동재개 가능한 것은 아님. |
| 28 | `docs/full_edge_training.md` | 108 | 7765 | 0/0 | full-edge streaming/chunking을 그래프 축소와 구분. 실제 GPU peak는 미측정. |
| 29 | `docs/level_v3_implementation.md` | 102 | 7877 | 0/0 | L1 final readout/L2 conditioning/shortcut mask 의도 대조. topology shortcut 잔존은 별도 F1. |
| 30 | `docs/online_bank_performance.md` | 190 | 7983 | 0/0 | ordered CPU 준비·spool/mmap·자원/heartbeat 기록. 완료 receipt와 진행 로그를 구분. |
| 31 | `docs/online_cp_curriculum.md` | 123 | 8177 | 0/0 | 과거 rank curriculum 경로. actual difficulty feedback으로 소급 재표기하지 않음. |
| 32 | `docs/online_cp_feedback.md` | 189 | 8304 | 0/0 | 현재 two-GNN와 train-only 관측·품질 gate/feedback 효과 혼재를 명시. |
| 33 | `docs/online_evaluation_verification.md` | 127 | 8497 | 0/0 | 기존 예측 재평가/새 output/provenance. current feedback training receipt의 end-to-end 연결과 구분. |
| 34 | `gpt_handoff.md` | 367 | 8628 | 0/0 | 별도 업로드와 byte-identical. 원격 상태·과거 테스트 수는 현재 실행 증거가 아님. |
| 35 | `hiercp/__init__.py` | 3 | 8999 | 0/0 | 패키지 진입 정보. |
| 36 | `hiercp/cache.py` | 2638 | 9006 | 56/0 | prototype/cache provenance, eligible donor, 후보 부족·no-placement 구분, training/inference sample 구축·publication. |
| 37 | `hiercp/cache_migration.py` | 331 | 11648 | 9/0 | 기존 failed cache 보존·명시적 migration 계약; 무조건 complete로 변경하지 않음. |
| 38 | `hiercp/causality_overlap.py` | 230 | 11983 | 10/0 | 공통 context와 후보 causal perturbation의 overlap 경계. cache/그래프 정의와 함께 해석. |
| 39 | `hiercp/common.py` | 1154 | 12217 | 43/6 | 원본 I/O·mask·후보 검사·완전 검색 확장·정확한 no-placement 증거. bool 거리연산 확인. |
| 40 | `hiercp/contracts.py` | 48 | 13375 | 3/0 | inner/outer/prototype case-ID 집합 경계 및 current architecture 강제. 실제 인물 identity 증명은 없음. |
| 41 | `hiercp/curriculum.py` | 381 | 13427 | 11/1 | 원위치 positive와 category별 negative/corruption pretext. 임상적 CP quality 정답은 아님. F1 연결. |
| 42 | `hiercp/data.py` | 500 | 13812 | 23/3 | 가변 후보/노드 disjoint-union batching, cache materialization 및 worker epoch 전달. |
| 43 | `hiercp/donor_preflight.py` | 156 | 14316 | 6/0 | raw donor eligibility·header geometry 전검사. 학습 cohort 자체를 donor 여부로 삭제하지 않음. |
| 44 | `hiercp/feedback.py` | 718 | 14476 | 32/3 | difficulty hierarchy·관측 BCE·graph provider·매 epoch update/predict. F4 자원/I/O와 F2 확장 recovery 창. |
| 45 | `hiercp/geometry.py` | 1135 | 15198 | 34/1 | 기존 geometry API와 입력/파라미터 계약. 활성 full_v22의 physical footprint는 spatial.py 경로와 구분. |
| 46 | `hiercp/hierarchy.py` | 605 | 16337 | 16/0 | L1/L2 graph 구성. source hosted_by-region topology 잔존 F1. |
| 47 | `hiercp/local.py` | 538 | 16946 | 14/4 | source/target 이종 local graph, canonical full edges, source→target 대응 관계와 candidate 구성. |
| 48 | `hiercp/loss.py` | 251 | 17488 | 11/1 | 가변 길이 후보의 batched CE·pairwise·ordinal·mining; consistency 결합 경로 대조. |
| 49 | `hiercp/model.py` | 1647 | 17743 | 61/12 | CNN·GATv2·sigmoid compatibility·L1/L2 conditioned readout·ablation/chunking. F1/F8 및 autograd probe. |
| 50 | `hiercp/nifti_geometry.py` | 167 | 19394 | 6/0 | 선택 affine의 voxel-cell 전역 extent와 양방향 voxel/mm 오차. registration 자체를 수행하지 않음. |
| 51 | `hiercp/pipeline.py` | 3481 | 19565 | 40/2 | quality 준비·학습·validation 선택·last/best checkpoint·생성 경로. main calibration과 feedback 차이 F4. |
| 52 | `hiercp/preparation_runtime.py` | 365 | 23050 | 11/1 | CPU/cgroup/RAM 관측, measured ordered worker 실행. 전체 수치 검증 대신 자원 정책. |
| 53 | `hiercp/prototype.py` | 238 | 23419 | 8/1 | inner-train liver-context clustering, feature 정규화/assignment/fingerprint. cancer-type clustering 아님. |
| 54 | `hiercp/region.py` | 733 | 23661 | 22/1 | liver 전체 partition, tumor-erased context descriptor, compact region cache. feedback cold path는 이 cache 미사용. |
| 55 | `hiercp/sample.py` | 678 | 24398 | 13/0 | context seed+interface+hop closure, full-edge sample view, fixed scoring view. 384를 최종 노드 cap으로 해석하면 안 됨. |
| 56 | `hiercp/schema.py` | 343 | 25080 | 3/1 | node/relation/dimension·budget·physical contract 정의, explicit fail 경계. |
| 57 | `hiercp/spatial.py` | 897 | 25427 | 26/3 | 활성 full_v22의 full footprint·anisotropic physical transform·확장 ROI·radius edges. 단순 clipping fallback 아님. |
| 58 | `hiercp/split.py` | 68 | 26328 | 3/0 | case-ID split 저장/검사. 같은 사람의 복수 case ID grouping은 별도 자료 필요. |
| 59 | `hiercp/tensor.py` | 555 | 26400 | 19/2 | load/checkpoint/batch device·hash/직렬화 경계. 외부 checkpoint는 trusted 입력 전제. |
| 60 | `hiercp/training_resources.py` | 111 | 26959 | 5/0 | 전체 cache 입력량·host budget·largest-first 대표 측정. feedback _calibrate와 대조. |
| 61 | `pyproject.toml` | 23 | 27074 | 0/0 | 패키지/엔트리 정의와 내부 참조 확인. native 환경 lockfile 검증은 별개. |
| 62 | `requirements-debug.txt` | 10 | 27101 | 0/0 | DEBUG 의존성·native 통합 경계. 누락 환경에서는 suite 전체 통과 불가. |
| 63 | `requirements.txt` | 5 | 27115 | 0/0 | 현재 nnU-Net/PyG 환경을 전제로 한 의존성 파일. 이 감사 환경에 모두 설치돼 있지 않음. |
| 64 | `run.py` | 875 | 27124 | 23/1 | 최상위 target/profile/규모 guard·stage 연결. full→tools.nnunet all의 late guard F3. |
| 65 | `tests/test_ablation_comparability_debug.py` | 178 | 28003 | 13/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=8. |
| 66 | `tests/test_bank_geometry_preparation_debug.py` | 247 | 28185 | 10/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 67 | `tests/test_bank_parallel_preparation_debug.py` | 215 | 28436 | 13/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 68 | `tests/test_bank_progress_debug.py` | 76 | 28655 | 7/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=4. |
| 69 | `tests/test_bank_scoring_canonical_debug.py` | 43 | 28735 | 4/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=2. |
| 70 | `tests/test_bank_scoring_resources_debug.py` | 341 | 28782 | 22/3 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 71 | `tests/test_batch_calibration_debug.py` | 112 | 29127 | 13/5 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 72 | `tests/test_cache_recovery_debug.py` | 357 | 29243 | 24/2 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 73 | `tests/test_cache_training_contract_debug.py` | 203 | 29604 | 13/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 74 | `tests/test_candidate_diagnostics_debug.py` | 263 | 29811 | 14/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=11. |
| 75 | `tests/test_candidate_pool_reuse_debug.py` | 128 | 30078 | 9/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 76 | `tests/test_causality_bank_contract_debug.py` | 331 | 30210 | 17/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 77 | `tests/test_causality_checkpoint_contract_debug.py` | 268 | 30545 | 13/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 78 | `tests/test_causality_overlap_debug.py` | 166 | 30817 | 14/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 79 | `tests/test_causality_preflight_debug.py` | 322 | 30987 | 26/5 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 80 | `tests/test_causality_resources_debug.py` | 253 | 31313 | 18/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=14. |
| 81 | `tests/test_causality_transform_memory_debug.py` | 304 | 31570 | 24/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 82 | `tests/test_causality_transforms_debug.py` | 262 | 31878 | 17/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 83 | `tests/test_cp_exact_masks_debug.py` | 125 | 32144 | 5/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=5. |
| 84 | `tests/test_curriculum_bank_contract.py` | 328 | 32273 | 28/3 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=22. |
| 85 | `tests/test_curriculum_launch_debug.py` | 48 | 32605 | 1/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=1. |
| 86 | `tests/test_donor_geometry_debug.py` | 248 | 32657 | 31/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 87 | `tests/test_donor_preflight_debug.py` | 187 | 32909 | 19/2 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 88 | `tests/test_donor_usage_debug.py` | 57 | 33100 | 6/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 89 | `tests/test_downstream_level_audit.py` | 575 | 33161 | 43/2 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=31. |
| 90 | `tests/test_downstream_reuse.py` | 294 | 33740 | 29/3 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=22. |
| 91 | `tests/test_edge_attention_streaming_debug.py` | 218 | 34038 | 16/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 92 | `tests/test_feedback_bank_upgrade_debug.py` | 558 | 34260 | 34/2 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=26. |
| 93 | `tests/test_feedback_experiment_debug.py` | 840 | 34822 | 51/3 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=26; errors=16. |
| 94 | `tests/test_feedback_gnn_debug.py` | 290 | 35666 | 19/2 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 95 | `tests/test_feedback_graph_binding_debug.py` | 126 | 35960 | 2/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 96 | `tests/test_feedback_launch_debug.py` | 111 | 36090 | 4/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=4. |
| 97 | `tests/test_feedback_trainer_integration.py` | 400 | 36205 | 37/4 | 회귀 테스트 정의·대상 계약/의존성 확인. skipped=1. |
| 98 | `tests/test_full_edge_view_debug.py` | 284 | 36609 | 13/2 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 99 | `tests/test_hierarchy_loss_debug.py` | 137 | 36897 | 9/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=6. |
| 100 | `tests/test_hierarchy_model_debug.py` | 328 | 37038 | 14/1 | 회귀 테스트 정의·대상 계약/의존성 확인. skipped=8. |
| 101 | `tests/test_level0_geometry.py` | 275 | 37370 | 25/3 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=14; skipped=5. |
| 102 | `tests/test_medical_aug_reference_debug.py` | 102 | 37649 | 7/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=5. |
| 103 | `tests/test_no_placement_cache_debug.py` | 155 | 37755 | 11/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 104 | `tests/test_online_bank_wiring_debug.py` | 377 | 37914 | 26/4 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 105 | `tests/test_online_cp_curriculum.py` | 281 | 38295 | 21/3 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=16. |
| 106 | `tests/test_online_cp_feedback_metrics.py` | 295 | 38580 | 26/2 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=14; skipped=8. |
| 107 | `tests/test_online_cp_feedback_policy.py` | 321 | 38879 | 27/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=21. |
| 108 | `tests/test_online_eval_io_regression.py` | 182 | 39204 | 18/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=11. |
| 109 | `tests/test_online_eval_provenance.py` | 222 | 39390 | 18/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=14. |
| 110 | `tests/test_online_eval_statistics.py` | 161 | 39616 | 9/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=8. |
| 111 | `tests/test_online_no_placement_debug.py` | 330 | 39781 | 17/2 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 112 | `tests/test_online_raw_bank_preparation_debug.py` | 105 | 40115 | 5/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=3. |
| 113 | `tests/test_online_source_mapping_debug.py` | 232 | 40224 | 21/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 114 | `tests/test_online_support_dispatch_debug.py` | 172 | 40460 | 14/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=8. |
| 115 | `tests/test_paired_gnn_cli_debug.py` | 114 | 40636 | 6/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 116 | `tests/test_preparation_recovery_debug.py` | 298 | 40754 | 21/2 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=14; errors=2. |
| 117 | `tests/test_preparation_runtime_debug.py` | 243 | 41056 | 25/2 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=18. |
| 118 | `tests/test_preprocessed_storage_debug.py` | 37 | 41303 | 2/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=2. |
| 119 | `tests/test_raw_bank_consumer_guards_debug.py` | 97 | 41344 | 8/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 120 | `tests/test_raw_bank_shared_sources_debug.py` | 86 | 41445 | 4/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=3. |
| 121 | `tests/test_raw_bank_storage_debug.py` | 77 | 41535 | 4/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=4. |
| 122 | `tests/test_raw_bank_type_boundary_debug.py` | 120 | 41616 | 5/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=4. |
| 123 | `tests/test_raw_cp_resampling_debug.py` | 258 | 41740 | 19/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 124 | `tests/test_raw_cp_trainer_debug.py` | 335 | 42002 | 21/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 125 | `tests/test_raw_cp_trainer_native_debug.py` | 316 | 42341 | 11/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 126 | `tests/test_recovery_orchestration_debug.py` | 406 | 42661 | 23/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 127 | `tests/test_retained_generation_debug.py` | 91 | 43071 | 7/1 | 회귀 테스트 정의·대상 계약/의존성 확인. errors=1. |
| 128 | `tests/test_smoke_view_contract_debug.py` | 48 | 43166 | 5/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=4. |
| 129 | `tests/test_training_resources_debug.py` | 50 | 43218 | 3/1 | 회귀 테스트 정의·대상 계약/의존성 확인. passed=2. |
| 130 | `tools/__init__.py` | 0 | 43272 | 0/0 | 도구 package marker. |
| 131 | `tools/ablation.py` | 859 | 43276 | 26/0 | GNN ablation profile·공정비교·별도 결과 경로. 새 feedback independent-effect 대조군을 자동 제공하는 것은 아님. |
| 132 | `tools/assemble.py` | 517 | 44139 | 14/1 | 검증 cohort·source/destination/hash·collision 사전검사 후 데이터셋 조립. |
| 133 | `tools/audit.py` | 229 | 44660 | 2/0 | 정적 감사와 architecture 계약 검사. 자체 audit 성공은 native 실행 증명이 아님. |
| 134 | `tools/case.py` | 275 | 44893 | 5/0 | 단일 NIfTI component/geometry 진단. 학습/평가 완료 도구 아님. |
| 135 | `tools/causality.py` | 2004 | 45172 | 55/0 | 실제 checkpoint perturbation 감사. upper_position_noise는 source-host topology를 바꾸지 않음 F1. |
| 136 | `tools/causality_resources.py` | 311 | 47180 | 14/1 | causality 자원·progress·소유 로그 경계. CPU synthetic resource tests 실행. |
| 137 | `tools/downstream_level_ablation.py` | 2133 | 47495 | 69/3 | 과거 exact-argmax Full/w/o-L1/w/o-L2 재채점·평가. raw feedback bank 거절은 정상 경계. |
| 138 | `tools/env.py` | 286 | 49632 | 4/1 | 원본 dataset/import/CPU/GPU/RAM/storage 환경 점검. 현재 원격 측정은 미수행. |
| 139 | `tools/feedback_bank_upgrade.py` | 687 | 49922 | 26/0 | 새 bank/runtime/views 생성·source preservation·legacy failed-row 호환성. 26개 관련 DEBUG 성공 확인. |
| 140 | `tools/feedback_preparation_recovery.py` | 600 | 50613 | 18/0 | 기존 준비 결과를 새 root로 회수·검증. native artifacts를 단순 marker만으로 승인하지 않음. |
| 141 | `tools/install.py` | 122 | 51217 | 4/0 | PyG 설치 안내/명령 경로. 이번에는 대상 환경에 설치하지 않음. |
| 142 | `tools/nnunet.py` | 626 | 51343 | 38/4 | legacy/offline orchestration. globally generated CP 거절과 full 문서 충돌 F3. |
| 143 | `tools/online_bank_preparation.py` | 102 | 51973 | 5/1 | 공유 source 준비·candidate graph worker dispatch. 원래 후보/그래프를 줄이는 최적화 아님. |
| 144 | `tools/online_bank_progress.py` | 96 | 52079 | 9/1 | heartbeat/progress/phase 로그. bank complete evidence와 분리. |
| 145 | `tools/online_cp_argmax_benchmark.py` | 3864 | 52179 | 97/3 | 과거 multi-pool argmax-v3 orchestration, source-anchored contract. current feedback과 혼용 금지. |
| 146 | `tools/online_cp_benchmark.py` | 4027 | 56047 | 103/4 | outer-train planning·raw bank builder·all eligible source slots·bank native acceptance/provenance. |
| 147 | `tools/online_cp_curriculum.py` | 137 | 60078 | 2/0 | current feedback sidecar도 같은 publisher 사용. final 파일 작성 후 검증/재시도 실패 F2. |
| 148 | `tools/online_eval_provenance.py` | 295 | 60219 | 15/1 | prediction/GT/cohort/definition SHA 및 새 출력. training checkpoint linkage까지 증명하지 않음. |
| 149 | `tools/online_eval_v2.py` | 1760 | 60518 | 46/5 | thresholded matching·patient-cluster inference·explicit trainer overrides. invalid-edge tie break F5. |
| 150 | `tools/online_raw_bank_preparation.py` | 153 | 62282 | 7/0 | case baseline native 대조·128 target payload·disk/resource 기록·zero support 유지. |
| 151 | `tools/online_scoring.py` | 469 | 62439 | 19/2 | ordered physical-batch scoring·mmap spool·batch calibration. 전체 GPU 처리량은 미측정. |
| 152 | `tools/online_trainer_contract.py` | 62 | 62912 | 1/0 | trainer/base source SHA와 지원 bank 선언 검사. |
| 153 | `tools/paired_benchmark.py` | 2263 | 62978 | 61/5 | fold-specific GNN/공통 preparation의 상위 연결. legacy offline all과 구분. |
| 154 | `tools/regress.py` | 288 | 65245 | 8/2 | 합성 regression harness. optional-import except는 실제 CP 실패를 조용히 숨기는 증거가 아님. |
| 155 | `tools/run_feedback_experiment.py` | 830 | 65537 | 31/0 | 현재 fresh/recovery/upgrade 단계 driver. F2 stage retry, F6 fresh resume, F7 평가 종결 경계. |
| 156 | `tools/smoke.py` | 678 | 66371 | 7/0 | synthetic connectivity/batch/chunk/shortcut tests. production-sized opt-in smoke는 이번 미실행. |
| 157 | `tools/train_online_curriculum.py` | 81 | 67053 | 2/0 | 과거 rank-only trainer의 strict launch. 새 raw feedback의 대체 launcher 아님. |
| 158 | `tools/train_online_feedback.py` | 93 | 67138 | 2/0 | 현재 Full/Basic trainer 연결·raw mapping·resume checkpoint 존재 확인·native run_training 호출. |
| 159 | `tools/validate.py` | 769 | 67235 | 25/0 | shape/affine/label/finite/tumor-change/cohort 검증 및 atomic 결과 publication/rollback. |
