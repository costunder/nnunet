# Single-scale research execution with recorded profile violations

The user instructed proceeding after the concrete proposal to use the current first partition for learning while recording uncalibrated bounds as diagnostics. The explicit option is `--profile-policy research-report`, required on both prepare and train. The default remains `strict`. This supersedes the earlier pending-approval/blocking status for this explicitly selected research run; it does not declare partition quality validated.

The initial node/edge/role-shell/bbox/feature-variance thresholds and reg are unchanged. Exceeding these thresholds is saved as a violation in per-pair data, full audits, cache counts and execution metadata. Full-cohort research execution can continue with these flags. `full_training_admitted` remains false for this mode; `research_training_admitted` records its different execution policy, while `partition_quality_validated` remains false. DEBUG caches and checkpoints cannot be promoted to a full run. A cache's policy must match the explicit training option; the policy is bound to the checkpoint identity, model extra state and final artifact, so a different policy cannot be silently used for exact resume.

Pair/role/shell separation, materialization identity, connectivity, full-node mass coverage, coordinates, finite values, edge indices, relation quotients, tensor mutation checks, full original paste-mask eligibility, explicit CUDA/RSS/resident limits and disk reserve are still enforced. This option reaches only the existing profile-bound allowance; structural and resource exceptions are not caught or downgraded. A data/structure/resource failure still stops execution, preserving outputs.

The model stays CNN → fixed first partition → SAGE3 on the first graph → single readout → 128D → existing L1/L2 and loss. The preparation uses the existing complete paired cache and copies only the existing CNN snapshot. A new GraphSAGE experiment starts; this is not exact resumption of an old GAT model. All observations, 128 production candidates, Basic CP and the full mask checks remain unchanged. No nnU-Net training is automatically launched by this GNN entry point.

## Server invocation parameters

Use the pushed, pinned fix commit in `/home/aicompetition06/Medical/HierCP-regions-8580e59`, conda `nnunet`. Select the UUID obtained from physical GPU index 2 via `nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader` and export it as `CUDA_VISIBLE_DEVICES`. PyTorch then refers to this selected GPU as `cuda:0`.

Preparation: original `HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json`; CNN snapshot `HierCP-v22-e1e34bf/work/v22_gnn40_resume114_20260928/training/checkpoint_latest.pt`; physical prepare batch32, workers16, scale1 reg0.02, fixed view0, CUDA40GiB, RSS192GiB. Train: resulting region cache, workers16, CUDA40GiB, RSS192GiB, resident128GiB, physical batch candidates32/48/64, existing configured full epochs. Both commands require `--profile-policy research-report`. There is no scale2 argument. Use a new output directory and chain train after successful prepare with `&&`. Foreground preparation receives Ctrl+C normally; foreground training saves and pauses at a batch boundary.

## Validation limits

Completed: 39 unit/regression checks passed; actual CT smoke passed with all ten profile-violating records retained, four DEBUG optimizer updates and exact model/optimizer/state/RNG fresh-process resume. Three-candidate CP smoke passed exact 2,368-voxel paste and full-mask eligibility. Recomputed-digest mutations of final policy, quality-validation claim and DEBUG identity were rejected. These are local short checks, not full training results. Evidence is under `validation/region_research_policy_20260929`.

The local actual-CT smoke uses the existing DEBUG train8/val2 set, preparation batch8, workers8, CUDA6/RSS12GiB, resident0.5GiB, batch candidates8/16/32 and four DEBUG updates. It checks support, optimization, validation, final selection and exact fresh-process resume. Non-DEBUG policy routing is checked separately with explicitly synthetic policy manifests and constructor-wiring unit tests; no synthetic case is passed off as real training data. Native nnU-Net training, full cohort quality and A6000 throughput are not measured by this local smoke. Existing first-partition overmerging remains an experimental quality concern rather than being relabelled PASS.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 후보와 전체규모 설정 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 로컬RTX5070Ti GPU 검사, 명시적 자원 한도 유지.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 변경은 OOM 회피가 아님.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
