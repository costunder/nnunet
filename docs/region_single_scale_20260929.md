# v2.2: remove the second region partition

The user explicitly requested removal of the second coarsening stage after its negligible additional compression, and asked whether stage 1 was correctly formed. The active prepare CLI now creates a single-scale region cache and accepts only `--reg-scale1`. `--reg-scale2` is rejected as an unknown argument. No dummy second assignment, second quotient, second graph transfer, second adjacency cache or two-scale readout is constructed on this path.

Pipeline: live CNN → fixed first partition aggregation → three relation-specific GraphSAGE blocks on the first quotient graph → one mass-aware readout → existing 128D fusion → unchanged L1/L2/loss. Removing a partition does not require reducing GNN depth: all three blocks, their gradients and optimizer connections remain. The explicit legacy two-scale diagnostic constructor remains available to test preservation of old behavior; it is not the new CLI preparation path. The new profile ID is `fixed_region_sage_single_scale_v1` with `region_scales=1`. Old cache/checkpoint source fingerprints are not overwritten or silently promoted.

## What stage 1 actually does

For each pair, role and existing shell, the adapter samples 32-channel frozen CNN features at the original fine-node coordinates. It L2-normalizes those features and calls the unchanged pinned official `merge_components_by_contour_prior` on existing same-role/shell adjacency. Official merge decisions use feature differences, original-node mass, contour weights and the explicitly supplied reg. They do not implement bbox or dispersion limits internally. Connected multi-node/chain merges are allowed. The adapter verifies role/pair/shell separation, connected clusters, complete mass coverage and the exact quotient of all 13 directed relation types. There is no new kNN graph and no fine node is discarded.

This verifies implementation and topology, **not semantic boundary quality**. The input CNN checkpoint was trained for the existing task, not established as a validated boundary-partition embedding. Explicit reg 0.02 and initial admission bounds remain uncalibrated. Similar features along connected adjacency can be merged into a large region; this is compatible with the official objective. Current evidence does not isolate CNN quality, reg and graph weighting as separate causal contributions, so no automatic reg tuning or algorithm substitution was made.

## Actual saved-pair comparison

The new single-scale and preserved old two-scale DEBUG preparation use the same frozen CNN contents, materialized CT/view and original observations. For all eight train pairs, cluster membership, original mass, physical bounds, stable IDs and all 13 quotient edge types are identical after canonical cluster relabeling. Raw numeric cluster IDs can be permuted; this is not proof of bitwise repeatability of cold official partition calls. Initial comparison by raw cluster number failed; canonical membership comparison resolved the distinction without changing partitions.

| Quantity, eight real pairs | Fine graph | First partition |
|---|---:|---:|
| Nodes | 52,666 | 13,868 |
| Directed edges | 2,761,602 | 379,049 |

The old second stage only reached 13,849 nodes (19 fewer). Its removal is therefore supported by observed marginal compression, but no A6000 epoch speedup is claimed here.

Stage-1 quality concern remains: `liver_72:84`, target_context has 3,150 fine nodes, of which 2,496 (79.24%) form one cluster spanning a 75.24mm bbox diagonal. This is a connected region, not a tumor diameter. Losing distinct local context within such a large pooled region is a concern, not a proven downstream accuracy loss. No claim that the first partition is semantically adequate is made.

## Validation scope and remaining blocker

Regression tests: 27 existing/report/preflight checks; four additional single-scale CUDA checks cover only one coarsening call, unchanged first topology, rejection of unused assignments, and nonzero finite gradients through CNN, all three SAGE blocks and L1/L2 with an optimizer update. Real CT smoke uses the existing DEBUG train8/val2 fixture, four updates, support refresh, validation, final selection and fresh-process resume. Model, optimizer, state and RNG match exactly. The CP smoke uses three explicit DEBUG candidates with original full-mask filtering and exact 2,368-voxel paste; it is not full 128-candidate/native segmentation training.

**Removing stage 2 does not remove the first-stage admission rejection.** All eight local train pairs still violate the initial profile in at least one way. The strict server path remains blocked at stage 1. No threshold increase, DEBUG promotion, failed-sample skip, training override or production-ready claim was added. The user has not approved changing the initial profile from a blocking condition to an advisory diagnostic. Do not supply another unchanged strict command as if this patch makes full training ready.

Basic CP, full observation inventory, production candidate128, whole donor mask eligibility, L1/L2, loss and training schedule are unchanged. No long training was started. Outputs are new directories under `work/region_single_scale_*_20260929`, preserving prior results. Final source-bound validation artifacts are recorded separately from the earlier intermediate smoke.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. SAGE3층·128D 유지.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자가 승인한 2차 병합 제거만 반영.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 DEBUG physical batch 후보8/16/32·workers8 계약 사용.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. RTX5070Ti, 명시적CUDA6/RSS12GiB·resident0.5GiB.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 오류 원인은 초기 admission임.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
