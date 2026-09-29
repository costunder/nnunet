# Saved fixed-partition inspection — 2026-09-29

Scope: the eight actual CT inner-train pairs saved by `regions_training_smoke_20260929_r8`, not a new partition or a training run. The saved partition, reg, min_size, admission limits, model, and prior results were not modified.

## Findings

| All eight pairs | Fine view | Scale 1 | Scale 2 |
|---|---:|---:|---:|
| Nodes | 52,666 | 13,868 | 13,849 |
| Directed edges, all 13 relations | 2,761,602 | 379,049 | 378,284 |

Scale 1 reduces node count by 73.67%. Scale 2 removes only another 19 nodes (0.137%). Three pairs have exactly unchanged scale-2 node and edge counts. Therefore an effective second spatial scale has not been demonstrated by these saved results.

The widest recipient-context region is `liver_72:84`, scale-1 cluster 229, shell 2. It merges 2,496 of that role's 3,150 fine nodes (79.24%). Its bounding-box extents are 52.19 × 43.37 × 32.50 mm, with a 75.24 mm diagonal. This is a bounding-box measurement, not a tumor diameter or maximum measured pairwise distance. The role has 231 clusters at both scales. Its cluster-size 25th, 50th and 75th percentiles are all one; a very large cluster coexists with many singletons. Scale 2 leaves this role's partition unchanged.

All saved clusters are connected in their original same-role, same-shell adjacency. Exact saved fine-node positions, grids, masses and member bounds were checked. No disconnected cluster, missing fine-node mass or mismatched record/view was found. This does not establish anatomical or recommendation quality: connectivity alone permits a wide region.

## Why scale 2 scarcely changes

The current fixed partition uses frozen CNN32 at both levels, with `reg_scale1=reg_scale2=0.02`. The first call already recursively merges energy-admissible components. Scale 2 uses the normalized mass mean of the **raw** frozen CNN features; it does not receive new SAGE or GAT features. This is not mathematically identical to reusing the first call's means of normalized features, so exact idempotence must not be claimed.

Using the same frozen snapshot, actual batched CUDA CNN output, exact fine view and saved first-stage assignments, an independent float64 evaluation of the pinned merge energy found only 13 admissible links among 128,721 undirected within-role/shell region adjacencies at the start of scale 2. No inspected value lay within 1e-6 of zero. These are reconstructed start energies, not a new official GPU merge trace. Saved scale-2 terminal diagnostics contain 65 nonempty groups with no energy-admissible merge, six groups with no remaining adjacency, and one empty shell. They do not indicate termination at the 32-iteration cap.

For the selected recipient role, the reconstructed scale-2 start has **zero admissible links out of 2,199**. This explains its unchanged 231 → 231 regions. Raising only the post-partition node/bbox limits would change admission status but would not change this partition or produce a second spatial scale.

The official energy uses feature discrepancy, original-node mass and accumulated adjacency weight. Bbox and feature-variance bounds remain post-partition checks; they are not constraints inside the official merger. A large connected component is therefore possible. The measured wide component is not by itself an implementation bug or proof of EZ-SP failure. Its usefulness and information loss have not been validated. The partition CNN comes from the existing DEBUG optimized snapshot, not an independently trained and validated boundary embedding model.

## Evidence and reproducibility

- `work/region_partition_inspection_20260929/report.json`: complete per-pair/role/cluster counts, bounds, cumulative original-feature variance, independent connectivity checks, saved diagnostics, reconstructed energies and input hashes.
- `work/region_partition_inspection_20260929/inspector.html`: all eight pairs and five roles; actual-coordinate fine, scale-1 and scale-2 graphs, cluster selection and normalized input-CT slices. No invented layout or graph sampling. The 3D panes display same-role relations; total 13-relation counts are separately labelled. Directed lines overlap visually and arrowheads are omitted.
- `work/region_partition_inspection_20260929/visual-check.json`: all 8 × 5 role views, scale selection, synchronized rotation, slice control, all-edge display, narrow layout and sandboxed inline rendering checked; no JavaScript errors.
- `tools/inspect_fixed_regions_visual.py`: read-only diagnostic exporter. Eight records are loaded with eight workers and encoded in one CUDA batch. The exact materialized batch and edge hashes match the saved preparation receipts.

Hardware: RTX 5070 Ti 16 GiB; 16 logical CPUs, approximately 38.9 GiB free RAM at start. CUDA peak allocated was 162,758,656 bytes. The measured analysis interval was 14.27 seconds before HTML serialization; this is not training/update performance. No backward, optimizer, partition rerun, production checkpoint, readiness marker, full training or full evaluation was run.

Visualization display rounding and CT grayscale windowing do not alter tensors used by validation. CT is the actual normalized 48³ model input, not raw HU or a whole-volume reconstruction. Cluster boxes are not tumor masks. The inline view focuses on the widest recipient-context example; the full viewer contains all eight pairs.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 저장된 DEBUG 8pair를 한 CUDA batch로 읽기 전용 분석했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행은 OOM 없이 끝났다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번 작업은 읽기 전용 분석이며 학습 연결 검사를 새로 수행하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
