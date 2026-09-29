# v2.2 support startup runtime correction

The supplied server completed all14,102 region records on f07b13f and entered `calibration support` (43/353 after5:09, about6.96s/batch). This is a different stage from the previous preparation-overlap patch. The supplied GPU snapshot was4%/2128MiB; host RAM/swap and CPU snapshots did not show system-wide capacity exhaustion. They do not identify a server function-level bottleneck or rule out per-process CPU/cgroup/I/O restrictions.

## Measured and corrected

A local cProfile of eight real cached CT records loaded sequentially took2.451s with profiling overhead; 104 `unique_dim` calls consumed1.607s in duplicate-edge validation. The original predicate used `torch.unique(edge.T,dim=0)` for every typed relation. The replacement counts unique injective int64 keys `source * target_count + target` after range checks. It retains shape/type/endpoint checks and falls back to the exact two-column predicate if the key range could overflow. No checks, nodes or edges are discarded. CPU/CUDA comparisons include duplicate edges, empty inputs, invalid endpoints and overflow collisions.

The runtime reader hashes a single byte buffer and deserializes that same buffer using `weights_only=True`; it no longer opens the tensor payload once for hashing and again for loading. All binding/content/structural validation remains. CPU work remains significant; this is not a claim that all loading overhead is eliminated.

Calibration now sets its CNN execution chunk to the explicit smallest physical-batch candidate instead of inheriting chunk4. It returns the complete ordered support rather than discarding it. Both the initial model and support hashes are checked before/after clone-based batch calibration. A new run starts optimization with that exact memory, avoiding a second full initial-support forward pass. Support is still refreshed after optimization and for the selected final model. L1/L2, ranking loss, CNN/GNN architecture, dataset, candidate128, CP and masks are unchanged. The change in CNN execution chunk is logged; it is not a width/depth or query-batch reduction.

Tqdm and support_timing.jsonl now show loader wait, H2D+validation, forward and actual CNN chunk. Completed reviewed preparation caches, including f07b13f, are accepted read-only by the training dataset after source verification. Index, bindings and tensors are not rewritten or copied. No preparation command is needed to use this patch. Unknown sources and incompatible geometry/profile remain rejected. Existing optimizer checkpoints from a different runtime are NOT automatically migrated or described as exact resumes.

## Short local evidence

- Existing f07b13f DEBUG cache, all8 inner_train records, physical batch8, workers8, RTX5070Ti. Baseline reproduces both the old two-read reader and original two-column unique predicate. Alternating timings: old cold2.2487s, new1.1739s, old warm1.6936s, new repeat1.3423s. Warm improvement about21–31%; too few batches to predict the A6000 cohort or epoch. Loader wait old warm1.3807s vs new0.9083/0.9646s. Output embeddings matched exactly in this fixture.
- Calibration support was constructed once, returned unchanged and covered all8 records. Model content hash unchanged after clone updates. No cache reconstruction or source-file edits occurred.
- Fresh-process actual-CT smoke PASS: uninterrupted 4 DEBUG updates versus pause after update 1 and resume in a new process produced identical final model, optimizer, RNG and state hashes. Refresh, validation, best selection and final artifact paths ran; calibration covered all 8 supplied train records exactly once. This is a short DEBUG run, not full-cohort training.
- 52 unit/regression checks PASS, including CPU/CUDA duplicate detection and existing cache/policy/resume/sparse-gradient contracts.
- Evidence is stored in `validation/region_support_runtime_20260930`. Full training, A6000 timing, segmentation accuracy and whole-epoch speedup remain unmeasured. The old server process is not patched in place.

## Safe server application

Keep the completed cache at `work/regions_frozen_reuse_20260929_230052/cache/index.json`. Do not rerun prepare. Switch source only after the foreground process has returned. If it is still in calibration, no optimizer update has occurred; a fresh train invocation can use this existing cache and a new output directory. If optimization has already begun, preserve the checkpoint and do not claim that a fresh invocation continues its steps. Retain physical GPU2 assignment, workers16, CUDA40/RSS192/resident128GiB, candidates32/48/64 and research-report. Do not change the running checkout under its process.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 서버후보 유지, calibration CNN chunk 연결.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 제공 자료와 로컬 GPU/CPU 프로파일 구분.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 원인은 support 검증 비용 분석.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위검사와 실제 CT 측정 구분.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 GPU 및 회귀 검사.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
