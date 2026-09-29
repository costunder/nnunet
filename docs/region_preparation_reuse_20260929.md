# Continue preparation after applying the packaging fix

The previous handoff offered only waiting or restarting the entire preparation. This implementation adds `prepare --reuse-prepared OLD_CACHE_DIR` and a **new output directory**. It reuses validated completed batches from the running server's `6be85aa` revision (or the packaging-only `6d5f0dc` revision), preserving original tensors, receipts, source identity and audits. It prepares only batches without a completion audit. The original directory is never edited or removed.

## Reuse contract

The original request must match the complete paired-cache digest, partition CNN checkpoint and CNN digest, configuration, fixed view, single-scale profile, DEBUG/full status and explicit profile policy. Its core source identity must match. Its preparation files must exactly match a pinned reviewed Git revision. All partition/integrity modules outside the reviewed preparation/packaging changes must still match the current files. Missing Git history or an unknown revision causes an explicit failure, not a source-hash override.

The existing writer saves its batch audit after all pair files and sidecars. That audit is the completed-batch marker. Reuse validates the whole original batch: original record/view bindings against the paired dataset, sidecar and file digests, admission receipts, coordinates, role/shell coverage, mass, bounds, finite values and all typed edge checks. The saved per-pair and whole-batch diagnostics, original batch owner positions and common fine-graph/materialization evidence must agree. Changed preparation batch size, missing files under a completed audit, corrupt contents or inconsistent diagnostics fail explicitly. A batch without an audit is unfinished and is regenerated in the new directory; its partial files remain in the old directory.

The new index records the current orchestration source plus `reused_preparation`, including the original request/source revision/digest, reused batch count and audit digests. Reused item bindings **retain their original preparation digest**. They are not stamped as newly partitioned. No graph reconstruction or official merge runs for completed batches. Integrity verification and writing the checked result to the new directory still take time and storage. Progress shows cumulative `reused` and `new` record counts.

This transition does not restore the old merger RNG cursor: the old preparation did not save it. Completed partitions are exact; previously unfinished partitions are generated under the same algorithm/settings, without a claim of matching an uninterrupted run's future cluster realization. This is preparation reuse, not exact resumption of a GNN optimizer checkpoint. Reusing an already migrated cache as the source of another migration is currently rejected explicitly. For another interruption of this transition, retain the original source directory; automated merging of multiple partial generations is outside this implementation.

## Local verification

Actual existing DEBUG train8/val2 from `6be85aa`, RTX5070Ti, physical preparation batch8, workers8, CUDA6/RSS12GiB. A separate interrupted-copy fixture retained the complete train batch and an intentionally incomplete val file. Eight original train items were reused bit-for-bit at the decoded payload level. Only the two validation items invoked materialization/partition. Ten profile violations remained recorded. Original file digests were unchanged. Different policy, view, DEBUG status, CNN digest, batch size, corrupt complete file and unknown source revision were rejected.

The recovered mixed-origin cache completed four DEBUG updates, full supplied DEBUG support/validation/final-memory selection and final export. Existing45 regression tests passed. Actual three-candidate CP verification is recorded separately with the full original-mask oracle. These are short local execution checks, not full training, full-cohort efficacy, A6000 throughput or native nnU-Net training. Evidence: `validation/region_preparation_reuse_20260929`.

## Server transition

Only if the server is still in `prepare`: interrupt the foreground task with **Ctrl+C once**, then wait for its shell prompt. Do not change source files while it is still unwinding. No signal-to-PID, parent shell, SSH or session termination command is needed. The previously chained training must not be manually started against an incomplete cache.

In the same terminal, `CP_RUN` is the output variable from the previous supplied command. Confirm `test -f "$CP_RUN/cache/request.json"` succeeds before changing it. Preserve its absolute path as `OLD_REGION_CACHE`; fetch and switch to the pinned reuse-fix revision only after preparation has stopped. Select physical GPU2 by UUID exactly as before. Run `prepare --reuse-prepared "$OLD_REGION_CACHE"` with the original paired-cache path, same CNN checkpoint, batch32/workers16/reg1=.02/view0/CUDA40/RSS192 and research-report. Use a new timestamped output directory. On successful completion, run train on that new index with workers16/CUDA40/RSS192/resident128 and batch candidates32/48/64. Both commands must use research-report, no DEBUG option. Neither the old directory nor its results are overwritten.

If training has already started rather than preparation, this preparation transition is not an exact training-checkpoint resume command. Preserve that training checkpoint and do not silently start a fresh optimizer in its place.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존batch 경계 유지, 병렬 파일 검증과 저장.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제GPU/peak CUDA, CPUworkers와RAM한도 기록.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 작업은 완료된 준비의 재계산 제거.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 의도적 손상파일은 별도 부정검사 복사본만 사용.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
