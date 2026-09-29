# Reuse the prepared CNN without reopening its parent checkpoint

The server's `72be3ce` reuse command stopped in `torch.load(..., mmap=True)` while reading the historical GNN checkpoint: `OSError: [Errno 116] Stale file handle`. This occurred before the new output directory was created or any batch was reused. The traceback identifies a filesystem read failure, not a GPU OOM or partition rejection. The underlying storage condition was not inspected remotely or repaired by this change.

The old preparation directory already contains the exact frozen CNN state and the request that binds its file digest, tensor-content digest, configuration, DEBUG status, source revision and historical parent-checkpoint digest. Reuse now reads and validates those saved CNN bytes directly. It checks the reviewed source identity, matching config/mode, file SHA256, tensor tree hash and strict CNN state-dict shape/key loading. Deserialization uses an in-memory byte buffer with weights_only=True. No parent-checkpoint open, mmap or final parent-checkpoint hash is performed in the reuse path. The historical parent digest remains recorded as provenance; metadata explicitly states `cnn_load_basis=verified_prepared_snapshot` and `original_checkpoint_reopened=false`. The actual snapshot and request remain checked again by the existing reuse verifier and end-of-preparation guards.

This is an explicit source selection, not an exception fallback. `--reuse-prepared` must **omit** `--partition-checkpoint`; supplying both is rejected rather than silently ignoring one. Fresh preparation still requires and verifies an explicit original checkpoint. The graph, partition algorithm, CNN weights, SAGE/L1/L2, loss, batch/cohort/profile and mask rules are unchanged. The original prepared files are not edited. Continued failure when reading the snapshot or paired cache is reported normally; this patch does not claim all server storage paths are healthy.

## Verification

- 52 unit/regression tests PASS on local RTX5070Ti, including 7 new CNN-source checks: identical loaded tensors, DEBUG promotion rejected, corrupt file rejected, changed contents rejected despite a new file hash, shape mismatch rejected, unknown source rejected and ambiguous supplied parent rejected before output creation.
- Actual CT DEBUG train8/val2 partial-reuse probe PASS with the original-checkpoint loader patched to raise Errno116 on any call. It was called zero times. Completed8 were reused with identical decoded payloads; only incomplete2 invoked preparation. Original files remained unchanged and all10 profile violations remained recorded.
- The actual CLI prepare path without --partition-checkpoint is also exercised separately against the existing complete DEBUG cache. No long training or remote process was launched. The previous revision's four-update training/CP evidence remains historical; this loader patch does not claim a newly completed full training experiment.
- Evidence is in `validation/region_frozen_cnn_reuse_20260929`.

## Server retry

The failed command has already returned to the shell; there is no worker from that failed preparation to interrupt. Keep the previously populated `OLD_REGION_CACHE`, which points to the original `6be85aa` preparation directory, rather than to the failed new output. Fetch/switch to this fix, select physicalGPU2 as before, choose a fresh timestamped output and rerun prepare with --reuse-prepared, **without --partition-checkpoint**. Preserve prepare batch32, workers16, reg1=.02, view0 and CUDA40/RSS192. Chain the existing train command only after successful preparation, with resident128 and physical batch candidates32/48/64. Use research-report on both commands. The completed old preparation is reused after verification; no full restart is required.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 병렬 재사용·준비 및 서버 설정 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제GPU/peak CUDA, workers와 자원한도를 DEBUG 근거에 기록.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 오류는 체크포인트 파일 읽기.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 부정검사 synthetic CNN은 단위검사로 명시.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 CUDA 회귀검사 통과.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
