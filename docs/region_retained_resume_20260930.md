# Retained activations and mixed-origin training resume — 2026-09-30

## Server failure and repair

The supplied server f07b13f run stopped fetching a batch after step73 with ValueError: Mixed partition contract. The earlier CSR beta warning is not the terminating exception. Preparation explicitly preserves reused item source fingerprints, while collate previously required every fingerprint in a batch to be identical. A later patient-grouped training batch can cross that reuse boundary even when contiguous support batches succeeded.

The repair reconstructs the allowed historical binding digests from the cache's verified current and reused source manifests. Each origin must match an explicitly reviewed preparation revision and each index entry must refer to one of those digests. Loader passes that verified set into collate. Collate retains each original binding, rejects unknown sources, and still requires identical cache/CNN/profile/view/coordinate/feature-evidence contracts. It does not rewrite receipts, skip records, rebuild regions or mix incompatible models. The mismatch message now identifies the field.

Actual CT regression combined unchanged items from reviewed f07b13f/6d5f0dc caches in an alternating DEBUG index. The unverified combination reproduced rejection; verified loading, full supplied DEBUG train/val, final export and fresh-process pause/resume passed. Final model, Adam, RNG and state hashes matched between uninterrupted and resumed runs. Original tensors/bindings were preserved. No partition generation was run.

## A6000 execution path

The real train CLI now accepts --activation-storage retained. It disables CNN and L0 block activation checkpointing and the L1/L2 recomputation branch, storing activations instead. The flag is per-model execution state, recorded in checkpoint identity and execution contract. No production monkeypatch, change of layers, width, graph, loss, precision, candidate128, observations, masks or BasicCP. The base config still supports checkpointed execution explicitly. The CLI allocator and Budget enforce the requested CUDA/RSS limits; there is no automatic model/batch shrink or silent checkpointing fallback on OOM.

For the user's assigned46GiB-class A6000, the command retains CUDA40GiB/RSS192GiB/resident128GiB, workers16 and existing batch candidate contract32/48/64. A resumed run retains its saved physical batch32 and existing support/plan/cursor, without calibration or initial support rebuilding. This is not a promise to allocate40GiB or a measured A6000 worst-case admission.

Tqdm displays actual update seconds and peak GiB. update_timing.jsonl separates loader wait, transfer+validation, forward, backward, checks/clip/optimizer and synchronous checkpoint saving. Overall step seconds include support-plan work, CPU overhead and save; GPU event spans include queued execution delays and are not kernel-only attribution.

## Explicit checkpoint continuation

--resume-execution-upgrade permits only the reviewed execution change. Checkpoint content hash is verified before loading. Dataset/cache hash, model configuration, epochs, precision, ranking, workers, resources and batch candidates must match. Historical runtime manifests are checked against pinned Git releases, and unchanged graph/loss modules must still match current source. The core model compatibility exception is exactly the reviewed old/new model.py digests for the checkpoint-support execution flag; default equations/weights remain unchanged.

A preparation-overlap helper and its caller that differ from f07b13f are accepted only if CURRENT contents equal the reviewed b398b3d Git blobs. Automatic review rejected an earlier filename-only addition; the final implementation enforces this current-content check, with rejection regression coverage. No unresolved approval blocker remains.

All weights, Adam, RNG, support, plan, selected best, epoch and next-batch cursor are restored. execution_upgrade.json records the transition; it is not labelled exact replay of the old runtime. A real old DEBUG optimization checkpoint was resumed from step1 through step4 in retained mode with no support regeneration. Its final model/Adam/RNG/state hashes also matched the prior checkpointed full run in this fixture. The f07b13f complete source manifest was checked separately from Git; the user's actual server checkpoint was not available locally.

## Verification and remaining limits

52 regression tests passed, then the strengthened source guard suite (6 tests) passed including current-content mismatch rejection. Actual mixed-origin CT training/validation/final export and separate-process resume passed. AST checks passed. Evidence: validation/region_retained_resume_20260930. A6000 full-batch/full-support time and peak memory remain to be measured by the user's actual run; no long training was launched locally or remotely. Partition-quality profile violations remain research-report, not clinical/production efficacy claims.

## Server use

The supplied traceback returned to the shell, so no process termination is required. Fetch and switch to the published fix, then run train with the existing cache and checkpoint_latest.pt, --resume-execution-upgrade and --activation-storage retained in a new output directory. Keep the original run and cache. Do not run prepare. Confirm startup reports retained, CNN/L0/L1_L2 checkpointing false, physical_batch32 and resumed_step matching the saved checkpoint (expected73; the file is authoritative). Ctrl+C requests save/pause in this foreground trainer. No command exits the shell or touches another process.

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
