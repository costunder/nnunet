# v2.2 preparation overlap and stage timing

## Confirmed server stage

The supplied PID 3614482 is `run_fixed_regions.py prepare`, not training. It uses physical preparation batch32, workers16, reg1=.02, fixed view0, CUDA40GiB/RSS192GiB and research-report. It reuses `work/regions_sage1_20260929_213659/cache` and writes `work/regions_frozen_reuse_20260929_230052/cache`. The supplied snapshot reports A6000 6330MiB process memory and GPU utilization10%; `ps` reports lifetime-average CPU131% after45minutes. These snapshots do not identify an exact dominant server stage. In particular, training checkpoint writes are not part of this running process. The CUDA budget is a ceiling, not a target allocation.

## Implemented scope

The original prepare loop finished CPU loading, GPU preparation and verified disk writes before starting the next batch. The new loop overlaps one next CPU batch and one previous CPU save with the current GPU batch. GPU calls remain on the main thread. Input order, batch boundaries, official merger, role/shell separation, first-scale partition, graph coverage, reg, CNN weights, SAGE, L1/L2, losses, candidate128, Basic CP and mask checks remain unchanged. The independently sealed loader receipt supplies the already-read donor component ID instead of opening each record again for that field.

Outstanding queues are bounded to one lookahead and one save, with RSS/resource checks retained. A completed batch audit is written only after all validated item writes succeed. Read/write errors propagate, and no completed index is written on failure. End-of-run source/input checks remain. Tqdm now shows load_s, wait_s, prepare_s and save_s; preparation_timing.jsonl also contains H2D, official merge, save-wait, resources and reused/new status. Overlapped spans cannot be summed to obtain elapsed time.

Repeated recovery accepts only current or pinned, reviewed source identities (6be85aa, 6d5f0dc, 72be3ce, f07b13f). Every record keeps its original source binding; unknown sources, changed profile/view/CNN, missing audits, changed payloads and incompatible batch boundaries remain errors. Already-reused and newly-produced batches can coexist. The parent GNN checkpoint is not reopened. Only immutable Git blob identities are memoized; live source hashes remain fresh.

## Measurement and limits

- Baseline instrumented actual CT batch8 on local RTX5070Ti: 3.7816s total; CPU read/materialize/collate/hash0.6106s; H2D0.0122s; preparation2.4075s; save0.6986s. Nested within preparation: CNN0.1037s, partition/quotient/diagnostics1.7746s, official merge0.9919s, final quotient check/packaging0.2731s. Do not sum nested spans.
- Entire existing DEBUG train8/val2, same physical batch4 and workers8 in both comparison arms: serial cold8.4423s, overlap6.4935s, overlap repeat6.4010s, serial warm7.1310s. Warm reduction about9–10%; small sample, not an A6000 batch32 or cohort estimate. No claim that the server low utilization or two-hour preparation is solved.
- Initial raw-ID comparison failed. The serial repeat also changes cluster numbering. After aligning by original stable member identity, all CT payloads, memberships, region attributes and typed quotient edges match exactly across all10 records and arms. Saved outputs are not renumbered; cold bitwise partition repeatability is not claimed.
- Partial recovery reused4, prepared only the remaining4+2, then a second recovery reused all10 without a partition call. A separate f07b13f chained-recovery probe reused10 with bitwise-identical decoded records, zero original-checkpoint reads and unchanged old files. Injected CPU-load/write failures produced neither a completed index nor failed-batch audit markers. Unknown source changes were rejected.
- 56 unit/regression checks PASS, including lookahead order, bounded scheduling and exception propagation. Explicit local DEBUG GPU checks are separate from full training. No remote process or remote checkout was modified and no long GNN/nnU-Net run was started.
- The modified cache also passed the actual CLI DEBUG training entry: all10 supplied records, 4 optimizer updates, refresh, validation, best selection, final memory and artifact export. This is a short integration smoke, not full training, exact-resume revalidation or a newly repeated segmentation/CP accuracy experiment.

Evidence: `validation/region_prepare_overlap_20260930`; detailed transient files are under `work/region_prepare_overlap_debug_20260929_r2` and `work/region_prepare_failure_debug_20260929`. `tools/profile_region_prepare_debug.py` is an explicit selected-record diagnostic without a cache index, training or ready marker.

## Applying the patch

Do not switch the checkout used by the live prepare/train process: its final source check would reject that mutation. The current process does not gain the patch automatically. A9–10% local improvement is insufficient reason by itself to discard substantial running work. On a later preparation invocation, fetch the pinned patch after the foreground pipeline has returned, use a fresh output directory and `--reuse-prepared` pointing to the most recent partial/completed cache. Omit `--partition-checkpoint`. Preserve server batch32/workers16/reg.02/view0/CUDA40/RSS192 and research-report. Completed batches are validated/reused; only missing batches are prepared. This is not an optimizer resume.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 서버32 유지, 비교는 명시적 DEBUG 양쪽4.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 로컬 GPU·CPU·RSS 및 서버 제공 snapshot 구분.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 작업은 준비 대기 비용 분석.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 스케줄러 단위검사는 synthetic 이벤트로 명시.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 모델을 변경하지 않았으며 GPU 회귀검사 수행.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
