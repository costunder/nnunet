# Region preparation: repeated packaging work removed

The server reported 21.37 seconds/batch at 14/353 batches. That measurement is full preparation, not an optimization epoch. This patch removes three identified repetitions without changing the partition, graph, model, profile, training objective or cohort.

## Changes

- Compute the fine batch's edge digest once after the independent 13-relation quotient comparison, rather than copying/hashing the same complete edge dictionary once per pair. Every pair retains the same digest.
- Compute preparation source provenance once per bounded batch. It is fresh on every call, never a global memoized digest. Each record/view binding is still independently constructed and checked against materialization. Final preparation source/input identity checks remain mandatory. Offline preparation and CP candidate preparation use this path.
- Keep each pair's complete role/shell group diagnostics in its own artifact. Save the complete batch audit once in the existing batch audit JSON. The per-pair audit includes its batch audit digest. This avoids repeatedly copying, sealing and serializing every other pair's diagnostics. No profile violation, group coverage check or graph tensor is removed.

The production dataset's repeated `record()` access already uses its verified shared RAM store; this patch does not claim a second full disk read was eliminated. Materialization content verification, frozen-CNN before/after checks, independent quotient verification and saved-file/receipt validation serve separate integrity boundaries and remain in place.

## Actual CT short measurement

RTX5070Ti, existing complete DEBUG train8 fixture, physical batch8 on both paths, workers8, CUDA budget6GiB/RSS12GiB. The benchmark loads the old packaging implementation from commit `6be85aa` without changing the checkout. Old/new packaging runs use the **same captured official GPU partition**, alternate order, synchronize CUDA and discard the first round. Three measured rounds follow. This isolates packaging, not the merger or full server pipeline.

| Measured work per 8-pair batch | Old | New |
| --- | ---: | ---: |
| Packaging including quotient check/extraction/sealing | 0.6195s | 0.2468s |
| Whole fine-edge dictionary hashes | 8 | 1 |
| Binding source verification, mean of 3 calls | 0.3089s | 0.0356s |
| Parallel verified save, single measurement | 0.8068s | 0.6723s |
| Embedded audit JSON bytes over all 8 pairs | 507,296 | 58,763 |
| Actual serialized pair-file bytes | 16,761,536 | 16,496,128 |

Packaging decreased about60.2%; this is **not** a60.2% reduction of full preparation. Audit bytes decreased88.4%; total pair-file size decreased only1.6%. The single new `prepare()` probe took2.285s, excluding load/bind/save. A6000 batch32 full-pipeline improvement has not been measured. Do not extrapolate this short batch8 result into a promised replacement for the server's approximately2-hour preparation estimate.

All CT tensors, fine coordinates, assignments, region bounds/mass, relation edges, record bindings, materialization evidence and profile flags were equal on the captured partition. Each retained pair's group diagnostics exactly matched its groups in the old full audit. The full audit digest was checked. Full report: `validation/region_preparation_dedup_20260929/packaging.json`.

## Verification

- 45 regression/unit checks PASS, including fresh per-batch fingerprint, unchanged binding results, invalid/duplicate record coverage rejection, own-group retention, out-of-range owner rejection and compact-audit mutation/missing-group rejection. Existing CUDA gradient/optimizer, policy, quotient and dependency checks remain covered.
- Actual DEBUG train8/val2 prepare, four optimizer updates, refresh, validation, best selection and fresh-process resume PASS. Model, optimizer, state and RNG hashes are exact between uninterrupted/resumed executions. All ten profile-violating records remain recorded. No profile was tuned to obtain PASS.
- Actual CT three-candidate CP smoke PASS: 2,368 voxels pasted exactly, complete original-mask checks retained, repeated event avoids repartition, resident and final-artifact mutation rejected. This is not a full128-candidate or native segmentation-training performance check.
- No full training, full evaluation or remote process was started. Production quality remains unverified. No cached data, CT or model weights are included in Git.

## Applying without disrupting the running server job

The ongoing server preparation remains on `6be85aa`. **Do not pull/switch source files in that active checkout.** Its final source-identity check would correctly reject a mixed-source preparation. Continue that run and its chained training with the same pinned code. These changes are for a separately prepared cache under the new revision. Existing region-cache and checkpoint fingerprints are not silently migrated or relabelled; the original paired cache remains reusable as input. A live upgrade or interruption was not performed.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 같은DEBUG batch8 비교, 병렬 저장 유지; 서버batch 설정 변경 없음.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. GPU모델/peak CUDA/RSS와workers 기록.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 변경은 반복해시·복사·저장 비용 제거.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
