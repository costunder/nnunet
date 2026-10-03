# Native v1 preparation ROI failure and explicit recovery

## What failed on the server

The pasted `ece-a6gpu4` run stopped in native CPU cache preparation. Training did
not start. This was `AdaptiveRoiBudgetError`, not a CUDA OOM. The preserved
baseline limits `graph.adaptive_roi_max_voxels` to 8,000,000. Failed ROI requests
in the log were:

| Case/sample | Requested native ROI | Voxels |
|---|---|---:|
| liver_116 / 0 | 219 × 213 × 223 | 10,402,281 |
| liver_116 / 1 | 219 × 213 × 223 | 10,402,281 |
| liver_129 / 1 | 189 × 219 × 203 | 8,402,373 |
| liver_84 / 0 | 213 × 225 × 237 | 11,358,225 |

The native ROI supports handcrafted geometry/context. The CNN input remains
48³. The 45.3777 mm margin in the last case follows the preserved liver-depth
rule; it was not a new manually enlarged context. Native ROI fields retain
roughly 70 bytes per voxel, but this is **not peak RAM**: intermediate arrays,
distance transforms and the original CT also occupy memory.

The final resource row (1.59 GB sampled RSS, one last task) does not measure the
failed requests, which were rejected before allocation. Host free RAM also does
not by itself prove a scheduler allocation. Twelve million voxels is an explicit
diagnostic candidate covering the four logged initial shapes, **not a validated
full-cache ceiling**. Target transforms, liver-anchor expansion and canonical
node/edge guards can still fail independently.

The error reports 188 eligible source attempts and 183 successful artifacts.
There are four unresolved ROI requests. The original completion accounting also
allows a proven `no_placement` attempt under `retain_original`; identifying that
remaining attempt requires the actual server manifest. It must not be presented
as a fifth ROI failure or as a newly permitted record skip.

## Delivered code and scope

1. `tools/probe_v1_roi_budget.py` reads every sample-level ROI failure in the
   native manifest. Each bounded child runs the original sample builder, including
   source selection, candidate search, curriculum transforms and earlier graph
   checks, until the exact first allocation failure recorded in the manifest.
   It verifies CT/label hashes, prepared-region metadata and the original
   prototype. It then executes the captured source or target ROI field call
   with an explicitly supplied candidate guard. It records shape, full mask
   voxel count, finite checks, elapsed time and process RSS. A failed child stays
   failed; other diagnostics continue. No cache, model or production config is
   changed. This is CPU geometry work, not a CPU substitute for neural tests.
2. Fresh-suite initialization optionally accepts a completed matching probe and
   one declared resource-only guard increase. The default remains 8,000,000.
   Model, GT, graph geometry rules, 416 profile, loss, two views, curriculum8,
   pool128, 84/21/26 split and40 epochs are unchanged. Both arms must have the
   same admission contract. Applying a larger guard requires explicit user
   approval; no production override has been applied in this local work.
3. `cache_budget_recovery.py` retains the current donor-aware eligibility
   contract and migrates verified successful artifacts into a disjoint fresh
   cache. Only the guard and its bound metadata may change. Tensor/geometry
   semantic hashes must agree before and after copying. Original results are
   never overwritten. The new certificate is checked around original cache
   preparation; the original pipeline alone may publish a complete cache.
4. Prepared regions and the prototype are copied as verified bytes rather than
   recomputed. Hash/copy I/O still costs time and disk space. No speedup for this
   recovery has been claimed without server measurements.

No long local or server training was started. The accepted r5 ZIP and pinned
202-file original archive were not rewritten. The original model/sampler files
remain exact in the new native snapshots; helper and resource contracts are
declared separately.

## Validation and next action

78 metadata/parser/recovery UNIT checks passed (50 existing/runtime integration,
14 admission, 9 cache migration, 5 ROI parser). Fixtures are explicitly synthetic
integrity inputs and never a clinical performance result. A sandbox-only Windows
temporary-directory problem was resolved by rerunning the same tests with
appropriate task-owned write access.

Actual failed-ROI cost measurement must run beside the server's existing
native region cache. No local copy of that failed server manifest/region cache
was available. A completed failed-ROI probe still does not establish canonical
graph or full-cache admission, ranking quality, full40 training, full21
evaluation, production CP128 or nnU-Net performance.

The immediate server operation is the read-only probe. After its real results,
choose and explicitly approve the resource ceiling before fresh recovery. Do not
edit the old frozen JSON or restart the unchanged8M preparation expecting a fix.

## Server probe received at 14:07:51

The user supplied the console output of
`/home/aicompetition06/Medical/experiments/v1_roi_probe_20261003_140751.json`
from revision `8b53e90`. This is console evidence; the JSON file itself has not
been copied to this workspace or independently hashed.

| Failed sample | Status | Observed RSS GiB | Wall seconds |
| --- | --- | ---: | ---: |
| liver_116[0] | PASS | 5.625 | 24.09 |
| liver_116[1] | PASS | 5.625 | 20.11 |
| liver_129[1] | FAILED | 2.220 | 6.29 |
| liver_84[0] | FAILED | 4.607 | 12.12 |

The report is incomplete (`completed=false`), preserves the originals, and
started no training. Neither failure was reported as `RSS_BUDGET` or
`TIME_BUDGET`. The subsequently supplied child stderr identifies incorrect
source-only assumptions in the diagnostic:

- `liver_129[1]`: the original source footprint is different from the failed
  footprint; the target transform must be replayed.
- `liver_84[0]`: the guard does not fail at the original source position; the
  original candidate location and its depth-dependent margin must be replayed.

These exceptions do not prove a model defect or insufficient RAM. The new
diagnostic captures the actual original ROI call without skipping earlier
candidate, canonical node/edge or geometry checks. The first original failure
must exactly match every parsed field in the old manifest before a larger
diagnostic guard is used. It retains the actual candidate center, rotation,
scale, corruption and target erasure, rather than measuring a guessed source
ROI.

The worker records `replay_seconds`, `transform_seconds` and `geometry_seconds`
separately. Whole-child wall time and RSS include the earlier original work;
they are not falsely reported as ROI-only cost. The initial allocation shape,
actual target-depth initial shape and final shape are recorded independently.
An original liver-surface expansion is allowed only within the explicitly
declared diagnostic ceiling, without shrinking geometry or changing the exact
centered full mask. A later real allocation failure remains a failure.

`--reuse-report` verifies the old report and matching request/answer sidecars,
frozen configuration, original source code, manifest, raw CT/label bytes and
resource limits. Only the two exact source PASS measurements are reused.
Their historical costs remain labeled `reused=true`; the two failed rows are
not adopted. Original report, sidecars, prototype and CT bytes are checked
again after the new diagnostics. Reuse saves the completed measurements without
relaxing a check or fabricating new timing.
Keep the original `v1_roi_probe_20261003_140751.json` as `--reuse-report` on
retries. The new combined report does not duplicate the old source sidecars;
it is not a replacement input for historical source-cost reuse.

The explicit limits remain 12,000,000 diagnostic voxels, 16 GiB RSS and 120
seconds per diagnostic child. The production default remains 8,000,000 voxels.
No automatic ceiling increase, new model configuration, cache publication or
training is introduced. Exact target replay may include earlier original graph
construction, so it may hit the declared timeout; that result must be reported,
not repaired by silently increasing the limit.

41 focused UNIT checks pass: 9 original-call replay, 8 bound historical reuse,
6 exact geometry/mask admission, 12 error display, 5 parser and 1 existing
admission integration regression. The fixtures are contract tests, not actual
server CT cost or GPU ranking evidence. The corrected server target measurements
remain pending. The accepted r5 ZIP, 202-file archive, four strict-nested sampler
modules, model, loss, candidate pool and curriculum are unchanged.
The existing admission/recovery suites also pass all 22 checks (one overlaps
the focused suite); 62 distinct UNIT checks pass in total. Eight modified/new
Python files parse, and the CLI exposes the reuse option. No new neural smoke,
full-cache preparation, training or accuracy evaluation was run for this patch.

## Snapshot inventory stop before the corrected probe

The next supplied server run at `1e5e615` stopped in `load_suite` with
`Unlisted or missing snapshot files: v1.0`, before starting an ROI worker.
That message alone does not enumerate the files, so the remote inventory has
not been independently inspected. A concrete defect was found locally: the
earlier probe imported the frozen snapshot without suppressing Python bytecode
writes, while `load_suite` counted every generated `.pyc` as unlisted source.

The correction distinguishes only an unlisted `.pyc` whose direct parent is
`__pycache__` and whose cache filename maps to an explicitly listed `.py`.
It still requires every listed source, exact source hashes, exact declared
overlays and configuration/manifest binding. Arbitrary extra files, caches
mapping to unlisted modules, `.pyc` outside `__pycache__`, and symlinks are
rejected. Real inventory errors now list the missing and unlisted paths.
The original snapshot, generated caches and old results are not deleted or
rewritten.

Suppressing bytecode writes alone does not prevent Python from reading an old
cache. Both ROI children and ordinary suite invocations therefore use a new,
non-existing `PYTHONPYCACHEPREFIX` plus `PYTHONDONTWRITEBYTECODE=1`. The ROI
worker enforces these settings again before frozen imports. Existing snapshot
bytecode cannot replace the verified source in these launched processes; no
replacement bytecode is written. A subprocess UNIT regression checks this with
timestamp-valid altered bytecode, while preserving its original bytes.

This is an execution/inventory correction. It does not change source manifests,
model mathematics, sampler rules, resources or the original 8M production guard.
The same original `140751` source PASS report remains the reuse input. Corrected
target ROI measurements and full preparation/training remain pending on the
server. If actual source files are missing or changed, the corrected checker
will still reject the run with their paths instead of treating them as caches.

Validation: all 60 existing runtime/ROI UNIT checks passed. The new bytecode
suite ran 20 checks: 17 passed and 3 real symlink-creation checks were skipped
because this Windows session lacks that privilege. The successful subprocess
check proves source execution despite a timestamp-valid altered cache; four
`load_suite` integration checks retain real inventory, source-hash and overlay
validation. No actual server inventory, target ROI cost, model forward or long
training was executed in this correction.

## Corrected server ROI probe completed at 15:36:22

The user supplied the complete console outcome of
`/home/aicompetition06/Medical/experiments/v1_roi_target_probe_20261003_153622.json`
at revision `dd601db`. This is received server console evidence; the JSON bytes
and its report SHA have not been independently obtained locally.

| Failed sample | Measurement provenance | Status | Peak RSS GiB | Whole-child wall seconds |
| --- | --- | --- | ---: | ---: |
| liver_116[0] | Verified historical source measurement | PASS | 5.625 | 24.09 |
| liver_116[1] | Verified historical source measurement | PASS | 5.625 | 20.11 |
| liver_129[1] | New exact original target replay | PASS | 6.405 | 54.77 |
| liver_84[0] | New exact original target replay | PASS | 13.249 | 74.19 |

The emitted report summary says `completed=true`, four failed-sample requests,
two reused and two new measurements, candidate guard 12,000,000, maximum RSS
13.248973846435547 GiB, and `originals_preserved=true`. It retains
`scope=debug_geometry_only`, `training_started=false`, `production_ready=false`.
This closes the four recorded first-failure ROI cost probes. It does not prove
complete canonical cache preparation or ranking quality.

The next reviewable server action uses this completed `153622` report as the
explicit 12M resource admission, the original failed native suite as recovery
source, and a fresh disjoint comparison root. The old `140751` report remains
historical evidence; it is not the completed admission input. Successful
artifacts are checked and copied with matching tensor/geometry semantics;
prototype and region files are copied as verified bytes. The expected 183
recovered artifacts must be confirmed by the actual server migration log.

The existing server shell performs native prepare, native 40 epochs, nested416
40 epochs and full21 paired evaluation in order. Native calibrates and freezes
the physical batch/worker lock; nested uses the same lock. The sole sampling
contrast remains native versus strict-nested 64/32/96/64/96/64. Model, target,
loss, two views, 84/21 split with outer26 excluded, seed42, curriculum8 and
candidate pool128 stay fixed.

The original runtime already accepts `HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES` as
a conservative per-case memory floor. Read its exact integer value from the
completed report, rather than reconstructing it from rounded console GiB.
This informs parallel resource admission; it is not a new whole-training RSS
limit, and the diagnostic 16GiB/120s constraints do not become training limits.
The explicit 8M-to-12M allocation-guard change is applied only when the user
executes the reviewed fresh-suite command. No production configuration or
server training has been modified or started by the local documentation work.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 측정 정책을 유지하며, 복구 복사는 원본 병렬 준비 runtime을 사용한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 제공된 CPU 자원 기록의 범위와 미측정 ROI peak를 구분했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 오류는 allocation 전 ROI guard였다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. UNIT fixture를 실제 데이터 결과로 보고하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 검증 경로를 변경하지 않았으며 이번 준비 패치에 새 신경망은 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 기본값 유지, 선택적 resource 변경, 서버 측정 대기를 기록했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
