# Native v1 preparation ROI failure and explicit recovery

## What failed on the server

The pasted `ece-a6gpu4` run stopped in native CPU cache preparation. Training did
not start. This was `AdaptiveRoiBudgetError`, not a CUDA OOM. The preserved
baseline limits `graph.adaptive_roi_max_voxels` to 8,000,000. Source ROI requests
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
   native manifest. Each bounded child replays the original source-component
   seed, verifies CT/label hashes and prepared-region metadata, reproduces the
   old guard failure, and executes the official source ROI field construction
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

Actual failed-source cost measurement must run beside the server's existing
native region cache. No local copy of that failed server manifest/region cache
was available. A completed source ROI probe still does not establish canonical
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

| Source request | Status | Observed RSS GiB | Wall seconds |
| --- | --- | ---: | ---: |
| liver_116[0] | PASS | 5.625 | 24.09 |
| liver_116[1] | PASS | 5.625 | 20.11 |
| liver_129[1] | FAILED | 2.220 | 6.29 |
| liver_84[0] | FAILED | 4.607 | 12.12 |

The report is incomplete (`completed=false`), preserves the originals, and
started no training. Neither failure was reported as `RSS_BUDGET` or
`TIME_BUDGET`. Their original child stderr is required to distinguish geometry
replay mismatch, original surface expansion, data validation, or another error;
the summary alone does not establish the cause. Do not increase the resource
ceiling, accept the two failures, or start cache recovery from this report.
The diagnostic display now exposes the actual child error while retaining its
complete stderr/stdout in the JSON; this changes reporting only.
Twelve focused failure-display UNIT checks pass, including multiline/chained
exceptions, stdout-only failures and unknown exit causes. The original worker
and ROI replay function are unchanged. This does not resolve the two server
failures whose stderr has not yet been supplied.

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
