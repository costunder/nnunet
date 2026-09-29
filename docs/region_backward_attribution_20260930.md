# Retained backward attribution — 2026-09-30

User-provided A6000 steps154–173: update9.356s, loader0s,
transfer/validation0.745s, forward1.936s, backward5.342s,
check/clip/optimizer0.037s, checkpoint1.113s. Backward is57.1%.
These measurements identify the phase, not the individual model component.

Existing profile_region_update_debug.py constructed the default checkpointed
model even for retained checkpoints. It now restores and reports the saved
activation policy. Existing historical reports are preserved.

New tools/profile_region_backward_debug.py restores the actual next batch,
physical batch, saved full support, model/Adam/RNG and original FP32 objective.
It checks content hashes, reviewed runtime, cache identity/configuration and
retained policy. Three disposable updates reset the same state: warmup,
baseline, and instrumented. No cache preparation, support refresh, production
checkpoint, long training or schema/model/dataset changes. Source checkpoint
and cache index hashes must be unchanged at completion. Use a paused source
and an otherwise idle assigned GPU. Training and diagnostics must not overlap.
The original source checkpoint must not be removed or modified.

Forward-created autograd nodes are attributed to CNN, fine feature sampling,
region pooling, SAGE blocks, L0 readout, support L1/L2, query L1/L2, loss and
parameter accumulation. Children retain ownership; shared nodes are counted
once. CUDA events surround actual backward nodes. Times include instrumented
launch/hook gaps, not pure kernel times. The report retains uninstrumented
baseline and instrumentation ratio. Do not apply diagnostic shares directly
to production seconds or claim epoch speedups.

Validation: 4 focused unit tests passed. Real CT DEBUG cache train8/val2,
actual query2, saved physical batch8, full DEBUG memory8/support6, RTX5070Ti.
Loss, all unclipped/clipped gradients, model and Adam match at rtol2e-5 /
atol2e-6; RNG is exact. Source checkpoint and cache index stayed unchanged.
Baseline backward~0.110s; attributed backward~0.235s (~2.147x overhead).
This validates attribution wiring/parity only; full server bottleneck remains
unmeasured. No production performance optimization is claimed in this patch.

An older local pre-commit fixture was rejected by the runtime verifier. The
verifier was not relaxed: a fresh current-source DEBUG step1 fixture was made
using the existing cache. Server87eafa6 source is already a reviewed release.
Evidence: validation/region_backward_attribution_20260930/report.json.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
