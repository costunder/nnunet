# Retain activations instead of recomputing — 2026-09-30

User explicitly requests spending available A6000 memory to reduce time. The previous checkpointed, fixed-workspace execution was not established as efficient on48GB. Batch capacity and activation storage are different choices.

Implemented tools/compare_region_memory_debug.py: clone the saved next update logically by resetting model, Adam and RNG for each arm; use its exact query group and all saved support. Compare original checkpointing/64MiB workspace, retained CNN/L0/L1/L2 activations with64MiB, and retained activations with explicit256MiB. The scope-local checkpoint execution override calls identical layer functions directly and restores controls even on error; it is isolated to this diagnostic process, not installed in production. No model/loss/precision/graph/batch/sampling change. Existing checkpoint/cache are read-only, no full support regeneration, no production checkpoint or ready marker. OOM/errors propagate with no smaller-model fallback.

Actual CT GPU DEBUG evidence: RTX5070Ti, configured batch8, actual next group2, saved memory8 and eligible support6. Explicit CUDA12GiB/RSS16GiB budget,8workers, one warmup and two measured trials per arm with alternating order. Baseline warm update0.449114s / peak266543104bytes; retained64MiB0.329416s /402708992bytes; retained256MiB0.342686s /402708992bytes. Retaining activations cut this fixture's update time26.65% with51.09% higher peak allocation.256MiB did not improve this small fixture; do not claim otherwise or promote it as a tuned default.

All loss terms, all trainable gradients after production clipping, model state and Adam tensors passed comparison rtol2e-5/atol2e-6; metadata and post-update RNG matched exactly. Five control/sparse CPU/CUDA tests passed. First diagnostic attempt failed only because torch.testing.assert_close cannot compare string metadata; comparator was fixed to compare metadata exactly and the entire experiment rerun. Failed attempt remains under work; evidence here is the completed r2.

Timing includes forward/backward/gradient validation+clip/Adam, excluding loading, support-plan preparation, parity copies and checkpoint save. This small fixture establishes execution equivalence and a local tradeoff, NOT full-support A6000 speed, worst-case memory admission, epoch reduction or exact old-runtime resume. Server execution has not changed. Run the same diagnostic only against a paused server checkpoint and otherwise free GPU; do not edit its live checkout or restart preparation. No automatic promotion to production.

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
