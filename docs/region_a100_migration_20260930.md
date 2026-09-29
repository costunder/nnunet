# A100 MIG CUDA budget migration — 2026-09-30

The user's ece-agpu16 command still selected the A6000 UUID from ece-a6gpu8.
CUDA was unavailable before training began. The previously confirmed allocation
on ece-agpu16 is A100 MIG 1g.10gb, UUID MIG-774a3cc0-0169-5e18-b5d9-fe1b7a1d6ce7,
9.5 GiB visible. This historical assignment must still be visible at execution.

Implemented --resume-cuda-budget-change together with --resume-execution-upgrade.
Only CUDA allocator bytes may differ; RSS/resident limits, workers, candidates,
FP32, cache, model/config, L1/L2, loss, candidate128 and paste masks stay unchanged.
Saved model/Adam/RNG/support/plan/step/physical batch are loaded, not recalibrated.
Retained activation execution remains enabled. Use 8 GiB allocator budget for the
9.5 GiB slice, leaving 1.5 GiB outside that budget. This is a resource ceiling,
not a full-dataset memory admission result; OOM is reported without batch/model
reduction or fallback. No prepare, cache copy, or automatic long training.

57 regression tests passed. Actual CT DEBUG train8/val2 on RTX5070Ti resumed
step1 through step4 under changed 6->8 GiB budget, including refresh, validation,
best selection and final export. Model/Adam/RNG/state hashes match the prior
uninterrupted local result. The exact 87eafa6 runtime manifest passes explicit
40->8 GiB migration; other resource or research-contract changes are rejected.
Evidence: validation/region_cuda_migration_20260930/report.json.
No A100 execution, cross-device numerical parity or full-scale OOM guarantee
is claimed. This is GPU migration support, not a new speedup claim.

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
