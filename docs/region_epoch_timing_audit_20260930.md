# Region epoch timing audit — 2026-09-30

The user's running A6000 process 3692164 reports initial_memory 353 batches / 6:07 followed by epoch1 step2/405 at17.11s/it. At that initial rate, optimization alone is115.5 minutes. It is not a stable full-epoch measurement. The GPU snapshot shows100% busy,295W,2740MiB; neither GPU arithmetic efficiency nor peak memory can be inferred from those snapshots. This is still the old running process, not evidence that the support-only patch changed its optimization.

## Correction to the earlier SAGE speed claim

validation/l0_gat_sage_20260929/report.json measured8 real DEBUG observations, physical batch8, patient-excluded support from that same8-record fixture. Its explicit unmeasured list includes production physical32, full observations/full support, epoch/validation and production checkpoints. 2.283s to1.203s does not establish half-time production epochs. The later training path enables stable_spmm (segmented fixed-order reductions); the original GraphSAGE comparison used PyG's standard message_and_aggregate path. Thus both workload and execution kernels differ. No accuracy or whole-epoch speed guarantee follows.

The production path retains CNN and L0 activation checkpointing, L1/L2 checkpointing, FP32, and fixed64MiB segmented sparse workspace. These can reduce live memory while increasing/restricting computation; they are candidates for resource-aware measurement, not proven causes of the server17s. Physical batches32/48/64 are calibrated on cloned models, selecting measured throughput. Their actual reports must be read before asserting the chosen batch is appropriate. Do not blindly increase batch or remove deterministic behavior to fill48GB.

## Diagnostic implementation and actual local execution

`tools/profile_region_update_debug.py` reads a saved optimization checkpoint, verifies its content hash/cache/config, loads all saved support, and executes its actual next group and saved physical batch in disposable state. It does not rebuild regions/support, change model/loss/candidates/masks, write production checkpoints or resume production. It records cold load, transfer/validation, support/plan, forward, backward, gradient checks/clip, optimizer and checkpoint CPU-copy/hash/serialization. GPU profiler ranges identify CNN, sampling, pooling, SAGE and L1/L2, including backward recomputation. Disk-save/NFS completion is explicitly excluded. A cold profiled update has instrumentation/first-use overhead; do not extrapolate it to an epoch.

Local smoke used an existing actual CT DEBUG checkpoint (configured batch8, actual next group2, full supplied memory8, excluded support6) on RTX5070Ti. It completed forward/loss/backward/gradient check/optimizer, unchanged source checkpoint verified. Loss1.49382758; synchronized times forward0.535s, backward0.599s, gradient/clip0.192s, optimizer0.011s, copy/hash/serialize0.352s. These are diagnostics, not server throughput. Original evidence is retained in validation/region_update_profile_20260930. CUDA table inclusive ranges are nested and must not be added; profiler output on Windows has inconsistent aggregate percentages, so percentages are not used as performance evidence.

## Application boundary

First read the running run's existing execution_contract.json and batch_calibration.json. No GPU work, interruption, source switch or cache rebuild is needed for that read. Only profile a stable paused checkpoint with the GPU free of its original training; do not run concurrent timing, modify the running checkout, or treat this tool as exact runtime migration. Full A6000 support profiling remains unperformed here. No production computation was changed in this audit.

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
