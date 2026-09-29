# Fixed-region preparation admission failure — 2026-09-29

The A6000 report passed CUDA/scatter/official merge dependency preflight, then stopped at first-scale partition admission. This was not a CUDA import failure or an out-of-memory exception. Its traceback does not contain the actual server pair counts or violated bounds. The displayed `train` command was not the running stage: the stack is `prepare_cache`, and preparation had not completed.

The delivery error was treating a DEBUG smoke as sufficient evidence for the strict server path. DEBUG preparation explicitly permits unvalidated profile violations; non-DEBUG preparation rejects them. Earlier local evidence already reported these violations. Repeating the strict preparation command does not resolve this known contract mismatch.

## Change made

Admission acceptance and model computation are unchanged. A strict rejection now retains every pair's record ID, N/E and violated limits; role/shell cluster counts, connectivity, size distributions, bbox/variance excess and merge termination remain in its complete audit. The CLI prints the violations and saves `admission_rejection.json` plus the same report in `failed.json`. When scale 1 rejects, scale 2 is explicitly `NOT_RUN`. Other pairs in that interrupted batch are not labelled admitted. Structural failures retain their original exception classification.

No research-training override, threshold change, reg/min_size adjustment, skip, fallback, training checkpoint or ready marker was added. Initial limits remain unvalidated; exceeding them is not evidence that the official merger is incorrectly implemented. Changing those bounds from blocking to diagnostic-only requires an explicit change to the earlier user contract. The proposed change is to retain and report all profile violations while continuing to enforce input/coordinate/role/coverage/finite/connectivity/edge integrity, original masks and explicit resource limits. This proposal is not implemented or approved by this patch.

## Verification

Eight focused tests passed, including four new report/gate checks and the existing four dependency/preflight checks (actual CUDA kernel included). A separate actual-CT DEBUG fixture of eight pairs was processed as one CUDA batch with **strict admission enabled**, using its existing optimized CNN snapshot. All eight rejected at scale 1, and scale 2 remained NOT_RUN. N ranged 691–3,856; E ranged 17,414–97,107. Other violations also occur below total N/E limits.

GPU: RTX 5070 Ti. Physical batch 8, workers 8, explicit allocation/RSS limits 6/12 GiB. Measured materialization-through-rejection interval 3.521 seconds; peak allocated CUDA 294,781,952 bytes and process RSS 1,898,426,368 bytes. This measures neither a training update nor the server batch of 32. Input checkpoint SHA256: `e70ef8a7e8b832259f5a70197129838d48804f526bf90cac41037b500b8cef96`. Full local report: `work/region_admission_report_actual_20260929.json`. Earlier sandbox attempts could not save the report/clean temporary files; the completed reruns used normal filesystem access. No user result was overwritten.

No full training or evaluation was performed. Changing preparation source files changes source fingerprints; old caches/checkpoints must not be relabelled. The failed server directory must be preserved. Do not present this reporting patch as a training unblock, and do not ask the user to repeat the same strict command as a cure.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 DEBUG 8pair를 한 GPU batch로 검사했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실패는 OOM이 아니다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사와 실제 CT 검사를 구분했다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번 작업은 차단 보고 검사이며 학습 연결 검사를 반복하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
