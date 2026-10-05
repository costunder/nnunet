# v1.7 D preparation: separate a live reference endpoint from canonical cache identity

The server run at `08f68bf6c0eb160c454f85629452ef955d9c80da` stopped in `import_preparation()` before importing any completed graph or starting training. Its generic exception did not name the unequal manifest fields. The actual old/new server manifests have not been received locally; therefore the exact server mismatch is not established by this patch.

Code inspection established a concrete false rejection: `IDENTITY_KEYS` required the complete `native_experiment_binding` to remain equal. That admission includes checkpoint SHA/bytes, saved cursor, completed epoch counts and phase-dependent checks. These change during ordinary reference training. D canonical preparation uses no reference neural weights: its request binds the raw inventory, full donor/observation assignment, base configuration, 10mm scope and local geometry implementation.

## Change and bounds

`transition_reference_identity.py` permits only a validated checkpoint endpoint advancing within the same reference attempt. The checkpoint path, all four metadata file paths/bytes/SHA values, native experiment SHA, inventory, native physical batch, full40 recipe and every unknown field stay exact. Every recorded admission check must be strictly Boolean true. Only six known phase-dependent check names can change; missing/failed admission fields, backwards epoch/step/phase changes and inconsistent counters are rejected.

The preparation importer continues to require exact source snapshot, geometry request, original curriculum, D model, scope, hardware and resource settings. It records every unequal JSON leaf, its previous/current value and its acceptance decision in the owned new run's `reuse_identity_comparison_<SHA>.json`. A rejected condition names its fields and diagnostic file before any canonical-cache destination is created. A native checkpoint advance does not transfer weights, optimizer, support memory or progress into D.

The current execution closure adds only the reference identity helper to the previous recipient absence extension. The old published aa28082 176-file inventory and its checkout-byte digest remain pinned. Both new modules are checked against their actual file hashes. Existing completed graph/shared-source SHA, assignment, ordinal, genuine two-view measurement and real nonempty-context checks remain active.

On an ordinary rerun of the same new crossed experiment, the stored experiment identity remains frozen. A validated later native admission is appended separately to `reference_admissions.jsonl`; the crossed training checkpoint and its exact identity are preserved. Changing reference attempts or their immutable metadata is still rejected and reported explicitly.

## Verification and remaining scope

- 96 metadata/contract UNIT tests executed: 95 passed, one actual Windows symlink-creation test skipped because the OS denies that privilege. The mocked symlink guard test passed.
- New full-import regression tests exercise checkpoint progress and phase changes through request admission, the pinned source migration, payload inspection, hardlinks and idempotent reuse. They verify original files remain byte-identical and no neural checkpoint is imported. These use explicitly labelled UNIT storage tensors and receipt metadata, not fabricated CT or learned quality results.
- Tests reject immutable native execution metadata changes, hardware changes, helper-source tampering, invalid equal reference admissions and backwards progress. Complete diagnostic old/new values are checked.
- Python AST parsing and diff whitespace checks run on the changed implementation/test files. No model, loss or graph-building code changed in this patch.
- The prior actual CT/CUDA evidence under `validation/v17_recipient_context_20261005/` is preserved byte-for-byte. This patch performs no new CUDA forward, optimizer update, long training or quality evaluation. The previous actual CUDA smoke is not relabelled as a run of this patch.

The supplied server configuration remains D only, all14102 P+128U observations, 10mm, two genuine original graph views per observation, physical32, workers16, CUDA40GiB, RSS192GiB, resident128GiB and40epochs. The previous fastprep cache with the server-reported835 completed observations is read-only. The server must verify these835 actual receipts before continuing the remaining work. This has not been executed on the server by this local task.

If the actual server mismatch concerns an immutable field rather than reference progress, it remains an explicit failure. This patch does not silently migrate model/data/hardware controls or regenerate the preserved835 records to conceal that mismatch.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 physical32와 workers16을 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 제공된 A6000/96CPU/RAM 로그를 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 오류는 메타데이터 검사이며 OOM이 아니다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. UNIT fixture는 UNIT로 명시했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이전 실제 CUDA 증거를 보존하며 이번 변경은 신경망 경로를 수정하지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
