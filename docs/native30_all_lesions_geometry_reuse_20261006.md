# Native30 full P+128U evaluation: retain all lesions and reuse complete geometry

The user-reported server run at revision `8e3d6f2` loaded the original ROI30/context28 BEST epoch30 successfully. Its preparation completed all 21 validation cases and 2,823 observation records in 51:35. Scoring then stopped at `liver_4` because 34 recipient lesion components exceeded the saved `graph.max_lesions=12`.

Failed output, preserved without modification:

`/home/aicompetition06/Medical/experiments/v1_native30_full128_20261006_033101_3235230`

## Cause and explicit evaluation policy

The saved value 12 is a graph-construction admission guard, not a learned tensor dimension. Both the original hierarchy and the evaluation adapter fail on excess components; they do not silently retain only 12. The donor-other-lesion enumeration also used the same guard and could fail later.

The server launcher now explicitly chooses `--lesion-policy all_observed`. The recipient keeps every actual annotated component. The donor keeps every component except the selected source, exactly as in the original feature equation. There is no new cap, subset, lesion deletion, fake SourceTumor, or P/U-label change. The original weights, full model dimensions, saved config (including 12), L0 operators, prototype bank, ROI30/context28 and complete cohort remain unchanged. Each upper audit records the saved and effective guard, retained counts and zero dropped components.

This is an explicitly expanded **external-donor upper-input policy**, not an exact replay of historical single-patient topology. Training-time source-byte provenance is unknown. The existing annotation-aware and prototype/validation-overlap disclosures remain; this evaluation does not establish blind CP recommendation quality.

The 34 components are L1 context nodes. They are not 34 new observed-positive ranking targets. Eligible observed P and the unchanged 128 comparison locations U remain the signed inventory's task.

## Reuse and execution order

`--reuse-geometry OLD/geometry` opens the old complete cache read-only. It verifies the original config, six source-module hashes, signed inventory/cohort/donor assignment, transport helper/AST receipt, exact ordered row publication, all per-row receipts and every target/shared payload SHA. Missing or inconsistent cache data fail explicitly without repair, fallback, build or write into the old output.

The reused `prepare()` returns its existing measured node/edge summaries and reuse provenance without repeating graph construction, CT decoding or tensor deserialization for preparation. L0 reads the existing canonical tensors with mmap, runs the original two-view sampler and uses measured physical GPU batching. Upper inputs still require actual CT/annotation arrays; these are decoded and verified for that purpose. Read-only donor lookup does not rebuild its local canonical graph.

Every complete-case CPU upper graph is constructed and validated before L0 GPU encoding. The admitted graphs are cached and then used once per complete-case L1/L2 scoring, so this preflight does not duplicate the upper construction work. Raw-case decoding and original local sampling are parallel; the shared mutable provider's upper preparation is coordinated serially. The saved model and all candidate inputs are preserved.

New output files include `geometry_receipt.json`, `upper_admission.json` and a durable `case_scores/<case>.json` after each actual whole-case forward. A final `report.json` is written only after the entire cohort succeeds. The failed old run has no durable first-case scores, so it is not possible to resume that first neural result; the graph preparation is reused and whole-case inference is recomputed. No training, optimizer step, checkpoint save, CP or nnU-Net training is started.

## Verification

Focused regression results: 69 tests passed for checkpoint loading, upper adaptation, entry contracts, data transport and full-case evaluation; all 20 geometry tests passed separately. The geometry tests include complete 21-case storage coverage, missing/corrupt/reordered publication rejection, unchanged existing bytes/mtime, no read-only write/rebuild/decode and original mmap/two-view sampling.

The actual RTX 5070 Ti CUDA UNIT ran the original 10,050,543-parameter model with 131 query nodes, 34 recipient and 14 donor-other components. All 48 lesion nodes and all 8,908 bidirectional candidate/recipient-lesion edges were retained and 131 finite upper scores were returned. This is a deterministic synthetic UNIT with fresh untrained weights, not real-CT ranking quality evidence.

An actual-CT CUDA DEBUG execution using a previously completed ROI30/context28 cache is recorded separately under `validation/native30_all_lesions_reuse_DEBUG_20261006/`. It uses a fresh untrained original-size model and a train-only mechanical prototype bank; it is not the server BEST and not a full21 performance result. Production all21 completion remains pending the user's server run.

The server command uses physical GPU5 and a new immutable checkout/output. The active D experiment, old checkpoint, prototype, source and failed output are preserved. A complete, matching remote geometry cache is admitted at runtime; the remote bytes have not been independently inspected from this local session.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실패는 OOM이 아닌 병변 admission guard였다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 UNIT과 미학습 DEBUG는 명시적으로 구분했다.
- [x] 핵심 모듈이 실제 forward에 연결되어 있다. 이번 요청은 평가 전용이며 loss/backward/optimizer 학습은 실행하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
