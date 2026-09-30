# v2.2 bug patch — bbox anchor and organ masking

The server stopped at calibration support 29/353, before optimization. Replaying
the same ordered records reproduces the cause: a selected donor component has a
bounding-box midpoint in background. `SourceCollection` defines that midpoint as
the placement anchor; it does not require the midpoint voxel to be tumor/liver.
The new local CNN crop loader incorrectly required all anchors to lie in liver.

The fix preserves the original donor anchor, recipient coordinates, native crop
extent, all observations, and full donor paste mask. It does not shift anchors,
skip records, resize inputs, or relax online full-mask placement eligibility.
Anchors must be finite integer coordinates inside the CT. The entire crop must
contain organ, external CT is masked before normalization, and every CNN layer
still masks non-organ features. Comparison observations still require label 1;
observed positives must match the original annotation component/anchor record.
Online candidate scoring retains the existing final full-mask filter.

## Verification

- 38 unit/regression tests passed, including concave background anchors, external
  NaN/gradient isolation, invalid observations, empty masks, out-of-CT anchors,
  existing ranking, and the online recommendation bridge.
- All 105 outer-train case annotation hashes and 14,102 recorded centers checked:
  13,440 comparison centers have label 1; among 662 observed anchors, 652 have
  label 2, seven label 1 and three label 0. A bbox anchor's label is not the label
  of the complete tumor component. No observations were dropped or relocated.
- The exact failing calibration batch, physical 32, passed real CT CUDA L0
  forward at margins 10/20/30. One actual pair with a background donor anchor
  passed backward and optimizer update at each margin, with zero external CT
  gradient. These are bounded DEBUG checks, not full training or MIG admission.
- Local detailed report: `work/local_cnn_anchor_fix_DEBUG_20261001/report.json`.
  Published summary excludes case identifiers, coordinates and raw CT.

## Execution and shared storage

The failed run never reached optimization, so there are no optimizer updates to
resume. Its old outputs stay intact. Start a separate output with the corrected
code; calibration is repeated. Other running experiments may use the shared
checkout, so fetch the fix and create a separate detached Git worktree instead
of switching the checkout they are using. Reuse the original read-only input
inventory and CT files. Each run keeps its own metadata and checkpoint directory.

Model size, physical batch, support/L1/L2, loss, 128 candidates, Basic CP and
original paste masks are unchanged. Long training was not started locally or
on the server. Earlier local CNN checkpoints have a different runtime identity;
this patch does not silently bypass their source checks.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험 명령을 사용하지 않았다.
- [x] 기존 파일·실험 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 축소하지 않았다.
- [x] 데이터·후보·mask·기준점을 축소하거나 이동하지 않았다.
- [x] 숨겨진 subset/cap/fallback을 추가하지 않았다.
- [x] 실제 실패 batch32를 GPU에서 처리하고 주석 검사는 4workers로 병렬화했다.
- [x] RTX5070Ti 및 GPU peak를 확인하고 명시적 CUDA/RAM 예산을 적용했다.
- [x] 실패 원인을 실제 데이터에서 재현했다. 모델 축소로 우회하지 않았다.
- [x] DEBUG 검사와 최종 학습 설정을 분리했다.
- [x] Synthetic 반례와 실제 CT 검사를 구분했다.
- [x] 실제 CT의 CNN loss·gradient·optimizer 연결을 검사했다.
- [x] 수정 범위와 shared checkout 실행 주의를 기록했다.
- [x] 전체 주석 검사·짧은 GPU 검사와 전체 학습/평가 미실행을 구분했다.
