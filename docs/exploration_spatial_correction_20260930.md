# Exploration L0: collapsed sampling correction — DEBUG candidate

## What failed

The original trace `exploration_L0_DEBUG_20260930_r1` placed all 48 nodes for
`liver_66:1` inside the original tumor annotation. Maximum radius was 4.743 mm,
and only 35 native voxels were occupied. Finite tensors and coordinate gradients
did not validate surrounding context exploration. The original artifacts remain.

## Actual change

`config/l0_exploration_spatial_debug.json` explicitly selects a new research
candidate. The original configuration and production defaults remain unchanged.

- The search head now receives query-relative position / 28 mm and incoming
  displacement / (28/3 mm), through a learned 6→384 projection. These are search
  state coordinates, not manually selected tissue descriptors or extra node data.
- Four offset biases start in tetrahedral directions. After tanh their initial
  direction components are ±1/sqrt(3). The prior step ceiling, 28/3 mm, remains.
  The existing random output weights remain trainable. No minimum radius,
  repulsion objective, GT exclusion, extra loss or post-forward node movement is
  added. Position-conditioned weights and offsets can change during learning.
- The idea of separated offset initialization is supported by the official
  [Deformable DETR implementation](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py).
  Its 2D radial initialization is NOT this 3D tetrahedral recurrent graph.
  Our geometry and CP application remain a project-specific research choice.
- Original CNN, 3 SAGE stages, 48 nodes, 47 parent-child edges, 128D output,
  L1/L2, same-donor loss, candidate128, Basic CP and paste checks remain.
- Total parameters: 1,674,210; L0: 841,824. Added parameters: 2,304. Common
  initial parameters except the intended offset bias are bitwise equal in the
  regression test; the additional layer preserves the shared initialization RNG.

## Actual CT evidence

GPU RTX5070Ti, 16 logical CPU, available RAM ~44.4 GiB. FP32, four reader/CPU
workers; GPU budget10 GiB and RSS24 GiB. Same named DEBUG 2 query /4 support
observations, three updates, native48³ crops. Not a full-cohort quality experiment.

| liver_66:1 after 3 updates | Previous | New candidate |
|---|---:|---:|
| Tumor annotation nodes |48|11|
| Nontumor liver nodes |0|37|
| Outside organ nodes |0|0|
| Maximum radius (mm) |4.743|27.771|
| Unique native voxels /48 |35|46|
| Median nearest neighbor distance (mm) |1.031|2.853|

The new candidate already had 41 nontumor-liver nodes BEFORE learning. After
three updates this became37. This is evidence of changed initial coverage, NOT
evidence that training discovered useful context, prevented future collapse, or
improved ranking. The second query occupies47/48 voxels after training. Native
voxel duplicates are reported; no node is dropped to make counts look better.

All 12 checked module groups, including CNN, search coordinates, offsets, SAGE,
L1/L2, receive nonzero gradients and optimizer changes. Outside-organ CT NaN
perturbation leaves output/coordinates unchanged; outside input gradient is zero.
Peak integrated smoke GPU allocation:251,397,632 bytes.

Separately, full native organ CTs (2 unique volumes, padded2×1×324×275×36) and
all128 real comparison centers ran at physical batches32/64/128. Warm L0-only
forward/backward/optimizer totals:0.3321/0.3392/0.3294 sec respectively. Batch128
peak:2,096,423,936 bytes (~1.95 GiB). This uses an output-square timing probe,
NOT the CP objective or a complete L1/L2 update. Only two passes per batch, not
an epoch estimate or rigorous throughput superiority claim.

## Verification and limitations

- 12 unit tests pass, including preserved legacy path, analytic interpolation
  gradient, PyG SAGE parity, boundary constraints, common initial weights, spatial
  search gradients, external CT invariance and collapsed-graph audit detection.
- Real CT 3-update integrated smoke passes. Full-native128 L0 cost probe passes.
- Actual trace visualization uses identical axes/scale for both versions and
  allows initialization versus later rounds; browser counts, controls and mobile
  width were checked.
- Fixed parent topology is unchanged. This does not learn branching/stopping.
- Smooth coordinate gradients are piecewise; boundary backtracking may change
  discretely. Exact physical spacing is preserved, including5 mm slice spacing.
- A wider initial graph does not ensure useful features. No new feature-diversity
  loss or GT-based "context" rule was silently introduced.
- CNN receptive fields and the root still read the center. Tumor-presence shortcut
  risk is unresolved; this is NOT a verified context-only recommendation model.
- No production integration, long training, server execution, checkpoint, git
  commit or push. No claim that project implementation is complete.

Raw evidence: `work/exploration_spatial_DEBUG_20260930_r1/report.json`,
`actual_graph.json`, `comparison_audit.json`, `comparison.png`;
`work/exploration_spatial_full_DEBUG_20260930/report.json`.
The earlier `work/exploration_spatial_DEBUG_20260930` run remains but is not the
paired comparison: it preceded initialization RNG preservation.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험 명령 없음.
- [x] 기존 파일/결과 파괴 없음; 기존 baseline 및 원시 결과 보존.
- [x] 모델 깊이/너비 편의 축소 없음.
- [x] 그래프/데이터 규모 편의 축소 없음; 기존48-node 연구 계약 유지.
- [x] 숨겨진 subset/cap/fast mode 없음; DEBUG 범위 명시.
- [x] physical32/64/128 및 공유 CNN batching 실제 확인.
- [x] GPU/CPU/RAM 확인 및 peak 측정.
- [x] OOM 없음; 모델 축소 없음.
- [x] DEBUG 설정과 production 분리.
- [x] 실제 CT 사용, dummy/random fallback 없음; 합성 unit fixture 별도.
- [x] 핵심 모듈 forward/loss/gradient/optimizer 연결 확인.
- [x] 변경값과 초기화 효과·미검증 범위 보고.
- [x] smoke와 전체 학습/평가 구분. 전체 학습/평가 미실행.
