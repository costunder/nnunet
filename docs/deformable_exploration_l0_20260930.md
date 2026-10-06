# Single-phase Deformable Exploration L0 — implementation and bounded checks

2026-09-30. Research model `single_phase_deformable_exploration_l0_v1`.
This is an explicit user-authorized new L0, not an exact optimization/resume of
GAT, fine SAGE, EZ-SP or CNN-only L0. Multiphase and vessels are not included.
Production defaults, Basic CP, L1/L2 equations, same-donor ranking loss, the
128-candidate policy, observation set, and full paste-mask checks were not edited.
No long training, remote command, commit, push, or production checkpoint occurred.

## Implemented architecture

`l0_exploration/model.py`:

1. Unique CT volumes, boolean liver UNION tumor mask, native physical spacing,
   recipient/donor indices and centers enter `ExplorationBatch`. Recipient tumor
   class is not a feature or extra image channel. Labels enter the existing loss.
2. External CT is removed before clipping/scaling in the loader and again before
   the CNN. Eight 3D convolutions retain the previous 12/24/32 channel sequence
   (2/3/3 convolutions), with full/stride2/stride4 maps. Per-voxel channel LayerNorm
   replaces spatial GroupNorm: external padding and remote spatial statistics do
   not contribute to normalization. Features are masked after each convolution.
3. CNN maps are computed once per UNIQUE INPUT VOLUME per forward/update, not
   per candidate. Current trainable maps cannot be cached across optimizer steps.
   Native full-organ loading pads volumes but does not resize or subsample them.
4. Project sampled concatenated 12+24+32 features to 128D. Donor is encoded by the
   same CNN. Current parent state + donor state + mean explored state predict four
   continuous 3D offsets. All queries/branches within a round run in a batch.
5. Exact indexed trilinear interpolation retains coordinate gradients without
   repeating the full feature volume Q times. Stride centers use native indices
   0,s,2s, not endpoint-normalized resized-grid coordinates. Invalid corners are
   excluded and interpolation mass renormalized. A coarse scale with no valid
   organ support contributes zero; native scale retains valid support.
6. Node schedule is **1 → 5 → 21 → 48**, with **47 directed parent→child edges**.
   Round allocations are 4,16,27 before prediction; the last round distributes
   children over the 16 preceding nodes. No nodes/records are discarded by top-k
   or after creation. Four children for EVERY parent for three rounds would give
   85 nodes and contradict the selected 48-node budget.
7. Three independent 128D mean-SAGE blocks update the growing graph, one per
   round. Each node has one incoming parent, so batched parent gather is exactly
   neighbor mean; the root transform is applied once. A residual FFN follows.
   Tests compare the block to PyG SAGEConv with identical weights.
8. Root, all-node mean, donor, and absolute mean-minus-donor features feed the
   pair fuser → 128D → the unchanged PromptGraphModel L1/L2 and ranking loss.

**Learned:** continuous node coordinates conditioned on CT, donor and accumulated
context. **Fixed:** node count, parent allocation, rounds and stop budget. This
revision does not learn discrete branch creation/deletion or termination. It is
not an extracted vessel tree; edges are information/exploration dependencies and
need not follow an anatomical path. There is no radius/kNN reconnection.

## Physical extent and organ boundary

`config/l0_exploration_research.json` explicitly exposes all limits. Existing
`config/train.json:graph.context_outer_radius_mm=28` supplies the initial maximum
path length, split equally among three steps. Tanh offsets are vector-norm
bounded in **mm**, then converted by each recipient's spacing. This is a research
initialization, not a validated optimal radius. CNN receptive fields around the
sampled points extend further than the path length: 5/15/35 native voxels across
the three levels. Do not describe 28mm as the total visible CT radius.

For an out-of-organ endpoint, the algorithm tests factors 1,1/2,...,1/65536 in
parallel and chooses the largest valid step. If none is valid, it raises; no
record skip, replacement embedding, or smaller graph. Step scales are exported.
Boundary selection is piecewise/discrete: gradients flow through the selected
continuous position and scale, **not through the boundary membership decision**.
Input/feature masking is independent of this constraint, so outside image values
cannot leak through interpolation or convolution even near a boundary.

Full-volume resource cost still exists. 48 graph nodes do not mean 48 CT voxels
were encoded. Default CNN activation checkpointing is explicit. The old CNN
checkpoint is incompatible because normalization and L0 representation changed.

## Verification

### Unit tests

`tests/test_exploration_l0.py`: 8 passing checks, synthetic fixtures clearly
separated from actual CT:

- native coordinate interpolation and analytic coordinate derivative;
- exact 5/21/48 sizes, 47 edges, one CNN call, every active parameter gradient;
- external NaN invariance and zero external CT gradient;
- query batching equivalence and donor-dependent coordinate changes;
- malformed input, empty organ and incompatible model contracts rejected;
- mean-SAGE numerical equality with PyG;
- forced boundary backtracking, inside endpoints and physical path bound;
- training-mode checkpointed CNN and offset-head gradients.

### Actual CT integrated smoke

`work/exploration_L0_DEBUG_20260930_r1/report.json` and `actual_graph.json`.
RTX 5070 Ti, FP32, seed42; physical query batch 2, support batch 4, accumulation1.
Six real observations: query liver_66, other-patient support liver_71/liver_72;
their fixed training donors liver_1/liver_9/liver_18. Raw hashes/affines checked.
Explicit DEBUG **native 48³ crops** are used here; these are not a full-volume or
128-candidate quality experiment. The normal loader does not impose that crop.

Three AdamW updates through the EXISTING same_donor_live_v1 loss, L1 and L2 pass.
CNN, projection, offset head, all three SAGE blocks, fuser, L1, L2, L2 residuals,
label seeds all show nonzero gradients and optimizer changes. Actual coordinate
gradient absolute sum: 0.0029929583. External CT changed to NaN: embedding and
coordinate maximum differences **0**; external input gradient **0**. Peak CUDA
251,349,504 bytes. Loss decreased on the repeated smoke examples; it is **not
evidence of validation accuracy, recommendation utility or segmentation benefit**.

Initial attempt `work/exploration_L0_DEBUG_20260930/failed.json` is preserved: the
metric caller supplied tuple keys where the existing evaluator requires strings.
Fixed to use existing geometry-based `record_key`, then reran the complete smoke.

### Full native CT / 128 real candidates: separate L0 cost probe

`work/exploration_full_native_DEBUG_20260930_r1/report.json`.
Recipient liver_66 native organ bbox 277×275×36; donor liver_1 324×249×29.
Native spacing retained; padded input [2,1,324,275,36]. All 128 existing comparison
centers for liver_66 from preserved pair_assignment.json are retained. Donor is
fixed to liver_1 component1 per the same-donor configuration; no candidate is
dropped. Raw loading 2.36s; transfer 0.0053s in this run.

| Physical queries | CNN forward s | Exploration + input checks s | Backward s | Optimizer s | Peak GiB |
|---:|---:|---:|---:|---:|---:|
| 32 | .0565 | .0392 | .2733 | .0015 | 1.952 |
| 64 | .0569 | .0418 | .2746 | .0014 | 1.952 |
| 128 | .0570 | .0319 | .2666 | .0014 | 1.952 |

These are single post-warmup observations, each preceded by one warmup, with
synchronized timing; CNN subtime uses CUDA events. Remaining forward time also
includes validation/dispatch. All three batches encoded exactly two volumes once.
CPU threads/readers4; measured final RSS2.10GiB; CUDA budget12GiB, RSS budget24GiB.
Full model has 1,671,906 trainable parameters; L0 839,520; CNN119,508;
offset50,828; SAGE495,360. Existing L1 has402,562 and attention L2 has132,096
(L2 residuals and label seeds are included in the total).

The cost probe uses output-square loss to exercise L0 backward/optimizer. It is
explicitly **not the CP loss, not a complete L1/L2 update, not model training for
quality**, and writes no weight checkpoint. Its times exclude support refresh,
validation, production checkpoint copying/writing and full-cohort loading. First
timing run is preserved separately and varied from this run; no stable speedup
factor, A6000/MIG timing or epoch-time forecast is claimed. Physical batch128 is
measured here, not silently applied to the production training contract.

## Reproduce short checks

Use a new output directory each time. No server/shell termination commands.

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_exploration_l0 -v
.\.venv\Scripts\python.exe -B -u tools/run_exploration_smoke.py --debug --metadata work/same_donor_learning_DEBUG_20260930/cache/index.json --output work/exploration_smoke_NEW --workers 4 --cuda-gib 10 --rss-gib 24
.\.venv\Scripts\python.exe -B -u tools/run_exploration_smoke.py --debug --full-case-profile --metadata work/same_donor_learning_DEBUG_20260930/cache/index.json --output work/exploration_full_case_NEW --workers 4 --cuda-gib 12 --rss-gib 24
```

`make_network(meta,cfg)` injects the new L0 into the unchanged PromptGraphModel;
`RawStore.batch(rows)` supplies unique full-organ volumes and indexed queries.
The existing live same-donor loss accepts this model/batch, as exercised above.
No full-cohort training/online-CP runner is registered for this architecture yet;
existing trainer/cache schemas do not accept this as an exact resume. A complete
dataset scheduling/support-refresh/checkpoint/online bank integration and larger
native-volume admission remain before a server training command can be claimed
ready. Full-mask paste checks must be reused at that integration boundary.

## References and remaining scientific questions

- Deformable DETR (Zhu et al.), https://arxiv.org/abs/2010.04159:
  learned offsets and sparse feature sampling. Does not validate our 3D CP graph.
- Recurrent Models of Visual Attention (Mnih et al.), https://arxiv.org/abs/1406.6247:
  sequential learned looking. Its discrete policy is not copied as our training.
- GraphSAGE (Hamilton et al.), https://arxiv.org/abs/1706.02216:
  neighbor aggregation; our parent tree uses exact one-parent mean, no sampling.

Node collapse/redundancy, direction diversity, initialization, whether surrounding
context rather than center tumor appearance drives ranking, full 128-candidate
validation, annotation limits and downstream CP benefit remain unvalidated.
No new regularizer or surrogate tumor-position label was silently added.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 새 구조의 12/24/32·128D·3단계를 명시했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자가 승인한 48노드 연구 경로만 추가했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. DEBUG crop/cohort와 명시적 node budget을 구분했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 32/64/128 query를 실제 CT에서 측정했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM은 발생하지 않았다. native 해상도·CNN 깊이/너비를 줄이지 않고 명시적 checkpointing을 사용했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사와 실제 CT 검사를 분리했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
