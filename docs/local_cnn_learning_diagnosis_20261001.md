# v2.2 local CNN learning failure diagnosis

## Current evidence and remaining question

The user's server validation log contains epochs 0 through 17. Initial MRR is
0.117, epoch 1 is 0.157, epoch 16 is 0.099 and epoch 17 is 0.213. Pairwise loss
rounds to 0.6931 for most epochs. This is a warning, not proof that every score is
identical: different score distributions can have the same mean logistic loss.
The historical v1 top1=1.0 is 100% top1 on its evaluation, but its source-anchor
curriculum and score head differ from the current observed-location problem.

The primary cause in the **server-trained model has not yet been established**.
Only short DEBUG weights are available locally. Claiming pooling, prototypes or
the changed objective as the proven main cause would exceed the evidence.
The separate 30mm/MIG calibration OOM is also not fixed by this diagnostic.

## Read-only server probe

`tools/diagnose_local_cnn_learning.py` finds one experiment whose CSV matches all
18 user-provided epochs (five printed metrics per epoch). It searches experiments
and HierCP project work directories under the specified medical root. It does not
pick the globally newest checkpoint or silently choose between matching runs.
Within the matched experiment it reads the checkpoint bound to its attempt ledger.
Legacy local-CNN output roots use their own training/checkpoint_latest.pt.

The loaded snapshot's content hash, runtime source, inventory hash and model
contract are verified. It uses the current saved weights and epoch support, not
an imagined reconstruction of epoch 17. The report includes epoch/step/phase and
explicitly notes that saved memory may be older than current model weights.
No optimizer, training subprocess, pause signal, checkpoint export or production
ready marker is created. The output JSON is new and separate from training files.

Measurements use evenly spaced positive-containing cases in sorted case order,
with an explicit diagnostic case count and **every** candidate in those cases.
No candidate is removed to improve reported ranks. The saved physical batch is
used, including natural partial final batches.

- Candidate dispersion before and after each L1 layer, including normalized
  dispersion so feature amplitude changes do not masquerade as collapse.
- Positive-minus-unobserved score gap, variance, strict pair wins, exact ties and
  pairwise loss; MRR and observed ranks retain the production metric definition.
- Between-class live prototype cosine and recorded cluster geometry.
- For the same train queries, episodic support versus full eligible support.
  Training uses the saved patient-episode count; evaluation uses full support.
  Each branch excludes query identity on recipient and donor sides.
- Saved train L0 memory versus newly encoded current features.
- Weighted ranking, auxiliary CE and alignment gradients, separately, on CNN,
  readout/fusion, L0 output, L1 and L2; gradient cosines expose loss conflict.
  One actual positive/unobserved tile uses the original epoch normalization and
  all current query features. Eval-mode disables dropout for this diagnostic;
  it is not a replay of a stochastic training update.

Interpretation must combine measurements. Low L0 dispersion suggests an encoder
or readout issue; additional normalized dispersion loss in L1 localizes a later
effect. High prototype similarity plus weak score gradients suggests class
readout collapse. A deterioration on identical queries when switching support
suggests support-distribution sensitivity. None is an automatic causal verdict:
controlled confirmation on server weights is still required.

## Verification

Nine unit/operator tests include log rounding, exact run disambiguation, bound
checkpoint selection, ties/score separation and GPU parity of the traced head
against production predict_embeddings. The GPU test uses explicitly synthetic
operator inputs, not medical accuracy evidence.

The complete tool also ran on RTX5070Ti with the preserved actual-CT DEBUG
checkpoint: one train case and one validation case, both with all two DEBUG
candidates, physical batch 2. All gradient probes ran and the model state hash
was unchanged. This validates the diagnostic path only. It does **not** diagnose
the server's 17-epoch checkpoint or establish CP quality.
Detailed local result: work/local_cnn_learning_diagnosis_DEBUG_20261001/report.json.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] production 그래프·데이터 규모를 변경하지 않았다.
- [x] 숨겨진 subset/cap/fast mode를 추가하지 않았다. 진단 사례 수는 명시한다.
- [x] 저장된 physical batch를 유지하고 병렬 CT loader를 사용한다.
- [x] 로컬 GPU와 진단 peak VRAM을 확인했다. 서버 사용량은 서버 실행 때 기록한다.
- [x] 기존 OOM을 해결했다고 표시하지 않았다.
- [x] 진단·DEBUG 검사와 production 학습을 분리했다.
- [x] 합성 연산 검사를 실제 CT 결과로 표시하지 않았다.
- [x] 실제 CT CNN·L1·L2의 loss별 gradient 경로를 검사했다. optimizer는 실행하지 않는다.
- [x] 실제 실행 설정과 미검증 범위를 기록했다.
- [x] 로컬 smoke 통과와 서버 진단·전체 학습 완료를 구분했다.
