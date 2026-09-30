# v2.2 local CNN — named experiments and checkpoint continuation

The earlier timestamped `run` command only started a new experiment. It did not
implement repeatable checkpoint selection. `tools/run_local_cnn_experiment.py`
adds a stable `--experiment` root while delegating to the existing, unchanged
CNN preparation, training, exact-resume and final-artifact validation code.

## GPU number selection

Use `--gpu 6` with `CP_GPU=6` in the shell. Before importing PyTorch, the runner
queries `nvidia-smi -L` and resolves physical GPU 6 to its current GPU UUID or,
when that device has one MIG instance, its current MIG UUID. No second MIG-number
variable is required. For multiple instances, a unique current environment
allocation takes precedence. If it is absent, an explicit host/user/physical-GPU
allocation from `config/server_gpu_allocations.json` is checked against the live
parent GPU UUID and MIG UUID. This is stored allocation evidence, not automatic
scheduling or a generic first/free-MIG fallback. Ambiguous selection or a stale
record is rejected rather than choosing an arbitrary instance. The numeric
index refers to the current host/container's nvidia-smi inventory, so use the
server host when intending to select its physical device index.

The initial seven tests missed the known server's seven-instance configuration:
`--gpu 6` failed before opening the experiment on ece-agpu16. The user's earlier
inventory explicitly identifies the assigned MIG under GPU 6, and their CUDA
output confirms successful use. That existing allocation is now recorded for
that host and account only. It is never transferred to another host or account.
If MIG is recreated and its UUID changes, current explicit allocation can select
the new instance; otherwise the stale record requires updating from new allocation
evidence. A physical number alone cannot establish ownership of one of seven slices.

Twelve selection tests now cover seven-instance resolution, host/account isolation,
stale parent/MIG refusal and current-allocation priority. These are synthetic
selection checks, not remote A100 training. The earlier actual RTX5070Ti selection
check remains valid; no new remote A100/MIG execution is claimed.
All 24 selection and experiment/checkpoint orchestration tests passed together.
Resolution against the original user-provided seven-instance inventory also passed.
The runner's GPU argument does not change the checkpoint's training contract.

## Behavior

- First invocation prepares `inventory/index.json` and trains in `attempts/0001`.
- Repeating the same command selects that experiment's recorded attempt and
  `checkpoint_latest.pt`. A new attempt directory preserves prior checkpoints,
  logs and learning history. It never searches other experiments or global mtimes.
- `experiment.json` binds input inventory content, FOV, runtime source and all
  training/resource options. Different settings are rejected. GPU identity itself
  remains an execution environment choice; the saved resource contract is retained.
- Missing/corrupt saved checkpoints are not silently replaced with fresh training.
  A failure before any checkpoint requires calibration again and says so explicitly.
  If a resumed attempt fails before its first save, its explicitly recorded parent
  checkpoint remains the resume source, provided no newer save has been recorded.
- Completed runs validate or recover final export and do not restart optimization.
- An atomic directory lock prevents two managed runs writing the same experiment
  on shared storage. Normal return/errors release only the owning lock. A hard
  crash can leave a lock; it is not automatically deleted based on a PID from a
  different host. Report the owner and verify termination before manual recovery.
- The foreground child handles terminal Ctrl+C using existing save/pause behavior.
  The wrapper retains the lock until the child returns. No detached training is
  created. Before initial calibration completes, no optimizer checkpoint exists.

## Existing runs and code checkout

An exact existing `run` root (containing `inventory/` and `training/`) may be
specified as `--experiment`. Adoption requires a PAUSED or completed marker;
an unmanaged run that may still be active is refused. All existing checkpoint
integrity/source/settings checks still apply. No implicit architecture or runtime
migration is added. Merely specifying a new path will start a new experiment,
not discover an older timestamped run.

Keep code in a pinned checkout while any experiment uses it. Fetching and creating
a separate detached worktree does not switch the running experiment's files.
Use a stable absolute experiment root outside the code checkout, for example
`/home/aicompetition06/Medical/experiments/v22_local_cnn_m20_seed42`.
Use distinct experiment roots for 10/20/30mm or independent repeats. The input CT
and original inventory remain shared read-only. No CT caches or weights are copied
into Git or the new code worktree.

## Verification scope

Twelve synthetic orchestration tests check first run/resume, independent roots,
settings mismatch, duplicate locks, error cleanup, exact legacy adoption, missing
checkpoints, bound-parent retry, calibration retry, path confinement, and refusal
to adopt an unmanaged active run, and completed-run reuse. These are not model
or medical accuracy tests. Actual CT DEBUG train8/val2, physical2, four updates:
pause after update1 then fresh-process resume matched continuous execution exactly
for model, optimizer, state and RNG. Repeating the completed experiment validated
its final artifact without creating another attempt or starting training. Summary:
`validation/v22_native_local_cnn_20261001/experiment_resume_summary.json`.
No full server training was started by this implementation task.

## 작업 완료 체크리스트

- [x] 서버/로그인 세션을 종료하는 명령을 사용하지 않았다.
- [x] 이전 checkpoint·결과를 덮어쓰거나 삭제하지 않았다.
- [x] 모델 깊이·너비를 변경하지 않았다.
- [x] 데이터·후보·mask·FOV 규칙을 변경하지 않았다.
- [x] 숨겨진 subset/cap/fallback을 추가하지 않았다.
- [x] 기존 physical batch 및 병렬 loader 설정을 명시적으로 전달한다.
- [x] GPU/RAM 예산과 자원 설정을 실험 계약에 결속한다.
- [x] OOM을 모델 축소나 새 학습으로 숨기지 않는다.
- [x] DEBUG 실행은 별도 실험 root에 저장한다.
- [x] Synthetic 검사 자료를 실제 학습 결과로 표시하지 않는다.
- [x] 기존 loss·gradient·optimizer·checkpoint 경로를 그대로 호출한다.
- [x] 새 실행·재개할 checkpoint·출력 경로를 화면에 출력한다.
- [x] 전체 학습·전체 평가는 자동 실행하지 않는다.
