## 2026-09-25 최신 — 서버 진행 화면 오류 수정

- 사용자 서버 출력: `work/v222_mig10gb_r6_scanfix`의 paired_cache가14,102/14,102에 도달한 뒤 viewer `Progress.event`에서 `KeyError: completed`. 이는 `stage_complete` envelope를 graph 진행 이벤트로 분류한 버그다. 로컬에서 실제 runner 이벤트 스트림으로 동일 오류 재현 후 lifecycle 분리를 수정했다. 관련11검사 통과. 테스트 범위는 UI/프로세스 제어이며 새 모델 학습 검증이 아니다.
- worker는 viewer와 별도 세션이므로 viewer traceback만으로 학습 종료를 단정하지 않는다. 현재 서버 worker 생존/후속 profile/GNN 시작 여부는 아직 출력으로 확인되지 않았다. 기존 run을 삭제하거나 새 학습을 시작하지 말고, Git fetch 후 수정된 독립 viewer만 `/tmp`에 추출하여 `--output work/v222_mig10gb_r6_scanfix`로 연결한다. 실행 중 checkout은 pull하지 않는다. 단계7개의 완료 비율로 표시되던 전체 ETA/rate도 제거했다. 모델·cache provenance·학습 설정 변경 없음.

## 2026-09-24 — 시각화한 v1 L0로 전체 GNN 학습 요청

- **최신 사용자 요구: tail 반복 대신 tqdm 화면.** `tools/watch_v222_server.py`를 독립 도구로 추가했다. Git fetch 후 `/tmp`로 이 파일만 추출하면 checkout 변경/학습 재시작 없이 기존 최신 run에 `--latest`로 연결한다. 새 `run_v222_server.py`는 직접 실행하며 자동으로 worker를 분리하고 상세 출력을 `console.log`에 저장, 전경은 tqdm 화면이다. 더 이상 새 실행 명령에 nohup/리디렉션/&를 붙이지 않는다. Ctrl+C는 viewer만 닫고 학습은 계속된다. 실제 완료 수 없는 단계는 경과 시간만 표시하며 전체 학습 완료는 최종 pipeline marker가 있어야 한다. 관련9검사 통과; 실제 별도 프로세스가 viewer 중단 뒤 완료되는 것 검증. 합성 프로토콜 테스트를 의료 모델 검증으로 보고하지 않는다. 모델 provenance 불변. 원격 MobaXterm에서 직접 화면 확인/서버 재시작은 수행하지 않았다.

- **최신 서버 정체 원인/수정:** 사용자 로그의 `preparation_benchmark_warmup workers128` 뒤 `workers1 tasks128 calibrationTrue`는 observations 준비의 중복 전체CT 벤치마크이며 학습이 아니다. `tools/v1_server.py`를 기존 v1 `run_case_jobs`로 연결해 각 CT를 한 번 처리하고 결과를 보존하게 수정했다. 개별 시작/완료·10초 heartbeat 추가. 실제 로컬131개/131attempts, 14,102관측·527donor, 모든 중심/정답/split/donor 배정 동일; 관련7검사 통과. 모델 provenance 불변. `validation/v222_r6/observation_single_scan.json` 참조. 현재 서버를 직접 중단하거나 재시작하지 않았다. 기존 run `work/v222_mig10gb_r6`의 observations 자식만 정확히 검증하는 stop helper를 Git fetch 후 임시파일로 받아 실행하고, 성공한 경우에만 pull 및 새 run `work/v222_mig10gb_r6_scanfix`를 시작한다. 현재 nnunet 환경과 할당 MIG UUID 유지. 다른 단계로 넘어갔으면 stop helper가 거절하며 중복 실행하지 않는다. Linux 신호 동작은 로컬 Windows에서 실검증하지 않았고 guard 단위검사만 완료했다. 별도 paired graph 준비 benchmark는 이번 변경 대상이 아니며 전체 pipeline 속도 개선을 주장하지 않는다.

- **사용자 서버 출력으로 확인한 최신 상태:** `ece-agpu16`, checkout `/home/aicompetition06/Medical/HierCP-v222-r6`, 원본 `Medical/Data/image`·`Data/labels` 존재, Python3.10.18의 `nnunet` 환경 활성화. 사용자 할당은 물리GPU6. 지정한 UUID `MIG-774a3cc0-0169-5e18-b5d9-fe1b7a1d6ce7`에서 PyTorch가 A100-SXM4-80GB MIG1g.10gb/9.5GiB/1device 인식한 출력을 받았다. 이를 다시 묻거나 GPU0/과거GPU5/A6000으로 취급하지 않는다. 긴 Python heredoc 대신 새 `tools/run_v222_server.py` 실행 명령을 제공한다. 현재 환경을 유지하고 원본 관측→실제graph DEBUG→전체paired cache→batch32/9GB DEBUG→auto-batch GNN40epoch를 순차 실행한다. 단계 실패 시 후속 실행 차단·기존 결과 보존. 실행기 제어 단위3검사 통과; 모델소스 hash 불변. 서버 본학습 시작 증거는 아직 받지 못했다.

- **MIG10GB 사전 메모리 smoke 실측:** RTX5070Ti에 PyTorch allocator9GB 상한을 강제하고 초과 할당 OOM을 확인했다. 실제 큰 그래프6배치씩과 저장된 실제 support1,216개, 전체5,550,806 parameter 사용. physical16 최대allocated3.695GB/reserved3.813GB, physical32 6.404GB/6.717GB로 forward/backward/모든 gradient/모듈 update/저장·재개 비트일치/eval 통과. physical64는9GB 상한에서 OOM. `tools/smoke_v1_memory_limit.py`, `validation/v222_r6/memory_limit_9GB_DEBUG.json`, `docs/v222_mig10gb_smoke_20260924.md`에 기록했다. 모델/그래프/production설정은 불변. 전체11,279 support/전체epoch/A100 MIG 속도는 검증하지 않았고 CUDA context 등 allocator 밖 메모리는 상한에 포함되지 않는다. profile 도구에 `--batch-size`, `--allocator-gb` 추가;10GB에서는 기존batch64 명령을 쓰지 않는다.

- **코드 전용 배포 검증 완료:** 새 Git checkout(autocrlf=false)에서 소스 해시/전체5,550,806 parameters 동일. 원본131개에서14,102관측·527donor를 재생성해 이전 중심/정답/donor 배정 전부 일치. 회귀20+재생성2검사 통과(0skip), 실제 원본→6개 full graph→L0/L1/L2 gradient/update 및 checkpoint 재개·epoch 평가 DEBUG 통과. 상세 `validation/v222_r6/code_only_release_checks.json`. 로컬 전체 그래프 재생성/40epoch/서버 학습/nnU-Net 평가를 완료했다고 주장하지 않는다. 캐시 전송 없이 서버에서 생성한다.

- **최신 배포 지시: 로컬 테스트 후 코드만 GitHub로 이동.** 캐시 전송 제안은 사용자가 거부했고 폐기했다. `SERVER_V222.md`와 `tools/v1_server.py observations`는 서버의 원본 CT에서 관측을 재생성한 뒤 `run_v222_v1_l0.py prepare`로 paired 그래프를 생성한다. `config/split_cp80_fold0.json`으로 현재 로컬 split을 고정하며 seed42/전체 데이터/모델을 유지한다. 로컬 캐시·가중치·DEBUG fixture는 전송하지 않는다. 새 raw 재생성 경로 검증 및 Git 업로드 결과는 후속 기록 확인. 서버 본학습 미실행.
- **사용자 후속 방향: 로컬 검증 / A6000 서버 본학습.** r6 로컬 전체 실행은 `STOP_AFTER_BATCH`로 정상 저장·중단했고 `paused.json` 확인. 현재 로컬 전체 학습이 계속 실행 중이라고 보고하지 않는다. optimizer는0회, support prefix는 최신 checkpoint_status 확인. 서버 학습은 아직 시작하지 않았다.
- **로컬 ETA를 명시:** 수정 후 실제516 query/421.04초=1.2255queries/s → train11,279개 optimizer약153분. production support 실측약4.04queries/s → epoch후 memory refresh약46.5분, validation inference최소약11.6분. 합계약211.5분(3시간32분), 현실적인 예측3.5~4시간/epoch. 최초support약47분 별도,40epoch약6~7일(GNN만, nnU-Net 제외). DEBUG support6개 benchmark와 partial production memory 외삽이므로 확정 wall time이 아니다.
- 저장된 MobaXterm endpoint는 `%APPDATA%/MobaXterm/MobaXterm.ini`에서 공개 접속 정보만 읽어 확인: `aicompetition06@147.46.121.38:22`, `.39:22`(07계정도 있으나 기존 프로젝트06만 시도). 암호/개인키를 읽거나 출력하지 않았다. read-only SSH 시도에서 .38은 OpenSSH에 검증된 host key가 없어 strict check 거부, .39는 connection reset. 보호를 끄거나 ~/.ssh·서버 설정을 변경하지 않았다. 서버GPU실측/학습 미실행. A6000 공식48GB는 확인했지만 실제 할당·속도는 미확인이다.
- **19:06:50 전체 r6 실행 확인:** `work/v222_v1_resumable_training_20260924_r6` 실제 worker PID32276(create_time1790244366.7457445), supervisor6176. 전체 설정 physical64/workers8/GNN40epochs, 모델5,550,806·전체14,102 graph 유지. 현재 initial_memory128/11,279 완료, checkpoint22,492,537bytes 실제 기록, 저장0.104초, stderr없음. optimizer step은 아직0이며 학습 재시작 전 support 계산 단계다. 정확한 최신 상태는 `.status.json`과 run의 `checkpoint_status.json` 확인. 원래49step은 복구되지 않으며 새 초기 가중치 실행이다.
- **r6 검증 완료, 전체 재시작 단계:** `v1_execution.py` 원자적 batch checkpoint/정확한 RNG·optimizer·L2 plan 재개 검증 통과. 실제 CT의 연속/재개 loss·전체 parameter·optimizer·support memory가 비트 동일. 실행기 전체 중단→재개→epoch refresh→validation→best checkpoint smoke도 통과(`work/v222_v1_lifecycle_smoke_20260924_r6/result.json`, 명시적 DEBUG). 같은10개 full-size query batch loss 전부 동일, 마지막64개·46,852,381edge batch277.81→48.00초, reserved21.39→12.77GB. 10batch 합계759.14→421.04초; 전체epoch 시간 확정 아님. unused CUDA cache를 batch 사이 반환하는 정책이 기본값. 실제 후속 실행은 `work/v222_v1_resumable_training_20260924_r6.status.json`으로 확인한다. 학습 시작은 first_optimizer_step 파일로 확인하며 support 계산과 구분한다. 실행 명세: `docs/v222_v1_execution_r6_20260924.md`.
- **실행 중단·수정 중:** 확인한 이 작업의 worker PID27840만 중단했다. 최종 기록은 epoch1/40, step49/9,360, 2,389/11,279 queries. 기존 실행은 epoch 끝에만 저장해서 이 49 step의 가중치는 디스크에 없다. 전체14,102 graph와 로그는 보존. `v1_execution.py`에 매 optimizer/support batch 원자적 checkpoint, RNG/optimizer/episode plan/coverage 복원, 새 디렉터리로 resume, 단계별 CUDA timing을 구현했다. `LocalBatch.to`의 tensor clone이 pinned memory를 잃던 문제는 PyG store shallow-copy로 수정. 아직 실제 재개·장시간 성능 검증을 진행 중이므로 아래 과거 '실행 중' 기록을 현재 상태로 읽지 않는다.
- **새 실측:** `work/v222_v1_profile_r6_default_20260924/steps.jsonl`은 실제 CT/full graph/physical64 DEBUG이며 support는 기존6개 실제 CT 검증 fixture이다. forward/backward가 주비용, H2D 약0.03~0.04초. CUDA reserved11.9→14.4→16.9GB와 함께 step9=186.4초 감속을 재현. 마지막 값만으로 전체 epoch ETA 확정 금지. 동일 batch 순서에서 사용하지 않는 allocator cache 반환 비교를 이어간다. 기본 profile 일부와 작은 CUDA 단위 테스트가 잠깐 겹쳤으므로 엄밀한 독점 A/B baseline으로 과장하지 않는다.
- **18:27 성능 문제 조사:** 사용자에게 1epoch 학습 약6시간을 보고한 뒤 이의를 받음. 실제 초기 step6(batch64,43,334,773edges)은 약50초, step44(batch64,43,439,758edges)는 약158.4초로 유사 규모에서3.17배 감속. 사전 측정1.387graphs/s 대비 누적0.524graphs/s. `performance_audit_1827.json`에 근거 보존. GPU process dedicated16.13GB/shared12.75GB가 관측됐으나 shared에는 pinned host가 포함될 수 있어 spill/누수로 단정 금지. 현재 runner에는 단계별 timing/allocator reserved 기록이 없고 calibration은 고정 GPU batch 반복이므로 지속적인 가변 batch 성능을 검증하지 못했다. 주병목은 아직 확정하지 못함. 실행은 유지했고 모델·그래프·batch를 변경하지 않았다. 최신 단계/step은 실제 로그를 확인한다.
- **17:19 실제 optimizer 학습 시작 확인:** `work/v222_v1_recovered2_training_20260924/training/first_optimizer_step.json` 생성. epoch1/40, step1/9,360, physical/effective batch64, loss1.8510187864, 첫 batch64개 관측·640,013nodes·43,369,297edges. `first_gradient_check.json` 모든 trainable parameter gradient 존재·유한. CPU loader8, inference batch64, 전체 train11,279/val2,823, 전체 graph14,102개 사용. `status.json` stage=optimization, optimizer_started=true. 전체40epoch 및 분할 평가 완료가 아니며 nnU-Net은 아직 시작하지 않았다. 아래 준비 상태는 과거 기록이다.
- **16:14 최신 상태:** `v222_v1_recovered2_training_20260924/cache/index.json`에 전체 14,102개 paired graph 준비 완료. 기존4,800개 재사용, 모든 관측 유지. `training/initialization.json` 생성 후 실제 GPU inference batch calibration 진행; optimizer는 이 시점 아직0회. 아래 준비 상태는 과거 기록이다. 실제 현재 상태는 `.status.json` 및 `training/first_optimizer_step.json`을 확인한다.
- **현재 복구 단계:** 두 번째 실행도 14:57에 빈 recipient context 입력 검사에서 실패했고 optimizer 0회였다. 기존 4,800 graph 보존. 실제 `liver_46:1/100`은 깊이>4mm context가 없지만 간 표면 CT node가 존재한다. recipient-only opt-in 빈 context와 기존 empty-shell token 경로를 연결했고 실제 두 위치의 AMP 전체 gradient/optimizer 및 CT 민감도 검증 통과(`work/v222_v1_empty_context_check_20260924/result.json`). 기존 nonempty graph/CT/epoch sampling 동일. 후속 실행 `work/v222_v1_recovered2_training_20260924/`, 같은 이름 `.status.json`에서 현재 상태 확인. 아래 상태들은 과거 스냅샷이다.
- **14:49 복구 실행:** `work/v222_v1_recovered_training_20260924/`. GNN pair 준비에서만 수신 격자에서 빈 mask가 되는 draw 3개를 seed 기반 유효 donor로 재배정. 관측 ID/중심/정답/총14,102개와 실제 사용527 donor 유지. 마스크 확대, 보간법 변경, native CP event 정책 변경 없음. 전체 해상도 조합 11,082개와 실제 실패 위치 3개 회귀 검증 통과(`work/v222_v1_recovery_check_20260924.json`). `v1_recovery.py`는 이전 3,196 graph와527 donor의 무결성을 검사해 새 경로에 hard link한다.
- 감독 프로세스 `tools/supervise_v1_training.py`가 `work/v222_v1_recovered_training_20260924.status.json`을 15초마다 기록한다. 로그는 `.supervised.stdout.log`/`.supervised.stderr.log`. 초기 supervisor PID16024, launcher9616; 실제 학습 자식 PID는 status에서 확인한다. `optimizer_started`와 `training_complete`를 확인해 실행/학습/완료를 구분한다. 14:49 최초 상태는 준비이며 학습 시작 전이다.
- **2026-09-24 14:37 확인 정정:** 실제 worker PID 15272는 종료되어 있다. 12:59:53에 paired cache 준비 중 `Real donor disappears at target spacing` 오류로 중단됐다. Donor 527개와 graph 파일 3,196개 생성, optimizer step 0. 아래 실행 중 스냅샷을 현재 상태로 읽지 않는다. `failure.txt`와 `.r2.stderr.log`에 오류가 있다.
- 전체 배정 14,102 관측/11,082 고유 해상도 조합을 읽기 전용으로 점검: `resampling_audit.json`. 실패는 `liver_28:17`, `liver_32:109`, `liver_45:81` 세 쌍이다. 공통 donor `liver_15` component 5는 5 voxels, 등가직경 1.59897mm, spacing 0.654296875/0.654296875/1mm이며 1/1/1mm 수신 해상도에서 nearest-neighbor mask가 모두 사라진다. 모델·표본·donor 배정·보간 규칙은 아직 변경하지 않았고 학습은 재시작하지 않았다.
- 실제 실행: `work/v222_v1_full_training_20260924/`. `.venv` launcher PID 14248, 실제 Python worker PID 15272. `work/v222_v1_full_training_20260924.r2.launch.json`, `.r2.stdout.log`, `.r2.stderr.log`를 확인한다. 마지막 확인에서는 donor 준비 단계: 3케이스 병렬, 공통 donor 파일 36개 생성, RSS 8.84 GB, process CPU 약 289%. 이 수치는 당시 스냅샷이며 현재 상태는 로그/프로세스로 재확인한다. Optimizer 단계는 아직 시작하지 않았다.
- 실제 최종 AMP smoke 통과: `work/v222_v1_training_smoke_20260924_r2/result.json`. 모든 parameter finite gradient, CNN/L0/L1/L2 optimizer update, 원본 대비 compact donor transport 및 압축 그래프 동일성 확인. 전체 train 11,279 / inner-val 2,823 관측. 첫 direct-conda launch는 `torch_geometric` 부재로 입력 준비 전에 종료됐고, 검증한 프로젝트 `.venv`로 다시 실행했다. 실패 로그는 보존했다.
- `run_v222_v1_l0.py fit --index <기존 관측 index> --output <새 run>`은 paired cache 전체 준비 뒤 40-epoch 학습을 실행한다. `hiercp_v222/v1_cache.py`, `v1_training.py`에 구현했다. 105 outer-train(84/21), 26 outer-val, 전체 14,102 관측과 inner-train donor 527개 조건 유지.
- 원본/기존 결과는 보존. 고정 seed로 관측별 donor를 정하며 정답을 donor 추출에 사용하지 않고 같은 recipient 그룹은 제외한다. Query가 양쪽 어느 역할로든 등장하는 support를 차단한다. 전체 donor–위치 조합 Cartesian product 실험은 아니다.
- 첫 AMP smoke에서 strict deterministic 3D grid sampler backward 오류 확인. 새 `deterministic_sampling.py`는 동일한 border/align_corners 보간을 batched gather로 계산한다. 결정론/seed 설정은 유지. CPU 수치 비교와 CUDA 반복 gradient 검사 통과. 실제 전체 모델 재검증 상태는 실행 결과를 확인한다.
- [학습 실행 명세와 상태 판단 파일](docs/v222_v1_training_20260924.md). `training/first_optimizer_step.json` 생성 전에는 학습 시작이라고 보고하지 않는다. 준비 과정과 전체 40 epoch 완료를 구분한다. Native online bank/nnU-Net은 아직 미연결이다.

## 2026-09-24 이전 — 사용자 지시로 v1 방식 L0 적용, 모듈 연결 검증

- 사용자 “L0는 V1처럼 적용해봐”에 따라 `hiercp_v222/v1_local.py`와 명시적 설정 `config/prompt_graph_v222_v1_l0.json`을 추가했다. 진입점 `run_v222_v1_l0.py`. v1의 물리 그래프/샘플링/source-target 문맥 비교 + 기존 CT-only 제외 조건을 적용한다. 원본 v1, 폐기 특징·내부 노드·CT 가림, 9천만 CNN을 그대로 복원하지 않는다.
- L0 4,718,420 / L0+L1+L2 5,550,806 parameters. 현재 모델 생성자의 L0 선택만 변경했고 L1/L2 메서드와 loss AST는 보존됐다. CP0.8/seed42/후보128 유지.
- 단위6개·구문·원본SHA 통과. 실제 CT3 recipients/6 graphs에서 전 parameter finite gradient 및 주요 L0/L1/L2 optimizer update, 실제 CP128 후보 점수·argmax 확인. 단 1-step DEBUG이고 학습된 추천 결과가 아니다. 초기 fixture 실패도 기록했다.
- [상세/범위](docs/v222_v1_l0_20260924.md), `work/v222_v1_l0_20260924/debug2/result.json`, `final_source_verification.json`. 전체 paired cache/40-epoch 학습/native bank·nnU-Net 전환/평가는 아직 미완료이며 production ready가 아니다. 기존 `run_v222.py`의 r4가 자동으로 이 새 입력을 처리한다고 설명하지 않는다.

## 2026-09-24 이전 정정 — L0 설계 이탈 확인, 당시 모델 수정 미완료

- `v22_cnn_l0_20260924`의 90,337,920-parameter ResEnc encoder와 전역 평균 readout은 assistant가 선택한 별도 미학습 진단이다. 승인된 L0 기준선, 학습된 CP 추천 모델, 완료된 수정으로 취급하지 않는다.
- 실제 v2.22 학습/점수 경로는 `hiercp_v222/training.py`와 `scoring.py`의 `PromptGraphModel`이다. 별도 CNN runner는 이 경로를 교체하지 않았고 L1/L2/CP를 실행하지 않았다. 직전 답변의 CNN+L1+L2 91,170,306개는 메타 장치에서 계산한 조합 크기이며 실제 학습된 모델이 아니다.
- 사용자는 뿌리 성장형 구현을 지시하지 않았다고 명시했다. 이를 미답 승인 질문이나 구현 지시로 해석하지 않는다. 직전 assistant의 DeepMedic 다중 스케일 변경 제안도 사용자가 원래 L0에서 벗어난다고 지적했으며 적용하지 않았다.
- 보존 v1 `hiercp/model.py`, `geometry.py`, `schema.py`, `loss.py`, `config/train.json`의 SHA가 `versions/v1/manifest.json`과 일치함을 확인했다. 그 코드의 L0는 CNN 120,804 + 이종 GAT 3층 4,038,912 + 변환/집계/관계 결합 1,441,024 = 5,600,740개다. 과거 서버 체크포인트를 직접 읽어 센 값은 아니다.
- v1의 source/target 문맥 및 관계 비교 경로가 목적에 더 부합한다는 판단은 과거 코드 전체의 복원 승인이 아니다. 이후 폐기된 수작업 특징, 종양 내부 노드, 마스킹 정책을 다시 넣지 않는다. 옛 그래프 자체의 타당성과 과거 학습 체크포인트의 정확한 revision은 별도 확인이 필요하다.
- 과거 Full GNN 40-epoch 학습 및 문맥 교란 결과는 `experiment_results/recovered_conversations_20260918/attachments/b611a9f81dd3cb24_붙여넣은 텍스트 (1).txt`에 존재한다. 현재 v2.22 성능 또는 L0 단독 기여로 표시하지 않는다.
- 완료한 작업은 코드/원본 SHA/파라미터/학습 기록 대조와 문서 정정이다. 모델 수정, 새 설계 확정, 전체 학습/평가는 완료하지 않았다.

## 2026-09-24 이전 정정 — CNN 특징맵 제출 반려, 요청한 그래프 미구현

- 사용자는 방금 CNN 특징맵 시각화를 명시적으로 반려하며 그래프를 요청했다고 정정했다. 아래 CNN-only 구현은 실제 실행된 별도 진단일 뿐, 사용자가 원한 L0 그래프의 구현/시각화 완료로 취급하지 않는다. CNN-only로 요구를 해석한 것은 assistant의 판단이었다.
- 해당 출력에는 노드/엣지/성장 경로가 없다. 이를 그림으로 꾸며 실제 그래프처럼 표시하거나 고정 격자 그래프를 다시 만드는 것으로 대응하지 않는다. 기존 코드/실행 자료는 증거로 보존한다.
- 당시 성장형 여부를 질문했으나 사용자는 성장형 구현을 지시한 적 없다고 답했다. 이 질문은 미답 상태가 아니다. 아래 v2.2 CNN-only 버전 기록을 사용자의 최종 그래프 설계 승인으로 읽지 않는다.

## 2026-09-24 이전 — CNN-only로 해석해 구현한 L0 한 케이스 진단

- 사용자는 3D CNN 기준선을 v2.2로 두고 L0만 한 케이스 생성하라고 요청했다. 구현 ID `v22_cnn_l0_20260924`, 새 모듈 `hiercp_v22_cnn/l0.py`, 설정 `config/v22_cnn_l0.json`. 과거 같은 숫자의 `hiercp_v22`/`run_v22.py`와 그래프 버전·결과는 보존한다. 새 실행기는 `tools/run_v22_cnn_l0_one_case.py`다.
- [정확한 구조·결과·한계](docs/v22_cnn_l0_20260924.md). 기존 nnU-Net ResEnc M plan의 6단계 encoder 전체(32/64/128/256/320/320, blocks1/3/4/6/6/6), 총90,337,920params. 기존 CT-only48³ 입력, 마지막 공간 평균→128D. full nnU-Net segmentation 재현이 아니며 그래프 후보 보간/GNN 없음.
- **미학습, optimizer0, L0-only DEBUG.** liver_108의 저장된187관찰 위치 전부 → [187,128]. 최종 성공 폴더 `work/v22_cnn_l0_one_case_20260924_release/`. 배치128, 전체추론0.9435초, peak allocated약7.15GiB. 모든 입력SHA/출력finite/배치독립/순서교환/입력민감도/전모듈자동미분 확인. 실제과제loss/optimizer/L1/L2/CP/전체학습/전체평가는 미실행.
- 큰 배치187 측정 지연으로 이번worker28220만 보고 후중단. `_final/`은 실패, 성공으로 읽지 않는다. 캐시해제로batch128 reserved14.86→7.78GB 감소. 실제 모델/데이터 축소 없음. 화면은 초기 CNN RMS이며 종양 확률/중요도가 아니다.

## 2026-09-24 이전 — L0 성장형 탐색과 대안 논의, 구현 변경 없음

- 사용자 요구: 가까운 문맥에서 시작해 CT 특징으로 다음 탐색 위치·방향·분기를 결정하며 노드/연결을 형성하는 뿌리 성장형 L0. 고정 격자의 점을 고르는 것으로 바꾸어 설명하지 않는다. 현재 고정 구 topology + CNN 보간 + GAT/SAG/U-Net 구현은 이 성장 과정의 구현이 아니다.
- 사용자는 이 아이디어의 평가와 다른 local graph 생성 아이디어를 요청했다. 성장형, 병렬 위치 학습(Deformable sampling 개념 확장), 학습 군집 집계(DiffPool), 특징 기반 동적 연결(DGCNN)을 비교했다. [REFERENCES](REFERENCES.md)의 2026-09-24 절에 R29–R31 및 기존 R16/R27 원문 근거·적용 차이를 기록했다.
- 이들은 설계 후보다. 새로운 모델/노드 수/탐색 예산/학습 계약을 확정하거나 코드를 변경하지 않았다. 기존 가중치·결과 보존. CNN-only로의 전체 교체도 승인·실행한 것으로 해석하지 않는다.

## 2026-09-23 이전 — 추가 무손실 압축 완료, 실제 점유 약 200 GiB

- [추가 압축 기록](docs/storage_compression_20260923.md). Basic CP의 현재 NPY 입력 1,076개를 NTFS 무손실 압축하고 각각 전체 SHA/dtype/shape/mmap 검증. 315개 manifest 보존. 해당 cache 디렉터리에만 새 파일 압축 상속 설정. reader/precision/model 변경과 파일 삭제 없음.
- 프로젝트 실제 파일 점유량은 첫 정리 전 563.181 → 첫 정리 후 452.014 → 압축 후 **200.121 GiB**. 이번 단계에서 파일럿 포함 약 251.893 GiB 추가 감소. 본 실행 감사: `work/storage_compression_20260923/{plan.json,files.jsonl,result.json,verification.json}`. 논리 크기와 하드링크 중복을 합산한 탐색기 표시를 실제 점유량으로 해석하지 않는다.
- Basic CP 105 case manifest, 대표 3 case production 로더/원본 SHA, 현재 L0 14,102 patch, frozen v1 검증 통과. 전체 학습/평가는 실행하지 않았고 압축 후 학습 처리량은 미측정. 기존 속도 결과를 압축 후 재측정치로 사용하지 않는다.

## 2026-09-23 이전 — 사용자 요청으로 불필요 저장 공간 정리 완료

- [정리 기록](docs/storage_cleanup_20260923.md). 실제 SHA 동일한1,713파일을 하드링크로 공유해101.735GiB 중복 저장 제거. 실패한 옛 context 준비와 shared-donor DEBUG bank의 NPY/NPZ11,870개(9.441GiB)만 삭제. JSON/CSV/로그/지표/가중치/원본은 보존했다. 폐기 경로에는 storage_cleanup 표식을 남겼다.
- `work/storage_cleanup_20260923/plan.json`, actions.jsonl, result.json, verification.json이 정확한 변경 근거다. 프로젝트 중복 제외 파일 크기563.181→452.014GiB. D 여유 공간이 실제119.423GB(111.222GiB) 늘었다. 경로마다 하드링크를 더하는 논리 합계는553.749GiB이므로 이를 실제 디스크 중복으로 오해하지 않는다.
- Basic CP105case manifest 및2,205배열 header/SHA 참조 검증, 대표3case 실제 production 로더/원본 SHA 검증 통과. 현재 L0의14,102patch 전부 보존. reader/model/precision/학습 계약 변경 없음. frozen v1 확인.
- 삭제된 `case_benchmark_run1/context/patches`와 `real_cp_DEBUG1/2/3/bank` payload를 완전한 현존 cache로 취급하지 않는다. 기존 검증 기록은 보존되며 해당 DEBUG 재실행에는 입력 재생성이 필요하다.

## 2026-09-23 이전 — 사용자 정정: U-Net형 다단계 축소·복원·skip 구현

- 사용자는 단일 late_sag를 요구한 것이 아니라 U-Net처럼 구현하라고 정정했다. 기존212ms 단일 풀링은 U-Net 결과가 아니다. `hiercp_v222/l0_graph_unet.py`에 encoder3단계, pool2회, bottleneck, unpool2회, skip concat2개, decoder2층을 구현하고 동일 L1/L2/loss에 연결했다. 비교 factory의 mode=graph_unet로 선택한다.
- [구조·근거·실측·한계](docs/l0_graph_unet_20260923.md). DEBUG 경로45,385→6,912→1,728→6,912→45,385; hidden128/heads4 보존, 총1,949,659params. SAG/GAT 기반 U-Net형 변형이며 원 Graph U-Nets의 GCN/gPool/A² 재현이라고 주장하지 않는다. coarse adjacency는 induced spatial이고 다중 성분이 있음을 기록했다.
- 별도 실행기 `tools/verify_l0_graph_unet_debug.py`, 설정 `config/l0_graph_unet_debug.json`, 실제 CT12patch 결과 `work/l0_graph_unet_debug_20260923/`. L0 전체 batch8 약449.44ms, peak2.61GiB. 실제 CT loss에서 두 skip/unpool 분기 각각 nonzero gradient, encoder/pools/decoder/skip merges/L1/L2 모두 optimizer update 확인.38테스트 통과.
- 전체 학습/평가와 production 승격은 미실행. 초기 가중치 속도이며 최종 pool budget은 미정. 45,385개 후보의 보간 중복 문제를 해결했다고 표현하지 않는다. 과거 code.txt/export는 이 최신 구현을 포함하지 않는다. 기존 결과·버전·r4 차단은 보존했다.

## 2026-09-23 이전 — 조기 선택 / 지연 학습 풀링 비교 구현·실제 CT 검증

- 사용자 지시에 따라 `hiercp_v222/l0_comparison.py`에 CNN-only/full_graph/early_ppr/early_sag/late_sag를 구현. 기존 L1/L2를 그대로 연결하며 GAT3층·hidden128·heads4 보존. late_sag는 전수 GAT1층 후 선택, early는 첫 GAT 이전 선택이다. 후보45,385개를 공통으로 평가하며 처음부터 작은 후보 격자를 만드는 구현이라고 주장하지 않는다.
- [구현·검증 기록](docs/l0_comparison_implementation_20260923.md). 실행기 `tools/benchmark_l0_comparison_debug.py`, 별도 DEBUG 설정 `config/l0_comparison_debug.json`. 유지 수1728/6912는 비교 후보이며 최종 설정이 아니다. production r4 gate는 유지한다. 전체 코호트 학습·nnU-Net 비교 실행기는 이번 범위에 포함되지 않았다.
- 실제 CT12patch/3case, physical batch2/4/8. 검증본 `work/l0_comparison_debug_20260923_verified/latency.csv` 및 result.json. K1728/batch8 L0 전체 latency: full467.97ms, early_ppr97.14ms, early_sag64.99ms, late_sag212.45ms. CNN/PPR/SAG/엣지/GAT/readout 포함, 전체 CP·nnU-Net latency는 아니다. SAG peak2.47GiB로 full2.07GiB보다 높다.
- 실제 loss에서 CNN/GAT/readout/L1/L2/scorer gradient와 optimizer update 확인. 34테스트 통과. 전체 학습·평가는 하지 않았다. 미학습 PPR 중심 편중과 SAG 다중 연결 성분도 NPZ/graph_audit에 기록했으며 정확도 성공으로 해석하지 않는다. 기존 code.txt와 이전 코드 snapshot은 이 구현까지 포함하는 최신 export가 아니다.

## 2026-09-23 이전 — L0 비교 실험과 노드 중복 점검

- 사용자는 격자형 그래프/PPR 표집/학습 graph pooling의 비교와 과도한 노드 수를 지적했다. [비교 설계](docs/l0_comparison_design_20260923.md)에 공통 입력 표현을 먼저 정한 후 CNN-only L0,전체 공간 그래프,PPR 표집,학습 풀링을 비교하는 안을 기록했다.
- 기존 후보45,385개는 CNN12³=1,728개 공간 특징의26.264배 보간 위치다. 새 영상 정보가 늘어난 것이 아니다. 다만1,728노드를 즉시 최종값으로 정하지 않고 소형 병변 표현과 실제 특징 해상도를 확인해야 한다. 모델·후보·기존 결과는 수정하지 않았다.
- C/D는 같은 선택 위치와 단계별 유지 노드 수로 먼저 통제 비교한다. PPR은 점수/분포이며 결정적 top-k인지 확률 추출인지 별도 명시해야 한다. 다단계 pooling의 시스템 성능 비교와 구분한다. 학습 설정 변경·비교 실행은 아직 없다.

## 2026-09-23 — 사용자 제안 Graph U-Net 계열 학습 풀링 원문 확인

- 사용자는 중요한 영상 특징을 학습해서 노드 표현을 줄이는 Graph U-Net 같은 pooling을 제안하고 실제 근거 확인을 요구했다. Graph U-Nets §3.1–3.4, SAGPool §3.1–3.2, DiffPool §3.2, ASAP §4.1–4.4 본문과 설치 PyG 2.6.1 pooling forward를 읽었다.
- [상세 기록](docs/graph_pooling_evidence_20260923.md), REFERENCES R25–R28. SAGPool을 첫 구현 비교 기준, ASAP를 국소 집계/연결 보존 비교 대안으로 선정한다. 숫자·pooling 비율·production 구현은 아직 확정하거나 변경하지 않았다.
- 앞선 고정 격자에 특징 kNN만 추가한 시각화나 수동 영역 병합 제안을 사용자 요구의 완성 해법으로 재사용하지 않는다. 현재 마지막 pool_gate는 global readout이며 중간 그래프를 줄이는 학습 pooling이 아니다.
- 새 pooling은 query CE의 gradient 경로에 연결해야 한다. 현재 detached support memory 때문에 alignment loss가 query CNN/pooling까지 직접 전달된다고 주장하지 않는다. ASAP의 학습 edge weight를 버리지 않도록 후속 message passing 통합도 필요하다. 논문 확인/코드 독해만 수행했으며 pooling 학습·성능 검증은 미실행, r4 학습 차단 유지.

## 2026-09-23 — 공간/특징 이웃 후보와 실제 CT 시각화 진단

- [방법 선정 기록](docs/v222_l0_method_selection_20260923.md): 공간 이웃과 동적 특징 이웃을 함께 쓰는 방법을 우선 후보로 선정했다. production 구현·최종 노드/표집 계약은 미확정이며 과거 철회 제안을 복구하지 않는다.
- 실제 CT liver_108 한 위치의 전체45,385후보에 초기 CNN(seed42) 특징 연결을 계산해 시각화했다. 진단용cosine8이웃이며 학습된 특징이나 최종 k가 아니다. 특징 연결 중6mm 초과는0.8221%; 대부분 공간 이웃과 겹쳤다. 이것을 비국소 문맥 학습 성공으로 보고하지 않는다.
- 근거 `work/v222_hybrid_visual_diagnostic_20260923/receipt.json`, `browser-check.json`. 화면2-hop은 보기 범위이며 production sampler가 아니다. 새 GNN/L1/L2 학습은 하지 않았고 r4 학습 차단을 유지한다. 기존 code export는 생성 시점 스냅샷이며 이후 진단 도구 변경을 포함하지 않는다.

## 2026-09-23 — r4 방사형 설계 반려·학습 차단

- 사용자가 실제 화면을 보고 “바깥쪽으로 쭉 뻗는 그래프”라고 지적했다. 확인 결과26개 고정 목적지의 경로/halo가 최종 노드의95.16~95.33%를 차지하고, 경로 길이는 직선거리의 평균1.0056배다. 계산이 맞는 것과 요구한 그래프 설계의 타당성을 또 혼동했다.
- `TRAINING_READY=False`와 명시적인 설계 실패 이유를 추가해 r4 `train` 및 production checkpoint 사용을 차단했다. 그래프 계산 코드·실제 DEBUG·실패 결과는 보존한다. r4를 완성 모델로 쓰거나 이전 모델로 자동 복구하지 않는다.
- 고정 목적지와 halo가 선택을 지배하는 결함은 미수정이다. 노드 선택이 연결 대상부터 결정하고 A*가 필요한 연결을 보완해야 하며, 정해 둔26방향을 새 숫자로 바꾸는 것을 해결책으로 간주하지 않는다. 다음 설계를 구현·검증했다고 주장하지 않는다.
- 아래60검사·GPU gradient·batch64 기록은 알고리즘/실행 검증 이력이며 사용자 요구 충족이나 연구 모델 완성 근거가 아니다. 실제 그래프 화면에도 반려 상태를 표시했다.
- 기존 `code.txt` 덮어쓰기는 자동 승인 심사에서 차단됐다. 따라서 해당 파일을 최신이라고 취급하지 않는다. 반려·학습 차단 상태의 현재 소스는 기존 파일을 대체하지 않는 `code_v222_r4_rejected_20260923.txt`로 별도 내보낸다.

## 2026-09-23 이전 — v2.22 r4 PPR/A* 실제 L0 구현·검증

- 사용자 요청한 PageRank 표집 + A* 외곽 연결을 `hiercp_v222/sampling.py`에 구현하고 기본 `PromptGraphModel.local` 경로에 연결했다. CNN 특징 비용 → 중심 개인화 PPR95% 질량 선택 →26방향 외곽 A* +1-hop 주변 → induced spatial graph → 기존3층GNN→L1→L2다. 상세 [r4 계약·근거·제한](docs/v222_ppr_astar_r4_20260923.md).
- 전체 후보45,385노드/2,383,482방향 엣지·CNN12/24/32·GNN128/3층/4heads·L1/L2 각2층·1,519,063params·CP80·seed42·40/250epoch 유지. 후보 전체를 GNN에 넣는 r3와 최종 sampled graph를 구분한다. 특징에 수작업 통계/좌표를 추가하지 않고 GT는 sampler 입력이 아니다.
- 실제 inner-train3case/6위치에서 최종8,934~8,969노드,345,574~347,382방향 엣지, 연결성·26목표 A* 비용의 독립 Dijkstra 일치·batch별 선택 동일성·CNN/L0/L1/L2 optimizer 갱신을 검증했다. 초기 CNN에서 노드 집합 Jaccard0.96~0.97이며 이를 강한 해부학적 적응이나 의료 성능으로 과장하지 않는다.
- 최종60회귀 통과. GPU checkpoint/autocast teacher cache 오류를 수정했고 보간8개 corner는 같은 계산을 streaming한다. A* 동적 nonzero GPU 동기화를 fixed-shape scatter로 바꿨다. 샘플/그래프별 순차 forward는 추가하지 않았다.
- 실제 batch2/4/8/16/32/64 측정 완료.64에서8.31graphs/s·6.32GB peak, 별도 재확인7.74graphs/s·6.32GB.128은 VRAM residency가 약15.9GB로 올라가며 장시간 지연되어 직접 생성한 정확한 PID30596/31388만 중단했다. 모델·그래프 축소 없이 보수적인 measured-memory envelope를 추가해 같은128 후보를 사전에 차단한다. 고정 batch cap은 아니다. 최종 프로파일:64개 L0 forward4.48s, 그중PPR0.374s/A*3.058s; backward3.645s.
- 전체14,102개 원본 CT 패치 SHA를 검증해 `context/index_ppr_astar_r4_final.json`으로 재연결한다. 데이터·분할·전처리 동일, 기존 graph/embedding/checkpoint는 재사용하지 않는다. r3 원본은 `versions/v2.22/before_ppr_astar_r4_20260923/` 보존.
- [실제 그래프 조작 화면](work/v222_ppr_astar_r4_20260923/actual-sampling.html), 검증 `work/v222_ppr_astar_r4_20260923/verification.json`, browser-check.json.6입력×4단계24보기와 실제 최종 노드/엣지 일치·회전·320px검사 통과. 전체 학습·전체 평가·최종 성능은 **미실행/미검증**. 현재 GPU DEBUG도 종료됐다.

## 2026-09-23 이전 — 그래프 설계 재감사, 학습 중단·복구 금지

- 사용자 지시: 이전 구조도 정확히 검토하며 복구하지 않는다. CNN cell + feature-kNN 제안은 철회한 제안이다. production 모델 변경이나 새 학습은 없다.
- 현재 v2.22 L0에는 그래프 노드/이웃 선택 sampler가 없다. 고정 격자 전체 → 반경 엣지 → CNN 특징 보간 → GAT다. `sample_features`는 특징 보간이며 graph sampling이 아니다.
- 이전 subset 선택 함수 존재를 요구한 샘플링 구조의 완성 근거로 보고한 설명을 정정한다. 기존 CT 가림, 표본 생성 비대칭, 미완성 L1, subset 선택 정책의 미검증을 확인했다. [감사와 근거](docs/v222_graph_design_audit_20260923.md).
- 보존본7파일 SHA 및 과거 실제 CT DEBUG의5개 주요 소스 SHA 검증. 현재 모델은 중단 실행 initialization과 동일, frozen v1 통과. 실제 CT 재실행이나 전체 모델 타당성 검증이 아니다.
- PID20828 학습 중단. 마지막 로그 epoch1 step411/30000, query6166/11279. 후속 조회에서 Python 프로세스 없음. 기존 캐시·로그 보존. 아래 실행 중 표기는 과거 이력이다.

## 2026-09-23 이전 — L0 구조 감사와 철회된 특징 기반 그래프 수정안

- 사용자 지적 후 CNN의12³=1728개 특징 위치를45385개 고정 격자 노드로 보간하고 있음을 확인했다. fixed topology에는 CT 인수가 없으며 그래프의 추가 효용은 입증되지 않았다. 이 점을 “제대로 된 구현”과 혼동하지 않는다.
- [철회된 수정 명세](docs/v222_l0_feature_graph_proposal_20260923.md): CNN 원래 map cell + layer별 특징 kNN 제안은 production에 적용하지 않았다. 사용자 지적 후 새 구조 제안 대신 이전 설계 감사로 전환했고 r3 학습도 중단했다.
- DGCNN/ViG 원문 확인, REFERENCES R16/R17 추가. 독립적 그래프 진단 함수4개 synthetic DEBUG 통과. 새 CT 학습이나 kNN 적용 완료를 의미하지 않는다.

## 2026-09-23 이전 — 실제 CT 시각화·VRAM admission 수정

- 후속 사용자 지적에 따라 점 표시 중심 화면 외에 **실제 전체 연결 그래프**를 추가했다. `tools/export_v222_actual_graph.py`가 production topology의45,385노드/2,383,482방향 엣지를 전수 복원 비교하고 `v222_actual_graph_fragment.html`이 전체 또는1/2/3-hop 연결을 표시한다. 범위 선택은 화면상 필터이며 모델 축소가 아니다. 현재 L0 연결은 CT별 동적 구조가 아닌 고정 격자·반경 그래프이고 CT마다 달라지는 것은 CNN 노드 특징임을 명시했다. 학습 코드는 변경하지 않았다.

- 실제 CT 단면·GT 별도 overlay·전체45,385개 노드 회전·선택 이웃 연결·L1/L2/CP 구조·실측 실행 상태를 조작형 화면으로 추가했다. 화면 시각을 명시하며 학습된 attention/군집/임상 성능을 만들어 표시하지 않는다. 생성 및 브라우저 검증 도구는 `tools/*v222*inspector*`, 실데이터 생성은 `tools/build_v222_visual_data.py`.
- VRAM 보정이 16GB GPU에서21.02/32.61GB까지 실행하던 결함을 수정했다. 다음 후보의 메모리를 실행 전에 검사하고 불합격에서 중단, probe 정리와 선택 batch 재검사를 수행한다. 이후 전체 support11,279개16.784graphs/s 완료(수정 전 약0.48).
- 첫 admission의 고정 support 메모리 과대 추정도 실측으로 발견했다. batch1/2 peak3.786/4.043GB인데 batch4를14.69GB로 예측했던 문제다. 두 실측 증가분과20% 증가분 여유로 수정했다. 새 VRAM6 + 기존 관련17 =23검사 통과. 기존89검사와 합쳐95종이지만 전체95 재실행은 아니다.
- 최종 실행 `work/v222_raw_ct_r3_training_20260923/gnn_vram_affine/`, wrapper30208. 정확한 실행 PID6256/29716은 확인 후 중단했고 이전 결과는 보존했다. CT14,102개 캐시는 그대로이며 새 `context/index_vram_affine.json`에 원본 SHA·runtime 변경·AST 동등성 근거를 남겼다. 모델·loss·전체 데이터·seed42·CP80·40epoch 유지. 이전 실행 step1은 완료 학습으로 취급하지 않는다.
- 현재 단계는 `work/v222_raw_ct_r3_20260923/gnn_vram_affine.stdout.log`, 시각화 snapshot은 같은 폴더 `visual_snapshot.json`으로 확인한다. 전체 GNN/segmentation/평가 미완료. [상세 수정·검증 기록](docs/v222_visual_runtime_verification_20260923.md). 아래 PID6256 기록은 이전 실행 이력이다.

## 2026-09-23 이전 실행 — v2.22 r3 실행 오류 재점검·수정

- 사용자 자원 저활용/전체 구현 점검 요청으로 추가 결함을 확인했다. 서로 다른 case의 처리량을 비교한 CPU 보정을 같은 작업 묶음 비교로 수정했다.4worker 실측0.348case/s로1worker0.193 대비 약1.8배이며 새 전체 작업에서 실제4코어 사용을 확인했다.
- GAT의 과소 타일을 실측에 맞게 수정했다. 노드·엣지·모델 축소 없이 실제 기본 경로 batch16에서3.128graphs/s·peak7.472GB, 이전 batch4 약0.885 대비 약3.5배다.32batch는14.788GB이며 더 느렸다. 최종 full-support batch는 별도 보정한다.
- Frozen scorer가 nnU-Net RNG/backend 상태를 바꾸던 문제를 scope/복원으로 수정했다. Seed42·비교표본128개 계약 검사도 보강했다. 전체89검사 통과(1건 git 접근 문제는 임시 저장소 범위 설정으로 재검사).
- 원본131case 감사 완료,105학습case 모두128개 비교표본 확보, outer-val26은 표집 제외. 학습 입력11,279개와 inner-val2,823개를 유지한다. 새 실행 `work/v222_raw_ct_r3_training_20260923/`, Python PID6256(wrapper23268). 전체 준비 후40epoch GNN 실행이며 optimizer 시작/완료는 로그로 구분한다.250epoch segmentation과 전체 평가는 아직 미실행이다.
- [결함·수정·실측·현재 실행 보고서](docs/v222_runtime_review_20260923.md). 아래44검사·모델파일 바이트 동일 기록은 최초 r3 적용 당시 이력이며 이번 실행 보완은 별도 기록이다.

## 2026-09-23 최신 — 승인된 v2.22 r3 원본 CT 구현·GPU 검증

- 사용자 전체 수정안 승인 후 `hiercp_v222_raw_ct_cluster_r3` 적용. CT 중심 가림 제거, 기존 외곽 전체 보존 + 내부 일반 공간 노드, 비교 표본의 종양 거리 제한 제거. CNN/L1/L2·loss·CP 실행 규모는 유지한다. 이전 코드/문서는 `versions/v2.22/before_raw_ct_r3_20260923/` 보존.
- 회귀44검사·정적18파일 통과. 실제 CT liver_1/5/108, 전체45,385노드/2,383,482엣지에서 모든 파라미터 finite gradient·CNN/L0/L1/L2 optimizer 갱신과 중심/주변 특징 전달 확인. Batch2/4 peak1.065/1.982GB, 0.825/0.885 graphs/s. DEBUG이며 full training 성능이 아니다.
- liver_108 비교 후보는0개에서3,402,279개로 정상화,128개 선택 완료. 전체131case 새 사전 검사는 진행 중이며 완료 receipt는 `work/v222_raw_ct_r3_20260923/preflight/summary.json`이다.
- Basic CP80 입력105case 준비 완료 상태는 그대로다. 새 전체 GNN40epoch/nnU-Net250epoch/전체 평가는 미실행. 관측 점수의 CP 배치 효용은 미검증이며 case 독립성 한계와 whole-method 비교 범위를 유지한다.
- [현재 파이프라인](docs/pipeline_v222.md), [r3 검증 기록](docs/v222_raw_ct_r3_verification_20260923.md). 아래의 승인 대기·고정 가림 기록은 과거 이력이다.

## 2026-09-23 최신 — 원본 CT 수정 명세 작성, Basic 준비105case 완료

- 사용자가 마스킹 필요성과 위치 선택 학습을 지적하고 제대로 수정하라고 요청했다. `tools/audit_v222_learning_contract.py`로 현재 학습/추론 경로와 inner-train527개 종양 크기를 감사했다. 결과 `work/v222_learning_contract_audit_20260923/audit.json`: extent 중앙값7.469852mm, 공통 가림27.290656mm, 종양 최대 extent보다 추가로 가리는 거리 중앙값19.820803mm. 관측 분류 점수를 CP 후보 순위로 쓰는 것의 유효성은 미검증이다.
- 원본 CT1채널 입력, 중심 제외 없는 일반 공간 그래프, 주변 종양 거리 제한 없는 주석상 간 비교 위치, CNN/L1/L2 규모·전체 데이터·CP80·seed42 유지안을 `docs/v222_raw_ct_proposal_20260923.md`에 작성했다. **학습 코드에 적용하지 않은 제안**이다. 읽기 전용 계산상 제안 graph45,385nodes/2,383,482edges. 가림 제거가 배치 목적 문제를 자동 해결한다고 주장하지 않는다.
- 이전 거리 제한만 변경하는 질문을 위 전체 수정안 질문으로 대체했다. 자동 승인 심사의 실험 계약 변경 차단 이후 정확한 수정안에 대한 답변 대기 상태다. 기존 거리/가림을 우회 변경하거나 새 GNN 학습을 실행하지 않았다.
- Basic CP80 입력 준비는105/105case 완료:30재사용+75새 준비. `work/cp80_comparison_20260922/resume_20260923_1/complete.json`; index SHA256 `8e2c77126195ba2014ff26ef41e1aafa04a5441db1f0c4adb337a5ba1a10ba6c`. **Basic/GNN optimizer 학습·전체 평가는 아직 미실행.** Native worker 보정과 실제 학습 실행이 남아 있다.
- 실제 liver_108 설명 자료 `work/v222_explanation_20260923/data.json`: 원본 GT label2와 모델 입력 가림을 구분한다. 원본 label1 voxel3,402,280개, 현재 거리 후보0개, 제안 후보3,402,279개(양성 anchor와 동일 좌표1개 제외). 설명용 계산이며 학습 정의 변경 아님.

## 2026-09-23 최신 — 병렬 처리·실행 전 검사 보완, 표본 정의 수정은 차단됨

- 사용자가 제대로 수정하고 병렬화·코드 검증을 수행하라고 요청했다. 주변 종양 거리 제한을 제거하는 변경 패치를 제출했지만 자동 승인 심사가 AGENTS.md의 실험 계약 변경 금지를 이유로 거부했다. 정확한 변경 내용을 async 질문으로 확인 중이다. 승인 없이 우회 적용하지 않았다. 현재 label/거리/가림/128개 규칙은 그대로다.
- 적용 완료: `hiercp_v222/parallel.py`에서 실제 전체 case 작업의 동시 실행 수·처리량·RAM을 측정하고 worker를 선택한다. 이후 작업은 완료될 때마다 보충한다. 실행 전 admission, 각 측정 wave, 최종 자원 보고서를 남긴다. CPU/RAM 범위를 넘는 명시적 worker 수는 오류로 처리한다.
- `hiercp_v222/data.py`는 모든 outer-train case의 비교 표본 검사를 패치 생성보다 먼저 수행한다. 기존 거리 규칙·난수·표본은 보존하며, 이후 raw hash 변경도 거부한다. liver_108 조건 자체가 해결된 것은 아니다. 새 GNN 준비/학습은 시작하지 않았다.
- DEBUG 병렬 검사5개, 사전 검사3개, 기존 누수/군집/case 분할24개 통과. 실제 CT3명/6그래프 CUDA DEBUG에서 전체 모델1,519,063params, 기존 DEBUG graph24,174nodes/1,229,900edges를 유지해 batch2/4 forward/backward와 모든 주요 모듈 optimizer 갱신 확인. Peak VRAM0.612/1.100GB,1.344/1.431graphs/s. Production graph는39,936nodes/2,040,356edges로 더 크며 이번 DEBUG 처리량을 production 성능으로 해석하지 않는다.
- 실행 근거 `work/v222_20260923_parallel_debug/result.json`; 수정 전 보존본 `versions/v2.22/before_center_fix_20260923/`. 자세한 현황과 미해결 사항은 `docs/v222_preflight_parallel_20260923.md`. Basic CP 준비는 별도로 계속 실행 중이다. frozen v1과 Basic source identity 유지 확인.

## 2026-09-23 13:26 KST 최신 — 사용자 요청으로 Basic CP80 준비 재개

- 사용자 “다시진행해”에 따라 중단했던 준비를 재개했다. 아래 야간 중단 기록은 이력이다. Wrapper PID19588, 실제 준비 PID16680. `work/cp80_comparison_20260922/resume_20260923_1.launch.json`과 같은 이름의 stdout/stderr 로그를 확인한다.
- 기존 완료30case는 manifest/배열/원본 CT·GT 해시를 검증한 뒤 재사용하고, 나머지75case는 기존 prepare 함수 본문 그대로 처리한다. 기존 부분 파일은 삭제하지 않으며 내용이 다른 파일은 덮어쓰지 않고 실패한다. 신규 helper `tools/resume_basic_cp80_preparation.py`, DEBUG 검사6개 통과. 원래 Basic source identity와 전처리 의존성 hash는 중단 전 receipt와 일치한다.
- CP0.8/seed42/105 outer-train case 및 전체 모델·학습 계약은 유지했다. 현재는 입력 준비이며 GNN/nnU-Net optimizer 학습·평가는 미시작이다. 준비 완료 후 native worker 처리량 보정과 학습 실행이 남아 있다.
- GNN의 liver_108 비교 중심 정의 변경은 미승인 상태다. 재개 요청을 별도 정의 변경 승인으로 해석하지 않았다. 기존 정의를 그대로 보존했고 GNN 준비·학습은 보류다. 상세 `docs/cp80_comparison_center_decision.md`.
- 재개 방법·검증·체크리스트: `docs/cp80_resume_20260923.md`. 이 항목은 착수 시점 기록이며 실시간 완료 수와 실패 여부는 attempt receipt와 로그로 확인한다.

## 2026-09-22 22:40 KST 최신 — 사용자 요청으로 작업 중단

- 사용자가 밤이므로 중단하라고 요청했다. 이번 Basic CP80 준비의 실제 PID13468을 명령/부모 PID22256 확인 후 종료했고 wrapper도 자동 종료됐다. 다른 프로세스·셸·원격 세션을 종료하지 않았다.
-105case 중30case 준비 완료 로그를 확인했다. 기존 결과·부분 입력·로그는 그대로 보존했다. 마지막 처리 중인 case의 부분 파일이 있을 수 있어 재개 전에 검증해야 한다. `work/cp80_comparison_20260922/stopped_20260922T134000Z.json`에 기록했다.
- GNN/native 모델 학습은 아직 시작하지 않았다. **사용자의 재개 지시 전까지 추가 준비·학습을 실행하지 않는다. 자동 재시작도 없다.** 비교 중심 정의 질문은 미답 상태로 남겨두며 중단 요청을 그 변경 승인으로 해석하지 않는다. 아래 실행 중 기록은 과거 상태다.

## 2026-09-22 최신 — 사용자 승인 CP80 비교 적용

- 사용자가 Basic와 v2.2 모두80%로 비교하라고 지시했다. 현재 활성 v2.22 r2에 적용했다. `config/comparison_cp80.json`, v2.22 config/bank/native 확률0.8, 새 `nnUNetTrainer_250epochs_BasicCP80Online`을 연결했다. 기존 OriginalBasicCPOnline 확률1.0과 literal reference는 보존했다. Basic CLI 기본은80%, 원본은 `--cp-probability 1.0` 명시.
- 두 loader의 gate를 공통 seed42/epoch/worker/batch별5draw 일정에 연결했다. 최초 draw<0.8이면 같은 샘플에서 양쪽 CP를 시도한다. Basic 위치/source 선택은 기존 별도 RNG로 유지한다. Donor/크기/위치/crop 차이는 남아 전체 방법 비교다. 실패 시 원본 유지, 재추첨 없음.
-66검사 통과(신규CP80 6 + Basic8 + RNG9 + 모델/누수43), 정적/frozen 확인. 문서 `docs/cp80_comparison_20260922.md`; snapshot `versions/v2.22/before_cp80_comparison_20260922/`.
- Basic 전체105case static 입력 준비를 `work/cp80_comparison_20260922/basic_inputs/`에 시작했다. Wrapper PID22256, 명령/로그는 `basic_prepare.launch.json` 및 `.stdout.log`/`.stderr.log`. GPU 학습은 아직 미실행이며 native worker 보정이 필요하다.
- **GNN은 여전히 liver_108의 비교 중심 규칙으로 보류 중이다.** 정답0을 '주변27.94mm 전체 무종양' 대신 '중심이 종양 마스크 밖 간 조직'으로 정의하되 모든case/128중심/가림반경을 유지하는 구체적 제안을 사용자에게 질문했다. 답변 전에는 적용하지 않는다. `docs/cp80_comparison_center_decision.md`.80% 요청을 이 별도 정의 변경까지 승인한 것으로 해석하지 않았다.

## 2026-09-22 최신 상태 정정 — 준비 실패, CP 확률 문헌 확인

- `case_benchmark_run1`은 context 준비 단계에서 실패했다. `liver_108`에 고정 blind radius 기반 종양 거리 규칙을 만족하는 비교 중심이0개라 필요한128개를 만들지 못했다. `work/v222_train_20260922/case_benchmark_run1/failed.json` 및 `.stderr.log`에 기록됐다. 실제 GNN optimizer/nnU-Net 학습은 시작하지 않았다. 해당 case를 버리거나 graph/patch/blind 규칙을 임의 축소하지 않았다. 아래 착수 기록을 현재 실행 중으로 읽으면 안 된다.
- CP 확률 질문으로 TumorCP 원문§3.2를 확인했다. CP 자체는0.8, CP 내부 개별 변환은0.5, intra/inter 선택도 별도0.5/0.5다. `REFERENCES.md` R13/R14에 기록했다. 본 프로젝트 Basic CP의 매 방문 시도와 논문의 설정은 다르며, v2.22의0.5가 표준/최적값이라는 근거는 없다. 문헌 확인만으로 설정을 변경하지 않았다.

## 2026-09-22 최신 실행 — 공개 case 기준 v2.22 전체 GNN40epoch 착수

- 사용자 “이렇게 학습하게 해”에 따라 `work/v222_train_20260922/case_benchmark_run1/`에 전체 context 준비→GNN40epoch 실행을 시작했다. Wrapper PID452, 상세 명령과 로그는 상위 폴더의 `case_benchmark_run1.launch.json` 및 `.stdout.log`/`.stderr.log`. 이 기록 시점은 준비 중이다. 실제 optimizer 시작은 `stage=optimization`, 완료는 `gnn/training_complete.json`으로 확인한다. 지금 학습 완료나 nnU-Net 시작으로 말하면 안 된다.
- CP0.5는 기존 프로젝트 설정이며 최적값의 실험 근거는 없다. 현재 승인 조건으로 유지했다. 전체 데이터 절반 사용이 아니라 이후 native 학습 샘플별 CP 시도 확률이다. 원본 Basic CP는 매 방문 시도한다.
- `hiercp_public_case_benchmark_v1`을 추가해 환자 독립성 미확인/주석 완전성 미확인을 명시했다. `patient_group=case:<ID>`, `annotation_complete=null`, `patient_independence_verified=false`. 기존 verified-patient 검사나 전체 CT 중복 검사, split 격리는 유지한다. 이전의 identity gate 미해결 기록은 이번 명시적 benchmark 계약으로 대체한다.
- 최초 전체 memory 생성 전에 inference batch/worker를 실측한다. 최신 memory를 epoch 경계에서 재사용해 중복 계산만 제거했다. Full graph39,936nodes/2,040,356edges, train11,279/val2,823예상, 모델1,519,063params,40epoch는 유지한다. Native250epoch/ResEncM/patch128³/batch2/seed42 계획은 별도이며 GNN 완료 후 native worker 보정이 필요하다.
- Case provenance4개 포함 회귀43개 및 정적 검사 통과. 보존본 `versions/v2.22/before_case_benchmark_training_20260922/`. 실행 source는 수정하지 않고 기록만 갱신한다. 상세 `docs/v222_case_training_run_20260922.md`.

## 2026-09-22 최신 — Basic CP 비교 난수 계약 수정

- Original Basic CP는 기본 nnU-Net의 seeds=None/비결정적 worker, v2.22는 seed42였다. 이를 공통 `comparison_randomness.py`로 수정했다. 모델 초기화42, ordered workers, epoch/worker/batch별 CT/crop/oversampling/standard augmentation RNG를 공유하고 CP RNG를 분리했다. 두 CLI는 seed 기본42와 필수 worker 수를 기록한다. Split 자체는 변경하지 않았다.
- DEBUG9 + Basic CP8 + v2.2~v2.22 회귀39 통과. 실제 전체 ResEncM 초기 state_dict 일치, 2-worker 순서, epoch 재개 검증. Native blur FFT/일반 연산 선택에 따른 반올림 오차는 보존하며 GPU bitwise 재현성을 주장하지 않는다.
- 현재 Basic CP와 v2.22는 donor/종양 크기/CP 확률/위치/crop 정책이 다르다. **전체 파이프라인 비교이며 GNN 위치 선택만의 ablation이 아니다.** 원본 Basic CP 정책을 임의로 바꾸거나 새 대조군을 실행하지 않았다. 상세: `docs/comparison_seed_audit_20260922.md`.
- 전체131개 CT/GT 사전 감사 완료: innertrain84/innerval21/outerval26, train eligible527개, blind radius27.290655635700848mm, full graph39,936nodes/2,040,356edges; exact decoded CT 중복 및 innerval blind 위반 없음. 감사: `work/v222_train_20260922/cohort_preflight/summary.json`.
- 로컬 Basic 준비와 raw SHA/분할/ResEncM plan이 같다는 것은 확인했다. 과거 서버 점수0.6454의 exact split/seed는 미확인. 수정 전 감사 `work/v222_train_20260922/basic_cp_comparison_audit.json`은 이력으로 보존한다.
- **전체 GNN40epoch·nnU-Net250epoch·평가 미실행.** Case ID를 검증된 환자 그룹이나 제공된 주석을 완전한 주석으로 위조하지 않았다. Production identity/annotation 계약 문제와 자원 보정은 미해결 상태다. 이번 시드 수정은 해당 gate를 우회하지 않는다.
- 이전 소스 `versions/v2.22/before_comparison_seed_fix_20260922/`; 새 검증 `versions/v2.22/verification_comparison_seed_20260922/`. L0/L1/L2 원본 소스와 기존 Basic CP reference hash는 유지했다. 새 source hash에 맞는 준비가 필요하며 기존 manifest/checkpoint hash를 덮어쓰지 않는다.

## 2026-09-22 v2.22 r2 — 환자 간 prototype 군집 정렬과 References

- 현재 format: `hiercp_v222_cluster_alignment_r2`. r1 소스·문서는 `versions/v2.22/before_cluster_alignment_r2_20260922/`에 보존했다. L0·L1 구조/너비/깊이 및 전체 그래프 규칙은 유지했다.
- L2: query 환자 그룹을 먼저 제외한 뒤 실제 관측 근거가 있는 환자별 L1 label을 클래스별 cosine average-linkage로 군집화한다. 비단독 cut 전체의 양의 silhouette로 K를 선택하고 근거가 부족한 경우 K=1 사유를 기록한다. 환자/그래프를 버리는 cap은 없다.
- 고정 teacher center·assignment는 환자 episode마다 fit하고 batch에서 재사용한다. 기존 L2 2층/128/4 heads에 class-balanced alignment CE를 연결했다. Total loss는 query CE + alignment CE, CP 점수는 live L2 군집 중심의 환자 수 가중 log-sum-exp다. 모든 128개 후보를 점수화하는 online CP 흐름은 유지한다.
- 단독 환자의 미관측 class label은 prototype fit에서 제외한다. Query/validation 정답은 군집 fit에 넣지 않는다. Epoch 로그에 K 후보/선택 사유/점유 수/support 환자 목록/제외 query group/군집 안정성 ARI/중심 유사도·분산을 기록한다.
- 참고문헌을 `REFERENCES.md`에 별도 정리했다. SwAV, DeepCluster, PRODIGY, GATv2, silhouette, 계층 군집, nnU-Net 및 과거 검토 자료에 대해 실제 적용·프로젝트 변형·미사용을 구분한다. **SwAV 그대로의 재현이나 view-only L2라고 주장하지 않는다.**
- 군집 합성 DEBUG 검사 8개 추가, 기존 검사를 포함한 회귀 39개가 모두 통과했다. 정적 검사 18파일과 r1 대비 L0/L1 AST 보존도 확인했다. 실제 CT 3명/6개, 그래프당 24,174 nodes/1,229,900 edges, 파라미터 1,519,063개를 유지한 CUDA batch 2/4에서 gradient·optimizer 갱신을 확인했다. Peak CUDA 0.612/1.100GB. 실제 CT fixture는 support 2명/클래스라 K=1/1이고 다중 군집의 의료 효능 검증은 아니다.
- 전체 GNN 40-epoch·nnU-Net 250-epoch 학습 및 전체 평가 미실행. 실제 임상 군집/소형 종양 성능 개선은 미검증이다. Production identity/annotation manifest, 전체 가림 범위 감사가 여전히 필요하다. 기존 결과를 덮어쓰거나 학습 checkpoint를 만들어 내지 않았다.
- 상세: `docs/pipeline_v222_cluster_r2.md`; References: `REFERENCES.md`; 검증: `versions/v2.22/verification_cluster_r2_20260922/`; 최종 소스 해시와 일치하는 실제 CT 검증: `work/v222_20260922/cluster_r2_final_debug/result.json`. `code.txt`는 현 소스로 갱신하며 export receipt를 같은 검증 폴더에 기록한다. 아래 r1 이하 절은 보존 이력이다.

## 2026-09-22 v2.22 r1 — 관측 관계 학습과 누수 차단 구현

- 사용자 승인에 따라 독립 `hiercp_v222/`, `config/prompt_graph_v222.json`, `run_v222.py`를 구현했다. v2.21은 `versions/v2.21/before_v222_20260922/`에 보존했고 기존 버전 소스/설정은 변경하지 않았다.
- L0: CT 한 채널 CNN(12/24/32) + 공간 GATv2 3층/128차원/4 heads. 수작업 특징·종양 내부/정답 표면 노드를 입력하지 않는다. 학습 분할 전체로 정한 동일한 중심 가림을 보간 전에 적용하고 고정 물리 격자 전체를 사용한다.
- L1: 실제 주석으로 정의한 두 관측 클래스와 T/F support edge를 사용하는 2층 관계 attention. Query의 정답은 loss에만 전달한다. 환자별 자유 latent 16개와 관측 클래스 2개를 혼동하지 않는다.
- L2: query 환자 그룹 전체를 제외한 training support의 환자별 표현을 2층 cross-patient attention으로 정렬한다. Query CE가 L2/L1/query L0에 연결된다. Support L0 memory는 전체 inner-train에서 매 epoch 갱신하는 detached embedding이며 이 학습 선택을 문서화했다.
- 준비/40-epoch GNN 학습/frozen scorer/온라인 CP bank/250-epoch nnU-Net/predict/기존 v5 CSV 평가 CLI 경로를 연결했다. CP 후보 128개 전체 점수와 확률 0.5를 유지한다. 구/DEBUG checkpoint는 production bank에서 거부한다.
- 새 검사 12개 및 회귀 검사 총 31개 통과. 실제 CT 3개/6개 그래프의 결정론 CUDA DEBUG에서 전체 CNN/L0/L1/L2 gradient·optimizer 갱신 및 누수 개입 검사를 통과했다. 별도 native runtime의 V222 트레이너 import도 통과했다.
- DEBUG 그래프당 24,174 nodes/1,229,900 edges; 전체 파라미터 1,519,063개. 전체 그래프를 유지한 attention 계산 분할로 batch 2/4 최대 CUDA 할당량 0.605/1.091GB를 측정했다. Production batch는 full-cohort 자원 보정에서 정한다.
- **전체 데이터 준비·40-epoch GNN·250-epoch nnU-Net·전체 online CP/평가는 실행하지 않았다. 학습 checkpoint나 성능 결과도 만들지 않았다.** 본 준비에는 검증된 환자 그룹/주석 범위 manifest와 전체 가림 범위 감사가 필요하다. 점수는 관측 문맥 순위이며 보정된 종양 발생 확률이 아니다.
- 상세: `docs/pipeline_v222.md`; 검증: `versions/v2.22/verification_20260922/checks.json`; 실제 CT DEBUG: `work/v222_20260922/real_debug4_chunked/`.

# HierCP GPT handoff

## 2026-09-22 v2.21 작업 중 — T/F/U와 기하 조건 분리, PRODIGY 근거 정정

**v2.21은 아직 전체 모델 완성본이 아니다.** 기존 v2.2-r5는 보존했고 별도 `hiercp_v221/`, `config/prompt_graph_v221.json`, `run_v221.py`에 작업 중이다.

- 사용자는 T/F/U가 data-label 관계이며 F가 기하 불가능을 뜻하지 않는다고 정정했다.
- 관계 부여 기준 질문에 “원래 PRODIGY에서는 어떻게 했는데?”라고 답하여 원문과 공식 코드를 확인했다. 원문은 task class가 먼저 있고 support 정답 클래스=T/다른 클래스=F다. 난수는 label 초기 표현이며, 정답 없는 자유 latent slot 자체가 아니다. Query는 정답을 입력하지 않는다.
- 자기지도 Neighbor Matching도 그래프 이웃으로 임시 클래스/정답을 정의한 후 task를 구성한다. 이 관계가 CT 종양 발생 가능성의 정답이라는 뜻은 아니다. 논문 embedding은 256이며 현재 16/128의 근거로 삼지 않는다.
- 현재 구현 범위: 명시된 episode class에서 support T/F와 query U 및 방향/edge 속성을 생성하는 `relations.py`. 합성 DEBUG3개 통과. 기하 정보를 F로 변환하지 않는다.
- 의료 task/class 정의와 L1/L2 통합은 미완료다. 새 loss/teacher/클래스 부여 규칙을 임의로 추가하지 않았다. copied base를 v2.21 완성 모델로 실행하지 못하도록 차단했다.
- 상세 및 출처: [PRODIGY 관계 기준과 진행 상태](docs/prodigy_relation_basis_v221.md). `run_v221.py check`는 통과했지만 학습/평가 성공을 뜻하지 않는다.

## 이하: 이전 구현 상태의 보존 기록


## 2026-09-22 최신 정정 — 폐기한 종양 내부 노드 제거, data/label 근거 구분 (v2.2 r5)

**이 절이 아래 r4/r3의 “기존 종양 내부 유지”, “L0/L1 설계 확정” 설명보다 우선한다.** 사용자는 종양 내부 노드가 폐기된 설정이라고 명시했다. 과거 코드 보존을 이유로 활성 모델에 다시 넣지 않는다.

- `tumor_interior` 생성·세 관계 엣지·전용 projection/GNN/pooling을 제거했다. v2.2 전용 schema는 5종 node/13종 edge다. legacy 공통 checkpoint helper에 의한 재유입도 차단했다. 구 node가 들어간 cache/model 입력은 오류로 거절한다.
- surface/context CT-only CNN과 남은 GNN/L1/L2 깊이·hidden 폭은 유지했다. 별도 내부 그래프 노드 제거이며 CNN receptive field에서 내부 CT의 영향까지 없앴다는 뜻은 아니다.
- 사용자 원문에서 data는 L0가 인코딩한 실제 local context다. 현재 구현은 donor–candidate pair embedding을 data로 삼는다. 이 구체적 단위를 사용자 정의로 설명하지 않는다.
- label은 task별 독립 학습 가능한 잠정 latent 기준이다. 현재 L1 attention에는 원문의 T/F/U edge 정보가 없다. data별 T를 L2에서 집계한다고 data–label 관계 학습 계약이 구현 완료인 것은 아니다.
- label 개수 16은 기존 capacity를, 차원128은 기존 hidden_dim을 계승한 값이다. 사용자 지정값/생물학적 종류/최적값이라는 근거가 없다. 현재 projection/attention의 차원 호환과 연구적 타당성을 혼동하지 않는다. 임의 대체 수치를 추가하지 않았다.
- `l1_contract_status=unverified_pair_data_unit_and_missing_TFU_edge_conditioning`, label 개수/차원 rationale를 미검증으로 정정했다. 전체 학습은 미실행이며 학습 목적함수 문제도 남아 있다.
- 회귀 DEBUG16/16, 실제 inner-train CT2명/4graph CUDA BF16 batch2/4와 남은 L0/L1/L2 gradient/optimizer 경로 통과. 이것은 의미적 설계나 의료 성능 검증이 아니다.
- 보존본 `versions/v2.2/before_retired_interior_removal_r5_20260922/`; 증거 `versions/v2.2/verification_no_interior_20260922/`; 현재 format `hiercp_prompt_graph_v22_surface_context_cnn_r5`.
- [정의·근거·폐기 설정 상세](docs/design_contract_corrections_v22_r5.md). code.txt와 패치노트를 같은 소스로 최신화한다. 전체 학습, GitHub push, 서버 배포는 하지 않았다.

## 이하: 이전 버전·설명 보존 기록

## 2026-09-22 정정 — CNN L0/L1 유지, 기존 L2 복구 (v2.2 내부 r4)

**이 절이 아래 r3의 L2 삭제·미구현 설명보다 우선한다.** L0 특징 단순화는 L2 삭제를 허가한 요구가 아니었다. 이전 코드에서 L2는 L1의 128차원 label만 받으며 수작업 특징을 직접 요구하지 않는다. 통계 기반 보조 loss와 L2 모듈을 혼동하여 함께 삭제했던 오류를 수정했다.

- `PromptGraphModel`에 기존 `CrossPatientAlignment` 2층/128/4 heads, 환자 간 label 대응과 외부 관측 T 전달을 복구했다. L0는 CT-only CNN 특징32, L1은 기존2block/16labels로 유지한다. 두 모듈 모두 가중치 학습 가능하다.
- L2 클래스·증거 전달·기존 관측 loss는 r2 보존본과 AST 동일하다. 전체 forward와 `forward_tasks`가 작동하며 CNN부터 L2까지 기존 관측 loss로 gradient와 optimizer update가 확인됐다.
- 통계 descriptor/복원 decoder/통계 유사도 teacher는 복구하지 않았다. **L2가 없거나 역할이 미정인 것이 아니다.** 전체 학습 목표가 통계 보조 loss 제거 이후 미완성인 것이다.
- 현재 준비 데이터는 T/U이며 F는 hard geometry gate에서 제외한다. 균등 label/대응 분포가 모든 후보 score=1, 관측 loss=0을 만족하는 반례를 검증했다. 따라서 기존 관측 loss만으로 전체 학습을 임의 재개하지 않는다. U를 F로 바꾸거나 새 teacher/loss를 추가하지 않았다.
- 단위/회귀 DEBUG 15/15, 실제 inner-train CT2명/4graph, CUDA BF16 batch2/4, L2 두 층을 포함한 주요 그룹 갱신 통과. 전체 학습·평가·소형 종양 성능 검증은 미실행이다.
- 증거: `versions/v2.2/verification_l2_restore_20260922/`. 수정 전 r3는 `versions/v2.2/before_l2_restore_r4_20260922/`에 보존했다. 현재 format은 `hiercp_prompt_graph_v22_ct_cnn_l0_l1_l2_r4`다.
- 상세: [L2 복구와 설계 결함 감사](docs/pipeline_v22_l2_restoration.md). code.txt도 현 소스로 재생성한다. GitHub push/서버 배포는 수행하지 않았다.

## 이하: 과거 구현·판단 보존 기록

## 2026-09-22 현재 확정 — v2.2 CT-only CNN L0 / 기존 L1 (r3)

**이 절이 아래 과거 r1/r2/audit 기록보다 우선한다.** 사용자가 `CNN으로만 특징 추출`을 선택한 뒤 L0·L1을 고정해 v2.2로 구현하라고 요청했다. 현재 format은 `hiercp_prompt_graph_v22_ct_cnn_l0_l1_r3`이며 상세는 [현재 L0/L1 계약](docs/pipeline_v22_cnn_l0_l1.md)이다.

- CT 1×48³ → CNN 12/24/32 → 각 node의 CNN 특징 32만 → projection 128 → 기존 GNN 3층/4 heads → data node 128 → 기존 환자별 L1 2 block/16 labels/128. 상대 좌표를 추가한 35차원 모델이 아니다.
- 노드 평균/표준편차/곡률/법선/SDF 등 입력과 엣지 10차원 속성을 제거했다. 마스크·좌표·기존 shell은 노드 구성/연결/집계의 구조 정보로만 유지한다. CNN 입력에도 마스크·거리장 채널이 없다.
- ‘고정’은 architecture 계약이다. 가중치 freeze가 아니며 모든 파라미터가 학습 가능하다. 6,453,844개는 환자 label table 2개를 둔 DEBUG 모델 기준이며 등록 환자 수에 따라 달라진다. L1 AST 보존 검증 완료.
- 완료 범위는 L0/L1 구현·검증. 수작업 통계 복원·L2 teacher를 제거했고 임의 대체 목표는 추가하지 않았다. L2 목표 미확정 때문에 train/bank/nnunet-train/predict/evaluate는 설명 있는 오류로 차단한다. 사용자가 새 목표를 정하기 전 전체 학습 가능 모델로 보고하지 않는다.
- 새 DEBUG 8/8, 정적 검사, 실제 CT 2명/4 graph, CUDA BF16 batch 2/4, 모든 파라미터 gradient 및 주요 그룹 optimizer 갱신을 확인했다. 실제 segmentation 성능·전체 학습 결과가 아니다. 증거: `versions/v2.2/verification_cnn_only_20260922/`.
- r2 원본 소스/문서/code.txt는 `versions/v2.2/before_cnn_only_r3_20260922/source.zip` 및 manifest에 보존. 이전 verifier/test는 같은 폴더 `historical_checks/`로 보존했다. 기존 의료 데이터·실험 결과·v1/v2.1 runtime은 유지했다.
- 이 인계를 반영해 code.txt를 새 소스 manifest와 함께 재생성한다. GitHub commit/push나 서버 배포는 수행하지 않았다.

## 이하: 이전 판단·구현 상태의 보존 기록

## 2026-09-22 우선 정정 — 특징 근거 대조, 모델 작업 중단 유지

이 절이 아래 보존 기록의 ‘현재/최신’보다 우선한다. [특징·L2 근거 대조](docs/feature_evidence_audit_20260922.md)에 사용자 첨부 원문, 논문 5편, IBSI, 실제 입력/손실 코드를 대조했다. 수작업 특징의 일반 선행 근거는 있지만 현재 CP용 16/24차원 조합과 통계 기반 L2 teacher의 직접 유효성 근거는 확인하지 못했다. 첨부 글에는 통계 특징 **제안**이 있으나 특정 논문 재현이라고 명시하지 않았고, 이후 사용자의 임의 특징 선택 반대를 무시할 근거가 되지 않는다.

현재 로컬 r2 코드는 `hiercp_prompt_graph_v22_physical_nodes_r2` / `physical_observation_boxes_mm_masked_r2`, 24차원 관측 특징 및 144차원 복원 descriptor다. 아래 16차원·116차원/r1 cache 변환 설명은 과거 r1 기록이며 현재 r2 계약이 아니다. r2 구현 및 실제 데이터 검증은 미완성·중단 상태다. 과거 r1 DEBUG 통과를 r2 완료로 보고하지 않는다. `code.txt`와 기존 v2.2 상세 문서는 이 r2 상태로 전체 동기화되지 않았다.

이번 작업은 근거 대조와 문서 정정이며 특징 삭제/추가·모델/설정 수정·테스트/학습 재개를 하지 않았다. 아래 과거 실행 시작 문구는 현재 실행 상태를 뜻하지 않는다. 사용자 중단 지시 이후의 중단 상태를 유지한다.

## 이하: 2026-09-20 작성 당시 기록 보존

갱신일: 2026-09-20 KST. **현재 구현은 v2.2 — CNN 없는 원본 관측 특징 기반 L0 + 기존 환자별 L1 / 환자 간 L2다.** 전달 소스 목록/SHA는 동반 code.txt를 따른다.
로컬 working tree 상태이며 GitHub commit/push 또는 서버 배포를 완료했다는 의미가 아니다.

버전 이력의 기준은 [PATCH_NOTES.md](PATCH_NOTES.md)다. v1은 기존 파이프라인, v2.0은 잘못된 view-only 구현(폐기), v2.1은 기존 환자 간 정렬 구현, v2.2는 CNN 없는 비교 구현이다.

## v2.2 최신 상태

- 진입점 `run_v22.py`, 모듈 `hiercp_v22/`, 설정 `config/prompt_graph_v22.json`, format `hiercp_prompt_graph_v22_raw_nodes_r1`. 상세 계약: [v2.2](docs/pipeline_v22.md).
- 원본 CT·통계·기하 특징 16개 → MLP → 기존 gated GNN. 학습 가능한 Conv3d 0개. KD-tree 물리 그래프·node/edge·후보·학습 규모를 유지했다. 옥트리/MST/FPS/새 cap/CNN 보조 경로를 함께 추가하지 않았다.
- dense patch GPU 전달·저장을 제거하되, L2 손실 목표의 패치 통계 20개는 보존했다. v2.1 GNN weight/support/bank는 재사용할 수 없고 새 학습이 필요하다. `convert-cache`는 전체 완료 cache의 geometry만 새 파일로 변환한다. `reuse-native`는 검증된 전처리를 참조하고 결과 폴더를 분리한다.
- 회귀 32개, CUDA BF16 전체 폭 loss/gradient/optimizer 1 step, 실제 의료 graph 4개의 geometry·descriptor 완전 일치 검증. 증거 `work/v22_20260920/debug2/verification.json`. CUDA profiler 표본에서 CNN 약 5–6 ms, 그래프 연산이 더 큰 비중이었다. 전체 학습·CP latency 추정으로 쓰지 않는다.
- 실제 후보 128개 전체와 실제 inner-train T anchor 3명의 support를 미학습 v2.2 Scorer에 넣어 유한 점수·argmax를 확인했다. 증거 `work/v22_20260920/pool_debug2/verification.json`. production 전체 support/자동 resource calibration/새 native paste까지 검증한 것은 아니다.
- 전체 cache 변환, GNN 40 epoch, nnU-Net 250 epoch, 전체 의료 평가는 미실행. 기존 v2.1 실행을 중단/교체하지 않았다. T-only 관측 손실의 순위 학습 타당성, L2 통계 기반 목표, train/scoring support 차이는 해결됐다고 보고하지 않는다.
- v2.1 실행 소스 및 이전 문서 원본: `versions/v2.1/before_v22_20260920/source.zip` / `manifest.json`. v1 원본 SHA와 함께 검사한다.

## 이하: 보존한 v2.1 인계·과거 실행 기록

아래의 ‘현재’ 또는 ‘최신’은 각 기록 당시 v2.1을 뜻한다. v2.2 학습 완료 상태로 읽지 않는다.

## 최신 설계 정정: view-only L2 폐기

**Basic CP 사용자 정정:** 원본 정책을 유지하고 실행 시점만 온라인으로 옮긴 `basic_cp_online/` / `run_basic_cp_online.py`를 분리했다. 자기 환자의 모든 종양에서 무작위 하나, 4,000개 이내 탐색 중 첫 유효 위치, 1회 paste, 원본 HU jitter다. ≤20 mm 제한·128 후보 풀·공통 donor·추가 50% gate가 없다. 원본의 uint8 거리 오류도 literal 재현이라는 사실을 명시했다. 일반 nnU-Net crop과 validation을 유지한다. 과거 변형 Basic 결과와 구분하며 상세·미검증 범위는 [원본 Basic CP 온라인](docs/original_basic_cp_online.md)을 따른다. 현재 GNN 공통 donor 정책은 별도 실험이며 이 원본 Basic과 동일 donor 조건이 아니다.

원본 온라인 Basic DEBUG 8개 통과. 실제 `liver_2`의 자기 종양 25.3166 mm / 14,131 voxel을 그대로 선택해 증강했고 원본 스크립트의 전체 raw CT/GT와 정확히 일치했다. 원본 파일 SHA 불변. 증거는 `work/original_basic_cp_online_20260919/debug1/verification.json`이다. 전체 Basic의 preparation·worker/physical batch 처리량 측정·250 epoch 학습은 아직 실행하지 않았다.

**최신 실행 — 2026-09-19 22:21 KST:** `work/local_v21_5070ti_20260919/attempt3_shared_donor/`에서 수정본 전체 context 준비→GNN 40 epoch launcher를 시작했다. 완료된 131명 원본을 SHA로 재검증하여 사용하며 이전 비대칭 graph cache는 쓰지 않는다. 실제 준비/학습 단계는 해당 디렉터리의 started/complete/failed 파일과 log를 따른다. 전체 학습·의료 평가는 아직 완료되지 않았다. 현재 소스와 일치하는 증거는 `versions/v2/verification_shared_donor_20260919/`다.

**공통 donor 실제 연결 검증 완료:** 회귀 39/39와 현재 소스의 CUDA optimizer DEBUG 1 step 통과. 소형 source 0개의 `liver_2`에 inner-train `liver_1/component1`을 사용해 전체 128위치를 평가했고 native CT/GT 3,165 voxel을 paste했다. 붙인 영역 밖 GT·원본 SHA는 유지됐다. 미학습 GNN 연결 검사이며 성능 점수나 전체 학습 완료가 아니다. 재실행 및 남은 production 자원 검증 상태는 아래 로컬 기록을 따른다.

**2026-09-19 공통 donor 수정:** 현재 정책은 `shared_inner_train_donors_uniform_per_cp_event_v1`이다. 모든 outer-train 105명이 소형 종양 유무와 무관하게 같은 inner-train donor 527개에서 CP 이벤트마다 선택받는다. 실제 선택된 pair만 128개 후보를 모두 평가하고 argmax의 native paste 한 개를 요청 시 생성한다. GNN은 실제 T 662개를 보존하고 donor와 환자를 모두 포함하는 균형 context round를 사용한다(train 67,983 / validation 67,591). 기존 비대칭 cache/bank는 재사용하지 않는다. split·원본 SHA·checkpoint memory·donor membership·일반 validation loader 분리를 검증하며 회귀 39개가 통과했다. 현재 계약은 [공통 donor·누수 방지](docs/shared_donor_leakage_v21.md), 최신 실행 증거는 [로컬 기록](docs/local_v21_5070ti_run.md)을 우선한다. 아래 중단/실행 기록은 수정 전 이력이다.

**최신 실행 정정 — 2026-09-19 donor 정책 검토로 중단:** 소형 source가 있는 81명은 자기 source만 사용하지만 없는 24명에게만 모든 inner-train source 527개를 사용하는 비대칭 분기가 사용자 지적으로 확인됐다. 이 정책을 L2 정렬에 필수인 설계로 간주하면 안 된다. graph 준비 PID 37104만 중단했고 전체 GNN 학습은 시작하지 않았다. 부분 cache는 미완성이며 원본·결과는 보존한다. native 전처리는 131/131명 완료. 중단 기록은 `work/local_v21_5070ti_20260919/attempt2/stopped_donor_policy_review.json`이다. 현재 후보 정책은 타당성 검토가 필요하며 아직 대체 정책을 임의로 적용하지 않았다. 아래 실행 중 설명은 이전 상태다.

**2026-09-19 전체 데이터 실행 상태 정정:** 공식 Task03 Liver 전체 archive 다운로드·MD5·추출 및 CT/GT 131명 검증 완료. 첫 GNN 준비는 source inventory 37/105에서 RAM admission 오류로 실패했고, mask의 lazy 생성과 공통 source 무손실 압축 저장으로 수정한 뒤 `work/local_v21_5070ti_20260919/attempt2/`에서 전체 준비→40 epoch를 다시 실행했다. 기존 설정상 예정 graph는 1,704,342개다. 131명 nnU-Net planning도 완료하고 전처리를 별도로 실행했다. **현재 완료 증거는 실데이터 DEBUG optimizer 1 step까지이며 전체 GNN/native 학습·평가는 완료되지 않았다.** 실제 CT 3명/graph 6개/BF16에서 loss 3.203799247741699, 모든 gradient 유한값과 핵심 모듈 갱신 확인. peak VRAM 1,086,012,416 bytes는 작은 DEBUG batch 측정이지 전체 support 메모리 검증이 아니다. 최신 정적 17개·회귀 26/26 및 실데이터 CUDA 증거는 `versions/v2/verification_storage_real_cuda_20260919/`에 보존한다. 실행 로그와 남은 전체 규모 검증은 [로컬 실행 기록](docs/local_v21_5070ti_run.md)을 따른다.

**2026-09-19 로컬 GPU 실행 추가:** RTX 5070 Ti 16GB에서 v2.1 BF16 실행 중 발견한 L1 `index_add_` dtype 불일치를 수정했다. `.venv`는 PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128 / PyG 2.6.1 / nnU-Net 2.8.1이다. 실제 CUDA full-width DEBUG 1 step 및 회귀 26/26 통과, 실패·skip 0개. 상세·실행 상태는 [로컬 실행 기록](docs/local_v21_5070ti_run.md), 변경 이유는 [패치 노트](PATCH_NOTES.md)를 따른다. 아래의 CPU-only·25개 검사 기록은 GPU 환경 구축 전 상태다. 의료 전체 학습·평가 완료로 읽지 않는다.

사용자가 명확히 한 L2는 동일 데이터의 augmentation view를 일치시키는 단계가 아니다.
**서로 다른 실제 data/task/patient의 local latent label space를 정렬하고, 관측 positive가
희소한 context로 다른 환자의 positive evidence를 전달하는 단계**다.

기존 v2.0의 두 stochastic L0 view→각각 k-means→contingency/view consistency 구현은 이 의도와
달랐다. 해당 소스·당시 gpt_handoff/code.txt 218개를
`versions/v2/superseded_view_alignment_20260919/source.zip`과 SHA manifest에 보존하고 교체했다.
기존 v1(내부 모델 revision v5)과 모든 과거 실험 결과도 보존한다.

## 현재 코드의 정확한 역할

- artifact format: `hiercp_prompt_graph_v2_cross_patient_r1`.
  `run_v2.py`, `hiercp_v2/`, `config/prompt_graph_v2.json`이 활성 구현이다.
  v1 및 폐기된 view-only v2.0의 cache/checkpoint/bank는 v2.1 format으로 재표기하거나 재사용하지 않는다.
- **L0:** 기존 full-scale KD-tree 국소 physical graph와 gated GATv2 encoder를 재사용한다.
  그래프 생성과 인코딩은 다른 단계다. 출력은 128-D data node다.
- **L1:** 실제 patient ID별 독립 random trainable label parameter table을 둔다.
  공유 연산자로 각 환자 안에서 data↔label 관계를 학습한다. query는 label로부터 받기만 한다.
  label ID는 global class가 아니며 다른 환자의 같은 번호와 같다는 가정을 하지 않는다.
  k-means는 제거했다. task를 두 stochastic view로 복제하지 않는다.
- **L2:** 환자 간 label cross-attention/correspondence를 학습하고, 다른 inner-train 환자의
  실제 T observation만 positive distribution으로 집계해 target 환자의 latent space로 전달한다.
  자기 환자의 T는 제외한다. B에 T가 없더라도 A/C의 T를 통해 B의 후보 점수를 계산할 수 있다.
- **T/F/U:** T=실제 적격 tumor source 원위치, F=명확한 기하학적 불가능,
  U=유효하지만 tumor positive 미관측. 정상 간을 F로 만들지 않는다.
  생산 cache는 T/U를 만들고 F 위치는 기존 hard geometry 계약으로 차단한다.
  U는 negative loss/분모에 들어가지 않으며 U를 T로 덮어쓰지 않는다.
- candidate score는 target의 data–label posterior와 다른 환자에서 전달된 positive distribution의
  compatibility다. 암 존재 확률이 아니다. 외부 evidence가 없으면 available=false로 구분하고
  production scorer는 오류를 낸다. 최소 2명의 inner-train 환자에 실제 T가 있어야 본훈련이 가능하다.
- 원본 CT/GT는 수정하지 않는다. 높은 compatibility만으로 normal liver를 tumor=2로 바꾸지 않는다.
  실제 donor paste를 한 augmentation sample에서만 pasted mask의 synthetic segmentation GT를 만든다.

## 준비·학습·추론의 연결

- 모든 실제 적격 source 원위치 T를 보존한다. U context는 모든 inner-train donor와 모든 환자를 포함하는 균형 round로 배정하며 이벤트마다 유효 후보 128개를 준비한다. 소형 종양 유무에 따른 donor 분기는 없다.
  donor footprint의 target native spacing 변환과 원본 source/target voxel 수를 기록한다.
- unseen/inner-val 환자는 독립 random 초기 label과 공유된 GNN 연산자로 context adaptation한다.
  그 환자의 정답으로 label parameter를 최적화하거나 그 환자를 evidence donor로 쓰지 않는다.
- L0 hidden 128 / heads 4 / layers 3, L1/L2 layers 2/2, CNN 12/24/32, patch 5×48³,
  GNN 40 epoch, nnU-Net ResEncM/3d_fullres 250 epoch, 후보 128개, CP 확률 0.5 유지.
- K=16은 기존 capacity 예산을 참고해 유지한 구현 설정이다. 사용자가 지정한 절대 범주 수가 아니다.
  각 training patient의 label은 16×128 parameter이며 환자 3명의 DEBUG 전체 모델은 7,086,156 parameters다.
- objective는 관측 T/F transfer 오차 + 실제 context 통계 reconstruction + 환자 간 soft alignment다.
  U도 reconstruction/정렬에는 참여한다. context 통계 유사성을 correspondence teacher로 쓰는 것은
  **이번 구현의 연구 가정**이며 biological ground truth 또는 입증된 최적 objective가 아니다.
  두-view consistency로 대체하지 않았지만 실제 collapse/shortcut/성능은 본실험에서 검증해야 한다.
- support L0 memory는 매 epoch 전체 갱신한다. L1/L2 및 query L0는 loss/backward/optimizer에 연결된다.
  frozen scoring/validation에서는 동일 support/L2 state를 query batch마다 다시 계산하지 않고 재사용한다.
- 선택 정책은 `cross_patient_positive_transport_argmax`. v1의 feedback curriculum과 동일 실험이 아니다.
  현재 native CP는 모든 outer-train 환자에 같은 inner-train 공통 donor 풀을 사용한다. CP 이벤트 확률은 0.5이며 validation은 증강하지 않는다.
- 출력 경로는 새 디렉터리를 요구한다. 자동 resume는 미구현이다.
  실제 실행 명령 및 설계 상세는 `docs/pipeline_v2.md`를 따른다.

## 결과·검증 상태와 다음 작업

- 출력: case_metrics.csv / lesion_metrics.csv / summary.csv / size_metrics.csv.
  ≤10 /10–20 />20 mm별 병변 수·검출 수·recall·lesion Dice를 기록한다.
  Dice≥0.10 등 매칭 기준과 크기 기준을 혼동하지 않는다.
- 새 검증은 `versions/v2/verification_cross_patient_final_20260919/`에 기록한다.
  **25/25 합성 DEBUG 회귀 통과, 실패/skip 0개, Python 16개 정적 검사 통과**다.
  이전 `verification_20260919`, `verification_handoff_20260919`의 검사는 폐기된
  view-only 구조에 대한 역사적 증거이며 현재 patient-transfer 검증으로 재인용하지 않는다.
- 핵심 DEBUG 검사는 independent trainable labels, U gradient=0, T가 없는 B로 외부 evidence 전달과
  점수 변화, own-T 배제, 환자별 L1 독립성, query 역류 차단, label 순열, BF16,
  전체 폭 L0/L1/L2 gradient/optimizer update, 실제 foreign-donor graph에서 CT/GT 비변경이다.
  128개 후보/환자 누락 방지 준비-loop 테스트는 geometry를 명시적으로 mock한 별도 정책 검사다.
- **실제 의료 데이터 전체 GNN 학습, nnU-Net 학습, native 처음부터 끝까지 실행, 의료 평가,
  target GPU 처리량 측정은 미실행**이다. 실제 v2.1 성능 수치나 학습된 가중치를 만들었다고 보고하지 않는다.
  로컬 PyTorch는 CPU 빌드이고 nnunetv2가 없다. CPU 16개/RAM 약 68.6 GB와 GPU 장치는 확인했다.
- GNN batch/worker 측정, segmented patient batching, case 병렬 준비 경로를 구현했지만
  DDP는 미구현이다. nnU-Net planner batch/worker의 target GPU 검증도 남아 있다.
- 다음 실험은 현재 CPU DEBUG 통과를 근거로 전체 학습이 완료됐다고 단정하지 않고,
  새 cache/data 규모·자원·native 연결·positive-transfer 효용을 실제 환경에서 확인하는 것이다.
  서버 실행/프로세스 중단/기존 결과 교체를 자동으로 수행하라는 지시가 아니다.

## v1 보존 및 과거 기록

`versions/v1/manifest.json`과 원본 ZIP에 commit
`74dcc2cf03d2d40d1f582223321d96004333f661` 기준 추적 파일 202개를 보존한다.
루트 gpt_handoff.md/code.txt는 현재 인계 문서이므로 원본은 ZIP에서 SHA 검증한다.
나머지 v1 실행/config 파일은 현재 파일도 원본 SHA와 일치해야 한다.
아래 기존 v5 서버 명령·GPU 5번 할당·성능 수치는 과거 v1 기록이며 현재 v2.1 명령이나 성능이 아니다.
과거 결과 회수 기록은 `docs/results_summary_20260918.md`, `experiment_results/`도 확인한다.
code.txt에는 현재 실행 소스/설정/설명/테스트를 포함하며 의료 데이터·실험 출력·환경·ZIP은 포함하지 않는다.

## 이하: v1의 역사적 인계 기록 — 2026-09-16 기준

당시 최신 변경은 **source-content / population-metric v5**였다.
이전 v4의 785개 중 780개 통과·5개 skip 기록은 과거 검사이며, v5나 Basic 재사용
기능의 검증 결과로 인용하면 안 된다. 현재 변경은 아래 별도의 v5 검사 결과를 따른다.
의료 데이터 전체 학습, GPU 전체 학습, 서버 결과 재평가는 실행하지 않았다.
배포 명령 갱신: 2026-09-15 KST, 사용자가 현재 할당받았다고 명시한 물리 GPU는 5번이다.
Git 배포 여부는 실제 원격 `origin/main`으로 확인하며 로컬 검사·문서 생성과 구분한다.

추가 공유 자료: `feedback/`의 2026-09-12 보고서·진단 코드·증거 ZIP 5개
(원본 합계 267,843 bytes)를 원문 보존하여 저장소에 포함한다. 증거 ZIP의 19개 항목은
보고서/진단 소스/목록/JSON/CSV/로그이며 의료영상은 없다. 이는 당시 코드에 대한 과거
검토 자료이지 최신 수정의 검증 결과가 아니다. 활성 코드 전달용 `code.txt`에는 계속
제외하고 별도로 공유한다. `Medical Data Aug.zip` 원본은 108,928,778 bytes여서
이번 소용량 자료 추가 대상에서 제외했다. 학습 결과·환경·cache도 변경하지 않았다.

## 2026-09-16 Medical Data Aug 원본 소스 통합

- `reference/medical_data_aug/`에 ZIP의 배치 코드 전체와 notebook 원본 24개 cell source를
  복원했다. 배치 `.py.txt`는 알려진 결함을 포함한 역사적 원문이며 활성 실행 파일이 아니다.
  배치의 줄바꿈/줄 끝 공백만 정규화했고 전체 AST는 원본과 같다.
  notebook은 출력·execution count·attachment·metadata를 제거하고 맨 앞에 안내/Run All
  방지 2개 cell만 추가했다. 원본 README의 데이터 공유 URL은 제외했다. ZIP/member SHA와
  변환 후 SHA·각 cell source SHA를 `provenance.json`에 남겼다. 이 소스들은 `code.txt`에도
  포함된다. CT/label 및 렌더링된 의료영상은 Git/전달 문서에 포함하지 않는다.
- 원본 batch와 notebook 전체 소스를 직접 대조했다. batch의 전처리는 3D/첫 채널 로딩,
  dtype/geometry 보존과 source patch 준비이며 별도 nnU-Net 정규화·spacing resampling은
  없다. notebook의 display normalization은 학습 전처리가 아니다. 기존 `hiercp.common`과
  raw-target bank→`onlinecp_raw_resampling.apply_candidate`→online trainer에 CP가 이미
  연결돼 있다. 함수별 대응은 `docs/cp_input_repair.md`에 정리했다.
- 따라서 별도의 Data_aug 생성 단계를 online 앞에 넣지 않았다. 이는 CP를 두 번 적용하는
  변경이 된다. 원본의 uint8 반전 오류·조용한 crop/실패·검증 없는 출력 재사용은 이식하지
  않는다. 원본 4,000회/첫 유효 위치와 현재 128후보/seeded online schedule은 동일 RNG
  실험이 아니며, 기존에 명시한 연구 확장이다. 모델·학습·CP 설정 및 runtime은 변경 없음.
- export 도구는 출력/attachment/metadata가 남은 notebook을 거절하고 기존 code.txt를
  보존한다. notebook을 재실행해 의료영상을 저장한 상태로 그대로 전달하지 않는다.
- 이번 작업은 소스 복원/대조와 DEBUG 검증이다. 실제 의료 데이터 추출·전체 전처리·GPU
  학습·평가를 새로 실행하지 않았으며, 기존 checkpoint/bank/preprocessing을 변경하지 않았다.
- 이번 관련 CPU 회귀 **77/77 통과, 11.304초**: 원본 연산/설정 대조, 소스 SHA/24 cell 보존,
  출력 포함 notebook 전달 거절, raw resampling/trainer, no-placement와 NIfTI geometry 검사.
  첫 패키지형 테스트 호출은 기존 테스트 간 import 경로 때문에 1개 모듈 로딩에 실패했으며,
  tests를 검색 경로에 둔 호출로 7개 모듈 모두 다시 실행했다. 테스트 실패를 skip하지 않았다.
  Python 166개 Python 3.10 AST·설정 JSON 5개 및 ZIP/member SHA/원본 batch AST/24 cell
  source 일치도 확인했다. 이번에는 전체 884개 회귀를 재실행하지 않았으며 아래 수치는 이전
  변경의 기록이다. 생산 runtime·설정·trainer 파일에는 이번 diff가 없다.

## 2026-09-15 전처리 재사용 완료 증거 수정

- 서버의 `Source completed preprocessing stage proof/configuration changed`에 대해
  재사용 검증기의 확정 결함을 재현했다. 기존 d904eeb와 현재 실행기의 `plan` 완료 증거는
  전처리 디렉터리의 `online_cp_preprocess_complete.json`, `splits_final.json`, `dataset.json`,
  설정에 지정된 plans JSON의 **정확한 4개 SHA**다. 3acef3c의 검증기는 다른 2개 marker를
  기대하여 정상 기록도 거절했다. 원본 journal을 재작성해서 맞추는 방식으로 해결하지 않는다.
- producer와 reader가 `_native_plan_evidence_files()`를 공유하도록 수정했다. 4개 항목·SHA와
  증거 전체 형식은 정확히 일치해야 한다. raw marker와 전체 원본/전처리 파일 검증은 기존
  `_verified_preprocess_contract()`에서 계속 수행한다. 두 marker만 있거나 임의의 추가 파일이
  있는 기록을 승인하는 호환 예외는 추가하지 않았다. 모델·학습·CP 조건은 변경하지 않았다.
- 아래 9월 13일 테스트에는 plan producer를 잘못된 2파일 fixture로 대체한 검사가 있어
  이 불일치를 잡지 못했다. 이제 실제 plan producer와 native artifact verifier를 사용하며,
  producer/helper에서 독립된 4개 전체 상대경로 assertion으로 형식을 고정한다.
- 수정 전 실제 producer 기반 재현 1개가 같은 오류로 실패했다. 수정 후 관련 모듈은
  21개 중 20개 통과·Windows symlink 권한 1개 skip, 실패 안내 모듈은 10개 모두 통과했다.
  SHA/항목/실파일 변경·누락, 추가 항목, 구형 2파일 증거, DEBUG 증거 거절도 포함한다.
- 수정 후 전체 CPU 회귀: **884개 중 879개 통과·5개 skip·실패/오류 0개**, 417.071초.
  skip은 Windows symlink 권한 관련 4개와 별도 opt-in production-sized DEBUG 1개다.
  로그: `work/debug_plan_proof_full_c7633e4360dc456bb43160817267d123/unittest.log`.
  Python 165개 Python 3.10 AST, JSON 5개, trainer/helper 10개 SHA 대조와 정적 감사도 통과했다.
  이 검사는 합성 입력/명시적 경계 fixture를 포함한 로컬 CPU 검사이며, 서버 실제 데이터·GPU
  전체 학습 완료 증거가 아니다. 이번 수정에서 서버 원본 파일을 직접 변경하지 않았다.
- 이번 서버 실패는 새 root·runtime·stage journal 생성 전이다. 수정 코드를 받은 뒤 같은
  최초 명령을 **`--resume-experiment` 없이** 실행한다. 첫 실행 전 실패, journal 존재,
  불완전한 root, 재개 요청인데 root가 사라진 경우의 안내를 분리했다. 재개 대상이 없다고
  새 학습을 권하지 않으며, 기존 파일을 지우거나 checkpoint 없이 재시작하지 않는다.

## 수정 전 v5 검증 — 2026-09-13 (과거 기록)

- 전체 `unittest discover -s tests -p 'test*.py' -v`: **873개 중 868개 통과,
  5개 skip, 실패/오류 0개**, 454.604초. skip은 Windows 심볼릭 링크 권한 관련 4개와
  별도 opt-in production-sized DEBUG smoke 1개다. 축소 테스트는 DEBUG이며 최종 학습이 아니다.
- 로컬 로그: `work/debug_v5_full_035e6fbe23464145832d75d3d301fa2b/unittest.log`.
  의료 데이터와 마찬가지로 실행 로그 자체는 `code.txt` 전달물에 넣지 않는다.
- Python 165개 Python 3.10 AST, 설정 JSON 5개와 중복 key 검사, trainer/helper 10개의
  실제 소스·installer·`SHA256SUMS` 대조 통과. `tools.audit`와 `git diff --check`도 통과했다.
  audit의 오래된 `val_margin` 문자열 검사를 현재 margin 계산·선택 연결 검사로 보정했다.
  추가 DEBUG 검사 3개에서 margin tie-break와 누락/비유한 값 거절을 확인했다.
- 기존 모델·raw CP·feedback·재개·평가 검사와 새 population metric, 최종 K-means 소속,
  관측 CT, 중단 후 비파괴 publication, Basic 재사용 provenance 검사를 함께 실행했다.
  초기 scoped 검사에서 발견한 테스트 호출 인자/예외 문구 불일치는 수정 후 전체 검사에 포함했다.
- 로컬 환경: PyTorch 2.6.0+cpu, PyG 2.6.1, CPU affinity 16, 물리 RAM 약 17.62 GiB,
  전체 검사 시작 전 사용 가능 RAM 약 3.69 GiB. 동시에 무거운 Python 검사를 실행하지 않았다.
  합성 1,499,368-edge 실제 attention DEBUG는 전 간선과 gradient를 유지했으며,
  forward+backward 0.708초, 해당 시점 process peak working set 1,025,908,736 bytes였다.
  이 검사의 C4/2-head 설정은 production H128 모델이나 서버 처리량의 검증이 아니다.
- 구현·정적 검사·CPU 회귀/DEBUG 검증과 **의료 데이터 전체 학습·GPU 자원 측정·최종 평가**는
  구분한다. 후자는 미실행이며, 실제 서버 Basic 재사용 가능 여부도 아직 승인하지 않았다.

## 최신 L2 변경과 과거 결과의 의미

- architecture는 `hiercp_source_content_population_metric_v5`, patient graph는
  `patient_source_content_population_v2`, upper policy는
  `source_content_observed_ct_population_v4`다. 기존 graph/checkpoint의 버전 문자열만
  바꾸어 새 결과로 승인하지 않는다.
- K-means는 기존 K16·최대 30회 중심 갱신을 유지한다. 마지막 중심으로 최종 소속을
  재계산하고 그 소속으로 support/dispersion을 만든다. 최종 소속 일치와 완전한 Lloyd
  수렴은 다르며, 반복 수·종료 사유·최종 재할당 및 중심 잔차를 별도로 기록한다.
- prototype bank v2는 전체 training region descriptor·case별 행 수·최종 label/count와
  fit 계약을 보존한다. load/build/save에서 재검산하되 매 candidate assignment마다
  전체 fit 검사를 반복하지 않는다. support는 환자 유병률이 아니라 region 표본 비율이다.
- 관측 CT와 whole-organ union으로 region CT 평균·표준편차를 계산한다. 종양 annotation
  영역을 평균값으로 치환하지 않는다. 이는 치환 footprint 제거이지, 실제 CT에 보이는
  병변이나 모든 source-anchor 단서가 제거됐다는 증명이 아니다. 16D 특징은 유지한다.
- candidate↔region 간선은 6D, prototype 관련 간선은 기존 6D에 표준화 descriptor 거리와
  `거리 - 군집 평균 중심거리`를 더한 8D다. prototype↔prototype은 두 군집 dispersion의
  평균을 참조한다. 상대 top-2 weight나 signed excess를 calibrated confidence, 의학적
  적합성 보증 또는 nnU-Net difficulty라고 해석하지 않는다.
- 원본 CT·라벨, split, 조건이 동일한 native 전처리는 보존·검증 재사용한다. 이번에 의미가
  바뀐 region descriptor·population bank·상위 graph·GNN은 새 workspace에서 만든다.
- `--reuse-basic-from`은 명시한 완료 Basic의 원래 checkpoint/bank/runtime/history를 보존한다.
  새 Full 전에 전체 CP 입력·128 후보 순서·native data/seg/properties/plans/split 및 훈련
  조건 동등성을 검사한다. bank 동등성만으로 checkpoint 재사용을 승인하지 않으며,
  검증 실패 시 Basic 재학습을 몰래 시작하지 않는다. 실제 서버 Basic 재사용 승인은 미실행이다.
- `python -B -m tools.audit_population_bank --prototype-bank <기존_bank.pt>`는 읽기 전용이다.
  구형 v1의 구조 검사와 원래 descriptor 부재에 따른 membership 미검증을 구분한다.
  이 감사만으로 outer/inner cohort 독립성이나 checkpoint 재사용 권한이 증명되지 않는다.

기존 학습 결과는 삭제 대상이 아니라 **이전 설계의 비교·진단 자료**다. 같은 사례·평가 정의로
별도 폴더에서 재평가하고 원래 모델/코드/seed/split/CP 조건을 함께 남긴다. 여러 변경을 함께
적용한 v5 차이를 L2 단독 기여로 주장하면 안 된다. 개발에 사용한 결과는 탐색적 결과로
구분하고, 좋지 않은 결과도 선택적으로 숨기지 않는다.

Quality GNN의 target은 여전히 원래 anchor와 설계된 negative/corruption을 구별하는 proxy다.
같은 prototype가 negative 구분에도 사용되므로 ranking accuracy/MRR만으로 L2의 실제 CP
효용을 증명하지 못한다. `no_population`은 학습되는 L2 제거이지 prototype를 활용하는 모든
전처리·negative 생성의 제거가 아니다. 별도 Difficulty GNN의 실제 segmentation 오차 관측과
Quality의 의미적 compatibility, 최종 downstream 개선은 서로 구분해야 한다.

## 앞서 반영한 source-content v4 수정의 경계

- L1의 실제 source-host 연결을 없애고 source 내용을 모든 24개 region에 같은
  규칙으로 전달한다. recipient 해부학적 문맥은 허용한다. 모든 위치 정보를
  없앴다는 뜻이 아니며, 고정된 생물학적 입력에서 source 주소 bookkeeping이
  점수에 전달되지 않는 계약이다. canonical graph와 checkpoint는 새 버전을 요구한다.
- 상수 0으로 마스킹되던 입력 열은 학습 projection에서 제외하되 H128/4 heads,
  3/2/2 blocks, 전체 graph/128 후보/40·250 epochs는 보존한다.
- 계약은 staging 검증 후 no-clobber 발행하며 같은 완성 계약은 전수 검증 후 재사용한다.
  feedback graph cache는 payload/receipt를 묶은 새 generation으로 발행한다.
- feedback은 환자 준비 결과를 공유하고 graph mmap/witness를 사용한다. 실제 관측만
  update target으로 사용하며, update와 전체 bank prediction의 largest/mixed batch 및
  host/cgroup/prefetch/optimizer/VRAM 예산을 구분한다. 서버 처리량 개선은 아직 미측정이다.
- 새 fresh run도 stage journal과 native child 종료 receipt로 재개한다. 불명확한
  실행을 중복 시작하거나 checkpoint 없는 학습을 새 학습으로 바꾸지 않는다.
  GNN state를 native segmentation state 변경 전에 검증하며, 복원 중 실패한 trainer는
  계속 사용할 수 없게 잠근다.
- 평가 v5는 유효 matching edge에만 secondary score를 적용한다. `--evaluate`는
  학습 완료 후 checkpoint에 결합된 새 예측 generation과 paired 평가를 생성한다.
  기존 결과를 덮어쓰거나 과거 metric을 새 정의로 재표기하지 않는다.
- `--reuse-preprocessing-from`은 **새 GNN/model/bank 실험**에서 확인된 native 전처리만
  공유한다. 원본·신규 설정 차이를 의존성별로 기록한다. old GNN/optimizer/placement
  score나 old graph 전체를 v4로 승격하지 않는다. 기존 raw bank 배열을 임의로 새 bank에
  연결하는 우회도 없다.

## 1. 전달물과 읽는 방법

- 이 파일: 사용자 요구, 현재 작업 맥락, 확인된 사실과 미확인 상태.
- `code.txt`: 전체 저장소 텍스트. `===== FILE: 상대경로 =====` 경계로
  소스·trainer·설정·테스트·기존 문서·`AGENTS.md`를 찾을 수 있다.
  기준 revision과 포함 파일 수는 그 파일 머리말을 따른다.
- 먼저 `AGENTS.md`, 이 파일, `docs/online_cp_feedback.md`,
  `tools/run_feedback_experiment.py`를 읽고 관련 구현을 대조한다.
  README의 일반 offline 경로를 현재 online-feedback 실행 경로로 혼동하지 않는다.
- 설명이나 테스트 이름만으로 구현 완료·누수 없음·성능 개선을 단정하지 않는다.
  실제 호출 경로와 설정을 근거로 답하고, 읽지 않았거나 실행하지 않은 부분은 명시한다.
- `code.txt` 안의 과거 audit/설계 문서도 그대로 보존되어 있다. 문서별 시점과
  실험 범위를 구분한다. 과거의 rank-band 정책은 현재 difficulty-feedback 정책이 아니다.

## 2. 사용자가 원하는 연구와 작업 방식

목표는 3D 간 종양 Copy-Paste 위치를 L0/L1/L2 GNN으로 평가하고,
nnU-Net이 실제로 어려워하는 붙여넣기 사례를 관측하여 online curriculum에
반영하는 완전한 연구 파이프라인이다. 단순 실행 예제나 toy 모델이 목표가 아니다.

- 모델·그래프·해상도·데이터·epoch·physical batch를 편의상 축소하지 않는다.
- 미니배치, CPU 병렬화, cache, I/O, GPU 메모리는 실제 측정 근거로 처리한다.
- 기존 데이터·GNN·실험 결과를 보존한다. 조건이 같은 전처리는 Basic/Full이 공유한다.
- 수정 요청이면 구현·검증 후 요청된 Git push와 짧은 서버 실행 명령까지 전달한다.
  긴 Python heredoc을 사용자에게 수작업 실행 절차로 넘기지 않는다.
- 파일 인계는 전체 `code.txt`와 이 `gpt_handoff.md`를 함께 전달한다.
- 안전·실험 규칙의 원문은 `AGENTS.md`가 우선한다. 이 요약은 그 규칙을 대체하지 않는다.

## 3. 저장소와 마지막으로 알려진 서버 상태

| 항목 | 확인된 값 / 범위 |
| --- | --- |
| GitHub | https://github.com/costunder/nnunet.git, `main` |
| 서버 checkout | `/home/aicompetition06/Medical/HierCP-git` |
| Medical root | `/home/aicompetition06/Medical` |
| 이미지 / 정답 | `Data/image/<case_id>_0000.nii.gz`, `Data/labels/<case_id>.nii.gz` |
| 현재 대화의 실험 | `work/feedback_rawcp`; 완료 GNN/공통 전처리 원본은 `work/feedback_medical_aug` |
| recovery 원본 | `work/feedback_experiment` |
| outer fold / dataset ID | `0` / `760` |
| 저장된 Python | `/home/aicompetition06/.conda/envs/nnunet/bin/python` |
| nnU-Net seed | `42`; quality GNN/bank는 별도 fold-specific 설정을 따른다 |
| 직전 bank 실패 | `liver_101`, component `3`, raw source 1 voxel → 원래 위치에서 독립 resampling한 mask 0 voxel |
| 최신 사용자 제공 오류 | `feedback_rawcp` bank의 `liver_15`, component `1`: raw-target case 저장 전 디스크 reserve 검사 실패 |

최신 첨부 `e1204233-a4a9-4695-a018-846c8ff87284/pasted-text.txt`에서 실패 당시
free는 77,643,907,072 bytes(약 72.31GiB), reserve는 85,899,345,920 bytes(80GiB),
해당 baseline 저장 하한은 1,103,806,600 bytes(약 1.03GiB)였다.
`raw_target_case` 시작 후 0.172초에 사전검사에서 중단됐으며 누적 시간은 약 36시간 13분이다.
RSS 약 290.6GiB, available RAM 약 694.2GiB였지만 **직접 실패 원인은 디스크 reserve**다.
사용자 후속 `df/du` 출력은 공유 NFS `/home` 74T 중 341G available, bank 총 36G
(`raw_cases` 34G, `raw_candidates` 1.8G, `raw_sources` 12M, entries 21M)였다.
따라서 이 bank가 공유 NFS 74T를 채웠다고 단정할 수 없고, `Use%=100%`만으로
사용 가능 공간이 0이라고 해석하지 않는다. 두 측정 사이 free 증가 원인은 미확인이다.
341G는 후속 시점의 공유 여유 공간이며 나머지 전체 bank 완료 용량 보장은 아니다.

기존 재개 경로는 완료 `ok` 행을 검증해 재사용하지만, 디스크 사전검사에서 남긴
`error` 행도 다른 오류와 똑같이 거절했다. 이제 `tools/online_bank_disk_retry.py`가
정확한 native 오류 문구·현재 입력 크기/설정·유일한 Measurement/BankProgress 증거·
해당 case/source의 부분 산출물 부재를 대조한다. 현재 free가 baseline+기존 reserve를
초과할 때만 CSV/config/실패 로그의 원본 bytes와 SHA를 `disk_retry_history/<uuid>.json`에
배타적으로 보관한 뒤 정상 생성 경로로 다시 진입한다. 실패 행을 수동 삭제하지 않으며,
성공/새 실패의 정상 manifest 갱신 전에 이전 실패 증거가 보존된다.
일반 오류, OOM, 저장 중 ENOSPC, 증거 누락/중복, 부분 산출물은 계속 거절한다.
새 CLI 우회 플래그나 reserve 축소는 없다. GNN·전처리·기존 완료 entry·private trainer
runtime·raw resampling 엔진·후보 128개·모델 설정은 바꾸지 않았다.
서버에서 이 수정의 bank 완료와 Full/Basic 학습이 확인된 것은 아니다.

아래 source-history 오류는 **이전 단계의 기록**이다. 당시 `validate_source` 안에서 발생했고,
그 invocation은 새 root 생성,
GPU 검사, bank 생성 및 학습 전에 멈췄다. 구 writer(`301482c` 이전)는 실패 행에
`name/status/error`만 저장했지만 새 upgrade reader가 `input_files` 누락도 설정 변경으로
오인하는 호환성 결함을 코드 이력에서 확인했다. 원격 journal 행 자체는 아직 제공되지
않았으므로 해당 서버 행이 누락인지 실제 SHA 차이인지까지 확정한 것은 아니다.
현재 수정은 구형 실패 행에 한해 **뒤의 동일 단계 완료 기록의 입력 SHA·native 완료
증거·원본 receipt를 모두 검증**한 후 허용한다. 완료 행 누락, 현대 기록 뒤 누락,
잘못된 형식·실제 SHA 변경은 계속 거절하며 단계·파일·양쪽 SHA를 출력한다.
source journal은 수정하지 않고 실패한 시도의 산출물을 승인하지 않는다.

첨부 로그 `4410ca46-0410-4935-aeb9-28d8216e42cc/pasted-text.txt`의 실패 이유는
`selected_source_disappeared_after_resampling`이고 해당 invocation은 약 2459.997초 후
bank에서 중단됐다. raw XYZ `[512,512,683]`, target `[478,476,476]`, spacing은
raw XYZ `[0.705078125,0.705078125,0.699999988079071]`, target reader-grid
`[1.0,0.7578125,0.7578125]`이며 seg order 1 / order_z 0이었다.
이 로그는 OOM이나 affine 불일치가 아니다. 원래 위치의 독립 source mask가 0이라는
사실은 전체 segmentation에서 그 병변 정보가 없거나 모든 CP 위치에서 불가능하다는
증거가 아니다. CT/정답을 붙인 뒤 전처리하는 순서와 source를 먼저 전처리해 옮기는
순서는 일반적으로 교환되지 않는다.
이 invocation이 이후 Full/Basic을 시작한 증거는 없다. 이후 다른 프로세스의 실행 여부와
전체 bank·학습 완료 상태는 최신 journal/서버 증거 없이는 여전히 **미확인**이다.
여러 GPU 서버 이름이 과거 대화에 등장했으므로 현재 호스트와 할당 GPU를 추정하지 않는다.
과거의 `/Medical/HierCP`는 Git 저장소가 아니었으며 현재 checkout과 다르다.
로컬 push 완료는 서버 pull/설치/실행 완료가 아니다.

## 4. 모델과 두 GNN의 역할

활성 계약은 `full_v22`, `level0_physical_closure_v2`,
`hiercp_source_content_population_metric_v5`, `patient_source_content_population_v2`,
`source_content_observed_ct_population_v4`이다. 실제 설정은 `config/train.json`과
기존 실험의 `recovery/train_config.json`을 함께 확인한다.

- **L0 — local:** 종양 표면·내부, source/target 실질 context, 간 표면 anchor의
  이종 그래프와 3D patch encoder. 물리 거리와 full-shape geometry를 사용한다.
  context seed 384는 최종 전체 노드 수 cap이 아니다. interface와 지정 hop closure가
  유지되며 자원 제한 초과를 조용한 잘림으로 처리하지 않는다.
- **L1 — patient:** source tumor, 후보, 간 region, 다른 병변, whole-liver 관계.
  최종 region/lesion/liver 상태가 candidate-conditioned readout에 연결된다.
  원래 source region을 가리키는 privileged host edge는 사용하지 않는다.
  source address/raw-column/permutation/context-response 검사를 구분하며,
  실제 학습 모델의 shortcut 활용 여부나 실제 환자 누수까지 입증했다고 하지 않는다.
- **L2 — population:** training-only **간 context** prototype clustering과
  source/patient/local 조건부 readout이다. 암 종류를 클러스터링하는 분류기가 아니다.
  건강한 위치가 비슷해 보인다고 양성 정답을 만들어 넣지 않는다. 후보의 기하학적
  적합성과 context 호환성을 평가하는 경로이며 실제 개선 효과는 별도 검증 대상이다.
- **Quality GNN:** 사전에 학습하는 위치 호환성/ranking 모델. bank의 128개
  후보 위치와 quality score는 이후 feedback 학습에서 고정한다.
- **Difficulty GNN:** 실제 nnU-Net 학습에서 얻은 CP 관측 오차를 학습하는 별도 모델.
  L0/L1/L2 경로를 사용하되 quality 모델이나 bank의 호환성 점수를 갱신하지 않는다.

기준 규모: hidden 128, attention 4 heads, L0/L1/L2 blocks 3/2/2,
48³ patches, CNN channels 12/24/32, patient regions 24, prototypes 16,
ranking 후보 8, placement pool 128, quality GNN 40 epochs,
각 nnU-Net arm 250 epochs. PyG disjoint-union physical batching을 사용한다.
physical batch/worker는 측정 정책을 따르며 서버에서 선택된 실제 값을 추측하지 않는다.
Quality batch 후보는 `powers_of_two_to_cohort`, gradient accumulation은 1,
target effective batch는 `null`이다. 과거 문서의 고정 effective batch 2를 가져오지 않는다.
현재 single-device 경로만 지원하며 검증된 DDP 구현이 있다는 뜻은 아니다.

낮은 quality rank는 높은 segmentation difficulty와 동의어가 아니다.
현재 feedback은 nnU-Net의 정상 학습 forward에서 나온 최고 해상도 예측을
optimizer update 전에 관측한다. 변환된 pasted support에 대해 foreground CE,
boundary error, adjacent FP를 측정하고 epoch 경계에서 EMA와 별도 difficulty
GNN을 갱신한다. 정상 segmentation loss/backward를 대체하거나 추가 forward를 요구하지 않는다.
관측 불가는 0점 난이도가 아니다. crop 소실·잘림·불충분 관측 상태를 구분한다.
다음 epoch의 불변 snapshot은 측정/예측 난이도, exploration, easy retention을 사용한다.

Full은 quality gate와 난이도 기반 선택을 사용하고 Basic은 전체 유효 pool에서
균등 선택한다. event/source/appearance/augmentation schedule은 paired하게 유지한다.
quality gate와 curriculum의 효능·의학적 안전성은 입증된 사실이 아니다.
두 arm은 quality gate도 다르므로 Full/Basic 비교만으로 difficulty feedback의
독립 효과를 분리했다고 주장하지 않는다.
세부 정책과 임의 선택값의 근거는 `config/online_cp_feedback.json`에 명시되어 있다.

누수 경계: outer held-out 환자의 정답·예측·validation loss는 prototype fitting,
quality GNN fitting/선택, CP donor/recipient, difficulty feedback에 들어가면 안 된다.
quality GNN의 inner split은 outer train 내부에 있다. Difficulty GNN은 outer train의
실제 관측을 쓰므로 원래 inner validation을 새로운 독립 최종 평가로 주장하지 않는다.
서로 다른 case ID가 같은 환자일 수 있는 데이터라면 환자 단위 grouping을 별도 검증해야 한다.

## 5. Medical Data Aug와 전처리 공유 — 다시 혼동하지 말 것

사용자 제공 `Medical Data Aug.zip`은 원래 전처리/Basic-CP의 기준 자료다.
실험 output 폴더나 필요 없는 생성물로 취급하면 안 된다. 로컬 archive에는
batch script, notebook, `liver_70` 영상/정답이 있고 실제 `liver_76`은 없다.
환자 데이터가 들어 있으므로 Git과 인계 소스에 포함하지 않았다.

현재 복원된 공통 CP 조건은 source padding 2, native voxel center distance 12,
liver coverage 0.85, occupied clearance 2 voxels, physical center separation 0이다.
거리 feature 자체는 물리 mm를 유지한다. Hard paste와 HU jitter도 유지한다.
원본 batch의 uint8 mask 반전 버그를 되살리는 것은 기준 복원이 아니다.
128 후보 pool, 8 ranking 후보, seeded schedule, 50,000 proposal budget 및
exhaustive extension은 연구 확장이므로 원본 4,000회 sampler와 bitwise 동일하다고 하지 않는다.

Basic/Full은 같은 nnU-Net 전처리를 공유한다. 입력/split이 동일하게 검증된
region feature와 population prototype까지 CP 조건 변경만으로 전부 폐기하지 않는다.
반면 후보 mask/target이 바뀐 CP-dependent GNN graph와 source-mapping이 바뀐 bank는
이전 계약 그대로 재사용할 수 없다. 무엇이 공유 가능하고 무엇이 재생성 대상인지
artifact의 실제 의존성과 provenance로 판단한다. 모델 코드 전체가 무효라는 뜻도 아니다.
원본 script는 raw grid에 CT/HU jitter와 정답을 붙이고 원래 affine/header로 저장한다.
새 bank는 source를 원래 위치에서 한 번 resample해서 정수 이동시키지 않는다.
각 raw 후보 위치에서 붙여넣은 CT와 **전체 label mixture**가 native nnU-Net 전처리를
거친 결과를 표현한다. 다른 가까운 병변이나 전체 연결 성분을 donor로 대신하지 않는다.
HU jitter → CTNormalization의 percentile clip/mean/std → native interpolation 순서를
유지한다. native float32 출력은 유지하되 cubic 보간의 전역 spline tail과 clip을
보존하기 위한 내부 baseline/operator는 float64다.

새 계약은 `onlinecp_raw_target_paste_v1` /
`online_cp_raw_target_resampling_v2` /
`npz_candidate_refs_raw_target_v1`으로 기존 source-anchored bank와 구분한다.
서로 다른 raw 후보가 같은 preprocessed center로 반올림되어도 삭제하지 않는다.
원래 raw CP 조건과 전체 128개 후보/eligible-source slot을 유지한다.
실제 raw CP 후에도 native 신규 tumor support가 0일 수 있다. 이 경우에도 raw CP와
source/candidate draw는 유지하고, segmentation 학습을 수행하며 해당 feedback
관측만 불가로 기록한다. donor 제외나 실패 시 가짜 zero mask 반환이 아니다.
신규 support의 정의는 `(붙인 뒤 전체 native seg == 2) & (원래 native seg != 2)`다.

case baseline/operator는 NPY mmap, 선택 후보의 작은 payload는 pickle 없는 NPZ/JSON으로
저장한다. training crop에는 CT·전체 seg·support를 함께 적용한다. training event마다
전체 volume을 다시 전처리하지 않으며 cubic tail을 임의 ROI 경계에서 자르지 않는다.
공통 raw source CT/mask는 content-addressed NPY로 한 번 저장하며 128번 복제하지 않는다.
시작 시 전수 SHA 검증한 파일 witness는 bank/index identity에 묶어 worker에 전달한다.
각 worker가 모든 대용량 baseline을 다시 해시하지 않지만 파일 stat 변경 검사는 유지한다.
큰 병변이 기존 nnU-Net 학습 patch보다 커도 source/candidate를 버리거나 patch를 줄이지 않는다.
전체 후보 native support와 실제 crop에서 관측된 support를 따로 기록한다.
`[OnlineCPNativeTransport]` 로그와 feedback unavailable 상태를 함께 확인한다.
원래 crop 및 filled-nonzero mask를 보존하는 geometry 증명과 native baseline 대조가 필요하다.
지원되지 않는 crop 변경을 raw CP 불가능/no-placement로 재분류하거나 후보 삭제로 숨기지 않는다.
원본의 liver coverage 0.85를 1.0으로 강화하지 않는다.

`retain_original`은 **완전한 raw search에서 후보가 0임이 입증된 경우**에만 적용한다.
GNN에는 가짜 ranking sample을 만들지 않고 별도 no-placement 기록을 남긴다.
Online bank는 해당 source slot을 유지하고 그 slot이 뽑히면 CP를 적용하지 않는다.
다른 donor로 바꿔 확률을 재분배하지 않는다. 손상 데이터, 실패/미완 검색,
0보다 크지만 부족한 후보 pool은 이 정책으로 숨기지 않는다.
모든 시도의 resolved 상태를 기록했다는 것과 모두 GNN 학습 sample로 생성됐다는 것은 다르다.
실제 `liver_76`에서 후보가 생겼거나 전체 학습이 성공했다고 확인한 것은 아니다.
자세한 근거는 `docs/cp_input_repair.md`를 확인한다.

## 6. 현재 실행 경로와 최근 수정

`tools/run_feedback_experiment.py`의 recovery/continuation 경로:

```text
verified preparation + private nnU-Net runtime
  -> quality GNN training / causality evidence
  -> shared nnU-Net planning and preprocessing
  -> immutable online bank
  -> feedback contract + arm dry-run checks
  -> Full feedback nnU-Net
  -> Basic control nnU-Net
  -> optional --evaluate: checkpoint-bound prediction -> paired evaluation v5
```

검증된 완료 단계는 재사용하고, 미완료 단계는 계약이 허용할 때만 이어간다.
기본 실행은 학습에서 끝난다. `--evaluate`를 명시하면 별도 receipt/output으로
downstream prediction·평가·통계를 연결한다. 학습 journal identity는 바꾸지 않는다.
bank 단계 시작은 기존 GNN이나 전처리를 처음부터 지웠다는 뜻이 아니다.

`--upgrade-bank-from`은 호환되는 같은 모델의 bank 표현 변경 경로다.
아래 설명은 이 경로의 원칙이며 old v3/v4 모델을 현 v5로 승격하는 명령이 아니다.
검증된 기존 paired GNN/cache/prototype/causality는 원래 경로로 참조하고,
공통 raw/preprocessed 결과는 원본을 보존하는 새 data view로 재사용한다.
native unpack이 원본 preprocessed 폴더에 새 파일을 쓰지 않도록 새 root에서 수행한다.
새 private runtime, 새 raw-target bank, 새 Full/Basic 결과를 만들며 기존 nnU-Net
체크포인트를 새 CP 방식의 학습으로 재표기하지 않는다. source에서 segmentation 학습이
이미 시작됐거나 provenance가 맞지 않으면 이를 임의로 계속하지 않는다.
새 bank baseline은 대략 native voxel당 10 bytes에 operator/후보 저장 공간이 추가된다.
디스크 예산과 기존 reserve를 확인하며, 이는 RAM peak 또는 실제 처리시간 측정값이 아니다.

최근 Git 변경의 의미:

- `c8edb72`: paired GNN의 causality 호출에 실제 CLI가 요구하는 prototype bank/run mode 전달.
- `28d29c8`: causality checkpoint 비교 시 ct_clip container 표현 정규화.
- `065c8a5`, `76bc3ee`: 빈 relation 처리 및 full-edge causality의 메모리/진단 경로 수정.
- `7848306`: bank reader가 host-bounded causality v2 receipt를 검증하도록 수정.
- `ebe58ff`: online bank 병목 수정. Standard bank의 버려지는 중복 full-target
  feature/edge 생성을 없애고, source 준비를 공유하며 measured ordered CPU graph
  preparation을 사용한다. Standard/argmax pending scoring은 disk spool과
  physical-batch mmap loading을 사용한다. Argmax에 원래 같은 중복 생성이 있었다는 뜻은 아니다.
  checkpoint의 불필요한 optimizer/CPU 상태와 이전 case/source 참조를 해제하는 경로도 포함한다.
  모델·후보·그래프·CP 조건은 축소하지 않았다. 저장장치 I/O와 단일 full-pool 메모리 비용은 남는다.
- `da2fdd8`: 기존 실행/복구 문서와 bank 성능 설명 갱신.
- `8e36228`: 오래된 `code.txt`를 당시 tracked 파일 145개 전체로 갱신.

Bank 진행 증거는 실제 bank root 아래
`preparation_progress.<uuid>.jsonl`,
`preparation_resources/candidate_graphs.<case>.<component>.<uuid>.resources.json`,
동반 summary 및 완료된 `index.json.scoring_execution`에서 확인한다.
첫 calibration은 충분한 source sample이 모일 때까지 시간이 들 수 있다.
heartbeat/`preparation_finished` 이벤트는 최종 bank acceptance 증거를 대체하지 않는다.
새 raw-target 단계는 `[RawTargetCase]`, `[RawTargetSource]`, `[RawTargetResources]` 및
`preparation_resources/raw_targets.*.json`에 baseline 일치, 전체 후보 support 수,
zero-support 수와 measured CPU 준비 자원을 구분해 기록한다.

표준 bank envelope만으로 legacy reader 호환성을 판단하지 않는다.
`paste_contract`, `source_mapping_format`, `entry_storage`와 엔진/파일 SHA를 함께 검증한다.
기존 exact-argmax/ArgmaxV3/rank-only curriculum/downstream ablation은 이 새 typed bank를
학습/재채점 전에 거절한다. 새 실행은 Full/Basic feedback runner를 사용한다.

## 7. 검증 범위와 과거 성능 결과

아래 v4 검사는 2026-09-12의 과거 기록이다. 현재 v5/Basic 재사용 검증 상태는 이 문서
첫 부분과 구분한다. 어떤 CPU 회귀 결과도 실제 의료 데이터 전체 파이프라인 검증이나
임상 효과 입증으로 표현하지 않으며, 서로 중복된 검사 수를 합산하지 않는다.

- 2026-09-12 source-content v4 최종 working tree: 전체
  `python -B -m unittest discover -s tests -p 'test*.py' -q`
  **785개 중 780개 통과·5개 skip·실패/오류 0**, 234.000초, 종료 코드 0.
  skip은 Windows symlink 권한 제약 4개와 별도 opt-in 대규모 production smoke 1개다.
  이 전체 회귀 후 문서 exporter의 EOF 빈 줄만 정리했고, 해당 exporter DEBUG 5개를
  다시 실행해 모두 통과했다. 이는 당시 기록이며 이후 v5/Basic 재사용 변경은
  이 과거 검사에 포함되지 않는다.
  모델/causality 21개, 전처리 공유 14개, feedback graph/resources 25개,
  발행/평가 연결 42개, native trainer 연결 23개, 실행/재개 58개,
  실행 잠금/인자 bridge 15개와 export 5개 집중 검사는 이 전체 검사와 중복되므로
  독립 표본이나 별도 전체 실행으로 합산하지 않는다.
- Python **153개**의 Python 3.10 AST, JSON **5개**의 중복 key/비표준 숫자 검사,
  trainer **10개**의 실제 SHA와 installer/SHA256SUMS 일치, `git diff --check` 통과.
  로컬 CPU 16 logical cores, RAM 18,918,256,640 bytes, 마지막 정적 검사 시 available
  2,995,515,392 bytes를 확인했다. PyTorch 2.6.0+cpu / PyG 2.6.1 환경이며 CUDA는 없다.
  `nnUNet_n_proc_DA=2`와 임시 matplotlib cache는 로컬 DEBUG 프로세스에만 적용했다.
- 실제 소형 NPZ/B2ND/NIfTI I/O, native Python child 종료 증거, torch 모델/SGD/Adam/RNG
  상태 검증, 중단·재시도·원본 보존, full128 logit/gradient 비교를 실행했다.
  평가 producer 테스트의 `cohort=2`, `parameters=3`은 명시적인 DEBUG predictor 대역이다.
  실제 전체 nnU-Net forward/학습을 했다는 뜻이 아니며 production 모델 설정은 그대로다.
- 실제 `feedback_medical_aug` source 자격 preflight, 전체 의료 cohort bank 생성,
  40-epoch quality GNN, 250-epoch Full/Basic nnU-Net, native 전체 epoch/checkpoint 재개,
  GPU 자원/처리량, NFS 잠금 동작, 실제 예측/전체 paired 평가·통계는 **미실행/미검증**이다.
  실제 서버의 보존된 source 설정·native 산출물·원본 SHA·실행 상태가 맞는지는 서버에서
  검사해야 한다. CPU DEBUG 성공을 학습 성능 향상·모든 shortcut 차단으로 확대하지 않는다.

다음 날짜별 기록은 변경 전 revision의 역사적 결과다.

- 2026-09-12 디스크 재개 수정: 전체 `python -B -m unittest discover -s tests -p 'test*.py' -q`
  **680개 중 677개 통과, 3개 건너뜀, 실패 0**, 231.251초, 종료 코드 0.
  새 helper/기존 wiring 범위 12개도 별도로 모두 통과했다(30.817초; 전체 suite의 부분집합).
  실제 native 저장 전 disk guard와 BankProgress/Measurement로 만든 작은 CPU DEBUG 실패 기록을
  사용해 CSV/config/실패 증거 보존, 완료 case 재사용, 실패 case의 128개 payload 생성 및
  native bank audit까지 검사했다. 잘못된 행/계약/중복·누락 증거/불완전 측정/부분 파일/
  일반 오류/여전한 공간 부족은 거절한다. GNN scoring/외부 실행의 명시적 DEBUG 경계 대역은
  서버 실행 검증을 대신하지 않는다. Python 3.10 문법 검사 136개와 `git diff --check` 통과.
  `hiercp`, `custom_trainers`, `config`의 기존 파일 변경 없음도 검사했다.
  로컬은 CPU 16 logical cores, RAM 약 17.62GiB/검사 전 available 약 2.25GiB,
  CUDA 없음이었다. 서버의 실제 CSV/측정 파일에 대한 admission, 전체 bank 완료, 학습·평가는
  실행하지 않았다. 사용자가 준 서버 로그/df/du와 로컬 DEBUG 검증을 구분해야 한다.
- 2026-09-10 source-history 수정(과거): upgrade DEBUG 26개 중 25개 통과·Windows symlink 권한
  1개 skip(8.640초), 기존 실행·재개 DEBUG 32개 모두 통과(39.098초).
  합계 **58개 중 57개 통과, 1개 skip, 실패 0**. 새 회귀 테스트 9개는 모두 통과했다.
  동일한 DEBUG legacy 기록을 이전 `a406574` reader에 넣으면 사용자와 같은 오류가
  발생하고, 수정 reader는 후속 동일 단계 증거 검증 후 통과함을 별도 대조했다.
  원본 journal·파일 내용/목록 보존 및 GNN·전처리 재실행 없는 stage driver를 검사했다.
  무거운 native 성공 검증에는 명시적 경계 double을 사용하며 실제 서버 journal이나
  환자 데이터 검증을 대신하지 않는다. 당시에는 아래 전체 662개 suite나 전체 학습·평가를
  재실행하지 않았다. 아래 662개 수치는 직전 raw-target 구현의 과거 결과다.
- raw-target 구현(`945f7a2`)의 로컬 전체 회귀: `python -B -m unittest discover -s tests -p 'test*.py' -v`
  **662개 중 659개 통과, 3개 건너뜀, 실패 0**, 267.604초, 종료 코드 0.
  건너뛴 것은 Windows symlink 권한 관련 2개와 명시적 opt-in이 필요한
  production-sized DEBUG optimizer-step smoke 1개다. 그 전체 규모 smoke를 실행한 것으로
  주장하지 않는다. 아래 범위별 수치는 이 suite의 부분집합이므로 합산하지 않는다.
- 이번 raw 엔진 CPU DEBUG 14개 통과. native nnU-Net CT normalization/crop/resampling
  대조 16건에서 CT 최대 절대 오차 0, 전체 label/support 정확 일치.
  원래 위치에서 native 소실되는 raw 1-voxel donor가 다른 target 위치에서는 생존하는
  경우, 전역 cubic tail/clip, transpose/crop, separate-z 3축, 0.85 coverage를 포함한다.
- storage/준비 helper CPU DEBUG 10개 통과. 실제 native baseline 대조, 후보별 0/양성 support,
  128개 후보 유지·공통 donor 2개 NPY만 저장, 변조 거절, mmap/spawn 상태를 확인했다.
  작은 인공 DEBUG 입력이며 `liver_101` 원본 영상이나 전체 의료 cohort를 실행한 결과가 아니다.
- trainer CPU DEBUG 16개 통과(경계 검사 13개, native 통합 3개).
  실제 128개 payload 저장/전수 감사, lazy loader, 같은 5개 RNG draw, native crop,
  feedback 변환과 segmentation CE backward까지 연결했다. 원본 위치에서 소실되는 작은
  donor의 target-phase 생존 차이, 큰 source의 고정 patch 부분 crop, 검증 witness 전달 후
  대용량 파일 재해시 0회와 stat 변조 거절도 검사했다. 최종 nnU-Net 전체 학습을 뜻하지 않는다.
- 구형/신형 bank 소비 경계 DEBUG 7개 통과. legacy pair/all/ArgmaxV3/ablation/rank-only
  경로가 새 bank를 기존 결과 변경 전에 거절하고, feedback은 실제 검증 경로를 유지한다.
- bank typed-schema/wiring/no-placement/준비 helper 범위 17개 통과.
  index의 raw/legacy 선언과 엔트리 형식의 혼합을 양방향 거절하며, 실제 raw helper로
  만든 128개 후보와 zero-placement 환자 다음 환자의 정상 처리를 확인했다.
- upgrade/기존 runner DEBUG 49개 중 48개 통과, Windows 심볼릭 링크 권한 제한으로
  디렉터리-link 검사 1개 건너뜀. native 경로의 원본 GNN 절대경로, 새 raw/preprocessed
  view의 marker/cohort/content SHA 계약도 소스로 대조했다. 서버 실자료 검증을 대체하지 않는다.
- 10개 trainer/helper SHA 검사 및 별도 임시 native nnU-Net 복사본의 실제 설치/import와
  legacy policy/paste smoke 통과. 기존 site-packages/실험 runtime은 수정하지 않았다.
  Python 파일 134개의 Python 3.10 AST 검사와 Git staged 원본 bytes 기준 10개 모듈 SHA
  일치도 확인했다. 로컬 GPU는 없으며 서버 GPU 성능은 측정하지 않았다.

- `ebe58ff`의 로컬 보고: Python 3.10 문법 검사 및 594 tests 중
  592 passed, 2 skipped. 환자 전체 학습이 아니라 DEBUG/단위 회귀 검사다.
- 작은 CPU DEBUG 비교: 동일 후보 graph 3개, 3회 반복에서 standard 경로의
  full-target edge 생성 호출이 6회에서 3회로 줄고 tensor/edge/patch가 일치했다.
  서버 GPU 속도 향상률이나 전체 bank OOM 해결을 입증한 측정은 아니다.
- 직전 `d861233`의 `code.txt`는 `13649142` 기준 텍스트 146개를 담았다.
  이번 갱신의 포함 파일 수와 기준 commit은 새 snapshot 머리말을 따른다.
- 현재 수정 경로의 real-data bank 완료, 전체 40/250-epoch 학습,
  full downstream 평가, multi-fold 효능은 이 로컬 검증으로 확인되지 않았다.

과거 사용자 제공 **fold-0 exact-argmax ablation**에서는 tumor Dice가
Basic 0.6454, Full M3 0.6722, w/o L1 0.6239, w/o L2 0.6977이었다.
Full minus w/o L2의 Dice 차이는 -0.0255, 95% CI [-0.0767, +0.0062], p=0.3241로
보고됐다. L2의 이득이 입증되지 않았다는 탐색적 결과이지, 현재 feedback 모델에서도
L2가 무효라는 결론이나 자동 제거 승인은 아니다.
별도 과거 `evaluation/summary.json`의 HierCP Dice 0.659247과도 같은 결과로 합치지 않는다.
새 feedback 결과로 재표기하거나 서로 다른 detection threshold/평가 경로를 혼합하지 않는다.
이 숫자는 대화에 제시된 과거 출력이며 원본 예측/체크포인트를 현재 재검증한 것이 아니다.

## 8. 다음 작업과 서버 재개 조건

다음 단계는 먼저 최신 서버 로그와 단계 증거로 실제 상태를 판별하는 것이다.
학습 진행 중이면 같은 checkout을 pull하거나 trainer를 교체하거나 중복 실행하지 않는다.
검증된 bank/GNN/preprocessing은 보존한다. 오류가 있으면 해당 실패 경로를 고치고
명시적인 검증 후 이어가며, journal/lock/manifest를 지워 통과시키지 않는다.

아래는 기존 작업이 실행 중이지 않음을 확인한 뒤, **원래 native 전처리가 완료된
`feedback_medical_aug`에서 전처리만 공유하는 새 v5 실험**의 명령 형식이다.
bank-upgrade 파생 root인 `feedback_rawcp`는 이 전처리 원본을 대체하지 않는다.
현재 검증 상태는 문서 첫 부분을 따른다. 아래 명령은 서버 checkout을 `git pull --ff-only`로
갱신한 뒤 실행하며, 실제 source 자격과 현재 디스크/GPU 할당은 첫 preflight에서 확인한다.
기존 GNN·bank·segmentation checkpoint를 새 버전으로 재표기하거나 이어 학습하지 않는다.
실행 중인 프로세스를 중단하라는 지시가 아니다.
아래 GPU 5번은 2026-09-15 사용자 할당에 따른 것이다. 할당이 바뀌면 해당 번호도 바꿔야 한다.
프로세스에는 이 GPU 한 개만 보이며 내부 `cuda:0`은 물리 GPU 5번을 가리킨다.

```bash
conda activate /home/aicompetition06/.conda/envs/nnunet &&
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
/home/aicompetition06/.conda/envs/nnunet/bin/python -B tools/run_feedback_experiment.py \
  --reuse-preprocessing-from work/feedback_medical_aug \
  --experiment-name feedback_population_v5 \
  --medical-root /home/aicompetition06/Medical \
  --outer-fold 0 \
  --dataset-id 760 \
  --seed 42 \
  --evaluate
```

완료된 raw-target Basic 원본이 별도로 확인된 경우에만
`--reuse-basic-from work/<완료_Basic_실험>`을 추가할 수 있다. 이 옵션은 원래 checkpoint와
전체 CP/native 입력·학습 이력 검증을 요구하며, 현재 서버의 어떤 Basic도 이 문서만으로
재사용 승인된 것은 아니다. 구형 legacy CP 결과는 자동으로 이 조건을 만족하지 않는다.

새 v5 root의 지원되는 중단 경계에서는 위와 **동일한 인자에 `--resume-experiment`만
추가**한다. 미완료나 출처 불명 파일을 자동 승인하는 옵션이 아니며, 기존 root를 삭제해서
새로 시작하지 않는다. 준비 단계 또는 native 종료 증거가 불명확하면 보존 후 거절한다.
공유되는 packed 전처리는 hard link이며, 메타데이터와 새 unpack 배열은 분리된다.
파일시스템 강제 read-only 복사본은 아니므로 공유 inode를 수작업 변경하면 양쪽에 영향을 준다.
지원되는 실행 경로는 공유 payload를 읽기만 하며 덮어쓰기·재전처리를 수행하지 않는다.
원본 `feedback_medical_aug` journal은 수정하지 않는다. 기존 `--recover-from` 기반
실험의 `--resume-preparation`/`--resume-experiment` 경로도 남아 있지만 새 bank 표현으로
구버전 runtime을 바꾸는 수단이 아니다. `--overwrite`나 journal 수작업 삭제로 우회하지 않는다.
저장된 Python 경로도 plan identity에 포함된다. 과거 `(base)` Python으로 검사했을 때는
checksum/source가 맞아도 Python 경로만 달라 plan mismatch가 발생했다.
이것을 journal checksum 완화로 고치지 않는다.
검증된 저장 전 디스크 실패 이외의 bank error row, 불충분 pool, 미완 runtime, running row/lock,
호환되지 않거나 없는 필수 checkpoint는 자동 재개가 거부될 수 있다.
최신 오류를 보지 않고 무조건 재실행 명령을 반복해서 주지 않는다.

Full/Basic 완료 후에는 해당 새 실험의 prediction·같은 정의의 paired evaluation과
검증된 결과 provenance가 필요하다. 추가 fold나 L2 재설계 여부는 사용자의 다음
실험 선택과 실제 결과에 따라 결정하며 이 인계로 자동 실행 범위를 넓히지 않는다.

## 작업 완료 체크리스트

아래는 다음 수정·실행 담당자가 실제 증거로 확인할 체크리스트다.
이번 문서 작성만으로 전체 연구 검증 항목을 완료 처리하지 않는다.

- [ ] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [ ] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [ ] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [ ] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [ ] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [ ] 디버그 설정과 최종 설정을 분리했다.
- [ ] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [ ] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [ ] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.


## 작업 완료 체크리스트


- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 문제에 그래프 축소 없이 기존 tiling/checkpointing 경로를 유지했다. 새 DEBUG에서 OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
