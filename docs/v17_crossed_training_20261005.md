# v1.7 교차 C/D: 빠진 전체 전환 블록의 실제 비교

이 설계는 기존 v1, A, B 및 native 결과를 보존한다. 새 C/D는 성공한 B와 실패한 native가 공유하는 native 상위 연산을 고정하고, 그 사이에 남은 입력/L0와 과제/학습의 변경을 교차한다. 기존 `v17_complete_transition_design.json`의 11/18 설계를 수정하거나 완료한 것으로 표현하지 않는다. 새 계약은 `config/v17_crossed_training.json`이다.

서버 수집은 native recipe/source121개의 일치와 누락0을 보고했다. 선택된 native checkpoint는 `v22_cnn_m10_seed42/attempts/0001/checkpoint_latest.pt`이고, 저장 상태는 epoch39/step21320/next_batch533/refresh_memory다. 40번째 optimization은 끝났지만 마지막 support refresh, validation, BEST 선택과 final memory는 완료되지 않았다. 실제 checkpoint SHA는 서버 receipt에서 읽어 실행 manifest에 결속한다. 이 문서에 없는 SHA를 만들지 않는다.

## 의존성을 닫은 두 블록

| 새 arm | 입력/L0 블록 | 과제/학습 블록 |
|---|---|---|
| C | native raw crop, 한 CT 채널, erasure 없음, CNN/organ mean/fused128, 한 view | 원래 source-anchor GT, 선별8후보 curriculum ranking loss, AMP/cosine, full own-task support |
| D | 실제 원본 v1 L0, erasure, resampled48/five channels, six roles/three GAT, genuine 두 view 평균 및 consistency0.1 | 실제 native 독립 donor, 모든 P+128U live loss, FP32/constant LR, support16 episode |

두 arm의 L1/L2/scalar scoring은 B와 같은 native upper다. CPU seed42 fork에서 같은 초기 upper tensor bytes를 만들고 optimizer update 전에 digest를 대조한다. 새 학습은 seed42, 10mm, 40epoch, 전체 원래 cohort와 모델 크기를 유지한다. 기존 학습 checkpoint를 새 초기 가중치인 것처럼 사용하지 않는다.

comparison corruption과 two-view consistency는 입력/L0 블록에 포함한다. C의 native raw crop에는 원래 relation/geometry corruption이 적용되지 않으며, 여섯 semantic 출력이나 consistency를 만들어낸 것처럼 보고하지 않는다. C의 source와 anchor index0 및 원래 선별 비교 후보의 GT는 유지한다. D의 P/U observation에는 curriculum corruption을 추가하지 않는다. D consistency는 실제 원본 여섯 semantic 출력으로 계산하며 input-owned 0.1 항목으로 native loss에 더한다. fused vector 복제나 loss0 우회는 금지한다.

corruption은 task가 제공하는 candidate datum과 결합되어 있어 C/D에서 모두 off다. 따라서 전체29축을 장부에 빠짐없이 포함한 것과29개의 독립 binary contrast를 구현한 것은 다르다. 이 coupling을 명시하며, 원래 two-view augmentation을 curriculum CandidateSpec corruption이라고 바꾸어 부르지 않는다. factor inventory coverage, 실제 source/forward 적용 증거, 독립적 원인 분리 가능성은 서로 다른 항목이다.

activation storage는 과제/실행 블록을 따른다. C의 native CNN은 checkpointing=True, 공통 native upper는 checkpoint_support=True로 원래 checkpointed 실행 정책을 사용한다. D는 실제 원본 L0의 config copy에서 checkpoint_dense_encoder=False와 checkpoint_local_blocks=False를 명시하고, 공통 native upper의 checkpoint_support=False도 명시하여 native retained 정책을 적용한다. 원본 L0와 native upper를 결합하는 dependency bridge와 실행 storage는 별도로 기록한다. 모델 깊이·너비·graph/readout/parameter는 바꾸지 않는다. 이 정책은 실행 manifest와 자원 측정에 기록하며, D32의 비용이 커져도 숨겨진 checkpointing이나 모델 축소로 바꾸지 않는다.

## 전체29 변경축 장부

아래 세 묶음은 각 factor를 정확히 한 번 포함한다. 실행 recipe는 config 장부와 대조하여 누락/중복을 거부한다.

- 공통 upper3: `upper_L1`, `upper_L2`, `scalar_score`.
- 입력/L0 10: `physical_input_sampling`, `input_channels`, `recipient_tumor_erasure`, `role_shell_geometry`, `local_encoder`, `local_message_passing`, `local_readout`, `pair_fusion`, `comparison_corruption`, `two_view_consistency`.
- 과제/학습/평가16: `donor_condition`, `GT_semantics`, `training_sample_population`, `training_candidate_set`, `ranking_loss`, `observation_alignment_auxiliaries`, `support_labels_population`, `support_refresh`, `support_episode_selection`, `update_schedule`, `optimizer_LR_schedule`, `precision_and_scaler`, `physical_batch_unit_and_work`, `activation_storage`, `evaluation_candidate_universe`, `checkpoint_selection`.

상위3축을 공통으로 고정한 이유는 B가 실제 원래 과제에서 그 연산을 연결하여 학습한 증거가 있기 때문이다. 이것이 다른 과제와의 interaction까지 증명하는 것은 아니다. 두 새 arm이 모두 학습해도 native endpoint 실패와의 상호작용이 남을 수 있다.

## own-task 평가와 공통 전체128 평가

C의 own-task BEST/성능은 고정 전체 난이도의 원래8후보 평가이며, D의 own-task BEST/성능은 native P/U 평가다. 서로 다른 own-task MRR 숫자의 차이로 더 좋은 모델이나 고장 블록을 결정하지 않는다.

C의 configured split은 원래84/21을 유지한다. 실제 candidate가 materialize된 case와 sample은 baseline preflight의 signed train_cache_files/val_cache_files와 결속된 cache index에서 산출한다. no-placement 사례가 있는 configured split 전체를 실제 cache case 수와 동일하다고 강제하지 않으며, 모든 실제 signed sample을 빠짐없이 사용한 증거와 configured-but-not-materialized 목록을 기록한다. 이 경계는 모든 native P/128U case를 사용하는 공통 평가의21case 계약과 별개다.

공통 benchmark는 같은 signed native inventory, 같은21 held-out case, 같은 seed42 독립 donor assignment를 사용한다. 128은 U 개수다. 실제 validation은 P135+U2688=2823 observations이며 각 case의 모든 P와128U를 한 번씩 score한다. zero-P case도 읽고 score하며, rank 분모에서만 제외하고 개수를 명시한다. outer26은 학습과 원인 비교에서 사용하지 않는다.

공통 support는 native train P527+U10752=11279 observations를 각 arm의 실제 학습된 encoder로 새로 encode하여 만든다. eval/no_grad의 detached full bank이며, 두 arm 모두 같은 native 관측 class를 사용한다. C own-task anchor/curriculum bank를 그대로 가져와 native P/U bank라고 부르면 안 된다. 각 support row의 recipient owner 또는 donor owner가 query recipient patient와 같으면 그 row를 제외한다. query의 donor patient를 별도로 모두 제외하는 규칙은 아니다. 이 조건을 통과한 full support를 case당 한 번 준비하며 validation에 support16 subset을 사용하지 않는다.

평가기는 L0만 명시적 측정 batch로 chunk하고, case 전체 embeddings를 합친 뒤 upper callback을 한 번 호출한다. 독립8후보 upper 결과를 합쳐 full128이라고 표시하지 않는다. callback에는 observation target/component가 들어가지 않는다. 원본 D graph의 erasure를 위한 준비 주석과 query GT를 prompt/label edge로 주입하는 행위는 구분해야 한다.

`case_first_P_mrr`, `case_hit_at_1`, `observed_micro_recall_at_1/5/10`을 각각 출력한다. microR1의 분모는 모든 P이며 P135라면21case가 전부 평가 가능해도 최대21/135다. native `.007407`은1/135이며 v1 sample top1과 다르다. P/U pair win/tie/softplus loss, 실제 score와 score 표준편차, 평가 가능 case와 zero-P 분모도 저장한다. 동점은 `score_desc_geometry_sha256_v1`의 GT-independent order를 따른다. 이는 원본 v1의 pessimistic tie 정책과 다르므로 이름과 규칙을 보존한다. P는 donor 적합도 GT가 아니고 U는 CP 부적합 GT가 아니다.

## 실제 실행 및 자원 경계

새 runner는 `tools/run_v17_crossed_training.py --arm C|D --baseline ... --native-run ... --output ... --gpu ... --workers ... --physical-batch-candidates ... --cuda-gib ... --rss-gib ... --resident-gib ...`이다. 새 server shell의 기본 arm은 D, GPU3, workers16, batch 후보32, CUDA40GiB/RSS192GiB/resident128GiB다. C의 후보는1/2/4 curriculum samples이며 sample당 query8개를 별도로 기록한다. 기준 실험에서 측정된1sample은 임의 축소 기본값과 다르다. D 후보32는 observation rows이고 두 원본 graph view의 실제 작업량을 따로 기록한다.

최종 physical batch는 명시된 후보의 실제 전체 shape CUDA clone calibration 3회로 확인한다. 같은 숫자가 같은 candidate/view 작업량을 뜻하지 않는다. D 후보32가 실패하면 자원 측정과 원인을 기록하고 중단한다. 더 작은 후보를 몰래 넣거나 L0/GT/profile/data를 바꾸지 않는다. CPU/RAM/VRAM과 loader/cache 비용은 새 arm에서 측정해야 하며, 기존 native32를 새 D의 비용 검증으로 보고하지 않는다.

서버에서는 검토한 새 checkout에서 `bash tools/server_v17_crossed.sh`를 실행한다. script는 실제 checkout의 `git rev-parse HEAD`를 읽으며, 명시한 `CP_CODE_COMMIT`과 다르면 학습 전에 오류를 낸다. 최종 server command에는 검토 후 게시된 정확한 commit을 명시한다. commit 확인과 실제 dependency source-byte identity 검사는 모두 필요하다. 실행 package와 그 package가 import하는 tools를 AST로 재귀 수집하여 새 timing/reference helper와 tools initializer까지 결속하며, 무관한 visualization renderer를 실행 의존성으로 포함하지 않는다. 알려진 baseline/native 경로를 사용하며 ZIP 전달이나 수동 업로드를 요구하지 않는다.

최종 명령은 arm별 고정된 새 `CP_OUTPUT`을 명시한다. 같은 출력과 같은 arm/data/source/resource 설정으로 다시 실행하면 자신의 checkpoint를 정확히 resume하며, 결속된40epoch 완료 marker가 있으면 재학습하지 않는다. `CP_OUTPUT`을 생략한 shell 기본값은 timestamp와 helper PID를 포함하는 새 experiment 경로를 만들므로 resume 명령으로 쓰지 않는다. `CP_ARM=C` 또는 `CP_ARM=D`는 사용자 선택이며 다른 arm이나 기존 A/B 재학습을 자동 시작하지 않는다.

D의 `--prepared-cache`는 이미 완성된 canonical `index.json`의 읽기 전용 재사용이다. inventory SHA, 모든 native observation/donor/GT assignment, 원본 scope/helper identity/base 및 sampled graph 측정 결속을 검사한다. 생산 cache는 전체14102 native observations를 포함해야 한다. 사용되는 graph와 shared donor source의 SHA도 검증하며 누락된 row를 다시 작은 subset으로 만들거나 건너뛰지 않는다. 이 옵션은 C에서 거부한다. 로컬 DEBUG의537 observations cache를 production cache로 사용할 수 없다. cache 재사용은 그래프 준비를 생략하는 것이며 새 L0의 full detached support encoding, 자원 calibration 또는 학습을 생략하는 것이 아니다.

native 비교 reference는 선택된 실제 attempt의 checkpoint를 CPU FakeTensorMode로 metadata만 읽는다. cache/config/base/local-CNN/learning-policy/source와 요청·execution·schedule·실제 cursor를 결속하고 필수 metadata 누락을 거부한다. 파일 SHA는 계산하지만 내부 tensor content hash를 재계산하거나 가중치를 새 C/D에 전달하지 않는다. 이는 기록된 reference 계약 admission이며 native 모델의 성능 재검증이 아니다. epoch39/refresh_memory/step21320은 optimizer40 이후의 미완료 reference로 기록한다.

DEBUG는 `--debug`와 별도 출력/profile에서만 허용한다. `--debug-updates`는 DEBUG에서만 허용하며 실제 CT mechanical 검사다. production40이나 전체128 평가 결과를 DEBUG 성공으로 대신하지 않는다.

## epoch 시간과 재시작

C/D epoch wall은 연속 monotonic 경과 시간으로 계산한다. loader/setup, 실제 optimizer update, support refresh, own/common validation 및 epoch 종료 checkpoint publication을 포함한다. async checkpoint와 GPU 작업의 phase 시간은 겹칠 수 있으므로 phase 시간을 더해 epoch wall을 만들지 않는다. D는 다음 epoch의 reset cursor를 저장하는 snapshot/hash/write/wait까지 기존 finished clock이 계속 측정한 후 curve의 epoch wall을 확정한다. C는 own-task와 공통 평가 및 closing checkpoint를 같은 epoch 시간에 포함한다.

재시작 시 이전 프로세스가 중단된 downtime은 더하지 않는다. C의 checkpoint timing receipt와 D의 checkpoint ID·identity·epoch/step/phase에 결속된 write journal로 마지막 atomic checkpoint write tail을 복원한다. 중단으로 완전한 receipt가 없으면 저장된 pre-write elapsed를 lower bound로 유지하고 불확실성을 명시한다. initial preparation/calibration/initial validation, 마지막 selected-model benchmark 및 프로세스 중단 시간은 epoch wall 밖의 별도 범위다. 전체21case production 시간·처리량은 서버 측정 전이며 로컬 DEBUG 시간을 대신 제출하지 않는다.

D는 다음 epoch reset cursor와 BEST를 먼저 durable checkpoint로 저장한 뒤 epoch curve/timing 행을 append한다. 이 둘 사이에서 프로세스가 중단되면 저장된 cursor와 BEST는 보존되지만 방금 완료한 epoch의 curve/timing 행이 없을 수 있다. 이 행을 임의로 재구성하거나 학습을 다시 실행하지 않는다. 따라서 학습 상태의 재시작과 모든 완료 epoch의 telemetry 행 보존은 별개의 계약이며, 이 crash window는 남아 있는 기록상의 제한이다.

## 현재 완료 범위

상태는 `IMPLEMENTED_LOCAL_CUDA_VERIFIED_SERVER_PREFLIGHT_REQUIRED`다. C/D 모델·학습·checkpoint/resume·공통 평가 entry와 전체29 factor 장부, coupling 및 source 계약을 구현했다. 최종 코드를 사용한 C_r4/D_r3의 실제 CT/CUDA DEBUG와 각각의 own-resume 검사를 완료했다. 기존 v1/A/B/native 가중치와 결과는 보존했다. CP나 nnU-Net 학습은 시작하지 않았다.

정적 검사에서는 새 코드22개 파일의 AST 검사와 서버 shell의 LF/문법 검사가 통과했다. 단위 검사는 transition98개와 evidence collector15개, 중복을 제외한 총113개가 통과했다. 단위 검사나 fixture metadata 검사를 실제 CT 정확도 결과로 부르지 않는다.

실제 CUDA에는 RTX5070Ti(15.92GiB), 한 GPU, CPU16 logical cores를 사용했다. DEBUG 자원 한도는 CUDA12GiB/RSS32GiB/resident16GiB, workers2였다. 모델 깊이·너비는 원래 각 블록을 유지했다. DEBUG의 cohort와 update 수만 별도 설정으로 제한했으며 production 계약은40epoch/전체 데이터다.

| 최종 실제 CT/CUDA DEBUG | C_r4 | D_r3 |
|---|---:|---:|
| 실제 trainable parameters | 1,125,718 | 6,433,126 |
| 성공한 optimizer updates | 2 | 2 |
| gradient가 전달된 trainable parameter tensors | 99/99 | 677/677 |
| missing-gradient 목록 | 없음 | 없음 |
| 실제 update의 physical batch 단위 | 최대2 curriculum samples=16 후보, 마지막1sample=8 후보 | 2 observations=4 genuine sampled-view graphs |
| core weight 변경 | CNN/readout/fusion/L1/L2/L2 updates 확인 | L0/L1/L2/L2 updates 확인 |

C의 AMP에서 발생한 두 overflow attempt는 optimizer update로 세지 않았다. 성공 update는2회이며 전체 backward attempts는4회다. D는 실제 원본 여섯 semantic 출력과 두 graph view를 사용했다. C에 가짜 여섯 출력을 만들지 않았다.

D의 원본 그래프 준비는 DEBUG537 observations/1074 sampled views 전체를 완료했으며 skip/donor redraw는0이다. 재사용 가능한 canonical 준비는1842.35초(30분42초), 최대 기록 RSS12.87GiB였다. 두 view를 합친 observation당 노드 수는3,086–10,734(평균5,736.39), edge 수는236,122–899,596(평균446,363.09)다. 압축 graph payload는338,718,431bytes, canonical 누적 디스크 기록은341,444,422bytes다. 이는 실제 준비 비용이며 서버 전체14,102 observations 준비 시간이나 이후 epoch 속도를 보장하지 않는다. D_r3는 이 완료 cache를 읽기 전용으로 재사용했다.

두 arm 모두 각자의 현재 L0로 native train401-row bank를 새로 만들고, 별도 CT `liver_31`의 모든136 observations(P8+128U)를 score했다. case embeddings를 모두 모은 뒤 upper를 한 번 호출했다. C의 own-task8후보 support를 가져오거나8후보 upper 결과를 붙여 전체128이라고 표시하지 않았다. 원래 원인 비교의 query GT와 P/U 의미도 유지했다.

| 마지막 공통 DEBUG 평가: P8+128U,1case | C_r4 | D_r3 |
|---|---:|---:|
| case-first-P MRR | 0.166667 | 0.333333 |
| case hit@1 | 0 | 0 |
| observed micro R@5 / R@10 | 0 / 0.25 | 0.125 / 0.125 |
| P/U pair win rate | 0.570313 | 0.436523 |
| P/U softplus loss | 0.691666 | 0.698496 |
| score standard deviation | 0.023617 | 0.052665 |

이 표는2-update mechanical DEBUG의 실제 산출값이다. C/D 우열, 고장 블록, 추천 품질 회복, 서버40epoch 성능을 판정하지 않는다. C own-task8후보 BEST와 위의 공통 P/U 지표는 별도 평가다. 전체21case에서 공통 support와 동일 관측집합으로 비교하는 실제 학습 결과는 서버 실행 후에만 얻는다.

같은 설정의 own-resume도 실제로 실행했다. C는 기존2updates/4backward attempts와 neural tensor SHA가 유지됐고 추가 optimizer update는0이었다. C의 재시작 전 checkpoint 파일 SHA를 별도로 측정하지 않았으므로 checkpoint 파일의 byte 동일성을 주장하지 않는다. 재평가한 float score의 작은 수치 차이 역시 bitwise 동일하다고 부르지 않는다. D는 이미 완료한2 DEBUG updates를 인식하여 추가 update를 실행하지 않았고, 재시작 전후 checkpoint 파일 SHA가 동일했다.

최종 근거는 [실제 CT/CUDA verification](../validation/v17_crossed_training_20261005/verification.json)과 [완료·resume 검사](../validation/v17_crossed_training_20261005/completion_checks.json)에 게시했다. verification은 최종 C/D의174개 실행 source byte, 동결 config, 원본 ZIP SHA 및 결속된 원본 snapshot source를 확인하며 실제 보고서를 그대로 복사한 각 파일의 SHA/size를 기록한다. collector는 모델을 다시 돌리거나 checkpoint tensor를 deserialize하지 않는다. CT, canonical graph tensor, checkpoint 파일 자체를 공개 evidence에 복사하지 않았다.

- verification SHA256: `ca1e67b2815012649122ac78476416c8c045725c64ac5db41d1461e69047262d`
- completion checks SHA256: `beef8f14f5875ae86328fda111eafb852ce5c0899379decd933f747274c3c136`
- 동결 config SHA256: `aafbcc74619a58ec414905b8ad712a74051e1979f2c1623f993bdc09c10e3b99`

동결 config의 `final_evidence_binding_pending`와 `final_code_recheck_required`는 최종 C_r4/D_r3 검사 전에 작성한 선언 상태다. 실제 실행의 source-byte 결속을 유지하기 위해 결과에 맞춰 config를 사후 수정하지 않았다. 최종 검사 완료 상태는 위 verification/completion receipt에서 읽는다. implementation ready=true와 real CT/CUDA smoke complete는 entry 구현 및 로컬 실제 연결 검사를 뜻한다. server_training_started, full_training_complete, common_full_evaluation_complete, quality_verified, production_ready는 false다.

다음 실행은 Git에 게시한 정확한 commit의 `tools/server_v17_crossed.sh`에서 선택한 C 또는 D 한 arm만 시작한다. script의 `CP_CODE_COMMIT` 검사는 실제 checkout과 다르면 중단한다. `CP_GPU`, `CP_ARM`, arm별 고정 `CP_OUTPUT`을 명시하며 같은 출력으로 재실행하면 그 arm의 checkpoint를 사용한다. 최종 서버 명령의 commit 값은 게시 후 확인된 실제 SHA를 사용한다. D 전체 graph 준비와 full-shape batch32/실제 할당 자원 preflight,40epoch 학습, 전체21case 공통 평가는 남아 있다. 두 arm을 자동 실행하거나 CP/nnU-Net으로 자동 진행하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토하고 서로 다른 작업 단위를 명시했다.
- [x] 기존 서버 설정과 로컬 실제 CT/CUDA DEBUG의 자원 기록을 확인했다. 서버 full-shape GPU/CPU/RAM preflight는 남아 있다.
- [x] OOM 시 모델 축소 대신 측정과 진단으로 중단하도록 계약했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다.
- [x] 최종 C_r4/D_r3 실제 CT/CUDA DEBUG에서 forward/loss/gradient/optimizer 연결과 whole-candidate 평가를 확인하고 source-bound receipt로 결속했다.
- [x] 실제 적용할 설정, 설계 변경과 아직 미검증인 범위를 보고했다.
- [x] UNIT/smoke와 전체 학습 또는 전체 평가를 구분했다.
