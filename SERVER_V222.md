# v2.22 r6 서버 실행: 코드만 Git으로 이동

## 2026-09-27 우선 확인 — 좌표 오류 수정과 기존 재개의 구분

독립 검토에서 CNN 입력→특징 좌표 오류가 확인됐다. [수정·검증·남은 연구 문제](docs/v222_independent_review_response_20260927.md). **기존 checkpoint 재개는 과거 입력 의미를 유지하며, 좌표가 수정된 새로운 실험으로 바뀌지 않는다.** 기존 학습과 corrected 학습을 같은 run/성능 결과로 섞지 않는다.

수정된 fresh 학습은 최신 코드 checkout에서 아래처럼 명시한다. 실행 중인 학습이 있으면 먼저 기존 viewer의 Ctrl+C→PAUSED를 확인한다. 새 코드 checkout 준비 방식은 아래의 code-only worktree 절차를 따르며 실행 중 checkout은 pull하지 않는다. 여기서는 서버 명령을 실행하지 않았다.

```bash
conda activate nnunet
CUDA_VISIBLE_DEVICES=MIG-774a3cc0-0169-5e18-b5d9-fe1b7a1d6ce7 python tools/run_v222_server.py \
  --runtime process \
  --feature-coordinates stride4 \
  --medical-root /home/aicompetition06/Medical \
  --cache /home/aicompetition06/Medical/HierCP-v222-r6/work/v222_mig10gb_r6_scanfix/paired_cache/index.json \
  --output work/v222_stride4_fresh_20260927
```

원본 graph cache를 재사용하지만 가중치와 learned support는 새로 학습/계산한다. `--resume` 없는 새 실험이다. 모델/graph/data/epoch를 축소하지 않으며 production batch/worker는 실제 calibration으로 결정된다. 최신 process fresh 기본 좌표도 `stride4`지만 명령에는 의도를 명시했다. 다른 `optimized/legacy` 경로는 과거 좌표를 보존한다.

과거 작업을 정확히 이어가려면 아래2026-09-26의 `--runtime process --resume 최신checkpoint` 명령을 사용하고 `--feature-coordinates stride4`를 추가하지 않는다. 필드가 없는 checkpoint는 `legacy`로 처리된다. 이미 corrected로 시작한 run은 saved `stride4`를 상속한다. 서로 다른 coordinate contract의 exact resume는 거부한다. 아래의 오래된 fresh 예시는 legacy 좌표를 사용하는 역사적 명령이다.

사용자 지시: **로컬 테스트 → 코드·설정 Git 업로드 → 서버 원본 CT로 준비 및 학습**. 로컬 graph cache, CT, checkpoint, DEBUG fixture를 전송하지 않는다.

현재 서버 진입점은 `tools/run_v222_server.py`이며 기본값은 중복 제거 backend인 `tools/run_v222_optimized.py`다. v1 방식 paired L0 + v2.22 L1/L2, 5,550,806 parameters, seed42, 전체 GNN40epochs를 유지한다. `run_v222.py`는 이전 L0 경로다. Native online CP bank/nnU-Net 연결과 segmentation 평가는 미완료이므로 이번 실행은 **paired GNN 학습**이다.

## 현재 확인된 서버에서 짧게 실행

### 2026-09-26 CPU 준비·pinning·support 복사 개선 경로

`--runtime process`는 동일 모델과 입력을 유지하면서 CPU graph 준비를 별도 producer4개로 분리한다. 현재 저장된 workers8을 2개씩 나누고 별도 pinning thread로 최대4batch를 선행 준비한다. 캐시는 단계 사이 유지한다. Support pass 중 불변인 model/Adam은 한 번 CPU에 복사하고, checkpoint 자체는 매 batch 완전한 파일로 저장한다. [측정 및 검증 범위](docs/v222_support_snapshot_20260926.md). A100 MIG 전체 epoch 실측은 아니다.

현재 대상은 **`v222_mig10gb_r6_fast_resume_20260926`**이다. 먼저 최신 viewer에서 Ctrl+C를 누르고 **PAUSED** 표시까지 기다린다. 구형 viewer가 `close viewer only`라고 표시한다면 아래 파일을 현재 작업에만 생성하고, 최신 viewer를 연결해 PAUSED를 확인한다. 서버나 SSH를 종료하는 명령은 사용하지 않는다.

```bash
touch /home/aicompetition06/Medical/HierCP-v222-r6-runtime/work/v222_mig10gb_r6_fast_resume_20260926/training/STOP_AFTER_BATCH
```

최신 viewer만 가져와 확인하려면 실행 중인 checkout을 변경하지 않고 다음을 사용한다. 기존 viewer가 이미 최신이고 PAUSED가 보이면 이 단계는 생략한다.

```bash
cd /home/aicompetition06/Medical/HierCP-v222-r6-runtime &&
git fetch origin codex/v222-server-r6 &&
V222_VIEW="$(mktemp /tmp/v222-progress.XXXXXX.py)" &&
git show origin/codex/v222-server-r6:tools/watch_v222_server.py > "$V222_VIEW" &&
python "$V222_VIEW" --output /home/aicompetition06/Medical/HierCP-v222-r6-runtime/work/v222_mig10gb_r6_fast_resume_20260926
```

PAUSED 이후 **코드만** 새 worktree로 받고 저장된 위치부터 재개한다. `fast_resume`의 최신 체크포인트를 사용하며 예전 `scanfix` 체크포인트로 되돌아가지 않는다. graph cache는 기존 절대 경로를 참조하고 재생성·복사하지 않는다. 기존 GPU6 MIG 할당을 유지한다.

```bash
conda activate nnunet
cd /home/aicompetition06/Medical/HierCP-v222-r6-runtime &&
git fetch origin codex/v222-server-r6 &&
git worktree add --detach /home/aicompetition06/Medical/HierCP-v222-r6-process2 origin/codex/v222-server-r6 &&
cd /home/aicompetition06/Medical/HierCP-v222-r6-process2 &&
CUDA_VISIBLE_DEVICES=MIG-774a3cc0-0169-5e18-b5d9-fe1b7a1d6ce7 python tools/run_v222_server.py \
  --runtime process \
  --medical-root /home/aicompetition06/Medical \
  --cache /home/aicompetition06/Medical/HierCP-v222-r6/work/v222_mig10gb_r6_scanfix/paired_cache/index.json \
  --resume /home/aicompetition06/Medical/HierCP-v222-r6-runtime/work/v222_mig10gb_r6_fast_resume_20260926/training/checkpoint_latest.pt \
  --output work/v222_process2_resume_20260926
```

새 worktree나 output이 이미 존재하면 덮어쓰지 않고 실패한다. 원본 학습이 아직 살아 있거나 알려진 runtime SHA와 다르면 실행을 거부한다. `256MiB`, batch, allocator 정책, optimizer/RNG, epoch/step, 부분 support 진행 위치는 checkpoint에서 그대로 이어진다. 이 명령에 `--migrate-workspace-mib`를 다시 추가할 필요는 없다. 위 명령은 마지막으로 확인된 `fast_resume`가 최신 실행인 경우다. 이미 다른 output에서 이어 학습했다면 그 최신 checkpoint를 사용해야 하며 `fast_resume`로 되돌리지 않는다.

### 이전 실행 기록과 명령

**2026-09-25 사용자 출력으로 기존 학습 실행 중 확인:** PID3350532는 `v222_mig10gb_r6_scanfix/training`을 사용하는 원래 trainer다. 새 resume의 OOM 당시 기존6.16GiB + 새3.16GiB가 같은 MIG를 점유했다. 이 상태에서는 아래 재개 명령을 다시 실행하지 않고 기존 화면에 연결한다. 이미 만들어진 runtime checkout의 viewer를 사용할 수 있다.

```bash
python /home/aicompetition06/Medical/HierCP-v222-r6-runtime/tools/watch_v222_server.py \
  --output /home/aicompetition06/Medical/HierCP-v222-r6/work/v222_mig10gb_r6_scanfix
```

수정된 runner는 원본 worker와 해당 output의 trainer가 살아 있으면 새 실행을 거부한다. 이 검사는 신호를 보내지 않는다. 최신 서버 상태는 viewer에서 확인하고, 아래 최적화 재개는 원본이 저장 후 중단된 경우에만 사용한다. 다른 GPU 작업의 점유까지 없음을 보장하는 전역 GPU lock은 아니다.

사용자 출력 확인: `ece-agpu16`, `nnunet` 환경 활성화, GPU6의 지정 MIG1g.10gb/9.5GiB. 기존 할당과 환경을 유지한다. **실행 중인 checkout은 pull하지 않는다.** 코드만 별도 checkout으로 준비한다. 아래 worktree 경로가 이미 있으면 덮어쓰지 않고 실패한다.

```bash
git -C /home/aicompetition06/Medical/HierCP-v222-r6 fetch origin codex/v222-server-r6 &&
git -C /home/aicompetition06/Medical/HierCP-v222-r6 worktree add --detach /home/aicompetition06/Medical/HierCP-v222-r6-runtime FETCH_HEAD
```

기존 worker가 종료/저장 중단된 것을 확인한 뒤 아래 둘 중 하나만 실행한다. 기존 worker에 대한 중단 명령은 이 문서에서 자동 실행하지 않는다. 현재 서버 진행 상태는 로컬에서 확인하지 못했다.

**기존 가중치에서 개선된256MiB로 이어가기 (2026-09-26):** 완료된14,102개 graph index와 rolling checkpoint를 그대로 참조한다. `--migrate-workspace-mib 256`은 가중치·optimizer·RNG·support·L2 plan·epoch/batch 위치를 복원하면서 GPU 연산 chunk만 변경한다. raw/graph/profile 단계를 반복하지 않는다. 기존 allocator 정책은 유지한다. 이후 부동소수점 누적 순서가 달라질 수 있음을 checkpoint 실행 정책에 명시하며, 최종 결과의 bitwise 동일을 주장하지 않는다. 이 옵션이 없으면 기존 workspace를 상속한다.

```bash
cd /home/aicompetition06/Medical/HierCP-v222-r6-runtime &&
python tools/run_v222_server.py \
  --medical-root /home/aicompetition06/Medical \
  --cache /home/aicompetition06/Medical/HierCP-v222-r6/work/v222_mig10gb_r6_scanfix/paired_cache/index.json \
  --resume /home/aicompetition06/Medical/HierCP-v222-r6/work/v222_mig10gb_r6_scanfix/training/checkpoint_latest.pt \
  --migrate-workspace-mib 256 \
  --output work/v222_mig10gb_r6_fast_resume_20260926
```

**가중치를 새로 학습하는 경우에만:** `--resume`을 빼고 새 output을 사용한다. 기존 cache는 재사용하며256MiB workspace + 중복 제거 loader/writer로 시작한다. 이것은 이어 학습이 아니다.

```bash
cd /home/aicompetition06/Medical/HierCP-v222-r6-runtime &&
python tools/run_v222_server.py \
  --medical-root /home/aicompetition06/Medical \
  --cache /home/aicompetition06/Medical/HierCP-v222-r6/work/v222_mig10gb_r6_scanfix/paired_cache/index.json \
  --output work/v222_mig10gb_r6_runtime_fresh
```

터미널에는 tqdm이 표시되고 로그는 자동 저장된다. `--cache` 없는 새 데이터 준비는 calibration 결과를 버리지 않는 준비기를 사용한다. DEBUG profile32/allocator9GB, production batch auto, GNN40epochs를 유지한다. 기존 입력·checkpoint는 수정하지 않으며 항상 새 output을 사용한다. [실측·검증·한계](docs/v222_runtime_optimization_20260925.md).

## GPU 엣지 연산의 작업 메모리 선택 (2026-09-25)

새 실행은 `--edge-workspace-mib 256`을 선택할 수 있다. 모델 크기, 모든 graph node/edge, physical batch 계약은 그대로이며 한 번에 계산하는 edge 조각만 커진다. 로컬 실제 batch32 두 개/9GB allocator에서64MiB 대비 step3.30~3.42배 개선, peak allocated약6.3GB를 측정했다. 서버 A100 처리량과 전체epoch 시간으로 일반화하지 않는다. [측정과 검증 범위](docs/v222_cached_execution_20260925.md).

2026-09-25 중복 제거 이후 **새 실행 기본값은256MiB**다. `--runtime legacy`는 기존64MiB 경로를 보존한다. `--resume` 기본값은 원래 workspace·allocator 정책 상속이다. 2026-09-26 추가한 `--migrate-workspace-mib 256`으로 workspace 전환을 명시할 수 있다. 실제 CT/full model의64MiB 저장→256MiB 재개를 메모리상의256MiB 연속 실행과 비교해 loss·전체가중치·optimizer가 일치하고 cursor가 유지됨을 확인했다. 관련24검사 통과. 전체epoch/A100속도 검증은 아니다.

## 터미널 진행 화면: tqdm 기본 표시 (2026-09-24)

2026-09-26 수정: **Ctrl+C 기본 동작은 해당 학습의 저장 후 중단 요청**이다. viewer가 `training/STOP_AFTER_BATCH`를 만들고 `PAUSED`까지 기다린다. optimizer/support는 현재 batch 뒤, validation은 해당 검증 뒤 중단한다. 프로세스에 종료 신호를 보내지 않는다. 학습 초기화 전 준비 단계는 이 batch 중단 프로토콜을 지원하지 않으며, 중단했다고 거짓 표시하지 않고 오류를 낸다. 화면만 닫으려면 독립 viewer의 `--detach-on-interrupt`를 명시한다.

화면은 `execution_contract.json` 및 `checkpoint_status.json`을 직접 읽어 전체 epoch 수·저장된 epoch/step·initial support/epoch support refresh/validation/final support를 구분해 표시한다. 마지막 support를 가짜 epoch41로 표시하지 않는다. 별도 `cat`으로 상태 파일을 읽을 필요가 없다.

새 실행은 아래처럼 직접 실행한다. 상세 출력은 실행 폴더 `console.log`에 자동 저장되고 터미널의 두 줄 tqdm을 갱신한다. `nohup`, 출력 리디렉션, `&`, 반복적인 `tail` 명령이 필요하지 않다.

```bash
python tools/run_v222_server.py --medical-root /home/aicompetition06/Medical --output work/v222_mig10gb_r6_progress
```

이미 실행 중인 작업에 화면만 연결하려면 아래를 실행한다. 작업 checkout을 pull하거나 프로세스를 중단하지 않는다. 구형 실행기의 외부 `.log`와 새 실행기의 `console.log`를 모두 읽는다. `--latest`는 현재 checkout의 `work/v222_mig10gb_r6*` 중 최신 기록된 run을 선택하며, 명시하려면 `--output work/<run>`을 쓴다.

```bash
git fetch origin codex/v222-server-r6 &&
V222_VIEW="$(mktemp /tmp/v222-progress.XXXXXX.py)" &&
git show FETCH_HEAD:tools/watch_v222_server.py > "$V222_VIEW" &&
python "$V222_VIEW" --latest
```

이미 열려 있는 구형 viewer는 기존 Ctrl+C 동작을 유지하므로 한 번 닫고 새 viewer에 연결한다. 현재 개선 재개 실행을 명시하려면 위 마지막 줄을 다음으로 바꾼다. 새 viewer부터 Ctrl+C가 저장 후 학습 중단을 요청한다.

```bash
python "$V222_VIEW" --output /home/aicompetition06/Medical/HierCP-v222-r6-runtime/work/v222_mig10gb_r6_fast_resume_20260926
```

2026-09-26 검증: 관련19검사 통과. 화면 Ctrl+C→선택한 run에만 pause marker 생성→PAUSED 확인까지 대기, 초기화 전 거절, marker 중복 요청 안전성, checkpoint payload 보존, support 중 epoch/step 표시, 마지막 support의 epoch41 표시 방지, 명시적 read-only detach, 실제 별도 자식의 협력 중단을 검증했다. 모델·GPU 계산 경로·기존 checkpoint는 변경하지 않았다. 서버 창 교체는 사용자가 위 명령을 실행해야 한다.

CT 완료 수, 그래프 완료 수, support 인코딩 수, epoch별 query 수·loss·step을 실제 로그에서 읽는다. 속도/ETA는 연결 후 새로 완료되는 작업을 기준으로 측정하며, 과거 로그를 빠르게 읽은 속도를 쓰지 않는다. 별도 측정이나 validation처럼 완료 수를 제공하지 않는 단계는 단계명과 경과 시간만 표시한다. 구형 벤치마크의 미출력 case 진행률을 추측하지 않는다. 전체 단계7개는 동일 소요시간이 아니므로 단계 완료 비율을 전체 시간의 비율로 해석하지 않는다. 오류 시 화면을 종료하고 마지막 로그와 실패 기록 경로를 표시한다.

검증: 진행 프로토콜/실제 분리 프로세스/화면 중단 후 지속/실패 표시와 기존 제어 검사9개 통과. 합성 입력은 UI 테스트로만 사용했고 학습 증거가 아니다. 모델·캐시 provenance 불변. Linux/MobaXterm 실제 화면은 원격으로 직접 확인하지 않았다.

## 실행 중인 observations 중복 벤치마크 교체 (2026-09-24)

`workers1 tasks128 calibrationTrue`는 학습이 아니라 기존 준비 코드가 동일 CT를 반복 읽는 측정 단계다. 수정본은 전체131개를 한 번씩 처리하고 결과를 저장한다. 개별 완료와10초 heartbeat를 출력한다. 로컬 전체 재생성 결과 동일성 및 관련7검사 통과. 모델·데이터·학습 설정은 유지한다.

현재 `nnunet` 환경과 지정된 GPU6의 MIG 환경변수를 유지한다. 아래는 기존 run의 소유자·정확한 자식 명령·진행 단계를 확인한 뒤 해당 observations 자식만 중단한다. 기존 파일과 로그는 보존한다. stop helper는 PID와 명령을 먼저 출력하고 wrapper의 실패 기록을 확인한다. 이미 다음 단계에 도달했거나 프로세스가 불명확하면 거절하고 새 실행도 시작하지 않는다. Linux 신호 실행 자체는 로컬 Windows에서 검증하지 않았으며 보호 조건 단위검사만 완료했다.

```bash
if cd /home/aicompetition06/Medical/HierCP-v222-r6 &&
   git fetch origin codex/v222-server-r6 &&
   V222_STOP_TOOL="$(mktemp /tmp/v222-stop.XXXXXX.py)" &&
   git show FETCH_HEAD:tools/stop_v222_observation_job.py > "$V222_STOP_TOOL" &&
   python "$V222_STOP_TOOL" --output work/v222_mig10gb_r6 --stop &&
   git pull --ff-only origin codex/v222-server-r6; then
  python tools/run_v222_server.py --medical-root /home/aicompetition06/Medical --output work/v222_mig10gb_r6_scanfix
fi
```

터미널에는 tqdm 진행 화면이 자동 연결된다. `work/v222_mig10gb_r6_scanfix/console.log`에 상세 로그가 남는다. 이 준비 성능 수정은 observations 단계에 한정하며 paired graph 준비와 전체 학습의 서버 소요시간은 아직 실측 전이다.

## 코드와 환경

브랜치 `codex/v222-server-r6`를 새 디렉터리에 checkout하고 최종 보고한 커밋 SHA를 확인한다. 기존 서버 checkout·실험을 덮어쓰지 않는다. Git에는 코드·설정·문서·작은 검증 보고서 및 원본 보존용 소스 archive만 포함한다.

새 checkout 최상위에서 실행한다. 사용 허가받은 GPU를 `CUDA_VISIBLE_DEVICES`로 지정한다. 로컬 검증 환경은 Python3.10, PyTorch2.8.0+cu128, PyG2.6.1이다. `requirements-v222-server.txt`는 추가 패키지 버전이며 PyTorch/CUDA/드라이버를 설치하지 않는다. 서버 환경을 먼저 확인하고 필요하면 별도 환경을 사용한다. 기록된 Python은 `/home/aicompetition06/.conda/envs/nnunet/bin/python`이며 아래 `python`은 확인한 환경을 뜻한다.

```bash
python tools/v1_server.py resources --output work/a6000_resources.json
python run_v222_v1_l0.py check
```

`.gitattributes`는 고정 v1 원본과 새 소스의 바이트를 보존한다. 해시를 재표기하거나 검사를 끄지 않는다. A6000 실제 처리량은 아직 측정 전이다.

## A100 MIG 10GB에서의 추가 조건

스케줄러가 지정한 `CUDA_VISIBLE_DEVICES`는 유지한다. 수동 지정이 필요하면 본인에게 할당된 MIG UUID를 사용하며, 물리 GPU 번호0으로 덮어쓰지 않는다. 아래 DEBUG profile은 `--batch-size 32`를 명시한다. 기존 batch64 기본 호출은10GB에서 사용하지 않는다. 본학습의 `batch_size=auto`는 실제 MIG에서 다시 측정하며, 로컬 테스트로 MIG 처리량을 추정하지 않는다.

RTX5070Ti의 PyTorch allocator를9GB로 제한하는 실제 CT 테스트 도구는 `tools/smoke_v1_memory_limit.py`다. 이는 CUDA context 등 allocator 외부 메모리나 MIG 연산량을 재현하지 않는다. 전체 모델을 유지하고 큰 실제 그래프에서 physical16/32/64, gradient·optimizer·정확한 재개·평가 출력을 검사한다. 완전한 본학습이 아닌 명시적 DEBUG이며 검증 결과는 별도 기록한다.

## 서버 원본으로 관측·그래프 생성

원본은 `/home/aicompetition06/Medical/Data/image` 및 `Data/labels`다. 전체131개 case를 사용한다. `config/split_cp80_fold0.json`은 현재 로컬 CP80 비교 split이며 재추첨하지 않는다. outer train/val105/26, inner train/val84/21이다. 과거 서버 실험과 split 동일성을 검증했다고 주장하지 않는다.

```bash
python tools/v1_server.py observations \
  --medical-root /home/aicompetition06/Medical \
  --output work/v222_server_observations_r6

python -u run_v222_v1_l0.py prepare \
  --index work/v222_server_observations_r6/index.json \
  --output work/v222_server_paired_cache_r6
```

첫 단계는 원본 CT/mask 전체를 읽어 기존 함수로 소형 병변 anchor, case당128개 비교 위치, inner-train donor 목록을 재생성한다. 중복 CT와 split 독립성을 검사하고 CPU/RAM 기준 병렬 worker를 측정한다. 폐기한 L0의 context patch는 생성하지 않는다. 공개 case 단위 구분과 제공 주석의 한계를 유지한다. 로컬 기준14,102관측·527donor이며 서버 데이터가 다르면 원인을 확인해야 한다.

두 번째 단계가 서버에서 현재 paired L0의 전체 그래프 캐시를 새로 생성한다. 출력은 존재하지 않는 새 경로여야 한다. 로컬 캐시 복사나 해시 재표기를 하지 않는다.

## 실제 입력 테스트

```bash
HIERCP_TEST_OBSERVATION_INDEX=work/v222_server_observations_r6/index.json \
python -m unittest tests.test_v1_execution tests.test_v222_v1_local tests.test_v1_paired_training tests.test_v1_deterministic_sampling tests.test_v1_empty_context

HIERCP_TEST_OBSERVATION_INDEX=work/v222_server_observations_r6/index.json \
python run_v222_v1_l0.py verify --output work/v222_server_graph_DEBUG

HIERCP_TEST_FIXTURE=work/v222_server_graph_DEBUG/actual_graphs_DEBUG.pt \
python tools/verify_v1_resume.py work/v222_server_resume_DEBUG

HIERCP_TEST_FIXTURE=work/v222_server_graph_DEBUG/actual_graphs_DEBUG.pt \
python tools/smoke_v1_execution_lifecycle.py work/v222_server_lifecycle_DEBUG

HIERCP_TEST_FIXTURE=work/v222_server_graph_DEBUG/actual_graphs_DEBUG.pt \
python tools/profile_v1_execution.py \
  work/v222_server_paired_cache_r6/index.json \
  work/v222_server_physical32_profile_DEBUG release_unused --batch-size 32
```

실제 metadata가 없으면 관련 테스트가 skip되므로 전체 통과로 세지 않는다. `verify`는 서버 원본에서3recipient/6관측의 전체 크기 그래프를 직접 생성하고 forward/loss/gradient/optimizer 연결을 검사한다. GPU 테스트는 순차 실행한다. profile은 지정한 physical batch로 실제 그래프를10배치 처리하며 본학습을10batch로 제한하지 않는다. lifecycle DEBUG는 실행 제어 검사 때문에 같은 fixture를 train/val로 재사용하므로 점수를 성능 결과로 쓰지 않는다.

## 본학습과 재개

```bash
python -u run_v222_v1_l0.py train \
  --cache work/v222_server_paired_cache_r6/index.json \
  --output work/v222_a6000_training_r6 \
  --release-unused
```

5070Ti의 calibration 파일을 가져오지 않는다. 서버에서 physical batch/worker를 재측정하며 선택값과 총 step은 `training_started.json`에 기록한다. 모델·전체 데이터·그래프·40epochs는 유지한다. 서버가 선택한 batch가 로컬64와 다르면 기록하며 학습 동작까지 동일하다고 주장하지 않는다.

`checkpoint_status.json`은 저장된 phase/step/support 진행, `first_optimizer_step.json`은 optimizer 시작, `step_timings.jsonl`은 연산별 시간과 VRAM이다. `training_complete.json`이 있어야 전체 GNN 완료이며 nnU-Net 완료와 다르다. `metrics.csv`는 관측 태스크 평가이고 segmentation Dice가 아니다.

중단 요청은 실행 폴더에 `STOP_AFTER_BATCH`를 만든다. 현재 batch 저장 후 정상 반환한다. 서버 checkpoint는 같은 코드/config/cache로 새 output에 재개한다.

```bash
python -u run_v222_v1_l0.py train \
  --cache work/v222_server_paired_cache_r6/index.json \
  --resume work/v222_a6000_training_r6/checkpoint_latest.pt \
  --output work/v222_a6000_training_r6_resume1 \
  --release-unused
```

로컬 본학습은 저장·중단 상태이며 서버는 seed42의 새 실행으로 시작한다. 서버 runtime 검증과 본학습은 아직 실행하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 로컬 측정; 서버 확인 명령 제공.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 로컬 실제 CT 검증.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
