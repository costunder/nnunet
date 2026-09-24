# v2.22 r6 서버 실행: 코드만 Git으로 이동

사용자 지시: **로컬 테스트 → 코드·설정 Git 업로드 → 서버 원본 CT로 준비 및 학습**. 로컬 graph cache, CT, checkpoint, DEBUG fixture를 전송하지 않는다.

현재 진입점은 `run_v222_v1_l0.py`다. v1 방식 paired L0 + v2.22 L1/L2, 5,550,806 parameters, seed42, 전체 GNN40epochs를 유지한다. `run_v222.py`는 이전 L0 경로다. Native online CP bank/nnU-Net 연결과 segmentation 평가는 미완료이므로 이번 실행은 **paired GNN 학습**이다.

## 현재 확인된 서버에서 짧게 실행

사용자 출력 확인: `ece-agpu16`, `nnunet` 환경 활성화, 코드 checkout·원본 경로 존재, GPU6의 지정 MIG가 PyTorch에1g.10gb/9.5GiB로 표시됐다. 다시 환경을 만들거나 GPU 할당을 바꾸지 않는다. 다음 명령은 새 run이며 같은 output이 이미 있으면 덮어쓰지 않고 실패한다.

```bash
git pull --ff-only origin codex/v222-server-r6
```

pull 성공 후 현재 `(nnunet)` 환경에서:

```bash
nohup python -u tools/run_v222_server.py \
  --medical-root /home/aicompetition06/Medical \
  --output work/v222_mig10gb_r6 \
  >> v222_mig10gb_r6.log 2>&1 < /dev/null &
```

```bash
tail -n 80 v222_mig10gb_r6.log
```

실행기는 아래의 기존 검증된 단계별 명령을 순서대로 호출한다. DEBUG profile32/allocator9GB이며 production batch는auto, GNN40epochs이다. 각 단계의 started/complete/failed JSON을 run root에 기록한다. 실패하면 다음 단계는 시작하지 않는다. `pipeline_complete.json`은 GNN 완료만 뜻한다. Python/GPU 환경 설치·재할당·원본 전송은 수행하지 않는다. 실행기 자체의 순서/실패 차단 단위3검사가 통과했으며 서버 전체 실행을 완료했다고 주장하지 않는다.

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
  nohup python -u tools/run_v222_server.py --medical-root /home/aicompetition06/Medical --output work/v222_mig10gb_r6_scanfix >> v222_mig10gb_r6_scanfix.log 2>&1 < /dev/null &
fi
```

로그: `tail -n 40 v222_mig10gb_r6_scanfix.log`. `raw_inventory completed/total`가 실제 완료 수다. 이 수정은 observations 단계에 한정하며 paired graph 준비와 전체 학습의 서버 소요시간은 아직 실측 전이다.

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
