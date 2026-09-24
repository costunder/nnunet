# v2.22 r6 서버 실행: 코드만 Git으로 이동

사용자 지시: **로컬 테스트 → 코드·설정 Git 업로드 → 서버 원본 CT로 준비 및 학습**. 로컬 graph cache, CT, checkpoint, DEBUG fixture를 전송하지 않는다.

현재 진입점은 `run_v222_v1_l0.py`다. v1 방식 paired L0 + v2.22 L1/L2, 5,550,806 parameters, seed42, 전체 GNN40epochs를 유지한다. `run_v222.py`는 이전 L0 경로다. Native online CP bank/nnU-Net 연결과 segmentation 평가는 미완료이므로 이번 실행은 **paired GNN 학습**이다.

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
