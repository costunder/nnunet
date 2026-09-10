# HierCP GPT handoff

작성일: 2026-09-10 KST. 실행 소스 기준: `8e362280e971e5075474cbca82031dd963ecd4a0`.
이 인계 파일 추가와 동반 `code.txt` 재생성은 문서 변경이며 모델·설정 변경이 아니다.

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
| 현재 대화의 실험 | `work/feedback_medical_aug` |
| recovery 원본 | `work/feedback_experiment` |
| outer fold / dataset ID | `0` / `760` |
| 저장된 Python | `/home/aicompetition06/.conda/envs/nnunet/bin/python` |
| nnU-Net seed | `42`; quality GNN/bank는 별도 fold-specific 설정을 따른다 |
| 마지막 사용자 제공 bank 로그 | `[Bank] case 1/105 liver_1` |

위 로그는 해당 시점에 online bank 단계에 도달했다는 증거일 뿐이다.
현재 프로세스가 실행 중인지, 중단됐는지, bank·Full·Basic이 완료됐는지는
최신 서버 로그·journal·완료 증거를 받지 않아 **미확인**이다.
여러 GPU 서버 이름이 과거 대화에 등장했으므로 현재 호스트와 할당 GPU를 추정하지 않는다.
과거의 `/Medical/HierCP`는 Git 저장소가 아니었으며 현재 checkout과 다르다.
로컬 push 완료는 서버 pull/설치/실행 완료가 아니다.

## 4. 모델과 두 GNN의 역할

활성 계약은 `full_v22`, `level0_physical_closure_v2`,
`hiercp_conditioned_readout_v3`이다. 실제 설정은 `config/train.json`과
기존 실험의 `recovery/train_config.json`을 함께 확인한다.

- **L0 — local:** 종양 표면·내부, source/target 실질 context, 간 표면 anchor의
  이종 그래프와 3D patch encoder. 물리 거리와 full-shape geometry를 사용한다.
  context seed 384는 최종 전체 노드 수 cap이 아니다. interface와 지정 hop closure가
  유지되며 자원 제한 초과를 조용한 잘림으로 처리하지 않는다.
- **L1 — patient:** source tumor, 후보, 간 region, 다른 병변, whole-liver 관계.
  최종 region/lesion/liver 상태가 candidate-conditioned readout에 연결된다.
  source 위치 shortcut은 해당 계약에서 차단한다.
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
선택한 raw donor mask는 실제 nnU-Net axis/crop/resampling 계약으로 매핑한다.
다른 가까운 병변이나 합쳐진 전체 component로 대신하지 않으며 preprocessed CT는
float16 왕복 변환 없이 float32를 유지한다.

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
```

검증된 완료 단계는 재사용하고, 미완료 단계는 계약이 허용할 때만 이어간다.
이 runner는 downstream prediction·평가·통계 분석까지 수행하지 않는다.
bank 단계 시작은 기존 GNN이나 전처리를 처음부터 지웠다는 뜻이 아니다.

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

## 7. 검증된 범위와 과거 성능 결과

- `ebe58ff`의 로컬 보고: Python 3.10 문법 검사 및 594 tests 중
  592 passed, 2 skipped. 환자 전체 학습이 아니라 DEBUG/단위 회귀 검사다.
- 작은 CPU DEBUG 비교: 동일 후보 graph 3개, 3회 반복에서 standard 경로의
  full-target edge 생성 호출이 6회에서 3회로 줄고 tensor/edge/patch가 일치했다.
  서버 GPU 속도 향상률이나 전체 bank OOM 해결을 입증한 측정은 아니다.
- 직전 `code.txt` 갱신: 145개 경로와 본문을 실제 파일과 LF 정규화 후 정확 대조했다.
  이번 인계문서 추가 후 포함 파일 수와 기준 commit은 새 snapshot 머리말을 따른다.
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

아래는 **동일 실험이 중단되어 있고, preparation 완료 및 재개 적합성이 확인된 경우**의
기존 명령이다. 실행 중인 프로세스를 중단하라는 지시가 아니다.
현재 할당 GPU 번호만 입력한다. 예전 GPU 번호를 재사용한다고 가정하지 않는다.

```bash
conda activate /home/aicompetition06/.conda/envs/nnunet &&
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
read -r -p "현재 할당받은 GPU 번호: " CP_GPU &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$CP_GPU" \
/home/aicompetition06/.conda/envs/nnunet/bin/python -B tools/run_feedback_experiment.py \
  --recover-from work/feedback_experiment \
  --experiment-name feedback_medical_aug \
  --medical-root /home/aicompetition06/Medical \
  --outer-fold 0 \
  --dataset-id 760 \
  --seed 42 \
  --resume-experiment
```

`--resume-preparation`은 적합한 미완 preparation용이고, 위 `--resume-experiment`는
preparation 완료 후 별도의 검증된 continuation이다. 둘 다 원래 `--recover-from`으로
생성한 실험에 한정된다. fresh run에 임의로 적용하지 않는다.
저장된 Python 경로도 plan identity에 포함된다. 과거 `(base)` Python으로 검사했을 때는
checksum/source가 맞아도 Python 경로만 달라 plan mismatch가 발생했다.
이것을 journal checksum 완화로 고치지 않는다.
기존 bank error row, 불충분 pool, 미완 runtime, running row/lock,
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
