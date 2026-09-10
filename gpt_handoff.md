# HierCP GPT handoff

작성일: 2026-09-10 KST. 실행 소스 기준과 파일 목록은 동반 `code.txt`의
`source_commit` 및 `Tree`를 따른다. raw CP → nnU-Net 전처리의 후보 위치별 표현과
trainer 연결 수정에 이어, 최신 변경은 bank-upgrade의 과거 실패 로그 호환성 수정이다.

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
| 직전 bank 실패 | `liver_101`, component `3`, raw source 1 voxel → 원래 위치에서 독립 resampling한 mask 0 voxel |
| 최신 사용자 제공 오류 | 새 `feedback_rawcp` 시작 전 source history 검증: `Source experiment configurations changed since its recorded attempt` |

최신 오류는 `validate_source` 안에서 발생했다. 이 invocation은 새 root 생성,
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
```

검증된 완료 단계는 재사용하고, 미완료 단계는 계약이 허용할 때만 이어간다.
이 runner는 downstream prediction·평가·통계 분석까지 수행하지 않는다.
bank 단계 시작은 기존 GNN이나 전처리를 처음부터 지웠다는 뜻이 아니다.

새 `--upgrade-bank-from` 경로는 위의 기존 recovery 재학습 경로와 별개다.
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

## 7. 검증된 범위와 과거 성능 결과

- 최신 source-history 수정: upgrade DEBUG 26개 중 25개 통과·Windows symlink 권한
  1개 skip(8.640초), 기존 실행·재개 DEBUG 32개 모두 통과(39.098초).
  합계 **58개 중 57개 통과, 1개 skip, 실패 0**. 새 회귀 테스트 9개는 모두 통과했다.
  동일한 DEBUG legacy 기록을 이전 `a406574` reader에 넣으면 사용자와 같은 오류가
  발생하고, 수정 reader는 후속 동일 단계 증거 검증 후 통과함을 별도 대조했다.
  원본 journal·파일 내용/목록 보존 및 GNN·전처리 재실행 없는 stage driver를 검사했다.
  무거운 native 성공 검증에는 명시적 경계 double을 사용하며 실제 서버 journal이나
  환자 데이터 검증을 대신하지 않는다. 이번에는 아래 전체 662개 suite나 전체 학습·평가를
  재실행하지 않았다. 아래 전체 suite 수치는 직전 raw-target 구현의 결과다.
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

아래는 **원본 실험이 중단되어 있고, GNN/공통 전처리가 완료되어 검증 가능한 경우**
새 raw-target 실험을 만드는 명령이다. 실행 중인 프로세스를 중단하라는 지시가 아니다.
변경 전 `feedback_medical_aug`에 새 trainer를 덮어씌워 resume하는 명령이 아니다.
현재 할당 GPU 번호만 입력한다. 예전 GPU 번호를 재사용한다고 가정하지 않는다.

```bash
conda activate /home/aicompetition06/.conda/envs/nnunet &&
cd /home/aicompetition06/Medical/HierCP-git &&
git pull --ff-only &&
read -r -p "현재 할당받은 GPU 번호: " CP_GPU &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$CP_GPU" \
/home/aicompetition06/.conda/envs/nnunet/bin/python -B tools/run_feedback_experiment.py \
  --upgrade-bank-from work/feedback_medical_aug \
  --experiment-name feedback_rawcp \
  --medical-root /home/aicompetition06/Medical \
  --outer-fold 0 \
  --dataset-id 760 \
  --seed 42
```

새 upgrade 실험의 안전 재개는 위와 같은 원래 인수에 `--resume-experiment`를 추가한다.
단, 최신 사용자 오류처럼 source history 검증에서 멈춘 invocation은 새 root를 만들기
전이므로 코드 업데이트 후 위의 최초 실행 인수를 그대로 쓴다. 다른 invocation이
이미 root를 만들었으면 삭제하지 말고 그 기록에 맞는 재개 여부를 확인한다.
원본 `feedback_medical_aug` journal은 수정하지 않는다. 기존 `--recover-from` 기반
실험의 `--resume-preparation`/`--resume-experiment` 경로도 남아 있지만 새 bank 표현으로
구버전 runtime을 바꾸는 수단이 아니다. `--overwrite`나 journal 수작업 삭제로 우회하지 않는다.
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
