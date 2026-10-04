# v1.5 half A — 성공한 10mm v1의 L0 교체 대조

## 실험 질문과 실행 순서

기준선은 서버에서 40epoch를 완료한 `v1_m10_seed42_20261004`다. 사용자가 전달한 validation 결과는 best epoch11과 마지막 epoch40 모두 MRR/top1=1이었다. 기존 결과를 재학습하거나 덮어쓰지 않고 **첫 절반 A(L0)만** 교체한다. A에서 저하가 재현되면 A 내부를 나누고, 재현되지 않으면 나머지 B를 다음 단계에서 검사한다. B와 A+B를 자동 실행하지 않는다. 자동 tolerance, 자동 품질 PASS, 한 부품이 단일 원인이라는 단정은 없다.

원본 v1의 source tumor original anchor가 정답 index0이며, sample당 curriculum8과 pool128을 유지한다. 관측 P/U 정답으로 전환하거나 U를 CP 부적합으로 재정의하지 않는다. 기존 curriculum, 난이도, corruption, 후보 순서, 두 view, 전체 ranking/consistency loss, original L1/L2/scalar scorer, seed42, optimizer/scheduler/AMP, 40epochs를 유지한다. configured cohort는 84train/21validation이며 outer26은 제외한다.

## 실제 교체 경계

이 arm의 정확한 이름은 **`v2CNN_on_preserved_v1_denseROI`**다. native-spacing v2.2 LocalCNN 전체를 복제한 모델이라고 주장하지 않는다.

```text
동일 10mm transformed footprint bbox ROI / 기존 target erasure / fixed48³
  → 간 내부 CT만 공유 OrganPyramid CNN (12/24/32 channels, conv 2/3/3)
  → 실제 footprint·원본 sampled role/shell별 scale 평균
  → shared 68→128 projection
  → [d, r, r-d, r*d] shared 512→256→128 fusion
  → 의미가 서로 다른 기존 12개 L0 dictionary 표현
  → 원본 L1 2층 / L2 2층 / 128D / 4heads / scalar scorer
  → 원본 complete loss → backward → fresh AdamW
```

원본 L0 CNN·GAT·attention readout의 parameter는 모두 제거하고 새 L0로 교체한다. 기존 L1/L2/readout/scorer는 같은 seed42 초기 tensor를 유지한다. 학습 완료 baseline 가중치와 Adam을 새 arm에 이식하지 않는다. 총 trainable parameter는 **5,127,124**다. 원본 constructor의 `local_layers=3` 인자는 기록하되 실제 L0 graph message passing은 실행하지 않는다고 명시한다.

단일 128D를 모든 dictionary key로 복제하지 않는다. 실제 tumor footprint/surface/interior, source/target context, 실제 liver surface, 원본 c0/c1/c2 shell, 관계 표현과 whole-organ-within-ROI pair fusion을 구분한다. 원본에서 존재하지 않는 선택적 liver surface와 빈 shell은 empty-set raw mean을 공유 projection 전에 사용한다. 필수 tumor/context 데이터 오류를 빈 값으로 숨기지 않는다.

원본 5×48³ cached payload와 organ/footprint mask를 보존하며 CNN이 보는 영상은 channel0의 간 내부 CT다. 원본 target erasure를 그대로 유지한다. 원본 sampled role 좌표는 readout에 사용한다. cached L0 edge는 CNN message passing에 사용하지 않는다. source feature map은 동일 source owner끼리 공유하며 원본 dense chunk4를 유지한다. 새 cap, 후보 drop, record skip, 작은 모델 fallback은 없다.

## 실행과 체크포인트 결속

`tools/run_v1_half_a.py`는 baseline의 실제 manifest/source202/config/split/cache/bank/regions, initial validation, completed best/last checkpoint, calibration 및 training signature를 검증한다. 마지막 epoch40 완료와 best의 `completed_epoch=40`, 실제 uint8 scope marker를 모두 확인한다. 검증 과정은 baseline을 읽기만 한다.

새 root의 config에서 바뀌는 항목은 기존 `auto` batch/worker를 baseline preflight에서 측정한 정수로 확정하는 두 항목뿐이다. L0 교체는 별도 runtime adapter와 해시 계약으로 연결한다. physical batch/worker는 비교 조건이므로 baseline lock을 재사용한다. 이를 새 모델의 처리량 최적 batch라고 부르지 않는다. 실제 GPU/allocation fingerprint가 baseline 측정과 다르면 조용히 다른 조건으로 실행하지 않는다.

새 manifest, scope, adapter, 실제 state marker, local extra state가 다른 checkpoint를 거부한다. 체크포인트는 새 arm의 `results/half_A/`에만 저장한다. 첫 실행은 fresh 초기화이고 같은 명령의 재실행은 **이 arm의 마지막 완료 epoch**에서 재개한다. 원본 v1과 다른 A/B의 checkpoint를 exact resume로 받지 않는다. 현재 epoch의 모든 update가 보존된다고 주장하지 않는다.

초기 validation 및 epoch telemetry를 원본 학습 loop에 연결한다. loss/ranking loss/MRR/top1/margin, loader/sampling/GPU optimization/validation/checkpoint 시간, VRAM이 기록된다. validation forward는 추가 실행하지 않고 기존 8-score 결과를 case별 JSONL에 기록한다. 원래 sample-weighted metric/best selection은 바꾸지 않는다.

validation cohort는 configured21을 임의로 채우지 않고, 이미 결속된 cache index의 실제 val entries와 baseline `training_signature.val_cache_files`를 대조한다. configured/materialized case 수, sample 수, case별 개수 및 `configured_but_not_materialized_cases`를 출력한다. 누락 이유를 확인하지 않고 donor eligibility 때문이라고 단정하지 않는다. raw score의 coverage는 invocation별로 완전/부분을 구분해 보존한다.

## 실제 로컬 CT/CUDA 학습 검사 — r2

현재 구현으로 RTX5070Ti 16GiB에서 8개의 **성공한 optimizer update**를 실행했다. train은 liver5/6, 별도 환자는 liver31이다. physical sample batch2, 동시 candidate graphs16, workers4, original 8 candidates/pool128, 원본 두 view와 complete loss를 사용한다. curriculum/view는 기존 DEBUG 대조의 epoch29로 고정했다. 이는 최종 cohort/epoch 설정과 분리된 DEBUG다.

| successful updates | train loss | train MRR | train top1 | train margin | 별도 환자 MRR | 별도 환자 margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.79423 | 0.6000 | 0.5 | −0.00763 | 0.3333 | −0.00244 |
| 2 | 3.77760 | 0.7500 | 0.5 | +0.00049 | 0.1667 | −0.04016 |
| 4 | 3.74607 | 0.7500 | 0.5 | +0.00916 | 0.1667 | −0.04749 |
| 8 | 3.67910 | 1.0000 | 1.0 | +0.02136 | 0.1667 | −0.08591 |

**학습 입력의 순위를 올리는 경로는 연결됐지만 일반화 개선은 확인하지 못했다.** 같은8-update 시점의 보존된 v1.4 DEBUG train MRR도1이며 margin은 +0.18915였다. A의 작은 train margin이나 한 환자의 rank 변화만으로 전체 학습 저하의 원인을 확정하지 않는다. 서버40epoch 성공 기준선과 로컬8-update를 직접 정확도 대조로 제출하지 않는다.

507/507 trainable parameter tensor에 gradient가 연결됐고 CNN·projection·fusion·L1·L2·readout·scalar scorer의 nonzero gradient와 실제 optimizer 변경을 확인했다. AMP skip은0이었다. 원본 입력/202 source/archive/기존 helper/상위 초기 tensor 및 설치 중 caller RNG 보존을 확인했다.

최대 GPU allocation은 약1.99GiB, process peak RSS는 약9.19GiB, 입력 재생을 포함한 전체 검사 wall은107.71초다. GPU compute update는 원시 `updates.jsonl`에 forward/backward/optimizer와 함께 기록한다. 입력이 RAM/GPU에 준비된 짧은 fixture 비용을 서버 epoch 전체로 환산하지 않는다. CNN batch와 원본 graph batching 경로를 사용했으며 sample별 개별 optimizer를 추가하지 않았다.

46개 외부 unit/contract 검사가 통과했다. 이 중 실제 archived constructor/serialization 검사는 내부11개 검사로 원본202 files, 상위 초기 tensor, 3종 marker 및 extra-state 거부와 own-checkpoint 재로딩을 확인했다. 완료된 last가 잘못되거나 손상된 best를 파일 존재 여부만 보고 통과하지 않도록 실제 best의 marker·완료 상태·signature·정적 metadata도 대조한다. CPU metadata/serialization 검사는 GPU 학습 증거와 분리한다. 실제 신경망 학습·gradient 검사는 위 CUDA r2다. 이전 임시 CPU 검사 프로세스의 종료 문제는 해당 직접 생성 PID만 확인 후 정리했으며 성공 결과에 포함하지 않았다.

원시 GPU 결과·curve·updates·attempts·execution contract·CPU unit audit·현재 source SHA와 검사 요약은 `validation/v15_half_a_20261004/`에 보존한다. CT fixture/대용량 checkpoint를 Git에 추가하지 않는다. `full_training=false`, `full_evaluation=false`, `quality_verified=false`를 유지한다.

## 서버 실행

물리GPU3의 기존 A6000 서버에서 다음 한 명령을 사용한다. Git fetch 후 branch의 revision을 고정해 실행한다. 최종 사용자 전달에는 push가 확인된 exact commit도 제공한다.

```bash
CP_GPU=3
CP_BASELINE=/home/aicompetition06/Medical/experiments/v1_m10_seed42_20261004
CP_EXPERIMENT=/home/aicompetition06/Medical/experiments/v15_m10_halfA_seed42_20261004

conda activate nnunet &&
cd /home/aicompetition06/Medical/HierCP-v1-47bdb58 &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach origin/codex/v222-server-r6 &&
python -u tools/run_v1_half_a.py \
  --gpu "$CP_GPU" \
  --baseline "$CP_BASELINE" \
  --experiment "$CP_EXPERIMENT" \
  --cuda-gib 40 --rss-gib 192
```

새 last checkpoint는 `$CP_EXPERIMENT/results/half_A/checkpoint_best.last.pt`다. margin10은 검증된 baseline에서 가져온다. 준비 작업과 기준선 재학습은 시작하지 않는다. full40epoch 및 full materialized validation 평가, CP128 production inference, Basic CP80/nnU-Net 비교는 로컬에서 실행하지 않았다. 이후 서버 곡선을 보고 이분 탐색의 다음 절반을 결정한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 명시적인 L0 교체 arm을 별도 구현했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 기존10mm cache 및 후보 계약을 유지했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 대조는 baseline의 측정 lock을 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
