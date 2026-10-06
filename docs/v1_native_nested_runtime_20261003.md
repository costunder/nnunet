# v1 원본 / strict-nested 실제 실행 연결

이번 변경의 목적은 이미 검증한 strict-nested sampler를 실제 v1 학습 실행기에 연결하는 것이다. 이전의 sampler와 104/416 측정 결과를 보존한다. 경로 손실을 0으로 만들기 위해 relay를 더 추가하거나 seed 예산을 자동 조절하지 않는다.

## 실제 실행 경로

```text
run_v1x_experiment.py init --local-sampling native 또는 strict_nested
  → 버전별 고정 source snapshot + sampling JSON + manifest

run ... --stage v1.0 --target train
  → sampling_entry: 원본 pipeline + worker 측정 RNG 복원
  → 원본 HierarchicalCacheDataset
  → materialize_sample_views (epoch별 두 view)
  → native build_local_view
      ├ native: 그대로 사용
      └ strict_nested: 이 native view 내부의 노드만 선택
                      모든 induced edge/속성/순서/중복 보존
  → disjoint-union PyG batch
  → 원본 CNN / 3층 GAT / patient / population / scalar head
  → 원본 loss → backward → AdamW → 원본 고정 validation

run ... --stage v1.0 --target generate
  → 같은 sampling_entry / 같은 sampling contract
  → 원본 build_inference_sample
  → 원본 collate → materialize_sample_views
  → 같은 native 또는 strict_nested sampler
  → 원본 score_inference_chunked → 원본 전체 mask 검사/paste
```

별도 DEBUG에서만 그래프를 줄이는 상태를 끝내고 train, validation, generation의 공통 sampler를 연결했다. 실제 generation 명령의 전체 실행과 128개 후보 전체 검증은 아래 smoke와 구분해야 한다.

## 고정되는 항목과 명시적으로 달라지는 항목

- 원본 v1의 source-anchor 정답, 128개 pool에서 8개 후보를 쓰는 curriculum, 난이도와 corruption, 두 view, train-only prototype, loss, 모델 크기, 40 epochs를 유지한다.
- 같은 84 train / 21 validation / 제외 26 case를 사용한다. seed42는 학습 seed이며, 입력 분할 파일에 없는 분할 생성 seed를 새 사실로 주장하지 않는다.
- nested는 원본과 별도 실험 폴더를 쓴다. 원본의 canonical cache와 prototype을 공유하고 원본의 측정된 physical batch / worker lock을 적용한다. 독립적으로 작은 batch를 다시 고르지 않는다.
- profile은 필수 설정이다. 여섯 역할 예산을 전부 지정해야 한다. 104와 416은 기존 DEBUG에서 사용한 예산의 합이며 최종 노드/edge 상한도 최적값도 아니다.
- 선택 정책은 FPS, 역할/shell 배정, witness와 weak-path relay이다. 따라서 ‘노드 수 숫자 하나의 무편향 대조’가 아니라 ‘동일 native view를 명시적인 정책으로 줄인 대조’이다.
- Basic CP, nnU-Net, v2.2의 observed P/U 정의를 바꾸지 않는다. v1에는 v2.2식 support refresh가 없으며 비용표에 추가하지 않는다.

## 원본 source와 체크포인트 보호

`versions/v1/pipeline_v1_source.zip`과 원본 모듈은 수정하지 않는다. 새 실험 폴더에 원본을 검증 후 해제한다. native의 v1.0 core는 원본 bytes 그대로이다. strict-nested snapshot의 sample/model/contracts에만 명세된 짧은 hook을 덧붙이고, 필요한 sampler 파일을 함께 고정한다. tensor와 pipeline의 원본 bytes는 유지한다.

snapshot hook은 spawned/persistent DataLoader worker에서도 실행된다. 부모 프로세스에서만 sampler를 교체하는 방식으로 끝내지 않는다. 원본 graph_config, source 파일 hash, profile, case/sample/candidate/view seed·epoch가 맞아야 이미 만든 view를 재사용할 수 있다. native 또는 다른 profile의 pre-materialized graph를 nested에 조용히 넣으면 오류로 종료한다.

strict-nested 모델은 architecture suffix와 32-byte 비학습 buffer에 sampling contract digest를 저장한다. 학습 가능한 parameter 수와 연산은 바꾸지 않는다. 원본 loader는 nested checkpoint를 거부하고, nested loader와 직접 `load_state_dict`도 다른 sampler의 상태를 거부한다. stage와 geometry 검사는 원본 validator에 계속 위임한다. 실험 폴더, manifest, launch contract, run contract도 정확히 맞아야 resume가 허용된다.

같은 seed만으로 실행이 동일하다고 가정하지 않았다. 원본의 자동 worker 측정이 Torch RNG를 소비하고 고정 worker 실행은 이 측정을 건너뛰는 차이가 있었다. 새 명시적 native/nested 실행기 양쪽에서 이 측정 전후 Python/NumPy/Torch CPU/CUDA RNG를 복원한다. 기존 실험을 이 새 실행 계약으로 몰래 이어받지 않는다.

## 로컬 검증 범위

`tools/verify_v1_sampling_runtime_debug.py`는 기존 실제 CT fixture를 사용한다. 모델은 양쪽 모두 원본의 10,434,532 parameters이다. physical 2 samples × 8 candidates와 두 view를 유지하며 각 branch 1 update만 수행하는 전용 DEBUG이다. 모델 깊이/너비/patch 해상도를 줄이지 않는다.

검사 항목은 실제 원본 Dataset의 spawned worker 2개, 학습과 validation의 sampler 적용, 전체 loss의 core-module gradient, optimizer 변화, canonical inference collation, chunked/ordinary scoring 수치 대조, 8개 중심에서 전체 원본 source mask를 사용한 paste, 실제 GPU 상태의 checkpoint 저장/동일 계약 reload이다.

첫 검사에서는 도구가 smoke helper를 import하며 ROOT를 sys.path 맨 앞으로 올려 spawned worker가 현재 코드를 읽는 오류를 발견했다. `work/v1_sampling_runtime_CUDA_DEBUG_20261003`의 부분 결과는 보존했다. 도구에서 worker spawn 전 snapshot 우선순위를 복원하고 새 폴더 `work/v1_sampling_runtime_CUDA_DEBUG_20261003_r2`에서 양쪽을 다시 검사했다. 실패한 첫 실행을 통과 수에 포함하지 않는다.

새 `work/v1_sampling_runtime_CUDA_DEBUG_20261003_r2/report.json`의 검사 결과는 PASS이다. 같은 초기 neural state hash를 확인했다. 한 학습 batch의 16 graphs/view에서 native의 평균 N/E는 13,138.88 / 889,816.56, nested는 376.69 / 3,166.56이었다. 같은 original model의 1 update compute는 31.303초 / 2.611초, peak allocated VRAM은 6.176 / 0.364 GiB였다. 모든 핵심 gradient 그룹이 양수·유한값이고 optimizer 변경, validation, checkpoint reload, 양쪽 596 voxel 전체 mask paste를 확인했다. Chunked/ordinary inference의 최대 score 차이는 각각 0.000244 / 0.000122로 기록했다.

Loader 최초 batch, H2D, forward/backward/optimizer와 gradient 진단의 시간을 구분한다. 최초 update의 cold-start 비용을 steady-state 또는 전체 epoch 비용이라고 부르지 않는다. 전체 validation, 저장, sampler를 모두 포함한 서버 epoch 시간은 아직 측정하지 않았다. 이번 수치는 이전 48-view 평균과 집계 대상이 다르며 profile104/416의 우열을 정하지 않는다.

8개 DEBUG 후보 paste는 128개 production 후보/transform 전체 또는 nnU-Net 최종 입력의 검증이 아니다. 한 번 optimizer가 실행됐다는 것으로 21개 case의 순위 품질 유지나 과거 v1 MRR=1 재현을 승인하지 않는다.

## 서버 비교를 준비하는 명령의 의미

init과 plan은 학습을 시작하지 않는다. 다음은 profile104를 **명시적으로 선택하는 예시**이며 권장/최적 기본값을 뜻하지 않는다. 실제 비교에는 원본과 profile 하나만 선택한다.

```bash
python tools/run_v1x_experiment.py init \
  --experiment /home/aicompetition06/Medical/experiments/v1_native_seed42 \
  --medical-root /home/aicompetition06/Medical \
  --local-sampling native

python tools/run_v1x_experiment.py init \
  --experiment /home/aicompetition06/Medical/experiments/v1_nested104_seed42 \
  --medical-root /home/aicompetition06/Medical \
  --local-sampling strict_nested \
  --role-seeds 16 8 24 16 24 16 \
  --reference-experiment /home/aicompetition06/Medical/experiments/v1_native_seed42

python tools/run_v1x_experiment.py plan \
  --experiment /home/aicompetition06/Medical/experiments/v1_nested104_seed42 \
  --stage v1.0 --target train
```

prepare는 native reference에서 한 번 수행하고 두 arm이 공유한다. 새 계약은 기존 캐시를 자동 채택하거나 재생성하지 않는다. 학습은 별도의 `run --target train --gpu <현재 환경에서 보이는 번호>` 명령으로만 시작한다. native의 measured calibration이 먼저 필요하다. nested가 더 작다는 이유로 원본과 batch를 다르게 설정하지 않는다. generation은 전체 40-epoch 완료/체크포인트 결속을 확인한 뒤에만 실행한다.

현재 연구 방향의 버전 표기는 v1.x 그래프 sampling 대조이다. 기존 누적 stage v1.1/v1.2/v1.3이 nested를 뜻하지 않는다. 이번 첫 그래프 대조는 두 arm 모두 모델 stage v1.0을 사용하며, 차이는 별도 sampling contract로 기록한다.

## 남은 품질 판단

`tools/evaluate_v1_sampling_pair.py`를 추가했다. 원본과 선택한 nested 실험의 전체 40-epoch 완료, 동일 native reference, 입력/모델/sampling/checkpoint 계약을 확인한 뒤에만 GPU 평가를 시작한다. native에서 측정된 동일 physical batch와 worker 설정을 사용하며 전체 21-case validation cache를 빠짐없이 소비한다. 체크포인트와 원본 cache의 hash를 전후 대조한다. optimizer나 production checkpoint는 생성하지 않는다.

MRR/top1의 동점 처리는 원본처럼 `negative >= positive`이다. sample별 지표를 먼저 case별로 평균하고 case 단위 paired bootstrap으로 nested-minus-native 차이와 신뢰구간을 계산한다. 후보나 view를 독립 환자로 취급하지 않는다. Bootstrap 반복 수와 confidence는 필수 입력이며 품질 허용 오차는 임의로 정하지 않는다. 통계는 validation에서 고른 best 모델에 조건부인 비교이며 독립 test나 CP 효용 검증이 아니다.

통계/완료 상태 검사 8개와 RTX 5070 Ti의 float32/float16 GPU 지표 대조 1개를 통과했다. GPU 대조는 보존 원본 ZIP의 실제 `ranking_metric_sums`를 사용한 명시적 score UNIT 검사이며 CT 정확도 평가를 대신하지 않는다. 전체 21-case 평가 도구는 구현됐지만 실행하지 않았다.

이번 실행 연결 변경의 서로 다른 검사 85개를 통과했다: runner/contract 44개, worker/checkpoint/RNG 13개, 결과 결속 13개, paired 통계/완료 상태 8개, 전달 자료 결속 6개, CUDA 지표 1개이다. 이 중 84개는 계약·통계·파일 검사이며 별도 실제 CT/CUDA 두 branch smoke와 구분한다. 이전 sampler 검사 수나 외부 GPT 검사 수를 합산하지 않았다. 원시 report와 검사 범위는 `validation/v1x_progressive_20261003/runtime_verification.json`에 결속했다.

원본 대비 profile 하나의 전체 84/21 학습과 평가를 수행한 뒤 21개 case별 MRR/top1/margin 차이를 본다. 두 view나 후보를 별도 환자처럼 세어 불확실성을 줄이지 않는다. 허용 품질 차이가 아직 정해지지 않았으므로 통과 기준의 임의 기본값을 만들지 않는다.

다음은 이번에 자동 실행하지 않는다: 전체 40 epochs, 전체 21-case paired 평가, 128-candidate production generation, nnU-Net/Basic CP 효과 비교. 실제 학습 비용은 sampler/loader/H2D/optimization/validation/save로 기록하고 v1에 없는 support refresh를 포함하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자가 요청한 명시적 sampling 대조만 분리했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 경로에서는 OOM을 숨기는 fallback이 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되는 경로를 구현하고 별도 실제 CT/GPU smoke로 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
