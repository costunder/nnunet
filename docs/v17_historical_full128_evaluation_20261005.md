# V1 / A / B / C: 기존 BEST의 공통 전체후보 평가

## 실험 질문

기존 8후보 curriculum validation에서 높은 MRR을 얻은 V1/A/B/C가,
같은 held-out 21명에 있는 관측 종양 위치 P와 미관측 비교 위치 128U를
전부 점수화할 때도 순위를 분리하는지 확인한다. 재학습 없이 기존 BEST를
읽는다. 이 결과만으로 원래 8후보 학습과 전체후보 학습의 차이를 인과적으로
확정하지 않는다.

P는 실제 CT에서 관측된 적격 종양 anchor이고 U는 기존 미관측 비교 center다.
P를 특정 donor의 CP 정답으로, U를 CP 부적합 정답으로 바꾸지 않는다.
donor 배정, center, split, 입력 CT와 원본 annotation은 기존 native inventory에
결속한다. 각 환자의 모든 P와 정확히 128U를 사용한다. outer 26명은 사용하지 않는다.

## 실행 경로

`tools/server_v17_historical_full128.sh` →
`tools/evaluate_v17_historical_full128.py` →
각 arm의 별도 CUDA worker → 실제 L0 → 원래 arm의 upper/scorer →
`hiercp_v1x/transition_evaluation.py`의 동일 whole-P+128U metric.

원본 `hiercp` namespace와 bounded-scope adapter가 process-global이므로 arm별
모델은 별도 프로세스에서 실행한다. 선택한 단일 GPU에서 한 arm씩 평가하며,
arm 내부 후보 L0는 실제 disjoint graph batch 또는 native CNN batch로 계산한다.
V1/A/B는 같은 실제 두-view geometry를 한 번 준비해서 공유한다. L0 계산만
청크로 나누고 upper는 환자 전체 후보를 한 번에 함께 처리한다.

| Arm | 읽는 checkpoint | L0 | Upper / support |
| --- | --- | --- | --- |
| V1 | `results/v1.0/checkpoint_best.pt` | 원본 3-layer heterogeneous GAT + dense CNN, 실제 두 view 평균 | 원본 patient/prototype GAT 및 scalar score |
| A | `results/half_A/checkpoint_best.pt` | 원본 fixed48 ROI 위 LocalCNN + 원래 role/shell bridge, 두 view 평균 | V1 upper/scalar score 그대로 |
| B | `results/half_B/checkpoint_best.pt` | V1 L0 그대로 | B의 prompt/cluster/cosine upper + 원래 train anchor/curriculum support |
| C | `training/checkpoint_best_own.pt` | 실제 native-spacing LocalCNN, fused128 단일 view | C의 학습 시 upper + 원래 train anchor/curriculum support |

V1/A/B의 LAST는 완료 40epoch 및 BEST 선택의 결속을 확인하는 데만 사용한다.
LAST를 BEST로 바꾸거나 전체후보 결과를 보고 checkpoint를 다시 선택하지 않는다.
C도 원래 8후보 task의 BEST를 사용한다. optimizer를 생성하거나 불러오지 않는다.

기존 C run이 출력했던 공통 전체후보 평가는 native 관측 P/U support를 사용했다.
이번 C 평가는 **C가 학습한 anchor/curriculum support**를 선택 BEST 가중치로
다시 계산한다. 따라서 그 이전 C의 전체후보 숫자를 이번 결과와 동일한 support
조건의 결과라고 부르면 안 된다. 두 비교의 support 정책을 report에 명시한다.

## 공통 평가의 해석 범위

모든 arm은 같은 query P/128U, donor, 환자, metric을 사용한다. 각 모델이
학습한 연산과 support 의미는 유지하므로 `same_support_semantics=false`다.

V1/A의 원래 upper는 recipient lesion annotation을 입력으로 사용한다.
관측 종양을 삭제하거나 가짜 빈 recipient source를 넣어서 원래 모델을 바꾸지
않는다. 외부 donor는 donor CT/좌표계에 남기고 recipient의 실제 lesion 목록과
분리해서 상위 그래프를 만든다. 이 adapter는 원본 v1의 own-source inference와
완전히 같은 입력 task라고 주장하지 않는다.

`class_target_argument_passed=false`는 P/U label 인자를 scorer에 전달하지 않는다는
뜻이다. V1/A에 annotation으로부터 유래한 입력이 없다는 뜻은 아니다.
`V1_A_annotation_aware=true`와 `blind_recommendation_quality_verified=false`를
함께 기록한다. 결과는 **공통 관측 위치 benchmark**이며, annotation 없이 실제
CP를 추천하는 성능이나 nnU-Net Dice를 증명하지 않는다.

## 비용과 재사용

- 10mm bounded scope를 그대로 유지한다. 원본 실제 종양 footprint가 기준이고
  그 바깥 margin이 10mm다. 전체 종양 크기를 10mm cube로 잘라내지 않는다.
- 완성된 D `canonical_cache/index.json`을 지정하면 bytes/row/source/scope 결속을
  검사한 뒤 읽기만 한다. D의 lock, 출력, checkpoint와 소스 checkout을 수정하지 않는다.
- D 캐시를 지정하지 않으면 공통 **validation 전체**의 실제 원본 geometry만 새로
  준비한다. train 14,102개를 다시 준비하지 않는다. 캐시의 case/record 완료 receipt와
  SHA를 기록하고 이후 V1/A/B가 같은 캐시를 사용한다.
- B/C support는 각 학습 시 사용한 **전체 signed training curriculum cache**를
  선택 BEST의 L0로 재인코딩한다. B는 원래 support calibration을, C는 저장된
  support batch lock을 유지한다. support를 일부 환자로 임의 축소하지 않는다.
- 실제 GPU/VRAM/CPU affinity/cgroup/RAM을 확인한다. 후보 L0 physical batch
  `8 16 32`를 실제로 측정하고, 요청한 후보 중 메모리 한도 안에서 처리량이 높은
  것을 선택한다. 학습 batch는 바꾸지 않는다. 모델/노드/edge/CT 해상도/후보를
  OOM fallback으로 줄이지 않는다. 전부 안 맞으면 명시적인 오류를 남긴다.
- 그래프 준비, L0(H2D 포함), whole-case upper, 평가 wall time을 나누어 기록한다.
  calibration 시 입력 shape, node/edge 기반 선택 record, peak CUDA도 기록한다.

## 출력과 보존

새 평가 폴더에 `request.json`, arm별 `attempts/`, `report.json`, `complete.json`,
그리고 한 화면 요약 `summary.json`을 만든다. arm별 MRR/Hit@1/pair-win/softplus
loss와 정확한 denominator, 선택 epoch/checkpoint SHA, support 정책을 남긴다.

기존 실험 폴더와 ancestor/descendant인 출력 경로는 거부한다. 파일은 exclusive
create를 사용한다. 실행 source closure와 요청을 SHA로 결속하며 실행 도중 코드가
바뀌면 완료로 표시하지 않는다. 완료 report를 재사용할 때도 checkpoint bytes,
report SHA, request와 completion receipt를 다시 검사한다. 실패한 출력은 보존한다.

새 detached worktree에서 실행해야 현재 D 작업의 checkout이 바뀌지 않는다.
기존 실험의 lock을 지우거나 D 프로세스를 종료하는 명령을 포함하지 않는다.

## 검증 상태

`test_historical_*.py`의 63개 단위 검사가 통과했다. checkpoint 선택/arm/scope,
원래 support 의미, 진짜 donor/recipient 좌표계, 두 view의 batched ordering,
파일 보존과 전체후보 처리 계약을 검사했다. C BEST의 실제 5개 선택 지표와
worker 사이 native inventory 변경 거부도 포함한다.

실제 CT의 전체 P 8개+U 128개를 네 경로에서 실행한 CUDA smoke가 통과했다.
결과와 source/report SHA는
`validation/v17_historical_full128_cuda_smoke_20261005.json`에 기록했다.
모델 parameter 수는 V1 10,434,532 / A 5,127,124 / B 6,433,126 /
C 1,125,718개이며, 모델 내부 규격을 축소하지 않았다.
DEBUG는 실제 CT 한 validation 환자의 모든 P+128U를 사용하되 full21 평가와
분리한다. V1/A/B는 로컬 DEBUG 도구가 학습 최종 가중치를 저장하지 않았으므로
fresh seed42 full architecture를 사용한 **실행 검사**다. C는 실제 DEBUG run의
saved own BEST를 사용한다. DEBUG 순위 숫자를 서버 학습 품질 결과로 제출하지 않는다.

서버의 기존 trained V1/A/B/C BEST에 대한 전체21 공통 평가는 이 코드가 실행된
후에만 확인할 수 있다. 이 작업에서 장기 학습, CP paste 또는 nnU-Net 학습을
시작하지 않는다. `quality_verified=false`를 유지한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 실제 입력→forward→전체후보 metric에 연결된다. 평가 전용이므로 loss backward/optimizer update는 실행하지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
