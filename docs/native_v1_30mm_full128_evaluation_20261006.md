# 원본 V1 30mm BEST의 전체 P+128U 평가

사용자가 선택한 대상은 `/home/aicompetition06/Medical/HierCP/work/full/model.pt`이다.
기존 기록의 BEST epoch30, 완료40epoch, 자체 curriculum 검증 MRR0.9869/top1 0.9804
모델을 현재의 전체 후보 평가에 사용한다. paired-fold 모델이나 새10mm 모델로 대체하지 않는다.

`tools/evaluate_native_v1_full128.py`는 실제 saved model_kwargs/graph_config/state_dict와
동일 publication의 `prototype.pt`를 읽는다. strict state load, fingerprint와 저장된 prototype
case IDs를 대조하고 전후 SHA256을 남긴다. optimizer를 만들거나 학습/CP/nnU-Net을 실행하지 않는다.

원본 운영 소스는 별도 immutable Git worktree의 `a818158a81fc09b9d11d2774d53fba152bb2b453`
이다. 전체 hiercp Python과 config를 기록하고 실제 import 경로를 검사한다. 이 소스는 원본
checkpoint schema와 연산을 실행하기 위한 명시적 기준이다. checkpoint가 학습 당시 source
SHA를 저장하지 않았으므로, strict state load 성공을 학습 당시 source byte 동일성 증거로
부르지 않는다. architecture tag 누락은 UNKNOWN으로 기록하며 임의의 revision을 주입하지 않는다.

## 평가 계약

- 현재 native inventory의 동일 inner_val21, 실제 P 전부, 환자별 U128, donor assignment를 유지한다.
- P는 관측 종양 위치, U는 미관측 비교 위치다. CP 적합/부적합 정답으로 바꾸지 않는다.
- 원본 ROI30mm, context28mm, shells4/12/28, fixed CNN48³, 전체 모델을 유지한다.
- 두 실제 original sampled views와 original heterogeneous GAT를 사용한다.
- L0 물리 배치 후보를 실제 CUDA에서 측정한다. OOM인 명시 후보만 거부하며 모델/범위/데이터를 바꾸지 않는다.
- L1/L2는 각 환자의 P+128U 전체를 한 그래프로 한 번 처리한다. L0 chunk마다 upper를 실행하지 않는다.
- graph 전처리는 inner_val과 필요한 donor만 수행한다. train14,102 전체를 새로 준비하지 않는다.
- 정답 target/component는 neural query에서 제거한다. 원본 V1의 recipient lesion annotation 입력은 유지한다.

원본 single-patient 학습 graph를 외부 donor 입력으로 옮길 때 두 환자의 region/liver를 각각
자기 좌표계에 둔다. donor tumor의 hosted_by는 실제 donor region에 연결된다. recipient
candidate/lesion은 실제 recipient region에 연결된다. 서로 다른 환자의 region ordinal을
같은 공간으로 간주하지 않는다. candidate↔tumor compatibility relation만 의미적 cross-frame
관계다. 학습된 연산과 relation schema는 보존하지만, 이것은 **paired donor/recipient 입력
adaptation**이며 original single-patient topology exact replay는 아니다.

기존 full105 prototype bank는 그대로 유지한다. 현재 validation21과 중첩하는 case를 제외하거나
bank를 다시 fit하지 않고, 실제 중첩 목록을 보고한다. 따라서 이 결과는 새 독립 held-out
성능이나 annotation-free 추천 품질을 증명하지 않는다. 자체8후보 검증과 같은 점수도 아니다.

## 출력과 실행

새 output에 request.json, resources.json, geometry/의 original canonical cache와 실제 두-view
node/edge 통계, calibration.json, report.json을 남긴다. 최종 terminal은 MRR, Hit@1, pair-win,
loss 및 report 절대 경로를 출력한다. 원본 checkpoint/source/prototype/inventory에 쓰지 않는다.
출력 경로가 이미 있으면 덮어쓰지 않고 실패한다. 실패 출력은 그대로 보존한다.

실행 옵션은 `python -B tools/evaluate_native_v1_full128.py --help`로 확인한다. 물리 GPU는
`--gpu` 하나로 선택하며 실제 현재 GPU UUID는 내부에서 해석한다.

## 검증 범위

`verify_native30_full128_cuda_debug.py`는 실제 CT의 train-only region으로 원본 prototype을 fit하고,
원본 전체 모델을 seed42로 새로 생성해 CUDA에서 별도 환자의 P+128U를 모두 처리한다.
이 경로는 UNTRAINED DEBUG이며 production checkpoint나 학습 완료 표시를 만들지 않는다.
서버의 실제 epoch30 trained checkpoint 전체21 결과는 사용자 서버 실행 후에만 확정할 수 있다.

로컬 검사의 수치와 완료 상태는 별도 validation evidence JSON에 기록한다. 성공한 smoke를
ranking 성능 개선으로 해석하지 않는다.

2026-10-06 최종 로컬 검증: regression68 PASS. RTX5070Ti actual CUDA DEBUG에서 liver_31의
P8+U128 전부, 272 genuine local graphs, 원본 전체10,050,543 parameters, 물리 배치2/4를
검사했다. 요청/source/inventory/prototype binding 전후 일치, 실제 whole-case upper1회와
1024 P×U metric 비교가 완료됐다. peak allocated CUDA1,009,979,392 bytes. 새 학습이나
trained checkpoint 로딩은 이 DEBUG에서 하지 않았다. 보고서:
`validation/native30_full128_DEBUG_20261006/report.json`.

완료된 동일-config region cache가 있는 경우 `--region-cache`로 읽기 전용 재사용할 수 있다.
case가 없거나 original metadata/integrity가 다르면 실패하며 그 cache에 새 case를 만들지 않는다.
최종 CUDA DEBUG는 먼저 실제 원본 연산으로 만든 두 환자의 region cache를 재사용했다.
서버 기본 명령은 새 평가 output에서 필요한 validation/donor region을 한 번 준비한다.

## 서버 legacy checkpoint 호환성 수정

첫 서버 실행은 실제 `model.pt`를 읽은 뒤, evaluation loader가 과거 파일에
`training_signature.seed=42/run_mode=production`을 필수로 요구해 중단됐다.
현재 선택한 원본 소스가 해당 형식을 저장한다는 사실만으로 과거 checkpoint도 그 형식을
가졌다고 요구한 조건이 잘못이었다. 이는 CUDA·graph·ranking 실패 결과가 아니다.

수정은 과거 기록과 현재 실행 설정을 분리한다. signature나 seed/run_mode가 저장되지
않았으면 training metadata를 UNKNOWN으로 기록한다. 실제로 저장된 seed는 그대로
남기며, 서로 모순되는 저장 seed나 명시적 debug/ablation 기록은 거부한다. 현재 평가의
seed42는 명시 원본 `config/train.json`에서 읽고 학습 당시 seed의 증거로 사용하지 않는다.
누락된 signature에 대한 직접 indexing도 제거했다. 원본 signature 형식 자체에 source_pad가
없으므로, 평가 padding4와 학습 당시 padding UNKNOWN을 구분한다.

가중치 strict load, 완료40epoch/BEST, 저장된 model/graph/CT clip, 실제 prototype fingerprint와
case IDs, 원본 ROI30/context28, 동일 전체21 P+128U 계약은 유지한다. 평가 시작 전에
`checkpoint_receipt.json`을 남기고 실제 저장 training seed와 evaluation seed를 출력한다.
원본 checkpoint나 실패 output은 수정하지 않으며 새 output에 재실행한다.

이번 수정의 회귀 검사는 74 PASS다. 누락 signature·부분 필드·현재 seed와 다른 과거 seed,
모순/잘못된 metadata, optional cache와 설정 보존을 UNIT으로 확인했다. 실제 trained 서버
weights는 로컬에 없으므로 해당 weights의 strict load와 전체21 평가 완료는 아직 미검증이다.
앞서 완료한 실제 CT/CUDA DEBUG 증거는 그대로 보존하며 이번 metadata 검사로 새 GPU
실행이나 ranking 품질이 검증됐다고 표시하지 않는다.

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
- [x] 핵심 모듈이 실제 forward에 연결된다. 평가 전용이며 loss/backward/optimizer update를 실행하지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
