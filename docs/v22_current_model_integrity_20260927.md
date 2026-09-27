# v2.2 C01 — 모든 resume 단계의 현재 모델 내용 검사

검토 기준: `96ab0531d9f088eb6cd73e53e2c8e03efd522ce5`. 사용자 첨부 텍스트 전체를 읽고 C01을 현재 코드에서 확인했다. 첨부 안의 sandbox ZIP/MD 링크는 실제 파일을 제공받지 않았으며 읽었다고 주장하지 않는다. 직전 B01–B03 수정과 정상 writer의 불변 snapshot은 유지됐다. C01은 **손상된 checkpoint admission 누락**이며 정상 trainer가 손상된 모델을 저장했다는 증거가 아니다.

## 수정 내용

`tools/v22_resume_integrity.py:seal_resume`가 이미 생성된 불변 CPU `payload['model']`의 전체 state_dict를 hash하여 `model_sha256`으로 저장한다. parameter뿐 아니라 persistent floating/integer buffer, 이름·shape·dtype·내용이 포함된다. 이 검사를 위해 GPU 모델을 추가 복사하지 않는다.

`tools/v22_artifacts.py`는 resume의 필수 필드에 `model_sha256`을 추가했고, `validate_integrity`는 **모든 phase**에서 현재 모델 내용과 저장된 hash를 비교한다. 검증은 `load_state_dict`/optimizer/RNG 복구 전에 수행된다. 현재 모델을 best·epoch reference·frozen teacher와 같게 만드는 변경은 없다. 별도의 final-memory best 검사는 유지한다.

계약은 `observed_rank_artifact_v5`다. v4에는 당시 current-model hash 증거가 없으므로 읽은 뒤 hash를 붙여 원래 저장 무결성을 검증했다고 처리하지 않는다. 이전 checkpoint/결과는 보존한다. core model/geometry/graph source는 그대로여서 **동일 최신 provenance/content-bound cache는 검증 후 재사용**한다. 오래된 geometry cache 재표시는 불가하다. runtime이 바뀌었으므로 calibration은 다시 측정한다.

모델 크기 5,550,806 parameters, graph 규칙, rank/CE/alignment 목적, donor 배정, seed42, CP80%, production physical-batch 정책·epoch·데이터 규모는 바꾸지 않았다. CPU hash의 추가 처리량 영향은 전체 규모에서 측정하지 않았으며 속도 개선을 주장하지 않는다. hash는 손상/불일치 검사이며 악의적으로 모든 hash를 다시 쓰는 경우의 전자서명이 아니다.

## 검증 기록

**회귀검사 86개 통과**: 직전 82개와 신규 test method 4개다. 내부 mutation 조합 수를 별도 test 수로 합산하지 않았다. 실제 CT process DEBUG에서 연속 실행 대 중간 재개의 다음 loss·544개 parameter tensor gradient·model·Adam, 마지막 Torch/CUDA/NumPy/Python RNG와 support가 bitwise 일치했다.

초기·optimization·refresh·validation·final-memory의 **다섯 정상 checkpoint를 각각 실제 재개**하여 최종 model/support가 기준 실행과 bitwise 일치했다. 별도의 final-memory 시작/부분/완료 prefix 경계 재개도 모두 일치했다. 다섯 단계 × finite parameter 변경/hash 누락 **10개 C01 반례**를 실제 GPU가 만든 checkpoint의 CPU admission에서 거부했다. 기존 B01–B03 계열 손상 10종도 다시 거부했다. 이것은 손상 checkpoint로 GPU update를 실행했다는 뜻이 아니다.

자원은 RTX 5070 Ti 16GB, CPU 8 physical/16 logical이다. loader 0/2/4/8을 실제 측정했고 warm 처리량 기준 4를 선택했다. DEBUG에서 physical=effective batch 2, accumulation 1을 유지했다. 해당 DEBUG calibration은 2.360 graphs/s, peak allocated 408,728,064 bytes였다. 이것을 전체 support/production physical batch의 메모리 근거로 사용하지 않는다. 실행별 실제 설정·RAM·node/edge 통계는 원문 로그와 JSON에 남겼다.

검증 원문은 `work/v22_model_integrity_20260927_DEBUG/`, 소스/출력 hash와 검사 범위는 `validation/v222_r6/current_model_integrity_20260927_DEBUG.json`에 기록한다. 실제 CT 검사는 명시적 DEBUG 8 train / 2 validation / support 8, physical=effective batch 2, 1epoch/4 updates로 production과 분리한다. 전체 architecture와 각 graph 생성 규칙을 유지한다.

새 CPU 검사는 다섯 phase 각각의 정상 모델을 허용하고, parameter·floating buffer·integer buffer의 유한값 변경, hash 누락·불일치를 거부한다. 실제 AsyncSaver 저장 후 live parameter/buffer 변경이 이미 만든 snapshot/hash에 유입되지 않는 것도 검사한다. final-memory current-model hash를 다시 계산해도 별도 best 결속 검사가 남는지 검사한다.

실제 GPU 재개 검사는 기존 `verify_v22_integrity_process_debug.py`에 명시적 `--all-resume-phases`를 추가했다. 초기·optimization·refresh·validation·final-memory checkpoint를 저장한 뒤 각각 admission 및 실제 재개 결과를 비교한다. 각 단계의 finite parameter 변경·hash 누락을 기존 hash를 다시 계산하지 않고 거부하는지 검사한다. buffer mutation은 buffer가 있는 CPU fixture로 별도 확인한다.

추가로 `verify_v22_current_model_admission_debug.py`는 실제 첫 Adam update가 끝난 `step=1/next_batch=1` checkpoint에서 정상 admission과 parameter 변경/hash 누락 2종 거부를 확인했다. 다섯 phase 검사의 optimization 시작 checkpoint와 중간 checkpoint를 구분하기 위한 보충 검사이며 전체 회귀 test method 수에는 합산하지 않는다.

같은 episode 검사는 `verify_v22_same_episode_debug.py`의 실제 CT batch 재사용 probe다. teacher 재생성을 금지하고 동일 saved plan으로 loss·L0/L1/L2 gradient·model·Adam·RNG를 비교하여 bitwise 동일성을 확인했다. 이것을 서로 다른 모든 batch를 가진 전체 episode 검증이라고 표현하지 않는다. earlier-best 다중 epoch 조건도 CPU fixture이며 실제 CT 1epoch DEBUG와 구분한다. 외부 best 읽기를 금지하고 rolling 파일 하나만 복사한 portable resume도 최종 모델/support가 bitwise 일치했다.

## 남은 범위

G3 전체 support·전체 physical batch·worst-batch 검증, G4 native bank/RPC/adapter/trainer와 nnU-Net 최종 입력 연결, G5 CP 대조 성능은 미완료다. pair estimator 비균등 가중, CE/종양 본체 shortcut, fixed-donor 추천 능력, GT 간 union 의존성도 이번 저장 무결성 수정과 별개다. 이전 raw paste 결과를 이번 실행 결과로 합산하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. DEBUG는 별도로 표시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번 측정은 DEBUG다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. 이번 실행 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 production에 사용하지 않았다. CPU fixture는 별도 표시했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 전체 모델의 gradient/Adam 비교로 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
