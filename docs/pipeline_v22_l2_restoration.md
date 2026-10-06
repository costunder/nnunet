# v2.2 — L2 삭제 오류 복구와 학습 목표 감사 (내부 r4)

2026-09-22. **r5 정정: 종양 내부 노드는 제거됐으며 L1 data/label 의미와 16/128 설정은 [최신 정정 문서](design_contract_corrections_v22_r5.md)를 따른다.** 아래 r4 L2 복구 이력과 당시 수치는 보존 기록이다. L0 입력을 CNN 특징만으로 바꾸라는 요구는 L2 삭제를 의미하지 않는다. r3에서 통계 기반 보조 loss 제거를 이유로 L2 모듈·증거 전달 경로까지 삭제한 것은 잘못된 범위 변경이었다. 사용자가 L2를 없애거나 역할을 다시 정하라고 요청한 적은 없다. 이 문서는 이전 r3의 “L2 미정/미구현” 설명을 정정한다.

## 원래 L2의 실제 설계

1. L1이 환자마다 만든 잠정 label 표현 `[환자 수, 16, 128]`을 입력받는다. CNN 채널이나 수작업 노드 특징을 직접 입력받지 않는다.
2. 전체 label을 펼치고 같은 환자의 label끼리 직접 attention하지 못하도록 마스크를 건다. 기존 2층 multi-head attention(4 heads, hidden 128)과 residual/FFN을 적용한다.
3. 정렬된 표현의 cosine similarity와 temperature 0.2로 환자 쌍의 label 대응 행렬 `[P,P,16,16]`을 만든다. 행 softmax이며 일대일 정합을 강제하는 행렬은 아니다.
4. 다른 환자에서 실제 종양 위치로 관측한 T의 data–label 분포를 이 대응 행렬로 전달한다. 자기 환자는 증거 전달에서 제외한다. U는 미관측이며 F로 만들지 않는다.
5. 후보의 L1 label 분포와 전달받은 T 분포의 cosine similarity를 위치 적합도 점수로 사용한다. 악성 확률이나 확정 정답이 아니다.

따라서 CNN 기반 L0도 L1의 128차원 표현을 생성하면 기존 L2에 그대로 연결할 수 있다. 특징 추출 방식 변경은 L2 삭제의 근거가 아니다.

## 잘못 결합했던 부분

이전 모델은 위 정렬 모듈과 별개로, L1 assignment로 수작업 descriptor를 집계한 `descriptor_basis`를 만들었다. 그 통계 벡터의 cosine similarity를 soft target으로 삼아 L2 대응 행렬을 학습시켰다. 별도 decoder에는 통계 벡터 복원을 학습시켰다.

- 통계 유사도는 생물학적으로 올바른 label 대응의 정답이 아니다.
- 통계 기반 보조 loss와 label 정렬 모듈은 서로 다른 구성요소다.
- 통계 특징을 제거할 때 수정해야 하는 부분은 descriptor·복원·통계 teacher 의존 경로다. L1 label을 받는 정렬 모듈까지 제거할 이유는 없다.

## r4에서 복구한 범위

`PromptGraphModel`은 `L0L1Model`을 확장해 기존 `CrossPatientAlignment`를 다시 연결한다. `Residual`, `PatientTaskGraph`, `CrossPatientAlignment`, `transport_positive`, `observed_evidence_loss`는 보존한 r2 원본과 AST가 동일하다.

```text
CT 1채널 → CNN 노드 특징 32 → 기존 공간 GNN → data 표현 128
→ 기존 환자별 L1 → 환자별 잠정 label 16×128
→ 기존 L2 attention 2층 → 환자 간 label 대응
→ 다른 환자의 관측 T 전달 → 후보 적합도 score
→ 기존 observed-transfer loss → backward → optimizer
```

- L0와 L1의 구조·너비·깊이는 바꾸지 않았다. `L0L1Model`은 독립 검사에 남고 실제 전체 모델 이름 `PromptGraphModel`은 L2를 포함한다.
- `prepare_task_state`, `forward_tasks`, 전체 `forward`와 관측 loss가 작동한다. loss는 종전 `observed_transfer` 항목과 가중치 1.0을 보존했다.
- support memory와 recipient context 결합을 복구했다. recipient context는 U만 추가하며 자신의 T를 donor 증거로 쓰지 않는다.
- CT/CNN/GNN에서 L2까지 미분 경로가 이어진다. BF16의 near-one 반올림을 줄이기 위해 최종 cosine score는 FP32로 계산한다.
- 통계 decoder, descriptor basis, 통계 유사도 teacher를 다시 넣거나 새로운 teacher를 만들지 않았다.
- 기존 r3 소스·문서·code.txt는 `versions/v2.2/before_l2_restore_r4_20260922/`에 보존했다. r4 포맷은 `hiercp_prompt_graph_v22_ct_cnn_l0_l1_l2_r4`다. 구 체크포인트를 재명명하지 않는다.

## 확인된 학습 목표의 결함

현재 준비 코드는 T=1과 U=-1을 생성하고, 기하적으로 불가능한 F=0은 후보에서 제외한다. 기존 관측 loss는 이용 가능한 T/F에서만 `(score-target)^2`를 계산하므로 U에는 직접적인 정답 손실이 없다.

모든 label assignment와 대응 행렬이 균등분포이면, 전달된 T 분포와 후보 분포도 균등하다. 둘의 cosine score는 모든 위치에서 1이고, T-only 관측 loss는 0이다. **구별 능력이 없는 해가 목적함수를 만족하는 반례**다. 실제 학습에서 이 붕괴가 발생했다고 주장하는 것은 아니다. 이 반례를 회귀 검사로 명시했다.

따라서 통계 보조 loss 두 항목을 지우고 observed-transfer 항목만으로 전체 학습을 시작하지 않는다. 기존 전체 목적함수의 대체 설계는 이번 복구 범위에서 임의 확정하지 않았다. L2 forward·기존 loss 연결의 복구와 전체 목적함수의 타당성은 구분한다.

`train`과 생산 checkpoint 기반 scorer의 차단 이유는 이제 “L2 없음”이 아니라 **전체 학습 목적함수 미완성/관측 loss 단독 붕괴 가능성**이다. 전체 학습·CP ranking 품질·segmentation 평가는 미실행이며 검증되지 않았다.

## 검증과 증거

- 합성 DEBUG 회귀 15개: 기존 L0/L1 8개, L2 복구 7개. 입력 계약, 원본 AST, 실제 loss의 전 계층 gradient/optimizer, score 영향, 환자 순서 변환, 자기 증거 제외, U/F 구분, recipient 격리, 균등분포 반례를 확인한다.
- 실제 CT CUDA DEBUG는 `tools/verify_v22_l2_restoration_debug.py`에서 별도 실행한다. inner-train 2명, 실제 T와 U 각 1개, 전체 크기 그래프 규칙, physical batch 2/4이며 최종 실험 dataset 또는 epoch를 줄이지 않는다.
- 검증 기록: `versions/v2.2/verification_l2_restore_20260922/`. 실제 CT 원 실행 기록은 `work/v22_l2_restore_r4_20260922/debug1/`이다. 실행 성공과 세부 수치는 해당 JSON을 따른다.

## 완료된 실제 CT CUDA DEBUG

- 실제 inner-train `liver_1`, `liver_5` T/U 4개 graph. 기존 canonical/sample node·edge 및 CT 채널 일치 확인.
- RTX 5070 Ti, CUDA BF16. warm-up 뒤 3회 forward/backward: physical batch2 **1.845 graph/s**, peak allocated **267.8 MiB**; batch4 **2.626 graph/s**, **489.5 MiB**.
- 두 환자 label table을 둔 DEBUG 모델 **6,883,412개 파라미터**, 모두 학습 가능. 등록 환자 수에 따라 파라미터 수가 달라진다.
- 모든 파라미터의 gradient가 존재하고 유한했다. CNN, GNN3층, L1양방향2block, L2attention/residual2층의 실제 AdamW 갱신을 확인했다.
- 이는 전체 cohort나 production batch 자원 측정, CP 순위 타당성, 학습 완료 또는 의료 성능 검증이 아니다. 체크포인트를 생성하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 수정 전 보존본을 만들었다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 기존 L2 2층/128/4 heads를 복구했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. DEBUG 범위는 별도 명시했다.
- [x] physical batch size와 병렬화 가능성을 검토했다. DEBUG에서 2/4를 비교하며 최종 batch 선정으로 보고하지 않는다.
- [x] GPU, CPU, RAM을 확인했다. RTX 5070 Ti 16303 MiB, CPU 16 logical, 사용 가능한 RAM 약 49.7 GB.
- [x] OOM을 이유로 모델을 축소하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 생산 경로에 사용하지 않았다.
- [x] L0/L1/L2 forward·기존 관측 loss·gradient·optimizer 연결을 DEBUG에서 검사했다. 전체 학습 목표 타당성 검증과 구별한다.
- [x] 실제 실행 설정과 변경 사항을 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분했다. 전체 학습/평가는 미실행이다.
