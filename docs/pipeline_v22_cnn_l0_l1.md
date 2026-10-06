# v2.2 — CT-only CNN L0 · L1 현 구현 (의미 계약 미검증)

2026-09-22, r5 정정. **종양 내부 노드 제거와 data/label 설명은 [최신 정정](design_contract_corrections_v22_r5.md)이 우선한다.** 기존 “설계 확정” 표현을 철회하며, 아래는 계산 구조 설명이다. 원문 관계 학습 계약이 충족됐다는 뜻이 아니다. **L2는 복구되었다.** [현재 L2 상태와 학습 목표 감사](pipeline_v22_l2_restoration.md)를 함께 따른다. 사용자 요청 범위는 **CNN으로만 노드 특징을 만드는 L0와 기존 L1 구조 확정**이다. 여기서 고정은 구조 계약을 고정한다는 뜻이며 가중치를 freeze한다는 뜻이 아니다. 모델 파라미터는 학습 가능하다.

진입점은 `run_v22.py`, 설정은 `config/prompt_graph_v22.json`, 구현은 `hiercp_v22/`다. 저장 포맷은 `hiercp_prompt_graph_v22_surface_context_cnn_r5`다. 사용자용 버전명은 v2.2이고 r5는 이전 구현과의 artifact 호환을 차단하는 내부 revision이다.

## 실제 입력과 forward

```text
donor / candidate CT 각각 [B,1,48,48,48]
          ↓ 공유 3D CNN (12 → 24 → 32 채널)
위치별 32차원 학습 특징
          ↓ 기존 노드 위치에서 trilinear sampling
노드 종류별 Linear(32,128) → LayerNorm → SiLU
          ↓ 기존 공간 그래프, 3층 GNN / 4 heads / hidden 128
역할별·기존 shell별 pooling → 6개 표현 결합 → data node 128
          ↓ 환자별 독립 label node 16×128
L1: data → label, label → data attention을 2개 block에서 수행
          ↓ 환자별 label states · data–label relation
```

| 항목 | 확정 구성 |
|---|---|
| CNN 입력 | 정규화 CT **1채널**. 마스크·SDF·간 깊이를 CNN 채널로 넣지 않음 |
| CNN 노드 특징 | **32차원만**. 상대 좌표·평균·표준편차·곡률·법선·거리장 추가 없음 |
| 엣지 | `edge_index` 연결만. `edge_attr` 없음; GAT의 `lin_edge` 파라미터도 없음 |
| 공간 처리 | 기존 mm 반경·KD-tree·노드 구성·문맥 seed 및 hop closure 유지 |
| 구조 메타데이터 | 좌표는 그래프 구성/특징 표본 위치 계산에만 사용. 노드 종류와 기존 shell membership은 그래프/pooling 구조로 유지 |
| L1 | 기존 `Residual`, `PatientTaskGraph`의 AST가 보존본과 동일하나 T/F/U edge 조건화가 없으므로 원문 계약 구현 완료로 간주하지 않음. 환자별 독립 random 초기 label, 공유 학습 연산자, query가 support를 갱신하지 않는 규칙 유지 |
| 가중치 | 학습 가능. `freeze_weights=false`. data/label 의미 계약 미검증으로 `architecture_fixed=false` |

**CNN-only는 노드 특징의 생성 방식이다.** 그래프가 마스크·좌표·노드 종류와 무관하다는 뜻은 아니다. 기존 shell별 집계도 유지하므로 “공간적 사전 정보가 전혀 없다”고 설명하면 안 된다. 종양 내부 노드는 r5에서 생성·엣지·모델 경로까지 제거했다. CNN 영상 수용영역의 내부 CT 영향까지 제거한 것은 아니다.

CNN 깊이와 출력 크기, GNN 3층/128/4 heads, L1 2층/16 labels/128은 줄이지 않았다. 후보 128개, 전체 split 및 donor 정책, GNN 40 epoch·nnU-Net 250 epoch의 예정 규모도 변경하지 않았다. 현재 epoch 값의 보존은 전체 학습이 가능하거나 실행됐다는 의미가 아니다.

CT clipping과 기존 candidate footprint 내부의 erasure 전처리는 유지한다. CT-only가 raw HU를 전처리 없이 그대로 넣는다는 뜻은 아니다. source의 even-size patch anchor 정렬 수정은 이전 r2에서 이어받는다.

## 수작업 정보 재유입 방지

- canonical node table은 sampling grid·물리 좌표·shell membership만 보관하며 16/24차원 특징 테이블을 만들지 않는다.
- materialize 후 물리 좌표는 GPU 모델 입력에서 제외한다. grid는 CNN 표본 위치이고 학습 벡터에 concatenate하지 않는다.
- 모델은 `x`, `observed`, `pos`, `pos_mm` 및 `edge_attr`가 입력에 있으면 오류를 낸다. 5채널 CNN 입력도 거절한다.
- shared-source 저장에는 CT patch가 포함되며, CPU/GPU input accounting에도 CT tensor 비용을 반영한다.
- 기존 통계 descriptor, 복원 decoder, 통계 유사도 L2 teacher는 활성 모델에 없다.

## L2와 전체 학습의 경계 (r4 정정)

기존 L2는 L1 label 128차원만 받으므로 L0 특징을 바꾼다고 삭제할 이유가 없다. r3의 삭제는 구현 오류였으며, r4에서는 원래 L2 attention 2층과 환자 간 대응/관측 T 전달/관측 loss 경로를 복구했다. L0와 L1 구조는 그대로 유지했다.

통계 descriptor·복원·통계 teacher는 복구하지 않았다. 현재 T/U-only 데이터에서 관측 loss만 사용하면 균등 분포/전 후보 score=1인 해도 허용한다. 이는 전체 목적함수 문제이며 L2 구조가 없다는 뜻이 아니다. full training gate를 이 정확한 이유로 유지하고 새 loss를 임의로 정하지 않았다.

[복구 코드·검증·설계 문제의 상세](pipeline_v22_l2_restoration.md)를 따른다. 아래 r3 검사 수치는 당시 L0/L1 검사 기록이며, 최신 L0/L1/L2 검사는 `versions/v2.2/verification_l2_restore_20260922/`다.

## 보존과 검증

기존 r2 소스·설정·문서·code.txt·검증 코드는 `versions/v2.2/before_cnn_only_r3_20260922/source.zip` 및 SHA manifest에 보존했다. 이전 r1/r2 테스트와 verifier는 같은 디렉터리의 `historical_checks/`로 이동했다. 기존 실험 결과·의료 데이터는 변경하지 않았다. v1/v2.1 소스 무결성 guard도 유지한다.

새 검증 파일:

- `tests/test_v22_cnn_l0_l1_debug.py`: 입력 계약, CT 개입, batch 동등성, gradient/optimizer, L1 환자 분리, 원래 topology/CT 보존, cache 왕복, L2 실행 차단.
- `tools/verify_v22_cnn_l0_l1_debug.py`: 명시한 inner-train 환자 2명의 실제 CT로 전체 그래프 규칙을 적용하고 physical batch 2/4 CUDA 실행을 측정.

DEBUG의 미분 가능한 probe는 forward/backward 연결 확인용이다. **CP 위치의 정답이나 L2 학습 목표가 아니며 생산 학습에 연결하지 않는다.** 실제 CT를 사용한 probe 성공도 소형 종양 성능·학습 완료를 의미하지 않는다. 완료된 검사 수치와 경로는 동반 패치 노트의 최신 검증 절을 따른다.

## 2026-09-22 r3 당시 완료된 검증 (보존 기록)

- 정적 검사 25개 파일, 새 단위/회귀 DEBUG **8/8 통과**. v1/v2.1 runtime/config SHA guard 통과. L1 `Residual`/`PatientTaskGraph` AST가 보존본과 동일하다.
- 실제 inner-train `liver_1`, `liver_5`에서 T/U 각 1개, 총 4개 graph를 DEBUG로 검사했다. canonical node/edge와 sampled node/edge가 기존 규칙과 정확히 일치하고 CT는 기존 첫 채널과 일치했다. 원래 128개 후보 생성 규칙은 유지했다.
- CUDA BF16에서 CNN·GNN 3층·L1 양방향 2 block을 포함한 gradient가 모두 유한했고 실제 AdamW DEBUG update가 확인됐다. 환자 label table 2개를 둔 DEBUG 모델의 파라미터 **6,453,844개**가 전부 학습 가능하다. 전체 파라미터 수는 등록 환자 수에 따라 달라진다.
- DEBUG physical batch 2: 1.871 graph/s, peak allocated 264.5 MiB. batch 4: 2.689 graph/s, 486.2 MiB. warm-up 후 각각 3회 forward/backward. production L2/cohort support가 없는 L0/L1 probe이므로 최종 학습 batch/속도/VRAM으로 일반화하지 않는다.
- 검증 근거: `versions/v2.2/verification_cnn_only_20260922/implementation_checks.json`, `real_CT_cuda_debug.json`, `geometry_debug.json`. 원 실행 로그 데이터: `work/v22_cnn_r3_20260922/debug1/`.
- 전체 GNN/nnU-Net 학습·전체 평가·의료 성능 검증은 미실행. L2 학습 목표는 미확정이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 변경 전 파일은 ZIP/manifest로 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 입력 채널 변경은 명시적인 사용자 요청이다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토하고 DEBUG 2/4 측정 경로를 마련했다. 최종 학습 batch 선정과 구분한다.
- [x] GPU, CPU, RAM을 확인했다. RTX 5070 Ti 16GB, 논리 CPU 16개, RAM 약 68.6GB.
- [x] OOM을 이유로 모델을 축소하지 않았다. 실행 상태/측정은 최신 검증 기록을 따른다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] 생산 경로에 dummy, placeholder, random fallback을 사용하지 않았다. 합성 fixture는 DEBUG에만 있다.
- [x] L0/L1 핵심 모듈의 forward·gradient·optimizer 연결을 DEBUG에서 검사했다. 생산 loss 타당성 검증을 뜻하지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 전체 학습/평가는 미실행이다.
