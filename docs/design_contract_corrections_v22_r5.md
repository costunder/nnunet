# v2.2 r5 — 폐기 설정과 data/label 설명 정정

2026-09-22. 사용자는 종양 내부 노드가 폐기된 설정임을 명시했고, data/label 정의와 label 128차원의 근거를 요구했다. 이 지시는 “기존 코드 유지”보다 우선한다. r4까지 종양 내부 노드를 유지하고 설명한 것은 오류다.

## 폐기 설정의 실제 제거

- `tumor_interior` canonical 좌표/노드 생성을 제거했다. 제거한 노드를 이름만 바꾸거나 다른 종류에 합쳐 넣지 않았다.
- 내부-내부, 내부-표면, 표면-내부의 세 관계를 제거했다. 현재 공간 노드 종류는 5개, 엣지 관계 종류는 13개다.
- 내부 노드 전용 CNN 특징 표본, projection, GNN 관계 모듈, attention pooling을 제거했다.
- 기존 종양 surface/interior 결합 대신 surface pooling 결과만 종양 표현에 사용한다. 후속 MLP는 `128→256→128`로 기존 hidden 폭과 깊이를 유지하며, 입력 결합 부분만 제거했다.
- v2.2 전용 `schema.py`를 만들었다. 기존 공통 모델의 checkpoint helper가 구 6종 node 상수를 참조하던 부분도 v2.2의 실제 block node 종류를 쓰도록 분리했다.
- model forward 및 canonical cache materialization은 폐기 node가 다시 들어오면 오류를 낸다. 구 캐시의 format 문자열만 고쳐서 쓰지 못한다.
- CNN patch 입력은 CT 1채널 그대로다. **별도 종양 내부 그래프 노드를 제거한 것이며, CNN receptive field에서 내부 CT의 영향까지 제거한 것은 아니다.** 내부 영상까지 지우거나 새로운 마스킹 정책을 추가하지 않았다.
- 남은 surface/context/간 표면 노드와 radius 연결, CNN/GNN/L1/L2 깊이·hidden 폭, 데이터/후보/epoch 설정은 변경하지 않았다. 내부 노드 제거는 성능 편의를 위한 축소가 아니라 명시적인 폐기 지시 반영이다.
- v1/v2.1용 공통 스키마·설정은 보존한다. 공통 `config/train.json`과 `GraphBuildConfig`에 남은 legacy interior 값은 v2.2 생성/forward에서 읽지 않는다. 활성 v2.2 계약은 `topology_contract=surface_context_no_interior_r5`, `tumor_interior_nodes=false`다.

## data와 label: 원문과 현재 구현을 구분

사용자 제공 원문은 `C:/Users/user/.codex/attachments/c329cbfe-af55-4be8-8803-8d7dfc7e0bbc/pasted-text.txt`이며, 일반적인 3D 그래프 참고 원문은 `93fae564-a423-4001-9642-5c5ac0f94e4b/pasted-text.txt`다. 원문에 담긴 이전 assistant의 제안·예시를 사용자 확정 수치로 승격하지 않는다.

| 항목 | 원문에서 설명한 의미 | 현재 구현 및 불일치 |
|---|---|---|
| data node | 실제 local context를 L0로 인코딩한 표현 | donor와 target의 여섯 pooled 표현을 합친 pair embedding을 data 하나로 사용한다. 이 결합 단위는 구현 선택이며 사용자 정의 자체로 설명하면 안 된다. |
| task | 서로 다른 실제 data/task/patient의 국소 label space | 실제 환자 ID 단위로 묶는다. 모든 task 정의가 반드시 환자 ID여야 한다는 일반적 근거를 뜻하지 않는다. |
| label node | task마다 독립적으로 초기화하고 관계를 학습하는 local latent 기준 | 환자별 trainable table이다. 초기 label index에 암/정상/해부학 의미를 부여하지 않는다. |
| data-label 관계 | 원문에 T/F/U 관계와 관측 positive의 전달이 등장 | 현재 L1 attention은 embedding·owner·label만 받는다. T/F/U edge 정보가 attention에 들어가지 않는다. |
| T/F/U 단위 | 관계의 관측/미관측 상태를 구분하는 의도 | 현재는 data record마다 한 개 evidence 값을 두고 L2 전달에서 T를 집계한다. 이것을 data-label edge 관계 학습 구현 완료라고 부르면 안 된다. |

현재 코드는 어떤 관측 근거로 개별 `data i ↔ latent label k`의 관계를 정의하는지 충분히 구현하지 않았다. feature similarity attention과 뒤의 T 집계가 있다는 이유로 원문 구조와 일치한다고 검증된 것은 아니다. 새로운 관계 정답·loss·data 단위를 이번 수정에서 임의로 추가하지 않았다.

## 16개와 128차원의 실제 근거

- `K=16`: label node의 **개수**다. 기존 16-slot 용량 설정을 이어받았고, 사용자 지정/클래스 수/데이터 기반 선택으로 확인된 값이 아니다. 적정 개수를 비교한 실험 근거가 없다.
- `d=128`: label 하나를 표현하는 **벡터 길이**다. 공통 모델의 `hidden_dim=128`을 가져와 data projection과 L1/L2 attention의 입력 폭을 맞췄다.
- label table은 환자마다 `[16,128]`이다. 128가지 암 종류나 128개의 의미 있는 수작업 특징이 아니다.
- 현재 attention/projection 인터페이스가 128차원을 쓰는 것은 구현상 호환성 설명이다. 128이 연구적으로 최선이거나 충분하다는 증거가 아니다.
- label 원래 차원과 attention 내부 차원은 projection으로 다르게 구성할 수도 있다. 현재 128이 수학적으로 필수라고 주장하지 않는다.
- 이번에는 16/128을 다른 임의 값으로 바꾸지 않았다. config의 rationale과 `l1_contract_status`를 미검증 상태로 정정했다. `architecture_fixed`도 전체 사용자 설계 확정이라는 의미로 유지하지 않는다.

## 실행 상태와 보존

전체 학습은 재개하지 않았다. L2 forward는 유지하지만, 관측 loss만 사용했을 때의 균등분포 붕괴 가능성과 L1 관계 정의 불일치가 남아 있다. DEBUG forward/backward 성공은 이 설계 문제 해결을 뜻하지 않는다.

- 변경 전 r4: `versions/v2.2/before_retired_interior_removal_r5_20260922/`.
- 현재 포맷: `hiercp_prompt_graph_v22_surface_context_cnn_r5`.
- 검사: L0/L1/L2 회귀 16개. 구 schema 재유입 거절, 남은 노드/연결/CT 보존, 전체 gradient/optimizer 경로 등을 확인한다.
- 실제 CT DEBUG: `work/v22_no_interior_r5_20260922/debug1/`. 완료된 실행 결과는 `versions/v2.2/verification_no_interior_20260922/`의 JSON 기록을 따른다.

## 완료된 실제 CT DEBUG

- 실제 `liver_1`, `liver_5`에서 T/U 4graph. 유지한 노드 좌표·연결·CT가 이전과 일치하며, 내부 노드/관련 연결은 없다.
- CUDA BF16 warm-up 후 각3회: physical batch2 12,752nodes/728,947edges, 1.744 graph/s, peak allocated 262.9MiB; batch4 26,234nodes/1,459,433edges, 2.363 graph/s, 483.6MiB.
- 모든 parameter의 gradient가 존재하고 유한했으며 CNN/GNN/L1/L2 주요 그룹의 AdamW 갱신을 확인했다. 두 환자 label table 기준 6,076,116parameters다.
- 전체 학습, CP ranking/segmentation 평가 또는 label 의미 검증 결과가 아니다. 모델 체크포인트를 생성하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 변경 전 소스·문서를 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 폐기 노드의 입력과 전용 모듈만 제거했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 명시적으로 폐기한 내부 노드/엣지만 제거했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토했다. DEBUG batch 2/4와 최종 학습 batch 선정을 구분한다.
- [x] GPU, CPU, RAM을 확인했다. RTX 5070 Ti 16303 MiB, 논리 CPU 16, available RAM 약 49.6 GB.
- [x] OOM을 이유로 모델을 축소하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 생산 경로에 사용하지 않았다.
- [x] 남은 모듈의 forward·기존 관측 loss·gradient·optimizer 연결을 DEBUG에서 확인했다. 의미적 타당성 검증과 구별한다.
- [x] 실제 설정 변경과 미검증 설계 선택을 명확하게 보고했다.
- [x] smoke test와 전체 학습/평가를 구분했다. 전체 학습과 평가는 미실행이다.
