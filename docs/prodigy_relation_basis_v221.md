# v2.21 작업 중 — PRODIGY의 관계 정답 출처 확인

2026-09-22. 사용자 요청은 관계 T/F/U와 기하 조건을 분리해 v2.21로 관리하는 것이다. 관계 부여 기준에 대한 질문에는 “원래 PRODIGY에서는 어떻게 했는가”라는 답변이 도착했다. 먼저 원문을 확인하며 의료 task 정의를 임의로 확정하지 않는다.

## 원문에서 확인한 내용

- Label node는 task의 각 클래스에 대응한다. Gaussian 난수 초기화는 표현 벡터의 초기값이며, 클래스 정의를 대신하지 않는다.
- Support는 알려진 정답 클래스 연결이 T, 다른 클래스 연결이 F다. Query는 정답을 입력하지 않고 label→query 방향으로 정보를 받는다. 여기서 U는 미관측 관계를 표현하는 우리 표기이며 원문의 세 번째 예측 클래스가 아니다.
- 클래스 정답이 없는 사전학습에서는 Neighbor Matching으로 표본 노드의 그래프 이웃을 기반으로 임시 분류 task를 만든다. 임시 class 정의와 support/query 정답이 먼저 존재한다.
- 예제/query 및 T/F edge 속성은 task attention에 들어간다. 최종 query 분류에는 cosine logits와 cross-entropy를 사용한다. 본문 설명과 달리 일부 KG 설정은 positive/query edge만 message passing에 사용한다.
- Appendix D의 제시 embedding dimension은 256이다. 현재 프로젝트의 16개/128차원을 정당화하는 결과가 아니다.

출처: [논문 §2.3–3.2](https://arxiv.org/html/2305.12600), [최종 NeurIPS 논문 Appendix C/D](https://cs.stanford.edu/~jure/pubs/prodigy-neurips23.pdf), [공식 edge-conditioned attention 코드](https://github.com/snap-stanford/prodigy/blob/main/models/metaGNN.py).

## 현재 프로젝트와의 차이

현재 v2.2의 환자별 자유 latent label16개는 위 클래스 기반 task를 재현한 것이 아니다. 종양 관측 여부와 기하 적격 여부만으로 각 latent label의 T/F가 생기지 않는다. 이 차이를 먼저 설명하지 않고 사용자에게 관계 정답을 요구했던 순서도 정정한다. Neighbor Matching을 CT에 그대로 적용해도 임시 분류 관계가 곧 종양 발생 가능성 정답이 되는 것은 아니다.

## v2.21에서 현재 구현된 범위

- 기존 v2.2 소스/설정/CLI는 `versions/v2.2/before_v221_20260922/`에 보존했다. 별도 `hiercp_v221/`, `config/prompt_graph_v221.json`, `run_v221.py`를 생성했다.
- `relations.from_episode_classes`는 명시된 클래스 ID로 support T/F와 query U를 만들고 support 양방향/query label→data 단방향 및 edge 속성을 생성한다. 기하 적격성 입력으로 F를 만들지 않는다.
- 합성 DEBUG 3개 통과: T/F/U 분리, query 역방향 차단, 클래스 순열 일관성, 명시적 class 정의 요구. 임의의 의료 관계 정답을 생성하지 않았다.
- `run_v221.py check`에서 소스/설정/환경을 확인했다. 모델 완성·학습 성공 검사가 아니다.
- 현재 `version_status=in_progress`, `implementation_scope=relation_contract_only`다. 복사한 구 모델은 새 의미의 모델처럼 실행되지 않도록 명시적으로 차단했다. 실제 L1/L2 통합, 의료 task/class 정의, 전체 학습·평가는 완료되지 않았다.
- 새로운 label 개수/차원·sigmoid 목적함수·통계 teacher를 채택하지 않았다. 보존한 기준 값과 설계 확정은 구분한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 관계 생성은 tensor 연산으로 처리한다. 이번에는 학습 batch를 선정하지 않았다.
- [x] check에서 GPU/CPU를 확인했다. 모델 실행/VRAM/RAM 측정은 이번 관계 계약 검사에 포함하지 않았다.
- [x] OOM 회피를 위한 구조 변경은 없었다.
- [x] DEBUG 관계 fixture와 실제 의료 데이터를 구분했다.
- [x] 가짜 관계 정답·성능·체크포인트를 만들지 않았다.
- [ ] v2.21 전체 모델이 forward·loss·gradient·optimizer에 연결되어 있다. 관계 계약만 구현했고 통합은 미완료다.
- [x] 구현 완료 범위와 미완료 범위를 구분했다.
- [x] 단위 검사와 전체 학습/평가를 구분했다. 전체 학습/평가는 미실행이다.
