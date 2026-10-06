# 학습 기반 그래프 풀링 — 원문·현재 코드 대조

2026-09-23. 사용자 질문: Graph U-Net과 같은 풀링으로 노드 표현을 줄이고 중요한 영상 문맥을 학습할 수 있는가? **가능하다. 이것은 기존 연구가 직접 다루는 문제다.** 이번 작업은 원문 확인과 방법 선정이며 production 변경·학습 실행이 아니다.

## 원문에서 확인한 메커니즘

### Graph U-Nets: 특징으로 노드를 선택하고 게이트로 학습

[Graph U-Nets](https://arxiv.org/abs/1905.05178), §3.1, Eq.2. 입력 특징 X와 학습 가능한 투영 벡터 p로 `y = Xp / ||p||`를 계산한다. 상위 k개 노드를 선택하고, 선택한 특징에 `sigmoid(y)`를 곱한다. 기본 인접 행렬은 선택된 노드의 유도 부분 그래프다. 단순 좌표·거리 순위가 아니다.

Top-k 인덱스 선택은 이산 연산이다. 원문이 설명한 학습 경로는 선택된 특징에 곱하는 게이트다. 따라서 'top-k 인덱스 자체가 미분 가능하다' 또는 '버린 모든 노드의 선택 점수가 직접 gradient를 받는다'고 쓰면 안 된다. 선행 GCN이 있으면 gPool에 들어가는 특징 자체에 이웃 정보가 포함된다. gPool 점수식에 A가 없다는 것과 전체 Graph U-Net이 그래프 구조를 무시한다는 것은 다르다.

§3.4, Eq.4에서는 풀링 이후 고립되는 노드에 대응하여 2-hop graph power를 사용한다. 원문의 A² 방식은 기존 엣지를 단순히 필터링하는 것과 다르며, 실제 구현 시 엣지 증가량을 측정해야 한다.

§3.2–3.3의 gUnpool은 저장한 인덱스 위치에 선택 노드 특징을 배치하고 나머지를 0으로 채운다. 버린 정보를 저절로 복구하는 연산이 아니며, decoder skip connection과 함께 사용된다.

### SAGPool: 특징과 이웃 관계로 선택 점수 학습

[Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082), §3.1, Eq.3–5. 점수는 `Z = tanh(D^(-1/2) (A+I) D^(-1/2) X Theta)` 형태의 GNN으로 구한다. 특징과 연결 관계가 모두 점수에 영향을 준다. 상위 비율의 노드를 선택하고 점수를 특징에 곱하며, 선택된 노드 사이 기존 엣지를 보존한다.

§3.2, Fig.2의 hierarchical 구조는 GNN과 pooling 블록을 쌓고 각 블록에서 mean/max readout한 표현들을 합쳐 graph-level 예측에 사용한다. 마지막에 남은 노드만 사용하는 것과 다르다. 이것이 소형 병변 보존을 보장한다는 논문 결과는 아니다. 우리 모델에 풀링 전 readout을 추가해 미세 문맥 경로를 남기는 것은 별도 프로젝트 설계다.

### ASAP: 이웃 정보를 학습해서 모은 뒤 묶음을 선택

[ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations](https://arxiv.org/abs/1911.07979), §4.1–4.4. 각 노드의 국소 이웃으로 겹칠 수 있는 후보 묶음을 만들고, 학습된 attention으로 membership과 묶음 특징을 계산한다. LEConv fitness 점수로 일부 묶음을 선택한다. 단순 유사도 임계값으로 공간 영역을 미리 확정하는 방법이 아니다.

새 연결은 sparse assignment를 사용하는 `A_pool = S_selected^T (A+I) S_selected`로 계산한다. 선택하지 않은 노드도 남은 묶음의 이웃 집계에 기여할 수 있으나, 모든 정보의 보존을 보장하지 않는다. 국소 묶음의 생성 범위는 초기 그래프에 의존한다. sparse 연산이라도 풀링 후 엣지가 증가할 수 있어 본 데이터에서 메모리/처리량을 확인해야 한다.

### DiffPool: 학습된 soft assignment로 병합

[Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804), §3.2, Eq.3–6. `S = softmax(GNN_pool(A,X))`, `X_pool = S^T Z`, `A_pool = S^T A S`. 노드 선택 대신 여러 노드를 학습된 가중치로 묶는다. S와 풀링 후 연결이 dense가 될 수 있어 현재 수만 노드 규모에서 우선 구현 대상으로 택하지 않는다. 실제 OOM을 측정했다는 뜻은 아니다.

## 우리 모델에서의 우선 선택

**SAGPool을 첫 구현·비교 기준으로, ASAP를 국소 정보 집계와 연결 보존을 비교할 대안으로 선택한다.** SAGPool은 특징과 이웃을 함께 점수화하고 기존 sparse 그래프의 노드를 선택한다. 현재 GAT의 엣지 처리와 연결하기가 상대적으로 명확하다. ASAP는 선택 전에 이웃 정보를 모으고 새 엣지 가중치를 만들기 때문에 주변 문맥 목적과 관련성이 높지만, 가중치 전달과 sparse 곱의 비용 검증까지 필요하다.

적용할 역할은 `원본 CT → CNN 특징 → L0 GNN → 학습 풀링 → 후속 L0 GNN → 여러 단계의 문맥 요약 → L1 → L2를 사용하는 최종 판단`이다. 아직 정확한 pool 배치 횟수·유지 비율은 확정하지 않았다. CNN 채널, hidden128, 기존 L0 깊이, L1/L2 깊이, CT FOV·해상도, 전체 split·seed42·CP80·학습 epoch를 조용히 바꾸지 않는다.

L0가 L1에 넘길 graph-level 문맥 벡터를 만드는 현재 목적에는 node-wise 복원을 위한 전체 Graph U-Net decoder가 반드시 필요한 것은 아니다. 이것은 우리 출력 형태에 대한 설계 판단이다. nnU-Net 분할기를 교체하겠다는 뜻도 아니다.

풀링에서 줄이는 주 대상은 N(노드 수)이며 각 노드의 C(특징 차원)를 임의 축소하는 것과 다르다. 유지 비율은 명시적인 모델 설정이다. 논문의 비율을 그대로 가져오거나 실행을 빠르게 하려고 정하지 않는다. 최초 입력 후보의 물리적 해상도와 CNN 특징 해상도 문제는 풀링을 도입한다고 자동으로 해결되지 않는다.

## 현재 코드와 실제 gradient 경로

- `hiercp_v222/model.py`의 현재 `pool_gate`는 마지막 GNN 출력의 weighted global readout이다. 중간 그래프의 노드/연결을 줄이는 학습 풀링이 아니다.
- r4 `sampling.py`의 PPR/A* 선택에는 별도의 학습 가능한 pooling scorer가 없고 선택 계산은 no-grad다. 이를 Graph U-Net/gPool 구현이라고 부르면 안 된다.
- 새 pooling을 `self.local` 내부에 연결하면 **query CE → query logits → query L1 표현 → L0 → pooling scorer/CNN** 경로를 검증해야 한다. 기존 support memory는 detach되어 있으므로 현재 alignment loss가 query CNN/pooling까지 직접 학습시킨다고 주장하면 안 된다. alignment branch와 query branch를 구분한다.
- 라벨·종양 마스크는 pooling forward 입력으로 넣지 않는다. 기존 학습 정답은 loss와 support 관계 구성에 사용하는 정책을 유지한다. pooling 점수는 학습된 선택 점수이며 보정된 암 확률·병변 위치 정답이 아니다.

## 설치된 구현에서 확인한 통합 주의점

설치된 PyG 2.6.1의 `SAGPooling.forward`, `ASAPooling.forward`를 직접 읽었다. SAG는 `self.gnn(attn, edge_index)` → 선택 → `x[perm] * score` → 엣지 필터링 순서다.

ASAP는 attention 이웃 집계 → 학습된 fitness → 선택 → sparse `S.T @ (A @ S)`를 수행한다. 이 버전은 입력 `edge_weight`가 None이면 반환할 가중치를 보존하지 않는 분기가 있다. 또한 현재 L0 GATv2는 edge_weight를 직접 사용하는 경로가 없다. 따라서 ASAP를 사용할 경우 초기 단위 edge weight, 새 가중치 반환, 후속 message passing의 가중치 소비·gradient까지 연결해야 한다. 라이브러리 호출만 추가하고 학습된 가중치를 버린 구현을 완료로 보고하지 않는다.

## 필요한 검증과 현재 상태

- scorer 파라미터가 optimizer에 포함되고 query CE에서 실제 gradient를 받는지 확인.
- 고정 기하·동일 seed에서 CT 특징 변화가 선택 결과에 미치는 영향, 선택 기록 재현성 확인.
- 작은 병변과 주변 문맥이 pooling 전후 어떻게 표현되는지 분석. GT는 검증 지표에만 사용하며 입력 선택을 유도하는 누수와 구분.
- 단계별 노드/엣지 수, 연결 성분, receptive field, fine-scale readout 경로와 손실을 측정.
- 실제 전체 그래프를 유지하는 후보들의 physical batch/VRAM/처리량 측정. pooling 없음, SAGPool, ASAP를 같은 데이터·학습 조건에서 비교.

이번에는 논문 본문과 설치 코드를 읽고 참고문헌/선정 기록을 추가했다. pooling production 구현, 실제 pooling forward/backward 시험, 전체 학습, 전체 의료 평가는 수행하지 않았다. r4 학습 차단은 유지한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 이번 작업은 문헌/코드 대조다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] 새 pooling의 physical batch size와 병렬화 성능을 실측했다. 미구현이다.
- [ ] 새 pooling의 GPU, CPU, RAM 활용 상태를 실측했다. 미구현이다.
- [x] OOM을 이유로 모델을 축소하지 않았다.
- [x] 디버그와 최종 실행을 구분했다. 새 실행은 하지 않았다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 새 pooling의 forward, loss, gradient와 optimizer 연결을 검증했다. 아직 코드 독해만 했다.
- [x] 실제 변경 범위를 명확하게 보고했다.
- [x] 논문 결과, 코드 독해와 실제 학습/평가 완료를 구분했다.
