# v2.2 Local-CNN과 PRODIGY의 실제 구현 대조

작성일: 2026-10-01. 검토 기준 local HEAD `5f01011e51a0779df474ea912bc25cc6663b7b93`.
이 문서는 **구현 감사**다. 새 모델 버전·학습 성능 결과·production 승인 문서가 아니다.

## 판정

**L0 문맥 표현을 data node로 쓰고, L1에서 support의 알려진 data–label 관계로 query를 판단하는 발상은 PRODIGY를 참고했다. 그러나 현재 CT L0와 L1 연산은 공식 구현을 그대로 가져온 것이 아니다.** 연결 그림을 유지하면서 self-message, message value, residual/정규화, 학습 경계와 점수 계산을 변경했다.

현재 부진을 “prompt graph는 직접 scalar head가 없어서 학습할 수 없다”거나 “CT 문제 자체가 원래 학습 불가능하다”로 설명할 근거는 없다. 공식 PRODIGY는 query–label 유사도와 query loss로 학습하며, v1에서도 실제 ranking 성공 기록이 있다. 먼저 본 프로젝트가 바꾼 경로를 검토해야 한다.

실제 코드·서버 증거가 가장 직접적으로 가리키는 구간은 **L0 fusion과 두 번째 query L1의 추가 FFN/정규화**다. 공식 원형에 있던 **attention self-message가 현재 빠져 있는 것도 확정된 차이**다. 이 차이들과 수축 측정이 관련될 가능성은 있지만, 어느 하나가 단독 원인이거나 복원만으로 성능이 회복된다고 확정하지 않는다.

## 공식 기준을 고정한 방법

공식 저장소 `snap-stanford/prodigy`의 commit
`107ba57234d3188227cda5b78a2dbcfb84a1c694`를 GitHub API로 확인하고, 그 commit의 다음 파일을 새 감사 폴더에 저장했다.

- `models/metaGNN.py`
- `models/general_gnn.py`
- `data/dataloader.py`
- `experiments/trainer.py`

원문 bytes·SHA256·고정 URL은 `validation/prompt_graph_audit_20261001/official/manifest.json`에 기록했다. 이 코드는 production에 설치·편입하거나 실행한 코드가 아니다. 최초 web의 `main` 조회본과 이 고정 원문의 줄 번호는 다르므로 아래는 **고정 원문**의 1-based 번호다.

1. [공식 MetaGNNLayer](https://github.com/snap-stanford/prodigy/blob/107ba57234d3188227cda5b78a2dbcfb84a1c694/models/metaGNN.py): 75–142, MetaGNN 선택/구성 223–290.
2. [공식 forward와 cosine decode](https://github.com/snap-stanford/prodigy/blob/107ba57234d3188227cda5b78a2dbcfb84a1c694/models/general_gnn.py): 28–47, 114–155.
3. [공식 task collator](https://github.com/snap-stanford/prodigy/blob/107ba57234d3188227cda5b78a2dbcfb84a1c694/data/dataloader.py): 358–366.
4. [공식 loss/optimizer](https://github.com/snap-stanford/prodigy/blob/107ba57234d3188227cda5b78a2dbcfb84a1c694/experiments/trainer.py): 58–63, 259–270, 384–390.
5. [논문 §2.3, §3.1, Appendix C](https://cs.stanford.edu/~jure/pubs/prodigy-neurips23.pdf).

공식 기준은 기본 `MetaGNNLayer` / `gat_layer=False` 경로다. 저장소의 alternate `MetaGATConvLayer`와 `MetaGATConvLayerBi`에는 pre-LayerNorm/FFN이 있으므로 **“PRODIGY에는 FFN이 전혀 없다”라고 일반화하지 않는다.** 공식 기본 코드의 `out_proj`는 edge별 가중 value에 적용된 후 합산된다. bias가 edge 수만큼 누적되는 이 구현은 논문의 합산 후 projection 수식과도 구분해야 한다.

## 현재 L0: 원형 encoder의 역할을 맡는 별도 CT 구현

`l0_local_cnn/model.py:39–62`의 실제 경로는 다음과 같다.

```text
donor/recipient의 native 국소 CT crop + 간 mask
→ 공유 OrganPyramid CNN: channels 12/24/32, convolutions 2/3/3
→ 각 crop 내부 간 영역의 세 scale masked mean
→ concat 68차원 → project 128차원
→ donor d, recipient r 조회
→ Fuse([d, r, r-d, r*d]) → 128차원 L0
```

**간 전체를 평균하는 경로가 아니다.** margin에 따른 국소 crop 안의 간 영역을 평균한다. 현재 이 L0에는 graph node/edge, GAT, GraphSAGE, EZ-SP가 없다. 훈련 파일이 `l0_regions` 디렉터리에 있다는 이유로 실제 L0에 GraphSAGE가 있다고 설명하면 안 된다.

PRODIGY의 DataGraph encoder가 만드는 data representation과 **기능상 대응**하지만, 이 CT crop/CNN/평균/fusion 연산이 PRODIGY에서 복사됐다는 뜻은 아니다. PRODIGY 논문의 레벨 이름 자체도 본 프로젝트의 L0/L1/L2와 동일한 명명 계약이 아니다.

## L1 topology: 맞는 부분과 원형에서 달라진 부분

| 항목 | 공식 기본 prompt graph | 현재 v2.2 | 판정 |
|---|---|---|---|
| 알려진 support | data↔task label | data↔자기 환자의 두 label | 방향 원칙은 맞음 |
| query | task label→query, query 정답 관계 미제공 | 모든 support 환자의 label→query, U edge | 방향 원칙은 맞음; 환자 집합 pooling은 확장 |
| label 수 | task의 class label | 환자마다 두 관측 class의 label | 별도 적용 설계 |
| edge 입력 | multiway T=[0,+1], F=[0,-1], U=[1,0] | T=[1,1], F=[1,0], U=[0,0] | 구별은 가능하나 같은 parameterization 아님 |
| self-loop | 각 노드→자기 자신, [0,0] | **support·label·query 모두 없음** | 원형의 경로 누락 |
| update 시점 | 각 층의 입력을 읽는 동시 message passing | 층 입력의 label history를 query에서 사용 | 현재 구현이 맞음; off-by-one 아님 |

현재 근거: `hiercp_v222/model.py:223–247`, `287–295`. 공식 self-loop 근거: 고정 `models/metaGNN.py:273–280`.

첫 query L1이 읽는 label은 환자별로 복제된 공통 class seed다. 아직 support 영상이 label에 반영되기 전이다. 둘째 query L1은 첫 support L1을 거친 label을 읽는다. 이는 동시 업데이트 구조이며, history를 해당 층의 **출력**으로 바꾸면 정보 전달 순서가 달라진다. 이 코드를 잘못된 history로 판정하거나 자동 교정하면 안 된다.

**Residual은 attention self-loop와 같지 않다.** 현재 query는 label의 value만 집계하고 자기 특징은 바깥 residual에 보존한다. 공식 self-loop는 자기 query의 key/value도 softmax 경쟁과 메시지 집계에 넣는다. 자기 정답을 사용하지 않으며 query→support 연결도 추가하지 않으므로, 정답 누수 방지를 이유로 반드시 없애야 하는 연결은 아니다.

다만 현재는 환자마다 두 label, 학습 시 통상 16명의 support를 묶는다. 공식 task의 두 class label과 동일한 source 집합은 아니다. 따라서 self-loop 추가 한 가지로 공식 task 전체가 재현되는 것은 아니다.

## L1 실제 연산 대조

| 연산 | 공식 기본 MetaGNNLayer | 현재 RelationLayer |
|---|---|---|
| q/k/v | 합친 biased Linear; k에 1/√head-width | 별도 bias 없는 Linear; 해당 k scaling 없음 |
| attention 입력 | [k,q,ReLU(edge)] | [q,k,SiLU(edge)] |
| attention MLP | ReLU | LeakyReLU(0.2) |
| 정규화 방향 | destination의 incoming source softmax | 같음 |
| value | V(source) | **V(source)+edge embedding** |
| attention dropout | 있음 | 없음 |
| update | residual + BatchNorm, 층 사이 GELU/ReLU | **LayerNorm → 추가 4D FFN residual → 두 번째 LayerNorm** |
| projection bias | edge별 out projection 후 aggregate | aggregate 후 out projection, bias 한 번 |

현재 위치: `hiercp_v222/model.py:58–68`, `174–197`. attention 순서/q-k bias/활성화 자체를 각각 버그로 단정하지 않는다. 모두 학습 가능한 다른 연산이다. 그러나 “공식 학습 block을 그대로 가져왔다”는 설명과는 일치하지 않는다.

또한 공식과 현재 둘 다 additive MLP attention을 사용한다. 활성화 구간에 따라 query의 공통 shift가 softmax에서 상쇄될 수 있지만, **모든 입력에서 query가 무시된다고 단정할 수 없다.** 이미 실행한 내적항 진단도 최종 순위를 회복하지 못했다.

BatchNorm만 그대로 끼워 넣는 것도 원형 복원이 아니다. 공식은 support/label/query를 함께 실행하고, 현재는 support와 query를 분리 실행한다. 두 번 별도로 계산한 BatchNorm의 학습 통계는 joint graph와 다르다. 공식 방식의 train 통계에서는 query 특징이 다른 노드의 통계에도 영향을 줄 수 있다. query 정답 사용과는 구별해야 한다.

## 학습·L2·점수의 차이

| 경계 | 공식 PRODIGY | 현재 v2.2 |
|---|---|---|
| episode encoder | support/query를 episode forward에서 인코딩 | **query P/U만 최신 CNN**, support L0는 detached epoch memory |
| L1/L2 trainable 여부 | task graph trainable | L1/L2·label seed는 매 update trainable |
| L2 | 현재 같은 환자 간 군집 정렬 없음 | cross-patient MHA 2층 + frozen cluster teacher + live prototypes |
| readout | query–task label cosine, 학습 logit scale | query–정렬 prototype cosine / 고정 temperature 0.2, class별 mixture logsumexp |
| query objective | multiway CE; binary BCE 또는 조건부 margin rank, 설정에 따른 attribute auxiliary | 전체 same-donor P×U softplus + balanced observation CE + support alignment CE |

현재 P/U는 `l0_regions/donor_learning.py:75–85`에서 같은 batch로 계산한다. 이전의 “다른 donor 비교”, “현재 양성 CNN만 gradient가 안 감”은 수정 전 문제이며 현재 원인으로 재사용하지 않는다.

Support memory: `l0_regions/training.py:181–204`, `423–437`, `l0_regions/support_episodes.py:98–99`. Query CNN은 업데이트되므로 epoch 중 두 encoder 표현의 시점이 달라진다. 코드상 확정된 차이지만 이것만으로 원인을 확정할 수 없다.

현재 rank/CE는 query L1→query L0/CNN, 그리고 live prototype→L2→support L1/label seed 양쪽에 gradient를 보낸다. **후자에서 support L0 memory까지만 멈춘다.** `hiercp_v222/clustering.py:106–129`의 score용 prototype은 live L2 값으로 미분 가능하게 만든다. Teacher 군집 배정·center만 고정되어 있으므로 “prototype까지 고정이라 학습이 안 된다”는 설명도 틀리다.

Rank는 전체 P×U mean으로, observation CE는 tile 반복 multiplicity를 역보정한다. 그러나 `donor_learning.py:85`의 support alignment는 각 tile에 같은 계수로 들어간다. 결과적으로 tile이 많은 환자 episode의 alignment를 더 많이 최적화하는 **step-weighted objective**다. equal-per-episode 계약이라면 weighting mismatch다. 현재 구현이 step-weighted라고 명시한 상태에서는 의도 확인 없이 버그로 단정하거나 자동 보정하지 않는다.

학습은 selected 16-patient support, 평가는 전체 eligible inner-train support를 사용한다. 이전 동일 query의 episode/full 대조에서 loss 차이는 작았다. 따라서 이 차이도 이번 collapse의 주원인으로 확정하지 않는다.

## 서버 측정과 결합한 판단

각 snapshot의 가중치와 측정 범위를 섞지 않는다.

| 증거 | 확인된 사실 | 확인되지 않은 주장 |
|---|---|---|
| epoch22 deep probe | fusion input→output 방향 분산 약39–120배 감소; 둘째 L1 약920–960배 감소, FFN residual에서 큰 수축 | CNN이 원천적으로 특징을 학습하지 못함; 정확도 손실률이 이 비율과 같음 |
| epoch27 FFN scale | 둘째 query FF residual scale0에서 spread 약14배 증가, ordering 효과 혼재 | FFN 삭제가 해결책으로 검증됨 |
| epoch30 interaction | 둘째 L1 방향 분산 약2,220–2,317배 감소; 내적항 후에도 pair loss ln2 부근, 4update validation pair win 약50% | β 증가가 학습 문제를 해결함 |
| v1 실제 terminal val | 36sample Top1=1/MRR=1 기록 | 현재 후보/정답/metric과 동일 조건에서 정확도100% |

Epoch30 원문 및 전사는 `validation/local_cnn_interaction_20261001/`와 `docs/local_cnn_interaction_epoch30_20261001.md`에 이미 보존돼 있다. 서버 원본 report JSON과 학습 checkpoint는 로컬 미수신이다. 이 감사에서 새로 epoch30 CT/가중치를 실행한 것으로 표현하지 않는다.

**수정 우선순위는 기존 prompt 경로의 self-message와 custom FFN/정규화, 그 앞의 fusion이다.** 이들을 구분해서 대조해야 한다. scalar head 우회, CNN 확대, 새 군집 방법, 또 다른 장기 학습부터 제안할 근거는 없다. 원형 대조에서도 현재 L2/loss/CP를 동시에 바꿔 효과를 뒤섞으면 안 된다.

## 이번 검증 경계와 산출물

읽기 전용 source audit와 별도 CUDA UNIT 검사를 완료했다. UNIT 결과는
`validation/prompt_graph_audit_20261001/cuda_unit_report.json`에 기록했다.
상태는 **PASS**이며 보고서 SHA256는
`7efb606303e6218857d626591ae755dcea8e4346e88cf5e74c78cdc81b003fd3`다.

UNIT는 현재 dim128/heads4/L1 2층/L2 2층을 보존하고 query embedding128개를 검사한다. physical chunk32, support16환자×16관측의 **명시적 tensor fixture**이며 실제 CT·CNN 성능 검사로 제출하지 않는다. 목적은 현재 split/joint L1의 동치, 실제 edge 방향·관계 입력, gradient 경계와 상태 보존이다. 공식 PRODIGY의 재학습/성능을 검증하는 경로가 아니다.

| CUDA UNIT 검사 | 실제 결과 |
|---|---|
| 실제 support edge capture | 256 data + 32 patient label, 양방향 T/F edge1,024개; self-loop 없음 |
| 실제 query edge capture | chunk32에서 label→query U edge1,024개; 역방향/self-loop 없음 |
| current-operator split/joint 동시 L1 | query 최대차7.15e-7, label 최대차1.43e-6; history 입력 일치 |
| input + L1/seed parameter gradient | input2개/parameter37개 비교, 최대차3.73e-9 |
| 전체128 query, chunk/permutation | chunk32×4, logits 최대차2.38e-7; 순서 역복원 차이0 |
| detached support/live query rank gradient | support input 없음; query/L1/L2/label seed gradient 유한·nonzero |
| query 정답 API | predict의 query_targets 거부, forward에 정답 argument 없음 |
| 상태/소스 보존 | model state·mode·gradient buffer·production source hash 일치 |

RTX5070Ti, PyTorch2.8.0+cu128, FP32, L1/L2 각2층/128D/4heads.
검사 내부1.072초, peak GPU 할당0.118GiB, process RSS 약1.75GiB,
CPU logical16, 시작 available RAM 약44.51GiB다. 이 작은 **embedding UNIT**의
메모리와 시간은 실제 CT update/서버 epoch 비용이 아니다.

Eval로 stochastic dropout을 끈 수치 동치 검사다. Train mode의 dropout random
stream까지 exact resume parity를 검사한 것이 아니다. 현재 split/history 경로가
잘못돼 수축한다는 가설을 지지하지 않으며, 공식 연산과 현재 연산의 동치도 아니다.
Gradient 확인은 optimizer 성능이나 학습 정확도의 검증이 아니다.

정적 AST 검사와 변경 파일 whitespace 검사도 통과했다. 첫 실행은 계산 완료 후
sandbox의 새 보고서 파일 쓰기에서 PermissionError가 발생했다. production 오류로
취급하지 않았고, 같은 새 경로에 승인된 재실행으로 보고서를 저장했다. 기존 결과는
덮어쓰지 않았다.

Production 코드·config·loss·후보128·Basic CP·원본 mask·checkpoint는 변경하지 않았다. 새 감사 source/문서/evidence만 추가했다. 장기 GNN/nnU-Net 학습, 전체 CT 평가와 production ready 생성은 하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. UNIT fixture와 최종 데이터는 분리했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 검토했다. UNIT의 chunk32는 기존 값이다.
- [x] GPU 상태를 확인했고 UNIT report에 CPU/RAM/GPU 자원을 기록했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. 이번 감사에서 최종 모델 축소 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다. UNIT tensor fixture는 명시했다.
- [x] 핵심 forward/loss/gradient/optimizer 연결을 source와 기존 실제 CT 증거로 대조했다. 이번 UNIT는 optimizer update 검사가 아니다.
- [x] 실제 검사 설정과 production 변경 없음을 명확하게 보고했다.
- [x] UNIT/smoke와 전체 학습 또는 전체 평가를 구분해서 보고했다.
