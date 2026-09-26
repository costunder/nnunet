# References — HierCP / nnU-Net

갱신: 2026-09-22, v2.22 r2. 논문에서 가져온 개념, 실제 구현, 프로젝트에서 새로 정한 조건을 분리한다. 논문 인용은 본 데이터의 소형 종양 성능을 입증하지 않는다. 현재 설계는 [L2 군집 정렬](docs/pipeline_v222_cluster_r2.md), 전체 흐름은 [파이프라인](docs/pipeline_v222.md)을 따른다.

## 현재 구현과 연결된 근거

| ID | 서지·원문 | 실제 연결 | 원문과 다른 점 / 적용 범위 |
|---|---|---|---|
| R1 — SwAV | Caron, Misra, Mairal, Goyal, Bojanowski, Joulin. **Unsupervised Learning of Visual Features by Contrasting Cluster Assignments**. NeurIPS 2020. [논문](https://arxiv.org/abs/2006.09882), [공식 코드](https://github.com/facebookresearch/swav) | 정규화된 prototype과 assignment 예측이라는 설계 참고. Appendix D의 DeepCluster-v2도 참고. | **SwAV 재현 아님.** 원문의 동일 영상 view 간 swapped prediction, Sinkhorn 균등 배정, multi-crop, 학습 가능한 prototype 파라미터를 구현했다고 주장하지 않는다. 현재 중심은 support의 군집별 평균이며 별도 Parameter가 아니다. SwAV가 K를 자동 선택한다는 주장도 하지 않는다. |
| R2 — DeepCluster | Caron, Bojanowski, Joulin, Douze. **Deep Clustering for Unsupervised Learning of Visual Features**. ECCV 2018. [논문](https://arxiv.org/abs/1807.05520), [공식 코드](https://github.com/facebookresearch/deepcluster) | 표현에서 군집 pseudo-target을 만들고 이를 고정한 상태에서 표현을 학습하는 교대 절차. `fit_support_clusters`, `alignment_loss`. | 원문 k-means를 그대로 쓰지 않는다. 본 구현은 관측 클래스별 cosine average-linkage, 환자 episode별 재계산이다. 의료 관측 클래스, query 환자 제외, loss 결합은 프로젝트 설계다. |
| R3 — PRODIGY | Huang, Ren, Chen, Kržmanc, Zeng, Liang, Leskovec. **PRODIGY: Enabling In-context Learning Over Graphs**. NeurIPS 2023. [논문](https://arxiv.org/abs/2305.12600), [최종 PDF](https://cs.stanford.edu/~jure/pubs/prodigy-neurips23.pdf), [공식 구현](https://github.com/snap-stanford/prodigy/blob/main/models/metaGNN.py) | L1 data–label prompt graph, 알려진 support 관계와 미지 query 관계 구분, query로부터 support로 정답을 보내지 않는 구조. `RelationLayer`, `encode_support`, `predict_embeddings`. | CT 문맥을 data로 삼는 것, 두 관측 클래스, 128차원, L2 환자 간 군집 정렬은 본 프로젝트의 적용 설계다. PRODIGY가 이 의료 L2를 제안했다고 쓰지 않는다. |
| R4 — GATv2 | Brody, Alon, Yahav. **How Attentive are Graph Attention Networks?** ICLR 2022. [논문](https://arxiv.org/abs/2105.14491), [공식 코드](https://github.com/tech-srl/how_attentive_are_gats) | L0 공간 GNN의 PyG `GATv2Conv`. 전체 그래프 계산을 streaming하는 `chunked_gat`는 PyG 출력·입력/파라미터 gradient와 대조한다. | L1은 별도 edge-conditioned relation attention이다. 모든 attention 층을 GATv2라고 부르지 않는다. 3층/128/4 heads는 기존 프로젝트 설정을 계승했다. |
| R5 — Silhouette | Peter J. Rousseeuw. **Silhouettes: A graphical aid to the interpretation and validation of cluster analysis**. Journal of Computational and Applied Mathematics 20, 53–65, 1987. [DOI](https://doi.org/10.1016/0377-0427(87)90125-7), [구현 정의](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) | `select_partition`에서 support 표현의 군집 내/군집 간 cosine 거리로 후보 K를 비교한다. | 모든 비단독 군집 cut을 비교하고 양의 최고 점수, 동점은 작은 K, 해상 불충분 시 K=1이라는 선택 규칙은 프로젝트 설계다. Silhouette는 실제 생물학적 아형이나 최적 K의 증명이 아니다. |
| R6 — Hierarchical clustering | Daniel Müllner. **Modern hierarchical, agglomerative clustering algorithms**. 2011. [논문](https://arxiv.org/abs/1109.2378), [SciPy linkage 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) | `scipy.cluster.hierarchy.linkage(method='average')`, `cut_tree`. 정규화된 환자 label 간 cosine 거리 사용. | 논문의 모든 linkage 방법을 사용하는 것이 아니다. Average-linkage를 택한 이유는 고정 K/무작위 초기 중심 없이 전체 계층의 cut을 비교하기 위해서다. CT에서 우수하다는 실험 근거는 아직 없다. |
| R7 — nnU-Net | Isensee, Jaeger, Kohl, Petersen, Maier-Hein. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation**. Nature Methods 18, 203–211, 2021. [DOI](https://doi.org/10.1038/s41592-020-01008-z), [공식 nnU-Net v2 코드](https://github.com/MIC-DKFZ/nnUNet) | 분할 학습·추론·기본 native loader 및 사용자 정의 online CP trainer의 기반. | 이 논문이 GNN 위치 선택, 본 관측 과제, 군집 loss 또는 CP 성능을 보증하는 것은 아니다. |

## 과거 검토·배경 자료 — 현재 특징으로 재도입하지 않음

| ID | 자료 | 검토 목적과 현재 상태 |
|---|---|---|
| R8 | Landrieu & Simonovsky. **Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs**. CVPR 2018. [논문](https://arxiv.org/abs/1711.09869) | 그래프/기하 특징 선행 사례를 검토했다. 해당 논문의 점군 특징을 CT에 임의로 결합하는 근거로 쓰지 않는다. 현재 수작업 특징 묶음 미사용. |
| R9 | Qi et al. **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**. NeurIPS 2017. [논문](https://arxiv.org/abs/1706.02413) | 국소 공간 이웃과 좌표 활용 비교 자료. 현재 PointNet++ 미구현, 상대 좌표를 CNN 특징에 concat하지 않음. |
| R10 | Wang et al. **Dynamic Graph CNN for Learning on Point Clouds**. [논문](https://arxiv.org/abs/1801.07829) | EdgeConv 및 동적 이웃 비교 자료. 현재 DGCNN/EdgeConv 미구현. |
| R11 | Wickramasinghe et al. **Voxel2Mesh: 3D Mesh Model Generation from Volumetric Data**. [논문](https://arxiv.org/abs/1912.03681) | CNN 영상 특징을 그래프 계열 처리에 전달하는 개념 참고. 현재 mesh decoder/mesh supervision 미구현. |
| R12 | **Image Biomarker Standardisation Initiative (IBSI)**. [공식 특징 정의](https://ibsi.readthedocs.io/en/latest/03_Image_features.html#intensity-based-statistical-features) | 과거 통계 특징 정의 감사에 참고했다. 공식 정의가 있다는 것과 본 과제에서 유효하다는 것은 다르다. 폐기된 통계 묶음을 복구하지 않는다. |

과거 감사 문서: [feature_evidence_audit_20260922.md](docs/feature_evidence_audit_20260922.md). 위 자료를 최신 구현 모듈인 것처럼 열거하지 않는다.

## Copy-Paste 적용 확률 확인 (2026-09-22)

- **R13 — TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation.** [원문 §3.2](https://arxiv.org/html/2107.09843#S3.SS2). 저자들은 모든 실험에서 CP 적용 확률 `p_cp=0.8`을 사용했다. CP가 발동했을 때 개별 object-level 변환은0.5이고, intra/inter를 결합한 설정의 선택 비율도 각각0.5다. 세 확률을 혼동하면 안 된다. KiTS19 실험 설정이며 현재 간 CT/GNN에서80%가 최적이라는 증거는 아니다. 기존 사용자 Basic CP를 TumorCP 재현이라고 바꾸어 부르지 않는다.
- **R14 — Ghiasi et al., Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. CVPR2021.** [원문](https://arxiv.org/abs/2012.07177). 두 영상과 복사할 객체의 무작위 부분집합을 사용한다. 객체 선택의 무작위성과 샘플별 CP 실행 gate는 서로 다른 개념이다. 이번 원문 확인만으로 모든 CP 방법의 공통 적용 확률을50%나100%로 정할 수 없다.
- 설치된 nnU-Net의 `get_training_transforms` 확인: 회전0.2, scaling0.2, Gaussian noise0.1, 밝기0.15, contrast0.15, 일반 gamma0.3. 각 transform의 바깥 적용 확률이며 채널별 추가 gate가 있는 다른 증강과 구분한다. 증강 하나만 택하는 확률분포가 아니므로 합을1로 맞추지 않는다.
- 본 프로젝트 원본 Basic CP는 별도의 확률 gate 없이 매 방문 CP를 시도한다. 변경 전 v2.22의0.5는 프로젝트 계승값이며 TumorCP 논문의 `p_cp`를 그대로 옮긴 설정도, 검증된 최적값도 아니다. 이후 사용자가 양쪽0.8 비교를 명시적으로 승인해 Basic CP80 변형과 v2.22의 설정을0.8로 맞췄다. 이 값의 현재 데이터 최적성을 주장하지 않는다. 원본1.0 트레이너는 보존했다.

## 문맥 기반 배치 검토 (2026-09-23)

- **R15 — Dvornik, Mairal, Schmid. Modeling Visual Context is Key to Augmenting Object Detection Datasets. ECCV2018.** [원문](https://openaccess.thecvf.com/content_ECCV_2018/html/NIKITA_DVORNIK_Modeling_Visual_Context_ECCV_2018_paper.html), [저자 코드](https://github.com/dvornikita/context_aug). 객체 주변 문맥을 학습해 copy-paste 위치를 고르는 선행연구다. 이번 학습 목표 감사에서 검토했으며 CT/GNN 또는 원본 CT 입력 수정안의 효능이 입증된 것으로 인용하지 않는다. 현재 구현을 해당 논문 재현이라고 부르지 않는다. 상세 `docs/v222_raw_ct_proposal_20260923.md`.

## 코드 출처와 프로젝트 고유 선택

- **R16 — Wang et al. Dynamic Graph CNN for Learning on Point Clouds.** [원문](https://arxiv.org/abs/1801.07829). L0 수정 검토의 근거: 매 layer의 현재 특징으로 kNN 관계를 갱신한다. 기존 r3에 적용된 방법이 아니며 CT CP 효능 근거도 아니다.
- **R17 — Han et al. Vision GNN: An Image is Worth Graph of Nodes.** [원문](https://arxiv.org/abs/2206.00272). 영상 patch 특징을 노드로 사용하는 설계 근거. 원문의 positional encoding과 graph operator를 포함해 재현했다고 주장하지 않는다. [미적용 L0 수정 명세](docs/v222_l0_feature_graph_proposal_20260923.md).

- **Basic CP**의 기준은 저장된 사용자 원본과 [원본 비교 기록](docs/original_basic_cp_online.md), [보존 구현](basic_cp_online/reference.py)이다. 이를 출처 확인 없이 다른 Copy-Paste 논문의 재현으로 바꾸어 인용하지 않는다.
- L2의 **관측 클래스별 군집**, **query 환자 그룹 전체 제외**, **군집당 최소 2명의 독립 환자라는 submode 조건**, **silhouette cut 선택**, **query CE + alignment CE의 동등 가중치**, **군집 환자 수 가중 log-sum-exp 점수**는 프로젝트 구현 선택이다. 위 논문이 이 조합을 의료 문제에서 검증했다는 근거는 없다.
- Temperature 0.2, hidden 128, L0/L1/L2 깊이 3/2/2와 4 heads는 기존 계약 유지다. SwAV 원문의 값을 임의 축소한 결과가 아니다.
- Source는 `hiercp_v222/model.py`, `clustering.py`, `training.py`, `scoring.py`에 연결된다. 참고문헌 추가만으로 구현 완료나 성능 개선으로 기록하지 않는다.

## PageRank와 외곽 경로를 통한 L0 표집 (v2.22 r4)

- **R18 — Andersen, Chung, Lang. Local Graph Partitioning using PageRank Vectors. FOCS2006.** [원문](https://snap.stanford.edu/class/cs224w-readings/andersen06localgraph.pdf). 개인화 PageRank와 degree로 정규화한 sweep의 근거다. r4는 논문의 conductance 최적화 partition 재현이 아니며 PPR 질량95% 선택 + A* + halo를 사용하는 프로젝트 변형이다. CT CP 효과의 근거로 인용하지 않는다. continuation0.85 출발값은 [NetworkX PageRank 공식 문서](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html)의 기본값과 같다. 이는 최적 하이퍼파라미터라는 뜻이 아니다.
- **R19 — A* 동작·heuristic 조건: NetworkX 공식 문서.** [astar_path](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html). r4 구현은 NetworkX의 Python 경로 반복 대신 graph×target 배치 GPU 탐색이다. `distance × (2 − cosine)` 비용은 직선거리 이상이므로 Euclidean heuristic을 사용한다. CNN 비용,26방향 목표,1-hop halo, PPR와의 합집합은 프로젝트 고유의 미검증 연구 선택이다. A*가 종양 위치나 해부학적 경로를 보장하지 않는다.

## 3D 공간 표현과 표집의 구분 — 설계 재검토, 미적용 (2026-09-23)

아래 자료는 r4 방사형 설계 반려 후 비교한 근거다. 새 L0 구현 완료나 CT 성능의 근거로 읽지 않는다. 공간 표현·엣지 정의가 먼저이고, 표집·경로 탐색은 그 위의 별도 연산이다.

- **R20 — Zeng et al. GraphSAINT: Graph Sampling Based Inductive Learning Method. ICLR 2020.** [원문](https://arxiv.org/abs/1907.04931). Appendix B의 RW는 전체 그래프에서 여러 시작 노드를 균등 복원 추출하고 이웃으로 random walk한 뒤 선택 노드의 유도 부분 그래프를 만든다. 지나간 경로 엣지만 유지하는 방식이 아니다. 기존 그래프를 입력으로 받으므로 CT에서 노드와 엣지를 무엇으로 정의할지 해결하지 않는다. 원문의 표집 확률에 따른 aggregation/loss 정규화와 본 프로젝트의 graph-level GAT/pooling 적용은 별도 검토가 필요하다. 다중 시작점 제안은 아직 미구현이며 GraphSAINT 재현이라고 부르지 않는다.
- **R21 — Hornung et al. OctoMap: An Efficient Probabilistic 3D Mapping Framework Based on Octrees. Autonomous Robots 2013.** [저자 공식 사이트·논문](https://octomap.github.io/). Octree를 사용한 다중 해상도 3D 점유 지도다. 점유·빈 공간·미관측 공간을 표현한다. Octree의 부모–자식 관계와 GNN에서 쓸 공간 이웃 엣지는 같은 정의가 아니다. CT는 점유/빈 공간 지도가 아니므로 로봇 지도용 분할 기준을 그대로 의료 문맥 기준으로 사용하지 않는다.
- **R22 — OMPL, Available Planners.** [공식 문서](https://ompl.kavrakilab.org/planners.html). PRM은 여러 질의에서 사용할 roadmap을 만들며, RRT 계열은 유효한 이동으로 연결된 상태 트리를 확장한다. 공간 위치뿐 아니라 문제의 configuration/state가 노드가 될 수 있다. 경로 계획 그래프의 이동 가능성과 CT 조직 문맥 관계는 목적이 다르다. 이를 A*를 대체할 L0로 채택하지 않았다.
- **R23 — 3D Multi-Object Tracking Using Graph Neural Networks with Cross-Edge Modality Attention (Batch3DMOT).** [원문](https://arxiv.org/abs/2203.10926). 노드는 시점별 3D 검출이며, 엣지는 다른 시점 검출 사이의 연관 후보다. 논문은 과거 시점의 운동학적 유사 이웃을 연결하는 방향 그래프를 사용한다. 3D 공간 전체를 분할한 그래프가 아니라 시공간 데이터 연관 그래프라는 점을 구분한다. 특정 항공기·미사일 추적 구현을 이 논문에서 확인했다고 주장하지 않는다.
- **R8 Superpoint Graphs 추가 확인:** §3.2에서 3D 점군을 나눈 영역(superpoint)이 노드이고 Voronoi 기반 영역 인접성이 엣지다. 영역 간 문맥을 학습한다는 목적은 이번 CT 요구와 비교할 가치가 있다. 원문은 기하 분할과 수작업 superedge 특징을 사용하므로 현재 CNN-only 특징 계약에 그대로 들어맞는 구현으로 간주하지 않는다. R16 DGCNN의 점/특징 이웃 방식, R17 ViG의 patch 노드 방식과 함께 표현 선택을 검토하는 자료다.

현재 판단: A* 경로, PageRank 점수, GraphSAINT 표집을 먼저 선택하고 그 결과 모양을 문맥 그래프로 부르는 순서를 중단한다. CT의 노드 단위와 관계 정의, 실제 message-passing 범위를 먼저 검증해야 한다. 생산 설정 변경과 신규 학습은 이번 문헌 비교에서 수행하지 않았다.

## L0 우선 후보 선정 (2026-09-23, 미구현)

[선정 기록](docs/v222_l0_method_selection_20260923.md): R16 DGCNN의 동적 특징 이웃과 R17 ViG의 영상 특징 노드를 참고해, 공간 인접성과 동적 특징 이웃을 함께 사용하는 그래프를 우선 연구 후보로 선택했다. 본 CT에서 두 관계를 결합하는 것은 프로젝트 가설이다. 기존 철회된 1,728노드·순수 feature-kNN 제안을 복구하거나 production 기본값으로 채택한 것이 아니다. 노드 해상도·이웃 수·표집 규모와 실행 성능은 미확정이다.

- **R24 — Landrieu & Boussaha. Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning. 2019.** [원문](https://arxiv.org/abs/1904.02113). 학습된 특징을 이용한 superpoint 분할의 비교 근거다. 별도 supervised oversegmentation/metric-learning 목적을 사용하므로 현재 CNN-only 문맥 모델에 추가 가정 없이 그대로 적용할 수 있다고 주장하지 않는다.

## 학습 기반 그래프 풀링 — 원문 및 설치 코드 확인 (2026-09-23)

사용자의 Graph U-Net 풀링 제안에 따라 원문 본문을 확인했다. 상세 [원문·코드 대조와 우선 선택](docs/graph_pooling_evidence_20260923.md). 첫 구현 비교 기준은 SAGPool, 국소 집계·연결 보존 대안은 ASAP다. 아직 production에는 적용하지 않았다.

- **R25 — Gao & Ji. Graph U-Nets. 2019.** [원문](https://arxiv.org/abs/1905.05178), §3.1 Eq.2의 학습 투영 점수·top-k·sigmoid gate, §3.2–3.3의 gUnpool/skip, §3.4 Eq.4의 graph power를 확인했다. hard top-k 인덱스 자체가 미분 가능한 것이 아니라 게이트를 통해 투영 벡터를 학습한다.
- **R26 — Lee, Lee & Kang. Self-Attention Graph Pooling. 2019.** [원문](https://arxiv.org/abs/1904.08082), §3.1 Eq.3–5의 GNN 점수·선택·게이트·유도 부분 그래프, §3.2 Fig.2의 단계별 mean/max readout과 합산을 확인했다. 본 CT의 소형 종양 보존이나 최적 유지 비율을 입증하는 결과는 아니다.
- **R27 — Ying et al. Hierarchical Graph Representation Learning with Differentiable Pooling. 2018.** [원문](https://arxiv.org/abs/1806.08804), §3.2 Eq.3–6의 학습 soft assignment, `S.T Z`, `S.T A S` 확인. 수만 후보에서 dense assignment/연결 비용 때문에 우선순위를 낮췄으며 본 장비에서 OOM을 직접 측정한 것은 아니다.
- **R28 — ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations.** [원문](https://arxiv.org/abs/1911.07979), §4.1–4.4의 국소 attention membership, LEConv fitness, 선택 후 sparse graph coarsening을 확인했다. 학습된 주변 집계라는 비교 근거이며 수동 기하/특징 임계값 분할의 재도입이 아니다.
- 설치 PyG 2.6.1 `sag_pool.py`와 `asap.py`의 forward를 읽었다. ASAP에서 반환된 학습 엣지 가중치를 후속 GNN까지 전달해야 하며, 현재 L0에는 그 통합이 없다. 단순 import 가능 여부를 구현 완료로 취급하지 않는다.

### 2026-09-23 U-Net형 encoder–decoder 구현 연결

사용자 정정에 따라 R25의 다단계 pool/unpool, 저장된 index, 대응 skip 구조를 실제 U자형 경로로 구현했다. 설치 PyG2.6.1 GraphUNet의 down/pool/permutation/up/skip 경로를 확인했다. `hiercp_v222/l0_graph_unet.py`는 기존 GAT와 R26 SAGPooling(GCNConv)을 사용한 변형으로 encoder3단계, pool2개, unpool/skip/decoder2단계다. **원문의 GCN/gPool/A² 구성 재현은 아니며** induced coarse graph가 분리될 수 있다는 한계를 숨기지 않는다. [실제 CT 검증과 차이](docs/l0_graph_unet_20260923.md).

### 2026-09-23 L0 비교 구현 연결

R26의 학습 scorer/top-k/gate/유도 부분 그래프를 `hiercp_v222/l0_comparison.py`에서 PyG2.6.1 SAGPooling(GCNConv)으로 사용했다. early_sag는 첫 GAT 이전, late_sag는 GAT1층 이후이며 기존 GAT총3층과 L1/L2는 유지한다. PyG forward 대비 값·gradient·선택 엣지 동등성을 검증했다. R25의 전체 Graph U-Net decoder/unpool 또는 논문의 전체 분류 모델 재현은 아니다. PPR는 CNN 비용 기반 PPR/degree의 결정적 상위K 비교이며 A*와 질량95% 선택을 사용하지 않는다. [구현/실측/미검증 항목](docs/l0_comparison_implementation_20260923.md).

## 성장형 탐색 및 대안 L0 — 아이디어 비교 (2026-09-24, 미구현)

사용자의 새 제안은 가까운 CT 문맥에서 시작해 관찰한 특징으로 다음 위치·방향·분기를 결정하고, 탐색 과정에서 노드와 연결을 형성하는 방식이다. 미리 채운 고정 격자의 부분집합을 고르거나 GAT 가중치만 바꾸는 것으로 이 요구를 구현했다고 설명하지 않는다. 아래는 설계 참고이며 모델 기본값·학습 계약을 변경하지 않았다.

- **R29 — Mnih, Heess, Graves & Kavukcuoglu. Recurrent Models of Visual Attention. 2014.** [원문](https://arxiv.org/abs/1406.6247), §3.1. 누적 관찰 상태를 이용해 다음 glimpse 위치를 결정한다. 신경망에는 역전파, 비미분 위치 선택에는 policy gradient를 사용한다. 순차적 관찰 정책의 근거이며 3D CT 분기 그래프의 검증 논문은 아니다. 사용자 제안의 방향·분기·종료 및 관계 그래프 통합은 별도 연구 설계다.
- **R30 — Cosmo et al. Differentiable Graph Module (DGM) for Graph Convolutional Networks.** [원문](https://arxiv.org/abs/2002.04999), §IV. 특징 임베딩으로 연결 확률을 학습하고 연속 가중 그래프 또는 Gumbel-Top-k 이산 연결을 만든다. 주어진 노드 간 구조 학습이며 새 CT 영역을 관찰하거나 공간 노드를 성장시키는 모듈은 아니다. 원문의 pairwise 계산 비용과 별도 연결 학습 목적을 고려해야 한다.
- **R31 — Dai et al. Deformable Convolutional Networks. 2017.** [원문](https://arxiv.org/abs/1703.06211), §2.1. 입력 특징으로 표집 offset을 예측하고 불규칙한 위치의 특징을 보간해 읽는다. 원문 주요 연산/실험은 2D이며 그래프 생성 모델이 아니다. 여러 3D 관찰 위치를 병렬 예측하고 위치별 특징을 노드로 연결하는 L0는 이 개념을 확장한 프로젝트 가설이다. 이 보간은 선택한 연속 좌표에서 특징을 읽는 연산이며 기존 1,728개 특징 위치를 45,385개 고정 후보로 늘리는 정책을 재승인하는 근거가 아니다.
- **R16/R10 DGCNN 재확인:** §3.2에서 각 층의 특징 공간 kNN으로 연결을 재계산한다. 노드 위치의 성장과 구분하며, 특징으로 연결만 바꾸는 비교군으로 사용할 수 있다. 기존 초기 CNN 진단 시각화는 학습된 DGCNN 결과가 아니다.
- **R27 DiffPool 재확인:** §3.2의 soft assignment로 특징과 연결을 함께 집계한다. §4.3은 떨어진 노드가 같은 군집에 들어갈 수 있다고 명시한다. 따라서 학습 군집을 곧바로 공간적으로 연결된 CT 조직 영역이라고 부르면 안 된다. 영역 집계형 L0 후보이며 새 관찰 위치를 찾아가는 성장형과 다르다. Dense assignment/coarse adjacency 비용과 소형 병변 특징의 혼합 위험을 평가해야 한다.

비교 아이디어는 성장형 탐색, 병렬 위치 학습, 학습 군집 집계, 특징 기반 동적 연결이다. 우선 성장형과 병렬 위치 학습의 비교는 순차적으로 새 관찰을 반영하는 이점과 실행 병렬성 사이의 차이를 확인하는 데 목적이 있다. 모든 방법의 CT 성능·최종 노드 수·학습 규모·계산 효율은 미검증이며 임의의 작은 기본값을 확정하지 않았다.

## v2.2 CNN L0 기준선 (2026-09-24)

- **R32 — Çiçek et al. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 2016.** [원문](https://arxiv.org/abs/1606.06650), §2. 3D 합성곱·다중 해상도 encoder/decoder·skip의 근거. 원문은 Xenopus kidney microscopy이며 본 CT의 CP 선택 성능을 검증한 것이 아니다.
- R7 nnU-Net과 프로젝트에 이미 생성된 `nnUNetResEncUNetMPlans.json`을 구조 출처로 사용한다. 이번 `CNNLocalEncoder`는 plan의 ResidualEncoder 6단계/채널/블록 전체를 직접 사용한다. 원 segmentation decoder는 만들지 않으며 기존 L0 48³ CT와128D readout을 적용한 별도 기준선이다. 원 nnU-Net의128³ segmentation 입력·정규화·학습·성능을 재현했다고 주장하지 않는다. 설치 library 버전과 실제 encoder 소스 SHA는 실행 started.json에 기록했다.
- [구현 및 실제 한 케이스 검증](docs/v22_cnn_l0_20260924.md). 초기 가중치 실행이며 learned feature/saliency나 CP 성능 개선의 증거로 사용하지 않는다.

## v2.2 관측 종양 순위 학습 — 2026-09-27

- **R33 — Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback.** [원문](https://arxiv.org/abs/1205.2618), §3–4. 관측된 양성과 미관측 항목의 상대 순위를 학습하며, 미관측끼리의 정답 순서를 알 수 없다는 점을 확인했다. 본 프로젝트는 `softplus(-(s_positive-s_unobserved))`를 같은 CT의 위치에 적용하고 기존 관측 CE/L2를 유지하는 변형이다. 원래 사용자–상품 MF/BPR 학습이나 종양 CP 성능을 재현했다고 주장하지 않는다. Epoch detached L0 reference와 unit loss weights는 프로젝트 구현 선택이며 원 논문에서 검증한 설정이 아니다.
- **R34 — Imagining the Unseen: Generative Location Modeling for Object Placement.** [원문](https://arxiv.org/abs/2410.13564), §3.1–3.2. 희소 양성 위치로 여러 가능한 삽입 위치를 학습하고, 명시적 negative annotation이 있을 때 선호학습을 사용하는 논의 참고. 미관측 위치를 전부 부적합으로 간주하는 문제의 근거다. 해당 논문의 generative transformer/DPO를 이 GNN에 구현한 것은 아니며 CT 종양 CP의 타당성 근거로 전용하지 않는다.

연결 코드: `tools/v22_rank_objective.py`, `tools/v22_ranking_training.py`, `tools/v22_rank_recommendation.py`. 구체적인 목표·근사·DEBUG 검증·미완료 항목은 [v2.2 순위 학습 기록](docs/v22_observed_ranking_20260927.md)을 따른다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 이번 변경에서 OOM은 없었으며 모델 축소를 적용하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 검사는 DEBUG로 구분한다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
