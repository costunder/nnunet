# L0 조기 선택 / 지연 풀링 비교 구현

2026-09-23. 사용자 승인 범위: 처음부터 적은 노드를 GNN에 넣는 방식과, 큰 그래프에서 문맥을 학습한 뒤 풀링하는 방식을 실제 비교 가능하게 구현한다.

## 구현된 경로

공통 입력은 기존 CT 48³, 원래 physical FOV, 45,385개 후보/2,383,482개 방향 엣지다. CNN은 12 base channels / 32 output channels, graph hidden128 / heads4 / GAT3층이며 L1 2층과 L2 2층을 그대로 연결한다. 기존 6mm 공간 연결과 2.5mm 후보 간격도 유지한다. 후보 위치는 관측 단위가 아니며 12³ CNN 특징의 보간 위치다.

| mode | GAT 1층 입력 N | GAT 2층 입력 N | GAT 3층 입력 N | 선택 |
|---|---:|---:|---:|---|
| cnn_only | 없음 | 없음 | 없음 | CNN 특징의 학습 readout → 기존 L1/L2 |
| full_graph | 45,385 | 45,385 | 45,385 | 전수 대조군 |
| early_ppr | K | K | K | CNN 특징 기반 비용 → 중심 PPR/degree 상위 K |
| early_sag | K | K | K | CNN 특징 투영 → 학습 GCN 점수 → 상위 K와 gate |
| late_sag | 45,385 | K | K | GAT 1층 문맥 → 같은 학습 GCN 점수 → 상위 K와 gate |

early_sag/late_sag는 같은 scorer와 같은 유지 수로 선택 시점의 효과를 비교한다. early_ppr/early_sag는 같은 선택 시점에서 선택 방법을 비교한다. late_sag의 scorer에도 추가 1회 GCN 메시지 전달이 있다. 기존 GAT 3층을 줄인 것이 아니다.

**‘처음부터 적게’는 여기서 첫 GAT 이전 선택을 뜻한다.** 후보 특징/PPR 또는 scorer 계산 비용은 여전히 있으며 시간에서 빼지 않았다. 원래부터 작은 후보 격자를 만드는 입력 해상도 비교는 이번 구현과 다른 실험이다. 45,385개 후보의 보간 중복 문제를 해결했다고 주장하지 않는다. 현재 후보를 공통으로 고정한 것은 선택 시점과 해상도의 효과를 섞지 않기 위해서다.

## 코드와 사용

- `hiercp_v222/l0_comparison.py`: ComparisonEncoder와 comparison_model. 모델은 DEBUG 전용 축소 모델이 아니며 위 전체 깊이·폭을 사용한다. K는 필수 명시 인자이고 숨겨진 cap/자동 fallback이 없다.
- `hiercp_v222/model.py`: PromptGraphModel에 선택적 local_encoder 주입만 추가. 기존 생성 호출, L1/L2, 손실은 유지한다. 사용되지 않는 구 L0를 생성해서 남겨두지 않는다.
- `config/l0_comparison_debug.json`: 실제 CT 실행 검증/지연 시간 측정을 위한 별도 DEBUG 프로필.
- `tools/benchmark_l0_comparison_debug.py`: 5개 경로, 두 K, 3개 physical batch, 실제 CT gradient/optimizer 검증, CSV/JSON/선택 그래프 NPZ 기록.
- `tests/test_l0_comparison_debug.py`: PyG SAG 결과/gradient/유도 엣지 동등성, 배치 독립성, 선택 시점, 초기 가중치 일치, scorer 갱신 검증.

재실행은 새로운 결과 디렉터리를 지정한다.

```powershell
.\.venv\Scripts\python.exe tools/benchmark_l0_comparison_debug.py --output work/l0_comparison_debug_next
```

새 comparison_model은 기존 L1/L2 forward와 supervised_loss를 사용하는 학습 가능한 모델이다. 이번 제공 runner는 비교 구현의 검증/벤치마크용이며 전체 코호트 학습·checkpoint 승격·nnU-Net 배치 비교 실행기는 아니다. r4 production 학습 차단을 우회하지 않는다. 최종 유지 노드 수, 전체 학습과 분할 성능 비교는 아직 실행하지 않았다.

## 알고리즘과 병렬 실행

SAG는 설치 PyG2.6.1 `SAGPooling(..., GNN=GCNConv)`의 scorer/SelectTopK를 사용한다. 원문의 정규화 GCN 형태와 일치하는 scorer이며 PyG 기본 GraphConv를 그대로 쓰는 설정과 구분한다. 선택 특징에 score gate를 곱하므로 query CE가 scorer까지 역전파된다. hard 선택 인덱스 자체를 미분한다고 주장하지 않는다. 전체 Graph U-Net decoder/unpool은 구현하지 않았다.

scorer는 disjoint-union 그래프 batch, GAT는 전체 batch를 병렬 처리하는 padded adjacency를 사용한다. 실행 tile은 모든 노드와 엣지를 방문하며 노드/엣지 수 제한이 아니다. 후보 topology, feature interpolation table, scorer의 batch edge index를 재사용한다. full_graph/PPR 모델 1,519,063params, SAG 모델 1,519,193params, CNN-only 모델 973,015params다.

선택한 노드 사이의 기존 공간 엣지를 모두 보존한다. 편의상 A* 연결, halo, 연결 성분 강제 복구, 상위 이웃 수 제한을 추가하지 않는다. 분리된 성분과 고립 노드는 감사 결과에 명시한다. Query 정답/마스크/patient ID는 L0 forward에 입력하지 않는다. support/query는 서로 다른 case다. 동일 환자의 재검사 여부에 대한 기존 공개 case 단위 분할의 한계는 유지한다.

## 실제 CT 측정

검증본: `work/l0_comparison_debug_20260923_verified/`. RTX5070Ti 16GB, CPU8 physical/16 logical, RAM68.64GB 중 시작 시 약49.5GB available. CPU/IO 각8 threads, 12개 고유 실제 CT 패치를 SHA 확인 후 병렬 로딩. support: liver_1/liver_5 각 두 관측 클래스, query: liver_108의 8개 패치. 전체 코호트 학습으로 간주하지 않는다.

K=1,728과6,912는 CNN 최종 공간 위치 수의1배/4배에 해당하는 명시적 DEBUG 비교 후보다. 최적값이나 production 기본값으로 확정하지 않았다. physical batch2/4/8, inference warmup1회 후3회 중앙값. gradient smoke는 각 구조당 실제 query2개로1 optimizer step이며 별도 기록한다. gradient accumulation1, 단일 GPU다. batch8이 최대 안정/최적 batch라는 주장은 하지 않는다.

아래는 **batch8의 L0 전체 forward** 시간이다. CNN, 후보 특징, PPR/SAG 선택과 유도 엣지 생성, 모든 GAT, readout을 포함한다. CT 전처리, 파일 I/O, H2D, full-cohort support 갱신, L1/L2/CP/nnU-Net 총 추론 시간은 이 표에 포함하지 않는다. cold topology와 I/O/H2D 및 cold forward는 별도 필드에 기록한다.

| mode | K | batch8 중앙값 ms | graphs/s | peak allocated VRAM GiB |
|---|---:|---:|---:|---:|
| cnn_only | 45,385 readout 위치 | 12.24 | 653.45 | 0.597 |
| full_graph | 45,385 | 467.97 | 17.09 | 2.070 |
| early_ppr | 1,728 | 97.14 | 82.36 | 1.315 |
| early_sag | 1,728 | 64.99 | 123.10 | 2.471 |
| late_sag | 1,728 | 212.45 | 37.66 | 2.471 |
| early_ppr | 6,912 | 152.33 | 52.52 | 1.759 |
| early_sag | 6,912 | 122.04 | 65.55 | 2.471 |
| late_sag | 6,912 | 246.42 | 32.47 | 2.471 |

풀링으로 후속 계산은 감소하지만 scorer의 전체 batch sparse 연결 처리가 추가되어 peak VRAM은 전수 GAT보다 클 수 있다. 실제로 이번 SAG 측정이 그러하다. latency만 보고 메모리도 감소했다고 주장하지 않는다. 초기 가중치 결과이므로 노드 선택·연결 수와 시간은 학습 후 달라질 수 있다.

## 실제 선택 그래프와 검증 한계

각 mode/K의 `*_actual_graphs.npz`는 실제 선택 ID/score/packed adjacency/좌표를 담는다. `*_graph_audit.json`은 다음 사실을 남긴다.

- 초기 early_ppr K1728: 두 CT 선택 집합 Jaccard0.9885. 중심에서 최대18.37~18.71mm 이내, 각1연결 성분. K6912도 최대29.69mm. 원래55.29mm FOV 전체를 고르게 보존하는 표집으로 설명하면 안 된다.
- 초기 early_sag K1728: 각1/3연결 성분. 두 CT 선택 집합 Jaccard0.000579.
- 초기 late_sag K1728: 각5/9연결 성분, 최대 성분 점유율0.922/0.512. K6912에서는 각1고립 노드가 확인됐다.
- 영상에 따라 선택이 달라지는 것은 확인했지만 미학습 scorer의 선택이 종양/주변 문맥에 유효하다는 근거가 아니다. top-k 분리 성분은 숨기지 않고 비교 지표로 남긴다. 연결 강제 복원으로 결과를 바꾸지 않는다.

전체 모델 gradient smoke에서 모든 trainable parameter의 gradient 존재/유한값을 확인했다. CNN, projection, 3 GAT/update, readout, L1, L2/update, label seed 및 SAG scorer가 실제 optimizer update에 연결됨을 확인했다. CNN-only에서는 제거한 GAT 파라미터가 남아 있지 않다. support memory는 detach되어 있으므로 alignment loss가 query CNN까지 직접 역전파된다고 주장하지 않는다.

신규4개 + 기존 leakage12개 + PPR/기존 실행10개 + clustering8개 = **34개 테스트 통과**. clustering 테스트는 기존 파일의 top-level import 때문에 `unittest discover -s tests -p test_v222_cluster_debug.py`로 실행한다. 초기 module-name 묶음 호출의 import 실패와 이후 정상 discovery 실행을 구분한다.

첫 실행은 출력 디렉터리 권한, 다음 실행은 Windows psutil Path 타입 오류로 실패했다. 승인된 workspace 실행과 str 경로 수정으로 해결했다. 성공한 run2를 보존하고 선택 그래프 감사와 메모리 acceptance 기록을 더한 verified 실행을 따로 생성했다. 임의 삭제/덮어쓰기로 실패 이력을 지우지 않았다.

## 근거와 남은 평가

REFERENCES R25 Graph U-Nets, R26 SAGPool, R18 PPR를 사용한다. SAG 전체 논문의 다단계 분류 아키텍처 재현 또는 CT 유효성 입증으로 표현하지 않는다. 이번 모델은 기존 L1/L2에 맞춘 단일 풀링 위치 비교다.

전체 코호트에서 동일 split/seed42/CP80와 기존 학습 예산을 사용하는 성능 비교는 아직 없다. 소형 병변 recall/precision/F1, lesion Dice, voxel Tumor Dice와 실제 전체 CP 처리량을 함께 평가한 뒤 유지 수와 위치를 선택해야 한다. 현재 DEBUG 속도만으로 더 늦게 풀링하는 방식의 정확도 이득 또는 우승 모델을 정하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. CNN-only는 명시적 대조군이다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. K는 승인된 비교 변수, CT subset은 명시적 DEBUG다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 2/4/8을 측정했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM이 발생하지 않았으며 메모리 사전 admission을 유지했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 테스트와 초기 가중치는 명시했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
