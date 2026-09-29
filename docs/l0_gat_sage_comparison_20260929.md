# 동일 fine graph의 GAT / GraphSAGE 실제 GPU 비교

2026-09-29. 사용자 요청에 따라 별도 GraphSAGE mean 경로를 구현하고 기존 GAT와 직접 비교했다. **로컬 DEBUG 비용 검증이며 최종 CP 성능 검증이나 production 전환이 아니다.**

## 구현과 비교 통제

`l0_sage/encoder.py`는 기존 L0의 **3개 block ×13개 relation convolution만** GraphSAGE mean/root 연산으로 교체한다. 각 relation의 출력은 기존처럼 합산한다. CNN12/24/32, stride4 좌표, projection128, 3층/128D, residual·LayerNorm·FFN·dropout, 역할·shell readout, fuser와 최종128D는 유지한다. L1/L2 및 rank/auxiliary/alignment loss 코드는 수정하지 않는다.

GAT의4 attention heads는 그대로 baseline에 있고, GraphSAGE에는 attention heads 및 attention dropout이 없다. 이는 비교 대상 연산의 차이이며 숨긴 너비 축소가 아니다. 두 모델의 공통 parameter310개가 동일한지 검사했다. 교체되지 않은 CNN·readout·L1/L2는 동일 가중치다.

표준 PyG SAGEConv의 `W_neighbor mean(x_neighbor) + W_root x_target`를 사용한다. CSR `D^-1 A`에 대한 sum으로 mean을 계산해 edge×hidden 크기의 메시지 tensor 생성을 피한다. 중복 edge는 기여도를 합쳐 원래 multiplicity를 보존한다. 이웃 sampling, coarsening, node/edge cap, 새로운 기하 입력은 없다. 평균 인접행렬은 현재 배치에 대해 한 번 준비해 3층·checkpoint backward에서 재사용한다. edge 내용/identity/규모 변경 시 무효화하고, 과거 여러 배치를 무제한 보유하지 않는다.

PyG 표준 `SAGEConv(aggr='mean')`의 COO 실행과 새 CSR 실행의 출력·입력 gradient·모든 가중치 gradient를 CPU/CUDA에서 비교했다. 중복 edge, 고립 destination, bipartite, 빈 edge도 검사했다. 빈 target context와 단독/배치 출력 동등성도 검사했다. 이 구현은 현재 **FP32 DEBUG**만 허용하고 미검증 AMP를 조용히 사용하지 않는다.

## 실제 측정 조건

- 실제 DEBUG cache inner_train8개 전체 / 4 recipient groups. Physical8, accumulation1/effective8. Production batch를 낮추거나 production observation을 줄인 것이 아니다.
- Fine graph **52,666 nodes /2,761,602 directed edges**, 두 경로에서 동일. CT48³, 같은 source/target patch·mask·좌표·모든13 relation 유지.
- RTX5070Ti16GB, CUDA GPU1개, torch2.8.0+cu128, PyG2.6.1, FP32, CPU workers8. CPU/RAM/GPU 시작 상태는 원본 report에 기록.
- CUDA allocator6GiB/RSS12GiB/180초의 명시적 DEBUG 예산. RSS/시간은 phase 경계 검사이며 세션/프로세스 강제 종료 없음.
- 각 arm 워밍업1update 후 **측정3update**. 실행 순서를 GAT→SAGE / SAGE→GAT로 번갈아 배치했다.
- 매 update는 seed42 초기 master의 독립 복제에서 시작. 실제 학습 checkpoint를 사용하지 않았고 master가 변하지 않았는지 hash를 검사했다. Master 초기값은 저장된 비교 report의 hash로 재생성 확인했다.
- L0는8개를 하나의 disjoint-union 배치로 인코딩한다. L1/L2는 기존 recipient/donor 제외 규칙을 적용한4개 recipient task에 기존 `observed_rank_v1.forward_loss`를 호출하고 record 수로 가중 평균한다. 기존 class weights·grad clip5·AdamW 설정을 사용한다. Production full-support11,279개나 production 단일 recipient batch와 동일한 작업량은 아니다.
- BasicCP, 후보128, 원본 mask와 paste 검사, EZ-SP adapter 및 이전 admission 프로파일은 변경하지 않았다.

## 결과

원본: [report.json](../validation/l0_gat_sage_20260929/report.json). 아래는 워밍업 제외3회 평균이다.

| 항목 | 기존 GAT | GraphSAGE mean |
|---|---:|---:|
| 전체 update | 2.283초 | 1.203초 |
| forward (L0+L1/L2+기존 loss) | 0.538초 | 0.511초 |
| backward | 1.681초 | 0.649초 |
| peak allocated | 1.010GiB | 0.772GiB |
| memory refresh (update 밖) | 0.247초 | 0.406초 |
| 전체 모델 parameter | 5,550,806 | 5,535,830 |
| L0 parameter | 4,718,420 | 4,703,444 |

이 측정에서는 전체 update **47.3% 단축(약1.90배 처리량)**, peak allocation 약23.5% 감소다. 주로 backward 비용이 줄었다. Parameter 차이14,976개는 convolution 종류 변경에서 생기며 깊이·hidden·CNN·그래프 규모를 줄인 결과가 아니다.

개별 update 범위는 GAT2.013–2.485초, SAGE1.176–1.245초다. SAGE 인접행렬 생성은 memory refresh 첫 실행에 포함되며 update에서 다시 만들지 않았다는 `adjacency_build_count=1`을 기록했다. **Memory refresh 자체는 이번 짧은 측정에서 더 느렸다.** 이를 숨기고 모든 단계가 빨라졌다고 주장하지 않는다. 전체 epoch에는 support refresh·validation·I/O·저장이 추가되므로 47.3%를 서버 epoch에 그대로 적용할 수 없다.

손실 값은 초기 GAT약1.812, SAGE약1.929다. 학습된 모델의 validation 점수가 아니므로 이 값으로 CP 정확도 우열을 판단하지 않는다. 연산 자체가 바뀌므로 GAT/SAGE 출력의 수치 일치를 요구하거나 주장하지 않는다.

## 검사 및 발견한 문제

GraphSAGE6개 + 기존 EZ-SP13개 = **19개 검사 통과**, skip0. 실제 비용 경로에서도 CNN·각SAGE/GAT block·L1·L2의 finite/nonzero gradient를 확인했다. 초기4개 통과 후 추가 RNG 회귀검사에서 PyG deepcopy가 CPU RNG를 소비하는 문제가 발견되어, 복사까지 fork_rng 범위에 포함했다. 실패 기록 `tests.txt`/`verification.json`은 보존했고 최종 통과는 `tests_final.txt`/`verification_final.json`에 별도 기록했다.

RNG 보존 수정은 update 경로 밖의 생성 과정 변경이다. 수정 후 두 master parameter hash가 실제 비용 측정 당시와 일치함을 확인했다. 원시 측정의 당시 소스 hash를 덮어쓰지 않았다. 기존 핵심8개 source anchor 해시도 그대로다. 최종 정적 검사와 단위 검사 완료, 실제 데이터 short update 완료, 전체 학습·전체 평가 미실행을 구분한다.

## 사용 경로와 남은 검증

```powershell
.venv\Scripts\python.exe -m unittest tests.test_l0_sage_debug tests.test_l0_ezsp_debug tests.test_l0_ezsp_diagnostic_debug -v
.venv\Scripts\python.exe tools/compare_l0_gat_sage_debug.py --cache work/v22_rereview_20260927_DEBUG/cache/index.json --output work/NEW_UNIQUE_GAT_SAGE_DEBUG --physical-batch 8 --workers 8 --warmup-updates 1 --repeats 3 --cuda-gib 6 --rss-gib 12 --seconds 180
```

Production 기본 encoder는 변경하지 않았다. 별도 SAGE 인스턴스의 checkpoint export는 금지한다. 이전 GAT checkpoint를 새 모델의 exact resume로 간주하지 않는다. BF16/AMP, MIG10GB, production physical32/full support, 전체 epoch 시간, 학습 수렴, CP 추천·segmentation 효용은 미검증이다. 장기 GNN/nnU-Net 학습을 시작하지 않았다. GraphSAINT/Cluster-GCN은 이번 비교에 섞지 않았다.

GraphSAGE는 지금 **계산 비용 측면의 유효한 비교 후보**다. 최종 모델을 선택하려면 성능을 별도로 평가해야 한다.

## References

- Hamilton et al., [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216).
- [PyG SAGEConv 공식 수식과 API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html). 웹 문서 버전과 별도로 실제 설치된 PyG2.6.1 소스의 forward/message_and_aggregate를 확인했다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 명시적 DEBUG cache 전체8개 사용.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 동일 disjoint-union batch8.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
