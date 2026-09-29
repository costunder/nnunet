# L0 8pair 진단과 별도 비용 경로

2026-09-29. **초기 프로파일 admission 거부는 구현 오류 또는 EZ-SP 자체 실패 판정이 아니다.** 사용자 요청에 따라 상한·reg·min_size를 조정하지 않고 진단을 확장했다. Production 가능 상태 아님.

## 보존과 변경 범위

변경 전 adapter·공식 vendor·프로파일·9개 검사 소스/결과·이전 실제 측정 23파일을 `versions/v2.22/ezsp_before_diagnostics_20260929/adapter_and_9_tests.zip`과 `manifest.json`에 보존했다. 최종 검사에서 ZIP CRC 및 모든 저장 파일 SHA256을 다시 확인했다. 이전 결과 파일을 덮어쓰지 않았다.

`partition.py`에 역할/shell별 원본·coarse 연결성분, cluster mass histogram/quantile, bbox·분산 초과 cluster 수와 fine mass 비율, 종료 상태와 공식 함수 시간을 추가했다. 종료 상태는 반환된 partition의 adjacency/energy 및 공식 depth로 검사한 조건이며 커널 내부의 추적하지 않은 shell별 iteration 횟수를 지어내지 않는다. 병합 시간은 role별 공식 함수 호출 시간이다. 하나의 호출에 들어간 여러 shell에 시간을 임의 배분하지 않았다.

`validation.py`는 CT·grid·mm 좌표·역할·pair ownership·canonical ID·fine node/edge coverage manifest·finite·edge index를 검사한다. 이 검사는 비용 진단에서도 필수다. Manifest 누락도 거부한다. 그룹 혼합, 불연결 cluster 및 mass coverage 손실은 미검증 상한과 분리된 hard error다.

`diagnostic.py`의 별도 `DiagnosticEZSPEncoder`만 초기 bbox/분산/node/edge 프로파일 초과를 기록한 채 계산을 계속할 수 있다. partition/feature/edge를 수리하거나 제거하지 않는다. Production encoder의 admission 거부 동작은 유지한다. 별도 진단 인스턴스의 checkpoint state export는 거부한다.

## 8pair 엄격한 admission 기록

원시 데이터는 [report.json](../validation/l0_ezsp_diagnostics_20260929/report.json), **역할·shell 72행과 cluster 크기 전체 histogram은 [pairs.md](../validation/l0_ezsp_diagnostics_20260929/pairs.md)**에 있다. record별 donor/recipient, 압축 전후 N/E, 빈 shell, bbox/분산 초과 비율, 연결성분 및 종료 조건을 모두 기록했다. 첫 프로파일 위반에서 다음 pair 진단을 중단하지 않았다.

| Record | Fine N / E | Scale1 N / E | Strict scale2 |
|---|---:|---:|---|
| liver_66:19 | 7,955 / 317,595 | 1,611 / 39,309 | NOT_RUN |
| liver_66:1 | 4,420 / 245,811 | 951 / 24,694 | NOT_RUN |
| liver_75:44 | 9,087 / 307,392 | 5,155 / 140,592 | NOT_RUN |
| liver_75:2 | 4,716 / 293,502 | 1,603 / 59,909 | NOT_RUN |
| liver_72:84 | 5,847 / 433,940 | 1,271 / 52,838 | NOT_RUN |
| liver_72:1 | 8,014 / 498,971 | 1,896 / 58,048 | NOT_RUN |
| liver_71:91 | 5,501 / 282,143 | 3,306 / 97,478 | NOT_RUN |
| liver_71:0 | 7,126 / 382,248 | 2,064 / 61,063 | NOT_RUN |

가중치는 **seed42 신규 초기화**, 학습된 checkpoint가 아니다. CNN/GAT parameter hash를 report에 별도 기록했다. Reg1/2=0.02는 이전 측정과 같은 명시적 후보이며 자동 탐색/조정하지 않았다. 1024/32768 및 기존 bbox/분산 상한은 미검증 초기 프로파일 그대로다.

## 복제 상태의 전체 update 비용

두 모델의 초기 전체 parameter hash가 같음을 확인했다. 각 측정 update는 CPU master의 독립 복제에서 시작하며 master hash가 바뀌지 않았는지 검사한다. 단기 측정 결과를 model checkpoint로 저장하지 않는다.

- RTX5070Ti16GB / torch2.8.0+cu128 / FP32 / CPU workers8.
- 실제 DEBUG inner_train8개 전체를 **동일 physical batch8**의 disjoint-union L0로 처리. accumulation1/effective8. Production physical batch 설정은 변경하지 않았다.
- 각 recipient group의 donor/recipient 제외를 그대로 적용한 기존 L1/L2와 `tools/v22_rank_objective.forward_loss`를 호출한다. 4개 recipient task별 기존 rank/auxiliary/alignment loss를 record 수로 가중 평균하고, 한 번 backward·기존 grad clip5·AdamW update를 실행한다. 전체 production support11,279개나 production의 단일 recipient batch와 동일한 실험이라고 주장하지 않는다.
- 명시적 자원 예산: PyTorch CUDA allocator6GiB, RSS12GiB, 180초. RSS/시간은 phase 경계에서 검사하는 cooperative 한도이며 실행 중인 kernel을 강제 종료하는 hard deadline은 아니다. 실제 update peak allocation은 두 경로 모두 한도 이내였고 종료 RSS는 약2.28GiB.
- 비용 경로의 scale2는 **별도 diagnostic-only 실행 결과**다. strict admission에서 멈춘 8pair의 scale2=NOT_RUN은 수정하지 않는다. 위반 pair를 정상 학습 sample/통과 sample로 표시하지 않는다.

두 번의 독립 복제 update 산술평균:

| 항목 | 기존 fine GAT | EZ-SP 비용 진단 |
|---|---:|---:|
| 전체 update | 2.649초 | 4.812초 |
| forward(기존 L1/L2·목표 포함) | 0.684초 | 4.071초 |
| backward | 1.886초 | 0.682초 |
| 공식 partition 함수 합계 | 해당 없음 | 3.160초 |
| memory refresh(별도) | 0.354초 | 3.087초 |
| support cluster plan(별도) | 0.086초 | 0.091초 |
| update peak allocated 평균 | 1.007GiB | 0.498GiB |

**이 짧은 측정에서 EZ-SP 비용 경로는 메모리를 덜 사용하지만 더 느렸다.** Update 시간에 memory refresh·plan·데이터 읽기·체크포인트 저장을 숨겨 포함하거나 제외 사실을 생략하지 않았다. 위 표에서 별도라고 표시한 비용은 update 밖이다. 공식 partition 시간과 구조 검사/quotient/GAT/전체 update 시간을 분리했다. 두 번의 짧은 실행이고 실행 순서는 baseline 후 EZ-SP이므로 통계적 속도 보장이나 서버 epoch 추정에 사용하지 않는다.

같은 초기 복제 상태의 EZ-SP 두 측정에서 scale2 partition과 loss가 달라졌다(1.912073/1.913239). 실제 GPU 경로의 반복 동일성은 확보됐다고 표시하지 않는다. 원인을 이 진단만으로 확정하지 않았으며, 이전 합성 repeat 검사를 실제 데이터의 결정론 보장으로 확대하지 않는다.

최종 report는 측정 당시 소스 해시와 실행 중 불변 여부를 저장했다. 측정 후 강화한 coverage manifest 필수/ID 일치 검사는 단위 검사에 포함된다. 원시 report의 당시 source hash를 현재 코드로 덮어쓰지 않았다.

## GAT / GraphSAGE / GraphSAINT / Cluster-GCN 검토

GAT를 CP 추천에 필수라고 입증한 비교는 없다. 기존 경로의 edge별 attention·중간 tensor·재계산 비용 문제가 있었으며, 현재 custom gated GAT가 가장 적합하다고 단정할 근거도 없다. 그러나 이번 EZ-SP 경로에서는 공식 partition3.160초에 비해 3 GAT block forward는 약0.150초였다. GAT backward만의 시간은 따로 분리하지 않았다. GAT 교체만으로 현재 병합 비용까지 해소되는 것은 아니다.

- **GraphSAGE mean aggregation**: 동일 graph/coverage에서 attention 연산을 집계 연산으로 바꾸는 직접적인 encoder 비교 후보. 이웃별 attention이 없어지는 표현력 차이는 효용 평가가 필요하다. neighbor sampling을 함께 도입할 필요는 없으며, 첫 비교에서는 기존 node/edge 전체를 유지해야 변경 효과를 분리할 수 있다.
- **GraphSAINT**: training graph의 subgraph sampling/정규화 방법. GAT를 포함한 여러 encoder와 결합할 수 있다. 단순한 GAT layer 대체가 아니며 우리 context coverage가 달라지는 별도 축이다.
- **Cluster-GCN**: 그래프 분할을 이용한 mini-batch GCN 학습 방법. sample별 문맥 readout과 cluster 경계 연결의 처리 계약을 별도로 정해야 한다. EZ-SP의 feature 기반 supernode 병합과 같은 작업이 아니다.

추천 비교 순서는 동일 graph의 GAT 대 mean GraphSAGE, 이후 필요한 경우 sampling/partition 학습 방식 비교다. 이는 설계 판단이며 우리 CP 정확도 우위를 확인한 결과가 아니다. 이번 요청에서는 GAT를 교체하지 않았다.

### References

- Hamilton et al., [Inductive Representation Learning on Large Graphs, NeurIPS2017](https://arxiv.org/abs/1706.02216).
- Zeng et al., [GraphSAINT, ICLR2020](https://arxiv.org/abs/1907.04931).
- Chiang et al., [Cluster-GCN, KDD2019](https://arxiv.org/abs/1905.07953).
- 공식 병합 소스: [torch-graph-components 고정 commit](https://github.com/drprojects/torch-graph-components/blob/e3db9f352fae52dff416616742b3c7ff1378451d/src/torch_graph_components/merge.py).

## 실행 및 검증

```powershell
.venv\Scripts\python.exe -m unittest tests.test_l0_ezsp_debug tests.test_l0_ezsp_diagnostic_debug -v
.venv\Scripts\python.exe tools/diagnose_l0_ezsp_pairs.py --cache work/v22_rereview_20260927_DEBUG/cache/index.json --output work/NEW_UNIQUE_DIAGNOSTIC_OUTPUT --reg-scale1 0.02 --reg-scale2 0.02 --physical-batch 8 --workers 8 --updates 2 --cuda-gib 6 --rss-gib 12 --seconds 180
```

기존9+새4 = **13개 통과**, skip0. Profile 위반 시 partition/설정 불변, hard integrity 거부8종, 자원한도 거부, diagnostic checkpoint 금지, fine-mass 초과 비율 검사 포함. 기록: `validation/l0_ezsp_diagnostics_20260929/summary.json`, `tests.txt`.

장기 GNN/nnU-Net 학습·전체 평가·production checkpoint·ready 표시 없음. BasicCP/L1/L2/loss/후보128/원본 mask 변경 없음. 서버 MIG/full support/epoch 시간 및 CP 효용은 미검증이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 별도 명시적 DEBUG 자료 전체8개만 측정.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 동일 batch8 비교.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. DEBUG 실제 pair 범위.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
