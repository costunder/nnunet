# L0 선행 병합 전달본 대조 — 2026-09-29

> 이후 상태: Reconciled ZIP을 실제로 수령·해제했고 정정 우선으로 구현/검사를 진행했다. [현재 구현·측정 기록](l0_ezsp_adapter_20260929.md)을 따른다. 아래는 ZIP 수령 전의 역사적 검토이며 현재 미수령/미구현 상태를 뜻하지 않는다.

상태: 두 첨부 전문, 활성 L0, 공식 논문 및 고정 커밋의 병합 함수 검토 완료. 새 encoder/partition 구현, GPU 병합 검사, 본학습은 미실행. 전달본이 언급한 CODEX_IMPLEMENTATION_SPEC.md와 ZIP/CPU 검사는 실제로 첨부되지 않았다. 본문에 적힌 CPU 8묶음 통과를 이 저장소에서 확인한 결과로 취급하지 않는다.

## 받은 자료

- 첫 첨부: d3e9aa03-380c-4f2f-8fbd-c5882654462b/붙여넣은 텍스트.txt, 16,629bytes. 9개 절 전체 확인.
- 두 번째 첨부: b1d1899f-d52c-4126-9943-e8077b89633c/붙여넣은 텍스트.txt, 8,240bytes. EZ-SP를 주 병합 기준으로 지정한 후속 설명 전체 확인.
- 두 폴더에는 위 텍스트만 존재. sandbox:/mnt/data 링크는 이 호스트 파일 경로가 아니다.

## 구현 계약으로 추출한 요구사항

| 범위 | 요구사항 | 확인/처리 |
|---|---|---|
| 입력 | 현재 CT CNN 유지, sampled 영상 특징32D 사용 | CTOnlyEncoder의 sample→project 사이에 첫 partition 필요 |
| 첫 병합 | 첫 GAT 전에 수행 | fine graph GAT 금지; fine geometry 조회 자체를 생략하는 것은 아님 |
| 2단계 | P1 GAT2층, P2 GAT1층 | 총3층/128D/4heads 유지 |
| 모델 출력 | 두 scale readout 동일 비중, 기존 pair fusion128D | fine 복원 decoder 추가 없음 |
| 병합 기준 | 후속 첨부의 EZ-SP contour-prior partition | Ward 단독/상호선택 구현을 EZ-SP 원형이라고 부르면 안 됨 |
| 격리 | pair/role/context shell 사이 병합 금지 | partition 입력 edge에서 분리; quotient 관계는 별도 유지 |
| 공간 연결 | 같은 역할의 원래 neighbor edge로만 병합 | isolated-node kNN 보완으로 새 edge 추가 금지 |
| 크기/오차 | component bbox와 feature dispersion 제약 | 본문에는 구체 수치·정의가 없음; 공식 API에도 해당 인자 없음 |
| gradient | assignment만 detach; aggregate feature는 live | 두 단계에서 fine-node mass 누적; gradient CNN까지 |
| checkpoint | partition은 GAT checkpoint 바깥 | backward 재분할 금지 |
| 비용 admission | P1 1024nodes/32768edges, P2 256nodes/8192edges | 역할·shell 세부 상한 포함. 잘라 버리는 cap이 아니라 실패 판정 조건 |
| 역할 상한 | P1 128/384/64/384/64, P2 32/96/16/96/16 | context shell별 P1 128/P2 32. 입력이 적으면 복제하지 않음 |
| quotient | 원래 directed relation을 parent ID로 매핑·중복 제거 | 13종 보존, 같은 cluster 내부 same-role edge 축약; 새 대응 생성 금지 |
| readout | softmax(g(h)+log mass) | 동일 feature 중복의 제한적 동등성만 주장 가능 |
| metadata | mm 좌표/original ID/shell 별도 sidecar | learned feature로 좌표/통계를 넣지 않음 |
| 좌표계 | stride4 sampler 명시 연결 | 기존 V1 __init__ patch 자동 상속을 가정하면 안 됨 |
| 공통 경로 | query/support 생성·갱신/validation/128후보 | 공통 factory·encoder 계약 필요 |
| cache | 학습 중 partition 영구 캐시 금지 | frozen 모델/content/view/geometry/config hash 모두 일치할 때만 재사용 |
| CP | paste mask·anchor·유효성 원형 유지 | 작은 representative mask로 바꾸지 않음 |
| L1/L2 | 기존 관측 rank/CE·L2 군집 그대로 | 새 pooling loss/별도 boundary network 없음 |
| 실험 조건 | Basic CP·후보128·physical batch·epoch 유지 | 새 L0의 노드/edge 예산은 명시적 설계 변경으로 기록 |
| 재개 | 기존 결과 보존, 새 run | CNN만 검증 후 명시적 초기화 가능, 나머지 자동 이전 없음 |
| 검증 | 실제 batch로 partition/GAT/L1L2/backward/저장 분리 | node 감소를 실제 update/epoch 속도로 바꾸어 보고하지 않음 |

## 실제로 확인한 차이 및 누락

1. **병합 알고리즘 차이.** 첫 첨부는 상호 선택한 서로 겹치지 않는 쌍만 병합한다. EZ-SP 논문과 공식 코드는 노드별 최선의 출발 edge를 선택하고 약연결성분을 병합한다. 연쇄 병합이 가능하다. 첫 첨부의 ‘chain 금지’와 동일하지 않다. 뒤의 EZ-SP 지정에 맞추더라도 앞의 bbox 제한까지 자동 충족되지는 않는다.
2. **목적함수 차이.** 첫 첨부 식은 weighted SSE 증가만이다. 공식 edge_merge_energy는 `weighted_SSE_increase - reg * boundary_weight`다. contour 항을 빠뜨리면 공식 기준을 반영한 구현이 아니다.
3. **공식 함수로 제약이 자동 보장되지 않는다.** pair/role/shell은 입력 adjacency에서 격리할 수 있다. 하지만 전체 bbox/분산 상한, P1/P2 대표 수·edge 상한 인자는 없다. 사후 검사 실패 처리 또는 명시적인 제약 확장이 필요하다. 아무 정책도 승인된 것처럼 임의 채택하지 않는다.
4. **min_size는 다른 조건을 무시하는 병합을 유발할 수 있다.** 공식 함수는 크기 미달인 component에 대해 energy 감소가 없어도 병합한다. 이를 ‘특징 손실 제한 보장’으로 해석하면 안 된다.
5. **관계 종류를 섞으면 의미가 달라진다.** partition의 same-role 무방향 adjacency와 GAT의 13종 방향성 quotient를 분리해야 한다. 공식 함수는 무방향 중복 edge weight를 합친다. 양방향 원본을 그대로 주고 W=1을 두 번 세면 contour regularization 강도가 달라진다.
6. **고립 노드 새 연결 금지.** 공식 함수의 `k>0` 옵션은 새 kNN edge를 만든다. 원래 edge 근거 요구와 충돌하므로 사용할 수 없다.
7. **shell 예시는 설정값이 아니다.** 두 번째 글의 2~8/8~16/16~28mm는 예시다. 실제 설정의 shell 경계는 4/12/28mm이며 3개 shell ID를 그대로 사용해야 한다. 반경 기준의 정확한 membership은 현재 metadata를 보존한다.
8. **2차 병합 특징을 명확히 해야 한다.** 최초32D CNN 특징의 mass 평균을 계속 partition 신호로 쓸지, GAT2층 후128D 특징을 쓸지 본문만으로 단정하지 않는다. 2차 projection을 임의 추가하지 않는다.
9. **필요한 수치가 미제공이다.** level별 reg/min_size, bbox 크기 기준(축별 최대/대각선 등), feature variance 허용값, boundary W 정의가 없다. 1024/256 예산을 맞추려고 자동으로 제약을 완화하면 안 된다.
10. **기존 cache provenance.** `hiercp_v222/v1_cache.py:provenance`는 관련 디렉터리의 모든 Python 파일을 해시한다. 새 모듈 추가만으로도 기존 cache 검사가 달라진다. 원형의 geometry/content를 검증하는 새 reuse 계약 없이 hash를 덮어쓰지 않는다.

## 공식 코드와 환경

공식 `torch-graph-components`의 확인 커밋: `e3db9f352fae52dff416616742b3c7ff1378451d`.

`src/torch_graph_components/merge.py` 전문을 읽었다. API는 X/S/E/W/reg/min_size 외에 merge_only_small/P/k/w_adjacency/depth/max_iterations/sharding/reduce/verbose를 받는다. 내부 연산은 순수 PyTorch/PyG/torch-scatter이며 ‘별도 custom CUDA kernel 하나’와 동일하지 않다. edge 중복 합치기 자체도 반복 비용이므로 실제 병합 시간을 계측해야 한다.

로컬 확인: RTX5070Ti16GiB, GPU 점유6819MiB/1% 관측, torch2.8.0+cu128, CPU physical8, available RAM32.4GiB. 다른 GPU 프로세스를 종료하지 않았다. torch_graph_components와 torch_scatter 미설치. 설치·GPU 병합·전체 학습은 실행하지 않았다.

## References

- [EZ-SP 원 논문, arXiv v2](https://arxiv.org/html/2512.00385v2): §III-B, Eq.7, 병합 단계/계층 partition 확인. 논문 성능 수치를 CT 처리량으로 전용하지 않는다.
- [공식 partition 함수, 고정 커밋](https://github.com/drprojects/torch-graph-components/blob/e3db9f352fae52dff416616742b3c7ff1378451d/src/torch_graph_components/merge.py): 목적함수, chain merge, min_size, isolated-node 연결 및 API 확인.
- [공식 dependency 안내](https://github.com/drprojects/torch-graph-components): torch/PyG/torch-scatter 의존성.
- [공식 Superpoint Transformer/EZ-SP 저장소](https://github.com/drprojects/superpoint_transformer): 원형 구현의 저장소 확인. 이번 검토는 그 전체 학습 코드를 검증했다는 뜻이 아니다.

## 다음 입력

첨부 본문에서 지칭한 `L0_Coarsen_Implementation_Handoff.zip` 또는 `CODEX_IMPLEMENTATION_SPEC.md` 실제 파일이 필요하다. 그 파일의 수치·병합 제약·CPU 반례 검사를 확인한 뒤 위 차이를 정리한다. 새 구조는 기존 학습과 exact resume가 아니다. 이번 검토만으로 구현/성능/학습 승인 완료라고 표시하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 구현 변경 미실행.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 batch 유지 및 공식 GPU 연산 확인.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검토 실행에서 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다. 이번에는 학습 실행 없음.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 새 encoder 미구현.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 모두 미실행.
